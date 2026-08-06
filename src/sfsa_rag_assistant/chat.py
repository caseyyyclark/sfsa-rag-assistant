# src/sfsa_rag_assistant/chat.py
"""
Interactive chat interface for SFSA Agentic RAG.

Provides a conversational experience with memory of previous Q&A pairs.
Maintains the last 5 question-answer exchanges for context.

Automatically manages Ollama server lifecycle:
- Starts Ollama if not running
- Stops Ollama on exit (only if this session started it)
"""

import sys
import logging
import subprocess
import time
import atexit
from typing import List, Dict, Any, Optional

from .graph import run_workflow
from .config import check_ollama_connection
from .app_config import settings
from .wiki_utils import format_terminal_link


logger = logging.getLogger(__name__)

# Track if we started Ollama in this session
_ollama_started_by_us = False


def _is_ollama_running() -> bool:
    """
    Check if Ollama server is running.
    
    Returns
    -------
    bool
        True if Ollama is running, False otherwise
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Failed to check Ollama status: {e}")
        return False


def _start_ollama() -> bool:
    """
    Start Ollama server in the background.
    
    Returns
    -------
    bool
        True if started successfully, False otherwise
    """
    global _ollama_started_by_us
    
    try:
        print("Starting Ollama server...")
        
        # Start Ollama in background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Wait a moment for Ollama to start
        time.sleep(3)
        
        # Verify it's running
        if check_ollama_connection():
            _ollama_started_by_us = True
            print("Ollama server started successfully")
            return True
        else:
            print("Failed to start Ollama server")
            return False
    
    except Exception as e:
        print(f"Error starting Ollama: {e}")
        return False


def _stop_ollama():
    """
    Stop Ollama server (only if we started it).
    """
    global _ollama_started_by_us
    
    if not _ollama_started_by_us:
        return
    
    try:
        print("\nStopping Ollama server...")
        subprocess.run(
            ["pkill", "-f", "ollama serve"],
            capture_output=True
        )
        subprocess.run(
            ["pkill", "-f", "Ollama.app"],
            capture_output=True
        )
        print("Ollama server stopped")
    except Exception as e:
        logger.warning(f"Failed to stop Ollama: {e}")


# Register cleanup handler
atexit.register(_stop_ollama)


class SFSAChatSession:
    """
    Interactive chat session manager with conversation memory.
    
    Attributes
    ----------
    conversation_history : List[Dict[str, str]]
        History of conversation turns, limited to last 5 Q&A pairs
    max_history_pairs : int
        Maximum number of Q&A pairs to remember (default: 5)
    """
    
    def __init__(self, max_history_pairs: int = 5):
        """
        Initialize a new chat session.
        
        Parameters
        ----------
        max_history_pairs : int
            Maximum number of Q&A pairs to keep in memory (default: 5)
        """
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_pairs = max_history_pairs
        self.max_history_turns = max_history_pairs * 2  # Each pair = 2 turns
    
    def add_turn(self, role: str, content: str):
        """
        Add a conversation turn to history.
        
        Parameters
        ----------
        role : str
            "user" or "assistant"
        content : str
            The message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Trim history to last N pairs (2N turns)
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear all conversation history."""
        self.conversation_history = []
    
    def get_history_summary(self) -> str:
        """
        Get a formatted summary of conversation history.
        
        Returns
        -------
        str
            Formatted string showing conversation history
        """
        if not self.conversation_history:
            return "No conversation history yet."
        
        num_pairs = len(self.conversation_history) // 2
        summary_parts = [f"Conversation history ({num_pairs} Q&A pairs):"]
        
        for i, turn in enumerate(self.conversation_history, 1):
            role = turn["role"].upper()
            content = turn["content"][:100]
            if len(turn["content"]) > 100:
                content += "..."
            summary_parts.append(f"  {i}. {role}: {content}")
        
        return "\n".join(summary_parts)


def print_welcome():
    """Print welcome message and instructions."""
    print("\n" + "=" * 80)
    print("SFSA AGENTIC RAG - INTERACTIVE CHAT")
    print("=" * 80)
    print("\nWelcome! I'm here to answer your questions about steel casting, foundry")
    print("operations, metallurgy, and related topics from the SFSA knowledge base.")
    print("\nI'll remember the last 5 question-answer pairs for context.")
    
    if _ollama_started_by_us:
        print("\nNote: Ollama server was started automatically and will stop when you quit.")
    
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - Type 'history' to see conversation history")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'quit', 'exit', or 'bye' to end the session")
    print("\n" + "=" * 80 + "\n")


def print_response(response: str, sources: List[Dict[str, Any]], show_sources: bool = True):
    """
    Print the assistant's response with optional sources.
    
    Parameters
    ----------
    response : str
        The generated response text
    sources : List[Dict[str, Any]]
        List of source metadata
    show_sources : bool
        Whether to display sources
    """
    print("\n" + "-" * 80)
    print("ASSISTANT:")
    print("-" * 80)
    print(response)
    
    if show_sources and sources:
        print("\n" + "-" * 40)
        print("Sources:")
        print("-" * 40)
        
        for i, source in enumerate(sources, 1):  # Show all sources
            source_type = source.get("source_type", "unknown")
            
            if source_type == "vector_db":
                # Get wiki URL and metadata
                wiki_url = source.get('wiki_url')
                source_path = source.get("source", "N/A")
                
                # Extract filename
                if "\\" in source_path or "/" in source_path:
                    filename = source_path.split("\\")[-1].split("/")[-1]
                else:
                    filename = source_path
                
                page_info = f", page {source.get('page')}" if 'page' in source else ""
                
                # Create clickable link if URL available
                if wiki_url:
                    clickable = format_terminal_link(wiki_url, filename)
                    print(f"  [{i}] SFSA Wiki: {clickable}{page_info}")
                else:
                    print(f"  [{i}] SFSA Wiki: {filename}{page_info}")
            
            elif source_type == "web_search":
                print(f"  [{i}] Internet: {source.get('title', 'N/A')}")
                print(f"      {source.get('url', 'N/A')}")
    
    print("\n" + "=" * 80 + "\n")


def chat_loop(
    show_sources: bool = True,
    max_history_pairs: int = 5,
    debug: bool = False
):
    """
    Run the interactive chat loop.
    
    Parameters
    ----------
    show_sources : bool
        Whether to display sources with each response
    max_history_pairs : int
        Maximum number of Q&A pairs to remember
    debug : bool
        Enable debug logging (logging already configured by CLI)
    """
    # Check if stdin is available (not when using conda run)
    if not sys.stdin.isatty():
        print("\nERROR: Interactive mode requires a terminal with stdin.")
        print("\nTo use chat mode, activate the conda environment first:")
        print("  1. conda activate sfsa_rag_assistant")
        print("  2. python -m sfsa_rag_assistant --chat")
        print("\nOr use single-query mode instead:")
        print("  conda run -n sfsa_rag_assistant python -m sfsa_rag_assistant \"Your question\"")
        sys.exit(1)
    
    # Note: Logging is already configured by cli.py setup_logging()
    # No need to reconfigure here - just use the configured logger
    
    # Initialize and start Ollama if needed
    print("Initializing SFSA Agentic RAG...")
    
    # Check if Ollama is already running
    if _is_ollama_running():
        print("Ollama server is already running")
    else:
        print("Ollama server not detected")
        if not _start_ollama():
            print("\nERROR: Failed to start Ollama server.")
            print("\nPlease try starting it manually:")
            print("  1. Open a new terminal")
            print("  2. Run: ollama serve")
            print("  3. Then try the chat again")
            sys.exit(1)
    
    # Verify connection and model availability
    if not check_ollama_connection():
        print("\nERROR: Cannot connect to Ollama.")
        print(f"Expected URL: {settings.ollama_base_url}")
        print(f"Expected model: {settings.ollama_model}")
        print("\nTry running: ollama pull llama3.1:8b")
        sys.exit(1)
    
    print(f"Using model: {settings.ollama_model}")
    
    # Initialize chat session
    session = SFSAChatSession(max_history_pairs=max_history_pairs)
    
    # Print welcome
    print_welcome()
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("YOU: ").strip()
            
            # Handle empty input
            if not user_input:
                print("Please enter a question.\n")
                continue
            
            # Handle commands
            user_input_lower = user_input.lower()
            
            if user_input_lower in ["quit", "exit", "bye", "q"]:
                print("\nThank you for using SFSA Agentic RAG. Goodbye!\n")
                break
            
            elif user_input_lower == "history":
                print("\n" + session.get_history_summary() + "\n")
                continue
            
            elif user_input_lower == "clear":
                session.clear_history()
                print("\nConversation history cleared.\n")
                continue
            
            elif user_input_lower in ["help", "?"]:
                print("\nCommands:")
                print("  - Type your question and press Enter")
                print("  - 'history' - Show conversation history")
                print("  - 'clear' - Clear conversation history")
                print("  - 'quit', 'exit', 'bye' - End session\n")
                continue
            
            # Process query with workflow
            print("\nThinking...")
            
            try:
                # Run workflow with conversation history
                result = run_workflow(
                    user_query=user_input,
                    conversation_history=session.get_history(),
                    max_validation_attempts=settings.max_validation_attempts
                )
                
                # Extract response
                response_text = result.get("response", "I apologize, but I couldn't generate a response.")
                sources = result.get("sources", [])
                
                # Add to conversation history
                session.add_turn("user", user_input)
                session.add_turn("assistant", response_text)
                
                # Print response
                print_response(response_text, sources, show_sources)
            
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                print(f"\nERROR: Failed to process your question: {str(e)}\n")
                print("Please try again or type 'quit' to exit.\n")
        
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Type 'quit' to exit or continue chatting.\n")
            continue
        
        except EOFError:
            print("\n\nThank you for using SFSA Agentic RAG. Goodbye!\n")
            break


def main():
    """
    Main entry point for interactive chat.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SFSA Agentic RAG - Interactive Chat Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive chat
  python -m sfsa_rag_assistant.chat
  
  # Hide sources
  python -m sfsa_rag_assistant.chat --no-sources
  
  # Custom history size (default: 5 pairs)
  python -m sfsa_rag_assistant.chat --history 10
  
  # Verbose logging
  python -m sfsa_rag_assistant.chat --verbose
"""
    )
    
    parser.add_argument(
        "--no-sources",
        dest="show_sources",
        action="store_false",
        default=True,
        help="Hide sources in responses"
    )
    
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Number of Q&A pairs to remember (default: 5)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Run chat loop
    chat_loop(
        show_sources=args.show_sources,
        max_history_pairs=args.history,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
