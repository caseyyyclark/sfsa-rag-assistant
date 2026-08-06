# src/sfsa_rag_assistant/cli.py
"""
Command-line interface for SFSA Agentic RAG.

Provides a simple CLI for single-query execution with formatted output.
"""

import argparse
import logging
import sys

from .graph import run_workflow, get_graph_visualization
from .config import check_ollama_connection
from .app_config import settings
from .chat import chat_loop
from .wiki_utils import format_terminal_link
from .batch_processor import process_batch
from .gui import launch_gui


def setup_logging(debug: bool = False):
    """
    Configure logging for CLI.
    
    Parameters
    ----------
    debug : bool
        If True, set logging to INFO level (detailed). If False, WARNING level (clean).
    """
    level = logging.INFO if debug else logging.WARNING
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Override any existing configuration
    )


def print_response(result: dict, show_sources: bool = True, show_metadata: bool = False):
    """
    Pretty print the response.
    
    Parameters
    ----------
    result : dict
        The final output from run_workflow
    show_sources : bool
        Whether to display sources
    show_metadata : bool
        Whether to display workflow metadata
    """
    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)
    print(result.get("query", ""))
    
    print("\n" + "=" * 80)
    print("RESPONSE")
    print("=" * 80)
    print(result.get("response", ""))
    
    if show_sources:
        print("\n" + "=" * 80)
        print("SOURCES")
        print("=" * 80)
        sources = result.get("sources", [])
        if sources:
            for i, source in enumerate(sources, 1):
                source_type = source.get("source_type", "unknown")
                
                if source_type == "vector_db":
                    print(f"\n[{i}] SFSA Wiki")
                    
                    # Show clickable link if available
                    wiki_url = source.get('wiki_url')
                    source_file = source.get('source', 'N/A')
                    if wiki_url:
                        # Extract filename for display
                        filename = source_file.split('/')[-1]
                        clickable = format_terminal_link(wiki_url, filename)
                        print(f"    Document: {clickable}")
                        print(f"    URL: {wiki_url}")
                    else:
                        print(f"    Source: {source_file}")
                    
                    if 'page' in source:
                        print(f"    Page: {source.get('page')}")
                
                elif source_type == "web_search":
                    print(f"\n[{i}] Internet")
                    print(f"    Title: {source.get('title', 'N/A')}")
                    print(f"    URL: {source.get('url', 'N/A')}")
                    print(f"    Relevance: {source.get('score', 0.0):.2f}")
        else:
            print("No sources available.")
    
    if show_metadata:
        print("\n" + "=" * 80)
        print("WORKFLOW METADATA")
        print("=" * 80)
        metadata = result.get("metadata", {})
        
        print(f"Workflow Path: {metadata.get('workflow_path', 'unknown')}")
        print(f"Agents Triggered: {', '.join(metadata.get('agents_triggered', []))}")
        
        # Query Contextualization
        context_query_info = metadata.get("query_contextualization", {})
        if context_query_info.get("was_contextualized"):
            print(f"\nQuery Contextualization:")
            print(f"  Original: {context_query_info.get('original_query', 'N/A')}")
            print(f"  Rewritten: {context_query_info.get('contextualized_query', 'N/A')}")
            print(f"  Reason: {context_query_info.get('reasoning', 'N/A')}")
        
        context_info = metadata.get("context_sufficiency", {})
        print(f"\nContext Sufficiency: {context_info.get('sufficient', 'unknown')}")
        print(f"  Reasoning: {context_info.get('reasoning', 'N/A')}")
        
        validation_info = metadata.get("validation", {})
        print(f"\nValidation:")
        print(f"  Attempts: {validation_info.get('attempts', 0)}/{validation_info.get('max_attempts', 2)}")
        print(f"  Status: {validation_info.get('final_status', 'unknown')}")
        print(f"  Reasoning: {validation_info.get('reasoning', 'N/A')}")
        
        web_info = metadata.get("web_search", {})
        if web_info.get("used"):
            print(f"\nWeb Search:")
            print(f"  Query: {web_info.get('query', 'N/A')}")
            print(f"  Results: {web_info.get('num_results', 0)}")


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="SFSA Agentic RAG - Steel Founders' Society of America Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple query
  python -m sfsa_rag_assistant "What is steel casting?"
  
  # Interactive chat with conversation memory
  python -m sfsa_rag_assistant --chat
  
  # Chat with debug logging (shows detailed workflow)
  python -m sfsa_rag_assistant --chat --debug
  
  # Batch process questions from CSV
  python -m sfsa_rag_assistant --batch input.csv --output results.csv
  
  # Batch with custom question column
  python -m sfsa_rag_assistant --batch input.csv --output results.csv --question-column "user_question"
  
  # With full metadata
  python -m sfsa_rag_assistant "What is steel casting?" --show-metadata
  
  # Hide sources
  python -m sfsa_rag_assistant "What is steel casting?" --no-sources
  
  # Show graph visualization
  python -m sfsa_rag_assistant --show-graph

  # Launch browser UI
  python -m sfsa_rag_assistant --gui --host 0.0.0.0 --port 7860
"""
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="The question to ask the RAG system"
    )
    
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive chat mode with conversation memory"
    )
    
    parser.add_argument(
        "--batch",
        type=str,
        metavar="INPUT_CSV",
        help="Batch process questions from CSV file (requires --output)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        metavar="OUTPUT_CSV",
        help="Output CSV file for batch processing results"
    )
    
    parser.add_argument(
        "--question-column",
        type=str,
        default="question",
        help="Name of the column containing questions in batch CSV (default: 'question')"
    )
    
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Number of Q&A pairs to remember in chat mode (default: 5)"
    )
    
    parser.add_argument(
        "--show-sources",
        dest="show_sources",
        action="store_true",
        default=True,
        help="Show sources (default: True)"
    )
    
    parser.add_argument(
        "--no-sources",
        dest="show_sources",
        action="store_false",
        help="Hide sources"
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the browser-based GUI"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface for the GUI server (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the GUI server (default: 7860)"
    )

    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Show detailed workflow metadata"
    )
    
    parser.add_argument(
        "--max-validation-attempts",
        type=int,
        default=None,
        help=f"Maximum validation attempts (default: {settings.max_validation_attempts})"
    )
    
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Display the workflow graph visualization and exit"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows detailed workflow information)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="(Deprecated: use --debug) Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging (use debug or verbose flag)
    debug_mode = args.debug or args.verbose
    setup_logging(debug_mode)
    
    # Show graph visualization if requested
    if args.show_graph:
        print(get_graph_visualization())
        return 0

    if args.gui:
        try:
            launch_gui(host=args.host, port=args.port)
        except ImportError as exc:
            print(f"Error: {exc}")
            print("Install the GUI dependency first, then relaunch the command.")
            return 1
        return 0
    
    # Start interactive chat if requested
    if args.chat:
        chat_loop(
            show_sources=args.show_sources,
            max_history_pairs=args.history,
            debug=debug_mode
        )
        return 0
    
    # Batch processing mode
    if args.batch:
        if not args.output:
            print("Error: --output is required for batch mode")
            parser.print_help()
            return 1
        
        try:
            print("=" * 80)
            print("SFSA AGENTIC RAG - BATCH PROCESSING")
            print("=" * 80)
            print(f"Input file: {args.batch}")
            print(f"Output file: {args.output}")
            print(f"Question column: {args.question_column}")
            print(f"Max validation attempts: {args.max_validation_attempts or settings.max_validation_attempts}")
            print("=" * 80)
            print()
            
            # Check Ollama connection
            print("Checking Ollama connection...")
            if not check_ollama_connection():
                print("\nError: Cannot connect to Ollama. Please ensure:")
                print(f"  1. Ollama is running at {settings.ollama_base_url}")
                print(f"  2. Model '{settings.ollama_model}' is pulled")
                print(f"     Run: ollama pull {settings.ollama_model}")
                return 1
            
            print(f"Using model: {settings.ollama_model}")
            print()
            
            # Run batch processing
            stats = process_batch(
                input_csv=args.batch,
                output_csv=args.output,
                question_column=args.question_column,
                max_validation_attempts=args.max_validation_attempts or settings.max_validation_attempts
            )
            
            return 0
            
        except KeyboardInterrupt:
            print("\n\nBatch processing interrupted by user")
            return 130
        
        except Exception as e:
            print(f"\nError in batch processing: {e}")
            if debug_mode:
                import traceback
                traceback.print_exc()
            return 1
    
    # Validate query for single-query mode
    if not args.query:
        parser.print_help()
        print("\nError: Please provide a query or use --chat for interactive mode")
        return 1
    
    try:
        # Check Ollama connection
        print("Initializing SFSA Agentic RAG...")
        
        print("Checking Ollama connection...")
        if not check_ollama_connection():
            print("\nError: Cannot connect to Ollama. Please ensure:")
            print(f"  1. Ollama is running at {settings.ollama_base_url}")
            print(f"  2. Model '{settings.ollama_model}' is pulled")
            print(f"     Run: ollama pull {settings.ollama_model}")
            return 1
        
        print(f"Using model: {settings.ollama_model}")
        print()
        
        # Run workflow
        result = run_workflow(
            user_query=args.query,
            max_validation_attempts=args.max_validation_attempts
        )
        
        # Print results
        print_response(
            result,
            show_sources=args.show_sources,
            show_metadata=args.show_metadata
        )
        
        print("\n" + "=" * 80)
        return 0
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    
    except Exception as e:
        print(f"\nError: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
