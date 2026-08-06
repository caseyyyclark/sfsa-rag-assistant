# src/sfsa_rag_assistant/nodes/contextualize_query.py
"""
Query Contextualization Node

Rewrites user queries to be self-contained by incorporating relevant context
from conversation history. This ensures the retriever can find relevant documents
even when users ask follow-up questions with pronouns or implicit references.

Example:
    History: "What is steel casting?" -> "Steel casting is..."
    Query: "What are its main applications?"
    Contextualized: "What are the main applications of steel casting?"
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

from ..generation import TextGenerator
from ..app_config import settings
from ..states import SFSAAgenticState

logger = logging.getLogger(__name__)


class ContextualizedQuery(BaseModel):
    """
    Pydantic model for query contextualization output.
    """
    contextualized_query: str = Field(
        description="The rewritten query that is self-contained and includes necessary context from conversation history"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of what context was added (optional)"
    )


def contextualize_query(state: SFSAAgenticState) -> SFSAAgenticState:
    """
    Rewrite user query to be self-contained using conversation history.
    
    This node runs BEFORE retrieval to ensure queries with pronouns or implicit
    references are properly contextualized. Without this, questions like
    "What are its applications?" would fail because the retriever doesn't know
    what "its" refers to.
    
    Flow:
    1. Check if conversation history exists
    2. If NO history: Return original query (no context needed)
    3. If YES: Use LLM to rewrite query incorporating relevant context
    4. Store contextualized query for retrieval
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state with user_query and conversation_history
    
    Returns
    -------
    SFSAAgenticState
        Updated state with contextualized_query field
    """
    logger.info("=" * 60)
    logger.info("NODE: contextualize_query")
    logger.info("=" * 60)
    
    user_query = state.get("user_query", "")
    conversation_history = state.get("conversation_history", [])
    
    if not user_query:
        logger.error("No user query found in state")
        return {
            "contextualized_query": "",
            "error": "No user query to contextualize"
        }
    
    # If no conversation history, use original query
    if not conversation_history:
        logger.info("No conversation history - using original query")
        logger.info(f"Query: {user_query}")
        return {
            "contextualized_query": user_query,
            "contextualization_reasoning": "No conversation history available"
        }
    
    logger.info(f"Original query: {user_query}")
    logger.info(f"Conversation history: {len(conversation_history)} turns")
    
    try:
        # Initialize TextGenerator with Ollama
        generator = TextGenerator(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,  # Use low temperature for consistent rewrites
            load_mode="ollama"
        )
        generator.load_model()
        
        # Configure for structured output
        structured_llm = generator.with_structured_output(ContextualizedQuery)
        
        # Build prompt for query contextualization
        prompt = _build_contextualization_prompt(user_query, conversation_history)
        
        logger.info("Contextualizing query with LLM...")
        
        # Get contextualized query
        result: ContextualizedQuery = structured_llm.invoke(prompt)
        
        logger.info(f"Contextualized query: {result.contextualized_query}")
        if result.reasoning:
            logger.info(f"Reasoning: {result.reasoning}")
        
        return {
            "contextualized_query": result.contextualized_query,
            "contextualization_reasoning": result.reasoning,
            "error": None
        }
    
    except Exception as e:
        logger.error(f"Error in query contextualization: {e}", exc_info=True)
        # Fallback to original query on error
        logger.warning("Falling back to original query due to error")
        return {
            "contextualized_query": user_query,
            "contextualization_reasoning": f"Error during contextualization - using original query: {str(e)}",
            "error": f"Query contextualization failed: {str(e)}"
        }


def _build_contextualization_prompt(query: str, conversation_history: list) -> str:
    """
    Build the prompt for query contextualization.
    
    Parameters
    ----------
    query : str
        The user's current query (potentially with pronouns/implicit refs)
    conversation_history : list
        List of (question, answer) tuples from previous conversation
    
    Returns
    -------
    str
        Formatted prompt for the LLM
    """
    # Format conversation history
    history_text = ""
    for i, (prev_q, prev_a) in enumerate(conversation_history, 1):
        # Truncate long answers to keep prompt manageable
        truncated_answer = prev_a[:300] + "..." if len(prev_a) > 300 else prev_a
        history_text += f"\nTurn {i}:\n"
        history_text += f"  User: {prev_q}\n"
        history_text += f"  Assistant: {truncated_answer}\n"
    
    prompt = f"""You are a query rewriter for a Steel Founders' Society of America (SFSA) RAG system.

Your task is to rewrite the user's current question to be self-contained by incorporating necessary context from the conversation history.

CONVERSATION HISTORY:
{history_text}

CURRENT USER QUESTION:
{query}

YOUR TASK:
Rewrite the current question to be self-contained and clear, so it can be used for document retrieval WITHOUT needing the conversation history.

REWRITING RULES:
1. **Replace pronouns** with specific nouns from context:
   - "it" → "steel casting"
   - "they" → "shrinkage defects"
   - "this method" → "the specific method mentioned"

2. **Add missing context** for implicit references:
   - "What are the applications?" → "What are the applications of steel casting?"
   - "How do I prevent them?" → "How do I prevent shrinkage defects?"

3. **Keep the question intent** - don't change what the user is asking

4. **If already self-contained** (no pronouns, no implicit references):
   - Return the original question unchanged

5. **Only use context from history** - don't add new information

EXAMPLES:

Example 1:
History: "What is steel casting?" → "Steel casting is a manufacturing process..."
Query: "What are its main applications?"
Rewrite: "What are the main applications of steel casting?"

Example 2:
History: "Tell me about shrinkage defects" → "Shrinkage defects occur when..."
Query: "How do I prevent them?"
Rewrite: "How do I prevent shrinkage defects in steel casting?"

Example 3:
Query: "What is steel casting?" (no relevant history)
Rewrite: "What is steel casting?" (unchanged)

Example 4:
History: "What causes porosity?" → "Porosity is caused by..."
Query: "What temperature should I use for pouring?"
Rewrite: "What temperature should I use for pouring?" (already self-contained)

Now rewrite the current user question.
"""
    
    return prompt
