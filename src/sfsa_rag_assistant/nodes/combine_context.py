# src/sfsa_rag_assistant/nodes/combine_context.py
"""
Node for combining retrieved context from vector DB with web search results.

This node merges information from two sources to provide comprehensive context
for the generation node when Agent 1 determined that RAG context alone was insufficient.
"""

import logging
from ..states import SFSAAgenticState

logger = logging.getLogger(__name__)


def combine_context(state: SFSAAgenticState) -> SFSAAgenticState:
    """
    Combine retrieved context from vector DB with web search results.
    
    This node:
    1. Takes the formatted retrieved context (from vector DB)
    2. Takes the formatted web search results (from Tavily)
    3. Merges them into a single combined context string
    4. Updates state with the combined context
    
    The combined context will be used by the generate_response node to produce
    a comprehensive answer that leverages both internal knowledge and external sources.
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state with retrieved_formatted and web_search_formatted
    
    Returns
    -------
    SFSAAgenticState
        Updated state with:
        - combined_context: str merged context from both sources
    """
    logger.info("=" * 60)
    logger.info("NODE: combine_context")
    logger.info("=" * 60)
    
    retrieved_context = state.get("retrieved_formatted", "")
    web_search_context = state.get("web_search_formatted", "")
    
    # Build combined context with clear section markers
    combined_parts = []
    
    # Section 1: Vector DB context
    if retrieved_context:
        combined_parts.append("=" * 60)
        combined_parts.append("INTERNAL KNOWLEDGE BASE (Vector DB)")
        combined_parts.append("=" * 60)
        combined_parts.append(retrieved_context)
        logger.info(f"Added retrieved context: {len(retrieved_context)} chars")
    else:
        logger.warning("No retrieved context available")
    
    # Section 2: Web search results
    if web_search_context and web_search_context != "No web search results found.":
        combined_parts.append("")  # Blank line separator
        combined_parts.append("=" * 60)
        combined_parts.append("EXTERNAL WEB SOURCES (Tavily Search)")
        combined_parts.append("=" * 60)
        combined_parts.append(web_search_context)
        logger.info(f"Added web search context: {len(web_search_context)} chars")
    else:
        logger.info("No web search context available")
    
    # Combine all parts
    combined_context = "\n".join(combined_parts)
    
    logger.info(f"Total combined context: {len(combined_context)} chars")
    
    # Handle edge case: no context at all
    if not combined_context.strip():
        combined_context = "No context available from either vector database or web search."
        logger.error("No context available from any source")
    
    return {
        "combined_context": combined_context
    }
