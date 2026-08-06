# src/sfsa_rag_assistant/nodes/check_context_sufficiency.py
"""
Agent 1: Context Sufficiency Checker

Analyzes whether the retrieved context from the vector database is sufficient
to answer the user's query. If insufficient, the workflow will trigger web search.
"""

import logging
from typing import Literal
from pydantic import BaseModel, Field

from ..generation import TextGenerator
from ..app_config import settings
from ..states import SFSAAgenticState

logger = logging.getLogger(__name__)


class ContextSufficiencyCheck(BaseModel):
    """
    Pydantic model for Agent 1's structured output.
    
    Ensures consistent decision format for routing logic.
    """
    decision: Literal["sufficient", "insufficient"] = Field(
        description="Whether the retrieved context is sufficient to answer the query"
    )
    reasoning: str = Field(
        description="Brief explanation of the decision (1-2 sentences)"
    )


def check_context_sufficiency(state: SFSAAgenticState) -> SFSAAgenticState:
    """
    Agent 1: Determine if retrieved context is sufficient to answer the query.
    
    This agent analyzes:
    - The user's query and its requirements
    - The retrieved context from the vector database
    - Whether the context contains enough information for a complete answer
    
    Decision logic:
    - "sufficient": Proceed to generate_response node
    - "insufficient": Trigger web_search node (Agent 2)
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state with user_query and retrieved_formatted
    
    Returns
    -------
    SFSAAgenticState
        Updated state with:
        - context_sufficient: bool indicating sufficiency
        - context_sufficiency_reasoning: str with decision explanation
    """
    logger.info("=" * 60)
    logger.info("AGENT 1: check_context_sufficiency")
    logger.info("=" * 60)
    
    # Use contextualized query (or fall back to original if not available)
    user_query = state.get("contextualized_query") or state.get("user_query", "")
    retrieved_context = state.get("retrieved_formatted", "")
    
    if not user_query:
        logger.error("No user query found in state")
        return {
            "context_sufficient": False,
            "context_sufficiency_reasoning": "Error: No query provided",
            "error": "No user query for sufficiency check"
        }
    
    if not retrieved_context:
        logger.warning("No retrieved context found - marking as insufficient")
        return {
            "context_sufficient": False,
            "context_sufficiency_reasoning": "No context retrieved from vector database"
        }
    
    logger.info(f"Query: {user_query}")
    logger.info(f"Context length: {len(retrieved_context)} chars")
    
    try:
        # Initialize TextGenerator with Ollama
        generator = TextGenerator(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            load_mode="ollama"
        )
        generator.load_model()
        
        # Configure for structured output
        structured_llm = generator.with_structured_output(ContextSufficiencyCheck)
        
        # Build prompt for Agent 1
        prompt = _build_sufficiency_prompt(user_query, retrieved_context)
        
        logger.info("Invoking Agent 1 with structured output...")
        
        # Get structured decision
        result: ContextSufficiencyCheck = structured_llm.invoke(prompt)
        
        logger.info(f"Decision: {result.decision}")
        logger.info(f"Reasoning: {result.reasoning}")
        
        # Convert decision to boolean
        is_sufficient = result.decision == "sufficient"
        
        return {
            "context_sufficient": is_sufficient,
            "context_sufficiency_reasoning": result.reasoning,
            "error": None
        }
    
    except Exception as e:
        logger.error(f"Error in Agent 1: {e}", exc_info=True)
        # Default to insufficient on error to trigger web search
        return {
            "context_sufficient": False,
            "context_sufficiency_reasoning": f"Error during check: {str(e)}",
            "error": f"Agent 1 failed: {str(e)}"
        }


def _build_sufficiency_prompt(query: str, context: str) -> str:
    """
    Build the prompt for Agent 1 to determine context sufficiency.
    
    Parameters
    ----------
    query : str
        The user's query
    context : str
        The retrieved context from vector database
    
    Returns
    -------
    str
        Formatted prompt for Agent 1
    """
    prompt = f"""You are a context sufficiency analyzer for a Steel Founders' Society of America (SFSA) RAG system.

Your task is to determine if the provided context contains sufficient information to answer the user's query comprehensively.

USER QUERY:
{query}

RETRIEVED CONTEXT:
{context}

EVALUATION CRITERIA:
1. Does the context directly address the key aspects of the query?
2. Is there enough detail to provide a complete, accurate answer?
3. Are there significant gaps or missing information?
4. Would web search provide substantially more relevant information?

DECISION RULES:
- Mark as "sufficient" if the context can answer the query adequately (even if not exhaustively)
- Mark as "insufficient" ONLY if:
  * The context is clearly off-topic or irrelevant
  * Critical information is missing that would require external sources
  * The query asks for recent/current information not in the context

Provide your decision and a brief reasoning (1-2 sentences).
"""
    
    return prompt
