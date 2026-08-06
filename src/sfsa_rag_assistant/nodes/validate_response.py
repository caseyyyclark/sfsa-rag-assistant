# src/sfsa_rag_assistant/nodes/validate_response.py
"""
Agent 3: Response Validator with Refinement Loop

This agent evaluates the quality of generated responses and decides whether
to accept them or generate a refined query for another attempt.
"""

import logging
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator, ValidationError

from ..generation import TextGenerator
from ..app_config import settings
from ..states import SFSAAgenticState

logger = logging.getLogger(__name__)


class ResponseValidation(BaseModel):
    """
    Pydantic model for Agent 3's structured output.
    
    Ensures consistent validation decision format for routing logic.
    """
    decision: Literal["satisfactory", "needs_refinement"] = Field(
        description="Whether the response meets quality standards"
    )
    reasoning: str = Field(
        description="Brief explanation of the validation decision (1-3 sentences)"
    )
    refined_query: Optional[str] = Field(
        default=None,
        description="If needs_refinement, a reformulated version of the original query to get a better answer"
    )
    
    @model_validator(mode='after')
    def validate_refined_query(self):
        """
        Enforce that refined_query MUST be provided when decision is needs_refinement.
        This ensures Agent 3 always provides a clarifying question when requesting refinement.
        """
        if self.decision == "needs_refinement":
            if not self.refined_query or not self.refined_query.strip():
                raise ValueError(
                    "refined_query is REQUIRED when decision is 'needs_refinement'. "
                    "You must provide a reformulated query that addresses the gaps in the response."
                )
        return self


def validate_response(state: SFSAAgenticState) -> SFSAAgenticState:
    """
    Agent 3: Validate response quality and decide on next action.
    
    This agent:
    1. Analyzes the generated response against the user's query
    2. Evaluates quality, completeness, and accuracy
    3. Decides whether to accept or request refinement
    4. If refinement needed AND under max attempts: generates refined_query
    5. If max attempts reached: accepts current response
    
    Decision logic:
    - "satisfactory": Proceed to format_final_output
    - "needs_refinement" + attempts < max: Loop back to generate_response with refined_query
    - "needs_refinement" + attempts >= max: Proceed to format_final_output (combine all)
    
    Parameters
    ----------
    state : SFSAAgenticState
        Current workflow state with current_response and validation_attempts
    
    Returns
    -------
    SFSAAgenticState
        Updated state with:
        - response_satisfactory: bool validation decision
        - validation_reasoning: str explanation of decision
        - refined_query: Optional[str] improved query for retry
    """
    logger.info("=" * 60)
    logger.info("AGENT 3: validate_response")
    logger.info("=" * 60)
    
    # Use contextualized query (or fall back to original if not available)
    user_query = state.get("contextualized_query") or state.get("user_query", "")
    current_response = state.get("current_response", "")
    validation_attempts = state.get("validation_attempts", 0)
    max_attempts = state.get("max_validation_attempts", settings.max_validation_attempts)
    
    if not user_query:
        logger.error("No user query found in state")
        return {
            "response_satisfactory": True,  # Accept by default
            "validation_reasoning": "Error: No query to validate against",
            "error": "No user query for validation"
        }
    
    if not current_response:
        logger.error("No current response to validate")
        return {
            "response_satisfactory": True,  # Accept by default
            "validation_reasoning": "Error: No response to validate",
            "error": "No response for validation"
        }
    
    logger.info(f"Query: {user_query}")
    logger.info(f"Response length: {len(current_response)} chars")
    logger.info(f"Validation attempt: {validation_attempts}/{max_attempts}")
    
    # Check if we've reached max attempts
    if validation_attempts >= max_attempts:
        logger.info(f"Max validation attempts ({max_attempts}) reached - accepting response")
        
        # Update validation history
        validation_history = state.get("validation_history", [])
        validation_history.append({
            "attempt": validation_attempts + 1,
            "response": current_response,
            "satisfactory": True,  # Forced acceptance
            "reasoning": f"Maximum validation attempts ({max_attempts}) reached. Using best available response.",
            "refined_query": None
        })
        
        return {
            "response_satisfactory": True,
            "validation_reasoning": f"Maximum validation attempts ({max_attempts}) reached. Using best available response.",
            "refined_query": None,
            "validation_history": validation_history
        }
    
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
        structured_llm = generator.with_structured_output(ResponseValidation)
        
        # Build prompt for Agent 3
        prompt = _build_validation_prompt(user_query, current_response, validation_attempts)
        
        logger.info("Invoking Agent 3 with structured output...")
        
        # Get structured validation decision - with retry logic for missing refined_query
        max_retries = 2
        result = None
        
        for retry_attempt in range(max_retries):
            try:
                result = structured_llm.invoke(prompt)
                # If we get here, the Pydantic validation passed
                # But let's double-check manually as a safety net
                if result.decision == "needs_refinement":
                    if not result.refined_query or not result.refined_query.strip():
                        raise ValueError(
                            "SAFETY CHECK FAILED: refined_query is empty or None despite passing Pydantic validation. "
                            f"Decision was '{result.decision}' but refined_query is '{result.refined_query}'"
                        )
                # Log success if this was a retry
                if retry_attempt > 0:
                    logger.info(f"✓ Agent 3 successfully provided refined_query on retry attempt {retry_attempt + 1}")
                break
            except (ValueError, ValidationError) as ve:
                error_msg = str(ve)
                if "refined_query" in error_msg.lower() or "SAFETY CHECK" in error_msg:
                    logger.warning(f"Agent 3 failed to provide refined_query on attempt {retry_attempt + 1}/{max_retries}")
                    logger.debug(f"Validation error details: {ve}")
                    
                    if retry_attempt < max_retries - 1:
                        logger.info("Retrying with even more explicit prompt...")
                        # Append an even stronger warning to the end of the prompt
                        prompt = prompt.rstrip() + "\n\n" + """
⚠️ CRITICAL VALIDATION REQUIREMENT ⚠️
Your previous response was rejected because it specified decision='needs_refinement' but did NOT provide a refined_query.

YOU MUST INCLUDE ALL THREE FIELDS:
1. decision: "satisfactory" OR "needs_refinement"
2. reasoning: Brief explanation (always required)
3. refined_query: 
   - NULL/empty if decision='satisfactory'
   - A complete rephrased question if decision='needs_refinement' (MANDATORY!)

Example valid response when refinement is needed:
{
  "decision": "needs_refinement",
  "reasoning": "The response lacks specific temperature values and time parameters.",
  "refined_query": "What are the specific temperature ranges and time durations for each degassing method?"
}

Do not omit the refined_query field if you choose needs_refinement!
"""
                    else:
                        # Last retry failed - accept the response to avoid blocking
                        logger.error("Agent 3 repeatedly failed to provide refined_query - accepting response")
                        validation_history = state.get("validation_history", [])
                        validation_history.append({
                            "attempt": validation_attempts + 1,
                            "response": current_response,
                            "satisfactory": True,
                            "reasoning": "Validation failed: Agent 3 could not generate refined_query. Accepting current response.",
                            "refined_query": None
                        })
                        return {
                            "response_satisfactory": True,
                            "validation_reasoning": "Validation error - accepting current response",
                            "refined_query": None,
                            "validation_history": validation_history,
                            "error": f"Agent 3 validation error: {str(ve)}"
                        }
                else:
                    # Different validation error - re-raise
                    raise
        
        if result is None:
            raise RuntimeError("Failed to get validation result after retries")
        
        logger.info(f"Decision: {result.decision}")
        logger.info(f"Reasoning: {result.reasoning}")
        
        if result.refined_query:
            logger.info(f"Refined query: {result.refined_query}")
        else:
            logger.info("No refined query (response satisfactory)")
        
        # Convert decision to boolean
        is_satisfactory = result.decision == "satisfactory"
        
        # Update validation history
        validation_history = state.get("validation_history", [])
        validation_history.append({
            "attempt": validation_attempts + 1,
            "response": current_response,
            "satisfactory": is_satisfactory,
            "reasoning": result.reasoning,
            "refined_query": result.refined_query
        })
        
        return {
            "response_satisfactory": is_satisfactory,
            "validation_reasoning": result.reasoning,
            "refined_query": result.refined_query if not is_satisfactory else None,
            "validation_history": validation_history,
            "error": None
        }
    
    except Exception as e:
        logger.error(f"Error in Agent 3: {e}", exc_info=True)
        # Default to accepting on error
        return {
            "response_satisfactory": True,
            "validation_reasoning": f"Validation error - accepting response: {str(e)}",
            "refined_query": None,
            "error": f"Agent 3 failed: {str(e)}"
        }


def _build_validation_prompt(query: str, response: str, attempt: int) -> str:
    """
    Build the prompt for Agent 3 to validate response quality.
    
    Parameters
    ----------
    query : str
        The user's original query
    response : str
        The generated response to validate
    attempt : int
        Current validation attempt number
    
    Returns
    -------
    str
        Formatted prompt for Agent 3
    """
    prompt = f"""You are a response quality validator for a Steel Founders' Society of America (SFSA) RAG system.

Your task is to evaluate whether the generated response thoroughly answers the user's query.

USER QUERY:
{query}

GENERATED RESPONSE:
{response}

VALIDATION CRITERIA:
1. **Relevance**: Does the response directly address the user's query?
2. **Completeness**: Are all important aspects of the query covered with sufficient detail?
3. **Thoroughness**: Does the response include relevant specifics, examples, or technical details where appropriate?
4. **Accuracy**: Is the information consistent and free of contradictions?
5. **Clarity**: Is the response well-structured and easy to understand?
6. **Actionability**: For "how-to" questions, are practical steps or methods explained?

DECISION RULES:
- Mark as "satisfactory" if:
  * The response addresses the query thoroughly (not just adequately)
  * All important aspects are covered with reasonable detail
  * Information is accurate, clear, and actionable
  * No significant gaps that would leave the user wanting more

- Mark as "needs_refinement" if:
  * The response misses important aspects of the query
  * The answer lacks sufficient detail or is too superficial
  * Key technical information is omitted when available
  * There are clarity or coherence issues
  * The response could be notably more complete without being verbose

CRITICAL REQUIREMENT - REFINED QUERY:
**IF you decide "needs_refinement", you MUST provide a refined_query. This is MANDATORY, not optional.**

Your refined_query should:
- Rephrase the original query to emphasize missing aspects
- Focus on the gaps or areas needing more detail
- Help the generator provide a more thorough answer
- Be still answerable with the same context (no new retrieval needed)
- Be a complete, well-formed question that addresses the identified gaps

Example: If the original query was "What are degassing methods?" and the response lacked specific procedures, 
your refined_query might be: "What are the specific procedural steps and parameters for degassing methods?"

Current attempt: {attempt + 1}

IMPORTANT: When providing your response:
- If decision='satisfactory': refined_query should be null/empty
- If decision='needs_refinement': refined_query is MANDATORY and must contain a complete, well-formed question

**Failure to provide refined_query when decision='needs_refinement' will cause a validation error.**
"""
    
    return prompt
