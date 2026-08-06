"""
Global configuration and environment handling for SFSA Agentic RAG.

This module handles:
- Loading environment variables from .env file
- Setting up LangSmith tracing
- Ensuring required API keys are present
- Initializing logging

Based on the KnowMat2 configuration pattern.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from .app_config import settings

# ─── Setup Logging ──────────────────────────────────────────────────────────
# Note: Logging is configured by cli.py or other entry points
# Default level is WARNING for clean output (use --debug for detailed logs)
logger = logging.getLogger(__name__)

# ─── Locate and Load .env File ──────────────────────────────────────────────
# Search order:
# 1. .env in current working directory
# 2. Path specified by SFSA_ENV_FILE environment variable
# 3. First .env found walking upwards from cwd

_cwd_dotenv = Path.cwd() / ".env"
if _cwd_dotenv.is_file():
    _env_path = str(_cwd_dotenv)
    logger.info(f"Found .env in current directory: {_env_path}")
else:
    _env_path = os.getenv("SFSA_ENV_FILE", "")
    if not _env_path:
        found = find_dotenv(usecwd=True)
        _env_path = found if found else ""
    
    if _env_path:
        logger.info(f"Loading environment from: {_env_path}")

if _env_path:
    load_dotenv(_env_path, override=False)
else:
    logger.warning("No .env file found. Using environment variables only.")

# ─── Setup LangSmith Tracing ────────────────────────────────────────────────
# Enable LangSmith for debugging and monitoring the agentic workflow
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "SFSA-Agentic-RAG")

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    logger.info("LangSmith tracing enabled")
    logger.info(f"LangSmith project: {os.getenv('LANGCHAIN_PROJECT')}")

# ─── Helper Function to Prompt for Missing Keys ─────────────────────────────
def _ensure_env_var(var_name: str, required: bool = False) -> None:
    """
    Check if an environment variable is set, optionally prompt for it.
    
    Parameters
    ----------
    var_name : str
        Name of the environment variable to check
    required : bool
        If True and variable is missing, prompt user for input
    """
    if var_name not in os.environ:
        if required:
            import getpass
            value = getpass.getpass(f"{var_name} not found. Please enter it now: ")
            os.environ[var_name] = value
            logger.info(f"Set {var_name} from user input")
        else:
            logger.warning(f"{var_name} not set (optional)")

# ─── Ensure Required API Keys ───────────────────────────────────────────────
# Only prompt for keys that are actually required for the workflow

# LangSmith key (required for tracing)
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    _ensure_env_var("LANGCHAIN_API_KEY", required=False)

# Tavily key (required for web search - Agent 2)
_ensure_env_var("TAVILY_API_KEY", required=False)

# Ollama doesn't need an API key, just needs to be running
logger.info(f"Ollama base URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")

# ─── Validate Ollama Connection ─────────────────────────────────────────────
def check_ollama_connection() -> bool:
    """
    Check if Ollama server is accessible.
    
    Returns
    -------
    bool
        True if Ollama is running and accessible
    """
    import requests
    
    base_url = (
        os.getenv("SFSA_OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or settings.ollama_base_url
    )
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("Ollama server is accessible")
            
            # Check if the required model is available
            model_name = (
                os.getenv("SFSA_OLLAMA_MODEL")
                or os.getenv("OLLAMA_MODEL")
                or settings.ollama_model
            )
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            
            if any(model_name in name for name in model_names):
                logger.info(f"Model {model_name} is available")
                return True
            else:
                logger.error(f"Model {model_name} not found in Ollama")
                logger.error(f"  Available models: {', '.join(model_names)}")
                logger.error(f"  Run: ollama pull {model_name}")
                return False
        else:
            logger.error(f"Ollama server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Cannot connect to Ollama at {base_url}")
        logger.error(f"  Error: {e}")
        logger.error("  Make sure Ollama is running: https://ollama.ai/")
        return False

# Note: Ollama check removed from module import to avoid startup errors
# The actual Ollama connection is checked and managed by chat.py and cli.py
