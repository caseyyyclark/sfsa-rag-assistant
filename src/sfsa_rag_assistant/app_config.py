"""
Application settings and configuration using Pydantic.

This module defines all configuration options for the SFSA Agentic RAG system.
Settings can be overridden via environment variables prefixed with SFSA_.

Example:
    SFSA_OLLAMA_MODEL=llama3.1:70b python -m sfsa_rag_assistant

Based on the KnowMat2 settings pattern.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for SFSA Agentic RAG.
    
    All settings can be overridden via environment variables with SFSA_ prefix.
    Example: SFSA_OLLAMA_MODEL=llama3.1:70b
    
    Attributes
    ----------
    ollama_base_url : str
        Base URL for Ollama server (default: http://localhost:11434)
    ollama_model : str
        Ollama model name to use (default: llama3.1:8b)
    ollama_temperature : float
        Temperature for LLM generation (0.0-1.0, default: 0.3)
    ollama_max_tokens : int
        Maximum tokens for LLM generation (default: 1000)
    
    vectordb_path : str
        Path to FAISS vector database directory (default: src/sfsa_rag_assistant/data/vectordb)
    embedding_model : str
        HuggingFace embedding model name (default: Alibaba-NLP/gte-large-en-v1.5)
    retrieval_k : int
        Number of documents to retrieve from vector DB (default: 5)
    
    tavily_api_key : Optional[str]
        Tavily API key for web search (default: None, loaded from TAVILY_API_KEY)
    tavily_max_results : int
        Maximum number of web search results from Tavily (default: 3)
    
    max_validation_attempts : int
        Maximum number of validation loops for Agent 3 (default: 2)
    
    output_dir : str
        Directory for saving outputs (default: data/outputs)
    
    langsmith_project : str
        LangSmith project name for tracing (default: SFSA-Agentic-RAG)
    """
    
    # ─── Ollama Configuration ───────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_temperature: float = 0.3
    ollama_max_tokens: int = 1000
    
    # ─── Vector Database Configuration ──────────────────────────────────────
    vectordb_path: str = "src/sfsa_rag_assistant/data/vectordb"
    embedding_model: str = "Alibaba-NLP/gte-large-en-v1.5"
    retrieval_k: int = 5
    
    # ─── Web Search Configuration ───────────────────────────────────────────
    tavily_api_key: Optional[str] = Field(
        default=None,
        validation_alias="TAVILY_API_KEY"  # Override SFSA_ prefix for this field
    )
    tavily_max_results: int = 3
    
    # ─── Agent Configuration ────────────────────────────────────────────────
    max_validation_attempts: int = 2
    
    # ─── Output Configuration ───────────────────────────────────────────────
    output_dir: str = "data/outputs"

    # ─── Authentication Configuration ───────────────────────────────────────
    auth_db_path: str = "data/auth/sfsa_auth.db"
    
    # ─── LangSmith Configuration ────────────────────────────────────────────
    langsmith_project: str = "SFSA-Agentic-RAG"
    
    # ─── Pydantic Configuration ─────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_prefix="SFSA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra env vars
    )
    
    def get_vectordb_path(self) -> Path:
        """
        Get the vector database path as a Path object.
        
        Returns
        -------
        Path
            Absolute path to vector database directory
        """
        path = Path(self.vectordb_path)
        # If relative, make it relative to project root
        if not path.is_absolute():
            # Try to find project root (where .env is)
            from pathlib import Path as P
            current = P.cwd()
            for parent in [current] + list(current.parents):
                if (parent / ".env").exists() or (parent / "pyproject.toml").exists():
                    return (parent / self.vectordb_path).resolve()
            # Fallback to relative from cwd
            return path.resolve()
        return path
    
    def get_output_dir(self) -> Path:
        """
        Get the output directory path as a Path object, creating it if needed.
        
        Returns
        -------
        Path
            Absolute path to output directory
        """
        path = Path(self.output_dir)
        if not path.is_absolute():
            path = Path.cwd() / self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_auth_db_path(self) -> Path:
        """
        Get the authentication database path as a Path object, creating parent directories.

        Returns
        -------
        Path
            Absolute path to the authentication SQLite database
        """
        path = Path(self.auth_db_path)
        if not path.is_absolute():
            path = Path.cwd() / self.auth_db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def validate_ollama_settings(self) -> bool:
        """
        Validate that Ollama settings are correct.
        
        Returns
        -------
        bool
            True if Ollama is accessible with the configured settings
        """
        import requests
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                return any(self.ollama_model in name for name in model_names)
        except Exception:
            pass
        return False
    
    def __repr__(self) -> str:
        """String representation showing key settings."""
        return (
            f"Settings(\n"
            f"  Ollama: {self.ollama_model} @ {self.ollama_base_url}\n"
            f"  Vector DB: {self.vectordb_path}\n"
            f"  Retrieval: top-{self.retrieval_k} docs\n"
            f"  Web Search: max {self.tavily_max_results} results\n"
            f"  Validation: max {self.max_validation_attempts} attempts\n"
            f")"
        )


# ─── Singleton Settings Instance ────────────────────────────────────────────
# Import this singleton throughout the application
settings = Settings()

# Print settings on first import (for debugging)
if __name__ != "__main__":
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded settings: {settings}")
