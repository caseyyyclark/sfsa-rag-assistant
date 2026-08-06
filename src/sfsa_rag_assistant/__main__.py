# src/sfsa_rag_assistant/__main__.py
"""
Entry point for the sfsa_rag_assistant package.

This module provides the main entry point for running the SFSA Agentic RAG system
as a Python module.

Example:
    Run single query:
        $ python -m sfsa_rag_assistant "What is steel casting?"
    
    Show workflow graph:
        $ python -m sfsa_rag_assistant --show-graph
    
    With metadata:
        $ python -m sfsa_rag_assistant "What is steel casting?" --show-metadata
"""

from sfsa_rag_assistant.cli import main

if __name__ == "__main__":
    main()