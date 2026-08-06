# src/sfsa_rag_assistant/generation.py
import torch
import logging
from typing import Optional
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TextGenerator:
    """
    Generates text responses using a specified LLM model.
    
    Supports three loading modes:
    - "ollama": Use Ollama for local model inference (recommended)
    - "api": Use HuggingFace Inference API
    - "local": Load HuggingFace model locally with transformers
    """

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        device: Optional[str] = None,
        load_mode: str = "ollama",
        api_token: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initializes the TextGenerator with a model name, device setting, and load mode.

        Parameters
        ----------
        model_name : str
            Model identifier. For Ollama: "llama3.1:8b", for HF: "meta-llama/..."
        device : Optional[str]
            Device for local models ("cuda", "cpu"). Only used for load_mode="local".
        load_mode : str
            Loading mode: "ollama" (default), "api" (HF API), or "local" (HF local)
        api_token : Optional[str]
            HuggingFace API token, required if load_mode is "api"
        base_url : str
            Ollama server URL (default: http://localhost:11434)
        temperature : float
            Generation temperature, 0.0-1.0 (default: 0.3)
        max_tokens : int
            Maximum tokens to generate (default: 1000)
        """
        self.model_name = model_name
        self.load_mode = load_mode
        self.api_token = api_token
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.llm = None
        
        logger.info(f"Initialized TextGenerator: mode={load_mode}, model={model_name}")

    def load_model(self):
        """
        Loads the LLM model and sets up the text generation pipeline.
        
        Supports three loading modes:
        - "ollama": Connect to Ollama server (recommended)
        - "api": Use HuggingFace Inference API
        - "local": Load HuggingFace model locally

        Raises
        ------
        ValueError
            If load_mode is "api" and api_token is not provided
        RuntimeError
            If load_mode is "ollama" and Ollama server is not accessible
        """
        if self.load_mode == "ollama":
            logger.info(f"Loading Ollama model: {self.model_name}")
            
            # Check Ollama connection first
            if not self._check_ollama_health():
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Make sure Ollama is running and the model '{self.model_name}' is pulled."
                )
            
            # Initialize ChatOllama
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
            )
            
            logger.info(f"Ollama model loaded: {self.model_name}")
            
        elif self.load_mode == "api":
            if not self.api_token:
                raise ValueError("Hugging Face API token is required for 'api' load mode.")
            # Use HuggingFaceEndpoint for loading the model via API
            self.llm = HuggingFaceEndpoint(
                endpoint_url=f"https://api-inference.huggingface.co/models/{self.model_name}",
                huggingfacehub_api_token=self.api_token,
                task="text-generation", 
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=1000,
                # timeout = 600
            )
        else:
            # Load the model and tokenizer locally
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Set pad_token_id to eos_token_id to avoid warnings
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            # Define terminators for stopping generation at end of response
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # Set up the text generation pipeline
            text_generation_pipeline = pipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                task="text-generation",
                device=self.device,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=1000,
                eos_token_id=terminators
            )

            # Wrap the pipeline for LangChain compatibility
            self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            
            logger.info(f"Local HuggingFace model loaded: {self.model_name}")
    
    def _check_ollama_health(self) -> bool:
        """
        Check if Ollama server is accessible and model is available.
        
        Returns
        -------
        bool
            True if Ollama is accessible and model exists
        """
        import requests
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                # Check if our model is available
                if any(self.model_name in name for name in model_names):
                    logger.info(f"Ollama model '{self.model_name}' is available")
                    return True
                else:
                    logger.error(f"Model '{self.model_name}' not found in Ollama")
                    logger.error(f"  Available models: {', '.join(model_names)}")
                    logger.error(f"  Run: ollama pull {self.model_name}")
                    return False
            else:
                logger.error(f"Ollama server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            logger.error(f"  Error: {e}")
            return False
    
    def invoke(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.
        
        Parameters
        ----------
        prompt : str
            The input prompt for generation
        
        Returns
        -------
        str
            Generated response text
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # For ChatOllama, use invoke with proper message format
        if self.load_mode == "ollama":
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        else:
            # For HF endpoints, direct invoke
            return self.llm.invoke(prompt)
    
    def with_structured_output(self, schema: type[BaseModel]):
        """
        Configure the LLM to output structured data matching a Pydantic schema.
        
        This is used by Agent 1, 2, and 3 to ensure consistent output format.
        
        Parameters
        ----------
        schema : type[BaseModel]
            Pydantic model class defining the expected output structure
        
        Returns
        -------
        Runnable
            LLM configured for structured output
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.load_mode == "ollama":
            # Ollama supports structured outputs via with_structured_output
            return self.llm.with_structured_output(schema)
        else:
            # For HF, we'll need to parse JSON manually (not ideal)
            logger.warning("Structured output not natively supported for HF mode")
            return self.llm


# Usage examples:
# 
# Ollama (recommended):
#   generator = TextGenerator(model_name="llama3.1:8b", load_mode="ollama")
#   generator.load_model()
#   response = generator.invoke("What is steel casting?")
#
# HuggingFace API:
#   generator = TextGenerator(
#       model_name="meta-llama/Llama-3.2-3B-Instruct",
#       load_mode="api",
#       api_token="hf_..."
#   )
#   generator.load_model()
#   response = generator.invoke("What is steel casting?")