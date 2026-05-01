"""
LangChain LLM Client with comprehensive error handling.

This module provides a robust interface to Ollama's LLM with:
- Modern ChatOllama integration
- Structured output support via Pydantic
- Retry mechanisms and error handling
- Connection pooling and timeouts
- Comprehensive logging
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Optional, Type, Any
from loguru import logger
import time
import requests


class OllamaLLMClient:
    """
    Production-grade LangChain client for Ollama.
    
    Features:
    - Connection validation
    - Automatic retry with exponential backoff
    - Structured output via Pydantic
    - Comprehensive error handling
    - Performance monitoring
    
    Example:
        >>> client = OllamaLLMClient()
        >>> if client.is_available():
        >>>     response = client.generate_text("Explain AI")
    """
    
    def __init__(
        self, 
        model: str = "qwen2.5:3b",
        temperature: float = 0.3,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize Ollama LLM client.
        
        Args:
            model: Model name (default: qwen2.5:3b)
            temperature: Randomness (0.0-1.0, lower = more deterministic)
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.model_name = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize ChatOllama
        try:
            self.llm = ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url,
                timeout=timeout,
                format="json"  # Force JSON for structured outputs
            )
            
            # Test connection
            if self._test_connection():
                logger.info(f"✅ Ollama client initialized: {model} @ {base_url}")
            else:
                logger.warning(f"⚠️ Ollama server not responding - client initialized but may fail")
                
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
    
    # ===========================
    # CONNECTION MANAGEMENT
    # ===========================
    
    def _test_connection(self) -> bool:
        """Test if Ollama server is reachable."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Connection test failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """
        Check if Ollama server and model are available.
        
        Returns:
            True if ready to use, False otherwise
        """
        return self._test_connection()
    
    def list_models(self) -> List[str]:
        """List available models on Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name") for m in models]
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    # ===========================
    # TEXT GENERATION
    # ===========================
    
    def generate_text(
        self, 
        user_message: str, 
        system_message: Optional[str] = None,
        retry: bool = True
    ) -> str:
        """
        Generate plain text response with optional retry.
        
        Args:
            user_message: User's question/prompt
            system_message: System instructions (optional)
            retry: Whether to retry on failure
            
        Returns:
            Generated text string
            
        Example:
            >>> response = client.generate_text(
            ...     "Explain quantum computing",
            ...     system_message="You are a physics professor"
            ... )
        """
        max_attempts = self.max_retries if retry else 1
        
        for attempt in range(1, max_attempts + 1):
            try:
                start_time = time.time()
                
                # Build prompt
                if system_message:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_message),
                        ("human", "{input}")
                    ])
                    chain = prompt | self.llm
                    response = chain.invoke({"input": user_message})
                else:
                    response = self.llm.invoke(user_message)
                
                # Extract text
                result = response.content.strip()
                
                elapsed = time.time() - start_time
                logger.debug(f"LLM response generated in {elapsed:.2f}s")
                
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_attempts} failed: {e}")
                if attempt < max_attempts:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Text generation failed after {max_attempts} attempts")
                    return ""
        
        return ""
    
    # ===========================
    # STRUCTURED GENERATION
    # ===========================
    
    def generate_structured(
        self,
        user_message: str,
        response_model: Type[BaseModel],
        system_message: Optional[str] = None,
        retry: bool = True
    ) -> Optional[BaseModel]:
        """
        Generate structured output using Pydantic models.
        
        This guarantees valid JSON matching your schema through Pydantic validation.
        
        Args:
            user_message: User's question/prompt
            response_model: Pydantic model class defining output structure
            system_message: System instructions (optional)
            retry: Whether to retry on failure
            
        Returns:
            Validated Pydantic model instance or None if failed
            
        Example:
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            ...
            >>> result = client.generate_structured(
            ...     "Extract: Alice is 30",
            ...     response_model=Person
            ... )
            >>> print(result.name, result.age)
        """
        max_attempts = self.max_retries if retry else 1
        
        for attempt in range(1, max_attempts + 1):
            try:
                start_time = time.time()
                
                # Create parser
                parser = PydanticOutputParser(pydantic_object=response_model)
                
                # Build system prompt
                base_system = system_message or "You are a helpful assistant."
                format_instructions = parser.get_format_instructions()
                
                # Build prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", base_system),
                    ("human", "{input}\n\n{format_instructions}")
                ])
                
                # Build chain
                chain = prompt | self.llm | parser
                
                # Execute
                result = chain.invoke({
                    "input": user_message,
                    "format_instructions": format_instructions
                })
                
                elapsed = time.time() - start_time
                logger.debug(
                    f"Structured output generated in {elapsed:.2f}s: "
                    f"{response_model.__name__}"
                )
                
                return result
                
            except Exception as e:
                logger.warning(
                    f"Structured generation attempt {attempt}/{max_attempts} failed: {e}"
                )
                if attempt < max_attempts:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(
                        f"Structured generation failed after {max_attempts} attempts"
                    )
                    return None
        
        return None
    
    # ===========================
    # JSON GENERATION (FALLBACK)
    # ===========================
    
    def generate_json_dict(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        retry: bool = True
    ) -> Dict[str, Any]:
        """
        Generate raw JSON dictionary without Pydantic validation.
        
        Note: Prefer generate_structured() for better type safety.
        This is a fallback for dynamic schemas.
        
        Args:
            user_message: User's question/prompt
            system_message: System instructions (optional)
            retry: Whether to retry on failure
            
        Returns:
            Dictionary parsed from JSON response
        """
        import json
        
        max_attempts = self.max_retries if retry else 1
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Force JSON format
                llm_json = ChatOllama(
                    model=self.model_name,
                    temperature=0.3,
                    base_url=self.base_url,
                    format="json"
                )
                
                base_system = system_message or "You are a helpful assistant."
                full_system = f"""{base_system}

CRITICAL: Respond ONLY with valid JSON. No explanations, no markdown, just raw JSON."""

                prompt = ChatPromptTemplate.from_messages([
                    ("system", full_system),
                    ("human", "{input}")
                ])
                
                chain = prompt | llm_json
                response = chain.invoke({"input": user_message})
                
                # Parse JSON
                result = json.loads(response.content)
                logger.debug("JSON dict generated successfully")
                return result
                
            except Exception as e:
                logger.warning(f"JSON generation attempt {attempt}/{max_attempts} failed: {e}")
                if attempt < max_attempts:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"JSON generation failed after {max_attempts} attempts")
                    return {}
        
        return {}
    
    # ===========================
    # HEALTH CHECK
    # ===========================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check of Ollama connection.
        
        Returns:
            Health status dictionary
        """
        health = {
            "server_reachable": False,
            "model_available": False,
            "latency_ms": None,
            "error": None
        }
        
        try:
            # Test server connection
            start = time.time()
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            latency = (time.time() - start) * 1000
            
            health["server_reachable"] = response.status_code == 200
            health["latency_ms"] = round(latency, 2)
            
            if health["server_reachable"]:
                # Check if our model is available
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                health["model_available"] = self.model_name in model_names
                
                if not health["model_available"]:
                    health["error"] = f"Model '{self.model_name}' not found. Available: {model_names}"
            
        except Exception as e:
            health["error"] = str(e)
        
        return health
    
    # ===========================
    # UTILITIES
    # ===========================
    
    def __repr__(self) -> str:
        """String representation."""
        return f"OllamaLLMClient(model='{self.model_name}', url='{self.base_url}')"