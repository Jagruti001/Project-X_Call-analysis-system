"""
Modern LangChain LLM Client with Structured Output.

This module provides a clean interface to interact with Ollama's Llama 3.1 model
using the latest LangChain features including:
- ChatOllama (modern, not deprecated)
- Pydantic models for guaranteed JSON output
- Proper error handling
- Type safety
"""

from langchain_ollama import ChatOllama  # Modern, not deprecated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Optional, Type
from loguru import logger


class OllamaLLMClient:
    """
    Clean LangChain client for Ollama Llama 3.1.
    
    Uses ChatOllama (modern) instead of deprecated Ollama class.
    Supports Pydantic models for guaranteed structured output.
    
    Example:
        >>> client = OllamaLLMClient()
        >>> response = client.generate("Hello")
        >>> print(response)
    """
    
    def __init__(
        self, 
        model: str = "qwen2.5:3b",
        temperature: float = 0.3,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama client with qwen2.5:3b.
        
        Args:
            model: Model name (default: "qwen2.5:3b")
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            base_url: Ollama server URL (default: localhost:11434)
            
        Note:
            Make sure Ollama is running: `ollama serve`
            And model is pulled: `ollama pull qwen2.5:3b`
        """
        self.model_name = model
        
        # Modern ChatOllama (not deprecated)
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            format="json"  # Force JSON output when needed
        )
        
        logger.info(f"✅ Ollama client initialized: {model}")
    
    def generate_text(
        self, 
        user_message: str, 
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate plain text response.
        
        Args:
            user_message: Your question/prompt
            system_message: Instructions for the model (optional)
            
        Returns:
            Generated text string
            
        Example:
            >>> client = OllamaLLMClient()
            >>> response = client.generate_text(
            ...     user_message="What is AI?",
            ...     system_message="You are a helpful teacher."
            ... )
        """
        try:
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
            
            # Extract text from AIMessage
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""
    
    def generate_structured(
        self,
        user_message: str,
        response_model: Type[BaseModel],
        system_message: Optional[str] = None
    ) -> Optional[BaseModel]:
        """
        Generate structured output using Pydantic models.
        
        This GUARANTEES valid JSON matching your schema.
        Uses Pydantic for validation - no manual parsing needed.
        
        Args:
            user_message: Your question/prompt
            response_model: Pydantic model class defining output structure
            system_message: Instructions for the model (optional)
            
        Returns:
            Validated Pydantic model instance or None if failed
            
        Example:
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            ...
            >>> client = OllamaLLMClient()
            >>> result = client.generate_structured(
            ...     user_message="Extract: John is 25 years old",
            ...     response_model=Person
            ... )
            >>> print(result.name)  # "John"
            >>> print(result.age)   # 25
        """
        try:
            # Create parser for this Pydantic model
            parser = PydanticOutputParser(pydantic_object=response_model)
            
            # Build prompt with format instructions
            base_system = system_message or "You are a helpful assistant."
            
            format_instructions = parser.get_format_instructions()

            full_system = (
                base_system
                + "\n\n"
                + format_instructions
                + "\n\nCRITICAL: Respond ONLY with valid JSON. No explanations, no markdown."
            )
       
            prompt = ChatPromptTemplate.from_messages([
    ("system", base_system),
    ("human", "{input}\n\n{format_instructions}")
])
            
            # Build chain: prompt -> llm -> parser
            chain = prompt | self.llm | parser
            
            # Execute and validate
            result = chain.invoke({
    "input": user_message,
    "format_instructions": parser.get_format_instructions()
})
            
            logger.info(f"✅ Structured output generated: {response_model.__name__}")
            return result
            
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            return None
    
    def generate_json_dict(
        self,
        user_message: str,
        system_message: Optional[str] = None
    ) -> Dict:
        """
        Generate raw JSON dictionary (without Pydantic validation).
        
        Use generate_structured() instead for better type safety.
        This is a fallback for dynamic schemas.
        
        Args:
            user_message: Your question/prompt
            system_message: Instructions for the model (optional)
            
        Returns:
            Dictionary parsed from JSON response
        """
        try:
            # Force JSON format
            llm_json = ChatOllama(
                model=self.model_name,
                temperature=0.3,
                format="json"
            )
            
            base_system = system_message or "You are a helpful assistant."
            full_system = f"""{base_system}

Respond ONLY with valid JSON. No explanations, no markdown, just JSON."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", full_system),
                ("human", "{input}")
            ])
            
            chain = prompt | llm_json
            response = chain.invoke({"input": user_message})
            
            # Parse JSON from content
            import json
            return json.loads(response.content)
            
        except Exception as e:
            logger.error(f"JSON generation failed: {e}")
            return {}