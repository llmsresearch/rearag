from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This interface allows the ReaRAG implementation to work with different LLM providers
    (such as OpenAI, Anthropic, or local models) through a common API.
    """
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM based on the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The generated response as a string
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the underlying model.
        
        Returns:
            The model name as a string
        """
        pass


class OpenAIProvider(LLMProvider):
    """
    LLM provider implementation for OpenAI models.
    """
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None, **kwargs):
        """
        Initialize the OpenAI provider.
        
        Args:
            model_name: The name of the OpenAI model to use
            api_key: OpenAI API key (optional, will use env var if not provided)
            **kwargs: Additional arguments to pass to the OpenAI API
        """
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai package to use OpenAIProvider: pip install openai")
        
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)
        self.kwargs = kwargs
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response using OpenAI API.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The generated response as a string
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.kwargs
        )
        return response.choices[0].message.content
    
    def get_model_name(self) -> str:
        """
        Get the name of the OpenAI model.
        
        Returns:
            The model name as a string
        """
        return self.model_name


class AnthropicProvider(LLMProvider):
    """
    LLM provider implementation for Anthropic models.
    """
    
    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: Optional[str] = None, **kwargs):
        """
        Initialize the Anthropic provider.
        
        Args:
            model_name: The name of the Anthropic model to use
            api_key: Anthropic API key (optional, will use env var if not provided)
            **kwargs: Additional arguments to pass to the Anthropic API
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic package to use AnthropicProvider: pip install anthropic")
        
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)
        self.kwargs = kwargs
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response using Anthropic API.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The generated response as a string
        """
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.kwargs
        )
        return response.content[0].text
    
    def get_model_name(self) -> str:
        """
        Get the name of the Anthropic model.
        
        Returns:
            The model name as a string
        """
        return self.model_name


class HuggingFaceProvider(LLMProvider):
    """
    LLM provider implementation for Hugging Face models.
    """
    
    def __init__(
        self, 
        model_name: str, 
        api_key: Optional[str] = None,
        use_pipeline: bool = True,
        **kwargs
    ):
        """
        Initialize the Hugging Face provider.
        
        Args:
            model_name: The name or path of the Hugging Face model to use
            api_key: Hugging Face API key if using Inference API
            use_pipeline: Whether to use the pipeline API for local models
            **kwargs: Additional arguments to pass to the model
        """
        self.model_name = model_name
        self.use_pipeline = use_pipeline
        self.kwargs = kwargs
        
        if use_pipeline:
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError("Please install transformers package to use HuggingFaceProvider with pipeline: pip install transformers")
            
            self.pipeline = pipeline("text-generation", model=model_name, **kwargs)
        else:
            try:
                from huggingface_hub import InferenceClient
            except ImportError:
                raise ImportError("Please install huggingface_hub package to use HuggingFaceProvider with InferenceAPI: pip install huggingface_hub")
            
            self.client = InferenceClient(api_key=api_key)
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response using Hugging Face model.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The generated response as a string
        """
        if self.use_pipeline:
            response = self.pipeline(
                prompt,
                **self.kwargs
            )
            return response[0]["generated_text"][len(prompt):]
        else:
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                **self.kwargs
            )
            return response
    
    def get_model_name(self) -> str:
        """
        Get the name of the Hugging Face model.
        
        Returns:
            The model name as a string
        """
        return self.model_name 