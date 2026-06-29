import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Verify that critical configuration keys are present
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "Critical Error: GROQ_API_KEY environment variable is missing. "
        "Please define it in your .env file."
    )

# Retrieve model name with a default fallback
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192")


def get_llm(temperature: float = 0.0, **kwargs) -> ChatGroq:
    """
    Factory function to initialize and retrieve a ChatGroq instance.
    
    Args:
        temperature (float): Controls response randomness (0.0 is deterministic).
        **kwargs: Additional configuration parameters for ChatGroq.
        
    Returns:
        ChatGroq: Initialized LangChain ChatGroq model wrapper.
        
    Raises:
        RuntimeError: If initialization fails.
    """
    try:
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL_NAME,
            temperature=temperature,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize ChatGroq LLM (Model: {GROQ_MODEL_NAME}). "
            f"Error details: {e}"
        ) from e


# Global reusable instance with default temperature
llm = get_llm(temperature=0.0)
