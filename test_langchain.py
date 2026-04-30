"""Verify LangChain integration works correctly."""

from core.llm_client import OllamaLLMClient
from loguru import logger
from pydantic import BaseModel

def test_langchain_client():
    """Test basic LangChain client functionality."""
    
    print("="*60)
    print("LANGCHAIN INTEGRATION TEST")
    print("="*60 + "\n")
    
    # Initialize client
    print("1. Initializing LangChain LLM client...")
    try:
        client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:3b",
            temperature=0.3
        )
        print("✅ Client initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}\n")
        return False
    
    # Test basic generation
    print("2. Testing basic text generation...")
    try:
        response = client.generate_text("Say 'LangChain works!' in one sentence.")
        if response:
            print(f"✅ Response: {response}\n")
        else:
            print("❌ Empty response\n")
            return False
    except Exception as e:
        print(f"❌ Generation failed: {e}\n")
        return False

    # Test structured output
    print("3. Testing structured JSON output...")

    class TestOutput(BaseModel):
        status: str
        framework: str
        test: bool

    try:
        prompt = """Generate a JSON object with these fields:
        - status: "working"
        - framework: "langchain"
        - test: true"""
        
        result = client.generate_structured(
            user_message=prompt,
            response_model=TestOutput,
            system_message="Return only valid JSON."
        )
        
        if result is not None:
            print("✅ Structured output parsed successfully:")
            print(result.model_dump())
        else:
            print("❌ Structured output failed")
            return False
    except Exception as e:
        print(f"❌ Structured output failed: {e}\n")
        return False

    print("="*60)
    print("✅ ALL TESTS PASSED - LangChain integration working!")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    test_langchain_client()