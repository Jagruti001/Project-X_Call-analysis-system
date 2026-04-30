"""Test Diarization Agent with LangChain."""

from core.llm_client import OllamaLLMClient
from agents.diarization_agent import DiarizationAgent

def test_diarization():
    """Test diarization agent."""
    
    print("="*60)
    print("DIARIZATION AGENT TEST (LangChain)")
    print("="*60 + "\n")
    
    # Sample transcript
    sample_transcript = """Hello thank you for calling support how can I help you today. 
My internet has been really slow for the past two days. 
I understand let me check your account can you provide your account number. 
Yes it's 12345."""
    
    print("Sample transcript:")
    print(sample_transcript)
    print("\n" + "-"*60 + "\n")
    
    # Initialize
    print("Initializing agent...")
    client = OllamaLLMClient()
    agent = DiarizationAgent(client)
    print("✅ Agent initialized\n")
    
    # Label speakers
    print("Labeling speakers...")
    result = agent.label_speakers(sample_transcript)
    
    print("\nLabeled transcript:")
    print(result['labeled_transcript'])
    print("\n" + "-"*60)
    
    print(f"\nMetrics:")
    print(f"  Agent turns: {result['agent_turns']}")
    print(f"  Customer turns: {result['customer_turns']}")
    print(f"  Total turns: {result['total_turns']}")
    
    # Validate
    validation = agent.validate_diarization(result['labeled_transcript'])
    print(f"\nValidation:")
    print(f"  Coverage: {validation['coverage']:.1%}")
    print(f"  Alternation rate: {validation['alternation_rate']:.1%}")
    print(f"  Valid: {validation['valid']}")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_diarization()
