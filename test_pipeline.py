"""Test all agents with LangChain integration."""

from core.llm_client import OllamaLLMClient
from agents.diarization_agent import DiarizationAgent
from agents.unified_analysis_agent import UnifiedAnalysisAgent
from agents.insight_agent import InsightAgent
from agents.root_cause_agent import RootCauseAgent
import json

def test_all_agents():
    """Test complete agent pipeline with LangChain."""
    
    print("="*70)
    print("COMPLETE LANGCHAIN AGENT PIPELINE TEST")
    print("="*70 + "\n")
    
    # Initialize
    print("1. Initializing LangChain client and agents...")
    client = OllamaLLMClient()
    
    diarization_agent = DiarizationAgent(client)
    analysis_agent = UnifiedAnalysisAgent(client)
    insight_agent = InsightAgent(client)
    root_cause_agent = RootCauseAgent()
    
    print("✅ All agents initialized\n")
    
    # Sample data
    sample_transcript = """Hello thank you for calling support how can I help you today. 
My internet has been really slow for the past two days and I'm really frustrated. 
I understand that must be frustrating let me check your account can you provide your account number. 
Yes it's 12345 I just want this fixed as soon as possible. 
I see the issue there was a network outage in your area yesterday we're working to resolve it. 
Okay when will it be fixed. 
The network team expects full resolution by end of day today. 
Fine thank you."""
    
    print("="*70)
    print("STEP 1: Diarization")
    print("="*70)
    print("\nRaw transcript:")
    print(sample_transcript[:200] + "...\n")
    
    diarization_result = diarization_agent.label_speakers(sample_transcript)
    labeled_transcript = diarization_result['labeled_transcript']
    
    print("Labeled transcript:")
    print(labeled_transcript[:300] + "...\n")
    print(f"Turns: {diarization_result['total_turns']} (Agent: {diarization_result['agent_turns']}, Customer: {diarization_result['customer_turns']})")
    
    # Analysis
    print("\n" + "="*70)
    print("STEP 2: Unified Analysis")
    print("="*70 + "\n")
    
    analysis = analysis_agent.analyze(labeled_transcript, "test_call_001")
    
    print(f"Intent: {analysis.get('intent')}")
    print(f"Sentiment: {analysis.get('sentiment', {}).get('overall')}")
    print(f"Resolution: {analysis.get('resolution_status')}")
    print(f"Summary: {analysis.get('summary', 'N/A')[:100]}...")
    print(f"Issues: {analysis.get('key_issues', [])}")
    
    # Mock data for insights (since we need multiple calls)
    print("\n" + "="*70)
    print("STEP 3: Root Cause & Insights (mock data)")
    print("="*70 + "\n")
    
    # Create mock analyses
    mock_analyses = [analysis] * 5  # Simulate 5 similar calls
    for i, a in enumerate(mock_analyses):
        a['call_id'] = f'test_call_{i:03d}'
    
    # Mock embeddings data
    import numpy as np
    mock_embeddings = {
        'ids': [f'test_call_{i:03d}' for i in range(5)],
        'embeddings': np.random.rand(5, 384).tolist(),
        'documents': [analysis.get('summary', '')] * 5,
        'metadatas': [
            {
                'call_id': f'test_call_{i:03d}',
                'intent': analysis.get('intent', 'unknown'),
                'sentiment': analysis.get('sentiment', {}).get('overall', 'neutral')
            }
            for i in range(5)
        ]
    }
    
    # Root cause
    root_cause_result = root_cause_agent.analyze_root_causes(
        mock_embeddings, 
        mock_analyses
    )
    
    print(f"Clusters found: {root_cause_result.get('num_clusters', 0)}")
    print(f"Silhouette score: {root_cause_result.get('silhouette_score', 0):.3f}")
    print(f"Quality: {root_cause_result.get('clustering_quality', 'Unknown')}")
    
    # Insights
    print("\nGenerating insights...")
    insights = insight_agent.generate_insights(root_cause_result, mock_analyses)
    
    print(f"\nKey Insights:")
    for idx, insight in enumerate(insights.get('key_insights', [])[:3], 1):
        print(f"{idx}. {insight}")
    
    print(f"\nRecommendations:")
    for rec in insights.get('recommendations', [])[:2]:
        print(f"  [{rec['priority']}] {rec['area']}: {rec['action']}")
    
    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE TEST PASSED")
    print("="*70)
    print("\nAll agents working with LangChain!")
    print("Structure ready for LangGraph integration.\n")

if __name__ == "__main__":
    test_all_agents()
