"""
Example script demonstrating how to use the Call Analysis System.

This script shows:
1. Processing a single call
2. Processing multiple calls
3. Generating insights
4. Searching for similar issues
"""

from utils.config import Config, ensure_directories
from core.orchestrator import CallAnalysisOrchestrator
from loguru import logger
import json


def main():
    """Main example workflow."""
    
    # Step 1: Initialize the system
    logger.info("Initializing Call Analysis System...")
    ensure_directories()
    config = Config()
    orchestrator = CallAnalysisOrchestrator(config)
    
    print("\n" + "="*60)
    print("CALL ANALYSIS SYSTEM - DEMO")
    print("="*60 + "\n")
    
    # Step 2: Check system status
    stats = orchestrator.get_storage_stats()
    print(f"📊 Current Stats:")
    print(f"   - Transcripts: {stats['total_transcripts']}")
    print(f"   - Analyses: {stats['total_analyses']}")
    print(f"   - Embeddings: {stats['total_embeddings']}\n")
    
    # Step 3: Process a single call (EXAMPLE - replace with your audio file)
    print("="*60)
    print("EXAMPLE 1: Process Single Call")
    print("="*60 + "\n")
    
    # NOTE: Replace this with an actual audio file path
    example_audio = "data/audio/sample_call.wav"
    
    print(f"⚠️  To process a call, place an audio file at: {example_audio}")
    print(f"   Then uncomment the processing code below.\n")
    
    # Uncomment to process:
    # result = orchestrator.process_single_call(example_audio)
    # 
    # if result['status'] == 'success':
    #     print(f"✅ Call processed: {result['call_id']}\n")
    #     
    #     print("📝 Analysis:")
    #     analysis = result['analysis']
    #     print(f"   Intent: {analysis['intent']}")
    #     print(f"   Sentiment: {analysis['sentiment']['overall']}")
    #     print(f"   Summary: {analysis['summary']}")
    #     print(f"   Issues: {', '.join(analysis['key_issues'])}\n")
    # else:
    #     print(f"❌ Error: {result['error']}\n")
    
    # Step 4: Generate insights (only if calls are processed)
    print("="*60)
    print("EXAMPLE 2: Generate Insights")
    print("="*60 + "\n")
    
    if stats['total_analyses'] > 0:
        print("🧠 Generating insights from all processed calls...")
        insights_data = orchestrator.generate_insights()
        
        if 'error' not in insights_data:
            # Root cause analysis
            root_cause = insights_data['root_cause_analysis']
            print(f"\n🔍 Root Cause Analysis:")
            print(f"   Clusters found: {root_cause['num_clusters']}")
            print(f"   Silhouette score: {root_cause['silhouette_score']:.3f}")
            print(f"   Quality: {root_cause['clustering_quality']}\n")
            
            # Business insights
            business = insights_data['business_insights']
            stats_data = business['statistics']
            
            print(f"📊 Statistics:")
            print(f"   Total calls: {stats_data['total_calls']}")
            print(f"   Negative sentiment: {stats_data['negative_sentiment_rate']*100:.1f}%")
            print(f"   Unresolved rate: {stats_data['unresolved_rate']*100:.1f}%\n")
            
            # Key insights
            print(f"💡 Key Insights:")
            for idx, insight in enumerate(business['key_insights'][:3], 1):
                print(f"   {idx}. {insight}")
            print()
            
            # Recommendations
            print(f"💼 Top Recommendations:")
            for idx, rec in enumerate(business['recommendations'][:2], 1):
                print(f"   {idx}. [{rec['priority']}] {rec['area']}")
                print(f"      Action: {rec['action']}")
                print(f"      Impact: {rec['expected_impact']}\n")
            
            # Save insights to file
            with open('insights_output.json', 'w') as f:
                json.dump(insights_data, f, indent=2)
            print(f"💾 Full insights saved to: insights_output.json\n")
        else:
            print(f"❌ Could not generate insights: {insights_data['error']}\n")
    else:
        print("⚠️  No calls processed yet. Process some calls first!\n")
    
    # Step 5: Search for similar issues
    print("="*60)
    print("EXAMPLE 3: Search Similar Issues")
    print("="*60 + "\n")
    
    if stats['total_embeddings'] > 0:
        query = "slow internet connection"
        print(f"🔍 Searching for issues similar to: '{query}'\n")
        
        results = orchestrator.search_similar_issues(query, n_results=3)
        
        if results['ids']:
            print(f"Found {len(results['ids'][0])} similar issues:\n")
            
            for idx, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                similarity = (1 - distance) * 100
                print(f"{idx}. Similarity: {similarity:.1f}%")
                print(f"   Call ID: {metadata['call_id']}")
                print(f"   Intent: {metadata['intent']}")
                print(f"   Issue: {doc[:100]}...\n")
        else:
            print("No similar issues found\n")
    else:
        print("⚠️  No embeddings in database yet.\n")
    
    print("="*60)
    print("Demo complete! Check the Streamlit UI for more features:")
    print("  streamlit run app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
