"""Streamlit UI for Call Analysis System."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from utils.config import Config, ensure_directories
from core.orchestrator import CallAnalysisOrchestrator

# Page config
st.set_page_config(
    page_title="Call Analysis System",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #000000;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #000000;
    }
    .recommendation-box {
        background-color: #fff4e6;
        padding: 1rem;
        border-left: 4px solid #ff7f0e;
        margin: 1rem 0;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the orchestrator (cached for performance)."""
    ensure_directories()
    config = Config()
    orchestrator = CallAnalysisOrchestrator(config)
    return orchestrator


def main():
    """Main application."""
    
    st.markdown('<p class="main-header">📞 Customer Call Analysis System</p>', 
                unsafe_allow_html=True)
    
    # Initialize system
    orchestrator = initialize_system()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Process Calls", "Insights & Analytics", "Search Similar Issues", "System Stats"]
    )
    
    if page == "Dashboard":
        show_dashboard(orchestrator)
    elif page == "Process Calls":
        show_process_calls(orchestrator)
    elif page == "Insights & Analytics":
        show_insights(orchestrator)
    elif page == "Search Similar Issues":
        show_search(orchestrator)
    elif page == "System Stats":
        show_stats(orchestrator)


def show_dashboard(orchestrator):
    """Dashboard overview."""
    st.header("📊 Dashboard Overview")
    
    # Get stats
    stats = orchestrator.get_storage_stats()
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Calls Processed", stats.get('total_transcripts', 0))
    
    with col2:
        st.metric("Analyses Complete", stats.get('total_analyses', 0))
    
    with col3:
        st.metric("Embeddings Stored", stats.get('total_embeddings', 0))
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Process New Call", use_container_width=True):
            st.switch_page("pages/process_calls.py")
    
    with col2:
        if st.button("📈 View Insights", use_container_width=True):
            st.switch_page("pages/insights.py")
    
    # Recent activity
    st.markdown("---")
    st.subheader("System Status")
    
    if stats.get('total_analyses', 0) > 0:
        st.success("✅ System operational - Ready to process calls")
    else:
        st.info("ℹ️ No calls processed yet - Upload your first audio file!")


def show_process_calls(orchestrator):
    """Process calls page."""
    st.header("🎤 Process Customer Calls")
    
    tab1, tab2 = st.tabs(["Single Call", "Batch Processing"])
    
    with tab1:
        st.subheader("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV, MP3, M4A)",
            type=['wav', 'mp3', 'm4a']
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = Path("data/audio") / uploaded_file.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            if st.button("Process Call", type="primary"):
                with st.spinner("Processing call... This may take a few minutes"):
                    result = orchestrator.process_single_call(str(temp_path))
                
                if result['status'] == 'success':
                    st.success(f"✅ Call processed: {result['call_id']}")
                    
                    # Show results
                    st.markdown("### Results")
                    
                    # Transcript
                    with st.expander("📝 Transcript", expanded=True):
                        st.text_area(
                            "Labeled Transcript",
                            result['labeled_transcript'],
                            height=300
                        )
                    
                    # Analysis
                    with st.expander("🔍 Analysis", expanded=True):
                        analysis = result['analysis']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Intent**")
                            st.info(analysis.get('intent', 'unknown').replace('_', ' ').title())
                            
                            st.markdown("**Resolution**")
                            st.info(analysis.get('resolution_status', 'unknown').title())
                        
                        with col2:
                            st.markdown("**Sentiment**")
                            sentiment = analysis.get('sentiment', {}).get('overall', 'neutral')
                            if sentiment == 'positive':
                                st.success(sentiment.title())
                            elif sentiment == 'negative':
                                st.error(sentiment.title())
                            else:
                                st.warning(sentiment.title())
                        
                        st.markdown("**Summary**")
                        st.write(analysis.get('summary', 'No summary available'))
                        
                        st.markdown("**Key Issues**")
                        issues = analysis.get('key_issues', [])
                        for issue in issues:
                            st.markdown(f"- {issue}")
                else:
                    st.error(f"❌ Processing failed: {result.get('error')}")
    
    with tab2:
        st.subheader("Batch Upload")
        st.info("Upload multiple audio files for batch processing")
        
        uploaded_files = st.file_uploader(
            "Choose audio files",
            type=['wav', 'mp3', 'm4a'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} files selected")
            
            if st.button("Process Batch", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                file_paths = []
                for idx, file in enumerate(uploaded_files):
                    temp_path = Path("data/audio") / file.name
                    with open(temp_path, 'wb') as f:
                        f.write(file.getbuffer())
                    file_paths.append(str(temp_path))
                    
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Uploading {idx + 1}/{len(uploaded_files)}")
                
                status_text.text("Processing...")
                results = orchestrator.process_batch(file_paths)
                
                success_count = sum(1 for r in results if r['status'] == 'success')
                
                st.success(f"✅ Processed {success_count}/{len(results)} calls successfully")


def show_insights(orchestrator):
    """Insights and analytics page."""
    st.header("📈 Insights & Analytics")
    
    if st.button("🔄 Generate Insights", type="primary"):
        with st.spinner("Analyzing all calls... This may take a moment"):
            insights_data = orchestrator.generate_insights()
        
        if 'error' in insights_data:
            st.error(insights_data['error'])
            return
        
        # Store in session state
        st.session_state['insights'] = insights_data
    
    if 'insights' not in st.session_state:
        st.info("ℹ️ Click 'Generate Insights' to analyze processed calls")
        return
    
    insights_data = st.session_state['insights']
    
    # Business Insights
    st.subheader("💡 Key Business Insights")
    
    business_insights = insights_data.get('business_insights', {})
    key_insights = business_insights.get('key_insights', [])
    
    for insight in key_insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Call Statistics")
    
    stats = business_insights.get('statistics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Calls", stats.get('total_calls', 0))
    
    with col2:
        neg_rate = stats.get('negative_sentiment_rate', 0)
        st.metric("Negative Sentiment", f"{neg_rate*100:.1f}%")
    
    with col3:
        unres_rate = stats.get('unresolved_rate', 0)
        st.metric("Unresolved", f"{unres_rate*100:.1f}%")
    
    with col4:
        st.metric("Top Intent", stats.get('top_intent', 'unknown').replace('_', ' ').title())
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        sentiment_dist = stats.get('sentiment_distribution', {})
        if sentiment_dist:
            fig = px.pie(
                values=list(sentiment_dist.values()),
                names=list(sentiment_dist.keys()),
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Intent distribution
        intent_dist = stats.get('intent_distribution', {})
        if intent_dist:
            fig = px.bar(
                x=list(intent_dist.keys()),
                y=list(intent_dist.values()),
                title="Intent Distribution",
                labels={'x': 'Intent', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Root Cause Clusters
    st.markdown("---")
    st.subheader("🔍 Root Cause Analysis")
    
    root_cause = insights_data.get('root_cause_analysis', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Clusters Found", root_cause.get('num_clusters', 0))
    
    with col2:
        st.metric("Silhouette Score", f"{root_cause.get('silhouette_score', 0):.3f}")
    
    with col3:
        st.metric("Quality", root_cause.get('clustering_quality', 'Unknown'))
    
    # Cluster details
    clusters = root_cause.get('clusters', [])
    
    for cluster in clusters:
        if cluster['cluster_id'] != -1:  # Skip noise
            with st.expander(f"📦 {cluster['label']} ({cluster['size']} calls)"):
                st.write(cluster.get('description', ''))
                
                st.markdown("**Top Issues:**")
                for issue in cluster.get('top_issues', []):
                    st.markdown(f"- {issue}")
    
    # Recommendations
    st.markdown("---")
    st.subheader("💼 Recommendations")
    
    recommendations = business_insights.get('recommendations', [])
    
    for rec in recommendations:
        priority_color = {
            'High': '🔴',
            'Medium': '🟡',
            'Low': '🟢'
        }.get(rec.get('priority', 'Low'), '⚪')
        
        st.markdown(f"""
        <div class="recommendation-box">
            <strong>{priority_color} {rec.get('priority', 'Low')} Priority - {rec.get('area', 'General')}</strong><br>
            <strong>Action:</strong> {rec.get('action', 'No action specified')}<br>
            <strong>Expected Impact:</strong> {rec.get('expected_impact', 'Unknown')}
        </div>
        """, unsafe_allow_html=True)


def show_search(orchestrator):
    """Search similar issues page."""
    st.header("🔎 Search Similar Issues")
    
    query = st.text_input("Enter issue description or keywords")
    n_results = st.slider("Number of results", 1, 10, 5)
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            results = orchestrator.search_similar_issues(query, n_results)
        
        if results['ids']:
            st.success(f"Found {len(results['ids'][0])} similar issues")
            
            for idx, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity = 1 - distance  # Convert distance to similarity
                
                with st.expander(f"Result {idx + 1} - Similarity: {similarity:.2%}"):
                    st.markdown(f"**Call ID:** {metadata.get('call_id')}")
                    st.markdown(f"**Intent:** {metadata.get('intent', 'unknown').replace('_', ' ').title()}")
                    st.markdown(f"**Sentiment:** {metadata.get('sentiment', 'neutral').title()}")
                    st.markdown(f"**Issue:** {doc}")
        else:
            st.warning("No similar issues found")


def show_stats(orchestrator):
    """System statistics page."""
    st.header("⚙️ System Statistics")
    
    stats = orchestrator.get_storage_stats()
    
    st.subheader("Storage Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Transcripts Stored", stats.get('total_transcripts', 0))
    
    with col2:
        st.metric("Analyses Stored", stats.get('total_analyses', 0))
    
    with col3:
        st.metric("Vector Embeddings", stats.get('total_embeddings', 0))
    
    st.markdown("---")
    
    st.subheader("System Configuration")
    
    config_info = {
        "Whisper Model": "base",
        "LLM Model": "Llama 3.1 (Ollama)",
        "Embedding Model": "all-MiniLM-L6-v2",
        "Vector DB": "ChromaDB",
        "Clustering": "HDBSCAN with Cosine Similarity"
    }
    
    for key, value in config_info.items():
        st.text(f"{key}: {value}")


if __name__ == "__main__":
    main()
