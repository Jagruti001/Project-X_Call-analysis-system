"""Orchestrator - Manages the workflow of all agents."""

from typing import Dict, List
from pathlib import Path
from loguru import logger
import uuid
from datetime import datetime

from core.transcription import TranscriptionEngine
from core.llm_client import OllamaLLMClient
from agents.diarization_agent import DiarizationAgent
from agents.unified_analysis_agent import UnifiedAnalysisAgent
from agents.root_cause_agent import RootCauseAgent
from agents.insight_agent import InsightAgent
from storage.storage_manager import StorageManager
from utils.config import Config, cleanup_audio


class CallAnalysisOrchestrator:
    """Orchestrates the complete call analysis pipeline."""
    
    def __init__(self, config: Config):
        """
        Initialize orchestrator with all agents.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize components
        logger.info("Initializing Call Analysis Orchestrator...")
        
        self.transcription_engine = TranscriptionEngine(
            model_size=config.get('whisper.model_size', 'base'),
            device=config.get('whisper.device', 'cpu')
        )
        
        self.llm_client = OllamaLLMClient(
            base_url=config.get('llm.base_url', 'http://localhost:11434'),
            model=config.get('llm.model', 'qwen2.5:3b'),
            temperature=config.get('llm.temperature', 0.3)
        )
        
        # Initialize agents
        self.diarization_agent = DiarizationAgent(self.llm_client)
        self.analysis_agent = UnifiedAnalysisAgent(self.llm_client)
        self.root_cause_agent = RootCauseAgent(
            min_cluster_size=config.get('clustering.min_cluster_size', 3),
            min_samples=config.get('clustering.min_samples', 2),
            metric=config.get('clustering.metric', 'cosine')
        )
        self.insight_agent = InsightAgent(self.llm_client)
        
        # Initialize storage
        self.storage = StorageManager(
            transcripts_dir=config.get('storage.transcripts_dir', 'data/transcripts'),
            analysis_dir=config.get('storage.analysis_dir', 'data/analysis'),
            chromadb_path=config.get('chromadb.persist_directory', 'data/chromadb'),
            collection_name=config.get('chromadb.collection_name', 'call_issues'),
            embedding_model=config.get('embeddings.model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
        
        logger.info("Orchestrator initialized successfully")
    
    def process_single_call(self, audio_path: str, call_id: str = None) -> Dict:
        """
        Process a single call through the complete pipeline.
        
        Args:
            audio_path: Path to audio file
            call_id: Optional call ID (auto-generated if not provided)
            
        Returns:
            Complete analysis results
        """
        if call_id is None:
            call_id = f"call_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Processing call: {call_id}")
        
        try:
            # Step 1: Transcription
            logger.info("Step 1/4: Transcription")
            transcript_data = self.transcription_engine.transcribe(audio_path)
            raw_transcript = transcript_data['text']
            
            # Save transcript
            self.storage.save_transcript(call_id, {
                **transcript_data,
                "call_id": call_id
            })
            
            # Step 2: Diarization
            logger.info("Step 2/4: Speaker Diarization")
            diarization_result = self.diarization_agent.label_speakers(raw_transcript)
            labeled_transcript = diarization_result['labeled_transcript']
            
            # Validate diarization
            validation = self.diarization_agent.validate_diarization(labeled_transcript)
            logger.info(f"Diarization quality: {validation}")
            
            # Step 3: Analysis
            logger.info("Step 3/4: Unified Analysis")
            analysis = self.analysis_agent.analyze(labeled_transcript, call_id)
            
            # Save analysis and embeddings
            self.storage.save_analysis(call_id, analysis)
            
            # Step 4: Cleanup
            cleanup_audio(audio_path)
            
            result = {
                "call_id": call_id,
                "status": "success",
                "raw_transcript": raw_transcript,
                "labeled_transcript": labeled_transcript,
                "diarization_quality": validation,
                "analysis": analysis,
                "processed_at": datetime.now().isoformat()
            }
            
            logger.info(f"Call {call_id} processed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process call {call_id}: {e}")
            return {
                "call_id": call_id,
                "status": "error",
                "error": str(e)
            }
    
    def process_batch(self, audio_files: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Process multiple calls in batches.
        
        Args:
            audio_files: List of audio file paths
            batch_size: Number of files to process per batch
            
        Returns:
            List of processing results
        """
        logger.info(f"Processing {len(audio_files)} calls in batches of {batch_size}")
        
        results = []
        
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(audio_files)-1)//batch_size + 1}")
            
            for audio_file in batch:
                result = self.process_single_call(audio_file)
                results.append(result)
        
        logger.info(f"Batch processing complete: {len(results)} calls processed")
        
        return results
    
    def generate_insights(self) -> Dict:
        """
        Generate insights from all processed calls.
        
        Returns:
            Complete insights including root causes and recommendations
        """
        logger.info("Generating insights from all calls...")
        
        # Load all analyses
        all_analyses = self.storage.load_all_analyses()
        
        if not all_analyses:
            logger.warning("No analyses available for insight generation")
            return {"error": "No data available"}
        
        # Get embeddings for clustering
        embeddings_data = self.storage.get_all_embeddings()
        
        # Root cause analysis
        logger.info("Performing root cause analysis...")
        root_cause_result = self.root_cause_agent.analyze_root_causes(
            embeddings_data,
            all_analyses
        )
        
        # Generate business insights
        logger.info("Generating business insights...")
        insights = self.insight_agent.generate_insights(
            root_cause_result,
            all_analyses
        )
        
        result = {
            "root_cause_analysis": root_cause_result,
            "business_insights": insights,
            "total_calls_analyzed": len(all_analyses),
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info("Insight generation complete")
        
        return result
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        return self.storage.get_stats()
    
    def search_similar_issues(self, query: str, n_results: int = 5) -> Dict:
        """
        Search for similar issues using semantic search.
        
        Args:
            query: Search query
            n_results: Number of results
            
        Returns:
            Similar issues with metadata
        """
        return self.storage.find_similar_issues(query, n_results)
