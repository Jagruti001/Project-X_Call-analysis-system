"""Orchestrator managing the complete call analysis workflow."""

from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
import uuid
from datetime import datetime
import time
import traceback

from core.transcription import TranscriptionEngine
from core.llm_client import OllamaLLMClient
from agents.diarization_agent import DiarizationAgent
from agents.unified_analysis_agent import UnifiedAnalysisAgent
from agents.root_cause_agent import RootCauseAgent
from agents.insight_agent import InsightAgent
from storage.storage_manager import StorageManager
from utils.config import Config, cleanup_audio


class CallAnalysisOrchestrator:
    """
    Features:
    - End-to-end call processing workflow
    - Comprehensive error handling at each stage
    - Progress tracking and logging
    - Batch processing support
    - Performance monitoring
    - Health checks for all components
    """
    
    def __init__(self, config: Config):
        """
        Initialize orchestrator with all components.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        logger.info("=" * 70)
        logger.info("Initializing Call Analysis Orchestrator")
        logger.info("=" * 70)
        
        try:
            # Initialize components
            self._initialize_components()
            
            logger.info("=" * 70)
            logger.info("✅ Orchestrator initialized successfully")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize orchestrator: {e}", exc_info=True)
            raise
    
    # ===========================
    # INITIALIZATION
    # ===========================
    
    def _initialize_components(self):
        """Initialize all system components."""
        
        # Transcription Engine
        logger.info("→ Initializing Transcription Engine...")
        self.transcription_engine = TranscriptionEngine(
            model_size=self.config.get('whisper.model_size', 'base'),
            device=self.config.get('whisper.device', 'cpu')
        )
        
        # LLM Client
        logger.info("→ Initializing LLM Client...")
        self.llm_client = OllamaLLMClient(
            base_url=self.config.get('llm.base_url', 'http://localhost:11434'),
            model=self.config.get('llm.model', 'qwen2.5:3b'),
            temperature=self.config.get('llm.temperature', 0.3)
        )
        
        # Check LLM availability
        if not self.llm_client.is_available():
            logger.warning(
                "⚠️ Ollama server not reachable - LLM-dependent features may fail. "
                "Make sure Ollama is running: 'ollama serve'"
            )
        
        # Initialize Agents
        logger.info("→ Initializing Agents...")
        
        self.diarization_agent = DiarizationAgent(self.llm_client)
        
        self.analysis_agent = UnifiedAnalysisAgent(self.llm_client)
        
        self.root_cause_agent = RootCauseAgent(
            min_cluster_size=self.config.get('clustering.min_cluster_size', 3),
            min_samples=self.config.get('clustering.min_samples', 2),
            metric=self.config.get('clustering.metric', 'cosine'),
            llm_client=self.llm_client.llm
        )
        
        self.insight_agent = InsightAgent(self.llm_client)
        
        # Initialize Storage
        logger.info("→ Initializing Storage Manager...")
        self.storage = StorageManager(
            transcripts_dir=self.config.get('storage.transcripts_dir', 'data/transcripts'),
            analysis_dir=self.config.get('storage.analysis_dir', 'data/analysis'),
            chromadb_path=self.config.get('chromadb.persist_directory', 'data/chromadb'),
            collection_name=self.config.get('chromadb.collection_name', 'call_issues'),
            embedding_model=self.config.get('embeddings.model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
    
    # ===========================
    # SINGLE CALL PROCESSING
    # ===========================
    
    def process_single_call(
        self, 
        audio_path: str, 
        call_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a single call through the complete pipeline.
        
        Pipeline stages:
        1. Transcription (audio → text)
        2. Diarization (speaker labeling)
        3. Analysis (extract insights)
        4. Storage (save results + embeddings)
        
        Args:
            audio_path: Path to audio file
            call_id: Optional call ID (auto-generated if not provided)
            metadata: Optional metadata (timestamp, agent name, etc.)
            
        Returns:
            Complete processing result with all stages
        """
        # Generate call ID if not provided
        if call_id is None:
            call_id = f"call_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("=" * 70)
        logger.info(f"🎯 Processing call: {call_id}")
        logger.info(f"Audio file: {Path(audio_path).name}")
        logger.info("=" * 70)
        
        pipeline_start = time.time()
        result = {
            "call_id": call_id,
            "status": "processing",
            "stages": {},
            "errors": []
        }
        
        try:
            # Stage 1: Transcription
            stage_result = self._stage_transcription(audio_path, call_id)
            result["stages"]["transcription"] = stage_result
            
            if stage_result["status"] == "failed":
                result["status"] = "failed"
                result["error"] = stage_result["error"]
                return result
            
            raw_transcript = stage_result["data"]["text"]
            
            # Stage 2: Diarization
            stage_result = self._stage_diarization(raw_transcript)
            result["stages"]["diarization"] = stage_result
            
            if stage_result["status"] == "failed":
                result["status"] = "partial"
                result["errors"].append(stage_result["error"])
                # Continue with raw transcript as fallback
                labeled_transcript = raw_transcript
            else:
                labeled_transcript = stage_result["data"]["labeled_transcript"]
            
            # Stage 3: Analysis
            stage_result = self._stage_analysis(labeled_transcript, call_id, metadata)
            result["stages"]["analysis"] = stage_result
            
            if stage_result["status"] == "failed":
                result["status"] = "partial"
                result["errors"].append(stage_result["error"])
            
            analysis = stage_result.get("data", {})
            
            # Stage 4: Storage
            stage_result = self._stage_storage(call_id, raw_transcript, analysis)
            result["stages"]["storage"] = stage_result
            
            # Cleanup
            cleanup_audio(audio_path)
            
            # Build final result
            pipeline_time = time.time() - pipeline_start
            
            result.update({
                "status": "success" if not result["errors"] else "partial",
                "raw_transcript": raw_transcript,
                "labeled_transcript": labeled_transcript,
                "analysis": analysis,
                "diarization_quality": result["stages"]["diarization"].get("data", {}).get("quality_metrics"),
                "pipeline_time_seconds": round(pipeline_time, 2),
                "processed_at": datetime.now().isoformat()
            })
            
            logger.info("=" * 70)
            logger.info(f"✅ Call {call_id} processed successfully in {pipeline_time:.1f}s")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed for call {call_id}: {e}", exc_info=True)
            
            result.update({
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            return result
    
    # ===========================
    # PIPELINE STAGES
    # ===========================
    
    def _stage_transcription(self, audio_path: str, call_id: str) -> Dict:
        """Execute transcription stage."""
        logger.info("📝 Stage 1/4: Transcription")
        
        try:
            start_time = time.time()
            
            transcript_data = self.transcription_engine.transcribe(audio_path)
            
            # Save transcript
            transcript_data["call_id"] = call_id
            self.storage.save_transcript(call_id, transcript_data)
            
            stage_time = time.time() - start_time
            
            logger.info(f"✅ Transcription complete: {len(transcript_data['text'])} characters in {stage_time:.1f}s")
            
            return {
                "status": "success",
                "stage_time_seconds": round(stage_time, 2),
                "data": transcript_data
            }
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "data": None
            }
    
    def _stage_diarization(self, raw_transcript: str) -> Dict:
        """Execute diarization stage."""
        logger.info("🎙️ Stage 2/4: Speaker Diarization")
        
        try:
            start_time = time.time()
            
            diarization_result = self.diarization_agent.label_speakers(raw_transcript)
            
            # Validate quality
            validation = self.diarization_agent.validate_diarization(diarization_result)
            
            stage_time = time.time() - start_time
            
            logger.info(
                f"✅ Diarization complete: {diarization_result['total_turns']} turns, "
                f"quality: {validation['quality_level']} in {stage_time:.1f}s"
            )
            
            return {
                "status": "success",
                "stage_time_seconds": round(stage_time, 2),
                "data": diarization_result,
                "validation": validation
            }
            
        except Exception as e:
            logger.error(f"❌ Diarization failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "data": None
            }
    
    def _stage_analysis(
        self, 
        labeled_transcript: str, 
        call_id: str,
        metadata: Optional[Dict]
    ) -> Dict:
        """Execute analysis stage."""
        logger.info("🔍 Stage 3/4: Call Analysis")
        
        try:
            start_time = time.time()
            
            analysis = self.analysis_agent.analyze(
                labeled_transcript, 
                call_id=call_id,
                metadata=metadata
            )
            
            stage_time = time.time() - start_time
            
            logger.info(
                f"✅ Analysis complete: intent={analysis.get('intent')}, "
                f"sentiment={analysis.get('sentiment', {}).get('overall')}, "
                f"confidence={analysis.get('confidence_score', 0):.2f} "
                f"in {stage_time:.1f}s"
            )
            
            return {
                "status": "success",
                "stage_time_seconds": round(stage_time, 2),
                "data": analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "data": None
            }
    
    def _stage_storage(
        self, 
        call_id: str, 
        transcript: str, 
        analysis: Dict
    ) -> Dict:
        """Execute storage stage."""
        logger.info("💾 Stage 4/4: Storage")
        
        try:
            start_time = time.time()
            
            # Save analysis
            self.storage.save_analysis(call_id, analysis)
            
            stage_time = time.time() - start_time
            
            logger.info(f"✅ Storage complete in {stage_time:.1f}s")
            
            return {
                "status": "success",
                "stage_time_seconds": round(stage_time, 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Storage failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    # ===========================
    # BATCH PROCESSING
    # ===========================
    
    def process_batch(
        self, 
        audio_files: List[str], 
        batch_size: int = 10
    ) -> List[Dict]:
        """
        Process multiple calls in batches.
        
        Args:
            audio_files: List of audio file paths
            batch_size: Number of files to process per batch
            
        Returns:
            List of processing results
        """
        logger.info("=" * 70)
        logger.info(f"📦 Batch Processing: {len(audio_files)} files in batches of {batch_size}")
        logger.info("=" * 70)
        
        results = []
        total_start = time.time()
        
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(audio_files) - 1) // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            for audio_file in batch:
                result = self.process_single_call(audio_file)
                results.append(result)
        
        total_time = time.time() - total_start
        successful = sum(1 for r in results if r["status"] == "success")
        partial = sum(1 for r in results if r["status"] == "partial")
        failed = sum(1 for r in results if r["status"] == "failed")
        
        logger.info("=" * 70)
        logger.info(f"✅ Batch complete in {total_time:.1f}s")
        logger.info(f"Results: {successful} success, {partial} partial, {failed} failed")
        logger.info("=" * 70)
        
        return results
    
    # ===========================
    # INSIGHTS GENERATION
    # ===========================
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights from all processed calls.
        
        Returns:
            Complete insights including:
                - Root cause analysis (clusters)
                - Business insights
                - Recommendations
                - Alerts
        """
        logger.info("=" * 70)
        logger.info("💡 Generating Insights from All Calls")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # Load all analyses
            all_analyses = self.storage.load_all_analyses()
            
            if not all_analyses:
                logger.warning("No analyses available for insight generation")
                return {
                    "error": "No data available - process some calls first",
                    "total_calls_analyzed": 0
                }
            
            logger.info(f"📊 Analyzing {len(all_analyses)} calls...")
            
            # Get embeddings for clustering
            embeddings_data = self.storage.get_all_embeddings()
            
            # Root cause analysis
            logger.info("🔍 Performing root cause analysis...")
            root_cause_result = self.root_cause_agent.analyze_root_causes(
                embeddings_data,
                all_analyses
            )
            
            # Generate business insights
            logger.info("💼 Generating business insights...")
            insights = self.insight_agent.generate_insights(
                root_cause_result,
                all_analyses
            )
            
            total_time = time.time() - start_time
            
            result = {
                "root_cause_analysis": root_cause_result,
                "business_insights": insights,
                "total_calls_analyzed": len(all_analyses),
                "analysis_time_seconds": round(total_time, 2),
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info("=" * 70)
            logger.info(f"✅ Insight generation complete in {total_time:.1f}s")
            logger.info(f"Found {root_cause_result.get('num_clusters', 0)} issue patterns")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Insight generation failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "total_calls_analyzed": 0
            }
    
    # ===========================
    # SEARCH & QUERY
    # ===========================
    
    def search_similar_issues(
        self, 
        query: str, 
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search for similar issues using semantic search.
        
        Args:
            query: Search query (issue description)
            n_results: Number of results to return
            
        Returns:
            Similar issues with metadata and similarity scores
        """
        logger.info(f"🔎 Searching for: '{query}' (top {n_results} results)")
        
        try:
            results = self.storage.find_similar_issues(query, n_results)
            
            if results and results.get('ids'):
                logger.info(f"Found {len(results['ids'][0])} similar issues")
            else:
                logger.info("No similar issues found")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": str(e)}
    
    # ===========================
    # SYSTEM STATUS
    # ===========================
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        return self.storage.get_stats()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive system health check.
        
        Returns:
            Health status of all components
        """
        logger.info("🏥 Running system health check...")
        
        health = {
            "overall_status": "healthy",
            "components": {},
            "issues": []
        }
        
        # LLM health
        llm_health = self.llm_client.health_check()
        health["components"]["llm"] = llm_health
        
        if not llm_health["server_reachable"]:
            health["overall_status"] = "degraded"
            health["issues"].append("Ollama server not reachable")
        
        # Storage health
        storage_stats = self.storage.get_stats()
        health["components"]["storage"] = {
            "status": "healthy",
            "total_calls": storage_stats.get("total_analyses", 0)
        }
        
        # Transcription engine
        transcription_info = self.transcription_engine.get_info()
        health["components"]["transcription"] = {
            "status": "healthy" if transcription_info["model_loaded"] else "failed",
            **transcription_info
        }
        
        logger.info(f"Health check complete: {health['overall_status']}")
        
        return health