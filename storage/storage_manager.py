""" Storage manager for transcripts, analyses, and embeddings."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger
import time


class StorageManager:
    """
    Features:
    - JSON storage for transcripts and analyses
    - ChromaDB for vector embeddings
    - Automatic embedding generation
    - Efficient retrieval and search
    - Data validation and error handling
    """
    
    def __init__(
        self,
        transcripts_dir: str = "data/transcripts",
        analysis_dir: str = "data/analysis",
        chromadb_path: str = "data/chromadb",
        collection_name: str = "call_issues",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize storage manager.
        
        Args:
            transcripts_dir: Directory for transcript JSON files
            analysis_dir: Directory for analysis JSON files
            chromadb_path: Path for ChromaDB storage
            collection_name: ChromaDB collection name
            embedding_model: Sentence transformer model for embeddings
        """
        logger.info("Initializing Storage Manager...")
        
        # Setup directories
        self.transcripts_dir = Path(transcripts_dir)
        self.analysis_dir = Path(analysis_dir)
        self.chromadb_path = Path(chromadb_path)
        
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.chromadb_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}...")
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Initialize ChromaDB
        logger.info(f"Connecting to ChromaDB at {chromadb_path}...")
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chromadb_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Call analysis issues and embeddings"}
            )
            
            logger.info(f"✅ ChromaDB connected: collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
        
        logger.info("✅ Storage Manager initialized")
    
    # ===========================
    # TRANSCRIPT STORAGE
    # ===========================
    
    def save_transcript(self, call_id: str, transcript_data: Dict) -> bool:
        """
        Save transcript to JSON file.
        
        Args:
            call_id: Unique call identifier
            transcript_data: Transcript dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.transcripts_dir / f"{call_id}.json"
            
            # Add metadata
            transcript_data["saved_at"] = datetime.now().isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Transcript saved: {call_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save transcript {call_id}: {e}")
            return False
    
    def load_transcript(self, call_id: str) -> Optional[Dict]:
        """Load transcript from JSON file."""
        try:
            file_path = self.transcripts_dir / f"{call_id}.json"
            
            if not file_path.exists():
                logger.warning(f"Transcript not found: {call_id}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load transcript {call_id}: {e}")
            return None
    
    # ===========================
    # ANALYSIS STORAGE
    # ===========================
    
    def save_analysis(self, call_id: str, analysis: Dict) -> bool:
        """
        Save analysis to JSON and generate embeddings.
        
        Args:
            call_id: Unique call identifier
            analysis: Analysis dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save to JSON
            file_path = self.analysis_dir / f"{call_id}.json"
            
            # Add metadata
            analysis["saved_at"] = datetime.now().isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Analysis saved: {call_id}")
            
            # Generate and store embeddings
            self._generate_and_store_embeddings(call_id, analysis)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save analysis {call_id}: {e}")
            return False
    
    def load_analysis(self, call_id: str) -> Optional[Dict]:
        """Load analysis from JSON file."""
        try:
            file_path = self.analysis_dir / f"{call_id}.json"
            
            if not file_path.exists():
                logger.warning(f"Analysis not found: {call_id}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load analysis {call_id}: {e}")
            return None
    
    def load_all_analyses(self) -> List[Dict]:
        """Load all analyses from storage."""
        analyses = []
        
        for file_path in self.analysis_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
                    analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(analyses)} analyses")
        return analyses
    
    # ===========================
    # EMBEDDING MANAGEMENT
    # ===========================
    
    def _generate_and_store_embeddings(self, call_id: str, analysis: Dict):
        """Generate embeddings from analysis and store in ChromaDB."""
        try:
            # Extract text for embedding
            issues_text = self._extract_issues_text(analysis)
            
            if not issues_text or len(issues_text) < 10:
                logger.warning(f"Insufficient text for embedding: {call_id}")
                return
            
            # Generate embedding
            embedding = self.embedding_model.encode(issues_text).tolist()
            
            # Prepare metadata
            metadata = {
                "call_id": call_id,
                "intent": analysis.get("intent", "unknown"),
                "sentiment": analysis.get("sentiment", {}).get("overall", "neutral"),
                "resolution_status": analysis.get("resolution_status", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to collection
            self.collection.add(
                ids=[call_id],
                embeddings=[embedding],
                documents=[issues_text],
                metadatas=[metadata]
            )
            
            logger.debug(f"Embedding stored: {call_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate/store embedding for {call_id}: {e}")
    
    def _extract_issues_text(self, analysis: Dict) -> str:
        """Extract concatenated text from analysis for embedding."""
        parts = []
        
        # Summary
        summary = analysis.get("summary", "")
        if summary:
            parts.append(summary)
        
        # Key issues
        issues = analysis.get("key_issues", [])
        if issues:
            parts.append("Issues: " + " | ".join(issues))
        
        # Intent
        intent = analysis.get("intent", "")
        if intent and intent != "unknown":
            parts.append(f"Intent: {intent}")
        
        return " ".join(parts).strip()
    
    # ===========================
    # SEARCH & RETRIEVAL
    # ===========================
    
    def find_similar_issues(
        self, 
        query: str, 
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Find similar issues using semantic search.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results with documents, metadata, and distances
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count())
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}
    
    def get_all_embeddings(self) -> Dict[str, Any]:
        """
        Retrieve all embeddings from ChromaDB.
        
        Returns:
            Dictionary with ids, embeddings, documents, metadatas
        """
        try:
            count = self.collection.count()
            
            if count == 0:
                return {
                    "ids": [],
                    "embeddings": [],
                    "documents": [],
                    "metadatas": []
                }
            
            results = self.collection.get(
                include=["embeddings", "documents", "metadatas"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings: {e}")
            return {
                "ids": [],
                "embeddings": [],
                "documents": [],
                "metadatas": []
            }
    
    # ===========================
    # STATISTICS
    # ===========================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            transcript_count = len(list(self.transcripts_dir.glob("*.json")))
            analysis_count = len(list(self.analysis_dir.glob("*.json")))
            embedding_count = self.collection.count()
            
            return {
                "total_transcripts": transcript_count,
                "total_analyses": analysis_count,
                "total_embeddings": embedding_count,
                "storage_paths": {
                    "transcripts": str(self.transcripts_dir),
                    "analyses": str(self.analysis_dir),
                    "chromadb": str(self.chromadb_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_transcripts": 0,
                "total_analyses": 0,
                "total_embeddings": 0
            }
    
    # ===========================
    # MAINTENANCE
    # ===========================
    
    def delete_call(self, call_id: str) -> bool:
        """Delete all data for a specific call."""
        try:
            # Delete transcript
            transcript_file = self.transcripts_dir / f"{call_id}.json"
            if transcript_file.exists():
                transcript_file.unlink()
            
            # Delete analysis
            analysis_file = self.analysis_dir / f"{call_id}.json"
            if analysis_file.exists():
                analysis_file.unlink()
            
            # Delete from ChromaDB
            try:
                self.collection.delete(ids=[call_id])
            except Exception as e:
                logger.warning(f"Could not delete from ChromaDB: {e}")
            
            logger.info(f"Deleted call: {call_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete call {call_id}: {e}")
            return False
    
    def reset_all(self):
        """
        Reset all storage (WARNING: Deletes all data).
        
        Use with caution!
        """
        logger.warning("Resetting all storage...")
        
        # Delete all JSON files
        for file_path in self.transcripts_dir.glob("*.json"):
            file_path.unlink()
        
        for file_path in self.analysis_dir.glob("*.json"):
            file_path.unlink()
        
        # Reset ChromaDB
        try:
            self.chroma_client.delete_collection(self.collection.name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection.name,
                metadata={"description": "Call analysis issues and embeddings"}
            )
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB: {e}")
        
        logger.warning("All storage reset complete")