"""Storage manager for transcripts, analysis, and vector embeddings."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
import chromadb
from sentence_transformers import SentenceTransformer


class StorageManager:
    """Manages storage of call data in JSON and ChromaDB."""

    def __init__(
        self,
        transcripts_dir: str = "data/transcripts",
        analysis_dir: str = "data/analysis",
        chromadb_path: str = "data/chromadb",
        collection_name: str = "call_issues",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.transcripts_dir = Path(transcripts_dir)
        self.analysis_dir = Path(analysis_dir)

        # Ensure directories exist
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info("Storage manager initialized")

    # =============================
    # SAVE TRANSCRIPT
    # =============================
    def save_transcript(self, call_id: str, transcript_data: Dict) -> str:
        transcript_data["call_id"] = call_id
        transcript_data["saved_at"] = datetime.now().isoformat()

        file_path = self.transcripts_dir / f"{call_id}_transcript.json"

        with open(file_path, "w") as f:
            json.dump(transcript_data, f, indent=2)

        logger.info(f"Transcript saved: {file_path}")
        return str(file_path)

    # =============================
    # SAVE ANALYSIS + EMBEDDING
    # =============================
    def save_analysis(self, call_id: str, analysis_data: Dict) -> str:
        analysis_data["call_id"] = call_id
        analysis_data["saved_at"] = datetime.now().isoformat()

        file_path = self.analysis_dir / f"{call_id}_analysis.json"

        with open(file_path, "w") as f:
            json.dump(analysis_data, f, indent=2)

        # ===== CHANGE 1: Better text extraction =====
        issues_text = self._extract_issues_text(analysis_data)

        # ===== CHANGE 2: Skip weak / garbage data =====
        if not issues_text or len(issues_text.strip()) < 10:
            logger.warning(f"Skipping weak embedding data for {call_id}")
            return str(file_path)

        # ===== CHANGE 3: Safe embedding generation =====
        try:
            embedding = self.embedding_model.encode(issues_text)
        except Exception as e:
            logger.error(f"Embedding failed for {call_id}: {e}")
            return str(file_path)

        if embedding is None or len(embedding) == 0:
            logger.error(f"Invalid embedding for {call_id}")
            return str(file_path)

        embedding = embedding.tolist()

        # ===== CHANGE 4: Prevent duplicate ID crash =====
        try:
            self.collection.delete(ids=[call_id])
        except Exception:
            pass

        # ===== CHANGE 5: Normalize metadata =====
        metadata = {
            "call_id": call_id,
            "intent": analysis_data.get("intent", "unknown").lower(),
            "sentiment": analysis_data.get("sentiment", {}).get("overall", "neutral").lower(),
            "timestamp": analysis_data["saved_at"],
        }

        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[issues_text],
            metadatas=[metadata],
            ids=[call_id],
        )

        logger.info(f"Analysis and embedding saved: {call_id}")
        return str(file_path)

    # =============================
    # LOAD FUNCTIONS
    # =============================
    def load_transcript(self, call_id: str) -> Optional[Dict]:
        file_path = self.transcripts_dir / f"{call_id}_transcript.json"

        if not file_path.exists():
            logger.warning(f"Transcript not found: {call_id}")
            return None

        with open(file_path, "r") as f:
            return json.load(f)

    def load_analysis(self, call_id: str) -> Optional[Dict]:
        file_path = self.analysis_dir / f"{call_id}_analysis.json"

        if not file_path.exists():
            logger.warning(f"Analysis not found: {call_id}")
            return None

        with open(file_path, "r") as f:
            return json.load(f)

    def load_all_analyses(self) -> List[Dict]:
        analyses = []

        for file_path in self.analysis_dir.glob("*_analysis.json"):
            try:
                with open(file_path, "r") as f:
                    analyses.append(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(analyses)} analyses")
        return analyses

    # =============================
    # RETRIEVE EMBEDDINGS (FOR RCA)
    # =============================
    def get_all_embeddings(self) -> Dict:
        result = self.collection.get(
            include=["embeddings", "documents", "metadatas"]
        )

        # ===== CHANGE 6: Safe retrieval handling =====
        if not result or not result.get("ids"):
            logger.warning("No embeddings found in DB")
            return {
                "ids": [],
                "embeddings": [],
                "documents": [],
                "metadatas": [],
            }

        logger.info(f"Retrieved {len(result['ids'])} embeddings")
        return result

    # =============================
    # SIMILARITY SEARCH
    # =============================
    def find_similar_issues(self, query_text: str, n_results: int = 5) -> Dict:
        query_embedding = self.embedding_model.encode(query_text).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        return results

    # =============================
    # TEXT PREPARATION
    # =============================
    def _extract_issues_text(self, analysis: Dict) -> str:
        """Create rich semantic text for embedding"""

        summary = analysis.get("summary", "")
        intent = analysis.get("intent", "")
        issues = analysis.get("key_issues", [])

        # ===== CHANGE 7: Better semantic structure =====
        text_parts = [
            f"Summary: {summary}",
            f"Intent: {intent}",
        ]

        if issues:
            text_parts.append(f"Issues: {' | '.join(issues)}")

        return " ".join(text_parts).strip()

    # =============================
    # STATS
    # =============================
    def get_stats(self) -> Dict:
        return {
            "total_transcripts": len(list(self.transcripts_dir.glob("*.json"))),
            "total_analyses": len(list(self.analysis_dir.glob("*.json"))),
            "total_embeddings": self.collection.count(),
        }