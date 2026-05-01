"""Root Cause Agent - Production-grade pattern identification across calls."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
import hdbscan
from collections import Counter
import time


class RootCauseAgent:
    """
    Production-grade root cause analysis using clustering.
    
    Features:
    - HDBSCAN clustering with multiple quality metrics
    - Automatic outlier detection
    - Multi-metric quality assessment
    - Robust error handling
    - Detailed cluster profiling
    """

    def __init__(
        self, 
        min_cluster_size: int = 3, 
        min_samples: int = 2, 
        metric: str = "cosine",
        llm_client=None
    ):
        """
        Initialize Root Cause Agent.
        
        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for core points
            metric: Distance metric ('cosine' or 'euclidean')
            llm_client: Optional LLM for cluster labeling
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.llm = llm_client
        
        logger.info(
            f"✅ Root Cause Agent initialized - "
            f"min_cluster_size={min_cluster_size}, metric={metric}"
        )

    def analyze_root_causes(
        self, 
        embeddings_data: Dict, 
        analyses: List[Dict]
    ) -> Dict:
        """
        Perform comprehensive root cause analysis.
        
        Args:
            embeddings_data: Dict with 'embeddings', 'documents', 'ids', 'metadatas'
            analyses: List of call analyses
            
        Returns:
            Complete root cause analysis with clusters and insights
        """
        logger.info("🔍 Starting root cause analysis...")
        start_time = time.time()
        
        # Validate input
        if not embeddings_data.get("ids"):
            logger.error("No embedding data available")
            return {"error": "No data available", "clusters": []}

        try:
            # Extract data
            embeddings = np.array(embeddings_data["embeddings"])
            documents = embeddings_data["documents"]
            call_ids = embeddings_data["ids"]
            metadatas = embeddings_data.get("metadatas", [])
            
            # Filter and clean
            embeddings, documents, call_ids, metadatas = self._filter_inputs(
                embeddings, documents, call_ids, metadatas
            )
            
            if len(embeddings) < self.min_cluster_size:
                logger.warning(
                    f"Insufficient data after filtering: {len(embeddings)} samples "
                    f"(minimum required: {self.min_cluster_size})"
                )
                return {
                    "error": "Insufficient clean data for clustering",
                    "total_calls": len(embeddings),
                    "required_minimum": self.min_cluster_size,
                    "clusters": []
                }
            
            # Normalize for cosine similarity
            if self.metric == "cosine":
                embeddings = normalize(embeddings)
                clustering_metric = "euclidean"  # After normalization
            else:
                clustering_metric = self.metric
            
            # Perform clustering
            logger.info(f"Clustering {len(embeddings)} samples...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=clustering_metric,
                cluster_selection_method='eom'  # Excess of Mass - more stable
            )
            
            labels = clusterer.fit_predict(embeddings)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(embeddings, labels)
            
            # Organize clusters
            clusters = self._organize_clusters(
                labels, call_ids, documents, metadatas, analyses
            )
            
            # Generate insights
            cluster_insights = self._generate_cluster_insights(
                clusters, len(call_ids), quality_metrics
            )
            
            analysis_time = time.time() - start_time
            
            result = {
                "total_calls": len(call_ids),
                "num_clusters": len([c for c in cluster_insights if c["cluster_id"] != -1]),
                "noise_points": sum(1 for l in labels if l == -1),
                "noise_rate": round(sum(1 for l in labels if l == -1) / len(labels), 3),
                "silhouette_score": quality_metrics["silhouette"],
                "davies_bouldin_score": quality_metrics["davies_bouldin"],
                "clusters": cluster_insights,
                "clustering_quality": self._assess_quality(quality_metrics, labels),
                "quality_metrics": quality_metrics,
                "analysis_time_seconds": round(analysis_time, 2),
                "clustering_params": {
                    "min_cluster_size": self.min_cluster_size,
                    "min_samples": self.min_samples,
                    "metric": self.metric
                }
            }
            
            logger.info(
                f"✅ Root cause analysis complete - "
                f"{result['num_clusters']} clusters found, "
                f"quality: {result['clustering_quality']}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Root cause analysis failed: {e}", exc_info=True)
            return {
                "error": f"Analysis failed: {str(e)}",
                "clusters": []
            }

    # ===========================
    # DATA FILTERING
    # ===========================

    def _filter_inputs(
        self, 
        embeddings: np.ndarray,
        documents: List[str],
        call_ids: List[str],
        metadatas: List[Dict]
    ) -> Tuple[np.ndarray, List[str], List[str], List[Dict]]:
        """
        Filter out invalid or low-quality data.
        
        Args:
            embeddings: Embedding vectors
            documents: Text documents
            call_ids: Call identifiers
            metadatas: Metadata dicts
            
        Returns:
            Filtered data arrays
        """
        clean_emb, clean_docs, clean_ids, clean_meta = [], [], [], []
        
        # Ensure metadatas has same length
        if len(metadatas) < len(embeddings):
            metadatas.extend([{}] * (len(embeddings) - len(metadatas)))
        
        for emb, doc, cid, meta in zip(embeddings, documents, call_ids, metadatas):
            # Filter criteria
            if not doc or len(doc.strip()) < 10:
                continue
            
            # Skip obvious error messages
            if any(word in doc.lower() for word in ["error", "failed", "unknown"]):
                continue
            
            # Check embedding validity
            if np.isnan(emb).any() or np.isinf(emb).any():
                logger.warning(f"Invalid embedding for call {cid}")
                continue
            
            # Check embedding norm
            if np.linalg.norm(emb) < 1e-6:
                logger.warning(f"Zero embedding for call {cid}")
                continue
            
            clean_emb.append(emb)
            clean_docs.append(doc)
            clean_ids.append(cid)
            clean_meta.append(meta)
        
        filtered_count = len(documents) - len(clean_docs)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} invalid records")
        
        return np.array(clean_emb), clean_docs, clean_ids, clean_meta

    # ===========================
    # QUALITY METRICS
    # ===========================

    def _calculate_quality_metrics(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate multiple clustering quality metrics.
        
        Args:
            embeddings: Embedding vectors
            labels: Cluster labels
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            "silhouette": 0.0,
            "davies_bouldin": float('inf'),
            "cluster_count": len(set(labels)) - (1 if -1 in labels else 0),
            "noise_ratio": sum(1 for l in labels if l == -1) / len(labels)
        }
        
        # Only calculate if we have valid clusters
        valid_mask = labels != -1
        valid_count = valid_mask.sum()
        unique_clusters = len(set(labels[valid_mask]))
        
        if valid_count > 1 and unique_clusters > 1:
            try:
                # Silhouette score (higher is better, range -1 to 1)
                metrics["silhouette"] = silhouette_score(
                    embeddings[valid_mask], 
                    labels[valid_mask]
                )
            except Exception as e:
                logger.warning(f"Silhouette calculation failed: {e}")
            
            try:
                # Davies-Bouldin score (lower is better, min 0)
                metrics["davies_bouldin"] = davies_bouldin_score(
                    embeddings[valid_mask], 
                    labels[valid_mask]
                )
            except Exception as e:
                logger.warning(f"Davies-Bouldin calculation failed: {e}")
        
        return metrics

    # ===========================
    # CLUSTER ORGANIZATION
    # ===========================

    def _organize_clusters(
        self,
        labels: np.ndarray,
        call_ids: List[str],
        documents: List[str],
        metadatas: List[Dict],
        analyses: List[Dict]
    ) -> List[Dict]:
        """
        Organize data into cluster structures.
        
        Args:
            labels: Cluster labels
            call_ids: Call IDs
            documents: Documents
            metadatas: Metadata
            analyses: Call analyses
            
        Returns:
            List of cluster dictionaries
        """
        clusters = {}
        lookup = {a["call_id"]: a for a in analyses if "call_id" in a}
        
        for i, label in enumerate(labels):
            cluster_id = int(label)
            
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    "cluster_id": cluster_id,
                    "call_ids": [],
                    "documents": [],
                    "intents": [],
                    "sentiments": [],
                    "resolutions": [],
                    "key_issues": [],
                    "products": []
                }
            
            cluster = clusters[cluster_id]
            cluster["call_ids"].append(call_ids[i])
            cluster["documents"].append(documents[i])
            
            # Extract metadata
            if i < len(metadatas) and metadatas[i]:
                cluster["intents"].append(metadatas[i].get("intent", "unknown"))
                cluster["sentiments"].append(metadatas[i].get("sentiment", "neutral"))
            
            # Extract from analysis
            if call_ids[i] in lookup:
                analysis = lookup[call_ids[i]]
                cluster["key_issues"].extend(analysis.get("key_issues", []))
                cluster["resolutions"].append(analysis.get("resolution_status", "unknown"))
                
                product = analysis.get("product_mentioned")
                if product:
                    cluster["products"].append(product)
        
        return list(clusters.values())

    # ===========================
    # CLUSTER INSIGHTS
    # ===========================

    def _generate_cluster_insights(
        self,
        clusters: List[Dict],
        total_calls: int,
        quality_metrics: Dict
    ) -> List[Dict]:
        """
        Generate insights for each cluster.
        
        Args:
            clusters: List of cluster data
            total_calls: Total number of calls
            quality_metrics: Clustering quality metrics
            
        Returns:
            List of cluster insights
        """
        insights = []
        
        for cluster in clusters:
            cid = cluster["cluster_id"]
            size = len(cluster["call_ids"])
            
            # Handle noise cluster
            if cid == -1:
                insights.append({
                    "cluster_id": cid,
                    "label": "Outliers / Unique Issues",
                    "description": "Calls that don't fit into major patterns",
                    "size": size,
                    "percentage": round(size / total_calls * 100, 1),
                    "confidence": 0.0,
                    "severity_score": 0.0,
                    "actionability": "low"
                })
                continue
            
            # Extract patterns
            top_intent = self._most_common(cluster["intents"])
            dominant_sentiment = self._most_common(cluster["sentiments"])
            resolution_pattern = self._most_common(cluster["resolutions"])
            
            # Get unique issues
            unique_issues = list(set(cluster["key_issues"]))
            top_issues = self._get_top_issues(cluster["key_issues"], limit=5)
            
            # Get top products
            top_products = self._get_top_items(cluster["products"], limit=3)
            
            # Generate label
            label = self._generate_cluster_label(unique_issues, top_intent, top_products)
            
            # Calculate metrics
            confidence = min(round(size / total_calls, 2), 1.0)
            
            # Severity based on size and negative sentiment
            neg_count = cluster["sentiments"].count("negative")
            severity = size * (1.0 if neg_count > size * 0.5 else 0.5)
            
            # Actionability
            actionability = self._assess_actionability(
                size, unique_issues, resolution_pattern
            )
            
            # Description
            description = self._generate_description(
                size, top_intent, dominant_sentiment, top_issues, top_products
            )
            
            insights.append({
                "cluster_id": cid,
                "label": label,
                "description": description,
                "size": size,
                "percentage": round(size / total_calls * 100, 1),
                "confidence": confidence,
                "severity_score": round(severity, 1),
                "actionability": actionability,
                "primary_intent": top_intent,
                "dominant_sentiment": dominant_sentiment,
                "resolution_pattern": resolution_pattern,
                "top_issues": top_issues,
                "top_products": top_products,
                "call_ids": cluster["call_ids"][:10]  # Limit for output size
            })
        
        # Sort by size (descending)
        return sorted(insights, key=lambda x: x["size"], reverse=True)

    # ===========================
    # HELPER FUNCTIONS
    # ===========================

    def _most_common(self, items: List[str]) -> str:
        """Get most common item from list."""
        if not items:
            return "unknown"
        
        counter = Counter(items)
        return counter.most_common(1)[0][0]

    def _get_top_issues(self, issues: List[str], limit: int = 5) -> List[Dict]:
        """Get top issues with frequency."""
        counter = Counter(issues)
        return [
            {"issue": issue, "frequency": freq}
            for issue, freq in counter.most_common(limit)
        ]

    def _get_top_items(self, items: List[str], limit: int = 3) -> List[str]:
        """Get top items by frequency."""
        if not items:
            return []
        
        counter = Counter(items)
        return [item for item, _ in counter.most_common(limit)]

    def _generate_cluster_label(
        self,
        issues: List[str],
        intent: str,
        products: List[str]
    ) -> str:
        """Generate a descriptive label for the cluster."""
        
        # Try LLM-based labeling if available
        if self.llm and len(issues) > 0:
            try:
                prompt = f"""Generate a concise 3-5 word label for this issue cluster:

Intent: {intent}
Products: {', '.join(products[:2]) if products else 'N/A'}
Top Issues: {', '.join(issues[:3])}

Label (3-5 words only):"""
                
                label = self.llm.invoke(prompt)
                label_str = str(label).strip().replace('"', '').replace("'", "")
                
                # Validate length
                if 5 <= len(label_str) <= 60:
                    return label_str
                    
            except Exception as e:
                logger.debug(f"LLM labeling failed: {e}")
        
        # Fallback: rule-based labeling
        if products:
            return f"{products[0]} - {intent.replace('_', ' ').title()}"
        elif issues:
            first_issue = issues[0]
            # Truncate if too long
            return first_issue[:50] + "..." if len(first_issue) > 50 else first_issue
        else:
            return f"{intent.replace('_', ' ').title()} Issues"

    def _assess_actionability(
        self,
        size: int,
        issues: List[str],
        resolution_pattern: str
    ) -> str:
        """Assess how actionable this cluster is."""
        
        # High actionability criteria
        if size >= 5 and len(issues) > 0 and resolution_pattern == "unresolved":
            return "high"
        
        # Medium actionability
        if size >= 3 and len(issues) > 0:
            return "medium"
        
        # Low actionability
        return "low"

    def _generate_description(
        self,
        size: int,
        intent: str,
        sentiment: str,
        top_issues: List[Dict],
        products: List[str]
    ) -> str:
        """Generate human-readable cluster description."""
        
        parts = []
        
        # Size and intent
        parts.append(f"{size} calls related to {intent.replace('_', ' ')}")
        
        # Sentiment
        if sentiment in ["negative", "positive"]:
            parts.append(f"with {sentiment} sentiment")
        
        # Top issue
        if top_issues:
            main_issue = top_issues[0]["issue"]
            parts.append(f"primarily concerning '{main_issue}'")
        
        # Products
        if products:
            parts.append(f"affecting {', '.join(products[:2])}")
        
        return ". ".join(parts) + "."

    # ===========================
    # QUALITY ASSESSMENT
    # ===========================

    def _assess_quality(
        self, 
        metrics: Dict[str, float], 
        labels: np.ndarray
    ) -> str:
        """
        Assess overall clustering quality.
        
        Args:
            metrics: Quality metrics
            labels: Cluster labels
            
        Returns:
            Quality assessment string
        """
        silhouette = metrics["silhouette"]
        noise_ratio = metrics["noise_ratio"]
        cluster_count = metrics["cluster_count"]
        
        quality_parts = []
        
        # Base quality from silhouette
        if silhouette > 0.5:
            quality_parts.append("Excellent")
        elif silhouette > 0.3:
            quality_parts.append("Good")
        elif silhouette > 0.1:
            quality_parts.append("Fair")
        else:
            quality_parts.append("Poor")
        
        # Modifiers
        if noise_ratio > 0.5:
            quality_parts.append("(high noise)")
        elif noise_ratio > 0.3:
            quality_parts.append("(moderate noise)")
        
        if cluster_count < 2:
            quality_parts.append("(limited clusters)")
        elif cluster_count > 10:
            quality_parts.append("(fragmented)")
        
        return " ".join(quality_parts)

    # ===========================
    # UTILITIES
    # ===========================

    def get_cluster_summary(self, analysis_result: Dict) -> str:
        """Generate text summary of clustering results."""
        
        if "error" in analysis_result:
            return f"Analysis failed: {analysis_result['error']}"
        
        summary_parts = [
            f"Analyzed {analysis_result['total_calls']} calls",
            f"Found {analysis_result['num_clusters']} distinct issue patterns",
            f"Quality: {analysis_result['clustering_quality']}",
            f"Silhouette score: {analysis_result['silhouette_score']:.3f}"
        ]
        
        if analysis_result['noise_points'] > 0:
            summary_parts.append(
                f"{analysis_result['noise_points']} unique/outlier cases"
            )
        
        return " | ".join(summary_parts)