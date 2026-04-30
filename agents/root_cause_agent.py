"""Root Cause Agent - Identifies common patterns across multiple calls."""

import numpy as np
from typing import Dict, List
from loguru import logger
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import hdbscan
from collections import Counter


class RootCauseAgent:

    def __init__(self, min_cluster_size=3, min_samples=2, metric="cosine", llm_client=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.llm = llm_client

    def analyze_root_causes(self, embeddings_data: Dict, analyses: List[Dict]) -> Dict:
        logger.info("Starting root cause analysis...")

        if not embeddings_data.get("ids"):
            return {"error": "No data available"}

        embeddings = np.array(embeddings_data["embeddings"])
        documents = embeddings_data["documents"]
        call_ids = embeddings_data["ids"]
        metadatas = embeddings_data.get("metadatas", [])

        # ===== FILTER =====
        embeddings, documents, call_ids, metadatas = self._filter_inputs(
            embeddings, documents, call_ids, metadatas
        )

        if len(embeddings) < self.min_cluster_size:
            return {"error": "Not enough clean data"}

        # ===== NORMALIZE =====
        if self.metric == "cosine":
            embeddings = normalize(embeddings)
            metric = "euclidean"
        else:
            metric = self.metric

        # ===== CLUSTER =====
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=metric
        )

        labels = clusterer.fit_predict(embeddings)

        # ===== SILHOUETTE =====
        valid = labels != -1
        silhouette = 0.0
        if valid.sum() > 1:
            try:
                silhouette = silhouette_score(embeddings[valid], labels[valid])
            except:
                pass

        clusters = self._organize_clusters(labels, call_ids, documents, metadatas, analyses)

        insights = self._generate_cluster_insights(clusters, len(call_ids))

        return {
            "total_calls": len(call_ids),
            "num_clusters": len([c for c in insights if c["cluster_id"] != -1]),
            "noise_points": sum(1 for l in labels if l == -1),
            "silhouette_score": float(silhouette),
            "clusters": insights,
            "clustering_quality": self._assess_quality(silhouette, labels)
        }

    # ===== FILTER =====
    def _filter_inputs(self, embeddings, documents, call_ids, metadatas):
        clean_emb, clean_docs, clean_ids, clean_meta = [], [], [], []

        for emb, doc, cid, meta in zip(embeddings, documents, call_ids, metadatas):
            if not doc or len(doc.strip()) < 10:
                continue
            if any(word in doc.lower() for word in ["error", "failed", "unknown"]):
                continue

            clean_emb.append(emb)
            clean_docs.append(doc)
            clean_ids.append(cid)
            clean_meta.append(meta)

        logger.info(f"Filtered {len(documents) - len(clean_docs)} bad records")
        return np.array(clean_emb), clean_docs, clean_ids, clean_meta

    # ===== ORGANIZE =====
    def _organize_clusters(self, labels, call_ids, documents, metadatas, analyses):
        clusters = {}
        lookup = {a["call_id"]: a for a in analyses}

        for i, label in enumerate(labels):
            clusters.setdefault(label, {
                "cluster_id": int(label),
                "call_ids": [],
                "documents": [],
                "intents": [],
                "sentiments": [],
                "key_issues": []
            })

            clusters[label]["call_ids"].append(call_ids[i])
            clusters[label]["documents"].append(documents[i])

            if i < len(metadatas):
                clusters[label]["intents"].append(metadatas[i].get("intent", "unknown"))
                clusters[label]["sentiments"].append(metadatas[i].get("sentiment", "neutral"))

            if call_ids[i] in lookup:
                clusters[label]["key_issues"].extend(
                    lookup[call_ids[i]].get("key_issues", [])
                )

        return list(clusters.values())

    # ===== INSIGHTS =====
    def _generate_cluster_insights(self, clusters, total_calls):
        insights = []

        for cluster in clusters:
            cid = cluster["cluster_id"]
            size = len(cluster["call_ids"])

            if cid == -1:
                insights.append({
                    "cluster_id": cid,
                    "label": "Potential New Issues",
                    "size": size,
                    "confidence": 0.0
                })
                continue

            issues = list(set(cluster["key_issues"]))
            intent = self._top(cluster["intents"])
            sentiment = self._top(cluster["sentiments"])

            label = self._safe_label(issues)

            confidence = round(size / total_calls, 2)
            severity = size * (1 if sentiment == "negative" else 0.5)

            insights.append({
                "cluster_id": cid,
                "label": label,
                "size": size,
                "confidence": confidence,
                "severity_score": severity,
                "primary_intent": intent,
                "dominant_sentiment": sentiment,
                "top_issues": issues[:3],
                "call_ids": cluster["call_ids"]
            })

        return sorted(insights, key=lambda x: x["size"], reverse=True)

    # ===== SAFE LABEL =====
    def _safe_label(self, issues):
        if not self.llm:
            return "General Issue"

        try:
            prompt = f"""
            Give a short 2-3 word label:
            {issues[:5]}
            """
            label = self.llm.invoke(prompt)
            return str(label).strip().replace('"', '')[:50]
        except:
            return "General Issue"

    def _top(self, items):
        return Counter(items).most_common(1)[0][0] if items else "unknown"

    def _assess_quality(self, silhouette, labels):
        noise = sum(1 for l in labels if l == -1) / len(labels)

        if silhouette > 0.5:
            q = "Excellent"
        elif silhouette > 0.3:
            q = "Good"
        else:
            q = "Poor"

        if noise > 0.5:
            q += " (high noise)"

        return q