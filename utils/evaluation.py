"""Evaluation module for tracking system performance."""

import json
from typing import Dict, List
from pathlib import Path
from loguru import logger
from sklearn.metrics import silhouette_score
import numpy as np


class SystemEvaluator:
    """Evaluates system performance using defined metrics."""
    
    def __init__(self, target_metrics: Dict = None):
        """
        Initialize evaluator.
        
        Args:
            target_metrics: Target thresholds for each metric
        """
        self.target_metrics = target_metrics or {
            "speaker_labeling": 0.80,
            "sentiment": 0.75,
            "issue_extraction": 0.85,
            "clustering_silhouette": 0.30
        }
        
        self.evaluation_results = []
    
    def evaluate_speaker_labeling(self, 
                                  labeled_transcripts: List[str],
                                  ground_truth: List[str]) -> Dict:
        """
        Evaluate speaker labeling accuracy.
        
        Args:
            labeled_transcripts: System-generated labeled transcripts
            ground_truth: Manually labeled ground truth
            
        Returns:
            Evaluation metrics
        """
        if len(labeled_transcripts) != len(ground_truth):
            raise ValueError("Labeled transcripts and ground truth must have same length")
        
        total_turns = 0
        correct_labels = 0
        
        for pred, truth in zip(labeled_transcripts, ground_truth):
            pred_lines = [l.strip() for l in pred.split('\n') if l.strip()]
            truth_lines = [l.strip() for l in truth.split('\n') if l.strip()]
            
            for p_line, t_line in zip(pred_lines, truth_lines):
                total_turns += 1
                
                # Extract speaker label
                pred_speaker = p_line.split(':')[0] if ':' in p_line else ''
                truth_speaker = t_line.split(':')[0] if ':' in t_line else ''
                
                if pred_speaker == truth_speaker:
                    correct_labels += 1
        
        accuracy = correct_labels / total_turns if total_turns > 0 else 0
        
        result = {
            "metric": "speaker_labeling_accuracy",
            "accuracy": round(accuracy, 3),
            "correct": correct_labels,
            "total": total_turns,
            "target": self.target_metrics["speaker_labeling"],
            "passed": accuracy >= self.target_metrics["speaker_labeling"]
        }
        
        logger.info(f"Speaker labeling accuracy: {accuracy:.1%} (target: {self.target_metrics['speaker_labeling']:.1%})")
        self.evaluation_results.append(result)
        
        return result
    
    def evaluate_sentiment(self,
                          predicted_sentiments: List[str],
                          ground_truth_sentiments: List[str]) -> Dict:
        """
        Evaluate sentiment classification accuracy.
        
        Args:
            predicted_sentiments: System predictions
            ground_truth_sentiments: Manual ratings
            
        Returns:
            Evaluation metrics
        """
        if len(predicted_sentiments) != len(ground_truth_sentiments):
            raise ValueError("Predictions and ground truth must have same length")
        
        correct = sum(1 for p, t in zip(predicted_sentiments, ground_truth_sentiments)
                     if p.lower() == t.lower())
        total = len(predicted_sentiments)
        
        accuracy = correct / total if total > 0 else 0
        
        result = {
            "metric": "sentiment_accuracy",
            "accuracy": round(accuracy, 3),
            "correct": correct,
            "total": total,
            "target": self.target_metrics["sentiment"],
            "passed": accuracy >= self.target_metrics["sentiment"]
        }
        
        logger.info(f"Sentiment accuracy: {accuracy:.1%} (target: {self.target_metrics['sentiment']:.1%})")
        self.evaluation_results.append(result)
        
        return result
    
    def evaluate_issue_extraction(self,
                                 analyses: List[Dict],
                                 manual_checks: List[bool]) -> Dict:
        """
        Evaluate issue extraction completeness.
        
        Args:
            analyses: System-generated analyses
            manual_checks: Boolean list indicating if main issue was caught
            
        Returns:
            Evaluation metrics
        """
        if len(analyses) != len(manual_checks):
            raise ValueError("Analyses and manual checks must have same length")
        
        caught = sum(manual_checks)
        total = len(manual_checks)
        
        completeness = caught / total if total > 0 else 0
        
        result = {
            "metric": "issue_extraction_completeness",
            "completeness": round(completeness, 3),
            "issues_caught": caught,
            "total": total,
            "target": self.target_metrics["issue_extraction"],
            "passed": completeness >= self.target_metrics["issue_extraction"]
        }
        
        logger.info(f"Issue extraction completeness: {completeness:.1%} (target: {self.target_metrics['issue_extraction']:.1%})")
        self.evaluation_results.append(result)
        
        return result
    
    def evaluate_clustering(self, 
                           embeddings: np.ndarray,
                           cluster_labels: np.ndarray) -> Dict:
        """
        Evaluate clustering quality using silhouette score.
        
        Args:
            embeddings: Embedding vectors
            cluster_labels: Cluster assignments
            
        Returns:
            Evaluation metrics
        """
        # Remove noise points (-1 labels)
        valid_mask = cluster_labels != -1
        
        if valid_mask.sum() < 2:
            logger.warning("Not enough valid clusters for silhouette score")
            score = 0.0
        else:
            score = silhouette_score(
                embeddings[valid_mask],
                cluster_labels[valid_mask],
                metric='cosine'
            )
        
        result = {
            "metric": "clustering_silhouette_score",
            "score": round(float(score), 3),
            "valid_points": int(valid_mask.sum()),
            "noise_points": int((~valid_mask).sum()),
            "target": self.target_metrics["clustering_silhouette"],
            "passed": score >= self.target_metrics["clustering_silhouette"]
        }
        
        logger.info(f"Clustering silhouette score: {score:.3f} (target: {self.target_metrics['clustering_silhouette']:.3f})")
        self.evaluation_results.append(result)
        
        return result
    
    def generate_evaluation_report(self) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Complete evaluation summary
        """
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
        passed_metrics = sum(1 for r in self.evaluation_results if r.get('passed', False))
        total_metrics = len(self.evaluation_results)
        
        report = {
            "overall_performance": {
                "metrics_passed": passed_metrics,
                "total_metrics": total_metrics,
                "pass_rate": round(passed_metrics / total_metrics, 3) if total_metrics > 0 else 0
            },
            "individual_metrics": self.evaluation_results,
            "summary": self._create_summary()
        }
        
        return report
    
    def _create_summary(self) -> str:
        """Create human-readable summary."""
        passed = sum(1 for r in self.evaluation_results if r.get('passed', False))
        total = len(self.evaluation_results)
        
        if passed == total:
            return f"✅ All {total} metrics passed target thresholds"
        elif passed >= total * 0.75:
            return f"⚠️  {passed}/{total} metrics passed - Good performance with room for improvement"
        else:
            return f"❌ Only {passed}/{total} metrics passed - Significant improvement needed"
    
    def save_report(self, filepath: str = "evaluation_report.json"):
        """Save evaluation report to file."""
        report = self.generate_evaluation_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved: {filepath}")
        
        return filepath
