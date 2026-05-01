"""Insight Agent - Production-grade business insights and recommendations."""

import json
from typing import Dict, List, Any, Optional
from collections import Counter
import time

from loguru import logger
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from core.llm_client import OllamaLLMClient


# ===========================
# STRUCTURED OUTPUT
# ===========================

class InsightOutput(BaseModel):
    """Structured business insights output."""
    insights: List[str] = Field(
        description="Key business insights (3-5 actionable items)",
        min_items=3,
        max_items=7
    )
    risks: List[str] = Field(
        description="Identified business risks (2-4 items)",
        min_items=2,
        max_items=5
    )
    opportunities: List[str] = Field(
        description="Improvement opportunities (2-4 items)",
        min_items=2,
        max_items=5
    )


class Recommendation(BaseModel):
    """Single actionable recommendation."""
    area: str = Field(description="Business area (e.g., Customer Service, Product)")
    action: str = Field(description="Specific action to take")
    expected_impact: str = Field(description="Expected business impact")
    priority: str = Field(description="Priority level: High, Medium, or Low")


# ===========================
# AGENT
# ===========================

class InsightAgent:
    """
    Production-grade business insights generator.
    
    Features:
    - Multi-dimensional statistical analysis
    - LLM-powered strategic insights
    - Automated recommendation generation
    - Priority-based action items
    - Trend detection and forecasting
    """

    def __init__(self, llm_client: OllamaLLMClient, max_retries: int = 2):
        """
        Initialize Insight Agent.
        
        Args:
            llm_client: LangChain LLM client
            max_retries: Maximum LLM retry attempts
        """
        self.llm = llm_client.llm
        self.max_retries = max_retries
        self._build_chain()
        logger.info("✅ Insight Agent initialized (production mode)")

    # ===========================
    # CHAIN BUILDING
    # ===========================

    def _build_chain(self):
        """Build LangChain chain for insight generation."""
        parser = PydanticOutputParser(pydantic_object=InsightOutput)

        system_prompt = """You are a senior business analyst and customer experience strategist.

Your task:
- Analyze call center data to identify strategic business insights
- Focus on actionable, high-impact recommendations
- Consider both immediate wins and long-term improvements
- Be specific, quantitative where possible, and business-focused
- Prioritize customer satisfaction and operational efficiency

Guidelines:
- Insights should be data-driven and specific
- Risks should be realistic and addressable
- Opportunities should be concrete and measurable
- Avoid generic statements - be specific to the data

Return ONLY valid JSON matching the schema."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", """Analyze this call center data:

{summary}

{format_instructions}""")
        ])

        self.chain = (
            prompt.partial(format_instructions=parser.get_format_instructions())
            | self.llm
            | parser
        )

    # ===========================
    # MAIN ENTRY POINT
    # ===========================

    def generate_insights(
        self,
        root_cause_analysis: Dict,
        all_analyses: List[Dict]
    ) -> Dict:
        """
        Generate comprehensive business insights.
        
        Args:
            root_cause_analysis: Root cause clustering results
            all_analyses: List of all call analyses
            
        Returns:
            Complete insights with statistics, trends, and recommendations
        """
        logger.info("💡 Generating business insights...")
        start_time = time.time()
        
        try:
            # Calculate statistics
            statistics = self._compute_comprehensive_stats(all_analyses)
            
            # Detect trends
            trends = self._compute_trends(all_analyses)
            
            # Build executive summary
            summary = self._build_executive_summary(
                statistics, trends, root_cause_analysis
            )
            
            # Generate LLM insights with retry
            llm_insights = self._generate_with_retry(summary)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                statistics, trends, root_cause_analysis, llm_insights
            )
            
            # Identify priority areas
            priority_areas = self._identify_priority_areas(statistics, trends)
            
            # Calculate confidence
            confidence = self._calculate_confidence_score(statistics, all_analyses)
            
            # Generate alerts
            alerts = self._generate_alerts(statistics, trends)
            
            analysis_time = time.time() - start_time
            
            result = {
                "statistics": statistics,
                "trends": trends,
                "key_insights": llm_insights.get("insights", []),
                "risks": llm_insights.get("risks", []),
                "opportunities": llm_insights.get("opportunities", []),
                "recommendations": recommendations,
                "priority_areas": priority_areas,
                "confidence_score": confidence,
                "alerts": alerts,
                "analysis_time_seconds": round(analysis_time, 2),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(
                f"✅ Insight generation complete - "
                f"{len(recommendations)} recommendations, "
                f"confidence: {confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}", exc_info=True)
            return self._create_fallback_insights()

    # ===========================
    # COMPREHENSIVE STATISTICS
    # ===========================

    def _compute_comprehensive_stats(self, analyses: List[Dict]) -> Dict:
        """
        Compute comprehensive statistics from all analyses.
        
        Args:
            analyses: List of call analyses
            
        Returns:
            Detailed statistics dictionary
        """
        if not analyses:
            return self._empty_stats()

        total = len(analyses)
        
        # Extract fields safely
        sentiments = [
            a.get("sentiment", {}).get("overall", "neutral")
            for a in analyses
        ]
        
        resolutions = [
            a.get("resolution_status", "unknown")
            for a in analyses
        ]
        
        intents = [
            a.get("intent", "unknown")
            for a in analyses
        ]
        
        # Satisfaction indicators
        satisfactions = [
            a.get("customer_satisfaction_indicators", {}).get("explicit_satisfaction", "unclear")
            for a in analyses
        ]
        
        # Confidence scores
        confidences = [
            a.get("confidence_score", 0.5)
            for a in analyses
        ]
        
        # Calculate rates
        negative_rate = sentiments.count("negative") / total
        positive_rate = sentiments.count("positive") / total
        
        resolved_rate = resolutions.count("resolved") / total
        unresolved_rate = (
            resolutions.count("unresolved") + 
            resolutions.count("escalated")
        ) / total
        
        satisfaction_rate = satisfactions.count("yes") / total
        
        # Intent distribution
        intent_counter = Counter(intents)
        intent_distribution = {
            intent: count for intent, count in intent_counter.most_common(10)
        }
        
        # Sentiment distribution
        sentiment_distribution = {
            "positive": sentiments.count("positive"),
            "neutral": sentiments.count("neutral"),
            "negative": sentiments.count("negative")
        }
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return {
            "total_calls": total,
            "negative_sentiment_rate": round(negative_rate, 3),
            "positive_sentiment_rate": round(positive_rate, 3),
            "neutral_sentiment_rate": round(1 - negative_rate - positive_rate, 3),
            "resolved_rate": round(resolved_rate, 3),
            "unresolved_rate": round(unresolved_rate, 3),
            "satisfaction_rate": round(satisfaction_rate, 3),
            "top_intent": intent_counter.most_common(1)[0][0] if intent_counter else "unknown",
            "intent_distribution": intent_distribution,
            "sentiment_distribution": sentiment_distribution,
            "average_confidence": round(avg_confidence, 2),
            "high_confidence_calls": sum(1 for c in confidences if c >= 0.7),
            "low_confidence_calls": sum(1 for c in confidences if c < 0.5)
        }

    def _empty_stats(self) -> Dict:
        """Return empty statistics structure."""
        return {
            "total_calls": 0,
            "negative_sentiment_rate": 0.0,
            "positive_sentiment_rate": 0.0,
            "resolved_rate": 0.0,
            "unresolved_rate": 0.0,
            "satisfaction_rate": 0.0,
            "top_intent": "unknown",
            "intent_distribution": {},
            "sentiment_distribution": {}
        }

    # ===========================
    # TREND DETECTION
    # ===========================

    def _compute_trends(self, analyses: List[Dict]) -> List[Dict]:
        """
        Compute issue trends and patterns.
        
        Args:
            analyses: List of call analyses
            
        Returns:
            List of trend dictionaries
        """
        issue_counter = Counter()
        
        # Collect all issues
        for analysis in analyses:
            issues = analysis.get("key_issues", [])
            issue_counter.update(issues)
        
        # Build trend list
        trends = []
        total_issues = sum(issue_counter.values())
        
        for issue, frequency in issue_counter.most_common(15):  # Top 15 trends
            percentage = (frequency / total_issues * 100) if total_issues > 0 else 0
            
            # Classify trend level
            if frequency >= 10:
                trend_level = "critical"
            elif frequency >= 5:
                trend_level = "high"
            elif frequency >= 3:
                trend_level = "moderate"
            else:
                trend_level = "low"
            
            trends.append({
                "issue": issue,
                "frequency": frequency,
                "percentage": round(percentage, 1),
                "trend_level": trend_level,
                "impact": self._assess_issue_impact(frequency, len(analyses))
            })
        
        return trends

    def _assess_issue_impact(self, frequency: int, total_calls: int) -> str:
        """Assess business impact of an issue based on frequency."""
        rate = frequency / total_calls
        
        if rate >= 0.3:
            return "Critical - Affects 30%+ of calls"
        elif rate >= 0.15:
            return "High - Affects 15%+ of calls"
        elif rate >= 0.05:
            return "Medium - Affects 5%+ of calls"
        else:
            return "Low - Isolated incidents"

    # ===========================
    # EXECUTIVE SUMMARY
    # ===========================

    def _build_executive_summary(
        self, 
        stats: Dict, 
        trends: List[Dict],
        root_cause: Dict
    ) -> str:
        """Build executive summary for LLM analysis."""
        
        # Overview
        summary_parts = [
            f"CALL CENTER ANALYSIS SUMMARY",
            f"Total Calls Analyzed: {stats.get('total_calls', 0)}",
            f"",
            f"SENTIMENT BREAKDOWN:",
            f"- Positive: {stats.get('positive_sentiment_rate', 0)*100:.1f}%",
            f"- Neutral: {stats.get('neutral_sentiment_rate', 0)*100:.1f}%",
            f"- Negative: {stats.get('negative_sentiment_rate', 0)*100:.1f}%",
            f"",
            f"RESOLUTION METRICS:",
            f"- Resolved: {stats.get('resolved_rate', 0)*100:.1f}%",
            f"- Unresolved/Escalated: {stats.get('unresolved_rate', 0)*100:.1f}%",
            f"- Customer Satisfaction: {stats.get('satisfaction_rate', 0)*100:.1f}%",
            f"",
            f"TOP INTENT: {stats.get('top_intent', 'unknown').replace('_', ' ').title()}",
            f""
        ]
        
        # Top trends
        if trends:
            summary_parts.append("TOP ISSUES:")
            for trend in trends[:5]:
                summary_parts.append(
                    f"- {trend['issue']} ({trend['frequency']} calls, {trend['percentage']:.1f}%)"
                )
            summary_parts.append("")
        
        # Cluster summary
        clusters = root_cause.get("clusters", [])
        valid_clusters = [c for c in clusters if c.get("cluster_id", -1) != -1]
        
        if valid_clusters:
            summary_parts.append("IDENTIFIED PATTERNS:")
            for cluster in valid_clusters[:3]:
                summary_parts.append(
                    f"- {cluster.get('label', 'Unknown')} "
                    f"({cluster.get('size', 0)} calls, "
                    f"{cluster.get('severity_score', 0):.0f} severity)"
                )
        
        return "\n".join(summary_parts)

    # ===========================
    # LLM INSIGHT GENERATION
    # ===========================

    def _generate_with_retry(self, summary: str) -> Dict:
        """Generate insights with retry mechanism."""
        
        for attempt in range(self.max_retries + 1):
            try:
                result = self.chain.invoke({"summary": summary})
                return result.model_dump()
                
            except Exception as e:
                logger.warning(f"LLM insight generation attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        logger.error("LLM insight generation failed completely - using fallback")
        return self._fallback_insights(summary)

    def _fallback_insights(self, summary: str) -> Dict:
        """Generate fallback insights when LLM fails."""
        return {
            "insights": [
                "High call volume detected - review staffing levels",
                "Multiple recurring issues identified - implement systematic fixes",
                "Customer sentiment patterns require immediate attention"
            ],
            "risks": [
                "Customer dissatisfaction may lead to churn",
                "Unresolved issues creating support backlog"
            ],
            "opportunities": [
                "Improve first-call resolution rate",
                "Implement proactive customer outreach for common issues"
            ]
        }

    # ===========================
    # RECOMMENDATIONS
    # ===========================

    def _generate_recommendations(
        self,
        stats: Dict,
        trends: List[Dict],
        root_cause: Dict,
        llm_insights: Dict
    ) -> List[Dict]:
        """
        Generate actionable recommendations.
        
        Args:
            stats: Statistics dictionary
            trends: Trend list
            root_cause: Root cause analysis
            llm_insights: LLM-generated insights
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        # High negative sentiment
        if stats.get("negative_sentiment_rate", 0) > 0.3:
            recommendations.append({
                "area": "Customer Experience",
                "action": "Launch customer satisfaction improvement initiative - focus on empathy training and issue resolution speed",
                "expected_impact": f"Could improve satisfaction for {int(stats['total_calls'] * stats['negative_sentiment_rate'])} calls",
                "priority": "High"
            })
        
        # Low resolution rate
        if stats.get("unresolved_rate", 0) > 0.4:
            recommendations.append({
                "area": "Support Efficiency",
                "action": "Implement knowledge base improvements and agent training on top unresolved issues",
                "expected_impact": f"Potential to resolve {int(stats['total_calls'] * 0.2)} more calls",
                "priority": "High"
            })
        
        # Top trending issues
        if trends and trends[0]["frequency"] >= 5:
            top_issue = trends[0]
            recommendations.append({
                "area": "Product/Service Quality",
                "action": f"Address recurring issue: '{top_issue['issue']}' affecting {top_issue['frequency']} calls",
                "expected_impact": f"Could prevent {top_issue['percentage']:.0f}% of support calls",
                "priority": "High" if top_issue['frequency'] >= 10 else "Medium"
            })
        
        # Cluster-based recommendations
        clusters = root_cause.get("clusters", [])
        high_severity_clusters = [
            c for c in clusters 
            if c.get("cluster_id", -1) != -1 and c.get("severity_score", 0) > 10
        ]
        
        for cluster in high_severity_clusters[:2]:  # Top 2 severe clusters
            recommendations.append({
                "area": "Issue Resolution",
                "action": f"Create targeted resolution playbook for '{cluster.get('label', 'Unknown')}' pattern",
                "expected_impact": f"Affects {cluster.get('size', 0)} calls ({cluster.get('percentage', 0):.0f}%)",
                "priority": "Medium"
            })
        
        # Low confidence calls
        low_conf = stats.get("low_confidence_calls", 0)
        if low_conf > stats.get("total_calls", 1) * 0.2:
            recommendations.append({
                "area": "Data Quality",
                "action": "Review and improve call transcription/analysis quality for better insights",
                "expected_impact": f"Improve reliability of insights from {low_conf} calls",
                "priority": "Low"
            })
        
        return recommendations[:10]  # Limit to top 10

    # ===========================
    # PRIORITY AREAS
    # ===========================

    def _identify_priority_areas(
        self, 
        stats: Dict, 
        trends: List[Dict]
    ) -> List[str]:
        """Identify priority areas for immediate attention."""
        areas = []
        
        # Check each dimension
        if stats.get("negative_sentiment_rate", 0) > 0.3:
            areas.append("Customer Experience (high negative sentiment)")
        
        if stats.get("unresolved_rate", 0) > 0.4:
            areas.append("Support Efficiency (low resolution rate)")
        
        if stats.get("satisfaction_rate", 0) < 0.5:
            areas.append("Customer Satisfaction (below 50%)")
        
        # Check for critical trends
        critical_trends = [t for t in trends if t.get("trend_level") == "critical"]
        if critical_trends:
            areas.append(f"Product Quality ({len(critical_trends)} critical issues)")
        
        return areas if areas else ["Operations Stable"]

    # ===========================
    # CONFIDENCE SCORING
    # ===========================

    def _calculate_confidence_score(
        self, 
        stats: Dict, 
        analyses: List[Dict]
    ) -> float:
        """Calculate confidence in the insights."""
        confidence = 0.5  # Base confidence
        
        # Factor 1: Sample size
        total = stats.get("total_calls", 0)
        if total >= 50:
            confidence += 0.2
        elif total >= 20:
            confidence += 0.1
        elif total < 10:
            confidence -= 0.2
        
        # Factor 2: Data quality (average confidence)
        avg_conf = stats.get("average_confidence", 0.5)
        confidence += (avg_conf - 0.5) * 0.3
        
        # Factor 3: Clear patterns
        if stats.get("negative_sentiment_rate", 0) > 0.3 or stats.get("positive_sentiment_rate", 0) > 0.5:
            confidence += 0.1
        
        # Factor 4: Low confidence calls
        if stats.get("low_confidence_calls", 0) > total * 0.3:
            confidence -= 0.1
        
        return round(min(max(confidence, 0.0), 1.0), 2)

    # ===========================
    # ALERTS
    # ===========================

    def _generate_alerts(
        self, 
        stats: Dict, 
        trends: List[Dict]
    ) -> List[str]:
        """Generate critical alerts for immediate attention."""
        alerts = []
        
        # High negative sentiment
        neg_rate = stats.get("negative_sentiment_rate", 0)
        if neg_rate > 0.5:
            alerts.append(f"🚨 CRITICAL: {neg_rate*100:.0f}% negative sentiment - immediate action required")
        elif neg_rate > 0.35:
            alerts.append(f"⚠️ WARNING: {neg_rate*100:.0f}% negative sentiment - trending concerning")
        
        # High unresolved rate
        unres_rate = stats.get("unresolved_rate", 0)
        if unres_rate > 0.6:
            alerts.append(f"🚨 CRITICAL: {unres_rate*100:.0f}% unresolved rate - support backlog building")
        elif unres_rate > 0.45:
            alerts.append(f"⚠️ WARNING: {unres_rate*100:.0f}% unresolved rate - review processes")
        
        # Critical trending issues
        critical_issues = [t for t in trends if t.get("trend_level") == "critical"]
        if len(critical_issues) >= 3:
            alerts.append(f"⚠️ {len(critical_issues)} critical recurring issues detected")
        
        # Low satisfaction
        sat_rate = stats.get("satisfaction_rate", 0)
        if sat_rate < 0.3:
            alerts.append(f"⚠️ Low customer satisfaction: {sat_rate*100:.0f}%")
        
        return alerts

    # ===========================
    # FALLBACK
    # ===========================

    def _create_fallback_insights(self) -> Dict:
        """Create minimal fallback insights structure."""
        return {
            "statistics": self._empty_stats(),
            "trends": [],
            "key_insights": ["Analysis incomplete - insufficient data"],
            "risks": ["Unable to assess risks"],
            "opportunities": ["Collect more data for better insights"],
            "recommendations": [],
            "priority_areas": ["Data Collection"],
            "confidence_score": 0.0,
            "alerts": ["⚠️ Insight generation failed - manual review required"]
        }