"""Insight Agent - Production-grade business insights generator."""

import json
from typing import Dict, List, Any
from collections import Counter

from loguru import logger
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from core.llm_client import OllamaLLMClient


# =========================
# STRUCTURED OUTPUT
# =========================

class InsightOutput(BaseModel):
    insights: List[str] = Field(description="List of business insights (3-5)")
    risks: List[str] = Field(description="Business risks")
    opportunities: List[str] = Field(description="Improvement opportunities")


# =========================
# AGENT
# =========================

class InsightAgent:
    """Production-grade Insight Agent."""

    def __init__(self, llm_client: OllamaLLMClient):
        self.llm = llm_client.llm
        self._build_chain()
        logger.info("Insight Agent initialized")

    # =========================
    # CHAIN
    # =========================

    def _build_chain(self):
        parser = PydanticOutputParser(pydantic_object=InsightOutput)

        system_prompt = """
        You are a senior business analyst.

        Generate:
        - Key insights
        - Business risks
        - Opportunities

        Be concise, actionable, and business-focused.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", """{summary}

{format_instructions}
""")
        ])

        self.chain = (
            prompt.partial(format_instructions=parser.get_format_instructions())
            | self.llm
            | parser
        )

    # =========================
    # MAIN ENTRY
    # =========================

    def generate_insights(
        self,
        root_cause_analysis: Dict,
        all_analyses: List[Dict]
    ) -> Dict:

        logger.info("Generating insights...")

        stats = self._compute_stats(all_analyses)
        trends = self._compute_trends(all_analyses)
        summary = self._build_summary(stats, trends, root_cause_analysis)

        # LLM with retry
        insights = self._generate_with_retry(summary)

        result = {
            "statistics": stats,
            "trends": trends,
            "insights": insights.get("insights", []),
            "risks": insights.get("risks", []),
            "opportunities": insights.get("opportunities", []),
            "priority_areas": self._priority_areas(stats),
            "confidence_score": self._confidence(stats),
            "alerts": self._alerts(stats)
        }

        logger.info("Insight generation complete")
        return result

    # =========================
    # STATS
    # =========================

    def _compute_stats(self, analyses: List[Dict]) -> Dict:
        if not analyses:
            return {
                "total_calls": 0,
                "negative_rate": 0.0,
                "unresolved_rate": 0.0
            }

        total = len(analyses)

        sentiments = [
            a.get("sentiment", {}).get("overall", "neutral")
            for a in analyses
        ]
        resolutions = [
            a.get("resolution_status", "unknown")
            for a in analyses
        ]

        neg_rate = sentiments.count("negative") / total
        unresolved = (
            resolutions.count("unresolved") +
            resolutions.count("escalated")
        ) / total

        return {
            "total_calls": total,
            "negative_rate": round(neg_rate, 3),
            "unresolved_rate": round(unresolved, 3)
        }

    # =========================
    # TRENDS
    # =========================

    def _compute_trends(self, analyses: List[Dict]) -> List[Dict]:
        issue_counter = Counter()

        for a in analyses:
            issue_counter.update(a.get("key_issues", []))

        trends = []
        for issue, freq in issue_counter.most_common(10):  # LIMIT added
            trends.append({
                "issue": issue,
                "frequency": freq,
                "trend": "high" if freq > 5 else "moderate"
            })

        return trends

    # =========================
    # SUMMARY
    # =========================

    def _build_summary(self, stats, trends, root_cause) -> str:
        cluster_summary = "\n".join([
            f"- {c.get('label')} ({c.get('size')} calls)"
            for c in root_cause.get("clusters", [])[:3]
            if c.get("cluster_id") != -1
        ])

        return f"""
Total Calls: {stats.get('total_calls')}
Negative Rate: {stats.get('negative_rate')*100:.1f}%
Unresolved Rate: {stats.get('unresolved_rate')*100:.1f}%

Top Issues:
{chr(10).join([f"- {t['issue']} ({t['frequency']})" for t in trends[:3]])}

Clusters:
{cluster_summary}
"""

    # =========================
    # LLM + RETRY
    # =========================

    def _generate_with_retry(self, summary: str) -> Dict:

        for attempt in range(2):  # retry once
            try:
                result = self.chain.invoke({"summary": summary})
                return result.model_dump()
            except Exception as e:
                logger.warning(f"LLM attempt {attempt+1} failed: {e}")

        logger.error("LLM completely failed → using fallback")
        return self._fallback(summary)

    # =========================
    # FALLBACK
    # =========================

    def _fallback(self, summary: str) -> Dict:
        return {
            "insights": ["High support volume detected"],
            "risks": ["Customer dissatisfaction risk"],
            "opportunities": ["Improve resolution efficiency"]
        }

    # =========================
    # PRIORITY
    # =========================

    def _priority_areas(self, stats: Dict) -> List[str]:
        areas = []

        if stats["negative_rate"] > 0.3:
            areas.append("Customer Experience")

        if stats["unresolved_rate"] > 0.4:
            areas.append("Support Efficiency")

        return areas or ["Stable"]

    # =========================
    # CONFIDENCE
    # =========================

    def _confidence(self, stats: Dict) -> float:
        base = 0.5

        if stats["total_calls"] > 50:
            base += 0.2
        if stats["negative_rate"] > 0.2:
            base += 0.1
        if stats["total_calls"] < 10:
            base -= 0.2

        return round(min(base, 1.0), 2)

    # =========================
    # ALERTS
    # =========================

    def _alerts(self, stats: Dict) -> List[str]:
        alerts = []

        if stats["negative_rate"] > 0.4:
            alerts.append(
                f"🚨 High negative sentiment: {stats['negative_rate']*100:.1f}%"
            )

        if stats["unresolved_rate"] > 0.5:
            alerts.append(
                f"⚠️ High unresolved rate: {stats['unresolved_rate']*100:.1f}%"
            )

        return alerts