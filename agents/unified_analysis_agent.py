"""Unified Analysis Agent - Production-grade call analysis with comprehensive error handling."""

from typing import List, Optional, Dict, Any, Literal
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import time
from functools import wraps

from core.llm_client import OllamaLLMClient


# ===========================
# RETRY DECORATOR
# ===========================

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator for handling transient failures."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay * attempt)  # Exponential backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
            raise last_exception
        return wrapper
    return decorator


# ===========================
# OUTPUT SCHEMAS
# ===========================

class Sentiment(BaseModel):
    """Sentiment analysis structure."""
    overall: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment of the conversation"
    )
    customer_emotion: str = Field(
        description="Customer's emotional state (e.g., frustrated, satisfied, confused)"
    )
    agent_tone: str = Field(
        description="Agent's tone (e.g., professional, empathetic, rushed)"
    )


class Satisfaction(BaseModel):
    """Customer satisfaction indicators."""
    explicit_satisfaction: Literal["yes", "no", "unclear"] = Field(
        description="Did customer explicitly express satisfaction?"
    )
    tone_shift: Literal["improved", "declined", "stable"] = Field(
        description="How did customer's tone change during call?"
    )


class AnalysisOutput(BaseModel):
    """Complete structured analysis output."""
    intent: str = Field(
        description="Primary intent/reason for the call"
    )
    sentiment: Sentiment = Field(
        description="Sentiment analysis of the conversation"
    )
    summary: str = Field(
        description="Brief summary of the call (2-3 sentences)"
    )
    key_issues: List[str] = Field(
        description="List of key issues discussed (3-5 items)",
        min_items=1,
        max_items=10
    )
    resolution_status: Literal["resolved", "unresolved", "escalated", "pending"] = Field(
        description="Current resolution status"
    )
    customer_satisfaction_indicators: Satisfaction = Field(
        description="Customer satisfaction signals"
    )
    product_mentioned: Optional[str] = Field(
        default=None,
        description="Product or service mentioned (if any)"
    )
    action_items: List[str] = Field(
        default_factory=list,
        description="Action items or follow-ups required"
    )


# ===========================
# AGENT
# ===========================

class UnifiedAnalysisAgent:
    """
    Production-grade agent for comprehensive call analysis.
    
    Features:
    - Structured output using Pydantic
    - Retry mechanism with exponential backoff
    - Comprehensive error handling
    - Confidence tracking
    - Fallback strategies
    """
    
    def __init__(self, llm_client: OllamaLLMClient, max_retries: int = 3):
        """
        Initialize Unified Analysis Agent.
        
        Args:
            llm_client: LangChain LLM client instance
            max_retries: Maximum retry attempts for LLM calls
        """
        self.llm = llm_client.llm
        self.max_retries = max_retries
        self._build_chain()
        logger.info("✅ Unified Analysis Agent initialized (production mode)")

    # ===========================
    # CHAIN BUILDING
    # ===========================

    def _build_chain(self):
        """Build LangChain chain for call analysis."""
        parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
        
        system_prompt = """You are an expert customer support analyst with years of experience.

Your task:
- Extract structured insights from call transcripts with HIGH ACCURACY
- Be precise, concise, and evidence-based
- Identify subtle sentiment cues and emotional undertones
- Provide actionable insights
- NEVER hallucinate or make assumptions

Quality standards:
- Summary must be factual and derived from transcript
- Issues must be specific and actionable
- Sentiment must reflect actual conversation tone
- Resolution status must be based on clear indicators

Return ONLY valid JSON matching the schema."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", """Analyze this customer support call transcript:

Transcript:
{transcript}

{format_instructions}""")
        ])
        
        # Build chain: prompt -> LLM -> parser
        self.chain = (
            prompt.partial(format_instructions=parser.get_format_instructions())
            | self.llm
            | parser
        )
        
        logger.debug("Analysis chain built successfully")

    # ===========================
    # MAIN ANALYSIS FUNCTION
    # ===========================

    @retry_on_failure(max_attempts=3, delay=1.0)
    def analyze(
        self, 
        labeled_transcript: str, 
        call_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Perform complete analysis of a call transcript.
        
        Args:
            labeled_transcript: Transcript with Agent:/Customer: labels
            call_id: Unique identifier for the call
            metadata: Additional context (e.g., timestamp, duration)
            
        Returns:
            Complete analysis with confidence scores and metadata
        """
        if not labeled_transcript or len(labeled_transcript.strip()) < 20:
            logger.warning(f"Transcript too short for call {call_id}")
            return self._create_fallback_analysis(
                call_id=call_id,
                reason="transcript_too_short"
            )
        
        logger.info(f"🔍 Analyzing call {call_id or 'unknown'}...")
        
        start_time = time.time()
        result = None
        
        # Execute LLM analysis with retry
        try:
            result = self.chain.invoke({"transcript": labeled_transcript})
            analysis_time = time.time() - start_time
            logger.debug(f"LLM analysis completed in {analysis_time:.2f}s")
            
        except Exception as e:
            logger.error(f"LLM analysis failed for call {call_id}: {e}")
            return self._create_fallback_analysis(
                call_id=call_id,
                reason=f"llm_error: {str(e)}"
            )
        
        # Convert Pydantic model to dict
        result_dict = result.model_dump()
        
        # Enrich with metadata
        result_dict.update({
            "call_id": call_id,
            "analyzed": True,
            "analysis_time_seconds": round(time.time() - start_time, 2),
            "confidence_score": self._calculate_confidence(result_dict, labeled_transcript),
            "quality_flags": self._assess_quality(result_dict),
            "metadata": metadata or {}
        })
        
        # Validation and normalization
        result_dict = self._validate_and_normalize(result_dict)
        
        logger.info(
            f"✅ Analysis complete - Intent: {result_dict.get('intent')}, "
            f"Sentiment: {result_dict.get('sentiment', {}).get('overall')}, "
            f"Confidence: {result_dict.get('confidence_score'):.2f}"
        )
        
        return result_dict

    # ===========================
    # CONFIDENCE CALCULATION
    # ===========================

    def _calculate_confidence(
        self, 
        analysis: Dict[str, Any], 
        transcript: str
    ) -> float:
        """
        Calculate confidence score for the analysis.
        
        Factors:
        - Transcript length (longer = more reliable)
        - Number of issues identified
        - Sentiment clarity
        - Resolution status clarity
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Transcript length
        transcript_length = len(transcript.split())
        if transcript_length > 200:
            confidence += 0.2
        elif transcript_length > 100:
            confidence += 0.1
        elif transcript_length < 30:
            confidence -= 0.2
        
        # Factor 2: Issue specificity
        issues = analysis.get("key_issues", [])
        if len(issues) >= 2:
            confidence += 0.1
        if all(len(issue) > 10 for issue in issues):  # Specific issues
            confidence += 0.1
        
        # Factor 3: Sentiment confidence
        sentiment = analysis.get("sentiment", {})
        if sentiment.get("overall") in ["positive", "negative"]:  # Clear sentiment
            confidence += 0.1
        
        # Factor 4: Resolution clarity
        resolution = analysis.get("resolution_status")
        if resolution in ["resolved", "unresolved"]:  # Clear outcome
            confidence += 0.1
        
        return round(min(max(confidence, 0.0), 1.0), 2)

    # ===========================
    # QUALITY ASSESSMENT
    # ===========================

    def _assess_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess quality of the analysis output.
        
        Returns:
            Quality flags and warnings
        """
        flags = {
            "has_warnings": False,
            "warnings": [],
            "completeness": 1.0
        }
        
        # Check for generic/vague issues
        issues = analysis.get("key_issues", [])
        generic_keywords = ["general", "issue", "problem", "help", "support"]
        
        for issue in issues:
            if any(keyword in issue.lower() for keyword in generic_keywords):
                if len(issue.split()) < 3:  # Very short and generic
                    flags["warnings"].append("Some issues may be too generic")
                    flags["has_warnings"] = True
                    break
        
        # Check summary quality
        summary = analysis.get("summary", "")
        if len(summary.split()) < 10:
            flags["warnings"].append("Summary may be too brief")
            flags["has_warnings"] = True
        
        # Check completeness
        required_fields = ["intent", "sentiment", "summary", "key_issues", "resolution_status"]
        missing_fields = [f for f in required_fields if not analysis.get(f)]
        
        if missing_fields:
            flags["completeness"] = 1.0 - (len(missing_fields) / len(required_fields))
            flags["warnings"].append(f"Missing fields: {', '.join(missing_fields)}")
            flags["has_warnings"] = True
        
        return flags

    # ===========================
    # VALIDATION & NORMALIZATION
    # ===========================

    def _validate_and_normalize(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize analysis output."""
        
        # Ensure key_issues exists and is non-empty
        if not analysis.get("key_issues"):
            logger.warning("No issues detected - adding placeholder")
            analysis["key_issues"] = ["General inquiry or support request"]
        
        # Normalize intent (lowercase, underscore-separated)
        intent = analysis.get("intent", "unknown")
        analysis["intent"] = intent.lower().replace(" ", "_").replace("-", "_")
        
        # Normalize sentiment
        if "sentiment" in analysis and "overall" in analysis["sentiment"]:
            analysis["sentiment"]["overall"] = analysis["sentiment"]["overall"].lower()
        
        # Ensure action_items is a list
        if not isinstance(analysis.get("action_items"), list):
            analysis["action_items"] = []
        
        # Truncate long fields
        if len(analysis.get("summary", "")) > 500:
            analysis["summary"] = analysis["summary"][:497] + "..."
            logger.warning("Summary truncated to 500 characters")
        
        return analysis

    # ===========================
    # FALLBACK STRATEGY
    # ===========================

    def _create_fallback_analysis(
        self, 
        call_id: str = None, 
        reason: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Create minimal fallback analysis when LLM fails.
        
        Args:
            call_id: Call identifier
            reason: Reason for fallback
            
        Returns:
            Safe fallback analysis structure
        """
        logger.warning(f"Using fallback analysis for call {call_id} - Reason: {reason}")
        
        return {
            "call_id": call_id,
            "analyzed": False,
            "fallback_used": True,
            "fallback_reason": reason,
            "intent": "unknown",
            "sentiment": {
                "overall": "neutral",
                "customer_emotion": "unclear",
                "agent_tone": "professional"
            },
            "summary": "Analysis could not be completed - manual review required",
            "key_issues": ["Analysis error - requires manual review"],
            "resolution_status": "unclear",
            "customer_satisfaction_indicators": {
                "explicit_satisfaction": "unclear",
                "tone_shift": "stable"
            },
            "product_mentioned": None,
            "action_items": ["Manual review required"],
            "confidence_score": 0.0,
            "quality_flags": {
                "has_warnings": True,
                "warnings": ["Fallback analysis used"],
                "completeness": 0.0
            }
        }

    # ===========================
    # UTILITY FUNCTIONS
    # ===========================

    def extract_issues_text(self, analysis: Dict[str, Any]) -> str:
        """
        Extract issues as concatenated text for embedding/clustering.
        
        Args:
            analysis: Analysis dictionary
            
        Returns:
            Concatenated text of summary and issues
        """
        issues = analysis.get("key_issues", [])
        summary = analysis.get("summary", "")
        
        return f"{summary} Issues: {' | '.join(issues)}".strip()

    def get_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metrics from analysis.
        
        Returns:
            Dictionary of metrics for monitoring/reporting
        """
        return {
            "intent": analysis.get("intent"),
            "sentiment": analysis.get("sentiment", {}).get("overall"),
            "resolved": analysis.get("resolution_status") == "resolved",
            "issue_count": len(analysis.get("key_issues", [])),
            "has_action_items": len(analysis.get("action_items", [])) > 0,
            "confidence": analysis.get("confidence_score", 0.0),
            "quality_score": analysis.get("quality_flags", {}).get("completeness", 0.0)
        }