"""Diarization Agent - Production-grade speaker labeling with comprehensive error handling."""

import re
from typing import List, Dict, Any, Literal
import time
from functools import wraps

from loguru import logger
from pydantic import BaseModel, Field, validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

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
                        time.sleep(delay * attempt)
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
            raise last_exception
        return wrapper
    return decorator


# ===========================
# OUTPUT SCHEMAS
# ===========================

class DialogueTurn(BaseModel):
    """Single dialogue turn with speaker label."""
    speaker: Literal["Agent", "Customer"] = Field(
        description="Speaker role: Agent or Customer"
    )
    text: str = Field(
        description="What the speaker said",
        min_length=1
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in speaker identification (0.0-1.0)"
    )
    
    @validator('text')
    def text_not_empty(cls, v):
        """Ensure text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class DiarizationOutput(BaseModel):
    """Complete diarization output with metadata."""
    dialogue: List[DialogueTurn] = Field(
        description="List of labeled dialogue turns",
        min_items=1
    )
    agent_turns: int = Field(
        ge=0,
        description="Number of agent turns"
    )
    customer_turns: int = Field(
        ge=0,
        description="Number of customer turns"
    )
    total_turns: int = Field(
        ge=1,
        description="Total number of turns"
    )


# ===========================
# AGENT
# ===========================

class DiarizationAgent:
    """
    Production-grade agent for speaker diarization.
    
    Features:
    - LLM-based intelligent speaker labeling
    - Rule-based fallback for reliability
    - Confidence scoring
    - Quality validation
    - Conversation flow analysis
    """
    
    def __init__(self, llm_client: OllamaLLMClient, max_retries: int = 3):
        """
        Initialize Diarization Agent.
        
        Args:
            llm_client: LangChain LLM client instance
            max_retries: Maximum retry attempts for LLM calls
        """
        self.llm = llm_client.llm
        self.max_retries = max_retries
        self._build_chain()
        logger.info("✅ Diarization Agent initialized (production mode)")

    # ===========================
    # CHAIN BUILDING
    # ===========================

    def _build_chain(self):
        """Build LangChain chain for diarization."""
        parser = PydanticOutputParser(pydantic_object=DiarizationOutput)

        system_prompt = """You are an expert call diarization system specialized in customer support conversations.

Task:
- Split transcript into sentences or natural conversation turns
- Label each turn as "Agent" or "Customer"
- Assign confidence score (0.0 to 1.0) based on contextual clues
- Maintain consistent speaker identity throughout conversation
- Look for cues like greetings, technical language, problem descriptions

Context clues:
- Agents typically: greet first, ask questions, provide solutions, use professional language
- Customers typically: describe problems, ask for help, express frustration/satisfaction
- Avoid random switching unless context clearly indicates speaker change

Return ONLY valid JSON matching the schema."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", """Diarize this customer support call transcript:

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
        
        logger.debug("Diarization chain built successfully")

    # ===========================
    # MAIN FUNCTION
    # ===========================

    @retry_on_failure(max_attempts=3, delay=1.0)
    def label_speakers(self, raw_transcript: str) -> Dict[str, Any]:
        """
        Label speakers in transcript using LLM with fallback.
        
        Args:
            raw_transcript: Raw transcript without speaker labels
            
        Returns:
            Labeled dialogue with metadata and quality metrics
        """
        if not raw_transcript or len(raw_transcript.strip()) < 10:
            logger.warning("Transcript too short for diarization")
            return self._create_minimal_fallback(raw_transcript)
        
        logger.info("🎙️ Running diarization...")
        start_time = time.time()
        
        try:
            # Attempt LLM-based diarization
            result = self.chain.invoke({"transcript": raw_transcript})
            diarization_time = time.time() - start_time
            
            logger.debug(f"LLM diarization completed in {diarization_time:.2f}s")
            
            # Validate output
            if not self._validate_diarization_output(result):
                logger.warning("LLM output failed validation - using fallback")
                return self._fallback(raw_transcript)
            
            # Build structured output
            dialogue_list = [turn.model_dump() for turn in result.dialogue]
            
            # Count turns safely
            agent_turns = sum(1 for d in result.dialogue if d.speaker == "Agent")
            customer_turns = sum(1 for d in result.dialogue if d.speaker == "Customer")
            
            output = {
                "dialogue": dialogue_list,
                "agent_turns": agent_turns,
                "customer_turns": customer_turns,
                "total_turns": len(dialogue_list),
                "labeled_transcript": self._format_labeled_transcript(dialogue_list),
                "diarization_time_seconds": round(diarization_time, 2),
                "method": "llm",
                "quality_metrics": self._calculate_quality_metrics(dialogue_list, raw_transcript)
            }
            
            logger.info(
                f"✅ Diarization complete - "
                f"{agent_turns} agent, {customer_turns} customer turns"
            )
            
            return output
            
        except Exception as e:
            logger.error(f"LLM diarization failed: {e}")
            return self._fallback(raw_transcript)

    # ===========================
    # VALIDATION
    # ===========================

    def _validate_diarization_output(self, result: DiarizationOutput) -> bool:
        """
        Validate LLM diarization output.
        
        Args:
            result: Diarization output from LLM
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check minimum turns
            if len(result.dialogue) < 2:
                logger.warning("Too few dialogue turns")
                return False
            
            # Check both speakers present
            speakers = {turn.speaker for turn in result.dialogue}
            if len(speakers) < 2:
                logger.warning("Only one speaker detected")
                return False
            
            # Check for excessive single-speaker runs
            max_consecutive = 0
            current_consecutive = 1
            prev_speaker = result.dialogue[0].speaker
            
            for turn in result.dialogue[1:]:
                if turn.speaker == prev_speaker:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 1
                prev_speaker = turn.speaker
            
            # Warning if one speaker dominates too much
            if max_consecutive > len(result.dialogue) * 0.7:
                logger.warning(f"Excessive consecutive turns: {max_consecutive}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    # ===========================
    # FALLBACK STRATEGY
    # ===========================

    def _fallback(self, transcript: str) -> Dict[str, Any]:
        """
        Rule-based fallback diarization.
        
        Uses simple heuristics:
        - Split by sentence
        - Alternate speakers
        - Adjust based on keyword patterns
        
        Args:
            transcript: Raw transcript
            
        Returns:
            Diarization output using rule-based approach
        """
        logger.warning("Using rule-based fallback diarization")
        
        # Split into sentences
        sentences = self._split_into_sentences(transcript)
        
        if len(sentences) < 2:
            return self._create_minimal_fallback(transcript)
        
        dialogue = []
        
        # First turn detection
        first_speaker = self._detect_first_speaker(sentences[0])
        
        for i, sentence in enumerate(sentences):
            # Alternate speakers by default
            if first_speaker == "Agent":
                speaker = "Agent" if i % 2 == 0 else "Customer"
            else:
                speaker = "Customer" if i % 2 == 0 else "Agent"
            
            # Adjust based on keywords
            speaker = self._refine_speaker_by_keywords(sentence, speaker)
            
            dialogue.append({
                "speaker": speaker,
                "text": sentence,
                "confidence": 0.6  # Lower confidence for rule-based
            })
        
        agent_turns = sum(1 for d in dialogue if d["speaker"] == "Agent")
        customer_turns = sum(1 for d in dialogue if d["speaker"] == "Customer")
        
        return {
            "dialogue": dialogue,
            "agent_turns": agent_turns,
            "customer_turns": customer_turns,
            "total_turns": len(dialogue),
            "labeled_transcript": self._format_labeled_transcript(dialogue),
            "method": "rule_based_fallback",
            "quality_metrics": self._calculate_quality_metrics(dialogue, transcript)
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split by common sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no sentence breaks, split by length
        if len(sentences) == 1 and len(sentences[0]) > 200:
            words = sentences[0].split()
            chunk_size = len(words) // 4  # Split into ~4 chunks
            sentences = [
                ' '.join(words[i:i+chunk_size]) 
                for i in range(0, len(words), chunk_size)
            ]
        
        return sentences

    def _detect_first_speaker(self, first_sentence: str) -> str:
        """Detect who speaks first based on greeting patterns."""
        first_lower = first_sentence.lower()
        
        # Agent greeting patterns
        agent_patterns = [
            "thank you for calling",
            "customer service",
            "how may i help",
            "how can i assist",
            "speaking",
            "my name is"
        ]
        
        # Customer problem patterns
        customer_patterns = [
            "i have a problem",
            "i need help",
            "my order",
            "not working",
            "issue with"
        ]
        
        # Check patterns
        for pattern in agent_patterns:
            if pattern in first_lower:
                return "Agent"
        
        for pattern in customer_patterns:
            if pattern in first_lower:
                return "Customer"
        
        # Default: agent speaks first in support calls
        return "Agent"

    def _refine_speaker_by_keywords(self, sentence: str, default_speaker: str) -> str:
        """Refine speaker identification using keyword patterns."""
        sentence_lower = sentence.lower()
        
        # Strong agent indicators
        agent_keywords = [
            "let me check",
            "i can help",
            "i'll transfer",
            "account shows",
            "according to",
            "our policy",
            "let me verify"
        ]
        
        # Strong customer indicators
        customer_keywords = [
            "i ordered",
            "i received",
            "my account",
            "i can't",
            "doesn't work",
            "i'm frustrated"
        ]
        
        # Check for strong signals
        for keyword in agent_keywords:
            if keyword in sentence_lower:
                return "Agent"
        
        for keyword in customer_keywords:
            if keyword in sentence_lower:
                return "Customer"
        
        # No strong signal - use default
        return default_speaker

    def _create_minimal_fallback(self, transcript: str) -> Dict[str, Any]:
        """Create minimal fallback for very short transcripts."""
        dialogue = [
            {
                "speaker": "Agent",
                "text": transcript[:len(transcript)//2] if len(transcript) > 20 else transcript,
                "confidence": 0.5
            }
        ]
        
        if len(transcript) > 20:
            dialogue.append({
                "speaker": "Customer",
                "text": transcript[len(transcript)//2:],
                "confidence": 0.5
            })
        
        return {
            "dialogue": dialogue,
            "agent_turns": 1,
            "customer_turns": min(1, len(dialogue) - 1),
            "total_turns": len(dialogue),
            "labeled_transcript": self._format_labeled_transcript(dialogue),
            "method": "minimal_fallback",
            "quality_metrics": {"quality_score": 0.3, "warnings": ["Minimal fallback used"]}
        }

    # ===========================
    # FORMATTING
    # ===========================

    def _format_labeled_transcript(self, dialogue: List[Dict[str, Any]]) -> str:
        """
        Format dialogue as labeled transcript.
        
        Args:
            dialogue: List of dialogue turns
            
        Returns:
            Formatted transcript with speaker labels
        """
        lines = []
        for turn in dialogue:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            lines.append(f"{speaker}: {text}")
        
        return "\n".join(lines)

    # ===========================
    # QUALITY METRICS
    # ===========================

    def _calculate_quality_metrics(
        self, 
        dialogue: List[Dict[str, Any]], 
        original_transcript: str
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics for diarization.
        
        Args:
            dialogue: Labeled dialogue
            original_transcript: Original transcript
            
        Returns:
            Quality metrics and warnings
        """
        metrics = {
            "quality_score": 0.0,
            "coverage": 0.0,
            "alternation_rate": 0.0,
            "avg_confidence": 0.0,
            "warnings": []
        }
        
        if not dialogue:
            metrics["warnings"].append("No dialogue turns")
            return metrics
        
        # Coverage: how much of original text is preserved
        labeled_text = " ".join(turn.get("text", "") for turn in dialogue)
        metrics["coverage"] = min(1.0, len(labeled_text) / max(len(original_transcript), 1))
        
        # Alternation rate: how often speakers change
        speakers = [turn.get("speaker") for turn in dialogue]
        switches = sum(1 for i in range(len(speakers) - 1) if speakers[i] != speakers[i + 1])
        metrics["alternation_rate"] = switches / max(len(speakers) - 1, 1)
        
        # Average confidence
        confidences = [turn.get("confidence", 0.5) for turn in dialogue]
        metrics["avg_confidence"] = sum(confidences) / len(confidences)
        
        # Overall quality score
        metrics["quality_score"] = (
            metrics["coverage"] * 0.4 +
            metrics["alternation_rate"] * 0.3 +
            metrics["avg_confidence"] * 0.3
        )
        
        # Generate warnings
        if metrics["coverage"] < 0.7:
            metrics["warnings"].append("Low coverage - some text may be missing")
        
        if metrics["alternation_rate"] < 0.2:
            metrics["warnings"].append("Low alternation - may indicate poor diarization")
        
        if metrics["avg_confidence"] < 0.5:
            metrics["warnings"].append("Low confidence scores")
        
        return metrics

    # ===========================
    # VALIDATION HELPER
    # ===========================

    def validate_diarization(self, diarization_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate quality of diarization output.
        
        Args:
            diarization_result: Output from label_speakers()
            
        Returns:
            Validation result with quality assessment
        """
        quality_metrics = diarization_result.get("quality_metrics", {})
        
        quality_score = quality_metrics.get("quality_score", 0.0)
        warnings = quality_metrics.get("warnings", [])
        
        # Determine if valid
        is_valid = (
            quality_score >= 0.5 and
            diarization_result.get("total_turns", 0) >= 2 and
            len(warnings) < 3
        )
        
        return {
            "valid": is_valid,
            "quality_score": quality_score,
            "quality_level": self._get_quality_level(quality_score),
            "warnings": warnings,
            "method": diarization_result.get("method", "unknown"),
            "total_turns": diarization_result.get("total_turns", 0)
        }

    def _get_quality_level(self, score: float) -> str:
        """Get quality level from score."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"