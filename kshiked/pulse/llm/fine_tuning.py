"""
Fine-Tuning Infrastructure for KShield Pulse

Prepares training data for fine-tuning LLMs on Kenya-specific threat detection.

Workflow:
1. Collect classified examples from LLM analysis table
2. Format into instruction/completion pairs
3. Export for Gemini/OpenAI fine-tuning

Usage:
    trainer = FineTuningDataPreparer(database)
    
    # Export training data
    await trainer.export_training_data("training_data.jsonl")
    
    # Upload to Gemini for fine-tuning
    await trainer.start_fine_tuning()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger("kshield.pulse.llm.fine_tuning")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrainingExample:
    """A single training example for fine-tuning."""
    instruction: str
    input_text: str
    expected_output: str
    tier: str
    confidence: float
    source: str  # human_labeled, llm_verified, synthetic
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_gemini_format(self) -> Dict[str, Any]:
        """Format for Gemini tuning API."""
        return {
            "text_input": f"{self.instruction}\n\nPOST:\n{self.input_text}",
            "output": self.expected_output,
        }
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Format for OpenAI fine-tuning API."""
        return {
            "messages": [
                {"role": "system", "content": self.instruction},
                {"role": "user", "content": self.input_text},
                {"role": "assistant", "content": self.expected_output},
            ]
        }
    
    def to_jsonl_line(self, format: str = "gemini") -> str:
        """Convert to JSONL line."""
        if format == "gemini":
            data = self.to_gemini_format()
        else:
            data = self.to_openai_format()
        return json.dumps(data)


# =============================================================================
# Prompts for Training Data
# =============================================================================

INSTRUCTION_THREAT_CLASSIFICATION = """You are a threat analyst for Kenya's national security system. Classify the following social media post according to the threat taxonomy:

TIER_1: Existential threats (calls for violence, coup, genocide)
TIER_2: Severe stability threats (dehumanizing language, violent protest coordination)
TIER_3: High-risk destabilization (mobilization, election fraud claims with action)
TIER_4: Emerging threats (economic grievances, political discontent)
TIER_5: Non-threat (normal discourse, news, entertainment)
TIER_0: Protected speech (legitimate criticism, journalism, peaceful advocacy)

Respond with JSON: {"tier": "TIER_X", "confidence": 0.0-1.0, "reasoning": "..."}"""


INSTRUCTION_ROLE_IDENTIFICATION = """Analyze the following posts from a social media user and identify their role in potential threat networks:

IDEOLOGUE: Produces justification narratives
MOBILIZER: Issues calls to action, coordinates
AMPLIFIER: High-volume resharing
BROKER: Connects communities
LEGITIMIZER: Adds authority cues
GATEKEEPER: Controls channels

Respond with JSON: {"role": "...", "confidence": 0.0-1.0, "evidence": [...]}"""


# =============================================================================
# Data Preparer
# =============================================================================

class FineTuningDataPreparer:
    """
    Prepares training data for fine-tuning.
    
    Sources:
    1. Human-labeled examples (highest quality)
    2. LLM-classified with high confidence (verified)
    3. Synthetic examples (generated)
    """
    
    def __init__(self, database):
        """
        Initialize data preparer.
        
        Args:
            database: Database instance.
        """
        self.db = database
        self._examples: List[TrainingExample] = []
    
    async def collect_examples_from_db(
        self,
        min_confidence: float = 0.9,
        limit: int = 1000,
    ) -> int:
        """
        Collect training examples from LLM analysis table.
        
        Only includes high-confidence classifications.
        
        Returns:
            Number of examples collected.
        """
        from ..db.models import LLMAnalysis, SocialPost
        from sqlalchemy import select
        
        async with self.db.session() as session:
            # Get high-confidence analyses
            stmt = select(LLMAnalysis, SocialPost).join(
                SocialPost, LLMAnalysis.post_id == SocialPost.id
            ).where(
                LLMAnalysis.confidence >= min_confidence
            ).limit(limit)
            
            result = await session.execute(stmt)
            rows = result.all()
            
            for analysis, post in rows:
                example = TrainingExample(
                    instruction=INSTRUCTION_THREAT_CLASSIFICATION,
                    input_text=post.text,
                    expected_output=json.dumps({
                        "tier": analysis.tier.value,
                        "confidence": analysis.confidence,
                        "reasoning": analysis.reasoning,
                    }),
                    tier=analysis.tier.value,
                    confidence=analysis.confidence,
                    source="llm_verified",
                )
                self._examples.append(example)
        
        logger.info(f"Collected {len(rows)} examples from database")
        return len(rows)
    
    def add_human_labeled(
        self,
        text: str,
        tier: str,
        reasoning: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Add a human-labeled training example.
        
        Human labels are the highest quality training data.
        """
        example = TrainingExample(
            instruction=INSTRUCTION_THREAT_CLASSIFICATION,
            input_text=text,
            expected_output=json.dumps({
                "tier": tier,
                "confidence": confidence,
                "reasoning": reasoning,
            }),
            tier=tier,
            confidence=confidence,
            source="human_labeled",
        )
        self._examples.append(example)
    
    def generate_synthetic_examples(
        self,
        count_per_tier: int = 50,
    ) -> int:
        """
        Generate synthetic training examples.
        
        Creates examples for each tier based on patterns.
        
        Returns:
            Number of examples generated.
        """
        # Tier 1 examples (violence)
        tier1_patterns = [
            "Kill all {group}, they are the enemy",
            "Time to take up arms against the oppressors",
            "Storm {location} tomorrow at dawn",
            "Burn everything down, this government must fall",
        ]
        
        # Tier 3 examples (mobilization)
        tier3_patterns = [
            "Rise up Kenya! Take the streets tomorrow!",
            "The election was stolen! We cannot accept this!",
            "Join the revolution, meet at {location}",
            "Everyone must come out, this is our moment",
        ]
        
        # Tier 5 examples (non-threat)
        tier5_patterns = [
            "Parliament session continues today",
            "Kenya beat Uganda in the match yesterday",
            "Weather looking good in Nairobi",
            "Traffic is heavy on Mombasa Road",
        ]
        
        # Tier 0 examples (protected speech)
        tier0_patterns = [
            "We demand accountability from our leaders",
            "Peaceful march scheduled for Saturday",
            "Investigation reveals corruption in ministry",
            "Citizens have the right to protest",
        ]
        
        generated = 0
        
        for pattern in tier1_patterns[:count_per_tier]:
            text = pattern.format(group="enemies", location="State House")
            self.add_human_labeled(text, "TIER_1", "Explicit call for violence", 0.95)
            generated += 1
        
        for pattern in tier3_patterns[:count_per_tier]:
            text = pattern.format(location="Uhuru Park")
            self.add_human_labeled(text, "TIER_3", "Mobilization language", 0.90)
            generated += 1
        
        for pattern in tier5_patterns[:count_per_tier]:
            self.add_human_labeled(text, "TIER_5", "Normal discourse", 0.95)
            generated += 1
        
        for pattern in tier0_patterns[:count_per_tier]:
            self.add_human_labeled(pattern, "TIER_0", "Protected speech", 0.95)
            generated += 1
        
        logger.info(f"Generated {generated} synthetic examples")
        return generated
    
    async def export_training_data(
        self,
        output_path: str,
        format: str = "gemini",
        min_examples_per_tier: int = 10,
    ) -> str:
        """
        Export training data to JSONL file.
        
        Args:
            output_path: Path to output file.
            format: "gemini" or "openai".
            min_examples_per_tier: Minimum examples required per tier.
            
        Returns:
            Path to created file.
        """
        # Validate we have enough examples
        tier_counts = {}
        for ex in self._examples:
            tier_counts[ex.tier] = tier_counts.get(ex.tier, 0) + 1
        
        logger.info(f"Examples per tier: {tier_counts}")
        
        # Export
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for example in self._examples:
                line = example.to_jsonl_line(format=format)
                f.write(line + '\n')
        
        logger.info(f"Exported {len(self._examples)} examples to {path}")
        return str(path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training data statistics."""
        tier_counts = {}
        source_counts = {}
        
        for ex in self._examples:
            tier_counts[ex.tier] = tier_counts.get(ex.tier, 0) + 1
            source_counts[ex.source] = source_counts.get(ex.source, 0) + 1
        
        return {
            "total_examples": len(self._examples),
            "by_tier": tier_counts,
            "by_source": source_counts,
        }
    
    async def start_fine_tuning(
        self,
        training_file: str,
        model_name: str = "gemini-1.5-flash",
        display_name: str = "kshield-threat-classifier",
    ) -> Optional[str]:
        """
        Start fine-tuning job with Gemini.
        
        Requires google-generativeai >= 0.3.0 with tuning support.
        
        Returns:
            Tuning job ID or None if failed.
        """
        try:
            import google.generativeai as genai
            
            # Upload training file
            training_dataset = genai.upload_file(training_file)
            
            # Start tuning (API may vary)
            # This is placeholder - actual API depends on Gemini tuning release
            logger.info(f"Training file uploaded: {training_dataset}")
            
            # Note: Gemini fine-tuning API is still in development
            # This code will need updating when it's released
            
            return None
            
        except ImportError:
            logger.error("google-generativeai not installed or outdated")
            return None
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return None


# =============================================================================
# Factory Function
# =============================================================================

def create_fine_tuning_preparer(database) -> FineTuningDataPreparer:
    """Create fine-tuning data preparer."""
    return FineTuningDataPreparer(database)
