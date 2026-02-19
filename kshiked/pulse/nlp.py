"""
NLP Utilities — Core NLP Processing for Signal Detection

Provides:
- Sentiment Analysis (VADER-based for social media)
- Named Entity Recognition (regex-based + extensible)
- Emotion Detection (lexicon-based)
- Text Embeddings (TF-IDF based, can be upgraded to transformers)
- Text Preprocessing utilities
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter
import numpy as np

logger = logging.getLogger("kshield.pulse.nlp")


# =============================================================================
# Text Preprocessing
# =============================================================================

class TextPreprocessor:
    """Basic text preprocessing utilities."""
    
    # Common social media patterns
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    MENTION_PATTERN = re.compile(r'@\w+')
    HASHTAG_PATTERN = re.compile(r'#(\w+)')
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """Remove noise from social media text."""
        text = cls.URL_PATTERN.sub(' ', text)
        text = cls.MENTION_PATTERN.sub(' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @classmethod
    def extract_hashtags(cls, text: str) -> List[str]:
        """Extract hashtags from text."""
        return cls.HASHTAG_PATTERN.findall(text)
    
    @classmethod
    def extract_mentions(cls, text: str) -> List[str]:
        """Extract @mentions from text."""
        return [m[1:] for m in cls.MENTION_PATTERN.findall(text)]
    
    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Simple word tokenization."""
        text = cls.clean_text(text.lower())
        # Split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    @classmethod
    def extract_emojis(cls, text: str) -> List[str]:
        """Extract emojis from text."""
        return cls.EMOJI_PATTERN.findall(text)


# =============================================================================
# Sentiment Analysis (VADER-inspired)
# =============================================================================

@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    compound: float     # [-1, 1] overall sentiment
    positive: float     # [0, 1] positive component
    negative: float     # [0, 1] negative component  
    neutral: float      # [0, 1] neutral component
    intensity: float    # [0, 1] emotional intensity


class SentimentAnalyzer:
    """
    VADER-inspired sentiment analyzer optimized for social media.
    
    Uses a lexicon-based approach with:
    - Sentiment lexicon with intensity scores
    - Negation handling
    - Intensifier/diminisher modifiers
    - Emoji sentiment
    """
    
    # Core sentiment lexicon (subset - can be expanded)
    LEXICON: Dict[str, float] = {
        # Strong negative
        "hate": -3.0, "kill": -3.5, "death": -3.0, "destroy": -3.0,
        "terrible": -2.5, "awful": -2.5, "horrible": -2.5, "disaster": -2.5,
        "corrupt": -2.5, "evil": -3.0, "crisis": -2.0, "collapse": -2.5,
        "suffering": -2.5, "dying": -3.0, "starving": -3.0, "panic": -2.0,
        
        # Moderate negative
        "bad": -1.5, "wrong": -1.5, "problem": -1.0, "fail": -2.0,
        "angry": -2.0, "frustrated": -1.5, "tired": -1.0, "exhausted": -1.5,
        "expensive": -1.5, "poor": -1.5, "hopeless": -2.5, "desperate": -2.0,
        "fear": -2.0, "worried": -1.5, "scared": -2.0, "threat": -2.0,
        
        # Mild negative
        "sad": -1.0, "disappointed": -1.0, "difficult": -1.0, "hard": -0.5,
        "concern": -1.0, "trouble": -1.0, "issue": -0.5,
        
        # Neutral-ish
        "ok": 0.2, "fine": 0.3, "normal": 0.0,
        
        # Mild positive
        "good": 1.5, "nice": 1.0, "okay": 0.5, "better": 1.0,
        
        # Moderate positive
        "great": 2.0, "happy": 2.0, "love": 2.5, "excellent": 2.5,
        "wonderful": 2.5, "amazing": 2.5, "hope": 1.5, "progress": 1.5,
        "success": 2.0, "win": 2.0, "victory": 2.5,
        
        # Strong positive
        "fantastic": 3.0, "incredible": 3.0, "perfect": 3.0,
    }
    
    # Negation words
    NEGATIONS: Set[str] = {
        "not", "no", "never", "neither", "nobody", "nothing",
        "nowhere", "hardly", "barely", "cannot", "cant", "dont",
        "doesnt", "didnt", "wont", "wouldnt", "shouldnt", "couldnt",
        "without", "lack", "lacking",
    }
    
    # Intensifiers (boost sentiment)
    INTENSIFIERS: Dict[str, float] = {
        "very": 1.3, "really": 1.3, "extremely": 1.5, "absolutely": 1.5,
        "totally": 1.4, "completely": 1.4, "so": 1.2, "too": 1.2,
        "incredibly": 1.5, "especially": 1.3,
    }
    
    # Diminishers (reduce sentiment)
    DIMINISHERS: Dict[str, float] = {
        "slightly": 0.6, "somewhat": 0.7, "barely": 0.5, "hardly": 0.5,
        "kind": 0.7, "kinda": 0.7, "sort": 0.7, "sorta": 0.7,
        "little": 0.7, "bit": 0.8,
    }
    
    # Emoji sentiment
    EMOJI_SENTIMENT: Dict[str, float] = {
        "😀": 2.0, "😊": 2.0, "😃": 2.0, "😄": 2.0, "🙂": 1.0,
        "😢": -2.0, "😭": -2.5, "😡": -3.0, "😠": -2.5, "🤬": -3.0,
        "😱": -2.0, "😨": -2.0, "💀": -2.0, "🔥": 0.5,  # context-dependent
        "❤️": 2.0, "💔": -2.0, "👍": 1.5, "👎": -1.5,
    }
    
    def __init__(self):
        self.lexicon = self.LEXICON.copy()
        self.preprocessor = TextPreprocessor()
    
    def add_words(self, words: Dict[str, float]) -> None:
        """Add custom words to lexicon."""
        self.lexicon.update(words)
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text.
        
        Returns compound score [-1, 1] and component scores.
        """
        tokens = self.preprocessor.tokenize(text)
        
        if not tokens:
            return SentimentResult(
                compound=0.0, positive=0.0, negative=0.0, 
                neutral=1.0, intensity=0.0
            )
        
        scores = []
        prev_word = None
        prev_prev_word = None
        
        for i, token in enumerate(tokens):
            score = self.lexicon.get(token, 0.0)
            
            if score != 0.0:
                # Check for negation in previous 3 words
                if self._has_negation(tokens, i):
                    score *= -0.5  # Flip and reduce
                
                # Check for intensifier
                if prev_word and prev_word in self.INTENSIFIERS:
                    score *= self.INTENSIFIERS[prev_word]
                elif prev_prev_word and prev_prev_word in self.INTENSIFIERS:
                    score *= self.INTENSIFIERS[prev_prev_word]
                
                # Check for diminisher
                if prev_word and prev_word in self.DIMINISHERS:
                    score *= self.DIMINISHERS[prev_word]
                
                scores.append(score)
            
            prev_prev_word = prev_word
            prev_word = token
        
        # Add emoji sentiment
        for char in text:
            if char in self.EMOJI_SENTIMENT:
                scores.append(self.EMOJI_SENTIMENT[char])
        
        if not scores:
            return SentimentResult(
                compound=0.0, positive=0.0, negative=0.0,
                neutral=1.0, intensity=0.0
            )
        
        # Compute components
        pos_sum = sum(s for s in scores if s > 0)
        neg_sum = sum(abs(s) for s in scores if s < 0)
        total = pos_sum + neg_sum
        
        if total > 0:
            positive = pos_sum / total
            negative = neg_sum / total
        else:
            positive = negative = 0.0
        
        neutral = 1.0 - (positive + negative)
        
        # Compound score (normalized to [-1, 1])
        raw_compound = sum(scores)
        compound = raw_compound / math.sqrt(raw_compound**2 + 15)  # Normalize
        
        # Intensity
        intensity = min(1.0, total / (len(tokens) * 2.0))
        
        return SentimentResult(
            compound=compound,
            positive=positive,
            negative=negative,
            neutral=max(0.0, neutral),
            intensity=intensity
        )
    
    def _has_negation(self, tokens: List[str], index: int) -> bool:
        """Check if there's a negation word before the given index."""
        start = max(0, index - 3)
        for i in range(start, index):
            if tokens[i] in self.NEGATIONS:
                return True
        return False


# =============================================================================
# Named Entity Recognition (Pattern-based)
# =============================================================================

@dataclass
class Entity:
    """A named entity extracted from text."""
    text: str
    label: str      # PERSON, ORG, GPE (location), GROUP, etc.
    start: int
    end: int
    confidence: float = 0.8


class EntityRecognizer:
    """
    Pattern-based Named Entity Recognition.
    
    Designed for social/political context:
    - Political leaders and figures
    - Government institutions
    - Ethnic/regional groups
    - Organizations
    """
    
    # Pattern definitions (can be customized per country)
    PATTERNS: Dict[str, List[str]] = {
        "INSTITUTION": [
            r"\b(?:government|parliament|ministry|police|military|army)\b",
            r"\b(?:central bank|treasury|supreme court|judiciary)\b",
            r"\b(?:IMF|World Bank|UN|AU|EU)\b",
        ],
        "GROUP": [
            r"\b(?:protesters?|demonstrators?|activists?|citizens?)\b",
            r"\b(?:workers?|unions?|opposition|ruling party)\b",
            r"\b(?:youth|women|students?|farmers?)\b",
        ],
        "ACTION": [
            r"\b(?:protest|demonstration|strike|riot|march)\b",
            r"\b(?:election|vote|referendum|coup)\b",
            r"\b(?:arrest|detention|crackdown|violence)\b",
        ],
        "ECONOMIC": [
            r"\b(?:inflation|prices?|costs?|taxes?|debt)\b",
            r"\b(?:unemployment|jobs?|wages?|currency)\b",
            r"\b(?:GDP|economy|recession|crisis)\b",
        ],
    }
    
    def __init__(self):
        # Compile patterns
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {
            label: [re.compile(p, re.IGNORECASE) for p in patterns]
            for label, patterns in self.PATTERNS.items()
        }
        
        # Custom entities (loaded from config)
        self.custom_entities: Dict[str, str] = {}  # text -> label
    
    def add_entities(self, entities: Dict[str, str]) -> None:
        """Add custom entity mappings."""
        self.custom_entities.update({k.lower(): v for k, v in entities.items()})
    
    def extract(self, text: str) -> List[Entity]:
        """Extract named entities from text."""
        entities = []
        text_lower = text.lower()
        
        # Check custom entities first
        for entity_text, label in self.custom_entities.items():
            for match in re.finditer(re.escape(entity_text), text_lower):
                entities.append(Entity(
                    text=text[match.start():match.end()],
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95
                ))
        
        # Check pattern-based entities
        for label, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Avoid duplicates
                    if not any(
                        e.start == match.start() and e.end == match.end()
                        for e in entities
                    ):
                        entities.append(Entity(
                            text=match.group(),
                            label=label,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.8
                        ))
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        return entities


# =============================================================================
# Emotion Detection (Plutchik-based)
# =============================================================================

@dataclass
class EmotionResult:
    """Result from emotion detection."""
    emotions: Dict[str, float]  # emotion -> intensity [0, 1]
    dominant: str               # Most intense emotion
    arousal: float             # Overall emotional arousal [0, 1]


class EmotionDetector:
    """
    Plutchik-inspired emotion detection.
    
    Detects 8 primary emotions:
    - Joy, Trust, Fear, Surprise
    - Sadness, Disgust, Anger, Anticipation
    """
    
    # Emotion lexicons
    EMOTION_LEXICON: Dict[str, Dict[str, float]] = {
        "joy": {
            "happy": 0.9, "joy": 1.0, "love": 0.8, "excited": 0.8,
            "wonderful": 0.7, "great": 0.6, "amazing": 0.8, "celebrate": 0.7,
            "laugh": 0.7, "smile": 0.6, "hope": 0.5,
        },
        "trust": {
            "trust": 1.0, "believe": 0.7, "reliable": 0.7, "honest": 0.8,
            "faith": 0.8, "loyal": 0.7, "safe": 0.6, "support": 0.6,
        },
        "fear": {
            "fear": 1.0, "scared": 0.9, "terrified": 1.0, "afraid": 0.8,
            "panic": 0.9, "horror": 0.9, "threat": 0.7, "danger": 0.7,
            "worried": 0.6, "anxiety": 0.7, "nervous": 0.6,
        },
        "surprise": {
            "surprise": 1.0, "shocked": 0.9, "amazed": 0.7, "unexpected": 0.7,
            "sudden": 0.5, "astonished": 0.8, "stunned": 0.8,
        },
        "sadness": {
            "sad": 1.0, "depressed": 0.9, "grief": 0.9, "sorrow": 0.9,
            "crying": 0.8, "tears": 0.7, "miserable": 0.8, "hopeless": 0.8,
            "lonely": 0.7, "heartbroken": 0.9, "suffering": 0.7,
        },
        "disgust": {
            "disgust": 1.0, "disgusting": 1.0, "revolting": 0.9, "sick": 0.6,
            "vile": 0.9, "repulsive": 0.9, "corrupt": 0.7, "filthy": 0.8,
        },
        "anger": {
            "angry": 1.0, "rage": 1.0, "furious": 1.0, "hate": 0.9,
            "outrage": 0.9, "mad": 0.7, "frustrated": 0.6, "irritated": 0.5,
            "resentment": 0.7, "hostile": 0.8, "violence": 0.7,
        },
        "anticipation": {
            "anticipate": 1.0, "expect": 0.7, "await": 0.7, "predict": 0.5,
            "prepare": 0.5, "ready": 0.5, "planning": 0.5, "soon": 0.4,
        },
    }
    
    def __init__(self):
        self.lexicon = {
            emotion: dict(words) 
            for emotion, words in self.EMOTION_LEXICON.items()
        }
        self.preprocessor = TextPreprocessor()
    
    def add_words(self, emotion: str, words: Dict[str, float]) -> None:
        """Add custom words to an emotion lexicon."""
        if emotion in self.lexicon:
            self.lexicon[emotion].update(words)
    
    def detect(self, text: str) -> EmotionResult:
        """Detect emotions in text."""
        tokens = self.preprocessor.tokenize(text)
        
        # Score each emotion
        emotion_scores: Dict[str, float] = {e: 0.0 for e in self.lexicon}
        
        for token in tokens:
            for emotion, words in self.lexicon.items():
                if token in words:
                    emotion_scores[emotion] += words[token]
        
        # Normalize
        max_score = max(emotion_scores.values()) if emotion_scores else 0.0
        if max_score > 0:
            emotion_scores = {
                e: min(1.0, s / max_score) 
                for e, s in emotion_scores.items()
            }
        
        # Find dominant emotion
        dominant = max(emotion_scores, key=emotion_scores.get) if emotion_scores else "neutral"
        
        # Compute arousal (overall emotional intensity)
        arousal = min(1.0, sum(emotion_scores.values()) / 4.0)
        
        return EmotionResult(
            emotions=emotion_scores,
            dominant=dominant,
            arousal=arousal
        )


# =============================================================================
# Text Embeddings (TF-IDF based)
# =============================================================================

class TextEmbedder:
    """
    Simple TF-IDF based text embeddings.
    
    For production, this can be upgraded to use:
    - Sentence Transformers (all-MiniLM-L6-v2)
    - OpenAI embeddings
    - Local LLM embeddings
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0
        self.preprocessor = TextPreprocessor()
    
    def fit(self, documents: List[str]) -> None:
        """Fit vocabulary and IDF weights from documents."""
        word_doc_counts: Counter = Counter()
        all_words: Counter = Counter()
        
        for doc in documents:
            tokens = set(self.preprocessor.tokenize(doc))
            for token in tokens:
                word_doc_counts[token] += 1
                all_words[token] += 1
        
        # Build vocabulary from most common words
        most_common = all_words.most_common(self.vocab_size)
        self.vocab = {word: i for i, (word, _) in enumerate(most_common)}
        
        # Compute IDF
        self.doc_count = len(documents)
        self.idf = {
            word: math.log(self.doc_count / (1 + word_doc_counts[word]))
            for word in self.vocab
        }
    
    def embed(self, text: str) -> np.ndarray:
        """
        Compute TF-IDF embedding for text.
        
        Returns:
            numpy array of shape (vocab_size,)
        """
        tokens = self.preprocessor.tokenize(text)
        
        if not self.vocab:
            # No vocabulary fitted, return simple word count vector
            return np.zeros(self.vocab_size)
        
        # Compute TF
        tf: Counter = Counter(tokens)
        total = len(tokens) if tokens else 1
        
        # Build embedding
        embedding = np.zeros(len(self.vocab))
        for word, idx in self.vocab.items():
            if word in tf:
                embedding[idx] = (tf[word] / total) * self.idf.get(word, 1.0)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        return embedding
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2))


# =============================================================================
# Combined NLP Pipeline
# =============================================================================

@dataclass
class NLPResult:
    """Combined result from full NLP pipeline."""
    text: str
    sentiment: SentimentResult
    emotions: EmotionResult
    entities: List[Entity]
    hashtags: List[str]
    mentions: List[str]
    token_count: int


class NLPPipeline:
    """
    Combined NLP pipeline for signal detection.
    
    Runs all NLP analysis in a single pass.
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment = SentimentAnalyzer()
        self.emotion = EmotionDetector()
        self.ner = EntityRecognizer()
        self.embedder = TextEmbedder()
    
    def analyze(self, text: str) -> NLPResult:
        """Run full NLP analysis on text."""
        tokens = self.preprocessor.tokenize(text)
        
        return NLPResult(
            text=text,
            sentiment=self.sentiment.analyze(text),
            emotions=self.emotion.detect(text),
            entities=self.ner.extract(text),
            hashtags=self.preprocessor.extract_hashtags(text),
            mentions=self.preprocessor.extract_mentions(text),
            token_count=len(tokens)
        )
    
    def analyze_batch(self, texts: List[str]) -> List[NLPResult]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]
