"""Sentiment analysis engine computing valence polarity, subjectivity, and sentiment class.

Provides a four-way sentiment dimension (Positive, Negative, Neutral, Mixed) with
valence scoring, negation flipping, intensifier boosting, and confidence calibration.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import pandas as pd

from src.data_ingestion.schemas import ClassificationStatus, SentimentLabel
from src.nlp.text_preprocessor import TextPreprocessor, PreprocessedText


class SentimentAnalyzer:
    """Computes sentiment polarity and sentiment categorization for social comments."""

    # Lexicon with base valences in [-1.0, 1.0]
    POSITIVE_LEXICON = {
        "love": 0.8, "loved": 0.8, "loving": 0.7, "great": 0.7, "awesome": 0.85,
        "amazing": 0.85, "good": 0.5, "best": 0.8, "brilliant": 0.8, "helpful": 0.6,
        "thanks": 0.5, "thank": 0.5, "thankful": 0.7, "inspire": 0.6, "inspiring": 0.7,
        "accurate": 0.5, "happy": 0.7, "blessed": 0.8, "masterpiece": 0.9,
        "valuable": 0.6, "gem": 0.7, "fire": 0.6, "cool": 0.4, "favorite": 0.7,
        "beautiful": 0.7, "wonder": 0.5, "wonderful": 0.8, "perfect": 0.9,
        "excited": 0.7, "appreciate": 0.6, "appreciated": 0.6, "enjoy": 0.6,
        "enjoyed": 0.6, "genius": 0.8, "like": 0.4, "liked": 0.4, "liked it": 0.5
    }

    NEGATIVE_LEXICON = {
        "hate": -0.8, "hated": -0.8, "terrible": -0.85, "awful": -0.85, "worst": -0.9,
        "bad": -0.5, "fake": -0.7, "toxic": -0.8, "boring": -0.5, "disagree": -0.4,
        "depressed": -0.8, "depression": -0.8, "struggle": -0.6, "struggling": -0.6,
        "anxiety": -0.7, "anxious": -0.7, "exhausted": -0.7, "burnout": -0.7,
        "stress": -0.6, "stressed": -0.6, "crying": -0.6, "sad": -0.6, "fail": -0.6,
        "failed": -0.6, "pain": -0.7, "painful": -0.7, "hard": -0.4, "difficult": -0.5,
        "overwhelmed": -0.7, "sucks": -0.7, "disaster": -0.8, "horrible": -0.85,
        "ugly": -0.6, "useless": -0.7, "waste": -0.6, "annoying": -0.6, "annoyed": -0.6,
        "angry": -0.7, "crap": -0.6, "poor": -0.5, "attacked": -0.4
    }

    INTENSIFIERS = {
        "so": 1.4, "very": 1.4, "extremely": 1.7, "super": 1.5, "really": 1.4,
        "literally": 1.3, "absolutely": 1.6, "truly": 1.4, "completely": 1.5,
        "too": 1.3, "deeply": 1.5
    }

    NEGATIONS = {
        "not", "never", "no", "cannot", "cant", "can't", "dont", "don't",
        "didnt", "didn't", "wont", "won't", "isnt", "isn't", "hardly", "barely"
    }

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        self.preprocessor = preprocessor or TextPreprocessor()

    def analyze_tokens(self, text: str) -> Dict[str, Any]:
        """Compute valence scores analyzing negations, intensifiers, and mixed polarity."""
        words = re.findall(r"\b[\w'-]+\b", text)
        if not words:
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "pos_sum": 0.0,
                "neg_sum": 0.0,
                "sentiment": SentimentLabel.NEUTRAL.value,
                "confidence": 0.5
            }

        pos_scores: List[float] = []
        neg_scores: List[float] = []

        for i, raw_word in enumerate(words):
            word_lower = raw_word.lower()
            is_all_caps = raw_word.isupper() and len(raw_word) > 1

            # Check if current word is in positive or negative lexicon
            base_score = 0.0
            is_pos = word_lower in self.POSITIVE_LEXICON
            is_neg = word_lower in self.NEGATIVE_LEXICON

            if is_pos:
                base_score = self.POSITIVE_LEXICON[word_lower]
            elif is_neg:
                base_score = self.NEGATIVE_LEXICON[word_lower]
            else:
                continue

            # Lookback up to 3 words for intensifiers and negations
            intensifier_multiplier = 1.0
            is_negated = False

            window_start = max(0, i - 3)
            lookback = [w.lower() for w in words[window_start:i]]

            for prev_word in lookback:
                if prev_word in self.INTENSIFIERS:
                    intensifier_multiplier *= self.INTENSIFIERS[prev_word]
                if prev_word in self.NEGATIONS:
                    is_negated = True

            if is_all_caps:
                intensifier_multiplier *= 1.25

            final_valence = base_score * intensifier_multiplier

            # If negated, invert valence
            if is_negated:
                final_valence = -final_valence * 0.75  # slightly attenuated inverted score

            if final_valence > 0:
                pos_scores.append(final_valence)
            elif final_valence < 0:
                neg_scores.append(abs(final_valence))

        total_pos = sum(pos_scores)
        total_neg = sum(neg_scores)
        word_count = max(len(words), 1)

        # Subjectivity: total emotional density
        subjectivity = min(1.0, (total_pos + total_neg) / (word_count * 0.4 + 1e-6))

        # Net polarity normalized into [-1.0, 1.0]
        net_valence = total_pos - total_neg
        denom = total_pos + total_neg + 1.0
        polarity = max(-1.0, min(1.0, net_valence / denom))

        # Exclamation boosting
        excl_count = text.count("!")
        if excl_count > 0:
            boost = min(0.3, excl_count * 0.08)
            if polarity > 0.05:
                polarity = min(1.0, polarity + boost)
            elif polarity < -0.05:
                polarity = max(-1.0, polarity - boost)

        # Mixed sentiment check: both strong positive and strong negative components present
        is_mixed = (total_pos >= 0.7 and total_neg >= 0.7)

        if is_mixed:
            sentiment = SentimentLabel.MIXED.value
            confidence = round(min(0.9, 0.5 + 0.2 * (total_pos + total_neg)), 3)
        elif polarity > 0.15:
            sentiment = SentimentLabel.POSITIVE.value
            confidence = round(min(0.95, 0.55 + 0.4 * abs(polarity)), 3)
        elif polarity < -0.15:
            sentiment = SentimentLabel.NEGATIVE.value
            confidence = round(min(0.95, 0.55 + 0.4 * abs(polarity)), 3)
        else:
            sentiment = SentimentLabel.NEUTRAL.value
            confidence = round(max(0.4, 0.8 - subjectivity * 0.5), 3)

        return {
            "polarity": round(float(polarity), 3),
            "subjectivity": round(float(subjectivity), 3),
            "pos_sum": round(float(total_pos), 3),
            "neg_sum": round(float(total_neg), 3),
            "sentiment": sentiment,
            "confidence": float(confidence)
        }

    def analyze_text(self, text: Optional[str], language_hint: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a single raw comment string through preprocessor and valence engine."""
        prep: PreprocessedText = self.preprocessor.preprocess(text, language_hint=language_hint)

        if not prep.is_valid:
            return {
                "raw_text": prep.raw_text,
                "cleaned_text": prep.cleaned_text,
                "polarity": None,
                "subjectivity": None,
                "sentiment_label": None,
                "sentiment_confidence": None,
                "classification_status": prep.status.value,
                "language": prep.language,
                "rejection_reason": prep.rejection_reason
            }

        res = self.analyze_tokens(prep.cleaned_text)
        return {
            "raw_text": prep.raw_text,
            "cleaned_text": prep.cleaned_text,
            "polarity": res["polarity"],
            "subjectivity": res["subjectivity"],
            "sentiment_label": res["sentiment"],
            "sentiment_confidence": res["confidence"],
            "classification_status": ClassificationStatus.SUCCESS.value,
            "language": prep.language,
            "rejection_reason": None
        }

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "comment_text",
        lang_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Process an entire dataframe of comments while preserving all original columns."""
        df_out = df.copy()
        if text_col not in df_out.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        results = []
        for idx, row in df_out.iterrows():
            txt = row[text_col]
            lang = row[lang_col] if (lang_col and lang_col in df_out.columns and pd.notna(row[lang_col])) else None
            results.append(self.analyze_text(txt, language_hint=lang))

        res_df = pd.DataFrame(results)
        df_out["sentiment_label"] = res_df["sentiment_label"].values
        df_out["sentiment_score"] = res_df["polarity"].values
        df_out["sentiment_confidence"] = res_df["sentiment_confidence"].values
        df_out["sentiment_status"] = res_df["classification_status"].values
        return df_out
