"""Relatability Classifier: Classifies audience comments as 'Relatable' vs 'Neutral'.

Validates audience problem awareness and emotional identification using a multi-tier
hybrid NLP pipeline:
1. Text validation and non-English / insufficient context rejection.
2. Heuristic domain guardrails (excluding generic gear/praise inquiries, anchoring authentic vulnerability).
3. Statistical TF-IDF + Regularized Logistic Regression with class weighting.
4. Uncertainty windowing: comments with ambiguous confidence (0.40 - 0.60) are tagged 'Uncertain'.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_ingestion.schemas import (
    ClassificationStatus,
    RelatabilityLabel,
    SentimentLabel,
    UNIFIED_COMMENT_FIELDS
)
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.nlp.text_preprocessor import TextPreprocessor, PreprocessedText

logger = logging.getLogger(__name__)


class RelatabilityClassifier:
    """Multi-tier classifier distinguishing authentic audience relatability from neutral/generic comments."""

    MODEL_VERSION = "nlp-relatability-v1.0.0"

    # Explicit experiential triggers demonstrating personal identification
    EXPERIENTIAL_TRIGGERS = [
        "literally me", "so true", "felt this", "called out", "same here",
        "story of my life", "relatable", "i feel this", "me every single morning",
        "me every day", "this is me", "going through this", "i needed to hear this",
        "hits home", "accurate", "too accurate", "why is this me", "why did you have to call me out",
        "attacking me", "stop describing me", "never felt so understood",
        "thought i was the only one", "thought i was alone", "happens to me every",
        "i do this every time", "my exact thoughts", "living this right now",
        "are you inside my head", "this called me out", "are you watching me",
        "felt that in my soul", "felt this in my soul", "realest post", "hits too close to home",
        "this is literally my life", "i struggle with this"
    ]

    # Explicit technical/informational/external inquiries that indicate non-relatable content reactions
    TECHNICAL_INQUIRY_PATTERNS = [
        r"\b(what camera|which camera|camera model|what lens|which lens)\b",
        r"\b(what software|editing software|which app|what app|which program)\b",
        r"\b(what song|song name|background music|track name|what track|what audio)\b",
        r"\b(what font|which font|color grading|lut|preset|specs)\b",
        r"\b(where did you buy|where is this from|link to purchase|link please)\b",
        r"\b(checked your page|nice aesthetics|good video quality|nice post|interesting perspective)\b"
    ]

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None
    ):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer(self.preprocessor)
        self.pipeline: Optional[Pipeline] = None
        self.tech_inquiry_regex = re.compile("|".join(self.TECHNICAL_INQUIRY_PATTERNS), re.IGNORECASE)

        if model_path:
            self.load(model_path)

    def _check_experiential_anchor(self, text: str) -> bool:
        """Check if comment contains strong first-person lived experience triggers."""
        clean = text.lower()
        for trig in self.EXPERIENTIAL_TRIGGERS:
            if trig in clean:
                return True
        # First-person vulnerability + struggle patterns (e.g. "my anxiety...", "i always overthink...")
        if re.search(r"\b(my anxiety|i always overthink|i struggle with|i hate when i do this|i thought i was weird)\b", clean):
            return True
        return False

    def _check_technical_or_generic_inquiry(self, text: str) -> bool:
        """Check if comment is a generic equipment/technical question or surface aesthetic comment."""
        return bool(self.tech_inquiry_regex.search(text))

    def fit(
        self,
        texts: List[str],
        labels: List[str],
        class_weight: str = "balanced"
    ) -> "RelatabilityClassifier":
        """Train a statistical NLP classifier using word and character n-grams."""
        # Clean training texts with preprocessor
        cleaned_texts = [self.preprocessor.preprocess(t).normalized_tokens for t in texts]

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=2500,
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight=class_weight,
                max_iter=1000,
                random_state=42
            ))
        ])
        self.pipeline.fit(cleaned_texts, labels)
        return self

    def classify_comment(
        self,
        text: Optional[str],
        language_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Classify a single comment into Relatable / Neutral / Uncertain with full diagnostic fields."""
        prep: PreprocessedText = self.preprocessor.preprocess(text, language_hint=language_hint)

        # 1. Handle invalid/unsupported text
        if not prep.is_valid:
            return {
                "raw_text": prep.raw_text,
                "cleaned_text": prep.cleaned_text,
                "language": prep.language,
                "relatability_label": None,
                "relatability_confidence": None,
                "classification_status": prep.status.value,
                "rejection_reason": prep.rejection_reason
            }

        norm_text = prep.normalized_tokens

        # 2. Layer 1 Guardrails: Check for strong heuristic anchors
        is_anchor = self._check_experiential_anchor(prep.cleaned_text)
        is_tech_inquiry = self._check_technical_or_generic_inquiry(prep.cleaned_text)

        # Technical/generic inquiry filter takes precedence over partial trigger matches
        if is_tech_inquiry and not is_anchor:
            return {
                "raw_text": prep.raw_text,
                "cleaned_text": prep.cleaned_text,
                "language": prep.language,
                "relatability_label": RelatabilityLabel.NEUTRAL.value,
                "relatability_confidence": 0.92,
                "classification_status": ClassificationStatus.SUCCESS.value,
                "rejection_reason": None
            }

        if is_anchor:
            return {
                "raw_text": prep.raw_text,
                "cleaned_text": prep.cleaned_text,
                "language": prep.language,
                "relatability_label": RelatabilityLabel.RELATABLE.value,
                "relatability_confidence": 0.95,
                "classification_status": ClassificationStatus.SUCCESS.value,
                "rejection_reason": None
            }

        # 3. Layer 2: Statistical Model inference
        if self.pipeline is not None:
            try:
                classes = list(self.pipeline.named_steps["clf"].classes_)
                probs = self.pipeline.predict_proba([norm_text])[0]
                prob_dict = dict(zip(classes, probs))
                p_relatable = float(prob_dict.get("Relatable", 0.0))
                p_neutral = float(prob_dict.get("Neutral", 0.0))
                if not (np.isfinite(p_relatable) and np.isfinite(p_neutral)):
                    raise ValueError("Statistical model returned non-finite class probabilities.")

                # Layer 3: Uncertainty Window (0.42 to 0.58)
                if 0.42 <= p_relatable <= 0.58:
                    return {
                        "raw_text": prep.raw_text,
                        "cleaned_text": prep.cleaned_text,
                        "language": prep.language,
                        "relatability_label": RelatabilityLabel.UNCERTAIN.value,
                        "relatability_confidence": round(float(max(p_relatable, p_neutral)), 3),
                        "classification_status": ClassificationStatus.UNCERTAIN.value,
                        "rejection_reason": "Statistical prediction probability falls within the ambiguous uncertainty boundary [0.42, 0.58]."
                    }

                if p_relatable > 0.58:
                    label = RelatabilityLabel.RELATABLE.value
                    conf = p_relatable
                else:
                    label = RelatabilityLabel.NEUTRAL.value
                    conf = p_neutral

                return {
                    "raw_text": prep.raw_text,
                    "cleaned_text": prep.cleaned_text,
                    "language": prep.language,
                    "relatability_label": label,
                    "relatability_confidence": round(float(conf), 3),
                    "classification_status": ClassificationStatus.SUCCESS.value,
                    "rejection_reason": None
                }
            except Exception as exc:
                raw_msg = str(exc)
                contains_input = any(
                    s and len(s.strip()) >= 3 and s in raw_msg
                    for s in (prep.raw_text, prep.cleaned_text, norm_text)
                )
                safe_detail = "[redacted]" if contains_input else raw_msg
                logger.error(
                    "Statistical relatability inference failed (%s: %s) [language=%s, token_len=%d]",
                    type(exc).__name__,
                    safe_detail,
                    prep.language,
                    len(norm_text),
                    exc_info=not contains_input,
                )
                return {
                    "raw_text": prep.raw_text,
                    "cleaned_text": prep.cleaned_text,
                    "language": prep.language,
                    "relatability_label": None,
                    "relatability_confidence": None,
                    "classification_status": ClassificationStatus.INFERENCE_FAILED.value,
                    "rejection_reason": f"Statistical classifier inference failed ({type(exc).__name__})."
                }

        # 4. Fallback when model is not fitted
        return {
            "raw_text": prep.raw_text,
            "cleaned_text": prep.cleaned_text,
            "language": prep.language,
            "relatability_label": RelatabilityLabel.NEUTRAL.value,
            "relatability_confidence": 0.60,
            "classification_status": ClassificationStatus.SUCCESS.value,
            "rejection_reason": None
        }

    def predict(self, texts: Union[str, List[str]]) -> List[Optional[str]]:
        """Simple list prediction interface for backwards compatibility (returns None when unavailable)."""
        if isinstance(texts, str):
            texts = [texts]
        return [self.classify_comment(t)["relatability_label"] for t in texts]

    def process_comments_pipeline(
        self,
        df: pd.DataFrame,
        text_col: str = "comment_text",
        comment_id_col: str = "comment_id",
        post_id_col: str = "post_id",
        platform_col: str = "platform",
        timestamp_col: str = "comment_timestamp",
        language_col: Optional[str] = "language"
    ) -> pd.DataFrame:
        """Full end-to-end NLP enrichment pipeline returning the canonical 13-field schema."""
        if df.empty or text_col not in df.columns:
            return pd.DataFrame(columns=UNIFIED_COMMENT_FIELDS)

        now_iso = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            cid = str(row[comment_id_col]) if comment_id_col in row and pd.notna(row[comment_id_col]) else f"CMT_{idx:05d}"
            pid = str(row[post_id_col]) if post_id_col in row and pd.notna(row[post_id_col]) else "UNKNOWN_POST"
            plat = str(row[platform_col]) if platform_col in row and pd.notna(row[platform_col]) else "instagram"
            ts = str(row[timestamp_col]) if (timestamp_col in row and pd.notna(row[timestamp_col])) else None
            lang_hint = str(row[language_col]) if (language_col and language_col in row and pd.notna(row[language_col])) else None
            raw_text = row[text_col]

            # 1. Relatability Classification
            rel_res = self.classify_comment(raw_text, language_hint=lang_hint)

            # 2. Sentiment Analysis (separate orthogonal dimension)
            sent_res = self.sentiment_analyzer.analyze_text(raw_text, language_hint=lang_hint)

            # Unified record adhering to strict contract
            rec = {
                "comment_id": cid,
                "post_id": pid,
                "platform": plat,
                "comment_text": str(raw_text) if pd.notna(raw_text) else "",
                "comment_timestamp": ts,
                "language": rel_res["language"],
                "sentiment_label": sent_res["sentiment_label"],
                "sentiment_confidence": sent_res["sentiment_confidence"],
                "sentiment_score": sent_res.get("polarity"),
                "relatability_label": rel_res["relatability_label"],
                "relatability_confidence": rel_res["relatability_confidence"],
                "classification_status": rel_res["classification_status"],
                "model_version": self.MODEL_VERSION,
                "analysis_timestamp": now_iso
            }
            records.append(rec)

        res_df = pd.DataFrame(records)
        return res_df

    def tag_dataframe(self, df: pd.DataFrame, text_col: str = "comment_text") -> pd.DataFrame:
        """Backwards compatible tagging returning DataFrame with 'relatability_tag'."""
        df_out = df.copy()
        preds = self.predict(df_out[text_col].tolist())
        df_out["relatability_tag"] = preds
        return df_out

    def compute_post_relatability_score(
        self,
        df_tagged: pd.DataFrame,
        post_id_col: str = "post_id",
        tag_col: str = "relatability_label",
        status_col: str = "classification_status"
    ) -> pd.DataFrame:
        """Aggregate post-level relatability percentage with transparent denominator handling.

        Denominator: Total successfully classified comments (excluding empty, unsupported, or uncertain).
        """
        # Backwards compatible tag column check
        active_tag_col = tag_col if tag_col in df_tagged.columns else "relatability_tag"
        if post_id_col not in df_tagged.columns or active_tag_col not in df_tagged.columns:
            raise ValueError(f"DataFrame must contain '{post_id_col}' and '{active_tag_col}' columns.")

        records = []
        for pid, group in df_tagged.groupby(post_id_col):
            total_raw = len(group)
            
            # Count valid evaluatable comments
            if status_col in group.columns:
                valid_group = group[group[status_col] == ClassificationStatus.SUCCESS.value]
            else:
                valid_group = group

            n_valid = len(valid_group)
            n_relatable = (valid_group[active_tag_col] == RelatabilityLabel.RELATABLE.value).sum()
            n_neutral = (valid_group[active_tag_col] == RelatabilityLabel.NEUTRAL.value).sum()
            n_uncertain = (group[active_tag_col] == RelatabilityLabel.UNCERTAIN.value).sum() if active_tag_col in group else 0

            ratio = n_relatable / n_valid if n_valid > 0 else 0.0

            records.append({
                "post_id": pid,
                "total_comments_raw": total_raw,
                "total_comments_evaluated": n_valid,
                "relatable_comments": int(n_relatable),
                "neutral_comments": int(n_neutral),
                "uncertain_comments": int(n_uncertain),
                "relatability_ratio": round(float(ratio), 4),
                "relatability_pct": round(float(ratio) * 100.0, 2)
            })

        return pd.DataFrame(records)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize trained pipeline artifact."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "version": self.MODEL_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat()
        }, path)

    def load(self, filepath: Union[str, Path]) -> "RelatabilityClassifier":
        """Load trained pipeline artifact."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        self.pipeline = data.get("pipeline")
        return self
