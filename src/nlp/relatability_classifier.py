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

    def _score_probabilities(
        self,
        prep: PreprocessedText,
        p_relatable: float,
        p_neutral: float,
    ) -> Dict[str, Any]:
        """Convert finite class probabilities into canonical relatability output with uncertainty windowing."""
        if not (np.isfinite(p_relatable) and np.isfinite(p_neutral)):
            raise ValueError("Statistical model returned non-finite class probabilities.")

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

    def _classify_preprocessed_non_rule(self, prep: PreprocessedText) -> Dict[str, Any]:
        """Run Layer 2 statistical inference (or unfitted fallback) on a preprocessed non-rule comment."""
        norm_text = prep.normalized_tokens
        if self.pipeline is not None:
            try:
                classes = list(self.pipeline.named_steps["clf"].classes_)
                probs = self.pipeline.predict_proba([norm_text])[0]
                prob_dict = dict(zip(classes, probs))
                p_relatable = float(prob_dict.get("Relatable", 0.0))
                p_neutral = float(prob_dict.get("Neutral", 0.0))
                return self._score_probabilities(prep, p_relatable, p_neutral)
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

        # 3 & 4. Layer 2 Statistical Model inference or unfitted fallback
        return self._classify_preprocessed_non_rule(prep)

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
        """Full end-to-end NLP enrichment pipeline returning the canonical 14-field schema.

        Uses single-pass preprocessing and vectorized batch TF-IDF + LogisticRegression
        inference across eligible non-rule comments while preserving exact single-comment
        classification semantics and per-row error isolation.
        """
        if df.empty or text_col not in df.columns:
            return pd.DataFrame(columns=UNIFIED_COMMENT_FIELDS)

        now_iso = datetime.now(timezone.utc).isoformat()
        n_rows = len(df)
        rel_results: List[Optional[Dict[str, Any]]] = [None] * n_rows
        batch_indices: List[int] = []
        batch_preps: List[PreprocessedText] = []
        row_meta: List[Tuple[str, str, str, Optional[str], Any, Dict[str, Any]]] = []

        reuse_prep_for_sentiment = (
            type(self.sentiment_analyzer) is SentimentAnalyzer
            and self.sentiment_analyzer.preprocessor is self.preprocessor
        )

        # Pass 1: Single preprocessing pass + deterministic Layer 1 guardrails + sentiment scoring
        indices = df.index.tolist()
        cid_vals = df[comment_id_col].tolist() if comment_id_col in df.columns else [None] * n_rows
        pid_vals = df[post_id_col].tolist() if post_id_col in df.columns else [None] * n_rows
        plat_vals = df[platform_col].tolist() if platform_col in df.columns else [None] * n_rows
        ts_vals = df[timestamp_col].tolist() if timestamp_col in df.columns else [None] * n_rows
        lang_vals = df[language_col].tolist() if (language_col and language_col in df.columns) else [None] * n_rows
        text_vals = df[text_col].tolist()

        for pos, (idx, raw_cid, raw_pid, raw_plat, raw_ts, raw_lang, raw_text) in enumerate(
            zip(indices, cid_vals, pid_vals, plat_vals, ts_vals, lang_vals, text_vals)
        ):
            cid = str(raw_cid) if pd.notna(raw_cid) else f"CMT_{idx:05d}"
            pid = str(raw_pid) if pd.notna(raw_pid) else "UNKNOWN_POST"
            plat = str(raw_plat) if pd.notna(raw_plat) else "instagram"
            ts = str(raw_ts) if pd.notna(raw_ts) else None
            lang_hint = str(raw_lang) if pd.notna(raw_lang) else None

            prep: PreprocessedText = self.preprocessor.preprocess(raw_text, language_hint=lang_hint)

            # Orthogonal Sentiment Analysis (reuses preprocessed text when using standard SentimentAnalyzer)
            if reuse_prep_for_sentiment:
                if not prep.is_valid:
                    sent_res: Dict[str, Any] = {
                        "polarity": None,
                        "sentiment_label": None,
                        "sentiment_confidence": None,
                    }
                else:
                    tok_res = self.sentiment_analyzer.analyze_tokens(prep.cleaned_text)
                    sent_res = {
                        "polarity": tok_res["polarity"],
                        "sentiment_label": tok_res["sentiment"],
                        "sentiment_confidence": tok_res["confidence"],
                    }
            else:
                sent_res = self.sentiment_analyzer.analyze_text(raw_text, language_hint=lang_hint)

            row_meta.append((cid, pid, plat, ts, raw_text, sent_res))

            # Relatability Layer 1 validation & heuristic guardrails
            if not prep.is_valid:
                rel_results[pos] = {
                    "raw_text": prep.raw_text,
                    "cleaned_text": prep.cleaned_text,
                    "language": prep.language,
                    "relatability_label": None,
                    "relatability_confidence": None,
                    "classification_status": prep.status.value,
                    "rejection_reason": prep.rejection_reason,
                }
                continue

            is_anchor = self._check_experiential_anchor(prep.cleaned_text)
            is_tech_inquiry = self._check_technical_or_generic_inquiry(prep.cleaned_text)

            if is_tech_inquiry and not is_anchor:
                rel_results[pos] = {
                    "raw_text": prep.raw_text,
                    "cleaned_text": prep.cleaned_text,
                    "language": prep.language,
                    "relatability_label": RelatabilityLabel.NEUTRAL.value,
                    "relatability_confidence": 0.92,
                    "classification_status": ClassificationStatus.SUCCESS.value,
                    "rejection_reason": None,
                }
            elif is_anchor:
                rel_results[pos] = {
                    "raw_text": prep.raw_text,
                    "cleaned_text": prep.cleaned_text,
                    "language": prep.language,
                    "relatability_label": RelatabilityLabel.RELATABLE.value,
                    "relatability_confidence": 0.95,
                    "classification_status": ClassificationStatus.SUCCESS.value,
                    "rejection_reason": None,
                }
            elif self.pipeline is None:
                rel_results[pos] = {
                    "raw_text": prep.raw_text,
                    "cleaned_text": prep.cleaned_text,
                    "language": prep.language,
                    "relatability_label": RelatabilityLabel.NEUTRAL.value,
                    "relatability_confidence": 0.60,
                    "classification_status": ClassificationStatus.SUCCESS.value,
                    "rejection_reason": None,
                }
            else:
                batch_indices.append(pos)
                batch_preps.append(prep)

        # Pass 2: Vectorized batch statistical inference with per-row fallback on failure
        if batch_preps and self.pipeline is not None:
            try:
                classes = list(self.pipeline.named_steps["clf"].classes_)
                rel_idx = classes.index("Relatable") if "Relatable" in classes else -1
                neu_idx = classes.index("Neutral") if "Neutral" in classes else -1
                norm_texts = [p.normalized_tokens for p in batch_preps]
                probs_matrix = np.asarray(self.pipeline.predict_proba(norm_texts), dtype=float)
                if probs_matrix.ndim != 2 or probs_matrix.shape[0] != len(batch_preps) or not np.all(np.isfinite(probs_matrix)):
                    raise ValueError("Batch statistical model returned malformed or non-finite probability matrix.")

                for pos, prep, probs_row in zip(batch_indices, batch_preps, probs_matrix):
                    p_relatable = float(probs_row[rel_idx]) if rel_idx >= 0 else 0.0
                    p_neutral = float(probs_row[neu_idx]) if neu_idx >= 0 else 0.0
                    rel_results[pos] = self._score_probabilities(prep, p_relatable, p_neutral)
            except Exception:
                # Isolate failures per comment so one malformed input never fails the batch
                for pos, prep in zip(batch_indices, batch_preps):
                    rel_results[pos] = self._classify_preprocessed_non_rule(prep)

        # Pass 3: Assemble canonical 14-field records
        records: List[Dict[str, Any]] = []
        for pos, (cid, pid, plat, ts, raw_text, sent_res) in enumerate(row_meta):
            rel_res = rel_results[pos]
            assert rel_res is not None
            records.append({
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
                "analysis_timestamp": now_iso,
            })

        return pd.DataFrame(records)

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
