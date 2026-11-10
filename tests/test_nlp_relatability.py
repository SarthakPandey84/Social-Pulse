"""Tests for NLP Sentiment & Relatability Classifier."""

import logging
import numpy as np
import pandas as pd

from src.data_ingestion.schemas import ClassificationStatus, RelatabilityLabel
from src.nlp.relatability_classifier import RelatabilityClassifier


def test_relatability_rule_trigger():
    """Unfitted heuristic mode must preserve rule matches and intentional Neutral fallback."""
    classifier = RelatabilityClassifier()
    assert classifier.pipeline is None
    assert classifier.predict("Literally me every morning")[0] == "Relatable"
    assert classifier.predict("Felt this in my soul")[0] == "Relatable"
    neutral_diag = classifier.classify_comment("The video quality is 1080p")
    assert neutral_diag["relatability_label"] == "Neutral"
    assert neutral_diag["relatability_confidence"] == 0.60
    assert neutral_diag["classification_status"] == ClassificationStatus.SUCCESS.value


def test_train_and_predict():
    classifier = RelatabilityClassifier()
    texts = [
        "so true literally me",
        "i feel this so much",
        "random post with no relation",
        "what is the camera model"
    ]
    labels = ["Relatable", "Relatable", "Neutral", "Neutral"]
    classifier.fit(texts, labels)
    preds = classifier.predict(["so true this happened to me", "what is the camera"])
    assert preds[0] == "Relatable"
    assert preds[1] == "Neutral"


def test_fitted_neutral_prediction_preserved():
    """Legitimate fitted Neutral statistical predictions (non-rule) must remain intact."""
    classifier = RelatabilityClassifier()
    texts = [
        "painfully true deep personal reflection",
        "personal burnout confession daily struggle",
        "overthinking sleepless anxiety nights",
        "emotional vulnerability confession story",
    ] * 3 + [
        "thumbnail typography comparison campaign schedule",
        "monthly spreadsheet export analytics overview",
        "weekly publishing calendar configuration notes",
        "render bitrate export documentation summary",
    ] * 3
    labels = ["Relatable"] * 12 + ["Neutral"] * 12
    classifier.fit(texts, labels)

    sample = "monthly spreadsheet export schedule overview"
    assert not classifier._check_technical_or_generic_inquiry(sample)
    assert not classifier._check_experiential_anchor(sample)
    res = classifier.classify_comment(sample)
    assert res["classification_status"] == ClassificationStatus.SUCCESS.value
    assert res["relatability_label"] == RelatabilityLabel.NEUTRAL.value
    assert res["relatability_confidence"] is not None and res["relatability_confidence"] > 0.58
    assert res["rejection_reason"] is None


def test_inference_failure_returns_explicit_error_and_logs_without_input_leak(caplog):
    """Genuine statistical inference errors must never silently become Neutral or leak user text."""
    classifier = RelatabilityClassifier()
    classifier.fit(
        ["literally me every morning", "felt this in my soul", "camera setup overview", "editing timeline tips"],
        ["Relatable", "Relatable", "Neutral", "Neutral"],
    )

    secret_comment = "Private sensitive struggle confession token XYZ_987"

    class BrokenPipeline:
        named_steps = {"clf": type("FakeClf", (), {"classes_": ["Neutral", "Relatable"]})()}

        def predict_proba(self, _items):
            raise RuntimeError(f"Simulated backend failure processing {secret_comment}")

    classifier.pipeline = BrokenPipeline()

    with caplog.at_level(logging.ERROR, logger="src.nlp.relatability_classifier"):
        diag = classifier.classify_comment(secret_comment)

    assert diag["classification_status"] == ClassificationStatus.INFERENCE_FAILED.value
    assert diag["relatability_label"] is None
    assert diag["relatability_confidence"] is None
    assert diag["rejection_reason"] is not None and "RuntimeError" in diag["rejection_reason"]
    assert secret_comment not in diag["rejection_reason"]

    # predict() must also propagate None rather than coercing failure to "Neutral"
    assert classifier.predict([secret_comment]) == [None]

    # Diagnostic error log must record failure without echoing user text
    assert any("RuntimeError" in rec.message for rec in caplog.records)
    assert all(secret_comment not in rec.getMessage() for rec in caplog.records)


def test_non_finite_probabilities_treated_as_inference_failure():
    """NaN probability outputs from a corrupted estimator must trigger INFERENCE_FAILED."""
    classifier = RelatabilityClassifier()
    classifier.fit(
        ["literally me every morning", "camera setup overview"],
        ["Relatable", "Neutral"],
    )

    class NanProbPipeline:
        named_steps = {"clf": type("FakeClf", (), {"classes_": ["Neutral", "Relatable"]})()}

        def predict_proba(self, _items):
            return np.array([[np.nan, 0.5]])

    classifier.pipeline = NanProbPipeline()
    diag = classifier.classify_comment("Some ordinary english sentence about editing workflows")
    assert diag["classification_status"] == ClassificationStatus.INFERENCE_FAILED.value
    assert diag["relatability_label"] is None
    assert diag["relatability_confidence"] is None


def test_pipeline_and_post_aggregation_exclude_inference_failures():
    """Batch pipeline and post-level aggregation must exclude failed rows from evaluated totals."""
    classifier = RelatabilityClassifier()
    classifier.fit(
        ["literally me every morning", "camera setup overview"],
        ["Relatable", "Neutral"],
    )

    class BrokenPipeline:
        named_steps = {"clf": type("FakeClf", (), {"classes_": ["Neutral", "Relatable"]})()}

        def predict_proba(self, _items):
            raise ValueError("Vectorizer mismatch")

    classifier.pipeline = BrokenPipeline()

    # Row 1 triggers Layer 1 experiential rule -> Relatable/SUCCESS
    # Row 2 triggers Layer 1 technical rule -> Neutral/SUCCESS
    # Row 3 reaches Layer 2 statistical pipeline -> raises -> INFERENCE_FAILED
    df_in = pd.DataFrame([
        {"comment_id": "C1", "post_id": "P1", "comment_text": "Literally me every morning.", "platform": "instagram"},
        {"comment_id": "C2", "post_id": "P1", "comment_text": "What camera did you use?", "platform": "instagram"},
        {"comment_id": "C3", "post_id": "P1", "comment_text": "Interesting breakdown of workflow habits.", "platform": "instagram"},
    ])
    df_proc = classifier.process_comments_pipeline(df_in)
    row3 = df_proc[df_proc["comment_id"] == "C3"].iloc[0]
    assert row3["classification_status"] == ClassificationStatus.INFERENCE_FAILED.value
    assert pd.isna(row3["relatability_label"])
    assert pd.isna(row3["relatability_confidence"])

    stats = classifier.compute_post_relatability_score(df_proc).iloc[0]
    assert stats["total_comments_raw"] == 3
    assert stats["total_comments_evaluated"] == 2
    assert stats["relatable_comments"] == 1
    assert stats["neutral_comments"] == 1

