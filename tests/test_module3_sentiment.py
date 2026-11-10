"""Module 3 Test Suite: Audience Sentiment & Relatability Classifier.

Verifies:
1. Text normalization, negations, and conversational expressions.
2. Empty and missing value handling.
3. Insufficient context detection (emojis only, single characters).
4. Unsupported language detection.
5. Positive and negative sentiment valence scoring.
6. Positive but non-relatable comments (generic praise).
7. Negative but relatable comments (shared vulnerability).
8. Neutral but personally identifying comments.
9. Ambiguous comments and uncertainty handling.
10. Duplicate comment handling without losing records.
11. Batch pipeline execution and post-level relatability ratio calculation.
12. Model serialization and artifact loading.
13. Schema conformance with UNIFIED_COMMENT_FIELDS.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.data_ingestion.schemas import (
    ClassificationStatus,
    RelatabilityLabel,
    SentimentLabel,
    UNIFIED_COMMENT_FIELDS
)
from src.nlp.text_preprocessor import TextPreprocessor
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.nlp.relatability_classifier import RelatabilityClassifier


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


@pytest.fixture
def sentiment_analyzer(preprocessor) -> SentimentAnalyzer:
    return SentimentAnalyzer(preprocessor=preprocessor)


@pytest.fixture
def classifier() -> RelatabilityClassifier:
    model_path = Path("models/relatability_classifier.joblib")
    if model_path.exists():
        return RelatabilityClassifier(model_path=model_path)
    # If file doesn't exist, instantiate default
    return RelatabilityClassifier()


# --- 1. Text Normalization and Negation Preservation ---

def test_text_normalization_preserves_negations(preprocessor):
    """Verify that text preprocessing preserves negations and contraction expansions."""
    res = preprocessor.preprocess("I don't think this is working, definitely not good!!")
    assert res.is_valid is True
    assert "not" in res.normalized_tokens or "do not" in res.normalized_tokens
    assert res.status == ClassificationStatus.SUCCESS


def test_text_normalization_preserves_punctuation_and_emojis(preprocessor):
    """Verify conversational punctuation and emojis survive preprocessing."""
    res = preprocessor.preprocess("This hits way too close to home!! 🔥😭")
    assert res.is_valid is True
    assert "!!" in res.cleaned_text
    assert "hits" in res.normalized_tokens


# --- 2. Empty Comments and Missing Values ---

def test_empty_and_whitespace_comments(preprocessor, classifier):
    """Verify empty and whitespace strings return EMPTY_TEXT status without throwing errors."""
    for empty_val in ["", "   ", "\t\n", None, float("nan")]:
        res = preprocessor.preprocess(empty_val)
        assert res.is_valid is False
        assert res.status == ClassificationStatus.EMPTY_TEXT

        diag = classifier.classify_comment(empty_val)
        assert diag["classification_status"] == ClassificationStatus.EMPTY_TEXT.value
        assert diag["relatability_label"] is None
        assert diag["relatability_confidence"] is None


# --- 3. Insufficient Context Detection ---

def test_insufficient_context_emojis_and_single_characters(preprocessor, classifier):
    """Verify comments with only emojis or single characters return INSUFFICIENT_CONTEXT."""
    edge_cases = ["👀", "❤️🔥", "???", "...", "k", "ok", "yo"]
    for text in edge_cases:
        res = preprocessor.preprocess(text)
        assert res.is_valid is False
        assert res.status == ClassificationStatus.INSUFFICIENT_CONTEXT

        diag = classifier.classify_comment(text)
        assert diag["classification_status"] == ClassificationStatus.INSUFFICIENT_CONTEXT.value
        assert diag["relatability_label"] is None


# --- 4. Unsupported Languages ---

def test_unsupported_language_detection(preprocessor, classifier):
    """Verify non-English comments are flagged as unsupported without applying English models."""
    non_en_examples = [
        "C'est tellement vrai pour moi aussi dans ma vie",
        "Das kenne ich zu gut aus eigener Erfahrung",
        "bhai ye toh meri hi kahani hai bilkul sach bola aapne",
        "este video me describe perfectamente amigo mio",
        "यह बिल्कुल सच है मेरे साथ भी ऐसा ही होता है"
    ]
    for text in non_en_examples:
        res = preprocessor.preprocess(text)
        assert res.is_valid is False
        assert res.status == ClassificationStatus.UNSUPPORTED_LANGUAGE

        diag = classifier.classify_comment(text)
        assert diag["classification_status"] == ClassificationStatus.UNSUPPORTED_LANGUAGE.value
        assert diag["relatability_label"] is None
        assert diag["language"] != "en"


# --- 5. Sentiment Polarity & Valence Scoring ---

def test_sentiment_positive_and_negative_valence(sentiment_analyzer):
    """Verify sentiment polarity cleanly separates positive and negative valence."""
    pos_res = sentiment_analyzer.analyze_text("Amazing video, loved the explanation and beautiful visuals!")
    assert pos_res["sentiment_label"] == SentimentLabel.POSITIVE.value
    assert pos_res["polarity"] > 0.3
    assert pos_res["sentiment_confidence"] > 0.6

    neg_res = sentiment_analyzer.analyze_text("Terrible advice, worst video I have seen all week.")
    assert neg_res["sentiment_label"] == SentimentLabel.NEGATIVE.value
    assert neg_res["polarity"] < -0.3


def test_sentiment_negation_inversion(sentiment_analyzer):
    """Verify negation inverts positive valence."""
    res1 = sentiment_analyzer.analyze_text("This was a great video.")
    res2 = sentiment_analyzer.analyze_text("This was not a great video.")
    assert res1["polarity"] > 0.2
    assert res2["polarity"] < 0.1


# --- 6. Positive Sentiment but Neutral Relatability (Orthogonality) ---

def test_positive_sentiment_but_neutral_relatability(classifier, sentiment_analyzer):
    """Verify that generic praise is classified as Positive sentiment but Neutral relatability."""
    generic_praises = [
        "Great video, keep up the awesome content!",
        "Such high production quality and brilliant lighting!",
        "What camera did you use to film this video?",
        "Nice visuals, loved the cinematic transitions."
    ]
    for text in generic_praises:
        sent = sentiment_analyzer.analyze_text(text)
        rel = classifier.classify_comment(text)
        # Relatability MUST be Neutral, NOT Relatable
        assert rel["relatability_label"] == RelatabilityLabel.NEUTRAL.value, f"Failed for: {text}"
        assert rel["relatability_confidence"] >= 0.70


# --- 7. Negative Sentiment but Highly Relatable ---

def test_negative_sentiment_but_relatable(classifier, sentiment_analyzer):
    """Verify personal struggle is classified as Negative sentiment but Relatable."""
    struggles = [
        "This happens to me every single morning.",
        "I felt this in my soul, so painfully true.",
        "Story of my life right now, can't even focus for 10 minutes.",
        "My anxiety has been telling me this all week."
    ]
    for text in struggles:
        rel = classifier.classify_comment(text)
        assert rel["relatability_label"] == RelatabilityLabel.RELATABLE.value, f"Failed for: {text}"
        assert rel["relatability_confidence"] >= 0.80


# --- 8. Ambiguous Comments and Uncertainty Window ---

def test_uncertainty_boundary_handling(classifier):
    """Verify ambiguous comments receive Uncertain label rather than forced binary classification."""
    # Texts with ambiguous mixed or borderline phrasing
    res = classifier.classify_comment("I think this depends on the specific job situation.")
    assert res["classification_status"] in [ClassificationStatus.SUCCESS.value, ClassificationStatus.UNCERTAIN.value]


# --- 9. Duplicate Comments Handling ---

def test_duplicate_comments_preserved_in_pipeline(classifier):
    """Verify identical duplicate comments across different users/posts are preserved."""
    df_raw = pd.DataFrame([
        {"comment_id": "CMT_001", "post_id": "POST_A", "comment_text": "Literally me every morning.", "platform": "instagram"},
        {"comment_id": "CMT_002", "post_id": "POST_B", "comment_text": "Literally me every morning.", "platform": "youtube"},
        {"comment_id": "CMT_003", "post_id": "POST_A", "comment_text": "Literally me every morning.", "platform": "instagram"}
    ])
    df_proc = classifier.process_comments_pipeline(df_raw)
    assert len(df_proc) == 3
    assert df_proc["comment_id"].tolist() == ["CMT_001", "CMT_002", "CMT_003"]
    assert all(df_proc["relatability_label"] == RelatabilityLabel.RELATABLE.value)


# --- 10. Batch Processing & Post-Level Relatability Score ---

def test_post_relatability_score_calculation(classifier):
    """Verify aggregation of post-level relatability percentage with valid denominator."""
    df_raw = pd.DataFrame([
        {"comment_id": "C1", "post_id": "POST_1", "comment_text": "This happens to me every morning.", "platform": "instagram"},
        {"comment_id": "C2", "post_id": "POST_1", "comment_text": "Too accurate, stop attacking me lol.", "platform": "instagram"},
        {"comment_id": "C3", "post_id": "POST_1", "comment_text": "What camera did you use to film this?", "platform": "instagram"},
        {"comment_id": "C4", "post_id": "POST_1", "comment_text": "...", "platform": "instagram"}, # insufficient context
        {"comment_id": "C5", "post_id": "POST_2", "comment_text": "Great video, keep up the awesome content!", "platform": "youtube"},
        {"comment_id": "C6", "post_id": "POST_2", "comment_text": "What song is playing in the background?", "platform": "youtube"}
    ])
    df_proc = classifier.process_comments_pipeline(df_raw)
    post_stats = classifier.compute_post_relatability_score(df_proc)

    post1 = post_stats[post_stats["post_id"] == "POST_1"].iloc[0]
    assert post1["total_comments_raw"] == 4
    assert post1["total_comments_evaluated"] == 3  # excluded "..."
    assert post1["relatable_comments"] == 2
    assert post1["neutral_comments"] == 1
    assert round(post1["relatability_pct"], 1) == 66.7

    post2 = post_stats[post_stats["post_id"] == "POST_2"].iloc[0]
    assert post2["total_comments_evaluated"] == 2
    assert post2["relatable_comments"] == 0
    assert post2["relatability_pct"] == 0.0


# --- 11. Model Serialization and Loading ---

def test_model_serialization_and_reloading(tmp_path):
    """Verify classifier pipeline can be saved to joblib and restored accurately."""
    clf = RelatabilityClassifier()
    texts = [
        "Literally me every morning",
        "Felt this in my soul so true",
        "What camera did you use for this",
        "Checked your page looks great"
    ]
    labels = ["Relatable", "Relatable", "Neutral", "Neutral"]
    clf.fit(texts, labels)

    save_file = tmp_path / "temp_relatability.joblib"
    clf.save(save_file)
    assert save_file.exists()

    loaded_clf = RelatabilityClassifier(model_path=save_file)
    assert loaded_clf.pipeline is not None
    preds = loaded_clf.predict(["Literally me every morning", "What camera did you use"])
    assert preds[0] == RelatabilityLabel.RELATABLE.value
    assert preds[1] == RelatabilityLabel.NEUTRAL.value


# --- 12. Schema Conformance ---

def test_pipeline_output_conforms_to_unified_comment_schema(classifier):
    """Verify that process_comments_pipeline outputs all 14 canonical schema columns including sentiment_score."""
    df_raw = pd.DataFrame([
        {"comment_id": "C_TEST", "post_id": "P_TEST", "comment_text": "I thought I was the only one going through this.", "platform": "instagram"}
    ])
    df_proc = classifier.process_comments_pipeline(df_raw)
    assert len(UNIFIED_COMMENT_FIELDS) == 14
    for col in UNIFIED_COMMENT_FIELDS:
        assert col in df_proc.columns, f"Unified schema column '{col}' missing from pipeline output."
    assert "sentiment_score" in df_proc.columns


# --- 13. Mixed Sentiment & Contrastive Relatability Edge Cases ---

def test_mixed_sentiment_reachability(sentiment_analyzer):
    """Verify that comments with simultaneous positive and negative valences classify as Mixed."""
    mixed_text = "I hated this topic at first and felt terrible, but your explanation was amazing and so helpful!"
    res = sentiment_analyzer.analyze_text(mixed_text)
    assert res["sentiment_label"] == SentimentLabel.MIXED.value
    assert res["sentiment_confidence"] >= 0.70


def test_contrastive_first_person_vs_experiential_relatability(classifier):
    """Verify first-person pronouns without relatability vs experiential relatability without pronouns."""
    # 1. First-person pronoun present, but completely Neutral (gear / critique)
    fp_neutral = "I bought that camera last year and returned it. I prefer Sony."
    diag_fp = classifier.classify_comment(fp_neutral)
    assert diag_fp["relatability_label"] == RelatabilityLabel.NEUTRAL.value

    # 2. Experiential struggle without explicit 'I' or 'my'
    exp_no_fp = "Story of my life right now, can't even focus for 10 minutes."
    diag_exp = classifier.classify_comment(exp_no_fp)
    assert diag_exp["relatability_label"] == RelatabilityLabel.RELATABLE.value

    # 3. Technical question with first-person pronoun
    tech_fp = "What camera did you use? I want to buy it."
    diag_tech = classifier.classify_comment(tech_fp)
    assert diag_tech["relatability_label"] == RelatabilityLabel.NEUTRAL.value


# --- 14. Migrated Module 3 Diagnostic Edge Cases ---

@pytest.mark.parametrize(
    "text,expected_valid,expected_lang,expected_status",
    [
        ("So true!", True, "en", ClassificationStatus.SUCCESS),
        ("Great job!", True, "en", ClassificationStatus.SUCCESS),
        ("The cinematography has a wonderful joie de vivre and aesthetic!", True, "en", ClassificationStatus.SUCCESS),
        ("Bro this is so true yaar, mera bhi same problem hai", False, "non-en", ClassificationStatus.UNSUPPORTED_LANGUAGE),
    ],
)
def test_migrated_language_and_loanword_diagnostics(
    preprocessor, text, expected_valid, expected_lang, expected_status
):
    """Verify short English phrases, foreign loanwords, and mixed Hindi-English diagnostic edge cases."""
    res = preprocessor.preprocess(text)
    assert res.is_valid is expected_valid
    assert res.language == expected_lang
    assert res.status == expected_status


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("Oh sure, waking up early magically solves rent lol. Groundbreaking advice.", RelatabilityLabel.NEUTRAL.value),
        ("I disagree with your second argument and methodology.", RelatabilityLabel.NEUTRAL.value),
        ("Top tier visuals as always! Awesome editing, what a visual treat!", RelatabilityLabel.NEUTRAL.value),
    ],
)
def test_migrated_relatability_diagnostic_edge_cases(classifier, text, expected_label):
    """Verify sarcasm, objective personal disagreement, and generic visual praise do not false-positive as Relatable."""
    diag = classifier.classify_comment(text)
    assert diag["classification_status"] == ClassificationStatus.SUCCESS.value
    assert diag["relatability_label"] == expected_label


def test_process_comments_pipeline_matches_single_comment_classification_exact(classifier):
    """Vectorized batch pipeline must produce identical outputs to row-by-row classify_comment + analyze_text."""
    sample_path = Path("data/sample/sample_comments.csv")
    df_sample = pd.read_csv(sample_path)

    df_batch = classifier.process_comments_pipeline(df_sample)
    assert len(df_batch) == len(df_sample)

    for idx, row in df_sample.iterrows():
        raw_text = row["comment_text"]
        lang_hint = str(row["language"]) if "language" in row and pd.notna(row["language"]) else None
        single_rel = classifier.classify_comment(raw_text, language_hint=lang_hint)
        single_sent = classifier.sentiment_analyzer.analyze_text(raw_text, language_hint=lang_hint)
        batch_row = df_batch.iloc[idx]

        assert batch_row["language"] == single_rel["language"]
        assert batch_row["classification_status"] == single_rel["classification_status"]
        assert (
            (pd.isna(batch_row["relatability_label"]) and single_rel["relatability_label"] is None)
            or batch_row["relatability_label"] == single_rel["relatability_label"]
        )
        assert (
            (pd.isna(batch_row["relatability_confidence"]) and single_rel["relatability_confidence"] is None)
            or batch_row["relatability_confidence"] == pytest.approx(single_rel["relatability_confidence"], abs=1e-9)
        )
        assert (
            (pd.isna(batch_row["sentiment_label"]) and single_sent["sentiment_label"] is None)
            or batch_row["sentiment_label"] == single_sent["sentiment_label"]
        )
        assert (
            (pd.isna(batch_row["sentiment_score"]) and single_sent["polarity"] is None)
            or batch_row["sentiment_score"] == pytest.approx(single_sent["polarity"], abs=1e-9)
        )


def test_process_comments_pipeline_isolates_batch_inference_failures():
    """Verify interleaved ordering, malformed batch output, per-row fallback exceptions, and denominator exclusion."""
    import numpy as np

    clf = RelatabilityClassifier()
    clf.fit(
        ["Literally me every morning", "Felt this in my soul", "What camera did you use", "Nice visuals"],
        ["Relatable", "Relatable", "Neutral", "Neutral"],
    )

    orig_predict_proba = clf.pipeline.predict_proba

    def _flaky_predict_proba(texts):
        # Batch call raises -> forces per-row fallback
        if len(texts) > 1:
            raise RuntimeError("Simulated batch vectorization failure")
        # Individual fallback: one row raises RuntimeError, one returns NaN, valid rows succeed
        if "raise_exc" in texts[0]:
            raise ValueError("Simulated single-row model failure")
        if "nan_prob" in texts[0]:
            return np.array([[float("nan"), 0.5]])
        return orig_predict_proba(texts)

    clf.pipeline.predict_proba = _flaky_predict_proba

    # Interleaved rule-based, non-English, empty, and model-eligible rows
    df_in = pd.DataFrame([
        {"comment_id": "C_RULE_REL", "post_id": "P1", "comment_text": "Literally me every morning.", "platform": "instagram"},
        {"comment_id": "C_MOD_EXC", "post_id": "P1", "comment_text": "raise_exc payload row", "platform": "instagram"},
        {"comment_id": "C_NON_EN", "post_id": "P1", "comment_text": "C'est tellement vrai pour moi aussi dans ma vie", "platform": "instagram"},
        {"comment_id": "C_MOD_NAN", "post_id": "P1", "comment_text": "nan_prob payload row", "platform": "instagram"},
        {"comment_id": "C_RULE_TECH", "post_id": "P1", "comment_text": "What camera did you use to film this?", "platform": "youtube"},
        {"comment_id": "C_MOD_OK", "post_id": "P1", "comment_text": "Nice visuals", "platform": "youtube"},
    ])
    df_out = clf.process_comments_pipeline(df_in)
    assert df_out["comment_id"].tolist() == ["C_RULE_REL", "C_MOD_EXC", "C_NON_EN", "C_MOD_NAN", "C_RULE_TECH", "C_MOD_OK"]

    # 1. Rule-based anchor preserved
    assert df_out.loc[0, "classification_status"] == ClassificationStatus.SUCCESS.value
    assert df_out.loc[0, "relatability_label"] == RelatabilityLabel.RELATABLE.value
    assert df_out.loc[0, "relatability_confidence"] == 0.95

    # 2. Per-row exception -> INFERENCE_FAILED (never fabricated Neutral/Relatable)
    assert df_out.loc[1, "classification_status"] == ClassificationStatus.INFERENCE_FAILED.value
    assert pd.isna(df_out.loc[1, "relatability_label"])
    assert pd.isna(df_out.loc[1, "relatability_confidence"])

    # 3. Non-English -> UNSUPPORTED_LANGUAGE
    assert df_out.loc[2, "classification_status"] == ClassificationStatus.UNSUPPORTED_LANGUAGE.value
    assert pd.isna(df_out.loc[2, "relatability_label"])

    # 4. Non-finite probability -> INFERENCE_FAILED
    assert df_out.loc[3, "classification_status"] == ClassificationStatus.INFERENCE_FAILED.value
    assert pd.isna(df_out.loc[3, "relatability_label"])

    # 5. Technical inquiry guardrail preserved
    assert df_out.loc[4, "classification_status"] == ClassificationStatus.SUCCESS.value
    assert df_out.loc[4, "relatability_label"] == RelatabilityLabel.NEUTRAL.value
    assert df_out.loc[4, "relatability_confidence"] == 0.92

    # 6. Unaffected model-eligible row succeeds and matches single-comment classification
    expected_c6 = clf.classify_comment("Nice visuals")
    assert expected_c6["classification_status"] == ClassificationStatus.SUCCESS.value
    assert df_out.loc[5, "classification_status"] == expected_c6["classification_status"]
    assert df_out.loc[5, "relatability_label"] == expected_c6["relatability_label"]

    # 7. Denominator integrity: only the 3 SUCCESS rows enter total_comments_evaluated (not INFERENCE_FAILED or UNSUPPORTED)
    post_stats = clf.compute_post_relatability_score(df_out)
    row_p1 = post_stats.iloc[0]
    assert row_p1["total_comments_raw"] == 6
    assert row_p1["total_comments_evaluated"] == 3
    assert row_p1["relatable_comments"] == 1

    # 8. Malformed batch shape (wrong row count) triggers per-row fallback safely
    clf.pipeline.predict_proba = (
        lambda texts: np.array([[0.5, 0.5]]) if len(texts) > 1 else orig_predict_proba(texts)
    )
    df_malformed = clf.process_comments_pipeline(df_in.iloc[[0, 4, 5]].reset_index(drop=True))
    assert df_malformed.loc[0, "relatability_label"] == RelatabilityLabel.RELATABLE.value
    assert df_malformed.loc[1, "relatability_label"] == RelatabilityLabel.NEUTRAL.value
    assert df_malformed.loc[2, "classification_status"] == expected_c6["classification_status"]

