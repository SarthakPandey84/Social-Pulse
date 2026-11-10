"""Out-of-sample NLP evaluation, calibration diagnostics, and multilingual guardrail checks."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.data_ingestion.schemas import ClassificationStatus, SourceProvenance
from src.nlp.relatability_classifier import RelatabilityClassifier
from src.nlp.sentiment_analyzer import SentimentAnalyzer


def compute_multiclass_calibration(
    y_true: List[str],
    probs: np.ndarray,
    classes: List[str],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute multiclass Brier score and Expected Calibration Error (ECE)."""
    if len(y_true) == 0:
        return {"brier_score": None, "ece": None}

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx.get(str(y), -1) for y in y_true], dtype=int)
    valid = y_idx >= 0
    if not np.any(valid):
        return {"brier_score": None, "ece": None}

    probs_v = probs[valid]
    y_idx_v = y_idx[valid]
    one_hot = np.zeros_like(probs_v)
    one_hot[np.arange(len(y_idx_v)), y_idx_v] = 1.0

    brier = float(np.mean(np.sum((probs_v - one_hot) ** 2, axis=1)))

    confs = np.max(probs_v, axis=1)
    preds = np.argmax(probs_v, axis=1)
    correct = (preds == y_idx_v).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for b_idx in range(n_bins):
        lo, hi = bins[b_idx], bins[b_idx + 1]
        in_bin = (confs >= lo) & (confs <= hi if b_idx == n_bins - 1 else confs < hi)
        cnt = int(np.sum(in_bin))
        if cnt > 0:
            acc_bin = float(np.mean(correct[in_bin]))
            conf_bin = float(np.mean(confs[in_bin]))
            ece += (cnt / n) * abs(acc_bin - conf_bin)

    return {"brier_score": round(brier, 4), "ece": round(float(ece), 4)}


def evaluate_heldout_instagram_sentiment(df_comments: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate strictly out-of-sample on instagram_comments_sentiment_hf (`test.csv` never seen in train)."""
    required = {"comment_text", "ground_truth_sentiment", "eval_split"}
    missing = required - set(df_comments.columns)
    if missing:
        raise ValueError(f"Missing required columns for held-out sentiment evaluation: {sorted(missing)}")

    work = df_comments.copy()
    work["eval_split_norm"] = work["eval_split"].astype(str).str.strip().str.lower()
    train_df = work.loc[work["eval_split_norm"] == "train"].copy()
    test_df = work.loc[work["eval_split_norm"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Both non-empty 'train' and 'test' splits are required.")

    # Strict cross-split duplicate audit on normalized lowercase stripped text
    train_norms = set(train_df["comment_text"].astype(str).str.strip().str.lower())
    test_norms = test_df["comment_text"].astype(str).str.strip().str.lower()
    overlap_mask = test_norms.isin(train_norms)
    test_unique_df = test_df.loc[~overlap_mask].copy()

    # 1. Supervised TF-IDF + LogisticRegression trained strictly on `train_df`
    tfidf_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    tfidf_pipe.fit(train_df["comment_text"].astype(str).tolist(), train_df["ground_truth_sentiment"].astype(str).tolist())
    classes = list(tfidf_pipe.named_steps["clf"].classes_)

    def _eval_subset(sub_df: pd.DataFrame) -> Dict[str, Any]:
        texts = sub_df["comment_text"].astype(str).tolist()
        y_true = sub_df["ground_truth_sentiment"].astype(str).tolist()

        y_pred_tfidf = tfidf_pipe.predict(texts).tolist()
        probs_tfidf = tfidf_pipe.predict_proba(texts)
        cal = compute_multiclass_calibration(y_true, probs_tfidf, classes)
        rep_tfidf = classification_report(y_true, y_pred_tfidf, output_dict=True, zero_division=0)

        # Lexicon SentimentAnalyzer baseline
        analyzer = SentimentAnalyzer()
        lex_preds: List[Optional[str]] = []
        lex_statuses: List[str] = []
        for t in texts:
            r = analyzer.analyze_text(t)
            lex_preds.append(r["sentiment_label"])
            lex_statuses.append(r["classification_status"])

        valid_lex = [i for i, p in enumerate(lex_preds) if p is not None]
        y_true_lex = [y_true[i] for i in valid_lex]
        y_pred_lex = [str(lex_preds[i]) for i in valid_lex]
        lex_acc = float(accuracy_score(y_true_lex, y_pred_lex)) if valid_lex else None
        lex_f1 = float(f1_score(y_true_lex, y_pred_lex, average="macro", zero_division=0)) if valid_lex else None

        return {
            "n_samples": len(sub_df),
            "tfidf_logreg": {
                "accuracy": round(float(accuracy_score(y_true, y_pred_tfidf)), 4),
                "macro_f1": round(float(f1_score(y_true, y_pred_tfidf, average="macro", zero_division=0)), 4),
                "weighted_f1": round(float(f1_score(y_true, y_pred_tfidf, average="weighted", zero_division=0)), 4),
                "brier_score": cal["brier_score"],
                "ece": cal["ece"],
                "per_class": {
                    k: {
                        "precision": round(float(v["precision"]), 4),
                        "recall": round(float(v["recall"]), 4),
                        "f1": round(float(v["f1-score"]), 4),
                        "support": int(v["support"]),
                    }
                    for k, v in rep_tfidf.items()
                    if isinstance(v, dict) and k not in {"macro avg", "weighted avg"}
                },
            },
            "lexicon_baseline": {
                "evaluated_count": len(valid_lex),
                "rejected_count": len(sub_df) - len(valid_lex),
                "accuracy_on_evaluated": round(lex_acc, 4) if lex_acc is not None else None,
                "macro_f1_on_evaluated": round(lex_f1, 4) if lex_f1 is not None else None,
                "mixed_count": sum(1 for p in lex_preds if p == "Mixed"),
            },
        }

    # Perturbation / Negation Consistency Probe on SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    pos_probe = analyzer.analyze_text("I love this advice so much!")
    neg_probe = analyzer.analyze_text("I do not love this advice at all.")
    negation_flips_polarity = bool(
        pos_probe["polarity"] is not None
        and neg_probe["polarity"] is not None
        and pos_probe["polarity"] > neg_probe["polarity"]
    )

    return {
        "dataset_id": "instagram_comments_sentiment_hf",
        "provenance": SourceProvenance.WEAKLY_LABELLED_DATA.value,
        "train_samples": int(len(train_df)),
        "test_samples_full": int(len(test_df)),
        "cross_split_duplicate_texts_in_test": int(overlap_mask.sum()),
        "test_samples_strict_unique": int(len(test_unique_df)),
        "full_heldout_test": _eval_subset(test_df),
        "strict_unique_heldout_test": _eval_subset(test_unique_df),
        "negation_perturbation_check": {
            "positive_polarity": pos_probe["polarity"],
            "negated_polarity": neg_probe["polarity"],
            "passed": negation_flips_polarity,
        },
        "limitations": [
            (
                f"Labels come from weakly labelled public social comments (N={len(train_df)} train / {len(test_df)} test) "
                f"with {int(overlap_mask.sum())} exact text duplicates overlapping train and test; "
                "both full and strict-unique metrics are reported."
            ),
            "Small test sample size yields wide confidence intervals on per-class F1.",
        ],
    }


def evaluate_lokananta_language_guardrail(df_comments: pd.DataFrame) -> Dict[str, Any]:
    """Verify non-English rejection guardrail preserves None labels on Indonesian comments."""
    analyzer = SentimentAnalyzer()
    rel = RelatabilityClassifier()

    texts = df_comments["comment_text"].astype(str).tolist()
    hints = (
        df_comments["language_hint"].tolist()
        if "language_hint" in df_comments.columns
        else [None] * len(texts)
    )

    auto_statuses: Dict[str, int] = {}
    auto_null_labels = 0
    for txt in texts:
        out = analyzer.analyze_text(txt, language_hint=None)
        st = out["classification_status"]
        auto_statuses[st] = auto_statuses.get(st, 0) + 1
        if out["sentiment_label"] is None:
            auto_null_labels += 1

    hinted_rejected = 0
    hinted_null_rel = 0
    for txt, h in zip(texts, hints):
        r_out = rel.classify_comment(txt, language_hint=str(h) if pd.notna(h) else "id")
        if r_out["classification_status"] == ClassificationStatus.UNSUPPORTED_LANGUAGE.value:
            hinted_rejected += 1
        if r_out["relatability_label"] is None:
            hinted_null_rel += 1

    n = len(texts)
    auto_unsupported = auto_statuses.get(ClassificationStatus.UNSUPPORTED_LANGUAGE.value, 0)
    auto_non_success = sum(v for k, v in auto_statuses.items() if k != ClassificationStatus.SUCCESS.value)

    return {
        "dataset_id": "zenodo_lokananta_comments_ig",
        "provenance": SourceProvenance.REAL_PUBLIC_DATA.value,
        "total_comments": n,
        "auto_detect_status_counts": auto_statuses,
        "auto_unsupported_language_pct": round((auto_unsupported / max(n, 1)) * 100.0, 2),
        "auto_abstained_total_pct": round((auto_non_success / max(n, 1)) * 100.0, 2),
        "auto_null_sentiment_preserved": int(auto_null_labels),
        "hinted_unsupported_language_count": int(hinted_rejected),
        "hinted_null_relatability_preserved": int(hinted_null_rel),
    }


def evaluate_instaart_comments(df_comments: pd.DataFrame) -> Dict[str, Any]:
    """Run M3 hybrid pipeline on real Instagram Human vs AI art comments cleanly."""
    clf = RelatabilityClassifier()
    enriched = clf.process_comments_pipeline(
        df_comments,
        text_col="comment_text",
        comment_id_col="comment_id",
        post_id_col="post_id",
        platform_col="platform",
        timestamp_col="comment_timestamp",
        language_col=None,
    )
    merged = df_comments.reset_index(drop=True).copy()
    merged["sentiment_label"] = enriched["sentiment_label"]
    merged["sentiment_score"] = enriched["sentiment_score"]
    merged["relatability_label"] = enriched["relatability_label"]
    merged["classification_status"] = enriched["classification_status"]

    status_counts = merged["classification_status"].value_counts(dropna=False).to_dict()
    group_stats: Dict[str, Any] = {}
    if "account_group" in merged.columns:
        for grp, gdf in merged.groupby("account_group"):
            ok = gdf[gdf["classification_status"] == ClassificationStatus.SUCCESS.value]
            group_stats[str(grp)] = {
                "total_comments": int(len(gdf)),
                "successfully_classified": int(len(ok)),
                "abstained_count": int(len(gdf) - len(ok)),
                "mean_sentiment_score": round(float(ok["sentiment_score"].dropna().mean()), 4) if len(ok) > 0 else None,
                "positive_pct_of_classified": round(float((ok["sentiment_label"] == "Positive").mean() * 100.0), 2) if len(ok) > 0 else None,
                "relatable_pct_of_classified": round(float((ok["relatability_label"] == "Relatable").mean() * 100.0), 2) if len(ok) > 0 else None,
            }

    rank_stats: Dict[str, Any] = {}
    if "comment_rank_index" in merged.columns:
        top10 = merged[merged["comment_rank_index"] <= 10]
        rest = merged[merged["comment_rank_index"] > 10]
        rank_stats = {
            "top_10_display_mean_sentiment": round(float(top10["sentiment_score"].dropna().mean()), 4) if top10["sentiment_score"].notna().any() else None,
            "rank_gt_10_mean_sentiment": round(float(rest["sentiment_score"].dropna().mean()), 4) if rest["sentiment_score"].notna().any() else None,
        }

    return {
        "dataset_id": "instaart_human_ai",
        "provenance": SourceProvenance.REAL_PUBLIC_DATA.value,
        "total_comments": int(len(merged)),
        "classification_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "by_account_group": group_stats,
        "display_rank_comparison": rank_stats,
    }

