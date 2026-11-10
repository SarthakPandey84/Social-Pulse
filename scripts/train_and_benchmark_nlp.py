"""Script to generate synthetic benchmark comments, benchmark NLP models, and serialize artifacts."""

import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
import sys
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.data_ingestion.schemas import ClassificationStatus, RelatabilityLabel
from src.nlp.relatability_classifier import RelatabilityClassifier

# --- 1. Diverse Benchmark Comment Corpus ---

RELATABLE_CORPUS = [
    # Explicit vulnerability & shared struggle
    ("This happens to me every single morning.", "Negative", "Relatable"),
    ("I thought I was the only one going through this.", "Neutral", "Relatable"),
    ("Exactly what I am experiencing right now with my job.", "Negative", "Relatable"),
    ("Why did you have to call me out like this?!", "Neutral", "Relatable"),
    ("Too accurate, please stop attacking me lol.", "Positive", "Relatable"),
    ("Never felt so understood by an Instagram post.", "Positive", "Relatable"),
    ("I needed to hear this today, thank you so much.", "Positive", "Relatable"),
    ("I felt this in my soul, so painfully true.", "Negative", "Relatable"),
    ("Literally me every single morning when the alarm rings.", "Negative", "Relatable"),
    ("Story of my life right now, can't even focus for 10 minutes.", "Negative", "Relatable"),
    ("Living this exact nightmare at my workplace right now.", "Negative", "Relatable"),
    ("My anxiety has been telling me this all week.", "Negative", "Relatable"),
    ("I always overthink every email I send to my manager.", "Negative", "Relatable"),
    ("I struggle with this daily, feels like everyone else has it figured out.", "Negative", "Relatable"),
    ("Hits way too close to home.", "Neutral", "Relatable"),
    ("Are you living inside my brain? This was literally my morning.", "Neutral", "Relatable"),
    ("I do this every single time I start studying.", "Neutral", "Relatable"),
    ("My exact thoughts when I opened LinkedIn today.", "Neutral", "Relatable"),
    ("I had this exact breakdown yesterday at my desk.", "Negative", "Relatable"),
    ("This called me out so hard I almost closed the app.", "Neutral", "Relatable"),
    ("Stop describing my daily routine please lol.", "Positive", "Relatable"),
    ("I thought I was weird for doing this, thank god I'm not alone.", "Positive", "Relatable"),
    ("Felt that deep in my chest. Career burnout is real.", "Negative", "Relatable"),
    ("The way my heart dropped because of how accurate this is.", "Negative", "Relatable"),
    ("Realest video on the internet today.", "Positive", "Relatable"),
    ("I am in this video and I don't like it haha.", "Neutral", "Relatable"),
    ("This hits home so hard. Grad school is breaking me.", "Negative", "Relatable"),
    ("Every word here describes what I've been silently enduring.", "Negative", "Relatable"),
    ("I hate how accurately this captures my imposter syndrome.", "Negative", "Relatable"),
    ("Bro is narrating my 20s better than my therapist.", "Positive", "Relatable"),
    ("Why is this me trying to submit an assignment at 11:59 pm?", "Negative", "Relatable"),
    ("I feel so seen right now, definitely sharing with my partner.", "Positive", "Relatable"),
    ("I literally cried watching this because I felt so isolated until now.", "Negative", "Relatable"),
    ("Same here, been dealing with this since college.", "Neutral", "Relatable"),
    ("I do the exact same thing whenever someone gives me a compliment.", "Neutral", "Relatable"),
    ("Going through this exact dilemma right now.", "Neutral", "Relatable"),
    ("Why does my brain always sabotage me like this?", "Negative", "Relatable"),
    ("Does anyone else get this sick feeling before team meetings?", "Negative", "Relatable"),
    ("This is me pretending to have my life together at family dinners.", "Negative", "Relatable"),
    ("I wake up at 5am, make coffee, and stare at the wall in existential dread.", "Negative", "Relatable"),
    ("I have 50 tabs open right now and can't complete one sentence.", "Negative", "Relatable"),
    ("I freeze whenever my boss asks for a quick sync.", "Negative", "Relatable"),
    ("I literally paced around my room for an hour avoiding one phone call.", "Negative", "Relatable"),
    ("I spent 4 hours rewriting one paragraph because of perfectionism.", "Negative", "Relatable"),
    ("This is 100% me during every exam week.", "Neutral", "Relatable"),
    ("I feel attacked by the accuracy of this reel.", "Neutral", "Relatable"),
    ("So true, I literally cancel plans just to sit in silence.", "Neutral", "Relatable"),
    ("I thought I was just lazy, but this explains the executive dysfunction.", "Positive", "Relatable"),
    ("My bank account and I felt this deeply.", "Negative", "Relatable"),
    ("I swear we all live the same collective experience in our 20s.", "Positive", "Relatable"),
    ("This is the most relatable thing I have seen all month.", "Positive", "Relatable"),
    ("I literally did this 10 minutes ago, get out of my head!", "Positive", "Relatable"),
    ("I can't even count how many times this has happened to me.", "Negative", "Relatable"),
    ("I felt so guilty about doing this until I saw your post.", "Positive", "Relatable"),
    ("The procrastination loop you described is my entire semester.", "Negative", "Relatable"),
    ("I am suffering from this exact issue at my startup.", "Negative", "Relatable")
]

NEUTRAL_CORPUS = [
    # Technical & Equipment Inquiries
    ("What camera did you use to film this video?", "Neutral", "Neutral"),
    ("What editing software was used for this transition?", "Neutral", "Neutral"),
    ("Which microphone is that on your desk?", "Neutral", "Neutral"),
    ("What song is playing in the background?", "Neutral", "Neutral"),
    ("Could you share the color grading LUT for this?", "Neutral", "Neutral"),
    ("What font did you use for the animated title text?", "Neutral", "Neutral"),
    ("Where did you buy that desk lamp?", "Neutral", "Neutral"),
    ("Is this filmed in 24fps or 60fps?", "Neutral", "Neutral"),
    ("Which lens did you pair with your Sony body?", "Neutral", "Neutral"),
    ("Can you drop the link to your studio setup?", "Neutral", "Neutral"),
    ("What app do you use to track your daily habits?", "Neutral", "Neutral"),
    ("What audio interface are you running?", "Neutral", "Neutral"),
    ("Are you using Premiere Pro or DaVinci Resolve?", "Neutral", "Neutral"),
    ("Where can I find the background music track?", "Neutral", "Neutral"),
    
    # Generic Praise (Positive Sentiment, but NON-relatable)
    ("Great video, keep up the awesome content!", "Positive", "Neutral"),
    ("Checked your page, looks super cool and clean.", "Positive", "Neutral"),
    ("Such high production quality and brilliant lighting!", "Positive", "Neutral"),
    ("Nice visuals, loved the cinematic transitions.", "Positive", "Neutral"),
    ("Amazing cinematography, you deserve way more subscribers!", "Positive", "Neutral"),
    ("Really well-crafted video, good pacing throughout.", "Positive", "Neutral"),
    ("Good job on the animation breakdown.", "Positive", "Neutral"),
    ("Awesome editing, what a visual treat!", "Positive", "Neutral"),
    ("Super clean aesthetics and color palette.", "Positive", "Neutral"),
    ("Subscribed! Looking forward to your next upload.", "Positive", "Neutral"),
    ("Brilliant video presentation, very engaging to watch.", "Positive", "Neutral"),
    ("Fantastic storytelling and crisp sound design.", "Positive", "Neutral"),
    ("Top tier visuals as always!", "Positive", "Neutral"),
    ("Love the aesthetic of your studio workspace.", "Positive", "Neutral"),
    ("Great thumbnail, really caught my attention.", "Positive", "Neutral"),
    ("Wonderful production value on this project.", "Positive", "Neutral"),

    # Objective Critique & Informational Feedback
    ("Not sure I agree with your second argument.", "Negative", "Neutral"),
    ("I disagree with the conclusion presented at 02:15.", "Negative", "Neutral"),
    ("This was very informative and clearly articulated.", "Positive", "Neutral"),
    ("Interesting perspective on modern education trends.", "Positive", "Neutral"),
    ("The audio was slightly quiet in the opening chapter.", "Negative", "Neutral"),
    ("Could you provide citations or data sources for these claims?", "Neutral", "Neutral"),
    ("Good overview of the topic for beginners.", "Positive", "Neutral"),
    ("I think you should cover time management tools in part 2.", "Neutral", "Neutral"),
    ("Interesting theory, but peer-reviewed evidence is mixed.", "Neutral", "Neutral"),
    ("This explains the concept much better than my textbook.", "Positive", "Neutral"),
    ("Could you do a follow-up video on YouTube automation?", "Neutral", "Neutral"),
    ("Shared this with my university research group.", "Positive", "Neutral"),
    ("The pacing was a bit rushed towards the end.", "Negative", "Neutral"),
    ("Please add chapters or timestamps in the description.", "Neutral", "Neutral"),
    ("A bit too repetitive in the second half.", "Negative", "Neutral"),
    ("Solid educational breakdown, thank you for sharing.", "Positive", "Neutral"),

    # First-person phrasing that is NOT relatable (Contrastive pairs!)
    ("I bought that camera last year and returned it.", "Negative", "Neutral"),
    ("I think the lighting in the background is a bit harsh.", "Negative", "Neutral"),
    ("I prefer the previous editing style you had.", "Neutral", "Neutral"),
    ("I found this video while browsing the explore tab.", "Neutral", "Neutral"),
    ("I watched this at 2x speed.", "Neutral", "Neutral"),
    ("I am a software engineer and I don't use this stack.", "Neutral", "Neutral"),
    ("I disagree with your methodology in this experiment.", "Negative", "Neutral"),
    ("I think you should invest in a better pop filter.", "Negative", "Neutral"),

    # Sarcastic / Dismissive Comments
    ("Oh sure, waking up early magically solves rent lol.", "Negative", "Neutral"),
    ("Groundbreaking advice, just stop procrastinating guys.", "Negative", "Neutral"),
    ("Thanks captain obvious for this revelation.", "Negative", "Neutral"),
    ("Another generic productivity guru giving generic advice.", "Negative", "Neutral")
]

# Edge case comments (Insufficient context & Non-English)
EDGE_CASE_CORPUS = [
    # Insufficient context
    ("lol", "Neutral", "Insufficient_Context", ClassificationStatus.INSUFFICIENT_CONTEXT.value),
    ("idk", "Neutral", "Insufficient_Context", ClassificationStatus.INSUFFICIENT_CONTEXT.value),
    ("ok", "Neutral", "Insufficient_Context", ClassificationStatus.INSUFFICIENT_CONTEXT.value),
    ("cool", "Positive", "Insufficient_Context", ClassificationStatus.INSUFFICIENT_CONTEXT.value),
    ("👍🔥", "Positive", "Insufficient_Context", ClassificationStatus.INSUFFICIENT_CONTEXT.value),
    ("???", "Neutral", "Insufficient_Context", ClassificationStatus.INSUFFICIENT_CONTEXT.value),
    ("...", "Neutral", "Insufficient_Context", ClassificationStatus.INSUFFICIENT_CONTEXT.value),
    ("👀", "Neutral", "Insufficient_Context", ClassificationStatus.INSUFFICIENT_CONTEXT.value),
    # Unsupported language
    ("C'est tellement vrai pour moi aussi dans ma vie.", "Positive", "Relatable", ClassificationStatus.UNSUPPORTED_LANGUAGE.value),
    ("Das kenne ich zu gut aus eigener beruflicher Erfahrung.", "Neutral", "Relatable", ClassificationStatus.UNSUPPORTED_LANGUAGE.value),
    ("bhai ye toh meri hi kahani hai bilkul sach bola aapne", "Positive", "Relatable", ClassificationStatus.UNSUPPORTED_LANGUAGE.value),
    ("este video me describe perfectamente amigo mio", "Positive", "Relatable", ClassificationStatus.UNSUPPORTED_LANGUAGE.value),
    ("merci beaucoup pour cette explication tres claire", "Positive", "Neutral", ClassificationStatus.UNSUPPORTED_LANGUAGE.value),
    ("que buena camara usas para grabar esto", "Positive", "Neutral", ClassificationStatus.UNSUPPORTED_LANGUAGE.value)
]


def generate_benchmark_dataset(
    n_samples: int = 500,
    base_dir: Optional[Path] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a diverse synthetic comment corpus mapped to post IDs from Module 1."""
    random.seed(seed)
    np.random.seed(seed)

    target_root = Path(base_dir) if base_dir is not None else root_dir
    posts_path = target_root / "data" / "processed" / "processed_posts.parquet"
    if posts_path.exists():
        df_posts = pd.read_parquet(posts_path)
        post_ids = df_posts["post_id"].unique().tolist()
        platforms_map = dict(zip(df_posts["post_id"], df_posts["platform"]))
    else:
        post_ids = [f"POST_{i:04d}" for i in range(1, 101)]
        platforms_map = {pid: "instagram" if int(pid.split("_")[1]) % 2 == 1 else "youtube" for pid in post_ids}

    records = []
    
    # 1. Base unique entries
    unique_pool = []
    for txt, sent, rel in RELATABLE_CORPUS:
        unique_pool.append((txt, sent, rel, ClassificationStatus.SUCCESS.value, "en"))
    for txt, sent, rel in NEUTRAL_CORPUS:
        unique_pool.append((txt, sent, rel, ClassificationStatus.SUCCESS.value, "en"))
    for txt, sent, rel, status in EDGE_CASE_CORPUS:
        lang = "non-en" if status == ClassificationStatus.UNSUPPORTED_LANGUAGE.value else "en"
        unique_pool.append((txt, sent, rel, status, lang))

    # 2. Replicate with realistic post mapping and timestamps
    idx = 1
    base_dates = pd.date_range("2026-09-01", "2026-11-03", freq="6h")

    while len(records) < n_samples:
        entry = random.choice(unique_pool)
        txt, sent, rel, status, lang = entry
        pid = random.choice(post_ids)
        plat = platforms_map.get(pid, "instagram")
        ts = random.choice(base_dates).isoformat() + "Z"
        likes = int(np.random.exponential(scale=8))

        records.append({
            "comment_id": f"CMT_{idx:05d}",
            "post_id": pid,
            "platform": plat,
            "comment_text": txt,
            "ground_truth_relatability": rel,
            "ground_truth_sentiment": sent,
            "classification_status": status,
            "language": lang,
            "likes": likes,
            "comment_timestamp": ts
        })
        idx += 1

    df_out = pd.DataFrame(records)
    return df_out


# --- 2. Baselines and Benchmark Evaluation ---

def evaluate_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Train and evaluate Candidate Classifiers strictly on English valid comments with zero text leakage."""
    # Filter to evaluatable English records
    valid_mask = (df["classification_status"] == ClassificationStatus.SUCCESS.value) & (df["ground_truth_relatability"].isin(["Relatable", "Neutral"]))
    df_eval = df[valid_mask].copy()

    # Split on UNIQUE comment texts to guarantee zero train/test text leakage
    unique_texts = df_eval[["comment_text", "ground_truth_relatability"]].drop_duplicates().reset_index(drop=True)
    
    train_texts, test_texts = train_test_split(
        unique_texts,
        test_size=0.25,
        random_state=42,
        stratify=unique_texts["ground_truth_relatability"]
    )

    train_set = set(train_texts["comment_text"])
    test_set = set(test_texts["comment_text"])

    # Split dataset based on text partitions
    train_df = df_eval[df_eval["comment_text"].isin(train_set)].reset_index(drop=True)
    test_df = df_eval[df_eval["comment_text"].isin(test_set)].reset_index(drop=True)

    X_train = train_df["comment_text"].tolist()
    y_train = train_df["ground_truth_relatability"].tolist()

    X_test = test_df["comment_text"].tolist()
    y_test = test_df["ground_truth_relatability"].tolist()

    print(f"Dataset split: Train={len(train_df)} comments ({len(train_set)} unique), Test={len(test_df)} comments ({len(test_set)} unique)")

    # Baseline 1: Rule-Based Keyword Matcher
    rule_classifier = RelatabilityClassifier()
    # Baseline 2: First-person pronoun heuristic
    first_person_preds = []
    for t in X_test:
        clean = t.lower()
        if re.search(r"\b(i|me|my|myself|we|us|our)\b", clean):
            first_person_preds.append("Relatable")
        else:
            first_person_preds.append("Neutral")

    # Baseline 1 Predictions (Rules only, unfitted pipeline)
    rule_preds = rule_classifier.predict(X_test)

    # Model 3: Statistical TF-IDF + Logistic Regression
    stat_classifier = RelatabilityClassifier()
    stat_classifier.fit(X_train, y_train)
    stat_preds = stat_classifier.predict(X_test)

    # --- 1. Repeated Instance Evaluation (114 rows across simulated posts) ---
    print("\n" + "=" * 50)
    print("RUNNING REPEATED INSTANCE BENCHMARK (N=114 rows)")
    print("=" * 50)
    repeated_results = {}
    for name, preds in [
        ("first_person_pronoun_heuristic", first_person_preds),
        ("rule_based_keyword_baseline", rule_preds),
        ("hybrid_tfidf_logistic_pipeline", stat_preds)
    ]:
        binary_preds = ["Relatable" if p == "Relatable" else "Neutral" for p in preds]
        n_uncertain = sum(1 for p in preds if p == "Uncertain")
        acc = accuracy_score(y_test, binary_preds)
        prec = precision_score(y_test, binary_preds, pos_label="Relatable", zero_division=0)
        rec = recall_score(y_test, binary_preds, pos_label="Relatable", zero_division=0)
        f1_rel = f1_score(y_test, binary_preds, pos_label="Relatable", zero_division=0)
        f1_macro = f1_score(y_test, binary_preds, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, binary_preds, labels=["Relatable", "Neutral"]).tolist()

        repeated_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision_relatable": round(float(prec), 4),
            "recall_relatable": round(float(rec), 4),
            "f1_relatable": round(float(f1_rel), 4),
            "f1_macro": round(float(f1_macro), 4),
            "uncertain_count": n_uncertain,
            "confusion_matrix": cm
        }
        print(f"[{name}] Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f} | Macro F1: {f1_macro:.4f} | Uncertain: {n_uncertain}")

    # --- 2. Strictly Unique Text Evaluation (N=28 non-redundant distinct comments) ---
    print("\n" + "=" * 50)
    print("RUNNING STRICTLY UNIQUE TEXT BENCHMARK (N=28 distinct texts)")
    print("=" * 50)
    unique_X_test = test_texts["comment_text"].tolist()
    unique_y_test = test_texts["ground_truth_relatability"].tolist()

    u_first_person_preds = ["Relatable" if re.search(r"\b(i|me|my|myself|we|us|our)\b", t.lower()) else "Neutral" for t in unique_X_test]
    u_rule_preds = rule_classifier.predict(unique_X_test)
    u_stat_preds = stat_classifier.predict(unique_X_test)

    unique_results = {}
    for name, preds in [
        ("first_person_pronoun_heuristic", u_first_person_preds),
        ("rule_based_keyword_baseline", u_rule_preds),
        ("hybrid_tfidf_logistic_pipeline", u_stat_preds)
    ]:
        binary_preds = ["Relatable" if p == "Relatable" else "Neutral" for p in preds]
        n_uncertain = sum(1 for p in preds if p == "Uncertain")
        acc = accuracy_score(unique_y_test, binary_preds)
        prec = precision_score(unique_y_test, binary_preds, pos_label="Relatable", zero_division=0)
        rec = recall_score(unique_y_test, binary_preds, pos_label="Relatable", zero_division=0)
        f1_rel = f1_score(unique_y_test, binary_preds, pos_label="Relatable", zero_division=0)
        f1_macro = f1_score(unique_y_test, binary_preds, average="macro", zero_division=0)
        cm = confusion_matrix(unique_y_test, binary_preds, labels=["Relatable", "Neutral"]).tolist()

        unique_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision_relatable": round(float(prec), 4),
            "recall_relatable": round(float(rec), 4),
            "f1_relatable": round(float(f1_rel), 4),
            "f1_macro": round(float(f1_macro), 4),
            "uncertain_count": n_uncertain,
            "confusion_matrix": cm
        }
        print(f"[{name}] Acc: {acc:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f} | Macro F1: {f1_macro:.4f} | Uncertain: {n_uncertain}")

    # Retrain on full evaluatable corpus for production deployment
    final_classifier = RelatabilityClassifier()
    final_classifier.fit(X_train + X_test, y_train + y_test)

    return repeated_results, unique_results, final_classifier, len(train_df), len(test_df), len(train_texts), len(test_texts)


REQUIRED_SAMPLE_COMMENT_COLUMNS = [
    "comment_id",
    "post_id",
    "platform",
    "comment_text",
    "ground_truth_relatability",
    "classification_status",
    "language",
    "comment_timestamp",
]


def validate_sample_comments_dataframe(df_comments: pd.DataFrame) -> None:
    """Validate sample_comments.csv before training and classifying."""
    if df_comments.empty:
        raise ValueError("sample_comments.csv is empty; expected labeled benchmark comments.")

    missing_cols = [c for c in REQUIRED_SAMPLE_COMMENT_COLUMNS if c not in df_comments.columns]
    if missing_cols:
        raise ValueError(f"sample_comments.csv is missing required columns: {missing_cols}")

    for non_null_col in ["comment_id", "post_id", "platform", "classification_status"]:
        if df_comments[non_null_col].isna().any():
            raise ValueError(f"sample_comments.csv contains nulls in required column '{non_null_col}'.")

    valid_mask = (
        (df_comments["classification_status"] == ClassificationStatus.SUCCESS.value)
        & (df_comments["ground_truth_relatability"].isin(["Relatable", "Neutral"]))
    )
    df_eval = df_comments[valid_mask]
    unique_texts = df_eval[["comment_text", "ground_truth_relatability"]].drop_duplicates()
    class_counts = unique_texts["ground_truth_relatability"].value_counts()

    if len(unique_texts) < 8 or int(class_counts.get("Relatable", 0)) < 2 or int(class_counts.get("Neutral", 0)) < 2:
        raise ValueError(
            "sample_comments.csv must contain at least 8 unique valid rows with both "
            "'Relatable' and 'Neutral' labels (>= 2 unique per class) for stratified evaluation."
        )


def run_nlp_pipeline(
    base_dir: Optional[Path] = None,
    overwrite_sample: bool = False,
    n_samples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train, benchmark, serialize, and score comments without modifying sample inputs when overwrite_sample=False."""
    target_root = Path(base_dir) if base_dir is not None else root_dir
    sample_csv_path = target_root / "data" / "sample" / "sample_comments.csv"

    if overwrite_sample:
        print("Generating synthetic benchmark comments dataset...")
        df_comments = generate_benchmark_dataset(n_samples=n_samples, base_dir=target_root, seed=seed)
        sample_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_comments.to_csv(sample_csv_path, index=False)
        print(f"Saved sample comments to {sample_csv_path} ({len(df_comments)} rows)")
    else:
        if not sample_csv_path.exists():
            raise FileNotFoundError(f"Required sample comments file not found at: {sample_csv_path}")
        df_comments = pd.read_csv(sample_csv_path)

    validate_sample_comments_dataframe(df_comments)

    # Run Benchmark Evaluation
    rep_results, uniq_results, final_model, n_train, n_test, n_train_u, n_test_u = evaluate_models(df_comments)

    # Serialize model artifact
    model_save_path = target_root / "models" / "relatability_classifier.joblib"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(model_save_path)
    print(f"Saved trained classifier to {model_save_path}")

    # Process all comments through the full pipeline to create processed dataset
    print("Processing entire dataset through canonical NLP pipeline...")
    processed_df = final_model.process_comments_pipeline(
        df_comments,
        text_col="comment_text",
        comment_id_col="comment_id",
        post_id_col="post_id",
        platform_col="platform",
        timestamp_col="comment_timestamp",
        language_col="language",
    )

    # Save processed comments in CSV and Parquet
    processed_dir = target_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    proc_csv = processed_dir / "processed_comments.csv"
    proc_parquet = processed_dir / "processed_comments.parquet"
    processed_df.to_csv(proc_csv, index=False)
    processed_df.to_parquet(proc_parquet, index=False)
    print(f"Saved processed comments to {proc_csv} and {proc_parquet}")

    trained_at_utc = pd.Timestamp.now("UTC").isoformat()

    # Save deterministic model metadata with full audit transparency
    metadata = {
        "model_version": RelatabilityClassifier.MODEL_VERSION,
        "dataset_provenance": "synthetic_illustrative_benchmark",
        "provenance_statement": (
            "Synthetic benchmark comments dataset created for architectural and NLP pipeline demonstration. "
            "Evaluation metrics demonstrate statistical separation and baseline comparison, not real-world audience validation."
        ),
        "metadata_reproducibility": {
            "artifact_type": "deterministic_model_metadata",
            "intentionally_variable_fields_excluded": ["trained_at", "saved_at", "analysis_timestamp"],
            "runtime_provenance_location": "models/build_provenance.json",
        },
        "label_provenance": {
            "source_type": "manually_authored_synthetic_benchmark_with_predefined_labels",
            "independent_human_annotated": False,
            "circularity_risk_identified": (
                "Rule guardrails in RelatabilityClassifier were designed with domain knowledge of synthetic benchmark phrases. "
                "Therefore, rule-based precision is circular and artificially elevated. No external human annotators were used."
            ),
        },
        "confidence_semantics": {
            "rule_confidence": "Uncalibrated heuristic weights (0.92 for technical queries, 0.95 for experiential anchors)",
            "model_confidence": "Uncalibrated Logistic Regression model probabilities (predict_proba)",
            "calibrated_posterior_probabilities": False,
        },
        "train_test_split_audit": {
            "total_benchmark_records": len(df_comments),
            "total_unique_evaluatable_texts": n_train_u + n_test_u,
            "repeated_split_counts": {"train_rows": n_train, "test_rows": n_test},
            "unique_text_split_counts": {"train_unique": n_train_u, "test_unique": n_test_u},
            "template_leakage_note": (
                "Partitioning on unique texts prevents verbatim string leakage. However, near-duplicate template families "
                "(e.g., 'What [gear] did you use?', 'Literally me every [morning/exam]') naturally exist across splits."
            ),
        },
        "models_benchmarked": rep_results,
        "strictly_unique_text_benchmark": uniq_results,
        "selected_architecture": "Hybrid Multi-Tier Pipeline (Rule Guardrails + TF-IDF Logistic Regression + Uncertainty Window)",
        "language_coverage": "English (non-English explicitly marked unsupported via script & stopword checks)",
        "unsupported_status_count": int((processed_df["classification_status"] == ClassificationStatus.UNSUPPORTED_LANGUAGE.value).sum()),
        "insufficient_context_count": int((processed_df["classification_status"] == ClassificationStatus.INSUFFICIENT_CONTEXT.value).sum()),
        "relatable_count": int((processed_df["relatability_label"] == RelatabilityLabel.RELATABLE.value).sum()),
        "neutral_count": int((processed_df["relatability_label"] == RelatabilityLabel.NEUTRAL.value).sum()),
        "uncertain_count": int((processed_df["classification_status"] == ClassificationStatus.UNCERTAIN.value).sum()),
    }

    meta_path = target_root / "models" / "relatability_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"Saved model metadata to {meta_path}")

    return {
        "comments_count": len(processed_df),
        "model_path": model_save_path,
        "metadata_path": meta_path,
        "processed_parquet_path": proc_parquet,
        "processed_csv_path": proc_csv,
        "metadata": metadata,
        "runtime_provenance": {"trained_at": trained_at_utc},
    }


def main() -> None:
    run_nlp_pipeline(base_dir=root_dir, overwrite_sample=True, n_samples=500, seed=42)


if __name__ == "__main__":
    main()
