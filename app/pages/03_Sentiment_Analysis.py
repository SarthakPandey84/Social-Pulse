"""Page 3: Audience Sentiment & Relatability Analyzer.

Multi-dimensional NLP analytics distinguishing:
1. Relatability (authentic audience identification and shared vulnerability)
2. Sentiment (emotional valence: Positive, Negative, Neutral, Mixed)
"""

import math
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from app.components.data_loader import (
    DatasetLoadError,
    SYNTHETIC_DEMO_DATASET_ID,
    get_relatability_classifier,
    list_available_datasets,
    load_comments_dataset,
    load_json_artifact,
    load_selected_comments_dataset,
)
from app.components.ui_widgets import (
    build_nlp_diagnostic_card_html,
    render_dataset_selector,
    render_header,
    render_metric_card,
    render_provenance_banner,
)
from src.data_ingestion.schemas import ClassificationStatus, RelatabilityLabel

render_header(
    title="Module 3: Audience Sentiment & Relatability",
    subtitle="Orthogonal NLP evaluation separating authentic problem identification from emotional polarity"
)

all_datasets = list_available_datasets(root_dir=root_dir)
comment_datasets = list_available_datasets(root_dir=root_dir, required_table="comments")
selected_dataset_id = render_dataset_selector(
    comment_datasets,
    all_dataset_ids=[d["dataset_id"] for d in all_datasets],
    key="sp_dataset_selector_p3",
    location="sidebar",
)

# 1. Load Data & Model Metadata via shared cached loaders
processed_path = root_dir / "data" / "processed" / "processed_comments.parquet"
sample_path = root_dir / "data" / "sample" / "sample_comments.csv"
model_path = root_dir / "models" / "relatability_classifier.joblib"
meta_path = root_dir / "models" / "relatability_metadata.json"

active_meta = {"dataset_id": selected_dataset_id, "provenance": "synthetic_data"}
if selected_dataset_id == SYNTHETIC_DEMO_DATASET_ID:
    st.info(
        "ℹ️ **Dataset Provenance & Demonstration Notice:** This module runs on a **synthetic benchmark comments dataset** "
        "designed to demonstrate multi-tier NLP pipelines, language safety guards, and orthogonal dimension separation. "
        "Evaluation metrics demonstrate statistical mechanics and baseline comparison, not empirical real-world validation."
    )
    df_comments, _ = load_comments_dataset(processed_path, sample_path)
    if df_comments is None:
        st.warning("Comments dataset not found. Please run the ingestion and benchmark pipeline.")
        st.stop()
else:
    try:
        df_comments, active_meta = load_selected_comments_dataset(selected_dataset_id, root_dir=root_dir)
    except DatasetLoadError as exc:
        st.error(f"❌ Dataset Load Error: {exc}")
        st.stop()
    render_provenance_banner(active_meta)

# Initialize / reuse cached classifier for interactive testing
classifier = get_relatability_classifier(model_path=model_path)

# Load metadata if available
metadata = load_json_artifact(meta_path)
real_benchmarks = load_json_artifact(root_dir / "models" / "real_data_benchmarks.json") or {}

# 2. Key Metrics Calculation
total_comments = len(df_comments)
status_series = df_comments["classification_status"]
valid_mask = status_series == ClassificationStatus.SUCCESS.value
valid_comments = df_comments[valid_mask]
n_valid = len(valid_comments)

# Count successful classifications strictly within valid_comments so failed/invalid rows never enter denominators
n_relatable = int((valid_comments["relatability_label"] == RelatabilityLabel.RELATABLE.value).sum())
n_neutral = int((valid_comments["relatability_label"] == RelatabilityLabel.NEUTRAL.value).sum())
n_uncertain = int((status_series == ClassificationStatus.UNCERTAIN.value).sum())
n_unsupported = int(status_series.isin([
    ClassificationStatus.UNSUPPORTED_LANGUAGE.value,
    ClassificationStatus.INSUFFICIENT_CONTEXT.value,
    ClassificationStatus.EMPTY_TEXT.value,
]).sum())
n_inference_failed = int((status_series == ClassificationStatus.INFERENCE_FAILED.value).sum())
n_invalid_or_unsupported = n_unsupported + n_inference_failed

# Denominator for relatability percentage: valid evaluatable comments
relatability_pct = (n_relatable / n_valid * 100.0) if n_valid > 0 else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_metric_card("Total Ingested", f"{total_comments:,}")
with col2:
    render_metric_card("Relatable Comments", f"{n_relatable:,}")
with col3:
    render_metric_card("Neutral Comments", f"{n_neutral:,}")
with col4:
    render_metric_card("Uncertain / Unsupported / Failed", f"{n_uncertain + n_invalid_or_unsupported:,}")
with col5:
    render_metric_card("Relatability Ratio", f"{relatability_pct:.1f}%")

st.caption(
    f"**Note on Denominator:** Relatability percentage is calculated over successfully evaluated comments "
    f"(N = {n_valid:,}), excluding {n_unsupported} unsupported/insufficient/empty comments, "
    f"{n_inference_failed} inference failures, and {n_uncertain} ambiguous cases."
)

st.markdown("---")

# 3. Interactive Analytical Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & Cross-Analysis",
    "🧪 Interactive Comment Tester",
    "🔍 Comment Explorer & Auditing",
    "🧠 Model Cards & Methodology"
])

# --- TAB 1: OVERVIEW & CROSS-ANALYSIS ---
with tab1:
    st.subheader("Orthogonal Dimensions: Relatability vs Sentiment")
    st.markdown(
        "A critical innovation of this architecture is separating **Audience Identification (Relatability)** "
        "from **Emotional Valence (Sentiment)**. A comment expressing acute personal pain or burnout is *Negative* in sentiment, "
        "yet represents the highest tier of *Relatable* problem awareness."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 1. Relatability Distribution")
        rel_display_mask = status_series.isin([
            ClassificationStatus.SUCCESS.value,
            ClassificationStatus.UNCERTAIN.value,
        ])
        rel_counts = (
            df_comments["relatability_label"]
            .where(rel_display_mask)
            .fillna("Excluded")
            .value_counts()
            .reset_index()
        )
        rel_counts.columns = ["Category", "Count"]
        fig_rel = px.pie(
            rel_counts,
            names="Category",
            values="Count",
            color="Category",
            color_discrete_map={
                "Relatable": "#10B981",
                "Neutral": "#64748B",
                "Uncertain": "#F59E0B",
                "Excluded": "#CBD5E1"
            },
            hole=0.45
        )
        fig_rel.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.06,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
            margin=dict(l=20, r=20, t=24, b=82),
            height=385,
        )
        st.plotly_chart(fig_rel, width="stretch")

    with c2:
        st.markdown("#### 2. Sentiment Polarity Breakdown")
        sent_counts = df_comments["sentiment_label"].fillna("Excluded").value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        fig_sent = px.bar(
            sent_counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#3B82F6",
                "Neutral": "#94A3B8",
                "Negative": "#EF4444",
                "Mixed": "#8B5CF6",
                "Excluded": "#CBD5E1"
            }
        )
        fig_sent.update_xaxes(automargin=True, title_standoff=6)
        fig_sent.update_yaxes(automargin=True)
        fig_sent.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                title_text=None,
                font=dict(size=10.5),
            ),
            margin=dict(l=48, r=32, t=24, b=118),
            height=385,
        )
        st.plotly_chart(fig_sent, width="stretch")

    st.markdown("#### 3. Cross-Tabulation Heatmap (Relatability × Sentiment)")
    # Pivot table of valid comments
    valid_df = df_comments[df_comments["classification_status"] == ClassificationStatus.SUCCESS.value]
    if not valid_df.empty:
        crosstab = pd.crosstab(
            valid_df["relatability_label"],
            valid_df["sentiment_label"],
            margins=False
        )
        fig_heat = px.imshow(
            crosstab,
            labels=dict(x="Sentiment Valence", y="Relatability Dimension", color="Count"),
            x=crosstab.columns,
            y=crosstab.index,
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_heat.update_layout(height=300, margin=dict(l=20, r=28, t=30, b=48))
        st.plotly_chart(fig_heat, width="stretch")

    # Platform breakdown
    st.markdown("#### 4. Relatability Across Platforms")
    if "platform" in df_comments.columns:
        plat_stats = (
            df_comments[rel_display_mask]
            .groupby(["platform", "relatability_label"])
            .size()
            .reset_index(name="count")
        )
    else:
        plat_stats = pd.DataFrame()
    if not plat_stats.empty:
        fig_plat = px.bar(
            plat_stats,
            x="platform",
            y="count",
            color="relatability_label",
            barmode="group",
            color_discrete_map={"Relatable": "#10B981", "Neutral": "#64748B", "Uncertain": "#F59E0B"},
            title="Comment Relatability by Platform"
        )
        fig_plat.update_xaxes(automargin=True, title_standoff=6)
        fig_plat.update_yaxes(automargin=True)
        fig_plat.update_layout(
            height=380,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.27,
                xanchor="center",
                x=0.5,
                title_text=None,
                font=dict(size=11),
            ),
            margin=dict(l=48, r=28, t=48, b=100),
        )
        st.plotly_chart(fig_plat, width="stretch")
    else:
        st.info("No English comments cleared language/context safety gates in this dataset to display a platform breakdown.")


# --- TAB 2: INTERACTIVE TESTER ---
with tab2:
    st.subheader("Live Multi-Dimensional Inference Engine")
    st.markdown(
        "Test individual comments against the hybrid pipeline. The engine evaluates text normalization, "
        "language safety checks, heuristic guardrails, statistical TF-IDF probabilities, and sentiment valence."
    )

    preset_col1, preset_col2, preset_col3 = st.columns(3)
    preset_text = "This happens to me every single morning, I felt this in my soul."
    if preset_col1.button("📌 Example 1: Negative + Relatable"):
        preset_text = "I struggle with this daily, feels like everyone else has it figured out."
    if preset_col2.button("📷 Example 2: Positive + Neutral (Gear Question)"):
        preset_text = "Great video! What camera and editing software did you use?"
    if preset_col3.button("🌍 Example 3: Non-English Comment"):
        preset_text = "C'est tellement vrai pour moi aussi dans ma vie."

    test_input = st.text_area("Enter comment text:", value=preset_text, height=90)

    if test_input:
        with st.spinner("Analyzing text..."):
            rel_diag = classifier.classify_comment(test_input)
            sent_diag = classifier.sentiment_analyzer.analyze_text(test_input)

        st.markdown("### Diagnostic Analysis Results")

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            rel_status = rel_diag["classification_status"]
            raw_rel_label = rel_diag["relatability_label"]
            if raw_rel_label:
                rel_label = raw_rel_label
            elif rel_status == ClassificationStatus.INFERENCE_FAILED.value:
                rel_label = "Inference Failed"
            else:
                rel_label = "Unavailable"
            badge_color = (
                "#10B981"
                if rel_label == "Relatable"
                else "#64748B"
                if rel_label == "Neutral"
                else "#EF4444"
                if rel_status == ClassificationStatus.INFERENCE_FAILED.value
                else "#F59E0B"
            )
            conf = rel_diag["relatability_confidence"]
            conf_float = None
            if conf is not None and not isinstance(conf, bool):
                try:
                    candidate = float(conf)
                    if math.isfinite(candidate):
                        conf_float = candidate
                except (TypeError, ValueError):
                    conf_float = None
            conf_str = f"{conf_float:.1%}" if conf_float is not None else "N/A"
            if conf_float is None:
                if rel_status == ClassificationStatus.INFERENCE_FAILED.value:
                    conf_desc = "Unavailable — statistical inference failed"
                else:
                    conf_desc = "Confidence unavailable (not scored)"
            elif conf_float in [0.92, 0.95]:
                conf_desc = "Heuristic rule weight"
            else:
                conf_desc = "Model probability (uncalibrated)"
            st.markdown(
                build_nlp_diagnostic_card_html(
                    eyebrow="RELATABILITY DIMENSION",
                    headline=rel_label,
                    metric_label="Classification Confidence:",
                    metric_value=conf_str,
                    accent_color=badge_color,
                    footnote=conf_desc,
                ),
                unsafe_allow_html=True,
            )

        with res_c2:
            sent_label = sent_diag["sentiment_label"] or "None"
            sent_color = "#3B82F6" if sent_label == "Positive" else "#EF4444" if sent_label == "Negative" else "#8B5CF6" if sent_label == "Mixed" else "#94A3B8"
            pol = sent_diag.get("polarity")
            pol_str = f"{pol:+.2f}" if pol is not None else "N/A"
            st.markdown(
                build_nlp_diagnostic_card_html(
                    eyebrow="SENTIMENT DIMENSION",
                    headline=sent_label,
                    metric_label="Heuristic Valence Score:",
                    metric_value=pol_str,
                    accent_color=sent_color,
                    footnote="Lexicon-derived polarity [-1.0, +1.0]",
                ),
                unsafe_allow_html=True,
            )

        with res_c3:
            status = rel_diag["classification_status"]
            status_color = (
                "#10B981"
                if status == "success"
                else "#EF4444"
                if status in ("unsupported_language", ClassificationStatus.INFERENCE_FAILED.value)
                else "#F59E0B"
            )
            st.markdown(
                build_nlp_diagnostic_card_html(
                    eyebrow="EXECUTION STATUS",
                    headline=status,
                    metric_label="Detected Lang:",
                    metric_value=str(rel_diag["language"]),
                    accent_color=status_color,
                ),
                unsafe_allow_html=True,
            )

        if rel_diag.get("rejection_reason"):
            st.warning(f"⚠️ **Reason / Uncertainty:** {rel_diag['rejection_reason']}")


# --- TAB 3: COMMENT EXPLORER & AUDITING ---
with tab3:
    st.subheader("Comment Explorer & Integrity Audit")
    st.markdown("Filter and inspect individual comments to audit model predictions and uncertainty boundaries.")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        plat_vals = sorted(df_comments["platform"].dropna().astype(str).unique().tolist()) if "platform" in df_comments.columns else []
        plat_filter = st.selectbox("Platform Filter", ["All"] + plat_vals)
    with f_col2:
        rel_filter = st.selectbox("Relatability Filter", ["All", "Relatable", "Neutral", "Uncertain"])
    with f_col3:
        sent_filter = st.selectbox("Sentiment Filter", ["All", "Positive", "Negative", "Neutral", "Mixed"])
    with f_col4:
        status_filter = st.selectbox(
            "Status Filter",
            [
                "All",
                ClassificationStatus.SUCCESS.value,
                ClassificationStatus.INSUFFICIENT_CONTEXT.value,
                ClassificationStatus.UNSUPPORTED_LANGUAGE.value,
                ClassificationStatus.EMPTY_TEXT.value,
                ClassificationStatus.UNCERTAIN.value,
                ClassificationStatus.INFERENCE_FAILED.value,
            ],
        )

    # Search filter
    search_query = st.text_input("Search in comment text:", "")

    # Apply filters
    filtered_df = df_comments.copy()
    if plat_filter != "All":
        filtered_df = filtered_df[filtered_df["platform"] == plat_filter]
    if rel_filter != "All":
        filtered_df = filtered_df[filtered_df["relatability_label"] == rel_filter]
    if sent_filter != "All":
        filtered_df = filtered_df[filtered_df["sentiment_label"] == sent_filter]
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["classification_status"] == status_filter]
    if search_query:
        filtered_df = filtered_df[filtered_df["comment_text"].str.contains(search_query, case=False, na=False)]

    st.write(f"Showing **{len(filtered_df):,}** matching comments:")

    display_cols = [
        c for c in [
            "comment_id", "post_id", "platform", "comment_text",
            "relatability_label", "relatability_confidence",
            "sentiment_label", "sentiment_score", "classification_status"
        ] if c in filtered_df.columns
    ]
    st.dataframe(filtered_df[display_cols].head(50), width="stretch")

    # Auditing Uncertain and Unsupported records
    with st.expander("🔍 Inspect Uncertain and Rejected Comments for Manual Review"):
        problematic_df = df_comments[df_comments["classification_status"] != ClassificationStatus.SUCCESS.value]
        if not problematic_df.empty:
            st.dataframe(problematic_df[["comment_id", "comment_text", "language", "classification_status"]], width="stretch")
        else:
            st.write("No uncertain or rejected records found.")


# --- TAB 4: MODEL CARDS & METHODOLOGY ---
with tab4:
    st.subheader("NLP Model Card & Empirical Benchmarks")

    st.markdown(
        """
        ### 1. Classification Architecture
        The Relatability Classifier uses a **Multi-Tier Hybrid Architecture**:
        1. **Layer 0 (Input Validation & Language Safety):** Discards empty/whitespace strings and detects non-English comments (e.g. Devanagari, French, German, Spanish). Unsupported languages are marked with `UNSUPPORTED_LANGUAGE` status rather than silently forcing English statistical models.
        2. **Layer 1 (Domain Heuristic Guardrails):**
           - **Experiential Anchor Rules:** Detects authentic vulnerability and first-person resonance phrases (*"literally me"*, *"felt this in my soul"*, *"stop describing me"*, *"thought I was the only one"*).
           - **Technical Inquiry Filter:** Filters out camera, lens, software, font, and track queries (*"what camera"*, *"editing software"*, *"checked your page"*) to prevent false positives.
        3. **Layer 2 (Statistical Model):** Sublinear TF-IDF (word n-grams `(1, 2)`) paired with class-weighted Logistic Regression to capture nuanced conversational phrases.
        4. **Layer 3 (Uncertainty Boundary):** Comments with predicted probabilities in the boundary `[0.42, 0.58]` without explicit rules are assigned `Uncertain` with status `ClassificationStatus.UNCERTAIN`.
        """
    )

    if metadata and "models_benchmarked" in metadata:
        st.markdown("### 2. Empirical Benchmark Comparison")

        # Benchmark 1: Repeated Instance Split
        st.markdown("#### A. Multi-Post Synthetic Benchmark (N = 114 test rows)")
        bench_data = []
        for model_key, stats in metadata["models_benchmarked"].items():
            name_display = model_key.replace("_", " ").title()
            bench_data.append({
                "Model / Baseline": name_display,
                "Accuracy": f"{stats['accuracy']:.1%}",
                "Relatable Recall": f"{stats['recall_relatable']:.1%}",
                "Relatable Precision": f"{stats['precision_relatable']:.1%}",
                "Macro F1": f"{stats['f1_macro']:.3f}",
                "Uncertain Count": stats.get("uncertain_count", 0)
            })
        st.table(pd.DataFrame(bench_data))

        # Benchmark 2: Strictly Unique Text Split
        if "strictly_unique_text_benchmark" in metadata:
            st.markdown("#### B. Non-Redundant Unique Text Evaluation (N = 28 distinct test comments)")
            uniq_data = []
            for model_key, stats in metadata["strictly_unique_text_benchmark"].items():
                name_display = model_key.replace("_", " ").title()
                uniq_data.append({
                    "Model / Baseline": name_display,
                    "Accuracy": f"{stats['accuracy']:.1%}",
                    "Relatable Recall": f"{stats['recall_relatable']:.1%}",
                    "Relatable Precision": f"{stats['precision_relatable']:.1%}",
                    "Macro F1": f"{stats['f1_macro']:.3f}",
                    "Uncertain Count": stats.get("uncertain_count", 0)
                })
            st.table(pd.DataFrame(uniq_data))

        st.caption(
            "**Evaluation Split Details:** Comments are partitioned on unique text hashes (75% train / 25% test). "
            "Table A evaluates on rows sampled across multi-post simulations; Table B evaluates strictly on non-redundant distinct comments."
        )

    m3_real = real_benchmarks.get("modules", {}).get("module3", {})
    if isinstance(m3_real, dict) and m3_real:
        st.markdown("#### C. External Real-World Benchmark Evaluation (Held-Out & Guardrails)")
        ig_sent = m3_real.get("instagram_sentiment_heldout", {})
        lok = m3_real.get("lokananta_indonesian_guardrail", {})
        instaart = m3_real.get("instaart_comments", {})
        if ig_sent:
            tf_full = (ig_sent.get("full_heldout_test") or {}).get("tfidf_logreg", {})
            tf_uniq = (ig_sent.get("strict_unique_heldout_test") or {}).get("tfidf_logreg", {})
            st.markdown(
                f"- **Held-out Instagram Sentiment (`instagram_comments_sentiment_hf`, zero training contamination):** "
                f"Train N = {ig_sent.get('train_samples')}, Full Test N = {ig_sent.get('test_samples_full')} "
                f"(Strict-Unique Test N = {ig_sent.get('test_samples_strict_unique')}). "
                f"TF-IDF + Logistic Regression achieved accuracy = `{tf_full.get('accuracy', 0.0):.3f}` "
                f"(unique `{tf_uniq.get('accuracy', 0.0):.3f}`), Macro-F1 = `{tf_full.get('macro_f1', 0.0):.3f}` "
                f"(unique `{tf_uniq.get('macro_f1', 0.0):.3f}`), ECE = `{tf_full.get('ece', 0.0):.3f}`."
            )
        if lok:
            total_lok = max(1, int(lok.get("total_comments", 0) or 0))
            hinted_rej = int(lok.get("hinted_unsupported_language_count", 0) or 0)
            st.markdown(
                f"- **Indonesian Language Guardrail Stress Test (`zenodo_lokananta_comments_ig`, N = {lok.get('total_comments', 0)}):** "
                f"Hint-free heuristic rejected {lok.get('auto_unsupported_language_pct', 0.0):.2f}% as `unsupported_language`; "
                f"metadata `language_hint='id'` routed {hinted_rej:,}/{total_lok:,} ({100.0 * hinted_rej / total_lok:.1f}%) "
                f"to `unsupported_language` without English model hallucination."
            )
        if instaart:
            st.markdown(
                f"- **Instagram Art Comments (`instaart_human_ai`, N = {instaart.get('total_comments', 0):,}):** "
                f"Segmented into human vs. AI creator cohorts while preserving `insufficient_context` and non-English abstentions."
            )

    st.markdown(
        """
        ### 3. Methodological Limitations & Provenance Disclosures
        - **Label Provenance & Circularity Risk:** Benchmark labels were manually authored with predefined ground truth labels. Rule guardrails in `RelatabilityClassifier` were formulated by the same author using common experiential phrases. Consequently, rule precision on synthetic tests is circular and artificially elevated; no independent human annotations or inter-rater reliability scores (e.g. Cohen's Kappa) exist.
        - **Template Overlap:** While exact string leakage between train and test is prevented, synthetic comments naturally share sentence templates (*"What [camera/lens]..."*, *"Literally me every [morning/exam]"*).
        - **Confidence Semantics:** Confidence values are uncalibrated. Rules output fixed weights (0.92–0.95), and the classifier outputs raw logistic regression probabilities. They are not calibrated Bayesian posterior probabilities.
        - **Monolingual Limitation:** Currently optimized strictly for English discourse. Multilingual sentiment requires multilingual transformer embeddings (e.g. XLM-RoBERTa) with dedicated GPU hardware.
        """
    )
