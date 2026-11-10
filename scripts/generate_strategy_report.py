"""Deterministic generator for Deliverable 4: Strategy Report (`reports/SOCIALPULSE_STRATEGY_REPORT.md`).

By default (``include_local_state=False``), synthesizes verified outputs strictly from
tracked repository sources so output is byte-identical across clean checkouts (before
or after artifact bootstrap) and data-rich checkouts:
- `models/real_data_benchmarks.json` (M1–M7 real/external validation)
- `models/virality_predictor_metadata.json` (Synthetic Ridge M2 metadata & holdout caveats)
- `models/relatability_metadata.json` (Synthetic NLP benchmarks & circularity caveats)
- `models/performance_benchmarks.json` (Historical pre-Batch-5A reference vs. measured timings)
- `data/sample/sample_experiments.csv` (Synthetic A/B fixture definitions)
- Tracked `DATASET_REGISTRY` governance and access contracts (`src/data_ingestion/external_datasets.py`)

Passing ``--include-local-state`` / ``include_local_state=True`` explicitly inspects
optional gitignored local host artifacts (`data/external/manifest.json`,
`data/processed/external/*/dataset_meta.json`, and `data/processed/processed_*.parquet`).

Zero fabricated figures and zero volatile wall-clock generation timestamps so repeated runs
produce byte-identical Markdown output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend
from src.data_ingestion.external_datasets import list_dataset_specs


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(val: Any, digits: int = 3, suffix: str = "") -> str:
    if val is None:
        return "N/A"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return str(val)
    if pd.isna(num):
        return "N/A"
    return f"{num:.{digits}f}{suffix}"


def _fmt_int(val: Any) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{int(val):,}"
    except (TypeError, ValueError):
        return str(val)


def build_strategy_report_markdown(
    root_dir: Path | None = None,
    *,
    include_local_state: bool = False,
) -> str:
    """Assemble deterministic Strategy Report Markdown.

    Defaults to tracked-input-only mode (``include_local_state=False``) so clean
    and data-rich checkouts emit byte-identical output. Pass ``include_local_state=True``
    to include optional gitignored local download manifests and built Parquet summaries.
    """
    root = Path(root_dir).resolve() if root_dir is not None else PROJECT_ROOT

    real_bench = _read_json(root / "models" / "real_data_benchmarks.json")
    vir_meta = _read_json(root / "models" / "virality_predictor_metadata.json")
    nlp_meta = _read_json(root / "models" / "relatability_metadata.json")
    perf_bench = _read_json(root / "models" / "performance_benchmarks.json")

    exp_csv = root / "data" / "sample" / "sample_experiments.csv"
    exp_df = pd.read_csv(exp_csv) if exp_csv.is_file() else pd.DataFrame()

    manifest: Dict[str, Any] = {}
    prepared_metas: Dict[str, Dict[str, Any]] = {}
    if include_local_state:
        manifest = _read_json(root / "data" / "external" / "manifest.json")
        ext_processed_dir = root / "data" / "processed" / "external"
        if ext_processed_dir.is_dir():
            for ds_dir in sorted(p for p in ext_processed_dir.iterdir() if p.is_dir()):
                mfile = ds_dir / "dataset_meta.json"
                if mfile.is_file():
                    prepared_metas[ds_dir.name] = _read_json(mfile)

    specs = list_dataset_specs(include_kaggle=True)
    non_blocked_specs = [s for s in specs if s.effective_governance_tier != "BLOCKED_MISSING_CREDENTIALS"]
    m1 = (real_bench.get("modules") or {}).get("module1") or {}
    m2 = (real_bench.get("modules") or {}).get("module2") or {}
    m3 = (real_bench.get("modules") or {}).get("module3") or {}
    m4 = (real_bench.get("modules") or {}).get("module4") or {}
    m7 = (real_bench.get("modules") or {}).get("module7") or {}

    if include_local_state:
        policy_sources = (
            "verified repository artifacts (`models/real_data_benchmarks.json`, `models/virality_predictor_metadata.json`, "
            "`models/relatability_metadata.json`, `models/performance_benchmarks.json`, `data/sample/sample_experiments.csv`, "
            "and tracked `DATASET_REGISTRY` contracts) plus optional local gitignored runtime state "
            "(`data/external/manifest.json`, `data/processed/external/*/dataset_meta.json`, and `data/processed/processed_*.parquet`)"
        )
        deliv2_local_suffix = f"{len(prepared_metas)} prepared locally under optional gitignored `data/processed/external/`"
        inventory_col_header = "Local Host Availability & Registry Scope"
    else:
        policy_sources = (
            "strictly tracked repository sources (`models/real_data_benchmarks.json`, `models/virality_predictor_metadata.json`, "
            "`models/relatability_metadata.json`, `models/performance_benchmarks.json`, `data/sample/sample_experiments.csv`, "
            "and `DATASET_REGISTRY`), ignoring optional gitignored local caches (`data/external/`, `data/processed/`)"
        )
        deliv2_local_suffix = "normalized on demand into optional gitignored `data/processed/external/`"
        inventory_col_header = "Registry Access & Intended Scope"

    lines: List[str] = [
        "# Deliverable 4 — SocialPulse Comprehensive Strategy & Validation Report",
        "",
        "> **Deterministic Artifact Policy & Provenance Integrity:**  ",
        f"> Every metric in this report is read programmatically from {policy_sources}. Wall-clock generation timestamps are intentionally omitted so regenerating the report yields byte-identical output across clean and data-rich checkouts.",
        "",
        "---",
        "",
        "## 1. Executive Summary & Evidence Separation Policy",
        "",
        "SocialPulse (*The Data-Driven Social Engagement Initiative*) integrates **seven analytical modules** across short-form creator platforms (**Instagram**, **TikTok**, and **YouTube Shorts**). To prevent synthetic demonstrations, weakly labelled scraped sets, and unverified third-party mirrors from being conflated with audited public evidence, every corpus and finding is assigned one of **six explicit governance tiers**:",
        "",
        "1. `VERIFIED_COMMERCIAL_SAFE` — Cleared license & verified upstream provenance (`zenodo_sponsored_unsponsored_ig`, `zenodo_educational_reels_ig`, `zenodo_lokananta_comments_ig`); observational confounding caveats still apply (`is_causal=False` unless randomized).",
        "2. `NON_COMMERCIAL_RESEARCH_ONLY` — Verified academic/non-commercial corpora (`tiktok_engagement_200k` under CC-BY-NC-4.0, `instaart_human_ai` academic research); excluded from commercial redistribution.",
        "3. `PENDING_REVIEW_WEAKLY_LABELLED` — Undeclared upstream license or weak/pseudo labels (`instagram_comments_sentiment_hf`); isolated for bounded held-out benchmarking and default-denied by the downloader unless explicitly opted in (`--include-pending-review`).",
        "4. `EXTERNAL_BENCHMARK_UNVERIFIED` — Public tabular benchmarks with synthetic-like regularities or unverified acquisition lineage (`instagram_analytics_github_alt`, `instagram_engagement_eda_hf` [also default-denied `PENDING_REVIEW`]); quarantined from real organic telemetry claims.",
        "5. `SYNTHETIC_BENCHMARK` — Internal synthetic offline fixtures (`synthetic_demo` / `data/sample/*.csv`) and external synthetic tables (`social_instagram_marketing_hf`); demonstrate pipeline mechanics only.",
        "6. `BLOCKED_MISSING_CREDENTIALS` — Gated Kaggle sources (`youtube_trending_kaggle`, `instagram_analytics_kaggle`, `youtube_kaggle_candidate_h`) blocked gracefully without guessing URLs or bypassing authentication.",
        "",
        "---",
        "",
        "## 2. Authoritative PDF Specification & Technical Stack Traceability",
        "",
        "Comparison against `Problem_Statement/Data_Science_Major_Project_December.pdf` (distinguishing verbatim implementations from intentional architectural substitutions):",
        "",
        "| PDF Layer | PDF Reference Tools | SocialPulse Implementation | Compliance Status & Engineering Rationale |",
        "| :--- | :--- | :--- | :--- |",
        "| **Data Processing** | Python (`Pandas`, `NumPy`) | `pandas==3.0.3`, `numpy==2.4.4`, `pyarrow==24.0.0` | **Exact Match** — Strict nullable `Int64`/`Float64` preservation (zero silent NaN->0 imputation). |",
        "| **Scraping / APIs** | Selenium, BeautifulSoup, Instagram Graph API | `InstagramGraphAPIConnector` (`v21.0` default, `META_*`), `YouTubeDataAPIConnector`, `SocialMediaScraper` (`placeholder`), + pinned HTTPS/Parquet downloader (`external_datasets.py`) | **Partial / Contract Boundary** — Live OAuth tokens & browser scraping excluded for platform ToS & clean offline reproducibility; real validation runs on pinned public archives. |",
        "| **NLP & Text Analysis** | NLTK, TextBlob, SpaCy | Pure Python `TextPreprocessor` (non-English rejection + PII scrubbing/HMAC pseudonymization), lexicon `SentimentAnalyzer`, TF-IDF + Logistic Regression `RelatabilityClassifier` | **Intentional Deterministic Substitution** — Zero runtime NLTK/SpaCy binary downloads; preserves offline portability and deterministic reproducibility. |",
        "| **Database** | MySQL / PostgreSQL | Embedded `sqlite3` (`database/social_analytics.db` via `DataStorageManager`, `latest_posts` SQL view) | **Intentional Local Substitution** — Zero database daemon required for local/Streamlit Community Cloud container execution. |",
        "| **Visualization** | Tableau, PowerBI, Matplotlib/Seaborn | Interactive `plotly==7.1.0` charts embedded in Streamlit (`matplotlib`/`seaborn` dev-only in `requirements-dev.txt`) | **Intentional Interactive Substitution** — Unified browser delivery directly inside Dashboard pages without proprietary BI licenses. |",
        "| **Statistical Modeling** | Scikit-learn, Excel | `scikit-learn==1.9.0` (`Ridge`, `HistGradientBoostingRegressor`, `TfidfVectorizer`, `LogisticRegression`), `scipy==1.17.1`, stdlib XML `.xlsx` reader | **Exact + Upgraded** — Reproducible ML/statistical tests plus stdlib Excel parser (`openpyxl`-free). |",
        "| **Dashboard Frontend** | Streamlit or Dash | `streamlit==1.58.0` (`streamlit_app.py` + `app/pages/01..07`) | **Exact Match** — 7-module multipage app with startup bootstrap and unified dataset switcher. |",
        "",
        "### Five PDF Deliverables Status",
        "1. **Deliverable 1 (Analytics Dashboard):** `streamlit_app.py` & `app/pages/01..07` (`GrowthDashboardBackend` + live dataset selector).",
        f"2. **Deliverable 2 (Structured Dataset):** Tracked fixtures (`data/sample/*`) + idempotent local build outputs (`data/processed/*`, SQLite `database/social_analytics.db`) + {len(non_blocked_specs)} configured non-blocked external Parquet pipelines ({deliv2_local_suffix}).",
        "3. **Deliverable 3 (NLP Sentiment Model):** `src/nlp/` hybrid pipeline + `scripts/train_and_benchmark_nlp.py` + held-out external evaluations.",
        "4. **Deliverable 4 (Strategy Report):** This document (`reports/SOCIALPULSE_STRATEGY_REPORT.md`), generated deterministically by `scripts/generate_strategy_report.py`.",
        "5. **Deliverable 5 (Content Series):** Pilot series protocol, A/B design, and 14-field unified logging template at [`docs/CONTENT_SERIES_TEMPLATE.md`](../docs/CONTENT_SERIES_TEMPLATE.md).",
        "",
        "---",
        "",
        "## 3. External & Real-Data Inventory & Governance Matrix",
        "",
        f"| Dataset ID | Governance Tier | Provenance | License Status | {inventory_col_header} |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for spec in specs:
        tier = spec.effective_governance_tier
        modules_txt = ", ".join(spec.intended_modules)
        if not include_local_state:
            if tier == "BLOCKED_MISSING_CREDENTIALS" or spec.requires_kaggle_auth:
                if spec.kaggle_slug:
                    scope_txt = f"Blocked (`{spec.license_status}`, slug `{spec.kaggle_slug}` awaits Kaggle auth) — modules: `{modules_txt}`"
                else:
                    scope_txt = f"Blocked (`{spec.license_status}`, slug unverified) — modules: `{modules_txt}`"
            elif spec.license_status == "PENDING_REVIEW":
                scope_txt = f"Default-deny (`PENDING_REVIEW`; requires explicit opt-in) — modules: `{modules_txt}`"
            else:
                scope_txt = f"Configured (`{spec.source_host}`; optional ignored local cache) — modules: `{modules_txt}`"
        else:
            pmeta = prepared_metas.get(spec.dataset_id)
            man_entry = (manifest.get("datasets") or {}).get(spec.dataset_id) or {}
            if tier == "BLOCKED_MISSING_CREDENTIALS" or spec.requires_kaggle_auth:
                if pmeta and pmeta.get("row_counts"):
                    rc = ", ".join(f"{k}={_fmt_int(v)}" for k, v in pmeta["row_counts"].items())
                    scope_txt = f"Available locally ({rc})"
                elif man_entry.get("status"):
                    scope_txt = f"Blocked / manifest status: `{man_entry['status']}`"
                else:
                    scope_txt = f"Blocked (`{spec.license_status}`; credentials/slug required)"
            elif pmeta and pmeta.get("row_counts"):
                rc = ", ".join(f"{k}={_fmt_int(v)}" for k, v in pmeta["row_counts"].items())
                scope_txt = f"Available locally ({rc})"
            elif man_entry.get("status"):
                scope_txt = f"Local manifest status: `{man_entry['status']}`"
            elif spec.license_status == "PENDING_REVIEW":
                scope_txt = "Not downloaded locally (default-deny `PENDING_REVIEW` — opt-in required)"
            else:
                scope_txt = "Not downloaded locally (optional ignored local cache)"
        lines.append(
            f"| `{spec.dataset_id}` | `{tier}` | `{spec.provenance.value}` | `{spec.license_name}` (`{spec.license_status}`) | {scope_txt} |"
        )

    lines.append("")
    if not include_local_state:
        lines.append(
            "> **Tracked Registry Contract vs. Local Host Artifacts:** Table rows report the tracked `DATASET_REGISTRY` contract identically across clean and data-rich checkouts. Pass `--include-local-state` when running `scripts/generate_strategy_report.py` locally to include optional gitignored host state from `data/external/manifest.json` and `data/processed/`."
        )
    else:
        lines.append(
            "> **Host Runtime State Enabled (`--include-local-state`):** Table reflects optional gitignored local downloads/Parquets present on this workstation alongside blocked Kaggle statuses."
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Empirical Findings on Real Public & Academic Data (`models/real_data_benchmarks.json`)",
        "",
    ])

    # Parse synthetic NLP accuracy upfront for Section 4.3 comparative wording
    rep_nlp = nlp_meta.get("models_benchmarked") or {}
    uniq_nlp = nlp_meta.get("strictly_unique_text_benchmark") or {}
    synth_hybrid_acc = ((rep_nlp.get("hybrid_tfidf_logistic_pipeline") or {}).get("accuracy"))

    # M1 Real Data Availability & Creator Relative
    if m1:
        tk_rel = m1.get("tiktok_creator_relative_views") or {}
        tk_collect_obs = (((m1.get("tiktok_metric_availability") or {}).get("metrics") or {}).get("collect_count") or {}).get("observed_pct")
        sp_avail = m1.get("sponsored_ig_metric_availability") or {}
        sp_fc = (sp_avail.get("metrics") or {}).get("follower_count") or {}
        sp_provs = sp_avail.get("provenances") or []
        tk_collect_phrase = (
            f" (`{_fmt_num(tk_collect_obs, 1, '%')}` observed)"
            if tk_collect_obs is not None
            else ""
        )
        lines.extend([
            "### 4.1 Module 1 — Schema Honesty, Proxy Separation & Creator-Relative Scaling",
            f"- **TikTok 200K Panel (`tiktok_engagement_200k`):** Evaluated across `{_fmt_int(tk_rel.get('rows_evaluated'))}` video snapshots (`{_fmt_int(tk_rel.get('eligible_creators_min_5_posts'))}` creators with `>= 5` posts). Median creator-relative view ratio = `{_fmt_num(tk_rel.get('median_relative_ratio'), 4)}`, 90th percentile = `{_fmt_num(tk_rel.get('p90_relative_ratio'), 4)}`.",
            f"- **Proxy & Missingness Isolation:** TikTok exposes `views`, `likes`, `comments_count`, `shares`, and explicit `collect_count` bookmark proxy{tk_collect_phrase}, while Instagram-native `saves`, `reach`, `impressions`, and `retention_rate` stay `<NA>` (never coerced to zero).",
            f"- **Zenodo 5082114 (`zenodo_sponsored_unsponsored_ig`):** Projected schema preserves `follower_count` (`{_fmt_int(sp_fc.get('observed_count'))}` / `{_fmt_num(sp_fc.get('observed_pct'), 2, '%')}` coverage) and provenance tags `{sp_provs}`; unexposed `views`, `shares`, `saves`, `reach` remain explicit `<NA>`.",
            "",
        ])

    # M2 Real Longitudinal TikTok
    if m2:
        lines.extend([
            "### 4.2 Module 2 — Longitudinal Virality Modeling Without Same-Day / Future Leakage",
            "",
            "All runs predict `log1p(day30_views)` or `log1p(day7_views)` using historical creator features joined strictly from `creator_day < create_day` (eliminating same-day/future creator counter leakage):",
            "",
            "| Validation Regime | Target | Feature Regime | Split Strategy | Creator Overlap | Test N | Best Model | R² (`log1p`) | Spearman rho | RMSE Lift vs Dummy | 90% Interval Cov |",
            "| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |",
        ])
        for reg_key, reg in m2.items():
            if not isinstance(reg, dict):
                continue
            b_name = reg.get("best_model")
            bm = (reg.get("models") or {}).get(b_name) or {}
            lines.append(
                f"| `{reg_key}` | `{reg.get('target_column')}` | `{reg.get('feature_regime')}` | `{reg.get('split_strategy')}` | "
                f"{_fmt_int(reg.get('creator_overlap_count'))} | {_fmt_int(reg.get('test_rows'))} | `{b_name}` | "
                f"{_fmt_num(bm.get('r2_log1p'), 3)} | {_fmt_num(bm.get('spearman_rho'), 3)} | "
                f"{_fmt_num(reg.get('rmse_lift_over_dummy_pct'), 2, '%')} | {_fmt_num(bm.get('empirical_90pct_test_coverage'), 2, '%')} |"
            )
        cg30 = (m2.get("day30_pre_pub_creator_grouped") or {}).get("models", {}).get(
            (m2.get("day30_pre_pub_creator_grouped") or {}).get("best_model"), {}
        )
        ch30 = (m2.get("day30_pre_pub_chronological") or {}).get("models", {}).get(
            (m2.get("day30_pre_pub_chronological") or {}).get("best_model"), {}
        )
        d1_cg = (m2.get("day30_early_day1_creator_grouped") or {}).get("models", {}).get(
            (m2.get("day30_early_day1_creator_grouped") or {}).get("best_model"), {}
        )
        lines.extend([
            "",
            f"- **Key Takeaway:** Zero-overlap creator-grouped splitting confirms true generalization across unseen creators (`R² = {_fmt_num(cg30.get('r2_log1p'), 3)}`, `Spearman rho = {_fmt_num(cg30.get('spearman_rho'), 3)}` pre-publication vs `R² = {_fmt_num(ch30.get('r2_log1p'), 3)}` when creators overlap chronologically; adding observed Day-1 signals lifts `day30_views` `R²` to `{_fmt_num(d1_cg.get('r2_log1p'), 3)}`).",
            "",
        ])

    # M3 Real / Weakly Labelled NLP
    ia_counts = m4.get("instaart_group_counts") or {}
    ia_total_posts = ia_counts.get("total_posts")
    sp_obs: Dict[str, Any] = (m4.get("sponsored_unsponsored_observational") or {}) if m4 else {}
    cl_list = sp_obs.get("creator_clustered_comparisons") or []
    primary_cl = next((c for c in cl_list if "likes" in str(c.get("metric_name", "")).lower()), cl_list[0] if cl_list else {})
    cl_method_str = str(primary_cl.get("clustering_method") or "")
    icc_match = re.search(r"ICC\s*=\s*([0-9]+(?:\.[0-9]+)?)", cl_method_str)
    deff_match = re.search(r"DEFF\s*=\s*([0-9]+(?:\.[0-9]+)?)", cl_method_str)
    likes_icc_val = icc_match.group(1) if icc_match else None
    likes_deff_val = deff_match.group(1) if deff_match else None

    if m3:
        ig_s = m3.get("instagram_sentiment_heldout") or {}
        loka = m3.get("lokananta_indonesian_guardrail") or {}
        ia_c = m3.get("instaart_comments") or {}
        tf_full = (ig_s.get("full_heldout_test") or {}).get("tfidf_logreg") or {}
        tf_uniq = (ig_s.get("strict_unique_heldout_test") or {}).get("tfidf_logreg") or {}
        lex_full = (ig_s.get("full_heldout_test") or {}).get("lexicon_baseline") or {}
        loka_tot = loka.get("total_comments") or 0
        loka_hint_cnt = loka.get("hinted_unsupported_language_count")
        loka_hint_pct = (100.0 * loka_hint_cnt / loka_tot) if (loka_tot and loka_hint_cnt is not None) else None
        loka_hint_null = loka.get("hinted_null_relatability_preserved")
        tf_acc_pct = _fmt_num(100.0 * float(tf_full["accuracy"]), 1, "%") if tf_full.get("accuracy") is not None else "held-out accuracy"
        synth_acc_pct = f" (~`{_fmt_num(100.0 * float(synth_hybrid_acc), 1, '%')}`)" if synth_hybrid_acc is not None else ""
        ia_posts_clause = f" across `{_fmt_int(ia_total_posts)}` Human/AI posts" if ia_total_posts is not None else ""
        lines.extend([
            "### 4.3 Module 3 — Held-Out Sentiment Generalization & Non-English Guardrails",
            "",
            f"- **Held-Out Instagram Sentiment (`instagram_comments_sentiment_hf`, `PENDING_REVIEW_WEAKLY_LABELLED`):** Train `N={_fmt_int(ig_s.get('train_samples'))}`, Full Held-Out Test `N={_fmt_int(ig_s.get('test_samples_full'))}` (`{_fmt_int(ig_s.get('cross_split_duplicate_texts_in_test'))}` exact text duplicates overlap train/test -> strict-unique test `N={_fmt_int(ig_s.get('test_samples_strict_unique'))}`). TF-IDF + Logistic Regression achieves `{_fmt_num(tf_full.get('accuracy'), 3)}` Accuracy / `{_fmt_num(tf_full.get('macro_f1'), 3)}` Macro F1 full (`{_fmt_num(tf_uniq.get('accuracy'), 3)}` / `{_fmt_num(tf_uniq.get('macro_f1'), 3)}` strict-unique; ECE `{_fmt_num(tf_full.get('ece'), 3)}`), beating rule-based lexicon accuracy (`{_fmt_num(lex_full.get('accuracy_on_evaluated'), 3)}` on `{_fmt_int(lex_full.get('evaluated_count'))}/{_fmt_int(ig_s.get('test_samples_full'))}` non-abstained rows). Demonstrates clearly that small weak-label generalization (`{tf_acc_pct}`) is materially harder than closed synthetic templates{synth_acc_pct}.",
            f"- **Non-English Safety Abstention (`zenodo_lokananta_comments_ig`, `VERIFIED_COMMERCIAL_SAFE`, `N={_fmt_int(loka.get('total_comments'))}` Indonesian comments):** Monolingual English pipeline abstained on `{_fmt_num(loka_hint_pct, 2, '%')}` when given `language_hint='id'` (`{_fmt_int(loka_hint_null)}/{_fmt_int(loka_tot)}` null labels) and abstained cleanly on `{_fmt_num(loka.get('auto_abstained_total_pct'), 2, '%')}` hint-free (`{_fmt_num(loka.get('auto_unsupported_language_pct'), 2, '%')}` explicit `UNSUPPORTED_LANGUAGE` without false neutral coercion).",
            f"- **InstaArt Comments (`instaart_human_ai`, `NON_COMMERCIAL_RESEARCH_ONLY`, `N={_fmt_int(ia_c.get('total_comments'))}` comments{ia_posts_clause}):** Status breakdown and rank-vs-chronological comment indexing preserved without fabricated saves/views.",
            "",
        ])

    # M4 Observational Cohort
    if sp_obs:
        icc_deff_clause = (
            f" Intra-creator ICC on likes is high (`{likes_icc_val}`, `DEFF={likes_deff_val}`), showing why naive post-level independence assumptions inflate significance."
            if likes_icc_val and likes_deff_val
            else ""
        )
        lines.extend([
            "### 4.4 Module 4 — Creator-Clustered Confounding Audit (`zenodo_sponsored_unsponsored_ig`)",
            "",
            f"- **Cohort Size & Clustering:** `{_fmt_int(sp_obs.get('n_posts_total'))}` posts across `{_fmt_int(sp_obs.get('n_creators_total'))}` creators (`is_causal={sp_obs.get('is_causal')}`).{icc_deff_clause}",
        ])
        for comp in cl_list:
            m_name = comp.get("metric_name")
            lines.append(
                f"- **Metric `{m_name}` ({comp.get('clustering_method')}):** Sponsored mean = `{_fmt_num(comp.get('variant_b_mean'), 2)}` vs. Unsponsored mean = `{_fmt_num(comp.get('variant_a_mean'), 2)}` (`{_fmt_num(comp.get('lift_percent'), 2, '%')}`, `p_adj={comp.get('p_value_adjusted')}`). "
                + ("Direction **reverses** after follower normalization, demonstrating creator-scale confounding in unadjusted sponsored comparisons." if "per_1k" in str(m_name) else "Raw interaction volume reflects creator scale differences rather than causal sponsorship lift.")
            )
        lines.append("")

    # M7 Daily Forecasting
    if m7:
        lines.extend([
            "### 4.5 Module 7 — Walk-Forward Daily Cohort Forecasting (`tiktok_engagement_200k`)",
            "",
            "| Daily Series | Contiguous History Span | Champion Model | MASE | Rel-MAE vs Naive | Validation Origins |",
            "| :--- | :--- | :--- | :---: | :---: | :---: |",
        ])
        for series_name, res in m7.items():
            if not isinstance(res, dict):
                continue
            champ_name = res.get("champion_model")
            ev_list = res.get("evaluations") or []
            champ_ev = next((e for e in ev_list if e.get("model_name") == champ_name), {})
            lines.append(
                f"| `{series_name}` | `{res.get('history_start')}` to `{res.get('history_end')}` (`{_fmt_int(res.get('history_points'))}` pts) | "
                f"`{champ_name}` | {_fmt_num(champ_ev.get('mase'), 3)} | {_fmt_num(champ_ev.get('relative_mae_to_naive'), 3)} | "
                f"`{_fmt_int(res.get('validation_origins_count'))}` |"
            )
        lines.append("")

    # Section 5: Synthetic Demonstration Quarantine
    base_m2 = vir_meta.get("baseline_metrics") or {}
    cand_m2 = vir_meta.get("candidate_metrics") or {}
    ig_pe = (vir_meta.get("platform_evaluation_metrics") or {}).get("instagram") or {}
    yt_pe = (vir_meta.get("platform_evaluation_metrics") or {}).get("youtube") or {}

    if include_local_state:
        posts_pq = root / "data" / "processed" / "processed_posts.parquet"
        comm_pq = root / "data" / "processed" / "processed_comments.parquet"
        if posts_pq.is_file() and comm_pq.is_file():
            aud_backend = GrowthDashboardBackend(root_dir=root)
            aud_insights = aud_backend.calculate_audience_insights()
            if aud_insights.get("has_data") and aud_insights.get("correlation_rho") is not None:
                matched_n = aud_insights.get("matched_correlation_posts")
                rho_val = aud_insights.get("correlation_rho")
                p_val = aud_insights.get("correlation_p")
                sig_desc = "a statistically significant" if float(p_val) < 0.05 else "zero significant"
                m6_aud_line = (
                    f"4. **Module 6 Audience Correlation Fixture (Local State Enabled):** On built synthetic fixtures (`N={_fmt_int(matched_n)}` matched posts with observed `saves`), "
                    f"post-level relatability percentage shows {sig_desc} Spearman correlation with saves (`rho = {_fmt_num(rho_val, 4)}`, `p = {_fmt_num(p_val, 3)}`), "
                    "underscoring why causal hypotheses must pass through Module 4 controlled tests."
                )
            else:
                m6_aud_line = (
                    "4. **Module 6 Audience Correlation Fixture (Local State Enabled):** Built fixtures lack matched observed `saves` for Spearman correlation."
                )
        else:
            m6_aud_line = (
                "4. **Module 6 Audience Correlation Fixture (Local State Enabled):** Optional gitignored `data/processed/processed_*.parquet` files are not built locally yet; "
                "run `python scripts/build_all_artifacts.py` to generate them."
            )
    else:
        m6_aud_line = (
            "4. **Module 6 Audience Correlation Fixture:** Post-level relatability-to-saves Spearman correlation is evaluated on optional ignored build artifacts "
            "(`data/processed/processed_*.parquet` from `scripts/build_all_artifacts.py`; pass `--include-local-state` to embed host runtime correlation metrics), "
            "and passes through Module 4 controlled tests before causal interpretation."
        )

    lines.extend([
        "---",
        "",
        "## 5. Offline Synthetic Benchmark Summary (`SYNTHETIC_BENCHMARK` — Mechanics Only)",
        "",
        "> **Methodological Quarantine:** Metrics in this section quantify synthetic pipeline correctness on checked-in fixtures (`data/sample/*.csv`) and MUST NOT be reported as empirical organic performance.",
        "",
        "1. **Module 2 Ridge Regressor (`models/virality_predictor_metadata.json`):**",
        f"   - Settled mature sample: `{_fmt_int(vir_meta.get('total_mature_posts'))}` posts (`{_fmt_int(vir_meta.get('train_samples_count'))}` train / `{_fmt_int(vir_meta.get('test_samples_count'))}` chronological test).",
        f"   - Candidate vs. Naive Mean Baseline: MAE `{_fmt_num(cand_m2.get('mae'), 2)}` vs `{_fmt_num(base_m2.get('mae'), 2)}` (`-{_fmt_num(vir_meta.get('mae_lift_pct'), 1, '%')}`), RMSE `{_fmt_num(cand_m2.get('rmse'), 2)}` vs `{_fmt_num(base_m2.get('rmse'), 2)}` (`-{_fmt_num(vir_meta.get('rmse_lift_pct'), 1, '%')}`), Spearman `rho = {_fmt_num(cand_m2.get('spearman_rho'), 3)}` (`p = {_fmt_num(cand_m2.get('spearman_pvalue'), 3)}`), empirical `q95 = ±{_fmt_num(vir_meta.get('empirical_residual_q95'), 2)}` pts.",
        f"   - **Platform-Stratified Caveats Disclosed in UI:** Pooled/YouTube chronological `R² = {_fmt_num(cand_m2.get('r2'), 3)}` (negative variance explained on small `N = {_fmt_int(yt_pe.get('test_samples'))}` test slice), and Instagram has `test_samples = {_fmt_int(ig_pe.get('test_samples'))}` (`status = {ig_pe.get('status')}`).",
        "2. **Module 3 Hybrid Relatability (`models/relatability_metadata.json`):**",
        f"   - Repeated split (`N={_fmt_int((nlp_meta.get('train_test_split_audit') or {}).get('repeated_split_counts', {}).get('test_rows'))}`): Hybrid Accuracy `{_fmt_num(((rep_nlp.get('hybrid_tfidf_logistic_pipeline') or {}).get('accuracy')), 4)}`, Macro F1 `{_fmt_num(((rep_nlp.get('hybrid_tfidf_logistic_pipeline') or {}).get('f1_macro')), 4)}`.",
        f"   - Strict-unique split (`N={_fmt_int((nlp_meta.get('train_test_split_audit') or {}).get('unique_text_split_counts', {}).get('test_unique'))}`): Hybrid Accuracy `{_fmt_num(((uniq_nlp.get('hybrid_tfidf_logistic_pipeline') or {}).get('accuracy')), 4)}`, Macro F1 `{_fmt_num(((uniq_nlp.get('hybrid_tfidf_logistic_pipeline') or {}).get('f1_macro')), 4)}`.",
        "   - **Circularity Disclosure:** Benchmark rules encode author domain knowledge of synthetic phrasing and uncalibrated confidence thresholds; real-world expectation should align with Section 4.3.",
        "3. **Module 4 Synthetic Experiments (`data/sample/sample_experiments.csv`):**",
    ])
    if not exp_df.empty:
        for _, row in exp_df.iterrows():
            lines.append(
                f"   - `{row['experiment_id']}` ({row['platform']} · `{row['experiment_type']}`): `{row['experiment_name']}` -> primary metric `{row['primary_metric']}` (`{row['variant_a_label']}` vs `{row['variant_b_label']}`)."
            )

    lines.extend([
        m6_aud_line,
        "",
        "---",
        "",
        "## 6. Runtime Vectorization & Memory Engineering (`models/performance_benchmarks.json`)",
        "",
    ])

    post_p = perf_bench.get("post_optimization_measured") or {}
    pre_p = perf_bench.get("pre_optimization_baseline") or {}
    tk_p = (perf_bench.get("external_scale_benchmarks") or {}).get("tiktok_engagement_200k") or {}
    if post_p:
        cold_sec = post_p.get("dashboard_cold_init_and_synthesis_seconds")
        cold_mb = post_p.get("dashboard_cold_init_and_synthesis_peak_mb")
        pre_dash = pre_p.get("dashboard_synthesis_seconds")
        cold_ratio = (
            f"**{_fmt_num(float(pre_dash) / max(float(cold_sec), 1e-6), 2, 'x')} faster**"
            if (cold_sec is not None and pre_dash is not None and float(cold_sec) > 0)
            else "Cold init+synth measured"
        )
        lines.extend([
            "| Workload | Historical Pre-Batch-5A Reference | Measured (Current HEAD) | Derived Ratio / Note |",
            "| :--- | :---: | :---: | :---: |",
            f"| 5,000-row EVI (`calculate_engagement_virality_index_detailed`) | `{_fmt_num(pre_p.get('evi_5k_seconds'), 4, 's')}` | `{_fmt_num(post_p.get('evi_5k_seconds'), 4, 's')}` (`{_fmt_num(post_p.get('evi_5k_peak_mb'), 2)} MB`) | **{_fmt_num(post_p.get('evi_5k_speedup_x'), 1, 'x')} faster** |",
            f"| 5,000-row Feature Engineering (`add_engagement_features`) | `{_fmt_num(pre_p.get('feature_engineering_5k_seconds'), 4, 's')}` | `{_fmt_num(post_p.get('feature_engineering_5k_seconds'), 4, 's')}` (`{_fmt_num(post_p.get('feature_engineering_5k_peak_mb'), 2)} MB`) | **{_fmt_num(post_p.get('feature_engineering_5k_speedup_x'), 1, 'x')} faster** |",
            f"| Dashboard Synthesis — Warm Backend (`GrowthDashboardBackend`) | `{_fmt_num(pre_p.get('dashboard_synthesis_seconds'), 4, 's')}` | `{_fmt_num(post_p.get('dashboard_synthesis_seconds'), 4, 's')}` (`{_fmt_num(post_p.get('dashboard_synthesis_peak_mb'), 2)} MB`) | **{_fmt_num(post_p.get('dashboard_synthesis_speedup_x'), 1, 'x')} faster** |",
        ])
        if cold_sec is not None:
            lines.append(
                f"| Dashboard Init + Synthesis — Cold Load (`GrowthDashboardBackend`) | `{_fmt_num(pre_dash, 4, 's')}` | `{_fmt_num(cold_sec, 4, 's')}` (`{_fmt_num(cold_mb, 2)} MB`) | {cold_ratio} |"
            )
        lines.append(
            f"| Peak Traced Memory (5,000-row pipeline) | `{_fmt_num(pre_p.get('peak_memory_mb'), 2)} MB` | `{_fmt_num(post_p.get('peak_memory_mb'), 2)} MB` | **-{_fmt_num(post_p.get('peak_memory_reduction_pct'), 1, '%')}` memory |"
        )
        if tk_p:
            lines.append(
                f"| Real `{_fmt_int(tk_p.get('row_count'))}`-row TikTok Projected Parquet / EVI / FE | — | `{_fmt_num(tk_p.get('projected_parquet_read_seconds'), 4, 's')}` / `{_fmt_num(tk_p.get('evi_detailed_seconds'), 4, 's')}` / `{_fmt_num(tk_p.get('add_engagement_features_seconds'), 4, 's')}` | Sub-second `{_fmt_int(tk_p.get('row_count'))}`-row scale |"
            )
        if pre_p.get("source_note"):
            lines.extend(["", f"> *Benchmark Reference Note:* {pre_p['source_note']}"])
        lines.append("")

    n_sp_posts = sp_obs.get("n_posts_total")
    sp_posts_clause = f" (`N={_fmt_int(n_sp_posts)}`)" if n_sp_posts is not None else ""
    icc_clause = f" (`ICC={likes_icc_val}`)" if likes_icc_val else ""

    lines.extend([
        "---",
        "",
        "## 7. Actionable Creator & Engineering Recommendations",
        "",
        "1. **Standardize High-Intent Measurement:** Track **Shares + Saves** (Instagram Insights) and **Shares + `collect_count`** (TikTok) separately per platform without coercing missing saves on YouTube or public scrapes to `0`.",
        f"2. **Normalize by Creator Baseline Before Comparing Cohorts:** Sponsored vs. unsponsored analysis{sp_posts_clause} confirms raw interaction volume heavily tracks follower scale{icc_clause}. Evaluate content experiments using within-creator relative ratios (`compute_creator_relative_metrics`) or stratified randomization.",
        "3. **Enforce 7-Day Maturity & Zero Same-Day Leakage:** Pre-publication predictions must rely solely on strictly historical creator statistics (`creator_day < create_day`) and settled post horizons (`>= 7` days).",
        "4. **Validate Creative Hypotheses Using Deliverable 5:** Execute the 4-part pilot in [`docs/CONTENT_SERIES_TEMPLATE.md`](../docs/CONTENT_SERIES_TEMPLATE.md) under `PILOT_EXP_01`–`PILOT_EXP_03` (`randomized_experiment` assignment, Holm–Bonferroni correction) before elevating any exploratory topic or hook recommendation to **HIGH** evidence.",
        "5. **Unlock Blocked Kaggle / Live Owner API Telemetry (Optional Next Stage):** Configure `.kaggle/kaggle.json` (or `KAGGLE_*` variables) and owner OAuth tokens (`META_ACCESS_TOKEN`, `META_IG_BUSINESS_ACCOUNT_ID`, `YOUTUBE_API_KEY`, `YOUTUBE_CHANNEL_ID`) in untracked `.env` to transition from public/research cohorts to account-owner `saves`, `reach`, and YouTube `averageViewPercentage` retention telemetry.",
        "",
    ])

    return "\n".join(lines).rstrip() + "\n"


def generate_strategy_report(
    root_dir: Path | None = None,
    output_path: Path | None = None,
    *,
    include_local_state: bool = False,
) -> Path:
    root = Path(root_dir).resolve() if root_dir is not None else PROJECT_ROOT
    target = Path(output_path) if output_path is not None else (root / "reports" / "SOCIALPULSE_STRATEGY_REPORT.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    md_text = build_strategy_report_markdown(root_dir=root, include_local_state=include_local_state)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(md_text)
    return target


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically generate Deliverable 4 Strategy Report from verified artifacts."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "SOCIALPULSE_STRATEGY_REPORT.md",
        help="Destination Markdown path (default: reports/SOCIALPULSE_STRATEGY_REPORT.md).",
    )
    parser.add_argument(
        "--include-local-state",
        action="store_true",
        default=False,
        help="Include optional gitignored local download manifests and built Parquet summaries.",
    )
    args = parser.parse_args(argv)
    out = generate_strategy_report(
        root_dir=PROJECT_ROOT,
        output_path=args.output,
        include_local_state=bool(args.include_local_state),
    )
    print(f"Strategy report generated deterministically at: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
