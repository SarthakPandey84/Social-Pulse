# Deliverable 4 — SocialPulse Comprehensive Strategy & Validation Report

> **Deterministic Artifact Policy & Provenance Integrity:**  
> Every metric in this report is read programmatically from strictly tracked repository sources (`models/real_data_benchmarks.json`, `models/virality_predictor_metadata.json`, `models/relatability_metadata.json`, `models/performance_benchmarks.json`, `data/sample/sample_experiments.csv`, and `DATASET_REGISTRY`), ignoring optional gitignored local caches (`data/external/`, `data/processed/`). Wall-clock generation timestamps are intentionally omitted so regenerating the report yields byte-identical output across clean and data-rich checkouts.

---

## 1. Executive Summary & Evidence Separation Policy

SocialPulse (*The Data-Driven Social Engagement Initiative*) integrates **seven analytical modules** across short-form creator platforms (**Instagram**, **TikTok**, and **YouTube Shorts**). To prevent synthetic demonstrations, weakly labelled scraped sets, and unverified third-party mirrors from being conflated with audited public evidence, every corpus and finding is assigned one of **six explicit governance tiers**:

1. `VERIFIED_COMMERCIAL_SAFE` — Cleared license & verified upstream provenance (`zenodo_sponsored_unsponsored_ig`, `zenodo_educational_reels_ig`, `zenodo_lokananta_comments_ig`); observational confounding caveats still apply (`is_causal=False` unless randomized).
2. `NON_COMMERCIAL_RESEARCH_ONLY` — Verified academic/non-commercial corpora (`tiktok_engagement_200k` under CC-BY-NC-4.0, `instaart_human_ai` academic research); excluded from commercial redistribution.
3. `PENDING_REVIEW_WEAKLY_LABELLED` — Undeclared upstream license or weak/pseudo labels (`instagram_comments_sentiment_hf`); isolated for bounded held-out benchmarking and default-denied by the downloader unless explicitly opted in (`--include-pending-review`).
4. `EXTERNAL_BENCHMARK_UNVERIFIED` — Public tabular benchmarks with synthetic-like regularities or unverified acquisition lineage (`instagram_analytics_github_alt`, `instagram_engagement_eda_hf` [also default-denied `PENDING_REVIEW`]); quarantined from real organic telemetry claims.
5. `SYNTHETIC_BENCHMARK` — Internal synthetic offline fixtures (`synthetic_demo` / `data/sample/*.csv`) and external synthetic tables (`social_instagram_marketing_hf`); demonstrate pipeline mechanics only.
6. `BLOCKED_MISSING_CREDENTIALS` — Gated Kaggle sources (`youtube_trending_kaggle`, `instagram_analytics_kaggle`, `youtube_kaggle_candidate_h`) blocked gracefully without guessing URLs or bypassing authentication.

---

## 2. Authoritative PDF Specification & Technical Stack Traceability

Comparison against `Problem_Statement/Data_Science_Major_Project_December.pdf` (distinguishing verbatim implementations from intentional architectural substitutions):

| PDF Layer | PDF Reference Tools | SocialPulse Implementation | Compliance Status & Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Data Processing** | Python (`Pandas`, `NumPy`) | `pandas==3.0.3`, `numpy==2.4.4`, `pyarrow==24.0.0` | **Exact Match** — Strict nullable `Int64`/`Float64` preservation (zero silent NaN->0 imputation). |
| **Scraping / APIs** | Selenium, BeautifulSoup, Instagram Graph API | `InstagramGraphAPIConnector` (`v21.0` default, `META_*`), `YouTubeDataAPIConnector`, `SocialMediaScraper` (`placeholder`), + pinned HTTPS/Parquet downloader (`external_datasets.py`) | **Partial / Contract Boundary** — Live OAuth tokens & browser scraping excluded for platform ToS & clean offline reproducibility; real validation runs on pinned public archives. |
| **NLP & Text Analysis** | NLTK, TextBlob, SpaCy | Pure Python `TextPreprocessor` (non-English rejection + PII scrubbing/HMAC pseudonymization), lexicon `SentimentAnalyzer`, TF-IDF + Logistic Regression `RelatabilityClassifier` | **Intentional Deterministic Substitution** — Zero runtime NLTK/SpaCy binary downloads; preserves offline portability and deterministic reproducibility. |
| **Database** | MySQL / PostgreSQL | Embedded `sqlite3` (`database/social_analytics.db` via `DataStorageManager`, `latest_posts` SQL view) | **Intentional Local Substitution** — Zero database daemon required for local/Streamlit Community Cloud container execution. |
| **Visualization** | Tableau, PowerBI, Matplotlib/Seaborn | Interactive `plotly==7.1.0` charts embedded in Streamlit (`matplotlib`/`seaborn` dev-only in `requirements-dev.txt`) | **Intentional Interactive Substitution** — Unified browser delivery directly inside Dashboard pages without proprietary BI licenses. |
| **Statistical Modeling** | Scikit-learn, Excel | `scikit-learn==1.9.0` (`Ridge`, `HistGradientBoostingRegressor`, `TfidfVectorizer`, `LogisticRegression`), `scipy==1.17.1`, stdlib XML `.xlsx` reader | **Exact + Upgraded** — Reproducible ML/statistical tests plus stdlib Excel parser (`openpyxl`-free). |
| **Dashboard Frontend** | Streamlit or Dash | `streamlit==1.58.0` (`streamlit_app.py` + `app/pages/01..07`) | **Exact Match** — 7-module multipage app with startup bootstrap and unified dataset switcher. |

### Five PDF Deliverables Status
1. **Deliverable 1 (Analytics Dashboard):** `streamlit_app.py` & `app/pages/01..07` (`GrowthDashboardBackend` + live dataset selector).
2. **Deliverable 2 (Structured Dataset):** Tracked fixtures (`data/sample/*`) + idempotent local build outputs (`data/processed/*`, SQLite `database/social_analytics.db`) + 9 configured non-blocked external Parquet pipelines (normalized on demand into optional gitignored `data/processed/external/`).
3. **Deliverable 3 (NLP Sentiment Model):** `src/nlp/` hybrid pipeline + `scripts/train_and_benchmark_nlp.py` + held-out external evaluations.
4. **Deliverable 4 (Strategy Report):** This document (`reports/SOCIALPULSE_STRATEGY_REPORT.md`), generated deterministically by `scripts/generate_strategy_report.py`.
5. **Deliverable 5 (Content Series):** Pilot series protocol, A/B design, and 14-field unified logging template at [`docs/CONTENT_SERIES_TEMPLATE.md`](../docs/CONTENT_SERIES_TEMPLATE.md).

---

## 3. External & Real-Data Inventory & Governance Matrix

| Dataset ID | Governance Tier | Provenance | License Status | Registry Access & Intended Scope |
| :--- | :--- | :--- | :--- | :--- |
| `tiktok_engagement_200k` | `NON_COMMERCIAL_RESEARCH_ONLY` | `real_public_data` | `CC-BY-NC-4.0` (`NON_COMMERCIAL_CC_BY_NC_4_0`) | Configured (`huggingface`; optional ignored local cache) — modules: `M1, M2, M5, M6, M7` |
| `instagram_engagement_eda_hf` | `EXTERNAL_BENCHMARK_UNVERIFIED` | `external_benchmark_unverified` | `NONE_DECLARED` (`PENDING_REVIEW`) | Default-deny (`PENDING_REVIEW`; requires explicit opt-in) — modules: `M1_benchmark, M2_benchmark, audit` |
| `instagram_analytics_github_alt` | `EXTERNAL_BENCHMARK_UNVERIFIED` | `external_benchmark_unverified` | `MIT` (`VERIFIED_MIT_ORIGIN_UNVERIFIED`) | Configured (`github`; optional ignored local cache) — modules: `M1_benchmark, M2_benchmark, audit` |
| `instaart_human_ai` | `NON_COMMERCIAL_RESEARCH_ONLY` | `real_public_data` | `OTHER_ACADEMIC_RESEARCH` (`RESEARCH_ONLY`) | Configured (`huggingface`; optional ignored local cache) — modules: `M1, M3, M6, M7` |
| `instagram_comments_sentiment_hf` | `PENDING_REVIEW_WEAKLY_LABELLED` | `weakly_labelled_data` | `NONE_DECLARED` (`PENDING_REVIEW`) | Default-deny (`PENDING_REVIEW`; requires explicit opt-in) — modules: `M3` |
| `social_instagram_marketing_hf` | `SYNTHETIC_BENCHMARK` | `synthetic_data` | `CC-BY-4.0` (`VERIFIED`) | Configured (`huggingface`; optional ignored local cache) — modules: `M1_synthetic, M5_synthetic` |
| `zenodo_sponsored_unsponsored_ig` | `VERIFIED_COMMERCIAL_SAFE` | `real_public_data` | `CC-BY-4.0` (`VERIFIED`) | Configured (`zenodo`; optional ignored local cache) — modules: `M1, M4_observational, M5` |
| `zenodo_educational_reels_ig` | `VERIFIED_COMMERCIAL_SAFE` | `real_public_data` | `CC-BY-4.0` (`VERIFIED`) | Configured (`zenodo`; optional ignored local cache) — modules: `M1, M5` |
| `zenodo_lokananta_comments_ig` | `VERIFIED_COMMERCIAL_SAFE` | `real_public_data` | `CC-BY-4.0` (`VERIFIED`) | Configured (`zenodo`; optional ignored local cache) — modules: `M3_multilingual_edge` |
| `youtube_trending_kaggle` | `BLOCKED_MISSING_CREDENTIALS` | `real_public_data` | `CC0-1.0_KAGGLE` (`REQUIRES_KAGGLE_AUTH`) | Blocked (`REQUIRES_KAGGLE_AUTH`, slug `rsrishav/youtube-trending-video-dataset` awaits Kaggle auth) — modules: `M1, M2, M6, M7` |
| `instagram_analytics_kaggle` | `BLOCKED_MISSING_CREDENTIALS` | `external_benchmark_unverified` | `UNKNOWN_REQUIRES_KAGGLE_AUTH` (`REQUIRES_KAGGLE_AUTH`) | Blocked (`REQUIRES_KAGGLE_AUTH`, slug `kundanbedmutha/instagram-analytics-dataset` awaits Kaggle auth) — modules: `M1_benchmark, audit` |
| `youtube_kaggle_candidate_h` | `BLOCKED_MISSING_CREDENTIALS` | `external_benchmark_unverified` | `UNKNOWN` (`BLOCKED_MISSING_CREDENTIALS_AND_UNVERIFIED_SLUG`) | Blocked (`BLOCKED_MISSING_CREDENTIALS_AND_UNVERIFIED_SLUG`, slug unverified) — modules: `audit` |

> **Tracked Registry Contract vs. Local Host Artifacts:** Table rows report the tracked `DATASET_REGISTRY` contract identically across clean and data-rich checkouts. Pass `--include-local-state` when running `scripts/generate_strategy_report.py` locally to include optional gitignored host state from `data/external/manifest.json` and `data/processed/`.

---

## 4. Empirical Findings on Real Public & Academic Data (`models/real_data_benchmarks.json`)

### 4.1 Module 1 — Schema Honesty, Proxy Separation & Creator-Relative Scaling
- **TikTok 200K Panel (`tiktok_engagement_200k`):** Evaluated across `209,543` video snapshots (`1,872` creators with `>= 5` posts). Median creator-relative view ratio = `1.0000`, 90th percentile = `3.1580`.
- **Proxy & Missingness Isolation:** TikTok exposes `views`, `likes`, `comments_count`, `shares`, and explicit `collect_count` bookmark proxy (`100.0%` observed), while Instagram-native `saves`, `reach`, `impressions`, and `retention_rate` stay `<NA>` (never coerced to zero).
- **Zenodo 5082114 (`zenodo_sponsored_unsponsored_ig`):** Projected schema preserves `follower_count` (`57,840` / `99.37%` coverage) and provenance tags `['real_public_data']`; unexposed `views`, `shares`, `saves`, `reach` remain explicit `<NA>`.

### 4.2 Module 2 — Longitudinal Virality Modeling Without Same-Day / Future Leakage

All runs predict `log1p(day30_views)` or `log1p(day7_views)` using historical creator features joined strictly from `creator_day < create_day` (eliminating same-day/future creator counter leakage):

| Validation Regime | Target | Feature Regime | Split Strategy | Creator Overlap | Test N | Best Model | R² (`log1p`) | Spearman rho | RMSE Lift vs Dummy | 90% Interval Cov |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `day30_pre_pub_chronological` | `day30_views` | `pre_pub` | `chronological` | 1,526 | 14,788 | `hist_gradient_boosting` | 0.635 | 0.780 | 43.96% | 89.57% |
| `day30_pre_pub_creator_grouped` | `day30_views` | `pre_pub` | `creator_grouped` | 0 | 13,636 | `hist_gradient_boosting` | 0.600 | 0.784 | 40.18% | 95.26% |
| `day30_early_day1_chronological` | `day30_views` | `early_day1` | `chronological` | 1,526 | 14,788 | `hist_gradient_boosting` | 0.915 | 0.960 | 72.99% | 90.79% |
| `day30_early_day1_creator_grouped` | `day30_views` | `early_day1` | `creator_grouped` | 0 | 13,636 | `hist_gradient_boosting` | 0.922 | 0.961 | 73.68% | 92.58% |
| `day7_pre_pub_creator_grouped` | `day7_views` | `pre_pub` | `creator_grouped` | 0 | 45,303 | `hist_gradient_boosting` | 0.545 | 0.742 | 36.35% | 85.06% |

- **Key Takeaway:** Zero-overlap creator-grouped splitting confirms true generalization across unseen creators (`R² = 0.600`, `Spearman rho = 0.784` pre-publication vs `R² = 0.635` when creators overlap chronologically; adding observed Day-1 signals lifts `day30_views` `R²` to `0.922`).

### 4.3 Module 3 — Held-Out Sentiment Generalization & Non-English Guardrails

- **Held-Out Instagram Sentiment (`instagram_comments_sentiment_hf`, `PENDING_REVIEW_WEAKLY_LABELLED`):** Train `N=200`, Full Held-Out Test `N=56` (`4` exact text duplicates overlap train/test -> strict-unique test `N=52`). TF-IDF + Logistic Regression achieves `0.554` Accuracy / `0.556` Macro F1 full (`0.538` / `0.518` strict-unique; ECE `0.083`), beating rule-based lexicon accuracy (`0.352` on `54/56` non-abstained rows). Demonstrates clearly that small weak-label generalization (`55.4%`) is materially harder than closed synthetic templates (~`90.3%`).
- **Non-English Safety Abstention (`zenodo_lokananta_comments_ig`, `VERIFIED_COMMERCIAL_SAFE`, `N=552` Indonesian comments):** Monolingual English pipeline abstained on `100.00%` when given `language_hint='id'` (`552/552` null labels) and abstained cleanly on `15.58%` hint-free (`15.58%` explicit `UNSUPPORTED_LANGUAGE` without false neutral coercion).
- **InstaArt Comments (`instaart_human_ai`, `NON_COMMERCIAL_RESEARCH_ONLY`, `N=2,810` comments across `79` Human/AI posts):** Status breakdown and rank-vs-chronological comment indexing preserved without fabricated saves/views.

### 4.4 Module 4 — Creator-Clustered Confounding Audit (`zenodo_sponsored_unsponsored_ig`)

- **Cohort Size & Clustering:** `58,204` posts across `40,170` creators (`is_causal=False`). Intra-creator ICC on likes is high (`0.945`, `DEFF=1.42`), showing why naive post-level independence assumptions inflate significance.
- **Metric `likes (creator-clustered)` (Creator-Level Mean Aggregation (mean posts/creator=1.45, est. ICC=0.945, DEFF=1.42)):** Sponsored mean = `599.45` vs. Unsponsored mean = `55.64` (`977.36%`, `p_adj=0.0`). Raw interaction volume reflects creator scale differences rather than causal sponsorship lift.
- **Metric `comments_count (creator-clustered)` (Creator-Level Mean Aggregation (mean posts/creator=1.45, est. ICC=0.872, DEFF=1.39)):** Sponsored mean = `35.67` vs. Unsponsored mean = `2.88` (`1138.10%`, `p_adj=0.0`). Raw interaction volume reflects creator scale differences rather than causal sponsorship lift.
- **Metric `likes_per_1k_followers (creator-clustered)` (Creator-Level Mean Aggregation (mean posts/creator=1.45, est. ICC=0.815, DEFF=1.36)):** Sponsored mean = `70.80` vs. Unsponsored mean = `75.06` (`-5.68%`, `p_adj=0.02543`). Direction **reverses** after follower normalization, demonstrating creator-scale confounding in unadjusted sponsored comparisons.

### 4.5 Module 7 — Walk-Forward Daily Cohort Forecasting (`tiktok_engagement_200k`)

| Daily Series | Contiguous History Span | Champion Model | MASE | Rel-MAE vs Naive | Validation Origins |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `daily_views` | `2024-06-24` to `2024-09-22` (`91` pts) | `AutoRegressive Ridge (AR)` | 0.817 | 0.812 | `10` |
| `daily_collects` | `2024-06-24` to `2024-09-22` (`91` pts) | `Moving Average` | 0.560 | 0.586 | `10` |

---

## 5. Offline Synthetic Benchmark Summary (`SYNTHETIC_BENCHMARK` — Mechanics Only)

> **Methodological Quarantine:** Metrics in this section quantify synthetic pipeline correctness on checked-in fixtures (`data/sample/*.csv`) and MUST NOT be reported as empirical organic performance.

1. **Module 2 Ridge Regressor (`models/virality_predictor_metadata.json`):**
   - Settled mature sample: `112` posts (`89` train / `23` chronological test).
   - Candidate vs. Naive Mean Baseline: MAE `20.91` vs `35.41` (`-40.9%`), RMSE `27.48` vs `38.52` (`-28.7%`), Spearman `rho = 0.492` (`p = 0.017`), empirical `q95 = ±49.99` pts.
   - **Platform-Stratified Caveats Disclosed in UI:** Pooled/YouTube chronological `R² = -0.136` (negative variance explained on small `N = 23` test slice), and Instagram has `test_samples = 0` (`status = insufficient_evaluation_samples`).
2. **Module 3 Hybrid Relatability (`models/relatability_metadata.json`):**
   - Repeated split (`N=114`): Hybrid Accuracy `0.9035`, Macro F1 `0.9026`.
   - Strict-unique split (`N=28`): Hybrid Accuracy `0.8929`, Macro F1 `0.8927`.
   - **Circularity Disclosure:** Benchmark rules encode author domain knowledge of synthetic phrasing and uncalibrated confidence thresholds; real-world expectation should align with Section 4.3.
3. **Module 4 Synthetic Experiments (`data/sample/sample_experiments.csv`):**
   - `EXP_001` (instagram · `randomized_experiment`): `Reels Hook Architecture: Story Hook vs Visual Hook` -> primary metric `share_rate` (`Story Hook` vs `Visual Hook`).
   - `EXP_002` (youtube · `randomized_experiment`): `YouTube Shorts Title Framing: Question vs Bold Statement` -> primary metric `views` (`Question Hook` vs `Bold Statement Hook`).
   - `EXP_003` (instagram · `randomized_experiment`): `Caption Depth: Short Punchy vs Long Vulnerable Story` -> primary metric `relatability_pct` (`Short Punchy Caption` vs `Long Vulnerable Story`).
   - `EXP_004` (instagram · `observational_comparison`): `Posting Window Cohort Comparison: Weekday Morning vs Weekend Evening` -> primary metric `views` (`Weekday Morning (8-10 AM)` vs `Weekend Evening (6-8 PM)`).
4. **Module 6 Audience Correlation Fixture:** Post-level relatability-to-saves Spearman correlation is evaluated on optional ignored build artifacts (`data/processed/processed_*.parquet` from `scripts/build_all_artifacts.py`; pass `--include-local-state` to embed host runtime correlation metrics), and passes through Module 4 controlled tests before causal interpretation.

---

## 6. Runtime Vectorization & Memory Engineering (`models/performance_benchmarks.json`)

| Workload | Historical Pre-Batch-5A Reference | Measured (Current HEAD) | Derived Ratio / Note |
| :--- | :---: | :---: | :---: |
| 5,000-row EVI (`calculate_engagement_virality_index_detailed`) | `1.8833s` | `0.0324s` (`2.39 MB`) | **58.1x faster** |
| 5,000-row Feature Engineering (`add_engagement_features`) | `3.3271s` | `0.0286s` (`5.13 MB`) | **116.3x faster** |
| Dashboard Synthesis — Warm Backend (`GrowthDashboardBackend`) | `0.4129s` | `0.1692s` (`0.44 MB`) | **2.4x faster** |
| Dashboard Init + Synthesis — Cold Load (`GrowthDashboardBackend`) | `0.4129s` | `0.2170s` (`0.58 MB`) | **1.90x faster** |
| Peak Traced Memory (5,000-row pipeline) | `15.53 MB` | `5.13 MB` | **-67.0%` memory |
| Real `209,543`-row TikTok Projected Parquet / EVI / FE | — | `0.0274s` / `0.2905s` / `0.5358s` | Sub-second `209,543`-row scale |

> *Benchmark Reference Note:* Historical pre-Batch-5A baseline recorded prior to vectorization on a local Python 3.11.9 workstation (fixed reference for comparison ratios; not re-executed on HEAD)

---

## 7. Actionable Creator & Engineering Recommendations

1. **Standardize High-Intent Measurement:** Track **Shares + Saves** (Instagram Insights) and **Shares + `collect_count`** (TikTok) separately per platform without coercing missing saves on YouTube or public scrapes to `0`.
2. **Normalize by Creator Baseline Before Comparing Cohorts:** Sponsored vs. unsponsored analysis (`N=58,204`) confirms raw interaction volume heavily tracks follower scale (`ICC=0.945`). Evaluate content experiments using within-creator relative ratios (`compute_creator_relative_metrics`) or stratified randomization.
3. **Enforce 7-Day Maturity & Zero Same-Day Leakage:** Pre-publication predictions must rely solely on strictly historical creator statistics (`creator_day < create_day`) and settled post horizons (`>= 7` days).
4. **Validate Creative Hypotheses Using Deliverable 5:** Execute the 4-part pilot in [`docs/CONTENT_SERIES_TEMPLATE.md`](../docs/CONTENT_SERIES_TEMPLATE.md) under `PILOT_EXP_01`–`PILOT_EXP_03` (`randomized_experiment` assignment, Holm–Bonferroni correction) before elevating any exploratory topic or hook recommendation to **HIGH** evidence.
5. **Unlock Blocked Kaggle / Live Owner API Telemetry (Optional Next Stage):** Configure `.kaggle/kaggle.json` (or `KAGGLE_*` variables) and owner OAuth tokens (`META_ACCESS_TOKEN`, `META_IG_BUSINESS_ACCOUNT_ID`, `YOUTUBE_API_KEY`, `YOUTUBE_CHANNEL_ID`) in untracked `.env` to transition from public/research cohorts to account-owner `saves`, `reach`, and YouTube `averageViewPercentage` retention telemetry.
