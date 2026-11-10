# Module 7: Trend Forecasting Engine Documentation

**Product:** *SocialPulse — An Intelligent Social Media Analytics & Engagement Optimization Platform*  
**Module:** 7 — Trend Forecasting Engine  
**Status:** Audited, Optimized, UI-Integrated & Verified (41/41 Module 7 Tests Passing; 273/273 Total Repository Tests Passing)

---

## 1. Executive Overview & Methodological Corrections

Following two rigorous independent methodological audits, Module 7 was enhanced to resolve all discrepancies between documented time-series methodology and actual implementation.

### Key Methodological Resolutions (First & Second Audits):

1. **Publication-Cohort Exposure-Age Bias Disclosure:**
   - In the project dataset, post publication dates span from September 1 to October 31, 2026, while snapshot collection occurred on November 2–3, 2026.
   - Older posts accumulated cumulative views over up to ~63 days, while late-October posts accumulated views over only ~2–3 days.
   - Intermediate daily tracking snapshots (e.g., day-1, day-7 view accrual curves) do not exist in the source dataset, making mathematical age standardization impossible without unsupported extrapolation.
   - The engine explicitly designates this target as **exploratory descriptive analysis** and surfaces prominent disclosures in both engine reports and dashboard views.

2. **Unverified Source Coverage Policy:**
   - Previous claims of 100% ingestion completeness inferred from min-to-max date spans were methodologically invalid because a continuous calendar range does not prove the absence of dropped or unobserved scrape records.
   - Source coverage for post cohorts is now formally designated as `UNVERIFIED` (`coverage_pct = None`, `missing_periods = None`), preserving zero-publication periods under an explicit stated assumption rather than claiming verified completeness.

3. **Strict Engagement-Rate NaN Handling (No False Zero Imputation):**
   - On zero-post days (`post_count = 0`), views are zero and engagement rate is mathematically undefined ($0/0$).
   - Previous `fillna(0.0)` imputation created false 0.0% engagement drops that distorted autocorrelation and variance estimation.
   - `forecast_metric()` now strictly verifies that undefined engagement rates cannot be forecasted as regular time series without domain-grounded aggregation, raising an explanatory `ValueError` that directs users to raw interaction metrics or coarser cadences.

4. **Platform Isolation & Interaction Semantics (Instagram, YouTube, TikTok):**
   - Instagram, YouTube, and TikTok are strictly modeled separately. Cross-platform pooling of raw volume metrics is prohibited (`allow_tiktok=True` enables explicit single-platform TikTok mode).
   - YouTube does not track post saves in its public API, and TikTok `collect_count` is a bookmark proxy rather than Instagram Insights `saves`. The engine eliminates `cohort_saves = 0` from YouTube/TikTok metrics and isolates `youtube_available_interactions` / `tiktok_available_interactions` (`Likes + Comments + Shares`) plus explicit `cohort_collects` where available.

5. **Short Keyword History Handling ($N < 17$ Safeguard):**
   - With a minimum training size of 14 observations and a requirement of $K \ge 3$ validation origins, a time series requires at least $N = 17$ observations ($17 - 14 = 3$).
   - Short series of lengths 14, 15, and 16 previously crashed with `IndexError: list index out of range` in `forecast_keyword_trend()` due to an unhandled empty evaluation list.
   - The engine now detects empty evaluation results and safely returns `ValidationStatus.INSUFFICIENT_VALIDATION_HISTORY` with zero forecast points and clear disclosures, while length 17 smoothly evaluates 3 rolling origins.

6. **Validation Boundary Instrumentation & Horizon Uncertainty Labeling:**
   - Evaluator boundaries are exposed via `get_rolling_origin_diagnostics()` to verify zero temporal leakage across all origins (`train_end < test_start` and disjoint index sets).
   - In uncertainty quantification, horizon steps $j \le H_{\text{eval}}$ with $n_j \ge 3$ are labeled `"Direct Empirical Quantile (90th percentile out-of-sample residual)"`, while steps $j > H_{\text{eval}}$ with $n_{H_{\text{eval}}} \ge 3$ are labeled `"Extrapolated Horizon Range — Evaluation-Horizon Expansion (...)"`.
   - Steps with insufficient empirical residuals ($n < 3$) use `"Step-1 Expansion Fallback"` ($\text{Spread}_1 \times \sqrt{s}$ when $n_1 \ge 3$) or `"MAE Fallback Spread"` ($\text{MAE} \times 1.645 \times \sqrt{s}$ when $n_1 < 3$) and are never mislabeled as direct empirical quantiles.
   - All intervals are presented with prominent caveats disclosing their uncalibrated exploratory status, and estimates based on $n < 10$ residuals are flagged as unstable.

7. **Model-Selection Optimism Bias Disclosure:**
   - Evaluator discloses that the champion model is selected using the exact same rolling-origin errors subsequently reported, meaning reported MAE/RMSE is selection-dependent and not an independent unbiased future generalization estimate.

8. **Observed Calendar-Date Continuity for Keywords:**
   - Keyword time-series validation measures "Observed Calendar-Date Continuity", explicitly disclosing that upstream source/API scrape completeness remains unverified without polling logs. Misleading claims of confirmed complete scrape coverage have been eliminated.

---

## 2. Architecture & Component Flow

```mermaid
flowchart TD
    subgraph Data Sources
        POSTS[(processed_posts.parquet<br/>120 posts, 61 days)]
        TRENDS[(sample_trends.csv<br/>170 records, 5 topics)]
    end

    subgraph Data Preparation & Validation
        DEDUP[Snapshot Deduplication<br/>get_latest_posts_view()]
        RESAMPLE[Publication-Cohort Resampling<br/>Daily 'D' or Weekly 'W']
        PLAT_GUARD[Platform Isolation Guard<br/>Block Cross-Platform Raw Pooling & YT Saves]
        MISS_GUARD[Missingness & Coverage Audit<br/>Distinguish Zero-Posts from Missing Data]
    end

    subgraph Candidate Models
        M_NAIVE[Naive Persistence<br/>y_hat = y_T]
        M_SNAIVE[Seasonal Naive<br/>m=7 Daily, m=4 Weekly]
        M_MA[Moving Average<br/>window k=7]
        M_HOLT[Holt Exponential Smoothing<br/>L-BFGS-B SSE Optimization]
        M_AR[AutoRegressive Ridge<br/>p lags, L2 alpha=1.0]
    end

    subgraph Rolling-Origin Walk-Forward Engine
        ROLL_SPLIT[K Rolling Forecast Origins<br/>Train: 0..t, Test: t..t+h]
        METRICS[Error Metrics Engine<br/>MAE, RMSE, MASE, Rel-MAE]
        CHAMP[Champion Selector<br/>Lowest Out-of-Sample MAE]
        SPREAD[Horizon-Specific Empirical Quantiles<br/>90th Percentile |e_j|]
    end

    subgraph Interactive Presentation Layer
        APP_PAGE[app/pages/07_Trend_Forecasting.py]
        SEC_A[Section A: Overview, Target Semantics & Coverage]
        SEC_B[Section B: Historical Cohort Visualizer]
        SEC_C[Section C: Forecast Fan Chart & CSV Export]
        SEC_D[Section D: Rolling-Origin MASE Comparison Table]
        SEC_E[Section E: Emerging Struggle Keywords Tracking]
        SEC_F[Section F: Methodological Formulations]
    end

    POSTS --> DEDUP --> RESAMPLE --> PLAT_GUARD --> MISS_GUARD
    TRENDS --> MISS_GUARD
    MISS_GUARD --> ROLL_SPLIT
    ROLL_SPLIT --> M_NAIVE & M_SNAIVE & M_MA & M_HOLT & M_AR
    M_NAIVE & M_SNAIVE & M_MA & M_HOLT & M_AR --> METRICS --> CHAMP --> SPREAD
    SPREAD --> SEC_A & SEC_B & SEC_C & SEC_D & SEC_E & SEC_F --> APP_PAGE
```

---

## 3. Mathematical Formulations & Validation Design

### 3.1 Publication-Cohort Formulation
One observation at interval $t$ represents the total cumulative lifetime views accrued by content published during interval $t$:
$$y_t = \sum_{i \in \text{Posts}_t} \text{Views}_i$$
For intervals with zero post uploads ($\text{post count} = 0$), $y_t = 0.0$.
For rate metrics:
$$\text{Rate}_t = \begin{cases} \frac{\text{Interactions}_t}{\text{Views}_t} \times 100\% & \text{if } \text{Views}_t > 0 \\ \text{NaN (Undefined)} & \text{if } \text{Views}_t = 0 \end{cases}$$

### 3.2 Rolling-Origin Walk-Forward Validation
For a series of length $N$, minimum training length $N_{\text{min}} = 14$, and forecast horizon $h$:
The validation engine iterates across $K = \min(10, N - N_{\text{min}})$ origins $t \in \{N - K, \dots, N - 1\}$:
1. **Training Split:** $y_{\text{train}} = y[0:t]$ (strictly past observations).
2. **Evaluation Window:** $y_{\text{test}} = y[t : \min(N, t + H_{\text{eval}})]$ of length $h_t$.
3. **Out-of-Sample Residuals:** For each model $m$ and step $j \in \{1 \dots h_t\}$:
   $$e_{t, j}^{(m)} = y_{\text{test}}[j-1] - \hat{y}_{t, j}^{(m)}$$
4. **Aggregate Error Metrics across All Origins:**
   - $\text{MAE} = \frac{1}{\sum h_t}\sum_{t, j} |e_{t, j}|$
   - $\text{RMSE} = \sqrt{\frac{1}{\sum h_t}\sum_{t, j} (e_{t, j})^2}$
   - $\text{MASE} = \frac{\text{MAE}}{\frac{1}{K}\sum_{k=1}^K S_k}$ where $S_k = \frac{1}{t_k-1}\sum_{i=1}^{t_k-1}|y_i - y_{i-1}|$ is in-sample naive scale.
   - $\text{RelMAE} = \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{naive}}}$.

### 3.3 Horizon-Specific Empirical Uncertainty Spreads & Method Labeling
Every evaluated candidate tracks its own per-step out-of-sample spreads (`step_error_spreads`), method labels (`step_error_methods`), residual counts (`step_error_counts`), and origin counts (`successful_origins`, `failed_origins`). Across forecast steps $s \in \{1 \dots H\}$ (validation cutoff $H_{\text{eval}} = \min(H, 5)$), `compute_horizon_uncertainty()` assigns one of four mutually distinct paths:

1. **Direct Empirical Quantile** (when $s \le H_{\text{eval}}$ and $n_s \ge 3$ residual observations):
   $$\text{Spread}_s = \text{Percentile}_{90}\left(\{|e_{t, s}| \mid \text{origins } t\}\right)$$
   - Method label: `"Direct Empirical Quantile (90th percentile out-of-sample residual)"`
2. **Evaluation-Horizon Expansion** (when $s > H_{\text{eval}}$ and $n_{H_{\text{eval}}} \ge 3$):
   $$\text{Spread}_s = \text{Spread}_{H_{\text{eval}}} \times \sqrt{\frac{s}{H_{\text{eval}}}}$$
   - Method label: `"Extrapolated Horizon Range — Evaluation-Horizon Expansion (...)"`
3. **Step-1 Expansion Fallback** (when step-$s$ or cutoff residuals have $n < 3$, and $n_1 \ge 3$):
   $$\text{Spread}_s = \text{Spread}_1 \times \sqrt{s}$$
   - Method label: `"Step-1 Expansion Fallback (...)"` *(never mislabeled as Direct Empirical Quantile)*
4. **MAE Fallback Spread** (when even step-1 has $n_1 < 3$):
   $$\text{Spread}_s = \text{MAE} \times 1.645 \times \sqrt{s}$$
   - Method label: `"MAE Fallback Spread (...)"`

- **Model-Specific Spreads on Manual Overrides:** When a manual `selected_model` override is eligible (`is_eligible_for_champion=True`, `failed_origins == 0`), forecast intervals use **that selected model's own** out-of-sample residual spreads (`MANUAL OVERRIDE UNCERTAINTY`). If an override candidate was excluded from champion eligibility (`failed_origins > 0` or insufficient origins), the engine emits `MANUAL OVERRIDE VALIDATION WARNING` and keeps empirical spreads anchored to the validated champion.

The prediction interval enforces a non-negative floor:
$$\text{CI}_s = \left[ \max(0.0, \hat{y}_{T+s} - \text{Spread}_s), \quad \hat{y}_{T+s} + \text{Spread}_s \right]$$

**Calibration Caveat:** These bounds represent empirical out-of-sample error spreads under observational stationarity. Nominal 95% coverage is **NOT calibrated or mathematically guaranteed**. When residual count $n < 10$, percentiles are exploratory and subject to high sampling variance.

### 3.4 Model-Selection Optimism Bias Disclosure
Candidate models (Naive, Seasonal Naive, Moving Average, Holt Exponential Smoothing, AutoRegressive Ridge) are evaluated across all rolling origins, and the champion model is selected as the one minimizing aggregate out-of-sample MAE. Because the selection criterion and reported evaluation metrics are computed on the same walk-forward validation observations, the champion's reported performance exhibits selection optimism. Reported errors must be interpreted as performance on the historical rolling evaluation set, not as an unbiased out-of-sample generalization estimate from an independent holdout.

---

## 4. Key Performance Benchmark Results

### 4.1 Content Performance Metrics (`processed_posts.parquet`)

- **Platform:** Instagram (Isolated)
- **Cadence:** Daily ('D')
- **Historical Scope:** 43 calendar days (38 active upload days, 5 zero-publication days)
- **Coverage Status:** `UNVERIFIED (Continuous calendar span does not prove absence of dropped scrape records)`
- **Walk-Forward Validation:** 10 rolling forecast origins, 40 out-of-sample prediction evaluations
- **Target Metric:** `cohort_views`
- **Candidate Model Evaluations (`horizon = 7` default):**
  - **Moving Average (Champion):** MAE = 12,296.65, RMSE = 15,694.61, MASE = 0.635, Relative MAE = **0.764** (23.6% error reduction vs persistence)
  - **AutoRegressive Ridge (AR):** MAE = 13,415.14, RMSE = 15,630.48, MASE = 0.693, Relative MAE = 0.834
  - **Holt Exponential Smoothing:** MAE = 13,844.82, RMSE = 16,830.57, MASE = 0.715, Relative MAE = 0.860
  - **Naive (Persistence):** MAE = 16,094.60, RMSE = 20,070.34, MASE = 0.831, Relative MAE = 1.000
  - **Seasonal Naive:** MAE = 21,918.35, RMSE = 30,247.01, MASE = 1.132, Relative MAE = 1.362

### 4.2 Real-Data TikTok Daily Increment Panel (`tiktok_engagement_200k`)

Using `forecast_daily_increment_series(..., segment_policy="longest_contiguous")` over `daily_platform_summary.parquet` (isolated strictly to `tiktok`, selecting the **91-day contiguous gap-free segment** `2024-06-24` to `2024-09-22` out of `159` days with `10` reset/gap days flagged explicitly in the full span, `10` rolling origins / `40` test points):

- **Target `daily_views`:** Champion = **`AutoRegressive Ridge (AR)`** (`MAE = 2,412,191.36`, `RMSE = 3,024,993.23`, `MASE = 0.817`, Relative MAE = **`0.812`** — `18.8%` error reduction vs persistence).
- **Target `daily_collects`:** Champion = **`Moving Average`** (`MAE = 10,822.08`, `RMSE = 13,788.45`, `MASE = 0.560`, Relative MAE = **`0.586`** — `41.4%` error reduction vs persistence).

---

## 5. Automated Regression Test Suite

All 41 unit, regression, optimization, and audit tests in `tests/test_trend_forecaster.py` pass cleanly:

```text
tests/test_trend_forecaster.py::test_trend_velocity_computation PASSED
tests/test_trend_forecaster.py::test_identify_rising_struggles_filtering PASSED
tests/test_trend_forecaster.py::test_rolling_origin_validation_multiple_origins_and_refitting PASSED
tests/test_trend_forecaster.py::test_no_temporal_leakage_between_origins PASSED
tests/test_trend_forecaster.py::test_horizon_specific_out_of_sample_residual_collection PASSED
tests/test_trend_forecaster.py::test_insufficient_validation_history_guard PASSED
tests/test_trend_forecaster.py::test_forecasting_target_publication_cohort_semantics PASSED
tests/test_trend_forecaster.py::test_missing_observations_vs_no_post_periods PASSED
tests/test_trend_forecaster.py::test_incomplete_coverage_guard_on_sparse_keywords PASSED
tests/test_trend_forecaster.py::test_platform_incompatibility_pooling_prohibited PASSED
tests/test_trend_forecaster.py::test_youtube_saves_reach_unavailable_guard PASSED
tests/test_trend_forecaster.py::test_missing_or_invalid_platform_metadata_raises_error PASSED
tests/test_trend_forecaster.py::test_weekly_cadence_string_handling_case_insensitive PASSED
tests/test_trend_forecaster.py::test_zero_and_negative_forecast_horizons_raise_error PASSED
tests/test_trend_forecaster.py::test_zero_naive_validation_mae_handling PASSED
tests/test_trend_forecaster.py::test_manual_model_selection_reporting_distinct_from_champion PASSED
tests/test_trend_forecaster.py::test_prediction_interval_construction_and_calibration_disclosures PASSED
tests/test_trend_forecaster.py::test_holt_optimizer_failure_graceful_fallback PASSED
tests/test_trend_forecaster.py::test_parquet_engine_availability_and_csv_fallback PASSED
tests/test_trend_forecaster.py::test_forecast_metric_end_to_end_on_processed_posts PASSED
tests/test_trend_forecaster.py::test_forecast_keyword_trend_end_to_end_on_sample_trends PASSED
tests/test_trend_forecaster.py::test_short_keyword_history_handling_consecutive_lengths PASSED
tests/test_trend_forecaster.py::test_engagement_rate_nan_handling_rejected PASSED
tests/test_trend_forecaster.py::test_youtube_interaction_semantics_and_available_interactions PASSED
tests/test_trend_forecaster.py::test_post_time_series_unverified_coverage_and_exposure_age_warning PASSED
tests/test_trend_forecaster.py::test_uncertainty_direct_vs_extrapolated_steps PASSED
tests/test_trend_forecaster.py::test_exactly_17_observations_3_step_forecast PASSED
tests/test_trend_forecaster.py::test_short_history_fallback_uncertainty_never_mislabeled_as_direct_quantile PASSED
tests/test_trend_forecaster.py::test_keyword_calendar_continuity_terminology_and_unverified_ingestion PASSED
tests/test_trend_forecaster.py::test_model_selection_optimism_disclosure_present PASSED
tests/test_trend_forecaster.py::test_uncertainty_formula_synchronization_sqrt_expansion PASSED
tests/test_trend_forecaster.py::test_single_failed_origin_excluded_from_champion_and_no_synthetic_residuals PASSED
tests/test_trend_forecaster.py::test_multiple_failed_origins_insufficient_and_unavailable_states PASSED
tests/test_trend_forecaster.py::test_failed_origins_do_not_contaminate_residual_counts_or_metrics PASSED
tests/test_trend_forecaster.py::test_uncertainty_methods_and_reconciled_description_all_four_paths PASSED
tests/test_trend_forecaster.py::test_forecaster_reports_all_models_failed_without_falling_back_to_eval_zero PASSED
tests/test_trend_forecaster.py::test_holt_vectorized_predict_equivalence PASSED
tests/test_trend_forecaster.py::test_ar_ridge_sliding_window_and_predict_equivalence PASSED
tests/test_trend_forecaster.py::test_seasonal_naive_vectorized_predict_equivalence PASSED
tests/test_trend_forecaster.py::test_prepare_post_time_series_and_velocity_edge_cases PASSED
tests/test_trend_forecaster.py::test_manual_override_uses_own_residual_spreads_instead_of_champion PASSED
```

**Full Repository Regression Suite:** **273 passed across Modules 1–7, external adapters/quality pipeline, PDF compliance & governance hardening, vectorization parity, UI shell/security/multi-dataset switching, shared caching/startup bootstrap, NLP/metadata hardening, and artifact build verification.**

---

## 6. Verification & Hardening History

An earlier Phase 4.2 / Phase A 21-file baseline lock table has been superseded by subsequent owner-directed engineering phases (**Phases B2, B2.2, B3 Batches 1–4.1, B4 Batches 1–3, and the Master Multi-Platform Upgrade**), which introduced targeted analytical, lifecycle, performance, external-corpus, and deployment-readiness updates across modules and UI pages:

- **Analytical & Lifecycle Hardening (B2 / B2.2 / B3 Batch 1):** Fixed resource closure in `SQLiteStorage`, preserved platform/metric semantics across ingestion and dashboard aggregations, added stratified `platform_evaluation_metrics` in Module 2 (`virality_predictor_metadata.json`), and added model-specific candidate residual spreads (`step_error_spreads`) for eligible manual overrides in Module 7.
- **Repository Consolidation & Optimization (B3 Batches 2–4.1 & Batch 5A):** Removed superseded forwarding pages, unused external YAML configuration helpers, and ad-hoc diagnostic scripts; batched NLP model inference in Module 3; vectorized feature engineering and EVI calculations; and centralized shared `@st.cache_data` / `@st.cache_resource` caching in `app/components/data_loader.py` (with signature-based cache invalidation).
- **Multi-Platform External Integration & Daily Increment Mode:** Added fail-loud external corpus switching (`sp_dataset_selector_p7`), strict TikTok platform isolation (`allow_tiktok=True`, `tiktok_available_interactions`), daily increment forecasting via `forecast_daily_increment_series` (`segment_policy="longest_contiguous"`), and graceful messages for static cross-sectional datasets without timestamps (`zenodo_sponsored_unsponsored_ig`).
- **Cloud Deployment Readiness (B4 Batches 1–3):** Added clean-checkout startup artifact bootstrapping (`ensure_artifacts_ready()`) in root `streamlit_app.py` and pinned exact Python 3.11+ runtime dependencies in `requirements.txt`.

Current verification rests on deterministic builds (`python scripts/build_all_artifacts.py --validate-only`), real-data benchmarks (`models/real_data_benchmarks.json`), and the full automated pytest suite (**41/41** in `tests/test_trend_forecaster.py`; **273/273** repository-wide).
