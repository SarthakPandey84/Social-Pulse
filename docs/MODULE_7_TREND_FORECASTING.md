# Module 7: Trend Forecasting Engine Documentation

**Product:** *SocialPulse — An Intelligent Social Media Analytics & Engagement Optimization Platform*  
*(Unlox Data-Driven Social Engagement Initiative)*  
**Module:** 7 — Trend Forecasting Engine  
**Status:** Audited, Optimized, UI-Integrated & Verified (40/40 Module 7 Tests Passing; 196/196 Total Repository Tests Passing)

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

4. **Platform Isolation & YouTube Interaction Semantics:**
   - Instagram and YouTube are strictly modeled separately. Cross-platform pooling of raw volume metrics is prohibited.
   - YouTube does not track post saves in its public API. The engine eliminates `cohort_saves = 0` from YouTube metrics, rejects the 4-component `cohort_total_engagement` on YouTube, and provides the 3-component `youtube_available_interactions` (Likes + Comments + Shares).

5. **Short Keyword History Handling ($N < 17$ Safeguard):**
   - With a minimum training size of 14 observations and a requirement of $K \ge 3$ validation origins, a time series requires at least $N = 17$ observations ($17 - 14 = 3$).
   - Short series of lengths 14, 15, and 16 previously crashed with `IndexError: list index out of range` in `forecast_keyword_trend()` due to an unhandled empty evaluation list.
   - The engine now detects empty evaluation results and safely returns `ValidationStatus.INSUFFICIENT_VALIDATION_HISTORY` with zero forecast points and clear disclosures, while length 17 smoothly evaluates 3 rolling origins.

6. **Validation Boundary Instrumentation & Horizon Uncertainty Labeling:**
   - Evaluator boundaries are exposed via `get_rolling_origin_diagnostics()` to verify zero temporal leakage across all origins ($train\_end < test\_start$ and disjoint index sets).
   - In uncertainty quantification, horizon steps $j \le H_{\text{eval}}$ with $n_j \ge 3$ are labeled `"Direct Empirical Quantile (90th percentile out-of-sample residual)"`, while steps $j > H_{\text{eval}}$ are labeled `"Extrapolated Horizon Range (sqrt(h) expansion from step H_eval)"`.
   - Steps with insufficient empirical residuals ($n_j < 3$) fall back to MAE-based or square-root extrapolation and are explicitly labeled as fallback spreads, never mislabeled as direct empirical quantiles.
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
For intervals with zero post uploads ($\text{post\_count} = 0$), $y_t = 0.0$.
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
For each evaluated validation horizon step $j \in \{1 \dots H_{\text{eval}}\}$ with $n_j \ge 3$ residual observations:
$$\text{Spread}_j = \text{Percentile}_{90}\left(\{|e_{t, j}| \mid \text{origins } t\}\right)$$
$$\text{Method}_j = \text{"Direct Empirical Quantile (90th percentile out-of-sample residual)"}$$

For multi-step extrapolations beyond the evaluated validation window ($j > H_{\text{eval}}$):
$$\text{Spread}_j = \text{Spread}_{H_{\text{eval}}} \times \sqrt{\frac{j}{H_{\text{eval}}}}$$
$$\text{Method}_j = \text{"Extrapolated Horizon Range (sqrt(h) expansion from step } H_{\text{eval}}\text{)"}$$

For steps with insufficient residual observations ($n_j < 3$, e.g., at the outer edges of short series):
$$\text{Spread}_j = \text{Fallback Spread via } \sqrt{j} \text{ or MAE} \times 1.645 \times \sqrt{j}$$
$$\text{Method}_j = \text{"Extrapolated Fallback Spread" or "MAE-based Fallback Spread"}$$
*(Never mislabeled as Direct Empirical Quantile)*

The prediction interval enforces a non-negative floor:
$$\text{CI}_j = \left[ \max(0.0, \hat{y}_{T+j} - \text{Spread}_j), \quad \hat{y}_{T+j} + \text{Spread}_j \right]$$

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
- **Candidate Model Evaluations:**
  - **Moving Average (Champion):** MAE = 12,296.65, RMSE = 15,694.61, MASE = 0.635, Relative MAE = **0.76** (24% error reduction vs persistence)
  - **AutoRegressive Ridge (AR):** MAE = 13,415.14, RMSE = 15,630.48, MASE = 0.693, Relative MAE = 0.83
  - **Holt Exponential Smoothing:** MAE = 13,844.82, RMSE = 16,830.57, MASE = 0.715, Relative MAE = 0.86
  - **Naive (Persistence):** MAE = 16,094.60, RMSE = 20,070.34, MASE = 0.831, Relative MAE = 1.00
  - **Seasonal Naive:** MAE = 21,918.35, RMSE = 30,247.01, MASE = 1.132, Relative MAE = 1.36

---

## 5. Automated Regression Test Suite

All 40 unit, regression, optimization, and audit tests in `tests/test_trend_forecaster.py` pass cleanly:

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
```

**Full Repository Regression Suite:** **196 passed across Modules 1–7, UI shell/security, NLP/metadata hardening, and artifact build verification.**

---

## 6. Baseline Immutability Check (Current Verified 21-File Lock Ledger)

Twelve of the 21 protected Module 1–6 files remain unchanged against their Phase 4.2 baseline MD5 prefixes. The remaining 9 files contain strictly owner-authorized **Phase A Reliability & Release Hardening** remediations (explicit `INFERENCE_FAILED` NLP failure handling, HTML boundary escaping, deterministic model-metadata separation, invalid escape-sequence cleanup, and `use_container_width=True` -> `width="stretch"` migration) with core analytical calculations, statistical methods, and sample datasets preserved:

| Module | File Path | Baseline MD5 (Phase 4.2) | Verified MD5 (16 hex) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **M1** | `src/data_ingestion/schemas.py` | `3546f7a4214c0b26` | `46cf54b7b0576f32` | **Phase A approved (`INFERENCE_FAILED` status)** |
| **M1** | `src/data_ingestion/storage.py` | `cd20024b92849f7e` | `cd20024b92849f7e` | **LOCKED (unchanged)** |
| **M1** | `src/data_ingestion/api_connectors.py` | `23e6ac8373cbad5b` | `23e6ac8373cbad5b` | **LOCKED (unchanged)** |
| **M1** | `src/data_ingestion/data_validator.py` | `f4529830d062d057` | `f4529830d062d057` | **LOCKED (unchanged)** |
| **M1** | `src/analytics/engagement_metrics.py` | `ea763a3ac75d4f29` | `ea763a3ac75d4f29` | **LOCKED (unchanged)** |
| **M1** | `src/preprocessing/cleaning.py` | `f07e8a899c591226` | `f07e8a899c591226` | **LOCKED (unchanged)** |
| **M1** | `src/preprocessing/feature_engineering.py` | `72e8c3da2eba1fc9` | `72e8c3da2eba1fc9` | **LOCKED (unchanged)** |
| **M2** | `src/modeling/virality_predictor.py` | `6fded69b4cb710ce` | `99615deb28ffa3eb` | **Phase A approved (deterministic metadata JSON)** |
| **M3** | `src/nlp/sentiment_analyzer.py` | `b64a2bd4fd7dd5af` | `b64a2bd4fd7dd5af` | **LOCKED (unchanged)** |
| **M3** | `src/nlp/relatability_classifier.py` | `2d709b0b90705f7b` | `86e081f608a41130` | **Phase A approved (explicit `INFERENCE_FAILED`)** |
| **M4** | `src/statistics/ab_testing.py` | `d22e08d98765e8c5` | `d22e08d98765e8c5` | **LOCKED (unchanged)** |
| **M5** | `src/recommender/engine.py` | `2bf9ad1177501505` | `2bf9ad1177501505` | **LOCKED (unchanged)** |
| **M5** | `src/recommender/schemas.py` | `17855006e4aa2753` | `17855006e4aa2753` | **LOCKED (unchanged)** |
| **Pages**| `app/pages/01_Content_Performance.py` | `ffae1b9ad76f6411` | `1f66bd2ad6ff9d89` | **Phase A approved (`width="stretch"`)** |
| **Pages**| `app/pages/02_Virality_Prediction.py` | `96547c884f617b1e` | `fd7b51db786146e4` | **Phase A approved (`width="stretch"`)** |
| **Pages**| `app/pages/03_Sentiment_Analysis.py` | `f684bb9dde35a0c2` | `e56e89f89fb495a8` | **Phase A approved (HTML escaping, status/confidence UI & `width="stretch"`)** |
| **Pages**| `app/pages/04_AB_Testing.py` | `24443e08e148ddda` | `d15e62ea124864d7` | **Phase A approved (HTML escaping, warning fix, `width`)** |
| **Pages**| `app/pages/05_Recommendations.py` | `458c4b15ddefa1fb` | `27a4e3ac7d32e815` | **Phase A approved (HTML escaping)** |
| **M6** | `src/dashboard/growth_dashboard_backend.py` | `6b51574e52964abe` | `6b51574e52964abe` | **LOCKED (unchanged)** |
| **M6** | `app/pages/06_Growth_Dashboard.py` | `86f4ec55fbf50209` | `1b0e3656791fd172` | **Phase A approved (HTML escaping & `width="stretch"`)** |
| **M6** | `tests/test_module6_dashboard.py` | `e171c613f76b5158` | `e171c613f76b5158` | **LOCKED (unchanged)** |
