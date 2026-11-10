# SocialPulse

**SocialPulse** is a modular Python and Streamlit data science major project for analyzing short-form social media performance across Instagram, YouTube Shorts, and TikTok. It focuses on high-intent engagement signals (shares, saves, and watch retention), audience comment relatability, controlled creative A/B testing, and short-horizon cohort forecasting.

## Live Application

- **Streamlit Community Cloud:** https://social-pulse-analytics.streamlit.app/

The live deployment runs automatically from `streamlit_app.py` using the built-in synthetic benchmark dataset in `data/sample/`. On first container startup, missing processed data, SQLite tables, and model files are generated cleanly without requiring external social-media API credentials.

## Project Overview

Creators and small teams often judge short-form video performance using passive vanity metrics such as raw likes or unsegmented impressions. **SocialPulse** (*The Data-Driven Social Engagement Initiative*) addresses this gap by organizing analytics around high-intent actions—shares, bookmark saves, and watch completion—alongside explicit safeguards against cross-platform metric mixing, missing-value zero-imputation, and causal over-claiming from observational data.

The original academic project brief is archived at [Problem_Statement/Data_Science_Major_Project_December.pdf](Problem_Statement/Data_Science_Major_Project_December.pdf).

## Key Features

- **Schema-Validated Ingestion:** Normalizes post metrics into a 14-field canonical schema while preserving unavailable fields (such as YouTube `saves`) as missing (`<NA>`) instead of converting them to zero.
- **Pre-Publication Virality Modeling:** Scores ex-post Engagement Virality Index (EVI) and trains a Ridge regression estimator using strictly pre-publication features to prevent target leakage.
- **Hybrid Sentiment and Relatability NLP:** Separates audience identification (`Relatable`, `Neutral`, `Uncertain`) from emotional sentiment (`Positive`, `Negative`, `Neutral`, `Mixed`) with non-English text detection.
- **Controlled Experimentation Tools:** Separates randomized A/B tests from observational cohort comparisons with Welch's *t*-test, two-proportion *z*-test, Fisher's exact fallback, Holm–Bonferroni / FDR adjustments, and sample-size planning.
- **Evidence-Tiered Recommendations:** Combines findings from Modules 1–4 into prioritized creative actions and draft A/B test proposals without presenting observational patterns as proven cause and effect.
- **Multi-Corpus Workspace Support:** Includes an interactive sidebar corpus selector covering the built-in synthetic fixtures and optional locally prepared external public datasets.

## The Seven Project Modules

1. **Content Performance Tracker** ([`app/pages/01_Content_Performance.py`](app/pages/01_Content_Performance.py), [`src/data_ingestion/`](src/data_ingestion/), [`src/preprocessing/`](src/preprocessing/)): Validates and deduplicates post snapshots, stores clean tables in Parquet and SQLite, and tracks engagement rates and data completeness across platforms.
2. **Virality Prediction Engine** ([`app/pages/02_Virality_Prediction.py`](app/pages/02_Virality_Prediction.py), [`src/modeling/virality_predictor.py`](src/modeling/virality_predictor.py)): Computes weighted Engagement Virality Index (EVI) scores and forecasts draft performance from pre-publication attributes with naive-baseline evaluation and topic rankings.
3. **Audience Sentiment Analyzer** ([`app/pages/03_Sentiment_Analysis.py`](app/pages/03_Sentiment_Analysis.py), [`src/nlp/`](src/nlp/)): Evaluates comment sentiment and audience relatability through a hybrid rule + TF-IDF logistic regression classifier, interactive comment tester, and auditing views.
4. **A/B Testing Framework** ([`app/pages/04_AB_Testing.py`](app/pages/04_AB_Testing.py), [`src/statistics/ab_testing.py`](src/statistics/ab_testing.py)): Evaluates creative experiments at the post level with parametric and non-parametric tests, confidence intervals, outlier diagnostics, multiple-testing corrections, and power calculators.
5. **Engagement Optimization Recommender** ([`app/pages/05_Recommendations.py`](app/pages/05_Recommendations.py), [`src/recommender/`](src/recommender/)): Synthesizes statistical and analytical outputs into ranked recommendations tagged by evidence strength (`HIGH`, `MEDIUM`, `LOW` / `EXPLORATORY`) and conflict checks.
6. **Growth Visualization Dashboard** ([`app/pages/06_Growth_Dashboard.py`](app/pages/06_Growth_Dashboard.py), [`src/dashboard/`](src/dashboard/)): Unifies overview KPIs, time-series trends, normalized platform comparisons, topic and schedule breakdowns, NLP summaries, experiment forest plots, and top action items.
7. **Trend Forecasting** ([`app/pages/07_Trend_Forecasting.py`](app/pages/07_Trend_Forecasting.py), [`src/modeling/trend_forecaster.py`](src/modeling/trend_forecaster.py)): Benchmarks Naive, Seasonal Naive, Moving Average, Holt's Exponential Smoothing, and AutoRegressive Ridge models using rolling-origin walk-forward validation under strict platform isolation.

## Technology Stack

- **Language & Environment:** Python 3.11+ (verified on 3.11.9)
- **Data Processing & Storage:** Pandas (`3.0.3`), NumPy (`2.4.4`), PyArrow (`24.0.0`), SQLite (`sqlite3`)
- **Machine Learning, Statistics & NLP:** Scikit-learn (`1.9.0`), SciPy (`1.17.1`), Joblib (`1.5.3`), custom rule/lexicon preprocessing
- **Frontend & Visualization:** Streamlit (`1.58.0`), Plotly (`7.1.0`)
- **Testing & CI:** Pytest, GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))

## Project Structure

```text
SocialPulse/
├── streamlit_app.py          # Main Streamlit entrypoint (artifact bootstrap & navigation)
├── app/                      # Home view, shared UI components, and 7 Streamlit module pages
├── src/                      # Core Python modules (ingestion, preprocessing, ML, NLP, stats, dashboard)
├── scripts/                  # CLI tools for artifact builds, external datasets, benchmarks & reports
├── data/                     # Checked-in demo CSVs (data/sample/) and local external/processed folders
├── database/                 # Local SQLite database generated at runtime (gitignored)
├── models/                   # Tracked JSON metadata/benchmarks and generated .joblib files (gitignored)
├── docs/                     # Module technical docs and pilot content series template
├── reports/                  # Generated Strategy & Validation Report
├── notebooks/                # Five unexecuted exploratory notebooks (EDA, ML, NLP, A/B, forecasting)
├── tests/                    # Automated pytest regression test suite
├── Problem_Statement/        # Archived academic major project problem statement PDF
├── requirements.txt          # Pinned runtime dependencies (used locally and on Streamlit Cloud)
├── requirements-dev.txt      # Development, testing, and notebook dependencies
└── constraints.txt           # Pinned Python 3.11.9 transitive closure for reproducible dev/CI installs
```

## Installation and Local Setup

### Prerequisites

- **Python 3.11+** (tested on Python `3.11.9`; earlier Python versions are incompatible with the pinned scientific packages).

### 1. Clone and Install Dependencies

#### Windows PowerShell

```powershell
git clone https://github.com/SarthakPandey84/Social-Pulse.git SocialPulse
cd SocialPulse

python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Runtime-only installation (matches Streamlit Cloud):
pip install -r requirements.txt

# Or development + testing installation:
pip install -r requirements-dev.txt -c constraints.txt
```

#### Linux / macOS

```bash
git clone https://github.com/SarthakPandey84/Social-Pulse.git SocialPulse
cd SocialPulse

python3 -m venv .venv
source .venv/bin/activate

# Runtime-only installation:
pip install -r requirements.txt

# Or development + testing installation:
pip install -r requirements-dev.txt -c constraints.txt
```

### 2. Build Artifacts (Optional) and Launch the App

Launching the root entrypoint builds and validates any missing local demo artifacts automatically from `data/sample/`:

```powershell
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser. Always launch `streamlit_app.py` (not `app/streamlit_app.py`), as the root entrypoint runs startup artifact preparation, shared theme setup, and multipage navigation.

To run the build and validation pipeline explicitly from the command line:

```powershell
python scripts/build_all_artifacts.py
python scripts/build_all_artifacts.py --validate-only
```

### Streamlit Community Cloud Configuration

The deployed application at https://social-pulse-analytics.streamlit.app/ tracks branch `main` with main file path `streamlit_app.py`, `requirements.txt`, and Python `3.11`. No API secrets are required to run the synthetic benchmark workspace.

## Testing and CI

With `requirements-dev.txt` installed, run the full test suite:

```powershell
pytest tests/ -v
```

- **Verified Test Results:** **`273` automated tests collected** across Modules 1–7, preprocessing/adapters, governance rules, metadata determinism, UI rendering, and startup bootstrapping (`273 passed` locally when optional prepared external datasets are present; `272 passed, 1 skipped` on clean checkouts where optional external datasets are absent).
- **Continuous Integration:** GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs dependency verification (`pip check`), artifact build & validation, tracked metadata determinism checks, syntax/deprecation compilation checks, `pytest`, and a headless Streamlit startup health check (`/_stcore/health`) on `ubuntu-latest` with Python `3.11`.

## Datasets, Demo Data, and Documentation

### Demo Fixtures vs. External Corpora

- **Built-in Demo Data (`data/sample/`):** Clean checkouts and the live Cloud app use small checked-in synthetic benchmark CSVs (`120` unique posts / `121` snapshots, `500` comments, `4` experiment fixtures, and `170` daily topic records) to demonstrate every module deterministically offline.
- **Optional External Public Datasets (Local Only):** [`scripts/download_external_datasets.py`](scripts/download_external_datasets.py) and [`scripts/prepare_external_datasets.py`](scripts/prepare_external_datasets.py) can download and normalize pinned public/academic corpora (TikTok, Instagram cohorts, Zenodo datasets, and Hugging Face comment benchmarks) into gitignored `data/external/` and `data/processed/external/` directories. Datasets awaiting license review stay excluded by default unless `--include-pending-review` is explicitly passed, and Kaggle sources remain blocked unless credentials are configured via [`.env.example`](.env.example).

### Detailed Documentation & Reports

- [Comprehensive Strategy & Validation Report](reports/SOCIALPULSE_STRATEGY_REPORT.md) — Empirical external benchmark results, governance tier matrix, synthetic vs. real comparisons, and runtime benchmarks.
- [Pilot Content Series Protocol & Schema Template](docs/CONTENT_SERIES_TEMPLATE.md) — Deliverable 5 four-part creator pilot design, A/B specifications (`PILOT_EXP_01`–`03`), and 14-field logging schema.
- [Module 6 Technical Documentation](docs/MODULE_6_GROWTH_DASHBOARD.md) — Growth Dashboard integration map, KPI formulas, and regression suite overview.
- [Module 7 Technical Documentation](docs/MODULE_7_TREND_FORECASTING.md) — Trend Forecasting walk-forward validation, uncertainty intervals, and benchmark tables.
- Benchmark Data Files: [`models/real_data_benchmarks.json`](models/real_data_benchmarks.json), [`models/performance_benchmarks.json`](models/performance_benchmarks.json), [`models/virality_predictor_metadata.json`](models/virality_predictor_metadata.json), and [`models/relatability_metadata.json`](models/relatability_metadata.json).

## Limitations and Future Work

- **Synthetic Demo Scope on Live Cloud:** Because external corpora are gitignored and Cloud containers use ephemeral storage, https://social-pulse-analytics.streamlit.app/ runs interactively on the built-in synthetic dataset rather than live creator accounts; real-data evaluation summaries are documented separately in [`reports/SOCIALPULSE_STRATEGY_REPORT.md`](reports/SOCIALPULSE_STRATEGY_REPORT.md).
- **Placeholder Social-Platform APIs:** [`src/data_ingestion/api_connectors.py`](src/data_ingestion/api_connectors.py) and [`src/data_ingestion/data_scraper.py`](src/data_ingestion/data_scraper.py) define validated interface contracts (`status="placeholder"`) without live OAuth sessions or automated scraping.
- **Content Series Deliverable Status:** [`docs/CONTENT_SERIES_TEMPLATE.md`](docs/CONTENT_SERIES_TEMPLATE.md) provides the structured pilot protocol and CSV logging template; no original creator videos or post-launch organic campaign metrics have been filmed or fabricated.
- **Unexecuted Reference Notebooks:** The five Jupyter notebooks in `notebooks/` are preserved as exploratory reference artifacts with zero stored cell outputs.
- **Future Improvements:** Connect owner-authorized Instagram Graph and YouTube Analytics credentials via local `.env` configuration, execute the 4-part pilot series for post-launch causal evaluation, and extend NLP comment handling beyond English-only validation.

## Credits and License

- Developed by **Sarthak Pandey** as a final-year Data Science major project ([`LICENSE`](LICENSE)).
- Project-owned code, documentation, UI assets, and sample benchmark fixtures are released under the **MIT License** ([`LICENSE`](LICENSE)).
- The external problem statement document ([`Problem_Statement/Data_Science_Major_Project_December.pdf`](Problem_Statement/Data_Science_Major_Project_December.pdf)) is redistributed with owner-confirmed permission solely for academic context and is excluded from the MIT License. Optional third-party public datasets referenced in the evaluation scripts remain governed by their respective upstream licenses.
