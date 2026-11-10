# Deliverable 5 — The Content Series: Creator Experiment Protocol & Schema Logging Template

> **Honest Scope Disclosure:**  
> Consistent with SocialPulse governance standards, this document provides the **executable pilot protocol, experimental design specification, and 14-field data logging template** paired with Modules 1–7. No live creator videos, social account URLs, or post-launch audience metrics are fabricated. Executing this series in production requires creator-owned accounts and real organic exposure.

---

## 1. Strategic Objective

Translate SocialPulse's empirical and exploratory findings into a **4-part short-form creator pilot series** (extended across **16 to 32 randomized posts**) designed to test causal drivers of high-intent engagement (**Shares**, **Saves / Bookmark Collects**, and **Watch Retention**) without relying on vanity likes or unadjusted observational correlations.

### Cross-Module Alignment
| Pilot Dimension | Informing Module | Empirical / Analytical Rationale |
| :--- | :--- | :--- |
| **Target Metric Priority** | Module 2 (`virality_predictor.py`) | Prioritizes weighted **Shares (`0.40`)** and **Saves (`0.35`)** over Comments (`0.15`) and Likes (`0.10`) in the Engagement Virality Index (EVI). |
| **Hook & Runtime Hypotheses** | Module 5 (`engine.py`) & Zenodo Educational Reels | Observational associations suggest runtime pacing and hook framing warrant controlled validation against retention and share rates. |
| **Audience Relatability Prompting** | Module 3 (`relatability_classifier.py`) | Orthogonal tagging tracks whether vulnerability/reflection cues shift **Relatable comment share** independently of positive/negative polarity. |
| **Causal Validation & Power** | Module 4 (`ab_testing.py`) | Replaces observational comparisons (`is_causal=False`) with randomized post-level variant assignment (`randomized_experiment`) and Holm–Bonferroni correction. |
| **Forecast & Maturity Window** | Modules 2 & 7 | Enforces a **minimum 7-day post-publication maturity window** before scoring cumulative performance, eliminating same-day/immature cohort bias. |

---

## 2. 4-Part Pilot Content Series Structure

Each episode uses the **Relatable Struggle -> Practical Reframe -> Experiential Reflection** narrative arc across **Instagram Reels**, **YouTube Shorts**, and **TikTok** (analyzed strictly per platform due to metric non-equivalence).

| Episode Slot | Core Content Pillar | Narrative Structure (0–45s) | Logged `topic` Label |
| :--- | :--- | :--- | :--- |
| **Part 1** | Expectation vs. Reality Burnout | Hook tension (0–3s) -> relatable workflow friction -> concise tactical takeaway -> CTA | `Career Burnout` |
| **Part 2** | Quiet Doubt / Credential Anxiety | High-contrast opening statement/question -> personal bottleneck breakdown -> reframe -> CTA | `Imposter Syndrome` |
| **Part 3** | Everyday Social Friction | Observable scenario -> emotional pattern naming -> actionable boundary prompt -> CTA | `Social Anxiety` |
| **Part 4** | Execution Paralysis & Perfectionism | Cold-open friction scene -> micro-step reset protocol -> peer reflection invitation | `Procrastination Loop` |

---

## 3. Controlled Factorial A/B Test Design (Module 4 Compliant)

To generate valid causal evidence readable directly by [`ABTestingFramework`](../src/statistics/ab_testing.py), assign variants randomly using balanced block randomization (`seed = 42`) rather than ad-hoc publishing:

### Primary Paired Experiments

| Test ID | Hypothesis | Control (`Variant A`) | Treatment (`Variant B`) | Primary Metric | Secondary Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`PILOT_EXP_01`** | Visual action cold-opens retain viewers and earn more shares than narrative intros. | `Story Hook` | `Visual Hook` | `share_rate` (`shares / views`) | `retention_rate`, `save_rate` (IG only) |
| **`PILOT_EXP_02`** | Concise 25–45s edits yield higher retention than 55–85s cuts of matching scripts. | `Long Cut (55-85s)` | `Compact Cut (25-45s)` | `retention_rate` | `views`, `share_rate` |
| **`PILOT_EXP_03`** | Experiential reflection prompts increase `Relatable` comments and IG bookmark saves vs. standard prompts. | `Standard Save CTA` | `Experiential Reflection Prompt` | `relatability_pct` | `save_rate` (IG), `collect_count` ratio (TikTok) |

### Statistical & Maturity Guardrails
1. **Cluster-Safe Unit of Analysis:** Evaluate rates at **post level** (or per-creator block when multiple creators participate) to mitigate viewer exposure dependence; never treat 10,000 raw impressions on 2 posts as 10,000 independent subjects.
2. **Minimum Sample Planning:** Run `ABTestingFramework.calculate_sample_size_proportions()` / power audit in `app/pages/04_AB_Testing.py` before locking test termination.
3. **Maturity Lock:** Record interim daily trajectories if desired, but run final hypothesis testing ONLY once every enrolled post reaches **`(collection_timestamp - timestamp) >= 7 days`**.
4. **Platform Separation Rules:**
   - Never pool YouTube (`saves = <NA>`) or TikTok (`saves = <NA>`, `collect_count` proxy tracked separately) directly into Instagram Insights `saves`.
   - Never coerce disabled comments/ratings or missing retention metrics to `0`.

---

## 4. Standardized Logging Templates

### 4.1 Post-Level Performance Log (`content_series_posts.csv`)
Maps directly to SocialPulse's **14 canonical unified fields** plus modeling/experiment metadata (validate via `DataValidator.validate_posts_schema()` and `scripts/build_all_artifacts.py`):

```csv
platform,post_id,timestamp,content_type,views,impressions,reach,likes,comments_count,shares,saves,collect_count,follower_count,retention_rate,collection_timestamp,topic,hook_type,duration_seconds,posting_hour,new_followers,experiment_id,variant_label,provenance
```

#### Field Rules & Constraints
| Column | Allowed / Required Type | Platform Rules |
| :--- | :--- | :--- |
| `platform` | `instagram` \| `youtube` \| `tiktok` | Required lowercase identifier |
| `post_id` | Non-empty unique string | Preserve full string ID (avoid float truncation) |
| `timestamp` | ISO-8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`) | Publication timestamp |
| `content_type` | `reel` \| `carousel` \| `image` \| `short` \| `short video` \| `video` | Canonical lowercase format |
| `views`, `likes`, `comments_count`, `shares` | Nullable `Int64` (`>= 0`) | Leave blank (`<NA>`) if unavailable — NEVER fill `0` |
| `impressions`, `reach`, `saves` | Nullable `Int64` | IG Insights (`owner_authorized_data`) only; `<NA>` on YouTube/public TikTok |
| `collect_count` | Nullable `Int64` | TikTok bookmark proxy only (`<NA>` elsewhere) |
| `retention_rate` | Nullable `Float64` in `[0.0, 1.0]` | e.g., `0.625` for `62.5%` average watch completion (`<NA>` if unexposed) |
| `collection_timestamp` | ISO-8601 UTC (`>= timestamp`) | Snapshot time (`>= +7 days` for final settled analysis) |
| `provenance` | `owner_authorized_data` \| `real_public_data` | Authentic campaign provenance tag |

### 4.2 Comment Extraction Log (`content_series_comments.csv`)
Ingested cleanly into Module 3 (`scripts/train_and_benchmark_nlp.py` and `app/pages/03_Sentiment_Analysis.py`):

```csv
platform,comment_id,post_id,comment_text,comment_timestamp,likes,language_hint
```

*Note:* Raw commenter usernames/handles MUST be dropped or scrubbed via `pseudonymize_identifier()` and `redact_pii_text()` in `src/preprocessing/quality_pipeline.py` prior to analytical archival.

---

## 5. End-to-End Ingestion Workflow After Series Launch

1. Save populated logs under `data/raw/` (e.g., `content_series_posts.csv`, `content_series_comments.csv`).
2. Validate & sanitize through the 12-stage quality pipeline (`run_cleaning_pipeline()`) verifying schema types, null preservation, negative-delta checks, and PII redaction.
3. Evaluate `PILOT_EXP_01`–`PILOT_EXP_03` via **Module 4 (`04_AB_Testing.py`)** using Holm–Bonferroni multiple-testing correction and 95% confidence intervals before updating any **Module 5** creative recommendations.

