"""Virality Prediction Engine & Pre-Publication Forecasting Module.

Provides:
1. Engagement Virality Index (EVI): Ex-post heuristic calculated from observed engagement.
2. Pre-Publication Predictive Model: Supervised regressor trained strictly on pre-publication features.
3. Target leakage prevention, maturity filtering (>=7 days), chronological train/test splitting,
   and baseline benchmarking against a naive Dummy Regressor.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.analytics.engagement_metrics import _vectorized_retention_applicable_mask, is_retention_applicable
from src.data_ingestion.schemas import SourceProvenance
from src.preprocessing.cleaning import get_latest_posts_view


# Canonical pre-publication features available BEFORE content is published
PRE_PUBLICATION_FEATURES = [
    "platform",
    "content_type",
    "topic",
    "hook_type",
    "duration_seconds",
    "posting_hour",
    "posting_day_of_week",
    "follower_count"
]

# Post-publication outcome metrics strictly forbidden in predictive features (Target Leakage Guard)
OUTCOME_METRICS = {
    "views",
    "impressions",
    "reach",
    "likes",
    "comments_count",
    "shares",
    "saves",
    "retention_rate",
    "new_followers",
    "viral_coefficient",
    "engagement_virality_index",
    "high_intent_actions",
    "save_to_share_ratio",
    "intent_to_passive_ratio",
    "views_engagement_rate",
    "reach_engagement_rate",
    "overall_engagement_rate",
    "effective_watch_views",
    "equivalent_full_watch_views",
    "collect_count",
    "download_count",
    "whatsapp_share_count",
    "engagement_rate_reported",
    "engagement_rate_recomputed",
    "share_rate_recomputed",
    "save_rate_recomputed",
    "followers_gained",
    "performance_bucket_label",
    "creator_relative_ratio",
}


def is_outcome_leakage_column(col_name: str) -> bool:
    """Return True if column represents a post-publication outcome or quarantined leakage metric."""
    low = str(col_name).strip().lower()
    if low in OUTCOME_METRICS:
        return True
    return low.startswith(("day0_", "day1_", "day7_", "day30_"))


class ViralityPredictor:
    """Calculates Engagement Virality Index and trains pre-publication predictive models.

    Methodological Notes:
    1. Terminology: The heuristic metric is an Engagement Virality Index (EVI), measuring high-intent
       interaction density per unit exposure. It is NOT a true epidemiological/referral viral coefficient K.
    2. Data Provenance: Benchmark evaluations use synthetic/illustrative data for pipeline demonstration.
    3. Target Maturity: Supervised training filters posts to mature observations (>= 7 days since publication).
    4. Prediction Uncertainty: Error bands represent approximate empirical residual dispersion,
       not calibrated conformal prediction intervals.
    """

    DEFAULT_WEIGHTS = {
        "shares": 0.40,
        "saves": 0.35,
        "comments_count": 0.15,
        "likes": 0.10
    }

    MIN_TRAINING_SAMPLES = 50

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        smoothing_constant_epsilon: float = 100.0
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.epsilon = smoothing_constant_epsilon
        self.model: Optional[Pipeline] = None
        self.metadata: Dict[str, Any] = {}

    # =========================================================================
    # PART A: EX-POST ENGAGEMENT VIRALITY INDEX (HEURISTIC)
    # =========================================================================

    def calculate_single_post_score(
        self,
        row: pd.Series,
        denominator_mode: str = "auto"
    ) -> Dict[str, Any]:
        """Compute the availability-aware Engagement Virality Index for an individual post."""
        # 1. Identify which engagement metrics are non-null and available
        available_metrics = {}
        for metric, w in self.weights.items():
            val = row.get(metric)
            if pd.notna(val):
                available_metrics[metric] = (float(val), w)

        # If zero metrics are available, score cannot be calculated (return NaN, never 0)
        if not available_metrics:
            return {
                "viral_coefficient": np.nan,
                "engagement_virality_index": np.nan,
                "metrics_contributed": "",
                "weight_coverage": 0.0,
                "denominator_used": "unmeasured_no_metrics",
                "confidence_tier": "unmeasured"
            }

        # 2. Availability-aware weight normalization
        total_available_weight = sum(w for _, w in available_metrics.values())
        if total_available_weight <= 0:
            return {
                "viral_coefficient": np.nan,
                "engagement_virality_index": np.nan,
                "metrics_contributed": "",
                "weight_coverage": 0.0,
                "denominator_used": "unmeasured_no_weight",
                "confidence_tier": "unmeasured"
            }

        normalized_numerator = sum(
            (w / total_available_weight) * val for val, w in available_metrics.values()
        )

        # 3. Determine exposure denominator (reach vs views vs impressions)
        views = row.get("views")
        reach = row.get("reach")
        impressions = row.get("impressions")

        views_val = float(views) if pd.notna(views) and views is not None else None
        reach_val = float(reach) if pd.notna(reach) and reach is not None else None
        impr_val = float(impressions) if pd.notna(impressions) and impressions is not None else None

        if denominator_mode == "reach":
            denom = reach_val if reach_val is not None else (views_val if views_val is not None else impr_val)
            denom_type = "reach" if reach_val is not None else "views_fallback"
        elif denominator_mode == "views":
            denom = views_val if views_val is not None else (reach_val if reach_val is not None else impr_val)
            denom_type = "views" if views_val is not None else "reach_fallback"
        else:  # "auto": prefers reach -> views -> impressions
            if reach_val is not None:
                denom = reach_val
                denom_type = "reach"
            elif views_val is not None:
                denom = views_val
                denom_type = "views"
            elif impr_val is not None:
                denom = impr_val
                denom_type = "impressions"
            else:
                denom = None
                denom_type = "unmeasured_no_exposure"

        # If exposure is completely unmeasured (neither views, reach, nor impressions is present),
        # return NaN rather than guessing an exposure number
        if denom is None:
            return {
                "viral_coefficient": np.nan,
                "engagement_virality_index": np.nan,
                "metrics_contributed": ",".join(sorted(available_metrics.keys())),
                "weight_coverage": round(total_available_weight, 2),
                "denominator_used": denom_type,
                "confidence_tier": "unmeasured"
            }

        # Denominator with additive smoothing (prevents division by zero when views/reach == 0)
        adjusted_denom = max(denom, 0.0) + self.epsilon
        base_rate = (normalized_numerator / adjusted_denom) * 1000.0

        # 4. Retention factor adjustment (video-only; static/carousel/unknown/incompatible content uses neutral 1.0)
        retention = row.get("retention_rate")
        if "content_type" in row.index:
            retention_allowed = is_retention_applicable(row.get("platform"), row.get("content_type"))
        else:
            retention_allowed = True

        if retention_allowed and pd.notna(retention) and retention is not None and 0.0 <= float(retention) <= 1.0:
            phi = 0.5 + float(retention)
        else:
            phi = 1.0  # Neutral multiplier if retention is unavailable, non-video, or platform-incompatible

        evi = round(float(base_rate * phi), 2)

        # 5. Confidence tiering based on metric availability
        if total_available_weight >= 0.75:
            tier = "high"
        elif total_available_weight >= 0.40:
            tier = "medium"
        else:
            tier = "low"

        return {
            "viral_coefficient": evi,
            "engagement_virality_index": evi,
            "metrics_contributed": ",".join(sorted(available_metrics.keys())),
            "weight_coverage": round(total_available_weight, 2),
            "denominator_used": denom_type,
            "confidence_tier": tier
        }

    def calculate_engagement_virality_index_detailed(
        self,
        df: pd.DataFrame,
        denominator_mode: str = "auto"
    ) -> pd.DataFrame:
        """Calculate availability-aware EVI with comprehensive audit tracking and within-platform percentiles."""
        if df.empty:
            return pd.DataFrame(
                {
                    "viral_coefficient": pd.Series(dtype="float64"),
                    "engagement_virality_index": pd.Series(dtype="float64"),
                    "metrics_contributed": pd.Series(dtype="object"),
                    "weight_coverage": pd.Series(dtype="float64"),
                    "denominator_used": pd.Series(dtype="object"),
                    "confidence_tier": pd.Series(dtype="object"),
                    "evi_platform_percentile": pd.Series(dtype="float64"),
                },
                index=df.index,
            )

        n = len(df)
        sorted_items = sorted(self.weights.items(), key=lambda kv: str(kv[0]))
        total_weight = np.zeros(n, dtype="float64")
        weighted_sum = np.zeros(n, dtype="float64")
        any_metric = np.zeros(n, dtype=bool)
        contrib_masks: List[Tuple[str, np.ndarray]] = []

        for metric, w in sorted_items:
            w_float = float(w)
            if metric in df.columns:
                m_arr = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
                m_mask = ~np.isnan(m_arr)
            else:
                m_arr = np.full(n, np.nan, dtype="float64")
                m_mask = np.zeros(n, dtype=bool)
            any_metric |= m_mask
            total_weight += np.where(m_mask, w_float, 0.0)
            weighted_sum += np.where(m_mask, w_float * m_arr, 0.0)
            contrib_masks.append((str(metric), m_mask))

        # Vectorized sorted comma-joined contributed metric labels
        if len(contrib_masks) <= 16:
            bits = np.zeros(n, dtype=np.int32)
            for idx_m, (_, mask_m) in enumerate(contrib_masks):
                bits |= mask_m.astype(np.int32) << idx_m
            uniq_bits = np.unique(bits)
            bit_map: Dict[int, str] = {
                int(b): ",".join(name for i, (name, _) in enumerate(contrib_masks) if int(b) & (1 << i))
                for b in uniq_bits
            }
            lut = np.empty(1 << len(contrib_masks), dtype=object)
            for b_k, b_v in bit_map.items():
                lut[b_k] = b_v
            metrics_contributed = lut[bits]
        else:
            metrics_contributed = np.array(
                [",".join(name for name, m_mask in contrib_masks if m_mask[i]) for i in range(n)],
                dtype=object,
            )

        def _num_col(col_name: str) -> Tuple[np.ndarray, np.ndarray]:
            if col_name not in df.columns:
                return np.full(n, np.nan, dtype="float64"), np.zeros(n, dtype=bool)
            arr = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
            return arr, ~np.isnan(arr)

        v_arr, has_v = _num_col("views")
        r_arr, has_r = _num_col("reach")
        i_arr, has_i = _num_col("impressions")

        if denominator_mode == "reach":
            has_denom = has_r | has_v | has_i
            denom_arr = np.where(has_r, r_arr, np.where(has_v, v_arr, i_arr))
            base_denom_type = np.where(has_r, "reach", "views_fallback")
        elif denominator_mode == "views":
            has_denom = has_v | has_r | has_i
            denom_arr = np.where(has_v, v_arr, np.where(has_r, r_arr, i_arr))
            base_denom_type = np.where(has_v, "views", "reach_fallback")
        else:
            has_denom = has_r | has_v | has_i
            denom_arr = np.where(has_r, r_arr, np.where(has_v, v_arr, i_arr))
            base_denom_type = np.select(
                [has_r, has_v, has_i],
                ["reach", "views", "impressions"],
                default="unmeasured_no_exposure",
            )

        has_pos_weight = any_metric & (total_weight > 0.0)
        valid_mask = has_pos_weight & has_denom

        evi_arr = np.full(n, np.nan, dtype="float64")
        if valid_mask.any():
            norm_num = weighted_sum[valid_mask] / total_weight[valid_mask]
            adj_denom = np.maximum(denom_arr[valid_mask], 0.0) + float(self.epsilon)
            base_rate = (norm_num / adj_denom) * 1000.0

            ret_arr, has_ret = _num_col("retention_rate")
            if "content_type" in df.columns:
                ret_allowed = _vectorized_retention_applicable_mask(df)
            else:
                ret_allowed = np.ones(n, dtype=bool)
            ret_valid = (
                ret_allowed[valid_mask]
                & has_ret[valid_mask]
                & (ret_arr[valid_mask] >= 0.0)
                & (ret_arr[valid_mask] <= 1.0)
            )
            phi = np.where(ret_valid, 0.5 + ret_arr[valid_mask], 1.0)
            evi_arr[valid_mask] = np.round(base_rate * phi, 2)

        weight_coverage = np.where(any_metric, np.round(total_weight, 2), 0.0)
        metrics_out = np.where(has_pos_weight, metrics_contributed, "")
        denom_out = np.select(
            [~any_metric, ~has_pos_weight],
            ["unmeasured_no_metrics", "unmeasured_no_weight"],
            default=base_denom_type,
        )
        tier_out = np.select(
            [~valid_mask, total_weight >= 0.75, total_weight >= 0.40],
            ["unmeasured", "high", "medium"],
            default="low",
        )

        detailed_df = pd.DataFrame(
            {
                "viral_coefficient": evi_arr,
                "engagement_virality_index": evi_arr.copy(),
                "metrics_contributed": metrics_out,
                "weight_coverage": weight_coverage,
                "denominator_used": denom_out,
                "confidence_tier": tier_out,
            },
            index=df.index,
        )
        if "platform" in df.columns and not detailed_df.empty:
            temp = pd.DataFrame(
                {
                    "platform": df["platform"],
                    "engagement_virality_index": detailed_df["engagement_virality_index"],
                },
                index=df.index,
            )
            detailed_df["evi_platform_percentile"] = self.normalize_target_by_platform(
                temp, target_col="engagement_virality_index"
            )
        else:
            detailed_df["evi_platform_percentile"] = pd.Series(np.nan, index=df.index, dtype="float64")
        return detailed_df

    def calculate_viral_coefficient_detailed(
        self,
        df: pd.DataFrame,
        denominator_mode: str = "auto"
    ) -> pd.DataFrame:
        """Alias for calculate_engagement_virality_index_detailed."""
        return self.calculate_engagement_virality_index_detailed(df, denominator_mode=denominator_mode)

    def calculate_engagement_virality_index(
        self,
        df: pd.DataFrame,
        denominator_mode: str = "auto"
    ) -> pd.Series:
        """Calculate the normalized Engagement Virality Index series."""
        detailed = self.calculate_engagement_virality_index_detailed(df, denominator_mode=denominator_mode)
        return detailed["engagement_virality_index"]

    def calculate_viral_coefficient(
        self,
        df: pd.DataFrame,
        denominator_mode: str = "auto"
    ) -> pd.Series:
        """Backwards-compatible alias for calculate_engagement_virality_index."""
        return self.calculate_engagement_virality_index(df, denominator_mode=denominator_mode)

    def rank_topics_by_virality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group and rank topics by mean Engagement Virality Index, preserving sample sizes."""
        df_calc = get_latest_posts_view(df).copy() if "post_id" in df.columns else df.copy()
        if "engagement_virality_index" not in df_calc.columns and "viral_coefficient" not in df_calc.columns:
            df_calc["engagement_virality_index"] = self.calculate_engagement_virality_index(df_calc)
        elif "engagement_virality_index" not in df_calc.columns:
            df_calc["engagement_virality_index"] = df_calc["viral_coefficient"]

        # Drop uncalculated NaN scores to prevent skewing
        valid_df = df_calc.dropna(subset=["engagement_virality_index"]).copy()
        if valid_df.empty:
            return pd.DataFrame()

        if "platform" in valid_df.columns:
            valid_df["evi_platform_percentile"] = self.normalize_target_by_platform(
                valid_df, target_col="engagement_virality_index"
            )
        else:
            valid_df["evi_platform_percentile"] = (
                valid_df["engagement_virality_index"].rank(pct=True) * 100.0
            ).round(1)

        ranking = valid_df.groupby("topic").agg(
            posts_count=("post_id", "count"),
            avg_viral_coefficient=("engagement_virality_index", "mean"),
            avg_platform_percentile=("evi_platform_percentile", "mean"),
            avg_shares=("shares", "mean"),
            avg_saves=("saves", "mean"),
            avg_retention=("retention_rate", "mean"),
            total_views=("views", "sum")
        ).reset_index()

        ranking["avg_viral_coefficient"] = ranking["avg_viral_coefficient"].round(2)
        ranking["avg_platform_percentile"] = ranking["avg_platform_percentile"].round(1)
        ranking = ranking.sort_values(by="avg_viral_coefficient", ascending=False).reset_index(drop=True)
        return ranking

    def normalize_target_by_platform(
        self,
        df: pd.DataFrame,
        target_col: str = "engagement_virality_index"
    ) -> pd.Series:
        """Compute deterministic within-platform percentile ranks in (0, 100] for intra-platform comparison.

        Methodology & Limitations:
        - Percentiles are computed strictly within each non-null platform cohort using average-rank tie
          handling (`rank(pct=True, method='average') * 100.0`, rounded to 1 decimal place).
        - Missing target scores or missing/unknown platform values remain `NaN` rather than pooling across platforms.
        - Normalization aligns intra-platform relative standing only; it does NOT make Instagram reach-denominated
          metrics and YouTube view-denominated metrics behaviorally or causally equivalent, and is not substituted
          as the supervised regression target.
        """
        df_calc = df.copy()
        if target_col not in df_calc.columns:
            df_calc[target_col] = self.calculate_engagement_virality_index(df_calc)

        percentiles = pd.Series(np.nan, index=df_calc.index, dtype="float64")
        if "platform" not in df_calc.columns:
            return percentiles

        plat_norm = df_calc["platform"].where(df_calc["platform"].notna(), other=np.nan)
        valid_plat_mask = plat_norm.notna() & (plat_norm.astype(str).str.strip() != "")
        for _, group in df_calc.loc[valid_plat_mask].groupby(plat_norm.loc[valid_plat_mask].astype(str).str.strip().str.lower()):
            valid = group[target_col].dropna()
            if len(valid) > 0:
                ranks = valid.rank(pct=True, method="average") * 100.0
                percentiles.loc[ranks.index] = ranks.round(1)

        return percentiles

    # =========================================================================
    # PART B: PRE-PUBLICATION PREDICTIVE MODEL & EVALUATION
    # =========================================================================

    @staticmethod
    def temporal_train_test_split(
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset strictly chronologically to eliminate future lookahead bias."""
        df_sorted = df.copy()
        df_sorted[timestamp_col] = pd.to_datetime(df_sorted[timestamp_col], utc=True, format="ISO8601")
        df_sorted = df_sorted.sort_values(by=timestamp_col).reset_index(drop=True)

        split_idx = int(len(df_sorted) * (1.0 - test_ratio))
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        return train_df, test_df

    def prepare_dataset_for_modeling(
        self,
        df: pd.DataFrame,
        target_col: str = "engagement_virality_index",
        min_elapsed_days: float = 7.0
    ) -> pd.DataFrame:
        """Prepare unique post observations, derive pre-publication features, and filter immature posts."""
        # 1. Snapshot Deduplication Guard: use strictly the latest observation per post
        df_unique = get_latest_posts_view(df).copy()

        # 2. Maturity Filter: Exclude posts observed for less than min_elapsed_days (default 7 days)
        if "timestamp" in df_unique.columns and "collection_timestamp" in df_unique.columns:
            pub_ts = pd.to_datetime(df_unique["timestamp"], utc=True, format="ISO8601")
            coll_ts = pd.to_datetime(df_unique["collection_timestamp"], utc=True, format="ISO8601")
            elapsed_days = (coll_ts - pub_ts).dt.total_seconds() / 86400.0
            df_unique["elapsed_days"] = elapsed_days
            df_unique = df_unique[df_unique["elapsed_days"] >= min_elapsed_days].copy()

        # 3. Derive Pre-Publication Day of Week
        if "timestamp" in df_unique.columns:
            df_unique["posting_day_of_week"] = pd.to_datetime(df_unique["timestamp"], utc=True, format="ISO8601").dt.day_name()
        else:
            df_unique["posting_day_of_week"] = "Unknown"

        # 4. Compute Target if absent
        if target_col not in df_unique.columns:
            df_unique[target_col] = self.calculate_engagement_virality_index(df_unique)

        # 5. Filter usable records with non-null target
        valid_df = df_unique.dropna(subset=[target_col]).copy()
        return valid_df

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        model_type: str = "ridge",
        test_ratio: float = 0.2,
        target_col: str = "engagement_virality_index",
        min_elapsed_days: float = 7.0
    ) -> Dict[str, Any]:
        """Train a pre-publication model on mature settled posts, benchmark against Dummy baseline."""
        # 1. Prepare data with maturity filtering
        clean_df = self.prepare_dataset_for_modeling(df, target_col=target_col, min_elapsed_days=min_elapsed_days)

        if len(clean_df) < self.MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Insufficient historical data to train a predictive model. "
                f"Requires at least {self.MIN_TRAINING_SAMPLES} unique mature post records, but found {len(clean_df)}."
            )

        # 2. Strict Chronological Train/Test Split
        train_df, test_df = self.temporal_train_test_split(clean_df, timestamp_col="timestamp", test_ratio=test_ratio)

        # 3. Feature Selection & Target Leakage Guard
        cat_features = ["platform", "content_type", "topic", "hook_type", "posting_day_of_week"]
        num_features = ["duration_seconds", "posting_hour", "follower_count"]
        all_features = cat_features + num_features

        # Assert no outcome leakage in feature set
        for f in all_features:
            if is_outcome_leakage_column(f):
                raise ValueError(f"Target leakage detected! Outcome metric '{f}' cannot be a predictive feature.")

        X_train = train_df[all_features].copy()
        y_train = train_df[target_col].copy()
        X_test = test_df[all_features].copy()
        y_test = test_df[target_col].copy()

        # 4. Construct Pipeline (Fitted strictly on training data)
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_features)
            ]
        )

        # Candidate Regressor Selection
        if model_type == "ridge":
            regressor = Ridge(alpha=1.0)
        elif model_type == "random_forest":
            regressor = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
        elif model_type == "gradient_boosting":
            regressor = GradientBoostingRegressor(n_estimators=40, max_depth=3, learning_rate=0.05, random_state=42)
        else:
            raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'ridge', 'random_forest', or 'gradient_boosting'.")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        # 5. Fit Candidate Model
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # 6. Fit Naive Dummy Baseline (Mean strategy)
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        y_dummy = dummy.predict(X_test)

        # 7. Compute Evaluation Metrics on Held-out Temporal Test Set
        dummy_mae = float(mean_absolute_error(y_test, y_dummy))
        dummy_rmse = float(np.sqrt(mean_squared_error(y_test, y_dummy)))
        dummy_r2 = float(r2_score(y_test, y_dummy))

        cand_mae = float(mean_absolute_error(y_test, y_pred))
        cand_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        cand_r2 = float(r2_score(y_test, y_pred))

        spear_rho, spear_pval = spearmanr(y_test, y_pred)
        spear_rho = float(spear_rho) if pd.notna(spear_rho) else 0.0
        spear_pval = float(spear_pval) if pd.notna(spear_pval) else 1.0

        # Lift over baseline
        mae_lift_pct = round(((dummy_mae - cand_mae) / (dummy_mae + 1e-9)) * 100.0, 1)
        rmse_lift_pct = round(((dummy_rmse - cand_rmse) / (dummy_rmse + 1e-9)) * 100.0, 1)
        beats_baseline = cand_rmse < dummy_rmse and cand_mae < dummy_mae

        # Per-platform held-out evaluation diagnostics (avoids tiny split-only models while exposing platform scale differences)
        platform_eval_metrics: Dict[str, Dict[str, Any]] = {}
        if "platform" in test_df.columns:
            for plat_name in sorted(clean_df["platform"].dropna().astype(str).str.strip().str.lower().unique()):
                tr_mask = (train_df["platform"].astype(str).str.strip().str.lower() == plat_name).to_numpy()
                te_mask = (test_df["platform"].astype(str).str.strip().str.lower() == plat_name).to_numpy()
                n_te_p = int(te_mask.sum())
                n_tr_p = int(tr_mask.sum())
                if n_te_p >= 3:
                    yt_p = y_test.to_numpy()[te_mask]
                    yp_p = np.asarray(y_pred)[te_mask]
                    yd_p = np.asarray(y_dummy)[te_mask]
                    p_mae = float(mean_absolute_error(yt_p, yp_p))
                    p_rmse = float(np.sqrt(mean_squared_error(yt_p, yp_p)))
                    d_mae = float(mean_absolute_error(yt_p, yd_p))
                    p_rho, _ = spearmanr(yt_p, yp_p)
                    p_rho_val = round(float(p_rho), 3) if pd.notna(p_rho) else None
                    platform_eval_metrics[plat_name] = {
                        "status": "evaluated",
                        "train_samples": n_tr_p,
                        "test_samples": n_te_p,
                        "candidate_mae": round(p_mae, 2),
                        "candidate_rmse": round(p_rmse, 2),
                        "dummy_mae": round(d_mae, 2),
                        "spearman_rho": p_rho_val,
                        "note": f"Evaluated on N={n_te_p} held-out '{plat_name}' posts from the chronological test split.",
                    }
                else:
                    platform_eval_metrics[plat_name] = {
                        "status": "insufficient_evaluation_samples",
                        "train_samples": n_tr_p,
                        "test_samples": n_te_p,
                        "candidate_mae": None,
                        "candidate_rmse": None,
                        "dummy_mae": None,
                        "spearman_rho": None,
                        "note": (
                            f"Insufficient held-out evaluation samples for platform '{plat_name}' (test N={n_te_p} < 3); "
                            "platform-specific error metrics omitted rather than training an unreliable small-sample model."
                        ),
                    }

        # Empirical residual quantile for approximate error band (95th percentile absolute residual)
        y_train_pred = pipeline.predict(X_train)
        train_residuals = np.abs(y_train - y_train_pred)
        empirical_error_q95 = float(np.quantile(train_residuals, 0.95)) if len(train_residuals) > 0 else float(cand_rmse * 1.96)

        # 8. Feature Importance / Coefficients extraction
        feature_importance_dict = {}
        try:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            if hasattr(regressor, "coef_"):
                coefs = regressor.coef_
                for name, coef in zip(feature_names, coefs):
                    clean_name = name.replace("cat__", "").replace("num__", "")
                    feature_importance_dict[clean_name] = round(float(coef), 3)
            elif hasattr(regressor, "feature_importances_"):
                importances = regressor.feature_importances_
                for name, imp in zip(feature_names, importances):
                    clean_name = name.replace("cat__", "").replace("num__", "")
                    feature_importance_dict[clean_name] = round(float(imp), 3)
            feature_importance_dict = dict(
                sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
            )
        except Exception:
            feature_importance_dict = {}

        self.model = pipeline
        self.metadata = {
            "model_type": model_type,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset_provenance": "synthetic_illustrative_benchmark",
            "provenance_statement": (
                "Synthetic benchmark dataset generated for architectural pipeline demonstration. "
                "Evaluation demonstrates pipeline mechanics and statistical separation, not empirical real-world validation."
            ),
            "target_observation_window": (
                f"Settled snapshot engagement after at least {min_elapsed_days:.0f} days since publication. "
                "Filtered out immature posts under 7 days of observation."
            ),
            "total_mature_posts": len(clean_df),
            "train_samples_count": len(X_train),
            "test_samples_count": len(X_test),
            "train_date_range": (str(train_df["timestamp"].min()), str(train_df["timestamp"].max())),
            "test_date_range": (str(test_df["timestamp"].min()), str(test_df["timestamp"].max())),
            "baseline_metrics": {
                "mae": round(dummy_mae, 2),
                "rmse": round(dummy_rmse, 2),
                "r2": round(dummy_r2, 3)
            },
            "candidate_metrics": {
                "mae": round(cand_mae, 2),
                "rmse": round(cand_rmse, 2),
                "r2": round(cand_r2, 3),
                "spearman_rho": round(spear_rho, 3),
                "spearman_pvalue": round(spear_pval, 4)
            },
            "mae_lift_pct": mae_lift_pct,
            "rmse_lift_pct": rmse_lift_pct,
            "beats_baseline": beats_baseline,
            "platform_evaluation_metrics": platform_eval_metrics,
            "empirical_residual_q95": round(empirical_error_q95, 2),
            "top_features": feature_importance_dict,
            "historical_scores_reference": {
                "mean": round(float(clean_df[target_col].mean()), 2),
                "std": round(float(clean_df[target_col].std()), 2),
                "p25": round(float(clean_df[target_col].quantile(0.25)), 2),
                "p50": round(float(clean_df[target_col].quantile(0.50)), 2),
                "p75": round(float(clean_df[target_col].quantile(0.75)), 2)
            },
            "methodological_limitations": {
                "selection_optimism_risk": (
                    "Model selection (choosing Ridge over tree ensembles) was conducted on the same held-out test split. "
                    "For streaming real-world data, rolling-origin cross-validation (walk-forward evaluation) is recommended."
                ),
                "prediction_interval_validity": (
                    "Reported intervals represent approximate empirical residual dispersion bands (+/- q95 or +/- 1.96*RMSE). "
                    "They are not guaranteed conformal prediction intervals with formal finite-sample coverage guarantees."
                ),
                "sample_size": f"N={len(clean_df)} mature posts. Indicative prototype only."
            }
        }

        return self.metadata

    def train_predictive_model(self, df: pd.DataFrame, **kwargs) -> Pipeline:
        """Backwards-compatible wrapper that trains the predictive model and returns the pipeline."""
        self.train_and_evaluate(df, model_type="ridge")
        return self.model

    def predict_pre_publication(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Predict expected Engagement Virality Index for unreleased drafts using pre-publication features."""
        df_input = df_features.copy()

        # 1. Assert no outcome leakage (checked FIRST)
        for col in df_input.columns:
            if is_outcome_leakage_column(col):
                raise ValueError(f"Target leakage detected! Outcome metric '{col}' cannot be in prediction input.")

        # 2. Check model readiness
        if self.model is None:
            raise ValueError("Predictive model has not been trained yet. Call train_and_evaluate() first.")

        # Ensure required pre-publication features exist (impute default if missing)
        if "posting_day_of_week" not in df_input.columns:
            df_input["posting_day_of_week"] = "Monday"
        if "follower_count" not in df_input.columns:
            df_input["follower_count"] = 35000
        if "duration_seconds" not in df_input.columns:
            df_input["duration_seconds"] = 30
        if "posting_hour" not in df_input.columns:
            df_input["posting_hour"] = 18

        predictions = self.model.predict(df_input)

        # Retrieve empirical residual dispersion for error band
        test_rmse = self.metadata.get("candidate_metrics", {}).get("rmse", 32.8)
        empirical_q95 = self.metadata.get("empirical_residual_q95", test_rmse * 1.96)
        ref = self.metadata.get("historical_scores_reference", {})
        p25 = ref.get("p25", 48.0)
        p50 = ref.get("p50", 77.0)
        p75 = ref.get("p75", 119.0)

        results = []
        for pred in predictions:
            pred_score = round(max(float(pred), 0.0), 2)
            band_lower = round(max(pred_score - empirical_q95, 0.0), 2)
            band_upper = round(pred_score + empirical_q95, 2)

            # Assign percentile tier relative to historical reference
            if pred_score >= p75:
                tier = "Top 25% (High Virality Potential)"
            elif pred_score >= p50:
                tier = "Above Average (50-75th Percentile)"
            elif pred_score >= p25:
                tier = "Moderate (25-50th Percentile)"
            else:
                tier = "Below Average (<25th Percentile)"

            results.append({
                "predicted_engagement_virality_index": pred_score,
                "approximate_error_band": (band_lower, band_upper),
                "error_band_type": "empirical_residual_dispersion_band",
                "predicted_performance_tier": tier,
                "model_used": self.metadata.get("model_type", "ridge"),
                "is_comparable_cross_platform": False
            })

        return pd.DataFrame(results, index=df_features.index)

    def predict_virality(self, df_features: pd.DataFrame) -> np.ndarray:
        """Backwards-compatible wrapper returning array of predictions."""
        if self.model is None:
            raise ValueError("Predictive model has not been trained yet. Call train_and_evaluate() first.")
        return self.model.predict(df_features)

    def save_model_pipeline(
        self,
        model_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        include_volatile_timestamps: bool = False,
    ) -> Path:
        """Serialize the trained pipeline and metadata to disk.

        By default (``include_volatile_timestamps=False``), runtime-variable
        execution timestamps such as ``trained_at`` are omitted from the
        serialized JSON metadata so tracked model metadata files remain
        byte-for-byte reproducible across unchanged inputs. Truthful
        training timestamps remain available in ``self.metadata`` and in the
        untracked build provenance manifest.
        """
        if self.model is None:
            raise ValueError("No model trained to save.")
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)

        if metadata_path is not None:
            metadata_path = Path(metadata_path)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            serializable_meta = dict(self.metadata)
            if not include_volatile_timestamps:
                serializable_meta.pop("trained_at", None)
                serializable_meta["metadata_reproducibility"] = {
                    "artifact_type": "deterministic_model_metadata",
                    "intentionally_variable_fields_excluded": ["trained_at"],
                    "runtime_provenance_location": "models/build_provenance.json",
                }
            with open(metadata_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(serializable_meta, f, indent=2)
                f.write("\n")

        return model_path

    def load_model_pipeline(self, model_path: Union[str, Path], metadata_path: Optional[Union[str, Path]] = None) -> "ViralityPredictor":
        """Load a serialized pipeline and metadata from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model = joblib.load(model_path)

        if metadata_path is not None and Path(metadata_path).exists():
            import json
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        return self

    def evaluate_tiktok_longitudinal(
        self,
        tiktok_posts_df: pd.DataFrame,
        *,
        target_horizon: str = "day30",
        feature_regime: str = "pre_pub",
        split_strategy: str = "chronological",
        test_ratio: float = 0.2,
        exclude_negative_reset_rows: bool = False,
        sample_limit: Optional[int] = None,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Evaluate pre-publication or early Day-1 predictors on real TikTok longitudinal cohorts.

        Parameters
        ----------
        target_horizon:
            ``'day7'`` or ``'day30'``. Predicts ``log1p(day{7|30}_views)``.
        feature_regime:
            ``'pre_pub'`` (strictly creator baseline, metadata, text/topic features only)
            or ``'early_day1'`` (adds ``day1_*`` engagement velocity signals; prohibited for draft scoring).
        split_strategy:
            ``'chronological'`` (strictly ``train_max < test_min``) or
            ``'creator_grouped'`` (disjoint ``creator_pseudo_id`` sets via GroupShuffleSplit).
        """
        horizon_norm = str(target_horizon).strip().lower()
        if horizon_norm not in {"day7", "day30"}:
            raise ValueError(f"Unsupported target_horizon '{target_horizon}'. Choose 'day7' or 'day30'.")
        regime_norm = str(feature_regime).strip().lower()
        if regime_norm not in {"pre_pub", "early_day1"}:
            raise ValueError(f"Unsupported feature_regime '{feature_regime}'. Choose 'pre_pub' or 'early_day1'.")
        split_norm = str(split_strategy).strip().lower()
        if split_norm not in {"chronological", "creator_grouped"}:
            raise ValueError(f"Unsupported split_strategy '{split_strategy}'. Choose 'chronological' or 'creator_grouped'.")

        target_raw_col = f"{horizon_norm}_views"
        if target_raw_col not in tiktok_posts_df.columns:
            raise ValueError(f"Missing target column '{target_raw_col}' in TikTok posts DataFrame.")

        work = tiktok_posts_df.copy()
        work["_target_views"] = pd.to_numeric(work[target_raw_col], errors="coerce")
        work = work.loc[work["_target_views"].notna() & (work["_target_views"] >= 0)].copy()

        reset_rows_total = int(work["has_negative_delta_reset"].fillna(False).sum()) if "has_negative_delta_reset" in work.columns else 0
        if exclude_negative_reset_rows and "has_negative_delta_reset" in work.columns:
            work = work.loc[~work["has_negative_delta_reset"].fillna(False)].copy()

        if sample_limit is not None and len(work) > int(sample_limit):
            work = work.sort_values("timestamp", kind="mergesort").tail(int(sample_limit)).reset_index(drop=True)

        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
        work = work.dropna(subset=["timestamp"]).copy()
        if len(work) < self.MIN_TRAINING_SAMPLES:
            raise ValueError(f"Insufficient mature TikTok observations ({len(work)} < {self.MIN_TRAINING_SAMPLES}).")

        work["_y_log1p"] = np.log1p(work["_target_views"].astype(float))
        work["posting_hour"] = work["timestamp"].dt.hour
        work["posting_day_of_week"] = work["timestamp"].dt.day_name().astype(str)

        # Derive log1p numerical transforms strictly within allowed regime
        num_features: List[str] = []
        raw_prepub_nums = [
            "duration_seconds",
            "posting_hour",
            "word_count",
            "emoji_count",
            "question_count",
            "hashtag_count",
            "speaking_rate",
            "is_english",
            "created_by_ai",
        ]
        log_prepub_nums = ["follower_count", "creator_video_count", "creator_total_favorited"]

        for col in raw_prepub_nums:
            if col in work.columns:
                if is_outcome_leakage_column(col):
                    raise ValueError(f"Target leakage detected in pre-pub numerical feature '{col}'.")
                work[col] = pd.to_numeric(work[col], errors="coerce")
                num_features.append(col)

        for col in log_prepub_nums:
            if col in work.columns:
                if is_outcome_leakage_column(col):
                    raise ValueError(f"Target leakage detected in pre-pub creator baseline feature '{col}'.")
                vals = pd.to_numeric(work[col], errors="coerce").clip(lower=0)
                derived_name = f"{col}_log1p"
                work[derived_name] = np.log1p(vals)
                num_features.append(derived_name)

        if regime_norm == "early_day1":
            for d1_col in ["day1_views", "day1_likes", "day1_comments", "day1_shares", "day1_collects"]:
                if d1_col in work.columns:
                    vals = pd.to_numeric(work[d1_col], errors="coerce").clip(lower=0)
                    derived_name = f"{d1_col}_log1p"
                    work[derived_name] = np.log1p(vals)
                    num_features.append(derived_name)

        cat_features = [c for c in ["topic", "posting_day_of_week"] if c in work.columns]
        if regime_norm == "pre_pub":
            for c in cat_features + num_features:
                if is_outcome_leakage_column(c):
                    raise ValueError(f"Target leakage guard triggered for '{c}' in pre_pub regime.")

        all_features = cat_features + num_features
        if not all_features:
            raise ValueError("No valid predictive features found in TikTok dataset.")

        for c in cat_features:
            work[c] = work[c].fillna("Unknown").astype(str)

        # Execute split without lookahead or creator overlap
        if split_norm == "chronological":
            work = work.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
            split_idx = int(len(work) * (1.0 - test_ratio))
            boundary_ts = work.loc[split_idx, "timestamp"]
            # Push split back cleanly so strict inequality train_max < test_min holds even with ties
            while split_idx > 1 and work.loc[split_idx - 1, "timestamp"] >= boundary_ts:
                split_idx -= 1
            train_df = work.iloc[:split_idx].copy()
            test_df = work.iloc[split_idx:].copy()
            if len(train_df) == 0 or len(test_df) == 0:
                raise ValueError("Chronological split collapsed due to identical timestamps.")
            if not (train_df["timestamp"].max() < test_df["timestamp"].min()):
                raise ValueError("Chronological split boundary overlap detected; insufficient distinct timestamps.")
            creator_overlap_count = (
                len(set(train_df["creator_pseudo_id"].dropna()) & set(test_df["creator_pseudo_id"].dropna()))
                if "creator_pseudo_id" in work.columns
                else None
            )
        else:
            if "creator_pseudo_id" not in work.columns or work["creator_pseudo_id"].dropna().nunique() < 5:
                raise ValueError("creator_grouped split requires >= 5 distinct 'creator_pseudo_id' values.")
            work = work.dropna(subset=["creator_pseudo_id"]).reset_index(drop=True)
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
            tr_idx, te_idx = next(splitter.split(work, work["_y_log1p"], groups=work["creator_pseudo_id"]))
            train_df = work.iloc[tr_idx].sort_values("timestamp", kind="mergesort").copy()
            test_df = work.iloc[te_idx].sort_values("timestamp", kind="mergesort").copy()
            creator_overlap_count = len(set(train_df["creator_pseudo_id"]) & set(test_df["creator_pseudo_id"]))
            if creator_overlap_count != 0:
                raise ValueError(f"Creator group leakage detected ({creator_overlap_count} overlapping creators).")

        X_train = train_df[all_features]
        y_train = train_df["_y_log1p"].to_numpy(dtype=float)
        X_test = test_df[all_features]
        y_test = test_df["_y_log1p"].to_numpy(dtype=float)

        # Split an internal calibration holdout (15% of train_df) respecting the active split structure:
        # - creator_grouped: GroupShuffleSplit on creator_pseudo_id with zero creator overlap
        # - chronological: strict temporal holdout without timestamp boundary ties
        # Never fall back silently to in-sample X_fit residuals when calibration is infeasible.
        calibration_available = True
        calibration_creator_overlap_count: Optional[int] = None
        calibration_note = ""
        fit_df = train_df
        cal_df = train_df.iloc[0:0]
        X_fit = X_train
        y_fit = y_train
        X_cal = X_train.iloc[0:0]
        y_cal = y_train[:0]

        if split_norm == "creator_grouped":
            n_train_creators = int(train_df["creator_pseudo_id"].dropna().nunique())
            if n_train_creators >= 2 and len(train_df) >= 2:
                cal_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
                fit_idx, cal_idx = next(
                    cal_splitter.split(train_df, train_df["_y_log1p"], groups=train_df["creator_pseudo_id"])
                )
                fit_df = train_df.iloc[fit_idx].copy()
                cal_df = train_df.iloc[cal_idx].copy()
                cal_overlap = len(set(fit_df["creator_pseudo_id"]) & set(cal_df["creator_pseudo_id"]))
                calibration_creator_overlap_count = int(cal_overlap)
                if cal_overlap != 0 or len(fit_df) == 0 or len(cal_df) == 0:
                    calibration_available = False
                    calibration_note = (
                        f"Creator-grouped calibration infeasible (overlap={cal_overlap}, fit={len(fit_df)}, cal={len(cal_df)}); "
                        "omitting interval calibration rather than using in-sample residuals."
                    )
                else:
                    X_fit = fit_df[all_features]
                    y_fit = fit_df["_y_log1p"].to_numpy(dtype=float)
                    X_cal = cal_df[all_features]
                    y_cal = cal_df["_y_log1p"].to_numpy(dtype=float)
                    calibration_note = (
                        f"Creator-grouped fit/calibration holdout via GroupShuffleSplit with zero creator overlap "
                        f"({fit_df['creator_pseudo_id'].nunique()} fit creators, {cal_df['creator_pseudo_id'].nunique()} cal creators)."
                    )
            else:
                calibration_available = False
                calibration_note = (
                    f"Insufficient distinct creators in training partition (n={n_train_creators}) for creator-grouped calibration; "
                    "omitting interval calibration rather than falling back to in-sample residuals."
                )
        else:
            cal_split = max(int(len(train_df) * 0.85), 1)
            if 0 < cal_split < len(train_df):
                cal_boundary_ts = train_df.iloc[cal_split]["timestamp"]
                while cal_split > 1 and train_df.iloc[cal_split - 1]["timestamp"] >= cal_boundary_ts:
                    cal_split -= 1
                if train_df.iloc[cal_split - 1]["timestamp"] >= train_df.iloc[cal_split]["timestamp"]:
                    alt_split = max(int(len(train_df) * 0.85), 1)
                    while alt_split < len(train_df) and train_df.iloc[alt_split]["timestamp"] <= train_df.iloc[alt_split - 1]["timestamp"]:
                        alt_split += 1
                    cal_split = alt_split if alt_split < len(train_df) else 0
            else:
                cal_split = 0

            fit_df = train_df.iloc[:cal_split].copy()
            cal_df = train_df.iloc[cal_split:].copy()
            if len(fit_df) == 0 or len(cal_df) == 0 or not (fit_df["timestamp"].max() < cal_df["timestamp"].min()):
                calibration_available = False
                calibration_note = (
                    "Chronological calibration holdout collapsed due to tied timestamps; "
                    "omitting interval calibration rather than falling back to in-sample residuals."
                )
            else:
                X_fit = fit_df[all_features]
                y_fit = fit_df["_y_log1p"].to_numpy(dtype=float)
                X_cal = cal_df[all_features]
                y_cal = cal_df["_y_log1p"].to_numpy(dtype=float)
                calibration_creator_overlap_count = (
                    len(set(fit_df["creator_pseudo_id"].dropna()) & set(cal_df["creator_pseudo_id"].dropna()))
                    if "creator_pseudo_id" in train_df.columns
                    else None
                )
                calibration_note = (
                    f"Chronological 85/15 fit/calibration split with strict timestamp boundary "
                    f"(fit_max={fit_df['timestamp'].max()}, cal_min={cal_df['timestamp'].min()})."
                )

        def _make_prep() -> ColumnTransformer:
            transformers = []
            if cat_features:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features))
            if num_features:
                transformers.append(
                    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_features)
                )
            return ColumnTransformer(transformers=transformers)

        candidates = {
            "dummy_median": DummyRegressor(strategy="median"),
            "ridge": Ridge(alpha=10.0),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                max_iter=80, max_depth=6, learning_rate=0.08, random_state=random_state
            ),
        }

        models_report: Dict[str, Dict[str, Any]] = {}
        for m_name, reg in candidates.items():
            # Calibration-fit for empirical 90% coverage check (strictly held-out; never falls back to X_fit)
            q90_band: Optional[float] = None
            covered: Optional[float] = None
            if calibration_available and len(y_cal) > 0:
                cal_pipe = Pipeline([("prep", _make_prep()), ("reg", clone(reg))])
                cal_pipe.fit(X_fit, y_fit)
                cal_res = np.abs(y_cal - cal_pipe.predict(X_cal))
                if len(cal_res) > 0:
                    q90_val = float(np.quantile(cal_res, 0.90))
                    q90_band = round(q90_val, 4)

            # Full train fit for test evaluation
            full_pipe = Pipeline([("prep", _make_prep()), ("reg", clone(reg))])
            full_pipe.fit(X_train, y_train)
            preds = full_pipe.predict(X_test)

            mae_log = float(mean_absolute_error(y_test, preds))
            rmse_log = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2_log = float(r2_score(y_test, preds))
            if len(y_test) > 1 and np.std(preds) > 1e-12 and np.std(y_test) > 1e-12:
                rho, _ = spearmanr(y_test, preds)
                rho_val = float(rho) if pd.notna(rho) else 0.0
            else:
                rho_val = 0.0
            if q90_band is not None and len(y_test) > 0:
                covered = round(float(np.mean(np.abs(y_test - preds) <= q90_val) * 100.0), 2)

            y_true_views = np.expm1(y_test)
            y_pred_views = np.expm1(np.clip(preds, 0.0, 20.0))
            med_abs_err_views = float(np.median(np.abs(y_true_views - y_pred_views)))

            models_report[m_name] = {
                "mae_log1p": round(mae_log, 4),
                "rmse_log1p": round(rmse_log, 4),
                "r2_log1p": round(r2_log, 4),
                "spearman_rho": round(rho_val, 4),
                "median_abs_error_views": round(med_abs_err_views, 1),
                "empirical_q90_log1p_radius": q90_band,
                "empirical_90pct_test_coverage": covered,
            }

        best_model_name = min(
            [k for k in models_report if k != "dummy_median"],
            key=lambda k: models_report[k]["rmse_log1p"],
        )
        dummy_rmse = models_report["dummy_median"]["rmse_log1p"]
        best_rmse = models_report[best_model_name]["rmse_log1p"]
        rmse_lift_pct = round(((dummy_rmse - best_rmse) / max(dummy_rmse, 1e-9)) * 100.0, 2)

        limitations = [
            "TikTok creator_total_favorited and follower_count reflect creator snapshot metadata and can correlate across videos from the same account.",
            "Creator-grouped splits are substantially harder than random/within-creator splits because creator baseline scale is unseen.",
            "Empirical 90% prediction bands are split-residual approximations without finite-sample conformal exchangeability guarantees under distribution shift.",
        ]
        if not calibration_available:
            limitations.append(f"Prediction interval calibration omitted: {calibration_note}")

        return {
            "dataset_id": "tiktok_engagement_200k",
            "provenance": SourceProvenance.REAL_PUBLIC_DATA.value,
            "platform": "tiktok",
            "target_column": target_raw_col,
            "target_transform": "log1p",
            "feature_regime": regime_norm,
            "split_strategy": split_norm,
            "features_used": all_features,
            "total_evaluated_rows": int(len(work)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "unique_creators": int(work["creator_pseudo_id"].nunique(dropna=True)) if "creator_pseudo_id" in work.columns else None,
            "creator_overlap_count": creator_overlap_count,
            "calibration_strategy": "creator_grouped" if split_norm == "creator_grouped" else "chronological",
            "calibration_available": bool(calibration_available),
            "calibration_fit_rows": int(len(fit_df)) if calibration_available else 0,
            "calibration_rows": int(len(cal_df)) if calibration_available else 0,
            "calibration_creator_overlap_count": calibration_creator_overlap_count,
            "calibration_note": calibration_note,
            "negative_delta_reset_rows_in_cohort": reset_rows_total,
            "exclude_negative_reset_rows": bool(exclude_negative_reset_rows),
            "train_date_range": [str(train_df["timestamp"].min()), str(train_df["timestamp"].max())],
            "test_date_range": [str(test_df["timestamp"].min()), str(test_df["timestamp"].max())],
            "models": models_report,
            "best_model": best_model_name,
            "rmse_lift_over_dummy_pct": rmse_lift_pct,
            "limitations": limitations,
        }
