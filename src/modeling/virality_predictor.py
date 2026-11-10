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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    "equivalent_full_watch_views"
}


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

        # 4. Retention factor adjustment (video-only)
        retention = row.get("retention_rate")
        if pd.notna(retention) and retention is not None and 0.0 <= float(retention) <= 1.0:
            phi = 0.5 + float(retention)
        else:
            phi = 1.0  # Neutral multiplier if retention is unavailable or static media

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
        """Calculate availability-aware EVI with comprehensive audit tracking."""
        results = [
            self.calculate_single_post_score(row, denominator_mode=denominator_mode)
            for _, row in df.iterrows()
        ]
        return pd.DataFrame(results, index=df.index)

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
        df_calc = df.copy()
        if "engagement_virality_index" not in df_calc.columns and "viral_coefficient" not in df_calc.columns:
            df_calc["engagement_virality_index"] = self.calculate_engagement_virality_index(df_calc)
        elif "engagement_virality_index" not in df_calc.columns:
            df_calc["engagement_virality_index"] = df_calc["viral_coefficient"]

        # Drop uncalculated NaN scores to prevent skewing
        valid_df = df_calc.dropna(subset=["engagement_virality_index"])
        if valid_df.empty:
            return pd.DataFrame()

        ranking = valid_df.groupby("topic").agg(
            posts_count=("post_id", "count"),
            avg_viral_coefficient=("engagement_virality_index", "mean"),
            avg_shares=("shares", "mean"),
            avg_saves=("saves", "mean"),
            avg_retention=("retention_rate", "mean"),
            total_views=("views", "sum")
        ).reset_index()

        ranking["avg_viral_coefficient"] = ranking["avg_viral_coefficient"].round(2)
        ranking = ranking.sort_values(by="avg_viral_coefficient", ascending=False).reset_index(drop=True)
        return ranking

    def normalize_target_by_platform(
        self,
        df: pd.DataFrame,
        target_col: str = "engagement_virality_index"
    ) -> pd.Series:
        """Compute within-platform percentile ranks [0, 100] for platform-fair comparison."""
        df_calc = df.copy()
        if target_col not in df_calc.columns:
            df_calc[target_col] = self.calculate_engagement_virality_index(df_calc)

        percentiles = pd.Series(index=df_calc.index, dtype="float64")
        for platform, group in df_calc.groupby("platform"):
            valid = group[target_col].dropna()
            if len(valid) > 0:
                ranks = valid.rank(pct=True) * 100.0
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
            if f in OUTCOME_METRICS:
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
            if col in OUTCOME_METRICS:
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
