"""Module 7: Trend Forecasting Engine.

Implements rigorous, time-aware forecasting for social media publication cohort indicators
and emerging relatable struggle topics.

Methodological Highlights & Audit Reconciliations:
1. Target Semantics & Exposure-Age Bias: Targets publication-cohort lifetime performance
   (lifetime metrics accrued by content assets published in interval t), distinguished from account-level
   daily traffic. Unequal accumulation window (exposure-age bias) is explicitly disclosed and treated
   strictly as exploratory descriptive analysis.
2. Unverified Coverage & Missing-Data: Distinguishes periods with no published posts (post_count=0) from
   missing source data; continuous calendar range is not claimed as verified 100% coverage.
   Rate metrics are undefined (NaN) on zero-views periods and cannot be converted to zero.
3. YouTube Metric Semantics: Disallows 4-component total_engagement on YouTube (saves are unavailable);
   provides 3-component 'youtube_available_interactions' (Likes + Comments + Shares).
4. Genuine Rolling-Origin Walk-Forward Validation: Multiple chronological forecast origins,
   models refit strictly on preceding data, out-of-sample predictions and residuals tracked per origin
   and horizon step.
5. Horizon Uncertainty Differentiation: Distinguishes direct empirical quantiles (steps <= H_eval)
   from extrapolated ranges (steps > H_eval) with explicit uncalibrated demonstration disclosures.
6. Robust Short History Handling: Explicit guard for keyword histories N in [14..16] returning
   INSUFFICIENT_VALIDATION_HISTORY without crashing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from src.preprocessing.cleaning import get_latest_posts_view


# --- 1. ENUMS & DATA SCHEMAS ---

class ForecastCadence(str, Enum):
    """Supported time-series aggregation cadences."""
    DAILY = "D"
    WEEKLY = "W"


class ForecastModelType(str, Enum):
    """Evaluated time-series forecasting models."""
    NAIVE = "Naive (Persistence)"
    SEASONAL_NAIVE = "Seasonal Naive"
    MOVING_AVERAGE = "Moving Average"
    EXPONENTIAL_SMOOTHING = "Holt Exponential Smoothing"
    AUTOREGRESSIVE = "AutoRegressive Ridge (AR)"


class ValidationStatus(str, Enum):
    """Chronological validation and data sufficiency status."""
    VALID = "VALID"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INSUFFICIENT_VALIDATION_HISTORY = "INSUFFICIENT_VALIDATION_HISTORY"
    SPARSE_SERIES = "SPARSE_SERIES"


@dataclass
class ModelEvaluationMetric:
    """Performance evaluation record from rolling-origin walk-forward validation."""
    model_name: str
    model_type: ForecastModelType
    mae: Optional[float]
    rmse: Optional[float]
    mase: Optional[float]
    relative_mae_to_naive: Optional[float]
    is_champion: bool
    validation_origins: int
    validation_observations: int
    successful_origins: int = 0
    failed_origins: int = 0
    attempted_origins: int = 0
    is_eligible_for_champion: bool = True
    evaluation_status: str = "Evaluated"
    horizon_step_mae: Dict[int, float] = field(default_factory=dict)
    notes: str = ""
    step_error_spreads: Dict[int, float] = field(default_factory=dict)
    step_error_methods: Dict[int, str] = field(default_factory=dict)
    step_error_counts: Dict[int, int] = field(default_factory=dict)


@dataclass
class ForecastPoint:
    """Discrete out-of-sample projection point with empirical uncertainty bounds."""
    step: int
    timestamp: str
    point_forecast: float
    ci_lower: float
    ci_upper: float
    empirical_error_spread: float = 0.0
    uncertainty_method: str = "Direct Empirical Quantile"
    residual_count: int = 0


@dataclass
class TrendForecastReport:
    """Comprehensive, serialized output of the Trend Forecasting Engine."""
    target_metric: str
    target_description: str
    platform: str
    cadence: str
    history_start: str
    history_end: str
    history_points: int
    active_observation_days: int
    zero_publication_days: int
    missing_calendar_days: Optional[int]
    coverage_pct: Optional[float]  # Represents Observed Calendar-Date Continuity (%) for keywords; None for posts
    coverage_status: str
    forecast_horizon: int
    champion_model: str
    selected_model: str
    is_manual_override: bool
    validation_origins_count: int
    validation_observations_count: int
    model_evaluations: List[ModelEvaluationMetric]
    forecast_points: List[ForecastPoint]
    historical_df: pd.DataFrame
    forecast_df: pd.DataFrame
    uncertainty_description: str
    is_calibrated: bool
    status: ValidationStatus
    disclosures: List[str] = field(default_factory=list)
    calendar_continuity_pct: Optional[float] = None


# --- 2. FORECASTING MODEL IMPLEMENTATIONS ---

class BaseTimeSeriesModel(ABC):
    """Abstract base class for deterministic univariate time-series forecasters."""

    @abstractmethod
    def fit(self, y: np.ndarray) -> "BaseTimeSeriesModel":
        """Fit model parameters to in-sample historical observations."""
        pass

    @abstractmethod
    def predict(self, h: int = 1, horizon: Optional[int] = None) -> np.ndarray:
        """Generate out-of-sample forecasts for horizon h."""
        pass


class NaiveModel(BaseTimeSeriesModel):
    """Naive persistence baseline: forecasts last observed value."""

    def __init__(self):
        self.last_value: float = 0.0

    def fit(self, y: np.ndarray) -> "NaiveModel":
        if len(y) == 0:
            raise ValueError("Cannot fit NaiveModel on empty series.")
        self.last_value = float(y[-1])
        return self

    def predict(self, h: int = 1, horizon: Optional[int] = None) -> np.ndarray:
        steps = horizon if horizon is not None else h
        return np.full(steps, self.last_value, dtype=float)


class SeasonalNaiveModel(BaseTimeSeriesModel):
    """Seasonal naive baseline: forecasts value from equivalent season m periods prior."""

    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.history: np.ndarray = np.array([])

    def fit(self, y: np.ndarray) -> "SeasonalNaiveModel":
        if len(y) == 0:
            raise ValueError("Cannot fit SeasonalNaiveModel on empty series.")
        self.history = np.asarray(y, dtype=float)
        return self

    def predict(self, h: int = 1, horizon: Optional[int] = None) -> np.ndarray:
        steps = horizon if horizon is not None else h
        if steps <= 0:
            return np.empty(0, dtype=float)
        if len(self.history) < 2 * self.season_length:
            return np.full(steps, self.history[-1], dtype=float)

        indices = (-self.season_length) + (np.arange(steps) % self.season_length)
        return self.history[indices].astype(float)


class MovingAverageModel(BaseTimeSeriesModel):
    """Moving average baseline: forecasts mean of the trailing k observations."""

    def __init__(self, window: int = 7):
        self.window = window
        self.ma_value: float = 0.0

    def fit(self, y: np.ndarray) -> "MovingAverageModel":
        if len(y) == 0:
            raise ValueError("Cannot fit MovingAverageModel on empty series.")
        k = min(self.window, len(y))
        self.ma_value = float(np.mean(y[-k:]))
        return self

    def predict(self, h: int = 1, horizon: Optional[int] = None) -> np.ndarray:
        steps = horizon if horizon is not None else h
        return np.full(steps, self.ma_value, dtype=float)


class HoltExponentialSmoothingModel(BaseTimeSeriesModel):
    """Holt's Linear Exponential Smoothing with optimal level and trend parameters."""

    def __init__(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        dampening_factor: float = 0.95
    ):
        self.alpha = alpha
        self.beta = beta
        self.dampening_factor = dampening_factor
        self.level: float = 0.0
        self.trend: float = 0.0

    def fit(self, y: np.ndarray) -> "HoltExponentialSmoothingModel":
        n = len(y)
        if n < 2:
            raise ValueError("Holt's model requires at least 2 observations.")

        # Unbox to native Python floats once so inner L-BFGS-B evaluations avoid numpy scalar overhead
        series_list = [float(v) for v in y]
        s0 = series_list[0]
        t0 = series_list[1] - s0
        series_tail = series_list[1:]

        def sse_objective(params: np.ndarray) -> float:
            a = float(params[0])
            b = float(params[1])
            one_minus_a = 1.0 - a
            one_minus_b = 1.0 - b
            lvl = s0
            trd = t0
            sse = 0.0
            for val in series_tail:
                pred = lvl + trd
                diff = val - pred
                sse += diff * diff
                new_lvl = a * val + one_minus_a * pred
                trd = b * (new_lvl - lvl) + one_minus_b * trd
                lvl = new_lvl
            return sse

        try:
            if self.alpha is None or self.beta is None:
                res = minimize(
                    sse_objective,
                    x0=[0.3, 0.1],
                    bounds=[(0.01, 0.99), (0.01, 0.99)],
                    method="L-BFGS-B"
                )
                if res.success and np.all(np.isfinite(res.x)):
                    opt_alpha, opt_beta = float(res.x[0]), float(res.x[1])
                else:
                    opt_alpha, opt_beta = 0.3, 0.1
            else:
                opt_alpha, opt_beta = float(self.alpha), float(self.beta)
        except Exception:
            opt_alpha, opt_beta = 0.3, 0.1

        one_minus_a = 1.0 - opt_alpha
        one_minus_b = 1.0 - opt_beta
        lvl = s0
        trd = t0
        for val in series_tail:
            pred = lvl + trd
            new_lvl = opt_alpha * val + one_minus_a * pred
            trd = opt_beta * (new_lvl - lvl) + one_minus_b * trd
            lvl = new_lvl

        self.alpha = opt_alpha
        self.beta = opt_beta
        self.level = float(lvl)
        self.trend = float(trd)
        return self

    def predict(self, h: int = 1, horizon: Optional[int] = None) -> np.ndarray:
        steps = horizon if horizon is not None else h
        if steps <= 0:
            return np.empty(0, dtype=float)
        damping = self.dampening_factor
        powers = damping ** np.arange(1, steps + 1, dtype=float)
        damped_trend = self.trend * np.cumsum(powers)
        preds = self.level + damped_trend
        return np.maximum(0.0, preds)


class AutoRegressiveRidgeModel(BaseTimeSeriesModel):
    """AutoRegressive model using lagged features with L2 Ridge regularization."""

    def __init__(self, p: int = 3, l2_alpha: float = 1.0):
        self.p = p
        self.l2_alpha = l2_alpha
        self.model = Ridge(alpha=self.l2_alpha, random_state=42)
        self.history: np.ndarray = np.array([])
        self.is_fitted: bool = False
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0

    def fit(self, y: np.ndarray) -> "AutoRegressiveRidgeModel":
        n = len(y)
        effective_p = max(1, min(self.p, n // 3))
        self.p = effective_p

        if n <= self.p + 1:
            raise ValueError(f"AR({self.p}) requires at least {self.p + 2} observations; got {n}.")

        self.history = np.asarray(y, dtype=float)
        X = np.lib.stride_tricks.sliding_window_view(self.history[:-1], self.p)[:, ::-1]
        Y = self.history[self.p:]

        self.model.fit(X, Y)
        self._coef = np.asarray(self.model.coef_, dtype=float)
        self._intercept = float(self.model.intercept_)
        self.is_fitted = True
        return self

    def predict(self, h: int = 1, horizon: Optional[int] = None) -> np.ndarray:
        steps = horizon if horizon is not None else h
        if not self.is_fitted:
            raise RuntimeError("AutoRegressiveRidgeModel must be fitted before predict.")
        if steps <= 0:
            return np.empty(0, dtype=float)

        p = self.p
        coef = getattr(self, "_coef", None)
        if coef is None:
            coef = np.asarray(self.model.coef_, dtype=float)
            intercept = float(self.model.intercept_)
        else:
            intercept = self._intercept

        buf = np.empty(p + steps, dtype=float)
        buf[steps:] = self.history[-p:][::-1]
        for i in range(steps - 1, -1, -1):
            next_val = float(np.dot(coef, buf[i + 1:i + 1 + p]) + intercept)
            if next_val < 0.0:
                next_val = 0.0
            buf[i] = next_val
        return buf[:steps][::-1].copy()


# --- 3. ROLLING-ORIGIN CHRONOLOGICAL WALK-FORWARD VALIDATION ---

def get_rolling_origin_diagnostics(
    series: np.ndarray,
    cadence: str = "D",
    horizon: int = 7,
    min_train_size: int = 14,
    max_origins: int = 10
) -> List[Dict[str, int]]:
    """Instrument and expose the exact rolling-origin training and validation slice indices.

    Verifies that every model fit receives data strictly preceding its validation origin (zero leakage).
    """
    n = len(series)
    if n < min_train_size + 3:
        return []

    num_origins = min(max_origins, n - min_train_size)
    if num_origins < 3:
        return []

    origins = list(range(n - num_origins, n))
    eval_step_horizon = min(horizon, 5)

    diagnostics = []
    for t in origins:
        h_t = min(eval_step_horizon, n - t)
        diagnostics.append({
            "origin_index": t,
            "train_start_idx": 0,
            "train_end_idx": t - 1,
            "train_len": t,
            "test_start_idx": t,
            "test_end_idx": t + h_t - 1,
            "test_len": h_t
        })
    return diagnostics


def compute_horizon_uncertainty(
    step_errors: Dict[int, List[float]],
    horizon: int,
    eval_step_horizon: int,
    mae: float,
) -> Tuple[Dict[int, float], Dict[int, str], Dict[int, int]]:
    """Compute horizon-specific empirical uncertainty spreads, method labels, and residual counts.

    Method hierarchy per forecast step s:
    1. Direct Empirical Quantile (90th percentile of |residuals|) when s <= eval_step_horizon and n_s >= 3.
    2. Evaluation-Horizon Expansion (Spread_{H_eval} * sqrt(s / H_eval)) when s > eval_step_horizon and n_{H_eval} >= 3.
    3. Step-1 Expansion Fallback (Spread_1 * sqrt(s)) when n_s < 3 (or n_{H_eval} < 3) and n_1 >= 3.
    4. MAE Fallback Spread (MAE * 1.645 * sqrt(s)) when n_1 < 3.
    """
    step_error_spread: Dict[int, float] = {}
    step_error_method: Dict[int, str] = {}
    step_error_count: Dict[int, int] = {}

    n_step1 = len(step_errors.get(1, []))
    n_eval_h = len(step_errors.get(eval_step_horizon, []))

    for s in range(1, horizon + 1):
        res_list = step_errors.get(s, [])
        res_count = len(res_list)
        step_error_count[s] = res_count

        if s <= eval_step_horizon and res_count >= 3:
            s_abs = np.abs(res_list)
            spread = float(np.percentile(s_abs, 90))
            method = "Direct Empirical Quantile (90th percentile out-of-sample residual)"
        elif s > eval_step_horizon and n_eval_h >= 3:
            base_spread = float(np.percentile(np.abs(step_errors[eval_step_horizon]), 90))
            spread = float(base_spread * np.sqrt(s / eval_step_horizon))
            method = (
                f"Extrapolated Horizon Range — Evaluation-Horizon Expansion "
                f"(Spread_{eval_step_horizon} * sqrt(h / {eval_step_horizon}), sqrt(h) scaling from step {eval_step_horizon})"
            )
        elif n_step1 >= 3:
            base_spread = float(np.percentile(np.abs(step_errors[1]), 90))
            spread = float(base_spread * np.sqrt(s))
            if s > eval_step_horizon:
                method = (
                    f"Step-1 Expansion Fallback (Spread_1 * sqrt(h) from step 1; "
                    f"insufficient step {eval_step_horizon} residuals: n={n_eval_h})"
                )
            else:
                method = (
                    f"Step-1 Expansion Fallback (Spread_1 * sqrt(h) from step 1; "
                    f"insufficient step {s} residuals: n={res_count})"
                )
        else:
            spread = float(mae * 1.645 * np.sqrt(s))
            method = (
                f"MAE Fallback Spread (MAE * 1.645 * sqrt(h); "
                f"insufficient step-1 and step-{s} residuals: n_1={n_step1}, n_{s}={res_count})"
            )

        step_error_spread[s] = spread
        step_error_method[s] = method

    return step_error_spread, step_error_method, step_error_count


def build_uncertainty_description(
    n_origins: int,
    eval_step_horizon: int,
    forecast_points: List[ForecastPoint],
) -> str:
    """Reconcile uncertainty_description text with the actual per-step calculation methods used."""
    direct_steps = [p.step for p in forecast_points if "Direct Empirical Quantile" in p.uncertainty_method]
    eval_exp_steps = [p.step for p in forecast_points if "Evaluation-Horizon Expansion" in p.uncertainty_method]
    step1_exp_steps = [p.step for p in forecast_points if "Step-1 Expansion" in p.uncertainty_method]
    mae_fallback_steps = [p.step for p in forecast_points if "MAE Fallback" in p.uncertainty_method]

    active_parts: List[str] = []
    if direct_steps:
        active_parts.append(f"steps {direct_steps}: Direct Empirical Quantile (90th percentile of |residual|)")
    if eval_exp_steps:
        active_parts.append(
            f"steps {eval_exp_steps}: Evaluation-Horizon Expansion (Spread_{eval_step_horizon} * sqrt(s / {eval_step_horizon}))"
        )
    if step1_exp_steps:
        active_parts.append(f"steps {step1_exp_steps}: Step-1 Expansion (Spread_1 * sqrt(s))")
    if mae_fallback_steps:
        active_parts.append(f"steps {mae_fallback_steps}: MAE Fallback (MAE * 1.645 * sqrt(s))")

    active_summary = "; ".join(active_parts) if active_parts else "No forecast steps generated"

    return (
        f"Uncertainty Range (Uncalibrated Demonstration Range): Evaluated across {n_origins} rolling origins "
        f"(validation cutoff H_eval={eval_step_horizon}). "
        f"Active calculation by step -> {active_summary}. "
        f"Fallback hierarchy per step s: "
        f"(1) Direct Empirical Quantile when s <= {eval_step_horizon} and residual count n_s >= 3; "
        f"(2) Evaluation-Horizon Expansion (Spread_{eval_step_horizon} * sqrt(s / {eval_step_horizon})) when s > {eval_step_horizon} and n_{eval_step_horizon} >= 3; "
        f"(3) Step-1 Expansion (Spread_1 * sqrt(s)) when n_s < 3 (or n_{eval_step_horizon} < 3) and n_1 >= 3; "
        f"(4) MAE Fallback (MAE * 1.645 * sqrt(s)) when n_1 < 3. "
        f"90th-percentile estimates based on few observations (n < 10) are statistically unstable and exploratory. "
        f"Nominal 95% coverage is NOT mathematically calibrated or guaranteed."
    )


def evaluate_models_rolling_origin(
    series: np.ndarray,
    cadence: str = "D",
    horizon: int = 7,
    min_train_size: int = 14,
    max_origins: int = 10,
    model_factories: Optional[Dict[ForecastModelType, Any]] = None,
    min_successful_origins: int = 3,
) -> Tuple[List[ModelEvaluationMetric], str, Dict[int, float], Dict[int, str], Dict[int, int], int, int]:
    """Execute rigorous rolling-origin walk-forward validation across candidate models.

    Requirements:
    1. Multiple chronological forecast origins (expanding window).
    2. Training data strictly precedes each validation window (zero lookahead).
    3. Refit candidate models at each origin.
    4. Collect strictly genuine out-of-sample predictions and residuals across successful origins.
    5. Track successful_origins and failed_origins separately for every candidate model; never insert
       artificial error residuals (e.g. 99999.0) or manufacture error values when evaluations fail.
    6. Exclude any model with failed validation origins (failed_origins > 0) or insufficient successful
       origins (< min_successful_origins) from competing for champion status.
    7. Compare candidates against Naive baseline with MAE, RMSE, MASE, and Relative MAE.
    8. Return step-specific empirical residual quantiles, actual uncertainty methods, and residual observation counts.
    """
    req_origins = max(1, int(min_successful_origins))
    n = len(series)
    if n < min_train_size + req_origins:
        return [], "None", {}, {}, {}, 0, 0

    num_origins = min(max_origins, n - min_train_size)
    if num_origins < req_origins:
        return [], "None", {}, {}, {}, 0, 0

    origins = list(range(n - num_origins, n))
    eval_step_horizon = min(horizon, 5)
    valid_origins = [t for t in origins if min(eval_step_horizon, n - t) >= 1]
    attempted_origins = len(valid_origins)
    if attempted_origins < req_origins:
        return [], "None", {}, {}, {}, 0, 0

    season_len = 7 if cadence == "D" else 4

    model_types = [
        ForecastModelType.NAIVE,
        ForecastModelType.MOVING_AVERAGE,
        ForecastModelType.EXPONENTIAL_SMOOTHING,
        ForecastModelType.AUTOREGRESSIVE,
        ForecastModelType.SEASONAL_NAIVE
    ]

    default_factories: Dict[ForecastModelType, Any] = {
        ForecastModelType.NAIVE: lambda yt: NaiveModel(),
        ForecastModelType.MOVING_AVERAGE: lambda yt: MovingAverageModel(window=min(7, max(1, len(yt) // 2))),
        ForecastModelType.EXPONENTIAL_SMOOTHING: lambda yt: HoltExponentialSmoothingModel(),
        ForecastModelType.AUTOREGRESSIVE: lambda yt: AutoRegressiveRidgeModel(p=min(3, max(1, len(yt) // 4))),
        ForecastModelType.SEASONAL_NAIVE: lambda yt: SeasonalNaiveModel(season_length=season_len),
    }
    if model_factories:
        default_factories.update(model_factories)

    error_store: Dict[ForecastModelType, Dict[str, Any]] = {
        m: {
            "all_errors": [],
            "step_errors": {s: [] for s in range(1, eval_step_horizon + 1)},
            "scales": [],
            "successful_origins": 0,
            "failed_origins": 0,
            "failure_reasons": [],
        }
        for m in model_types
    }

    total_val_obs = 0

    for t in valid_origins:
        y_train = np.asarray(series[:t], dtype=float)
        h_t = min(eval_step_horizon, n - t)
        y_test = np.asarray(series[t:t + h_t], dtype=float)
        total_val_obs += h_t

        naive_diffs = np.abs(np.diff(y_train))
        mean_diff = float(np.mean(naive_diffs)) if len(naive_diffs) > 0 and np.all(np.isfinite(naive_diffs)) else 0.0
        scale_t = mean_diff if mean_diff > 0.0 else 0.0

        for m_type in model_types:
            store = error_store[m_type]
            try:
                inst = default_factories[m_type](y_train)
                inst.fit(y_train)
                raw_preds = inst.predict(h_t)
                preds = np.asarray(raw_preds, dtype=float)
                if preds.shape != (h_t,) or not np.all(np.isfinite(preds)):
                    raise ValueError(f"Invalid forecast output shape={preds.shape} or non-finite values.")
                errors = y_test - preds
                if not np.all(np.isfinite(errors)):
                    raise ValueError("Non-finite residual encountered.")
            except Exception as exc:
                store["failed_origins"] += 1
                store["failure_reasons"].append(f"origin={t}: {type(exc).__name__}: {exc}")
            else:
                store["successful_origins"] += 1
                store["scales"].append(scale_t)
                for step_idx, err in enumerate(errors, start=1):
                    err_f = float(err)
                    store["all_errors"].append(err_f)
                    store["step_errors"][step_idx].append(err_f)

    eval_metrics: List[ModelEvaluationMetric] = []
    raw_mae_map: Dict[ForecastModelType, float] = {}
    naive_mae: Optional[float] = None

    for m_type in model_types:
        store = error_store[m_type]
        succ = store["successful_origins"]
        fail = store["failed_origins"]
        errs = np.array(store["all_errors"], dtype=float)

        if succ == 0:
            eval_metrics.append(
                ModelEvaluationMetric(
                    model_name=m_type.value,
                    model_type=m_type,
                    mae=None,
                    rmse=None,
                    mase=None,
                    relative_mae_to_naive=None,
                    is_champion=False,
                    validation_origins=0,
                    validation_observations=0,
                    successful_origins=0,
                    failed_origins=fail,
                    attempted_origins=attempted_origins,
                    is_eligible_for_champion=False,
                    evaluation_status="Unavailable",
                    horizon_step_mae={},
                    notes=(
                        f"Unavailable: failed at all {attempted_origins} rolling origins "
                        f"({fail} failed, 0 successful). No error metrics manufactured."
                    ),
                )
            )
            continue

        if succ < req_origins:
            eval_metrics.append(
                ModelEvaluationMetric(
                    model_name=m_type.value,
                    model_type=m_type,
                    mae=None,
                    rmse=None,
                    mase=None,
                    relative_mae_to_naive=None,
                    is_champion=False,
                    validation_origins=succ,
                    validation_observations=len(errs),
                    successful_origins=succ,
                    failed_origins=fail,
                    attempted_origins=attempted_origins,
                    is_eligible_for_champion=False,
                    evaluation_status="Insufficiently Evaluated",
                    horizon_step_mae={},
                    notes=(
                        f"Insufficiently Evaluated: {succ}/{attempted_origins} rolling origins succeeded "
                        f"({fail} failed; minimum {req_origins} required). Error metrics omitted."
                    ),
                )
            )
            continue

        mae = float(np.mean(np.abs(errs)))
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        model_scales = store["scales"]
        mean_scale = float(np.mean(model_scales)) if model_scales and np.mean(model_scales) > 0 else None
        mase = float(mae / mean_scale) if mean_scale and mean_scale > 0 else None
        raw_mae_map[m_type] = mae

        if m_type == ForecastModelType.NAIVE and fail == 0:
            naive_mae = mae

        step_mae: Dict[int, float] = {}
        for s, s_errs in store["step_errors"].items():
            if len(s_errs) > 0:
                step_mae[s] = round(float(np.mean(np.abs(s_errs))), 2)

        eligible = (fail == 0 and succ == attempted_origins)
        if eligible:
            status_label = "Evaluated"
            note_str = f"Evaluated across {succ} rolling origins ({len(errs)} out-of-sample observations)."
        else:
            status_label = "Excluded (Partial Failure)"
            note_str = (
                f"Excluded from champion selection: failed at {fail}/{attempted_origins} rolling origin(s) "
                f"({succ} succeeded, {len(errs)} genuine out-of-sample observations)."
            )

        if m_type == ForecastModelType.SEASONAL_NAIVE and len(series) < 2 * season_len:
            note_str += f" History < 2 cycles (N < {2 * season_len}); fell back to persistence."

        m_spreads, m_methods, m_counts = compute_horizon_uncertainty(
            step_errors=store["step_errors"],
            horizon=horizon,
            eval_step_horizon=eval_step_horizon,
            mae=mae,
        )

        eval_metrics.append(
            ModelEvaluationMetric(
                model_name=m_type.value,
                model_type=m_type,
                mae=round(mae, 2),
                rmse=round(rmse, 2),
                mase=round(mase, 3) if mase is not None else None,
                relative_mae_to_naive=None,
                is_champion=False,
                validation_origins=succ,
                validation_observations=len(errs),
                successful_origins=succ,
                failed_origins=fail,
                attempted_origins=attempted_origins,
                is_eligible_for_champion=eligible,
                evaluation_status=status_label,
                horizon_step_mae=step_mae,
                notes=note_str,
                step_error_spreads=m_spreads,
                step_error_methods=m_methods,
                step_error_counts=m_counts,
            )
        )

    for em in eval_metrics:
        if em.mae is None or naive_mae is None:
            em.relative_mae_to_naive = None
        elif naive_mae == 0.0:
            em.relative_mae_to_naive = 1.0 if em.mae == 0.0 else float("inf")
        else:
            em.relative_mae_to_naive = round(em.mae / naive_mae, 3)

    eligible_metrics = [
        m for m in eval_metrics
        if m.is_eligible_for_champion and m.mae is not None and m.rmse is not None
    ]
    if not eligible_metrics:
        return (
            eval_metrics,
            "None",
            {},
            {},
            {},
            attempted_origins,
            total_val_obs,
        )

    champion = min(eligible_metrics, key=lambda x: (x.mae, x.rmse))
    champion.is_champion = True
    champion_name = champion.model_name
    champion_type = champion.model_type

    champ_step_errors = error_store[champion_type]["step_errors"]
    step_error_spread, step_error_method, step_error_count = compute_horizon_uncertainty(
        step_errors=champ_step_errors,
        horizon=horizon,
        eval_step_horizon=eval_step_horizon,
        mae=raw_mae_map[champion_type],
    )

    return (
        eval_metrics,
        champion_name,
        step_error_spread,
        step_error_method,
        step_error_count,
        attempted_origins,
        total_val_obs,
    )


def evaluate_models_chronologically(
    series: np.ndarray,
    cadence: str = "D",
    test_horizon: int = 7,
    min_train_ratio: float = 0.70
) -> Tuple[List[ModelEvaluationMetric], str, float]:
    """Backward-compatible adapter forwarding to evaluate_models_rolling_origin."""
    evals, champ_name, spreads, methods, counts, origins, obs = evaluate_models_rolling_origin(
        series, cadence=cadence, horizon=test_horizon
    )
    residual_std = spreads.get(1) if spreads else float("nan")
    return evals, champ_name, residual_std


# --- 4. DATA PREPARATION PIPELINES ---

def prepare_post_time_series(
    df: pd.DataFrame,
    target_metric: str = "cohort_views",
    cadence: str = "D",
    platform: str = "instagram",
    allow_tiktok: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Deduplicate, validate, and resample post performance into regular publication-cohort time series.

    Strict Methodological Rules:
    - Target Semantics: Publication Cohort Performance (lifetime metrics of posts published in interval t).
    - Exposure-Age Bias: Unequal accumulation durations are explicitly disclosed.
    - Coverage: Calendar span is NOT assumed to represent verified 100% ingestion completeness.
    - Zero Handling: No-post intervals have cohort_views=0.0, but rate metrics are undefined (NaN).
    - Platform Isolation: Cross-platform pooling of raw views is prohibited.
    - YouTube / TikTok: Unavailable saves are rejected, not treated as zero; provides 3-component
      'youtube_available_interactions' / 'available_interactions' (Likes + Comments + Shares).
      TikTok 'collect_count' is treated strictly as a separate bookmark proxy metric.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(), {
            "status": ValidationStatus.INSUFFICIENT_DATA,
            "reason": "Empty dataset provided."
        }

    # 1. Platform Isolation & Incompatibility Guard
    if platform is None or str(platform).strip().lower() in ["all", "all platforms"]:
        raw_incompatible = [
            "views", "cohort_views", "likes", "cohort_likes", "shares", "cohort_shares",
            "saves", "cohort_saves", "collect_count", "cohort_collects", "comments_count",
            "total_engagement", "cohort_total_engagement"
        ]
        if target_metric.lower() in raw_incompatible:
            raise ValueError(
                f"Cross-platform pooling of raw '{target_metric}' is prohibited. "
                "YouTube views (30-second playback standard), TikTok short-video views, and Instagram views "
                "are fundamentally non-equivalent. Please specify a single platform ('instagram', 'youtube', or 'tiktok'), "
                "or analyze a normalized cross-platform engagement rate."
            )

    p_norm = platform.strip().lower() if platform else "instagram"
    allowed_platforms = ["instagram", "youtube", "tiktok"] if allow_tiktok else ["instagram", "youtube"]
    if p_norm not in allowed_platforms:
        raise ValueError(
            f"Unsupported platform '{platform}'. Supported platforms: {allowed_platforms}."
        )

    # YouTube Specific Metric Guards
    if p_norm == "youtube":
        if target_metric.lower() in ["saves", "cohort_saves", "reach", "reach_engagement_rate"]:
            raise ValueError(
                f"Metric '{target_metric}' is not available on YouTube. "
                "YouTube Data API does not expose post-level bookmarks/saves or profile reach."
            )
        if target_metric.lower() in ["total_engagement", "cohort_total_engagement"]:
            raise ValueError(
                "The four-component 'total_engagement' metric (Likes + Comments + Shares + Saves) is not valid on YouTube "
                "because post-level saves are unavailable in the YouTube Data API. "
                "Please use 'youtube_available_interactions' (Likes + Comments + Shares) or select platform='instagram'."
            )

    # TikTok Specific Metric Guards
    if p_norm == "tiktok":
        if target_metric.lower() in ["saves", "cohort_saves", "reach", "reach_engagement_rate"]:
            raise ValueError(
                f"Metric '{target_metric}' is not available on TikTok cohorts. "
                "Canonical 'saves' and 'reach' are unmeasured (use 'collect_count' as an explicit bookmark proxy if present)."
            )
        if target_metric.lower() in ["total_engagement", "cohort_total_engagement"]:
            raise ValueError(
                "The four-component 'total_engagement' metric (Likes + Comments + Shares + Saves) is not valid on TikTok "
                "because canonical 'saves' are unavailable. Please use 'available_interactions' (Likes + Comments + Shares)."
            )

    # 2. Cadence Validation (Case-Insensitive)
    c_raw = str(cadence).strip().lower()
    if c_raw in ["daily", "d"]:
        chosen_freq = "D"
    elif c_raw in ["weekly", "w"]:
        chosen_freq = "W"
    else:
        raise ValueError(f"Unsupported cadence '{cadence}'. Supported cadences: 'daily' ('D') or 'weekly' ('W').")

    # 3. Snapshot Deduplication (Module 1 contract)
    latest_df = get_latest_posts_view(df) if "post_id" in df.columns else df

    # Filter platform
    if "platform" in latest_df.columns:
        latest_df = latest_df[latest_df["platform"].str.lower() == p_norm]
        if len(latest_df) == 0:
            return pd.DataFrame(), {
                "status": ValidationStatus.INSUFFICIENT_DATA,
                "reason": f"No posts found for platform '{platform}'."
            }

    # 4. Timestamp Standardization
    ts_candidates = ["timestamp_dt", "timestamp", "published_at", "date"]
    ts_col = next((c for c in ts_candidates if c in latest_df.columns), None)
    if not ts_col:
        return pd.DataFrame(), {
            "status": ValidationStatus.INSUFFICIENT_DATA,
            "reason": "Missing timestamp or published_at column."
        }

    norm_metric = target_metric.lower()
    has_views_col = "views" in latest_df.columns
    need_engagement_components = norm_metric in [
        "total_engagement", "cohort_total_engagement",
        "youtube_available_interactions", "available_interactions",
        "views_engagement_rate", "engagement_rate",
    ]
    need_likes = need_engagement_components or norm_metric in ["likes", "cohort_likes"]
    need_comments = need_engagement_components or norm_metric in ["comments_count", "cohort_comments"]
    need_shares = need_engagement_components or norm_metric in ["shares", "cohort_shares"]
    need_saves = need_engagement_components or norm_metric in ["saves", "cohort_saves"]
    need_collects = (p_norm == "tiktok") and norm_metric in ["collect_count", "cohort_collects"]

    # Copy only required columns before datetime parsing and chronological sorting
    if has_views_col:
        needed_cols = [ts_col, "views"]
        if need_likes and "likes" in latest_df.columns:
            needed_cols.append("likes")
        if need_comments and "comments_count" in latest_df.columns:
            needed_cols.append("comments_count")
        if need_shares and "shares" in latest_df.columns:
            needed_cols.append("shares")
        if need_saves and p_norm == "instagram" and "saves" in latest_df.columns:
            needed_cols.append("saves")
        if need_collects and "collect_count" in latest_df.columns:
            needed_cols.append("collect_count")
    elif target_metric in latest_df.columns:
        needed_cols = [ts_col, target_metric]
    else:
        needed_cols = [ts_col]

    work_df = latest_df[needed_cols].copy()
    work_df["ts"] = pd.to_datetime(work_df[ts_col], utc=True, errors="coerce")
    work_df = work_df.dropna(subset=["ts"]).sort_values("ts")

    if len(work_df) == 0:
        return pd.DataFrame(), {
            "status": ValidationStatus.INSUFFICIENT_DATA,
            "reason": "No valid timestamps found in dataset."
        }

    # 5. Publication-Cohort Resampling (native Cython aggregations; avoid Python lambdas)
    resampled = work_df.set_index("ts").resample(chosen_freq)

    if has_views_col:
        agg_spec: Dict[str, Tuple[str, str]] = {
            "post_count": ("views", "count"),
            "cohort_views": ("views", "sum"),
        }
        has_likes = need_likes and "likes" in work_df.columns
        has_comments = need_comments and "comments_count" in work_df.columns
        has_shares = need_shares and "shares" in work_df.columns
        has_ig_saves = need_saves and p_norm == "instagram" and "saves" in work_df.columns
        has_tk_collects = need_collects and "collect_count" in work_df.columns

        if has_likes:
            agg_spec["cohort_likes"] = ("likes", "sum")
        if has_comments:
            agg_spec["cohort_comments"] = ("comments_count", "sum")
        if has_shares:
            agg_spec["cohort_shares"] = ("shares", "sum")
        if has_ig_saves:
            agg_spec["cohort_saves"] = ("saves", "sum")
        if has_tk_collects:
            agg_spec["cohort_collects"] = ("collect_count", "sum")

        agg_df = resampled.agg(**agg_spec).reset_index()

        if need_likes:
            if not has_likes:
                agg_df["cohort_likes"] = 0
            agg_df["likes"] = agg_df["cohort_likes"]
        if need_comments:
            if not has_comments:
                agg_df["cohort_comments"] = 0
            agg_df["comments_count"] = agg_df["cohort_comments"]
        if need_shares:
            if not has_shares:
                agg_df["cohort_shares"] = 0
            agg_df["shares"] = agg_df["cohort_shares"]
        if need_saves:
            if not has_ig_saves:
                agg_df["cohort_saves"] = np.nan
            if p_norm == "instagram":
                agg_df["saves"] = agg_df["cohort_saves"]
        if need_collects:
            if not has_tk_collects:
                agg_df["cohort_collects"] = np.nan
            agg_df["collect_count"] = agg_df["cohort_collects"]

        if need_engagement_components:
            if p_norm == "instagram":
                agg_df["cohort_total_engagement"] = (
                    agg_df["cohort_likes"] + agg_df["cohort_comments"] + agg_df["cohort_shares"] + agg_df["cohort_saves"].fillna(0.0)
                )
                agg_df["total_engagement"] = agg_df["cohort_total_engagement"]
            else:
                # YouTube & TikTok: 3-component available interactions (Likes + Comments + Shares)
                interactions_3 = (
                    agg_df["cohort_likes"] + agg_df["cohort_comments"] + agg_df["cohort_shares"]
                )
                if p_norm == "youtube":
                    agg_df["youtube_available_interactions"] = interactions_3
                agg_df["available_interactions"] = interactions_3

        if norm_metric in ["views_engagement_rate", "engagement_rate"]:
            # Engagement rate: UNDEFINED (NaN) on zero-views periods
            interactions_series = agg_df["cohort_total_engagement"] if p_norm == "instagram" else agg_df["available_interactions"]
            agg_df["views_engagement_rate"] = np.where(
                agg_df["cohort_views"] > 0,
                (interactions_series / agg_df["cohort_views"] * 100.0).round(2),
                np.nan
            )

        agg_df["views"] = agg_df["cohort_views"]
    elif target_metric in work_df.columns:
        agg_df = resampled.agg(
            post_count=(target_metric, "count"),
            **{target_metric: (target_metric, "mean" if "rate" in target_metric else "sum")}
        ).reset_index()
    else:
        valid_cols = ["cohort_views", "views", "cohort_total_engagement", "total_engagement", "youtube_available_interactions", "available_interactions", "views_engagement_rate"]
        raise ValueError(f"Unknown target metric '{target_metric}'. Supported metrics: {valid_cols}")

    # Map requested metric
    if norm_metric in ["views", "cohort_views"]:
        chosen_metric = "cohort_views"
        desc = "Publication Cohort Views: Aggregate lifetime views generated by content published in interval t."
    elif norm_metric in ["total_engagement", "cohort_total_engagement"]:
        if p_norm == "youtube":
            raise ValueError(
                "The four-component 'total_engagement' metric is not valid on YouTube. "
                "Please use 'youtube_available_interactions' (Likes + Comments + Shares)."
            )
        if p_norm == "tiktok":
            raise ValueError(
                "The four-component 'total_engagement' metric is not valid on TikTok because canonical 'saves' are unavailable. "
                "Please use 'available_interactions' (Likes + Comments + Shares)."
            )
        chosen_metric = "cohort_total_engagement"
        desc = "Instagram Publication Cohort Total Interactions (Likes + Comments + Shares + Saves)."
    elif norm_metric == "youtube_available_interactions":
        if p_norm != "youtube":
            raise ValueError("Metric 'youtube_available_interactions' is specific to YouTube.")
        chosen_metric = "youtube_available_interactions"
        desc = "YouTube Available Interactions: Sum of observable interactions (Likes + Comments + Shares; Saves are unavailable)."
    elif norm_metric == "available_interactions":
        if p_norm not in ["youtube", "tiktok"]:
            raise ValueError("Metric 'available_interactions' is supported for YouTube and TikTok (saves-free 3-component sum).")
        chosen_metric = "available_interactions"
        desc = f"{p_norm.title()} Available Interactions: Sum of observable interactions (Likes + Comments + Shares; canonical Saves unavailable)."
    elif norm_metric in ["collect_count", "cohort_collects"]:
        if p_norm != "tiktok":
            raise ValueError("Metric 'collect_count' / 'cohort_collects' is specific to TikTok bookmark proxy data.")
        chosen_metric = "cohort_collects"
        desc = "TikTok Publication Cohort Bookmark Collects (collect_count proxy — strictly separate from Instagram canonical saves)."
    elif norm_metric in ["views_engagement_rate", "engagement_rate"]:
        chosen_metric = "views_engagement_rate"
        desc = "Cohort Engagement Rate (%): Interactions divided by views for posts published in interval t (NaN on zero-post days)."
    elif norm_metric in agg_df.columns:
        chosen_metric = norm_metric
        desc = f"Publication Cohort Metric: {norm_metric}"
    else:
        raise ValueError(f"Target metric '{target_metric}' not available in resampled dataset.")

    total_periods = len(agg_df)
    active_periods = int((agg_df["post_count"] > 0).sum())
    zero_publication_periods = total_periods - active_periods

    meta = {
        "status": ValidationStatus.VALID if total_periods >= 14 else ValidationStatus.INSUFFICIENT_DATA,
        "cadence": chosen_freq,
        "total_periods": total_periods,
        "active_periods": active_periods,
        "zero_publication_periods": zero_publication_periods,
        "missing_periods": None,  # Ingestion logs do not verify missing scrape days
        "coverage_pct": None,     # Completeness unverified
        "coverage_status": "UNVERIFIED (Continuous calendar span does not prove absence of dropped scrape records)",
        "history_start": agg_df["ts"].min().strftime("%Y-%m-%d"),
        "history_end": agg_df["ts"].max().strftime("%Y-%m-%d"),
        "target_metric": chosen_metric,
        "target_description": desc,
        "platform": p_norm.title(),
        "exposure_age_bias_warning": (
            "Unequal accumulation duration: content published earlier had up to 63 days to accrue views, "
            "while later content had only 2–3 days. Multi-snapshot tracking is required for age-standardized metrics."
        )
    }

    return agg_df[["ts", chosen_metric, "post_count"]], meta


# --- 5. TREND FORECASTING ENGINE ---

class TrendForecaster:
    """Rigorous time-series forecasting engine with rolling-origin walk-forward validation,

    publication-cohort target semantics, and empirical uncertainty ranges.
    Preserves backward compatibility for hashtag trend velocity tracking.
    """

    def __init__(self, rolling_window: int = 7):
        self.rolling_window = rolling_window

    def forecast_metric(
        self,
        df: pd.DataFrame,
        target_metric: str = "cohort_views",
        platform: str = "instagram",
        cadence: str = "D",
        horizon: int = 7,
        selected_model: Optional[ForecastModelType] = None,
        allow_tiktok: bool = False,
    ) -> TrendForecastReport:
        """Forecast a publication cohort performance indicator with rolling-origin walk-forward validation."""
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"Forecast horizon must be a positive integer >= 1; got {horizon}.")

        # 1. Prepare and validate time series
        ts_df, meta = prepare_post_time_series(
            df,
            target_metric=target_metric,
            cadence=cadence,
            platform=platform,
            allow_tiktok=allow_tiktok,
        )

        disclosures = [
            "PUBLICATION COHORT TARGET: One observation represents the lifetime cumulative performance of content assets "
            "published in interval t, NOT calendar-day account-wide views.",
            "EXPOSURE-AGE BIAS: Cohorts published earlier in the collection window had up to 63 days of accumulation before collection, "
            "while late-October cohorts had only 2–3 days. Projections must be interpreted strictly as exploratory descriptive analysis.",
            "SOURCE COMPLETENESS UNVERIFIED: Ingestion logs do not verify whether zero-post intervals reflect genuine zero publication activity "
            "or unobserved/dropped API records. Zero-publication intervals are preserved under an explicit working assumption.",
            "PLATFORM ISOLATION: Platforms are modeled separately. YouTube views, TikTok short-video views, and Instagram exposures are non-equivalent.",
            "OBSERVATIONAL STATUS-QUO: Projections represent statistical continuations and do not predict causal returns from creative changes.",
            "UNCALIBRATED UNCERTAINTY: Uncertainty bounds reflect empirical out-of-sample error quantiles and are not calibrated 95% intervals."
        ]
        if str(platform or "").strip().lower() == "tiktok":
            disclosures.append(
                "TIKTOK PROXY & METRIC SEMANTICS: Canonical Instagram 'saves' and 'reach' are unavailable on TikTok. "
                "Any 'collect_count' metric is treated strictly as a bookmark proxy, not canonical saves."
            )

        if meta.get("status") != ValidationStatus.VALID or len(ts_df) < 14:
            return TrendForecastReport(
                target_metric=target_metric,
                target_description=meta.get("target_description", "Publication Cohort Performance"),
                platform=meta.get("platform", platform.title() if platform else "Instagram"),
                cadence=cadence,
                history_start=meta.get("history_start", "N/A"),
                history_end=meta.get("history_end", "N/A"),
                history_points=len(ts_df),
                active_observation_days=meta.get("active_periods", 0),
                zero_publication_days=meta.get("zero_publication_periods", 0),
                missing_calendar_days=None,
                coverage_pct=None,
                coverage_status="UNVERIFIED",
                forecast_horizon=horizon,
                champion_model="None",
                selected_model="None",
                is_manual_override=False,
                validation_origins_count=0,
                validation_observations_count=0,
                model_evaluations=[],
                forecast_points=[],
                historical_df=ts_df,
                forecast_df=pd.DataFrame(),
                uncertainty_description="Insufficient historical observations for rolling-origin walk-forward validation (minimum 14 observations required).",
                is_calibrated=False,
                status=ValidationStatus.INSUFFICIENT_DATA,
                disclosures=disclosures
            )

        chosen_metric_name = meta["target_metric"]

        # Engagement Rate NaN Check (Do NOT fill with zero)
        if ts_df[chosen_metric_name].isna().any():
            nan_count = int(ts_df[chosen_metric_name].isna().sum())
            raise ValueError(
                f"Cannot model '{target_metric}' as a regular time series: series contains {nan_count} "
                "undefined (NaN) observations on intervals with zero views. Imputing undefined rates as zero "
                "is statistically invalid and distorts trend dynamics. To forecast engagement, model raw interaction "
                "counts (e.g. 'cohort_total_engagement') or aggregate across intervals with consistent post activity."
            )

        y_series = ts_df[chosen_metric_name].astype(float).to_numpy()

        # 2. Genuine Rolling-Origin Chronological Walk-Forward Validation
        eval_metrics, champion_name, step_error_spreads, step_error_methods, step_error_counts, n_origins, n_val_obs = evaluate_models_rolling_origin(
            y_series,
            cadence=meta["cadence"],
            horizon=horizon,
            min_train_size=14,
            max_origins=10
        )

        # 3. Model Selection: Automatic Champion vs User Manual Override
        champion_metric = next((m for m in eval_metrics if m.is_champion), None)

        if not eval_metrics or n_origins < 3 or champion_metric is None:
            no_champ_reason = (
                "No candidate model completed all rolling-origin validations without failures; "
                "champion selection and uncontaminated uncertainty intervals are unavailable."
                if (eval_metrics and n_origins >= 3)
                else "Series length insufficient for multiple rolling-origin validation windows."
            )
            failure_disclosures = list(disclosures)
            if eval_metrics and n_origins >= 3:
                failure_disclosures.append(
                    "VALIDATION ELIGIBILITY FAILURE: All candidate models experienced failed rolling-origin evaluations "
                    "and were excluded from champion selection without manufacturing synthetic error residuals."
                )
            return TrendForecastReport(
                target_metric=chosen_metric_name,
                target_description=meta["target_description"],
                platform=meta["platform"],
                cadence=meta["cadence"],
                history_start=meta["history_start"],
                history_end=meta["history_end"],
                history_points=len(ts_df),
                active_observation_days=meta["active_periods"],
                zero_publication_days=meta["zero_publication_periods"],
                missing_calendar_days=None,
                coverage_pct=None,
                coverage_status="UNVERIFIED",
                forecast_horizon=horizon,
                champion_model="None",
                selected_model="None",
                is_manual_override=False,
                validation_origins_count=n_origins if eval_metrics else 0,
                validation_observations_count=n_val_obs if eval_metrics else 0,
                model_evaluations=eval_metrics,
                forecast_points=[],
                historical_df=ts_df,
                forecast_df=pd.DataFrame(),
                uncertainty_description=no_champ_reason,
                is_calibrated=False,
                status=ValidationStatus.INSUFFICIENT_VALIDATION_HISTORY,
                disclosures=failure_disclosures
            )

        champion_model_str = champion_metric.model_name

        if selected_model is not None:
            chosen_type = selected_model
            selected_model_str = selected_model.value
            is_override = (selected_model != champion_metric.model_type)
        else:
            chosen_type = champion_metric.model_type
            selected_model_str = champion_metric.model_name
            is_override = False

        # 4. Fit Final Selected Model on Complete Historical Series
        season_len = 7 if meta["cadence"] == "D" else 4
        model_map: Dict[ForecastModelType, BaseTimeSeriesModel] = {
            ForecastModelType.NAIVE: NaiveModel(),
            ForecastModelType.SEASONAL_NAIVE: SeasonalNaiveModel(season_length=season_len),
            ForecastModelType.MOVING_AVERAGE: MovingAverageModel(window=min(7, max(1, len(y_series) // 2))),
            ForecastModelType.EXPONENTIAL_SMOOTHING: HoltExponentialSmoothingModel(),
            ForecastModelType.AUTOREGRESSIVE: AutoRegressiveRidgeModel(p=min(3, max(1, len(y_series) // 4)))
        }

        disclosures_list = list(disclosures)
        active_spreads = step_error_spreads
        active_methods = step_error_methods
        active_counts = step_error_counts
        uncertainty_origins_count = n_origins

        override_eval = next((m for m in eval_metrics if m.model_type == chosen_type), None)
        if is_override and override_eval is not None:
            if override_eval.is_eligible_for_champion and override_eval.step_error_spreads:
                active_spreads = override_eval.step_error_spreads
                active_methods = override_eval.step_error_methods
                active_counts = override_eval.step_error_counts
                uncertainty_origins_count = override_eval.successful_origins
                disclosures_list.append(
                    f"MANUAL OVERRIDE UNCERTAINTY: Empirical uncertainty ranges reflect '{override_eval.model_name}' "
                    f"out-of-sample residuals across {override_eval.successful_origins} successful rolling origins."
                )
            elif not override_eval.is_eligible_for_champion:
                disclosures_list.append(
                    f"MANUAL OVERRIDE VALIDATION WARNING: Selected model '{override_eval.model_name}' was excluded from champion "
                    f"eligibility (status: {override_eval.evaluation_status}; {override_eval.failed_origins}/{override_eval.attempted_origins} "
                    f"origins failed). Empirical uncertainty ranges remain anchored to the validated champion '{champion_metric.model_name}'."
                )

        forecaster = model_map.get(chosen_type, NaiveModel())
        try:
            forecaster.fit(y_series)
            point_forecasts = np.asarray(forecaster.predict(horizon), dtype=float)
            if point_forecasts.shape != (horizon,) or not np.all(np.isfinite(point_forecasts)):
                raise ValueError("Final forecast contained non-finite values or mismatched horizon shape.")
        except Exception as exc:
            fallback_type = champion_metric.model_type if chosen_type != champion_metric.model_type else ForecastModelType.NAIVE
            fallback_model = model_map.get(fallback_type, NaiveModel())
            fallback_model.fit(y_series)
            point_forecasts = np.asarray(fallback_model.predict(horizon), dtype=float)
            failed_name = chosen_type.value
            selected_model_str = fallback_type.value
            is_override = (fallback_type != champion_metric.model_type)
            active_spreads = step_error_spreads
            active_methods = step_error_methods
            active_counts = step_error_counts
            uncertainty_origins_count = n_origins
            disclosures_list.append(
                f"FINAL FIT FALLBACK WARNING: Full-series estimation for '{failed_name}' failed ({exc}); reverted to '{selected_model_str}'."
            )

        # 5. Horizon-Specific Empirical Uncertainty Ranges & Method Distinctions
        last_date = ts_df["ts"].max()
        freq_alias = "D" if meta["cadence"] == "D" else "W"
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq_alias)[1:]
        eval_step_horizon = min(horizon, 5)

        points: List[ForecastPoint] = []
        forecast_rows: List[Dict[str, Any]] = []

        for h, (f_date, val) in enumerate(zip(future_dates, point_forecasts), start=1):
            spread = float(active_spreads[h])
            u_method = active_methods[h]
            u_count = active_counts[h]
            ci_low = max(0.0, float(val - spread))
            ci_high = float(val + spread)
            p_val = round(float(val), 2)
            ts_str = f_date.strftime("%Y-%m-%d")

            pt = ForecastPoint(
                step=h,
                timestamp=ts_str,
                point_forecast=p_val,
                ci_lower=round(ci_low, 2),
                ci_upper=round(ci_high, 2),
                empirical_error_spread=round(spread, 2),
                uncertainty_method=u_method,
                residual_count=u_count
            )
            points.append(pt)
            forecast_rows.append({
                "timestamp": ts_str,
                "point_forecast": p_val,
                "ci_lower": round(ci_low, 2),
                "ci_upper": round(ci_high, 2),
                "empirical_error_spread": round(spread, 2),
                "uncertainty_method": u_method,
                "residual_count": u_count
            })

        forecast_df = pd.DataFrame(forecast_rows)
        hist_df = ts_df.rename(columns={"ts": "timestamp", chosen_metric_name: "actual"})
        hist_df["timestamp"] = hist_df["timestamp"].dt.strftime("%Y-%m-%d")

        # Check for small-sample uncertainty warnings
        small_sample_steps = [
            s for s, m in active_methods.items()
            if "Direct Empirical Quantile" in m and active_counts.get(s, 0) < 10
        ]
        if small_sample_steps:
            disclosures_list.append(
                f"SMALL-SAMPLE UNCERTAINTY WARNING: Horizon step(s) {small_sample_steps} utilize direct empirical quantiles based on "
                f"very few residual observations (n < 10). A 90th-percentile quantile estimate derived from sparse observations is "
                f"statistically unstable and must be treated as exploratory."
            )

        # Disclose model-selection optimism bias
        disclosures_list.append(
            "MODEL-SELECTION OPTIMISM BIAS: The champion model was selected by minimizing error across the same rolling-origin "
            "validation windows used to report its evaluation metrics. Consequently, the champion's reported MAE/RMSE is "
            "selection-dependent and does not represent an independent, unbiased estimate of future out-of-sample generalization error."
        )

        uncertainty_note = build_uncertainty_description(
            n_origins=uncertainty_origins_count,
            eval_step_horizon=eval_step_horizon,
            forecast_points=points,
        )

        return TrendForecastReport(
            target_metric=chosen_metric_name,
            target_description=meta["target_description"],
            platform=meta["platform"],
            cadence=meta["cadence"],
            history_start=meta["history_start"],
            history_end=meta["history_end"],
            history_points=len(ts_df),
            active_observation_days=meta["active_periods"],
            zero_publication_days=meta["zero_publication_periods"],
            missing_calendar_days=None,
            coverage_pct=None,
            coverage_status=meta["coverage_status"],
            forecast_horizon=horizon,
            champion_model=champion_model_str,
            selected_model=selected_model_str,
            is_manual_override=is_override,
            validation_origins_count=n_origins,
            validation_observations_count=n_val_obs,
            model_evaluations=eval_metrics,
            forecast_points=points,
            historical_df=hist_df,
            forecast_df=forecast_df,
            uncertainty_description=uncertainty_note,
            is_calibrated=False,
            status=ValidationStatus.VALID,
            disclosures=disclosures_list
        )

    # --- B. TOPIC & KEYWORD VOLUME FORECASTING ---

    def forecast_keyword_trend(
        self,
        trend_df: pd.DataFrame,
        keyword: str,
        horizon: int = 7,
        selected_model: Optional[ForecastModelType] = None
    ) -> TrendForecastReport:
        """Forecast future search/mention volume trajectory for an external struggle topic.

        Checks missing date coverage without unconditional interpolation or zero-filling.
        Safely returns INSUFFICIENT_VALIDATION_HISTORY on short series (N in [14..16]).
        """
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"Forecast horizon must be a positive integer >= 1; got {horizon}.")

        if trend_df is None or len(trend_df) == 0:
            return TrendForecastReport(
                target_metric="mention_volume",
                target_description="External Topic Search & Mention Volume",
                platform="External Search/Social",
                cadence="D",
                history_start="N/A",
                history_end="N/A",
                history_points=0,
                active_observation_days=0,
                zero_publication_days=0,
                missing_calendar_days=None,
                coverage_pct=None,
                coverage_status="Empty Dataset",
                forecast_horizon=horizon,
                champion_model="None",
                selected_model="None",
                is_manual_override=False,
                validation_origins_count=0,
                validation_observations_count=0,
                model_evaluations=[],
                forecast_points=[],
                historical_df=pd.DataFrame(),
                forecast_df=pd.DataFrame(),
                uncertainty_description="Dataset is empty.",
                is_calibrated=False,
                status=ValidationStatus.INSUFFICIENT_DATA,
                disclosures=[]
            )

        sub = trend_df[trend_df["keyword"] == keyword].copy()
        if len(sub) == 0:
            raise ValueError(f"Keyword '{keyword}' not found in trends dataset.")

        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.sort_values("date")

        start_date = sub["date"].min()
        end_date = sub["date"].max()
        expected_range = pd.date_range(start_date, end_date, freq="D")
        observed_dates = set(sub["date"].dt.normalize())
        missing_dates = set(expected_range.normalize()) - observed_dates

        total_expected = len(expected_range)
        missing_count = len(missing_dates)
        coverage_pct = round(((total_expected - missing_count) / total_expected) * 100.0, 1)

        # Insufficient coverage guard: do not unconditionally fabricate missing dates
        if coverage_pct < 80.0 or missing_count > 3:
            return TrendForecastReport(
                target_metric="mention_volume",
                target_description="External Topic Search & Mention Volume",
                platform="External Topics",
                cadence="D",
                history_start=start_date.strftime("%Y-%m-%d"),
                history_end=end_date.strftime("%Y-%m-%d"),
                history_points=len(sub),
                active_observation_days=len(sub),
                zero_publication_days=0,
                missing_calendar_days=missing_count,
                coverage_pct=coverage_pct,
                calendar_continuity_pct=coverage_pct,
                coverage_status=f"Incomplete Observed Calendar Continuity ({missing_count} missing days)",
                forecast_horizon=horizon,
                champion_model="None",
                selected_model="None",
                is_manual_override=False,
                validation_origins_count=0,
                validation_observations_count=0,
                model_evaluations=[],
                forecast_points=[],
                historical_df=sub.rename(columns={"date": "timestamp", "mention_volume": "actual"}),
                forecast_df=pd.DataFrame(),
                uncertainty_description=f"Inadequate observed calendar-date continuity ({coverage_pct}%). {missing_count} calendar dates missing.",
                is_calibrated=False,
                status=ValidationStatus.SPARSE_SERIES,
                disclosures=[
                    "OBSERVED CALENDAR CONTINUITY WARNING: Calendar-date continuity below 80% threshold; missing observations not interpolated.",
                    "SOURCE COMPLETENESS UNVERIFIED: Source/API ingestion completeness remains unverified without crawler polling logs."
                ]
            )

        daily_sub = sub.set_index("date").resample("D").agg(mention_volume=("mention_volume", "mean")).reset_index()
        daily_sub["timestamp"] = daily_sub["date"]
        y_series = daily_sub["mention_volume"].astype(float).to_numpy()

        # Run rolling-origin forecasting
        eval_metrics, champion_name, step_error_spreads, step_error_methods, step_error_counts, n_origins, n_val_obs = evaluate_models_rolling_origin(
            y_series, cadence="D", horizon=horizon, min_train_size=14, max_origins=10
        )

        champion_metric = next((m for m in eval_metrics if m.is_champion), None)

        # Explicit Short-History & Validation Eligibility Guard
        if not eval_metrics or n_origins < 3 or champion_metric is None:
            is_short_history = (not eval_metrics or n_origins < 3)
            if is_short_history:
                reason = (
                    f"Keyword series length (N={len(daily_sub)}) is insufficient for rolling-origin walk-forward validation "
                    f"(minimum 17 daily observations required to establish >= 3 origins with min_train_size=14)."
                )
                fail_disclosures = [
                    f"INSUFFICIENT VALIDATION HISTORY: Series has only {len(daily_sub)} observations. "
                    "Minimum 17 daily observations required to evaluate >= 3 rolling forecast origins.",
                    "SOURCE COMPLETENESS UNVERIFIED: Observed calendar-date continuity does not verify source/API ingestion completeness."
                ]
            else:
                reason = (
                    "No candidate model completed all rolling-origin validations without failures; "
                    "champion selection and uncontaminated uncertainty intervals are unavailable."
                )
                fail_disclosures = [
                    "VALIDATION ELIGIBILITY FAILURE: All candidate models experienced failed rolling-origin evaluations "
                    "and were excluded from champion selection without manufacturing synthetic error residuals.",
                    "SOURCE COMPLETENESS UNVERIFIED: Observed calendar-date continuity does not verify source/API ingestion completeness."
                ]

            return TrendForecastReport(
                target_metric="mention_volume",
                target_description="External Topic Search & Mention Volume",
                platform="External Topics",
                cadence="D",
                history_start=start_date.strftime("%Y-%m-%d"),
                history_end=end_date.strftime("%Y-%m-%d"),
                history_points=len(daily_sub),
                active_observation_days=len(daily_sub),
                zero_publication_days=0,
                missing_calendar_days=missing_count,
                coverage_pct=coverage_pct,
                calendar_continuity_pct=coverage_pct,
                coverage_status="Unbroken Observed Calendar Continuity (Source ingestion unverified)" if missing_count == 0 else f"Incomplete Observed Calendar Continuity ({missing_count} missing days)",
                forecast_horizon=horizon,
                champion_model="None",
                selected_model="None",
                is_manual_override=False,
                validation_origins_count=n_origins,
                validation_observations_count=n_val_obs,
                model_evaluations=eval_metrics,
                forecast_points=[],
                historical_df=daily_sub[["date", "mention_volume"]].rename(columns={"date": "timestamp", "mention_volume": "actual"}),
                forecast_df=pd.DataFrame(),
                uncertainty_description=reason,
                is_calibrated=False,
                status=ValidationStatus.INSUFFICIENT_VALIDATION_HISTORY,
                disclosures=fail_disclosures
            )

        if selected_model is not None:
            chosen_type = selected_model
            selected_model_str = selected_model.value
            is_override = (selected_model != champion_metric.model_type)
        else:
            chosen_type = champion_metric.model_type
            selected_model_str = champion_metric.model_name
            is_override = False

        model_map: Dict[ForecastModelType, BaseTimeSeriesModel] = {
            ForecastModelType.NAIVE: NaiveModel(),
            ForecastModelType.SEASONAL_NAIVE: SeasonalNaiveModel(season_length=7),
            ForecastModelType.MOVING_AVERAGE: MovingAverageModel(window=min(7, max(1, len(y_series) // 2))),
            ForecastModelType.EXPONENTIAL_SMOOTHING: HoltExponentialSmoothingModel(),
            ForecastModelType.AUTOREGRESSIVE: AutoRegressiveRidgeModel(p=min(3, max(1, len(y_series) // 4)))
        }

        disclosures = [
            f"TOPIC TRAJECTORY: Forecast reflects historical search/social mention volume for keyword '{keyword}'.",
            "OBSERVED CALENDAR-DATE CONTINUITY: Series exhibits unbroken daily calendar-date presence across the observed span. "
            "However, source/API ingestion completeness remains unverified without crawler polling logs.",
            "UNCALIBRATED DEMONSTRATION RANGE: Residual bounds are based on empirical percentiles across rolling origins.",
            "MODEL-SELECTION OPTIMISM BIAS: The champion model was selected by minimizing error across the same rolling-origin "
            "validation windows used to report its evaluation metrics. Consequently, the champion's reported MAE/RMSE is "
            "selection-dependent and does not represent an independent, unbiased estimate of future out-of-sample generalization error."
        ]

        active_spreads = step_error_spreads
        active_methods = step_error_methods
        active_counts = step_error_counts
        uncertainty_origins_count = n_origins

        override_eval = next((m for m in eval_metrics if m.model_type == chosen_type), None)
        if is_override and override_eval is not None:
            if override_eval.is_eligible_for_champion and override_eval.step_error_spreads:
                active_spreads = override_eval.step_error_spreads
                active_methods = override_eval.step_error_methods
                active_counts = override_eval.step_error_counts
                uncertainty_origins_count = override_eval.successful_origins
                disclosures.append(
                    f"MANUAL OVERRIDE UNCERTAINTY: Empirical uncertainty ranges reflect '{override_eval.model_name}' "
                    f"out-of-sample residuals across {override_eval.successful_origins} successful rolling origins."
                )
            elif not override_eval.is_eligible_for_champion:
                disclosures.append(
                    f"MANUAL OVERRIDE VALIDATION WARNING: Selected model '{override_eval.model_name}' was excluded from champion "
                    f"eligibility (status: {override_eval.evaluation_status}; {override_eval.failed_origins}/{override_eval.attempted_origins} "
                    f"origins failed). Empirical uncertainty ranges remain anchored to the validated champion '{champion_metric.model_name}'."
                )

        forecaster = model_map.get(chosen_type, NaiveModel())
        try:
            forecaster.fit(y_series)
            point_forecasts = np.asarray(forecaster.predict(horizon), dtype=float)
            if point_forecasts.shape != (horizon,) or not np.all(np.isfinite(point_forecasts)):
                raise ValueError("Final forecast contained non-finite values or mismatched horizon shape.")
        except Exception as exc:
            fallback_type = champion_metric.model_type if chosen_type != champion_metric.model_type else ForecastModelType.NAIVE
            fallback_model = model_map.get(fallback_type, NaiveModel())
            fallback_model.fit(y_series)
            point_forecasts = np.asarray(fallback_model.predict(horizon), dtype=float)
            failed_name = chosen_type.value
            selected_model_str = fallback_type.value
            is_override = (fallback_type != champion_metric.model_type)
            active_spreads = step_error_spreads
            active_methods = step_error_methods
            active_counts = step_error_counts
            uncertainty_origins_count = n_origins
            disclosures.append(
                f"FINAL FIT FALLBACK WARNING: Full-series estimation for '{failed_name}' failed ({exc}); reverted to '{selected_model_str}'."
            )

        last_date = daily_sub["date"].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq="D")[1:]
        eval_step_horizon = min(horizon, 5)

        points: List[ForecastPoint] = []
        forecast_rows: List[Dict[str, Any]] = []

        for h, (f_date, val) in enumerate(zip(future_dates, point_forecasts), start=1):
            spread = float(active_spreads[h])
            u_method = active_methods[h]
            u_count = active_counts[h]
            ci_low = max(0.0, float(val - spread))
            ci_high = float(val + spread)
            p_val = round(float(val), 2)
            ts_str = f_date.strftime("%Y-%m-%d")

            pt = ForecastPoint(
                step=h,
                timestamp=ts_str,
                point_forecast=p_val,
                ci_lower=round(ci_low, 2),
                ci_upper=round(ci_high, 2),
                empirical_error_spread=round(spread, 2),
                uncertainty_method=u_method,
                residual_count=u_count
            )
            points.append(pt)
            forecast_rows.append({
                "timestamp": ts_str,
                "point_forecast": p_val,
                "ci_lower": round(ci_low, 2),
                "ci_upper": round(ci_high, 2),
                "empirical_error_spread": round(spread, 2),
                "uncertainty_method": u_method,
                "residual_count": u_count
            })

        forecast_df = pd.DataFrame(forecast_rows)
        hist_df = daily_sub[["date", "mention_volume"]].rename(columns={"date": "timestamp", "mention_volume": "actual"})
        hist_df["timestamp"] = hist_df["timestamp"].dt.strftime("%Y-%m-%d")

        small_sample_steps = [
            s for s, m in active_methods.items()
            if "Direct Empirical Quantile" in m and active_counts.get(s, 0) < 10
        ]
        if small_sample_steps:
            disclosures.append(
                f"SMALL-SAMPLE UNCERTAINTY WARNING: Horizon step(s) {small_sample_steps} utilize direct empirical quantiles based on "
                f"very few residual observations (n < 10). A 90th-percentile quantile estimate derived from sparse observations is "
                f"statistically unstable and must be treated as exploratory."
            )

        uncertainty_note = build_uncertainty_description(
            n_origins=uncertainty_origins_count,
            eval_step_horizon=eval_step_horizon,
            forecast_points=points,
        )

        return TrendForecastReport(
            target_metric="mention_volume",
            target_description="External Topic Search & Mention Volume",
            platform="External Topics",
            cadence="D",
            history_start=start_date.strftime("%Y-%m-%d"),
            history_end=end_date.strftime("%Y-%m-%d"),
            history_points=len(daily_sub),
            active_observation_days=len(daily_sub),
            zero_publication_days=0,
            missing_calendar_days=missing_count,
            coverage_pct=coverage_pct,
            calendar_continuity_pct=coverage_pct,
            coverage_status="Unbroken Observed Calendar Continuity (Source ingestion unverified)" if missing_count == 0 else f"Incomplete Observed Calendar Continuity ({missing_count} missing days)",
            forecast_horizon=horizon,
            champion_model=champion_metric.model_name,
            selected_model=selected_model_str,
            is_manual_override=is_override,
            validation_origins_count=n_origins,
            validation_observations_count=n_val_obs,
            model_evaluations=eval_metrics,
            forecast_points=points,
            historical_df=hist_df,
            forecast_df=forecast_df,
            uncertainty_description=uncertainty_note,
            is_calibrated=False,
            status=ValidationStatus.VALID,
            disclosures=disclosures
        )

    # --- B2. PANEL DAILY-INCREMENT SERIES FORECASTING (REAL LONGITUDINAL PANELS) ---

    def forecast_daily_increment_series(
        self,
        daily_df: pd.DataFrame,
        *,
        metric_col: str = "daily_views",
        platform: str = "tiktok",
        horizon: int = 7,
        selected_model: Optional[ForecastModelType] = None,
        segment_policy: str = "longest_contiguous",
    ) -> TrendForecastReport:
        """Forecast daily panel-increment activity (e.g., TikTok daily_views / daily_collects).

        Audits calendar continuity across observed snapshot dates without blindly fabricating
        zeros across crawler/snapshot outages. When `segment_policy` is `'longest_contiguous'` or
        `'latest_contiguous'`, extracts the largest (or latest) unbroken 1-day gap-free sub-span
        (requiring >= 17 days and >= 80% full-span continuity), while transparently disclosing any
        gaps, panel entry/exit (`active_videos` variation), and reset suppressions. When
        `segment_policy == 'strict'`, any calendar gap (>3 days or <80% continuity) returns
        `ValidationStatus.SPARSE_SERIES`.
        """
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"Forecast horizon must be a positive integer >= 1; got {horizon}.")

        p_norm = (platform or "tiktok").strip().lower()
        p_label = p_norm.title() if p_norm != "tiktok" else "TikTok"
        target_desc_map = {
            "daily_views": f"{p_label} Daily Panel Incremental Views (Consecutive 1-Day Non-Negative Snapshot Deltas)",
            "daily_likes": f"{p_label} Daily Panel Incremental Likes",
            "daily_comments": f"{p_label} Daily Panel Incremental Comments",
            "daily_shares": f"{p_label} Daily Panel Incremental Shares",
            "daily_collects": f"{p_label} Daily Panel Incremental Bookmark Collects (Proxy Metric — non-equivalent to canonical saves)",
        }
        target_description = target_desc_map.get(metric_col, f"{p_label} Daily Panel Increment: {metric_col}")

        if daily_df is None or len(daily_df) == 0:
            return TrendForecastReport(
                target_metric=metric_col,
                target_description=target_description,
                platform=p_label,
                cadence="D",
                history_start="N/A",
                history_end="N/A",
                history_points=0,
                active_observation_days=0,
                zero_publication_days=0,
                missing_calendar_days=None,
                coverage_pct=None,
                coverage_status="Empty Daily Summary Dataset",
                forecast_horizon=horizon,
                champion_model="None",
                selected_model="None",
                is_manual_override=False,
                validation_origins_count=0,
                validation_observations_count=0,
                model_evaluations=[],
                forecast_points=[],
                historical_df=pd.DataFrame(),
                forecast_df=pd.DataFrame(),
                uncertainty_description="Empty dataset provided.",
                is_calibrated=False,
                status=ValidationStatus.INSUFFICIENT_DATA,
                disclosures=[],
            )

        if "date" not in daily_df.columns or metric_col not in daily_df.columns:
            raise ValueError(f"daily_df must contain 'date' and '{metric_col}' columns.")

        work = daily_df.copy()
        work["date"] = pd.to_datetime(work["date"], utc=True, errors="coerce").dt.normalize()
        work = work.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        if len(work) == 0:
            return TrendForecastReport(
                target_metric=metric_col,
                target_description=target_description,
                platform=p_label,
                cadence="D",
                history_start="N/A",
                history_end="N/A",
                history_points=0,
                active_observation_days=0,
                zero_publication_days=0,
                missing_calendar_days=None,
                coverage_pct=None,
                coverage_status="No Valid Dates",
                forecast_horizon=horizon,
                champion_model="None",
                selected_model="None",
                is_manual_override=False,
                validation_origins_count=0,
                validation_observations_count=0,
                model_evaluations=[],
                forecast_points=[],
                historical_df=pd.DataFrame(),
                forecast_df=pd.DataFrame(),
                uncertainty_description="No valid dates found.",
                is_calibrated=False,
                status=ValidationStatus.INSUFFICIENT_DATA,
                disclosures=[],
            )

        if work[metric_col].isna().any():
            nan_cnt = int(work[metric_col].isna().sum())
            raise ValueError(
                f"Cannot model '{metric_col}': series contains {nan_cnt} NaN observation(s); "
                "missing panel increments cannot be zero-imputed."
            )

        span_start = work["date"].min()
        span_end = work["date"].max()
        expected_range = pd.date_range(span_start, span_end, freq="D", tz="UTC")
        observed_dates = set(work["date"])
        missing_count = int(len(set(expected_range) - observed_dates))
        total_expected = int(len(expected_range))
        overall_coverage_pct = round(((total_expected - missing_count) / total_expected) * 100.0, 1)

        total_resets = int(work["reset_events"].fillna(0).sum()) if "reset_events" in work.columns else None

        # Identify contiguous 1-day gap-free segments
        diff_days = work["date"].diff().dt.days.fillna(1.0)
        work["_seg_id"] = (diff_days != 1.0).cumsum()
        seg_stats = (
            work.groupby("_seg_id", sort=False)
            .agg(seg_len=("date", "count"), seg_start=("date", "min"), seg_end=("date", "max"))
            .reset_index()
        )

        if segment_policy == "strict":
            use_sparse = (overall_coverage_pct < 80.0 or missing_count > 3)
            chosen_seg_id = int(seg_stats.iloc[0]["_seg_id"]) if missing_count == 0 else None
        elif segment_policy == "latest_contiguous":
            valid_segs = seg_stats[seg_stats["seg_len"] >= 17]
            use_sparse = (overall_coverage_pct < 80.0) or valid_segs.empty
            chosen_seg_id = int(valid_segs.iloc[-1]["_seg_id"]) if not valid_segs.empty else None
        else:  # "longest_contiguous" (default)
            valid_segs = seg_stats[seg_stats["seg_len"] >= 17]
            use_sparse = (overall_coverage_pct < 80.0) or valid_segs.empty
            if not valid_segs.empty:
                # Tie-break toward latest longest segment
                best_row = valid_segs.sort_values(["seg_len", "seg_end"], ascending=[True, True]).iloc[-1]
                chosen_seg_id = int(best_row["_seg_id"])
            else:
                chosen_seg_id = None

        if use_sparse or chosen_seg_id is None:
            return TrendForecastReport(
                target_metric=metric_col,
                target_description=target_description,
                platform=p_label,
                cadence="D",
                history_start=span_start.strftime("%Y-%m-%d"),
                history_end=span_end.strftime("%Y-%m-%d"),
                history_points=len(work),
                active_observation_days=len(work),
                zero_publication_days=0,
                missing_calendar_days=missing_count,
                coverage_pct=overall_coverage_pct,
                calendar_continuity_pct=overall_coverage_pct,
                coverage_status=f"Incomplete Observed Calendar Continuity ({missing_count} missing snapshot day(s))",
                forecast_horizon=horizon,
                champion_model="None",
                selected_model="None",
                is_manual_override=False,
                validation_origins_count=0,
                validation_observations_count=0,
                model_evaluations=[],
                forecast_points=[],
                historical_df=work[["date", metric_col]].rename(columns={"date": "timestamp", metric_col: "actual"}),
                forecast_df=pd.DataFrame(),
                uncertainty_description=(
                    f"Inadequate observed daily continuity ({overall_coverage_pct}% across {total_expected} days; "
                    f"{missing_count} missing day(s)); missing snapshot days are not interpolated."
                ),
                is_calibrated=False,
                status=ValidationStatus.SPARSE_SERIES,
                disclosures=[
                    "OBSERVED CALENDAR CONTINUITY WARNING: Panel snapshot outages prevented unbroken daily evaluation; "
                    "missing dates were not fabricated or zero-imputed.",
                ],
            )

        seg_df = work[work["_seg_id"] == chosen_seg_id].sort_values("date").reset_index(drop=True)
        seg_start_str = seg_df["date"].min().strftime("%Y-%m-%d")
        seg_end_str = seg_df["date"].max().strftime("%Y-%m-%d")
        y_series = pd.to_numeric(seg_df[metric_col], errors="raise").astype(float).to_numpy()

        eval_metrics, champion_name, step_error_spreads, step_error_methods, step_error_counts, n_origins, n_val_obs = evaluate_models_rolling_origin(
            y_series, cadence="D", horizon=horizon, min_train_size=14, max_origins=10
        )
        champion_metric = next((m for m in eval_metrics if m.is_champion), None)

        if missing_count == 0:
            cov_status = "Unbroken Observed Calendar Continuity (Source snapshot ingestion unverified)"
        else:
            cov_status = (
                f"Modeled Contiguous Gap-Free Segment ({len(seg_df)}d: {seg_start_str} to {seg_end_str}; "
                f"Full Span Continuity {overall_coverage_pct}% with {missing_count} missing snapshot day(s))"
            )

        base_disclosures: List[str] = [
            f"PANEL DAILY INCREMENT TARGET: Models consecutive 1-day non-negative deltas for '{metric_col}' across tracked panel assets.",
            "PANEL ENTRY / SURVIVORSHIP BIAS: Aggregate daily totals reflect active cohort panel size on each date rather than platform-wide traffic.",
            "UNCALIBRATED UNCERTAINTY: Empirical rolling-origin error quantiles are exploratory and are not calibrated 95% intervals.",
        ]
        if missing_count > 0:
            base_disclosures.append(
                f"GAP-FREE SEGMENTATION: Full observed span ({span_start.strftime('%Y-%m-%d')} to {span_end.strftime('%Y-%m-%d')}) "
                f"contained {missing_count} missing snapshot date(s); modeling used unbroken contiguous segment "
                f"{seg_start_str} to {seg_end_str} (N={len(seg_df)}) without interpolating missing days."
            )
        if "active_videos" in seg_df.columns:
            min_act = int(seg_df["active_videos"].min())
            max_act = int(seg_df["active_videos"].max())
            base_disclosures.append(
                f"PANEL SIZE VARIATION: Active tracked videos within the modeled segment ranged from {min_act:,} to {max_act:,} across days."
            )
        if total_resets is not None:
            seg_resets = int(seg_df["reset_events"].fillna(0).sum()) if "reset_events" in seg_df.columns else 0
            base_disclosures.append(
                f"NEGATIVE-DELTA RESET SUPPRESSION: {seg_resets:,} intra-segment ({total_resets:,} full-span) negative inter-day snapshot deltas were quarantined prior to aggregation."
            )
        if metric_col == "daily_collects":
            base_disclosures.append(
                "PROXY METRIC SEMANTICS: 'daily_collects' measures TikTok bookmark collects, which is a platform proxy distinct from canonical Instagram 'saves'."
            )

        if not eval_metrics or n_origins < 3 or champion_metric is None:
            return TrendForecastReport(
                target_metric=metric_col,
                target_description=target_description,
                platform=p_label,
                cadence="D",
                history_start=seg_start_str,
                history_end=seg_end_str,
                history_points=len(seg_df),
                active_observation_days=len(seg_df),
                zero_publication_days=0,
                missing_calendar_days=missing_count,
                coverage_pct=overall_coverage_pct,
                calendar_continuity_pct=overall_coverage_pct,
                coverage_status=cov_status,
                forecast_horizon=horizon,
                champion_model="None",
                selected_model="None",
                is_manual_override=False,
                validation_origins_count=n_origins,
                validation_observations_count=n_val_obs,
                model_evaluations=eval_metrics,
                forecast_points=[],
                historical_df=seg_df[["date", metric_col]].rename(columns={"date": "timestamp", metric_col: "actual"}),
                forecast_df=pd.DataFrame(),
                uncertainty_description="Series segment length insufficient for >= 3 rolling-origin windows.",
                is_calibrated=False,
                status=ValidationStatus.INSUFFICIENT_VALIDATION_HISTORY,
                disclosures=base_disclosures,
            )

        if selected_model is not None:
            chosen_type = selected_model
            selected_model_str = selected_model.value
            is_override = (selected_model != champion_metric.model_type)
        else:
            chosen_type = champion_metric.model_type
            selected_model_str = champion_metric.model_name
            is_override = False

        model_map: Dict[ForecastModelType, BaseTimeSeriesModel] = {
            ForecastModelType.NAIVE: NaiveModel(),
            ForecastModelType.SEASONAL_NAIVE: SeasonalNaiveModel(season_length=7),
            ForecastModelType.MOVING_AVERAGE: MovingAverageModel(window=min(7, max(1, len(y_series) // 2))),
            ForecastModelType.EXPONENTIAL_SMOOTHING: HoltExponentialSmoothingModel(),
            ForecastModelType.AUTOREGRESSIVE: AutoRegressiveRidgeModel(p=min(3, max(1, len(y_series) // 4))),
        }

        active_spreads = step_error_spreads
        active_methods = step_error_methods
        active_counts = step_error_counts
        uncertainty_origins_count = n_origins

        override_eval = next((m for m in eval_metrics if m.model_type == chosen_type), None)
        if is_override and override_eval is not None and override_eval.is_eligible_for_champion and override_eval.step_error_spreads:
            active_spreads = override_eval.step_error_spreads
            active_methods = override_eval.step_error_methods
            active_counts = override_eval.step_error_counts
            uncertainty_origins_count = override_eval.successful_origins

        forecaster = model_map.get(chosen_type, NaiveModel())
        try:
            forecaster.fit(y_series)
            point_forecasts = np.asarray(forecaster.predict(horizon), dtype=float)
            if point_forecasts.shape != (horizon,) or not np.all(np.isfinite(point_forecasts)):
                raise ValueError("Invalid forecast shape or non-finite output.")
        except Exception as exc:
            fallback_type = champion_metric.model_type if chosen_type != champion_metric.model_type else ForecastModelType.NAIVE
            fallback_model = model_map.get(fallback_type, NaiveModel())
            fallback_model.fit(y_series)
            point_forecasts = np.asarray(fallback_model.predict(horizon), dtype=float)
            failed_name = chosen_type.value
            selected_model_str = fallback_type.value
            is_override = (fallback_type != champion_metric.model_type)
            active_spreads = step_error_spreads
            active_methods = step_error_methods
            active_counts = step_error_counts
            uncertainty_origins_count = n_origins
            base_disclosures.append(
                f"FINAL FIT FALLBACK WARNING: Full-series estimation for '{failed_name}' failed ({exc}); reverted to '{selected_model_str}'."
            )

        last_date = seg_df["date"].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq="D")[1:]
        eval_step_horizon = min(horizon, 5)

        points: List[ForecastPoint] = []
        forecast_rows: List[Dict[str, Any]] = []
        for h, (f_date, val) in enumerate(zip(future_dates, point_forecasts), start=1):
            spread = float(active_spreads[h])
            u_method = active_methods[h]
            u_count = active_counts[h]
            ci_low = max(0.0, float(val - spread))
            ci_high = float(val + spread)
            p_val = round(float(val), 2)
            ts_str = f_date.strftime("%Y-%m-%d")

            points.append(ForecastPoint(
                step=h,
                timestamp=ts_str,
                point_forecast=p_val,
                ci_lower=round(ci_low, 2),
                ci_upper=round(ci_high, 2),
                empirical_error_spread=round(spread, 2),
                uncertainty_method=u_method,
                residual_count=u_count,
            ))
            forecast_rows.append({
                "timestamp": ts_str,
                "point_forecast": p_val,
                "ci_lower": round(ci_low, 2),
                "ci_upper": round(ci_high, 2),
                "empirical_error_spread": round(spread, 2),
                "uncertainty_method": u_method,
                "residual_count": u_count,
            })

        hist_df = seg_df[["date", metric_col]].rename(columns={"date": "timestamp", metric_col: "actual"}).copy()
        hist_df["timestamp"] = hist_df["timestamp"].dt.strftime("%Y-%m-%d")

        base_disclosures.append(
            "MODEL-SELECTION OPTIMISM BIAS: The champion model minimized MAE across the same rolling-origin "
            "windows reported in evaluations; reported error is selection-dependent."
        )

        uncertainty_note = build_uncertainty_description(
            n_origins=uncertainty_origins_count,
            eval_step_horizon=eval_step_horizon,
            forecast_points=points,
        )

        return TrendForecastReport(
            target_metric=metric_col,
            target_description=target_description,
            platform=p_label,
            cadence="D",
            history_start=seg_start_str,
            history_end=seg_end_str,
            history_points=len(seg_df),
            active_observation_days=len(seg_df),
            zero_publication_days=0,
            missing_calendar_days=missing_count,
            coverage_pct=overall_coverage_pct,
            calendar_continuity_pct=overall_coverage_pct,
            coverage_status=cov_status,
            forecast_horizon=horizon,
            champion_model=champion_metric.model_name,
            selected_model=selected_model_str,
            is_manual_override=is_override,
            validation_origins_count=n_origins,
            validation_observations_count=n_val_obs,
            model_evaluations=eval_metrics,
            forecast_points=points,
            historical_df=hist_df,
            forecast_df=pd.DataFrame(forecast_rows),
            uncertainty_description=uncertainty_note,
            is_calibrated=False,
            status=ValidationStatus.VALID,
            disclosures=base_disclosures,
        )

    # --- C. PRESERVED LEGACY APIS (BACKWARD COMPATIBILITY) ---

    def compute_trend_velocity(self, time_series_df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling growth velocity and acceleration for topics/hashtags.

        Preserved for full backward compatibility with Module 7 baseline tests.
        """
        if time_series_df is None or len(time_series_df) == 0:
            return pd.DataFrame()

        df = time_series_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if df["keyword"].isna().any():
            df = df[df["keyword"].notna()]
        df = df.sort_values(by=["keyword", "date"]).reset_index(drop=True)
        if len(df) == 0:
            return pd.DataFrame()

        rolling_mean = (
            df.groupby("keyword", sort=False)["mention_volume"]
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["rolling_mean"] = rolling_mean
        df["velocity_pct"] = (
            df.groupby("keyword", sort=False)["rolling_mean"]
            .pct_change(periods=self.rolling_window)
            .fillna(0.0)
            * 100
        )
        df["acceleration"] = (
            df.groupby("keyword", sort=False)["velocity_pct"]
            .diff()
            .fillna(0.0)
        )
        return df

    def identify_rising_struggles(
        self,
        trend_df: pd.DataFrame,
        min_velocity: float = 15.0
    ) -> pd.DataFrame:
        """Identify keywords exhibiting rapid upward adoption.

        Preserved for full backward compatibility with Module 7 baseline tests.
        """
        if trend_df is None or len(trend_df) == 0 or "velocity_pct" not in trend_df.columns:
            return pd.DataFrame()

        latest_date = trend_df["date"].max()
        latest = trend_df[trend_df["date"] == latest_date].copy()

        rising = latest[latest["velocity_pct"] >= min_velocity].sort_values(
            by="velocity_pct", ascending=False
        ).reset_index(drop=True)

        rising["trend_status"] = np.where(
            rising["acceleration"] > 0, "Accelerating Breakout", "Sustained Rise"
        )
        return rising[["keyword", "mention_volume", "velocity_pct", "acceleration", "trend_status"]]
