"""12-stage reusable Data Cleaning & Quality Pipeline for external and internal tables.

Design principles:
- Strictly preserves pandas nullable semantics (`Int64`, `Float64`, `string`) — NEVER zero-imputes missing metrics.
- Preserves longitudinal snapshots via composite identity keys rather than collapsing timelines.
- Flags cumulative counter resets (`has_negative_delta_reset`) without silently deleting/altering real trajectories.
- Redacts text PII (`@handles`, emails, URLs) and applies deterministic pseudonymization to creator identifiers (`creator_pseudo_id`), supporting secret-keyed HMAC-SHA256 (`SOCIALPULSE_PII_HMAC_KEY`) with a documented legacy public-salt SHA-256 compatibility mode (explicitly not irreversible anonymization).
- Audits train/test split separation and quarantines post-outcome target-leakage columns from feature matrices.
- Emits deterministic machine-readable JSON + Markdown quality reports alongside sanitized Parquet outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_ingestion.schemas import (
    MetricAvailabilityStatus,
    PROXY_METRIC_DEFINITIONS,
    SourceProvenance,
    get_applicable_metrics,
)

# Reverse mapping for Windows-1252 printable characters > 0x7F that differ from ISO-8859-1
_CP1252_CHAR_TO_BYTE: Dict[str, int] = {}
for _b in range(0x80, 0x100):
    try:
        _ch = bytes([_b]).decode("cp1252")
        if _ch not in _CP1252_CHAR_TO_BYTE:
            _CP1252_CHAR_TO_BYTE[_ch] = _b
    except UnicodeDecodeError:
        _CP1252_CHAR_TO_BYTE[chr(_b)] = _b

_MOJIBAKE_SIGNATURE_RE = re.compile(r"(?:Ã.|Â.|ðŸ|â€|âœ|ï¸)")
_ZERO_WIDTH_RE = re.compile(r"[\ufeff\u200b-\u200f\u202a-\u202e]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")

_PII_HANDLE_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_.]{2,32}")
_PII_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PII_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

_COMPACT_COUNT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([kmb])?\s*$", re.IGNORECASE)
_COMPACT_MULTIPLIERS = {"": 1.0, "k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}


def fix_mojibake_and_whitespace(text: Any) -> Tuple[Any, bool]:
    """Recover double-encoded CP1252/Latin-1 UTF-8 mojibake (when unambiguous) and normalize Unicode/whitespace."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or text is pd.NA:
        return text, False
    s = str(text)
    changed = False

    cleaned = _ZERO_WIDTH_RE.sub("", s)
    if cleaned != s:
        s = cleaned
        changed = True

    if _MOJIBAKE_SIGNATURE_RE.search(s):
        buf = bytearray()
        convertible = True
        for ch in s:
            o = ord(ch)
            if o < 0x80:
                buf.append(o)
            elif ch in _CP1252_CHAR_TO_BYTE:
                buf.append(_CP1252_CHAR_TO_BYTE[ch])
            elif 0x80 <= o <= 0xFF:
                buf.append(o)
            else:
                convertible = False
                break
        if convertible:
            try:
                decoded = bytes(buf).decode("utf-8")
                if decoded != s:
                    s = decoded
                    changed = True
            except UnicodeDecodeError:
                pass

    norm = unicodedata.normalize("NFC", s)
    if norm != s:
        s = norm
        changed = True

    collapsed = _MULTI_SPACE_RE.sub(" ", s).strip()
    if collapsed != s:
        s = collapsed
        changed = True

    return s, changed


def parse_compact_count_value(val: Any) -> float | Any:
    """Parse counts like '30.5K', '1.2M', '12,340' into numeric values while keeping nulls as `np.nan`."""
    if val is None or val is pd.NA:
        return np.nan
    if isinstance(val, (int, np.integer)):
        return float(val)
    if isinstance(val, (float, np.floating)):
        return np.nan if np.isnan(val) else float(val)
    s = str(val).strip().replace(",", "")
    if not s or s.lower() in {"na", "n/a", "nan", "none", "null", "-", "--"}:
        return np.nan
    m = _COMPACT_COUNT_RE.match(s)
    if m:
        num = float(m.group(1))
        suffix = (m.group(2) or "").lower()
        return num * _COMPACT_MULTIPLIERS[suffix]
    return np.nan


def parse_compact_count_series(series: pd.Series) -> pd.Series:
    """Vectorized fast path when already numeric; shorthand string parsing otherwise."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    direct = pd.to_numeric(series, errors="coerce")
    non_null_mask = series.notna()
    # Fall back element-wise only where direct numeric coercion failed on non-null entries
    needs_shorthand = non_null_mask & direct.isna()
    if not needs_shorthand.any():
        return direct
    res = direct.copy().astype("float64")
    res.loc[needs_shorthand] = [parse_compact_count_value(v) for v in series.loc[needs_shorthand]]
    return res


PII_HMAC_KEY_ENV = "SOCIALPULSE_PII_HMAC_KEY"
LEGACY_PUBLIC_SALT = "SocialPulse-PII-Salt-v1"


def resolve_pseudonymization_mode(
    salt: Optional[str] = None,
    *,
    hmac_key: Optional[str] = None,
    allow_legacy_public_salt: bool = True,
) -> Dict[str, Any]:
    """Determine active deterministic pseudonymization mode without logging secrets.

    Security boundary:
    - Pseudonymization preserves deterministic linkability across tables/snapshots; it is
      NOT irreversible cryptographic anonymization.
    - Prefer secret-keyed HMAC-SHA256 via explicit ``hmac_key`` or external environment
      variable ``SOCIALPULSE_PII_HMAC_KEY`` so offline enumeration against public handles
      is infeasible without the external secret.
    - When neither secret key nor custom salt is supplied and ``allow_legacy_public_salt=True``,
      falls back to ``LEGACY_PUBLIC_SALT`` solely for reproducible compatibility with previously
      generated public benchmark Parquet files.
    """
    explicit_key = "" if hmac_key is None else str(hmac_key).strip()
    if explicit_key:
        return {"mode": "hmac_sha256", "is_secret_keyed": True, "key": explicit_key}

    if salt is not None and str(salt).strip():
        salt_str = str(salt).strip()
        if salt_str == LEGACY_PUBLIC_SALT:
            if not allow_legacy_public_salt:
                raise ValueError(
                    f"Legacy public salt rejected when allow_legacy_public_salt=False; set {PII_HMAC_KEY_ENV} or pass hmac_key."
                )
            return {"mode": "legacy_public_salt_sha256", "is_secret_keyed": False, "salt": LEGACY_PUBLIC_SALT}
        return {"mode": "custom_salt_sha256", "is_secret_keyed": False, "salt": salt_str}

    env_key = os.environ.get(PII_HMAC_KEY_ENV, "").strip()
    if env_key:
        return {"mode": "hmac_sha256", "is_secret_keyed": True, "key": env_key}

    if allow_legacy_public_salt:
        return {"mode": "legacy_public_salt_sha256", "is_secret_keyed": False, "salt": LEGACY_PUBLIC_SALT}

    raise ValueError(
        f"Secret HMAC key ({PII_HMAC_KEY_ENV} or explicit hmac_key) or non-default salt required when allow_legacy_public_salt=False."
    )


def pseudonymize_identifier(
    value: Any,
    salt: Optional[str] = None,
    *,
    hmac_key: Optional[str] = None,
    allow_legacy_public_salt: bool = True,
) -> Any:
    """Deterministically pseudonymize a raw account/user identifier (`cr_<hex>` surrogate).

    Uses keyed HMAC-SHA256 (`cr_<16-hex>`) when `hmac_key` or `SOCIALPULSE_PII_HMAC_KEY`
    is set; otherwise uses explicit `salt` or the documented `LEGACY_PUBLIC_SALT`
    compatibility SHA-256 hash (`cr_<12-hex>`). Explicitly performs deterministic
    pseudonymization (preserves within-corpus linkability) rather than irreversible anonymization.
    """
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "na"}:
        return pd.NA

    mode_info = resolve_pseudonymization_mode(
        salt=salt,
        hmac_key=hmac_key,
        allow_legacy_public_salt=allow_legacy_public_salt,
    )
    if mode_info["mode"] == "hmac_sha256":
        digest = hmac.new(
            mode_info["key"].encode("utf-8"),
            s.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:16]
    else:
        digest = hashlib.sha256(f"{mode_info['salt']}:{s}".encode("utf-8")).hexdigest()[:12]
    return f"cr_{digest}"


def redact_pii_text(text: Any) -> Tuple[Any, Dict[str, int]]:
    """Replace @handles, email addresses, and URLs in public text columns with neutral tokens."""
    counts = {"handles": 0, "emails": 0, "urls": 0}
    if text is None or text is pd.NA or (isinstance(text, float) and np.isnan(text)):
        return text, counts
    s = str(text)
    if "@" in s:
        s, n_email = _PII_EMAIL_RE.subn("[EMAIL]", s)
        counts["emails"] += n_email
        s, n_handle = _PII_HANDLE_RE.subn("[USER]", s)
        counts["handles"] += n_handle
    if "http" in s.lower() or "www." in s.lower():
        s, n_url = _PII_URL_RE.subn("[URL]", s)
        counts["urls"] += n_url
    return s, counts


def _parse_mixed_timestamp_series(series: pd.Series, *, is_excel_serial: bool = False) -> Tuple[pd.Series, pd.Series]:
    """Parse timestamps/dates/Excel serials and emit precision labels cleanly."""
    n = len(series)
    if n == 0:
        return pd.Series(dtype="datetime64[ns, UTC]"), pd.Series(dtype="string")

    if is_excel_serial:
        nums = pd.to_numeric(series, errors="coerce")
        valid_num = nums.notna() & (nums > 20_000) & (nums < 80_000)
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
        if valid_num.any():
            parsed.loc[valid_num] = pd.to_datetime(
                nums.loc[valid_num], unit="D", origin="1899-12-30", utc=True
            )
        # Fallback to string parsing for any non-serial entries (e.g. '31.07.2025')
        non_serial = series.notna() & (~valid_num)
        if non_serial.any():
            parsed.loc[non_serial] = pd.to_datetime(
                series.loc[non_serial], format="mixed", dayfirst=True, utc=True, errors="coerce"
            )
        prec = pd.Series(
            np.where(parsed.notna(), "date_only", "missing_or_unparseable"),
            index=series.index,
            dtype="string",
        )
        return parsed, prec

    if pd.api.types.is_numeric_dtype(series):
        nums = pd.to_numeric(series, errors="coerce")
        med = float(nums.dropna().median()) if nums.notna().any() else 0.0
        if 1_000_000_000_000 <= med <= 3_000_000_000_000:
            parsed = pd.to_datetime(nums, unit="ms", utc=True, errors="coerce")
        elif 1_000_000_000 <= med <= 3_000_000_000:
            parsed = pd.to_datetime(nums, unit="s", utc=True, errors="coerce")
        else:
            parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
        prec = pd.Series(
            np.where(parsed.notna(), "full_datetime", "missing_or_unparseable"),
            index=series.index,
            dtype="string",
        )
        return parsed, prec

    s_str = series.astype("string").str.strip()
    is_epoch_str = s_str.str.fullmatch(r"\d{10,13}").fillna(False)
    if is_epoch_str.all() and series.notna().any():
        nums = pd.to_numeric(s_str, errors="coerce")
        unit = "ms" if float(nums.dropna().median()) >= 1e12 else "s"
        parsed = pd.to_datetime(nums, unit=unit, utc=True, errors="coerce")
        prec = pd.Series(
            np.where(parsed.notna(), "full_datetime", "missing_or_unparseable"),
            index=series.index,
            dtype="string",
        )
        return parsed, prec

    # Classify partial patterns prior to parsing so date/time-only formats don't synthesize fake calendar dates
    is_time_only = s_str.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?").fillna(False)
    is_month_day = s_str.str.fullmatch(r"\d{1,2}[-/]\d{1,2}").fillna(False)
    is_date_only = s_str.str.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}").fillna(False)
    full_mask = series.notna() & (~is_time_only) & (~is_month_day)

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    if full_mask.any():
        parsed.loc[full_mask] = pd.to_datetime(s_str.loc[full_mask], format="mixed", utc=True, errors="coerce")

    prec_arr = np.full(n, "missing_or_unparseable", dtype=object)
    valid_dt = parsed.notna().to_numpy()
    prec_arr[valid_dt & is_date_only.to_numpy()] = "date_only"
    prec_arr[valid_dt & (~is_date_only.to_numpy())] = "full_datetime"
    prec_arr[is_time_only.to_numpy()] = "time_only"
    prec_arr[is_month_day.to_numpy()] = "month_day"
    return parsed, pd.Series(prec_arr, index=series.index, dtype="string")


@dataclass
class CleaningConfig:
    """Declarative specification for cleaning and auditing a single table."""

    dataset_id: str
    table_name: str
    platform: str
    provenance: SourceProvenance
    access_tier: str = "public_api"
    composite_key_cols: Tuple[str, ...] = ()
    mandatory_cols: Tuple[str, ...] = ("platform",)
    int_cols: Tuple[str, ...] = ()
    float_cols: Tuple[str, ...] = ()
    shorthand_int_cols: Tuple[str, ...] = ()
    string_cols: Tuple[str, ...] = ()
    timestamp_cols: Tuple[str, ...] = ()
    excel_serial_date_cols: Tuple[str, ...] = ()
    non_negative_cols: Tuple[str, ...] = ()
    rate_0_to_1_cols: Tuple[str, ...] = ("retention_rate",)
    monotonic_group_col: Optional[str] = None
    monotonic_order_col: Optional[str] = None
    monotonic_counter_cols: Tuple[str, ...] = ()
    pii_id_cols: Tuple[str, ...] = ()
    drop_raw_pii_cols: Tuple[str, ...] = ()
    pii_text_cols: Tuple[str, ...] = ()
    mojibake_text_cols: Tuple[str, ...] = ()
    split_col: Optional[str] = None
    split_dedup_text_col: Optional[str] = None
    quarantine_leakage_cols: Tuple[str, ...] = ()
    proxy_metrics_used: Tuple[str, ...] = ()
    recompute_engagement_rate: bool = False
    recompute_save_rate: bool = False
    recompute_share_rate: bool = False
    hmac_key: Optional[str] = None
    allow_legacy_public_salt: bool = True
    notes: Tuple[str, ...] = ()


def run_cleaning_pipeline(df: pd.DataFrame, config: CleaningConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute all 12 standardized data-cleaning and quality-audit stages on `df`."""
    work = df.copy()
    rows_initial = int(len(work))

    # --- STAGE 1: Encoding / Mojibake / Whitespace Normalization ---
    encoding_fixes = 0
    for col in config.mojibake_text_cols:
        if col in work.columns:
            col_vals = work[col].tolist()
            fixed_vals: List[Any] = []
            for v in col_vals:
                fv, chg = fix_mojibake_and_whitespace(v)
                if chg:
                    encoding_fixes += 1
                fixed_vals.append(fv)
            work[col] = pd.Series(fixed_vals, index=work.index, dtype="object")

    # --- STAGE 2: Timestamp Parsing & UTC Standardization ---
    parsed_timestamps_meta: Dict[str, Dict[str, Any]] = {}
    for col in config.timestamp_cols:
        if col in work.columns:
            is_excel = col in config.excel_serial_date_cols
            parsed_col, prec_col = _parse_mixed_timestamp_series(work[col], is_excel_serial=is_excel)
            work[col] = parsed_col
            if f"{col}_precision" not in work.columns:
                work[f"{col}_precision"] = prec_col
            valid = parsed_col.dropna()
            parsed_timestamps_meta[col] = {
                "valid_count": int(len(valid)),
                "missing_or_partial_count": int(len(parsed_col) - len(valid)),
                "min": str(valid.min()) if not valid.empty else None,
                "max": str(valid.max()) if not valid.empty else None,
                "precision_counts": {str(k): int(v) for k, v in prec_col.value_counts().items()},
            }

    # --- STAGE 3: Composite Key Deduplication (Longitudinal-Safe) ---
    dup_removed = 0
    valid_keys = [c for c in config.composite_key_cols if c in work.columns]
    if valid_keys and len(work) > 0:
        before = len(work)
        work = work.drop_duplicates(subset=valid_keys, keep="last").reset_index(drop=True)
        dup_removed = int(before - len(work))

    # --- STAGE 4: Nullable Dtype Casting & Shorthand Count Parsing (NO Zero-Imputation) ---
    for col in config.shorthand_int_cols:
        if col in work.columns:
            parsed_nums = parse_compact_count_series(work[col])
            work[col] = parsed_nums.round().astype("Int64")

    for col in config.int_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").round().astype("Int64")

    for col in config.float_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").astype("Float64")

    for col in config.string_cols:
        if col in work.columns:
            work[col] = work[col].astype("string")

    # --- STAGE 5: Range Validation & Cumulative Counter Reset Flagging ---
    negative_raw_flagged: Dict[str, int] = {}
    for col in config.non_negative_cols:
        if col in work.columns:
            num = pd.to_numeric(work[col], errors="coerce")
            neg_mask = num < 0
            cnt = int(neg_mask.fillna(False).sum())
            if cnt > 0:
                negative_raw_flagged[col] = cnt
                # Mask impossible negative raw counters to NA rather than silently clamping to 0
                work.loc[neg_mask.fillna(False), col] = pd.NA

    out_of_range_rates: Dict[str, int] = {}
    for col in config.rate_0_to_1_cols:
        if col in work.columns:
            num = pd.to_numeric(work[col], errors="coerce")
            bad = num.notna() & ((num < 0.0) | (num > 1.0))
            cnt = int(bad.sum())
            if cnt > 0:
                out_of_range_rates[col] = cnt
                work[col] = num.clip(lower=0.0, upper=1.0).astype("Float64")

    reset_counts_by_col: Dict[str, int] = {}
    if (
        config.monotonic_group_col
        and config.monotonic_order_col
        and config.monotonic_counter_cols
        and config.monotonic_group_col in work.columns
        and config.monotonic_order_col in work.columns
    ):
        work = work.sort_values([config.monotonic_group_col, config.monotonic_order_col]).reset_index(drop=True)
        same_group = work[config.monotonic_group_col].eq(work[config.monotonic_group_col].shift(1)).fillna(False)
        any_reset = np.zeros(len(work), dtype=bool)
        for col in config.monotonic_counter_cols:
            if col not in work.columns:
                continue
            vals = pd.to_numeric(work[col], errors="coerce")
            delta = vals - vals.shift(1)
            neg_delta = (same_group & delta.lt(0).fillna(False)).to_numpy()
            cnt = int(neg_delta.sum())
            reset_counts_by_col[col] = cnt
            any_reset |= neg_delta
        work["has_negative_delta_reset"] = pd.Series(any_reset, index=work.index, dtype="boolean")

    # --- STAGE 6: Derived Ratio Recalculation & Safe Zero-Denom Division ---
    derived_recalculated: List[str] = []
    if config.recompute_engagement_rate and "likes" in work.columns:
        likes = pd.to_numeric(work["likes"], errors="coerce")
        comms = pd.to_numeric(work["comments_count"], errors="coerce") if "comments_count" in work.columns else 0.0
        shares = pd.to_numeric(work["shares"], errors="coerce") if "shares" in work.columns else 0.0
        saves = pd.to_numeric(work["saves"], errors="coerce") if "saves" in work.columns else 0.0
        denom_col = "reach" if ("reach" in work.columns and work["reach"].notna().any()) else ("views" if "views" in work.columns else None)
        if denom_col is not None:
            denom = pd.to_numeric(work[denom_col], errors="coerce")
            num = likes.fillna(0.0) + (comms.fillna(0.0) if isinstance(comms, pd.Series) else 0.0) + (shares.fillna(0.0) if isinstance(shares, pd.Series) else 0.0) + (saves.fillna(0.0) if isinstance(saves, pd.Series) else 0.0)
            valid_ratio = likes.notna() & denom.notna() & (denom > 0)
            res = np.full(len(work), np.nan, dtype="float64")
            res[valid_ratio.to_numpy()] = (num[valid_ratio] / denom[valid_ratio]).to_numpy()
            work["engagement_rate_recomputed"] = pd.Series(res, index=work.index, dtype="Float64")
            derived_recalculated.append("engagement_rate_recomputed")

    if config.recompute_save_rate and "saves" in work.columns and "reach" in work.columns:
        saves = pd.to_numeric(work["saves"], errors="coerce")
        reach = pd.to_numeric(work["reach"], errors="coerce")
        valid_sr = saves.notna() & reach.notna() & (reach > 0)
        res = np.full(len(work), np.nan, dtype="float64")
        res[valid_sr.to_numpy()] = (saves[valid_sr] / reach[valid_sr]).to_numpy()
        work["save_rate_recomputed"] = pd.Series(res, index=work.index, dtype="Float64")
        derived_recalculated.append("save_rate_recomputed")

    if config.recompute_share_rate and "shares" in work.columns:
        shares = pd.to_numeric(work["shares"], errors="coerce")
        denom_col = "reach" if ("reach" in work.columns and work["reach"].notna().any()) else ("views" if "views" in work.columns else None)
        if denom_col is not None:
            denom = pd.to_numeric(work[denom_col], errors="coerce")
            valid_shr = shares.notna() & denom.notna() & (denom > 0)
            res = np.full(len(work), np.nan, dtype="float64")
            res[valid_shr.to_numpy()] = (shares[valid_shr] / denom[valid_shr]).to_numpy()
            work["share_rate_recomputed"] = pd.Series(res, index=work.index, dtype="Float64")
            derived_recalculated.append("share_rate_recomputed")

    # --- STAGE 7: Metric Availability Status & Proxy Annotations ---
    applicable_metrics = sorted(get_applicable_metrics(config.platform, "video", config.access_tier))
    availability_summary: Dict[str, str] = {}
    canonical_metrics = ("views", "impressions", "reach", "likes", "comments_count", "shares", "saves", "follower_count", "retention_rate")
    for metric in canonical_metrics:
        if metric in work.columns and work[metric].notna().any():
            availability_summary[metric] = MetricAvailabilityStatus.OBSERVED.value
        elif metric in applicable_metrics:
            availability_summary[metric] = MetricAvailabilityStatus.MISSING.value
        elif metric in {"saves", "impressions", "reach", "retention_rate"} and config.platform in {"tiktok", "youtube"} and metric == "saves":
            availability_summary[metric] = MetricAvailabilityStatus.NOT_APPLICABLE.value
        else:
            availability_summary[metric] = MetricAvailabilityStatus.UNAVAILABLE.value

    proxy_notes = {k: PROXY_METRIC_DEFINITIONS[k] for k in config.proxy_metrics_used if k in PROXY_METRIC_DEFINITIONS}

    # --- STAGE 8: PII Pseudonymization & Text Redaction ---
    pii_hashed_cols: List[str] = []
    present_pii_cols = [col for col in config.pii_id_cols if col in work.columns]
    pseudo_mode = (
        resolve_pseudonymization_mode(
            hmac_key=config.hmac_key,
            allow_legacy_public_salt=config.allow_legacy_public_salt,
        )
        if present_pii_cols
        else {"mode": "not_applicable", "is_secret_keyed": False}
    )
    for col in present_pii_cols:
        # Map unique non-null values via dict for speed & determinism
        uniq = work[col].dropna().unique()
        mapping = {
            u: pseudonymize_identifier(
                u,
                hmac_key=config.hmac_key,
                allow_legacy_public_salt=config.allow_legacy_public_salt,
            )
            for u in uniq
        }
        pseudo_series = work[col].map(mapping).astype("string")
        if "creator_pseudo_id" not in work.columns:
            work["creator_pseudo_id"] = pseudo_series
        else:
            work[f"{col}_pseudo"] = pseudo_series
        pii_hashed_cols.append(col)

    dropped_pii_cols: List[str] = []
    for col in config.drop_raw_pii_cols:
        if col in work.columns:
            work = work.drop(columns=[col])
            dropped_pii_cols.append(col)

    text_redaction_totals = {"handles": 0, "emails": 0, "urls": 0}
    for col in config.pii_text_cols:
        if col in work.columns:
            cleaned_texts: List[Any] = []
            for v in work[col].tolist():
                rv, cnts = redact_pii_text(v)
                for k, c in cnts.items():
                    text_redaction_totals[k] += c
                cleaned_texts.append(rv)
            work[col] = pd.Series(cleaned_texts, index=work.index, dtype="string")

    # --- STAGE 9: Split Hygiene & Cross-Split Overlap Audit ---
    split_audit: Dict[str, Any] = {"split_column": config.split_col, "counts": {}, "cross_split_text_overlap": 0}
    if config.split_col and config.split_col in work.columns:
        split_counts = {str(k): int(v) for k, v in work[config.split_col].value_counts(dropna=False).items()}
        split_audit["counts"] = split_counts
        if config.split_dedup_text_col and config.split_dedup_text_col in work.columns:
            tr = work.loc[work[config.split_col].astype(str) == "train", config.split_dedup_text_col].dropna().astype(str).str.strip().str.lower()
            te = work.loc[work[config.split_col].astype(str) == "test", config.split_dedup_text_col].dropna().astype(str).str.strip().str.lower()
            overlap = set(tr) & set(te)
            split_audit["cross_split_text_overlap"] = len(overlap)

    # --- STAGE 10: Target-Leakage Quarantine ---
    quarantined_present = [c for c in config.quarantine_leakage_cols if c in work.columns]

    # --- STAGE 11: Schema & Completeness Validation ---
    mandatory_nulls: Dict[str, int] = {}
    for col in config.mandatory_cols:
        if col not in work.columns:
            mandatory_nulls[col] = len(work)
        else:
            mandatory_nulls[col] = int(work[col].isna().sum())

    null_rates_pct: Dict[str, float] = {}
    n_final = int(len(work))
    for col in work.columns:
        null_rates_pct[str(col)] = round((int(work[col].isna().sum()) / n_final) * 100.0, 2) if n_final > 0 else 0.0

    # --- STAGE 12: Assemble Audit Report Metadata ---
    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": config.dataset_id,
        "table_name": config.table_name,
        "platform": config.platform,
        "provenance": config.provenance.value,
        "access_tier": config.access_tier,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_counts": {
            "initial": rows_initial,
            "duplicates_removed": dup_removed,
            "final": n_final,
        },
        "stage_1_encoding_fixes": encoding_fixes,
        "stage_2_timestamps": parsed_timestamps_meta,
        "stage_5_anomalies": {
            "negative_raw_values_masked": negative_raw_flagged,
            "out_of_range_rates_clipped": out_of_range_rates,
            "cumulative_counter_negative_deltas": reset_counts_by_col,
        },
        "stage_6_derived_ratios": derived_recalculated,
        "stage_7_metric_availability": availability_summary,
        "stage_7_proxy_definitions": proxy_notes,
        "stage_8_pii_protection": {
            "protection_mechanism": "pseudonymization",
            "pseudonymization_mode": pseudo_mode["mode"],
            "is_secret_keyed": bool(pseudo_mode["is_secret_keyed"]),
            "irreversibility_claim": False,
            "reidentification_disclosure": (
                "Deterministic pseudonymization substitutes direct account identifiers for cross-post linking while preserving linkability; "
                "pass an external secret via SOCIALPULSE_PII_HMAC_KEY (HMAC-SHA256) for non-enumerable tokens. Legacy public-salt mode is retained strictly for benchmark compatibility and is not irreversible anonymization."
            ),
            "hashed_columns": pii_hashed_cols,
            "dropped_raw_columns": dropped_pii_cols,
            "inline_text_redactions": text_redaction_totals,
        },
        "stage_9_split_hygiene": split_audit,
        "stage_10_leakage_quarantined_columns": quarantined_present,
        "stage_11_validation": {
            "mandatory_column_nulls": mandatory_nulls,
            "column_null_rates_pct": null_rates_pct,
        },
        "notes": list(config.notes),
    }
    return work, report


def save_clean_artifacts(
    df_clean: pd.DataFrame,
    report: Dict[str, Any],
    output_dir: Path,
    table_name: str,
    *,
    write_markdown: bool = True,
) -> Dict[str, Path]:
    """Persist cleaned table as Parquet alongside structured JSON and Markdown quality reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{table_name}.parquet"
    df_clean.to_parquet(parquet_path, index=False)

    json_path = output_dir / f"{table_name}_clean_quality.json"
    with open(json_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    paths = {"parquet": parquet_path, "json_report": json_path}
    if write_markdown:
        md_path = output_dir / f"{table_name}_clean_quality.md"
        rc = report["row_counts"]
        lines = [
            f"# Clean Quality Report — `{report['dataset_id']}` / `{table_name}`",
            "",
            f"- **Platform:** `{report['platform']}` | **Provenance:** `{report['provenance']}` | **Access Tier:** `{report['access_tier']}`",
            f"- **Rows:** initial=`{rc['initial']:,}`, duplicates_removed=`{rc['duplicates_removed']:,}`, final=`{rc['final']:,}`",
            f"- **Stage 1 Encoding Fixes:** `{report['stage_1_encoding_fixes']}`",
            f"- **Stage 5 Cumulative Counter Negative Deltas Flagged:** `{report['stage_5_anomalies']['cumulative_counter_negative_deltas']}`",
            f"- **Stage 7 Metric Availability:** `{report['stage_7_metric_availability']}`",
            f"- **Stage 8 PII Protection:** `{report['stage_8_pii_protection']}`",
            f"- **Stage 9 Split Hygiene:** `{report['stage_9_split_hygiene']}`",
            f"- **Stage 10 Target-Leakage Quarantined Columns:** `{report['stage_10_leakage_quarantined_columns']}`",
            f"- **Stage 11 Mandatory Nulls:** `{report['stage_11_validation']['mandatory_column_nulls']}`",
            "",
        ]
        if report["stage_7_proxy_definitions"]:
            lines.append("## Proxy Metric Disclosures")
            for k, desc in report["stage_7_proxy_definitions"].items():
                lines.append(f"- `{k}`: {desc}")
            lines.append("")
        if report["notes"]:
            lines.append("## Notes")
            for note in report["notes"]:
                lines.append(f"- {note}")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths["md_report"] = md_path

    return paths

