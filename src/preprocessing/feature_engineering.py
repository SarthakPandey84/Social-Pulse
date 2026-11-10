"""Feature engineering module extracting high-intent signals and engagement ratios.

Handles nullable fields cleanly without assuming missing metrics are zero.
Safely guards against division-by-zero across all calculated ratios.
Audits metric availability, partial vs complete derivations, and explicit zero handling.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from src.data_ingestion.schemas import get_applicable_metrics


_CONTRIB_INTERACTION_COLS = ("likes", "comments", "shares", "saves")
_CONTRIB_LUT: Tuple[str, ...] = tuple(
    ",".join(col for bit, col in enumerate(_CONTRIB_INTERACTION_COLS) if (mask & (1 << bit))) or "none"
    for mask in range(16)
)

_AUDIT_FIELDS = (
    "views",
    "impressions",
    "reach",
    "likes",
    "comments_count",
    "shares",
    "saves",
    "retention_rate",
)


def _extract_numeric_col(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(float64_values, non_null_mask)`` without raising on missing columns."""
    if col not in df.columns:
        n = len(df)
        return np.full(n, np.nan, dtype="float64"), np.zeros(n, dtype=bool)
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    return arr, ~np.isnan(arr)


def add_engagement_features(df: pd.DataFrame, access_tier: str = "owner_oauth") -> pd.DataFrame:
    """Calculate key engagement ratios while strictly preserving missing-value semantics.

    Audit Standards Applied:
    1. high_intent_actions: Tracks whether the derivation is 'complete' (both shares & saves observed)
       or 'partial_shares_only' / 'partial_saves_only'.
    2. save_to_share_ratio: Handles zero shares explicitly (returns pd.NA with 'undefined_zero_shares'
       status rather than silently substituting 1).
    3. intent_to_passive_ratio: Handles zero likes explicitly (returns pd.NA with 'undefined_zero_likes').
    4. equivalent_full_watch_views (effective_watch_views): Documents the calculation as estimated
       equivalent 100% full-duration watches (views * retention_rate).
    5. views_engagement_rate vs. reach_engagement_rate: Kept strictly separate with explicit denominator labels.
    6. data_completeness_pct: Evaluated strictly against applicable, collectible fields for each platform
       and content format rather than an arbitrary static denominator.
    """
    if df.empty:
        return df.copy()

    df_feat = df.copy()
    n = len(df_feat)

    s_arr, has_s = _extract_numeric_col(df_feat, "shares")
    b_arr, has_b = _extract_numeric_col(df_feat, "saves")
    l_arr, has_l = _extract_numeric_col(df_feat, "likes")
    c_arr, has_c = _extract_numeric_col(df_feat, "comments_count")
    v_arr, has_v = _extract_numeric_col(df_feat, "views")
    r_arr, has_r = _extract_numeric_col(df_feat, "reach")
    ret_arr, has_ret = _extract_numeric_col(df_feat, "retention_rate")

    # -------------------------------------------------------------
    # 1. High-Intent Actions & Completeness Tracking (Vectorized)
    # -------------------------------------------------------------
    both_sb = has_s & has_b
    only_s = has_s & (~has_b)
    only_b = (~has_s) & has_b
    has_hi = has_s | has_b

    hi_raw = np.where(both_sb, s_arr + b_arr, np.where(only_s, s_arr, np.where(only_b, b_arr, 0.0)))
    hi_int = np.trunc(hi_raw)

    high_intent_series = pd.Series(pd.NA, index=df_feat.index, dtype="Int64")
    if has_hi.any():
        high_intent_series.iloc[has_hi] = hi_int[has_hi].astype("int64")

    high_intent_completeness = np.select(
        [both_sb, only_s, only_b],
        ["complete", "partial_shares_only", "partial_saves_only"],
        default="unmeasured",
    )
    high_intent_contrib = np.select(
        [both_sb, only_s, only_b],
        ["shares,saves", "shares", "saves"],
        default="none",
    )

    # -------------------------------------------------------------
    # 2. Save-to-Share Ratio (Zero-Share Explicit Handling)
    # -------------------------------------------------------------
    ssr_valid = both_sb & (s_arr > 0)
    ssr_zero_shares = both_sb & (s_arr == 0) & (b_arr > 0)
    ssr_zero_both = both_sb & (s_arr == 0) & (b_arr == 0)
    ssr_other_undef = both_sb & (~ssr_valid) & (~ssr_zero_shares) & (~ssr_zero_both)

    ssr_series = pd.Series(pd.NA, index=df_feat.index, dtype="Float64")
    if ssr_valid.any():
        ssr_series.iloc[ssr_valid] = np.round(b_arr[ssr_valid] / s_arr[ssr_valid], 2)
    if ssr_zero_both.any():
        ssr_series.iloc[ssr_zero_both] = 0.0

    save_share_status = np.select(
        [ssr_valid, ssr_zero_shares, ssr_zero_both, ssr_other_undef],
        ["valid_ratio", "undefined_zero_shares", "zero_both", "undefined"],
        default="unmeasured",
    )

    # -------------------------------------------------------------
    # 3. Intent-to-Passive Ratio (Zero-Likes Explicit Handling)
    # -------------------------------------------------------------
    both_hil = has_hi & has_l
    ipr_valid = both_hil & (l_arr > 0)
    ipr_zero_likes = both_hil & (l_arr == 0) & (hi_int > 0)
    ipr_zero_both = both_hil & (l_arr == 0) & (hi_int == 0)
    ipr_other_undef = both_hil & (~ipr_valid) & (~ipr_zero_likes) & (~ipr_zero_both)

    ipr_series = pd.Series(pd.NA, index=df_feat.index, dtype="Float64")
    if ipr_valid.any():
        ipr_series.iloc[ipr_valid] = np.round(hi_int[ipr_valid] / l_arr[ipr_valid], 3)
    if ipr_zero_both.any():
        ipr_series.iloc[ipr_zero_both] = 0.0

    intent_passive_status = np.select(
        [ipr_valid, ipr_zero_likes, ipr_zero_both, ipr_other_undef],
        ["valid_ratio", "undefined_zero_likes", "zero_both", "undefined"],
        default="unmeasured",
    )

    # -------------------------------------------------------------
    # 4. Overall Engagement Rate (Views vs Reach Strictly Separated)
    # -------------------------------------------------------------
    interaction_sum = (
        np.where(has_l, l_arr, 0.0)
        + np.where(has_c, c_arr, 0.0)
        + np.where(has_s, s_arr, 0.0)
        + np.where(has_b, b_arr, 0.0)
    )
    contrib_bits = (
        has_l.astype(np.int8)
        | (has_c.astype(np.int8) << 1)
        | (has_s.astype(np.int8) << 2)
        | (has_b.astype(np.int8) << 3)
    )
    has_any_inter = contrib_bits > 0
    contrib_lut = np.asarray(_CONTRIB_LUT, dtype=object)
    contrib_arr = contrib_lut[contrib_bits]

    v_valid = has_v & (v_arr > 0) & has_any_inter
    views_eng_series = pd.Series(pd.NA, index=df_feat.index, dtype="Float64")
    if v_valid.any():
        views_eng_series.iloc[v_valid] = np.round(interaction_sum[v_valid] / v_arr[v_valid], 4)
    views_eng_contrib = np.where(v_valid, contrib_arr, "none")

    r_valid = has_r & (r_arr > 0) & has_any_inter
    reach_eng_series = pd.Series(pd.NA, index=df_feat.index, dtype="Float64")
    if r_valid.any():
        reach_eng_series.iloc[r_valid] = np.round(interaction_sum[r_valid] / r_arr[r_valid], 4)
    reach_eng_contrib = np.where(r_valid, contrib_arr, "none")

    # -------------------------------------------------------------
    # 5. Equivalent Full-Watch Views (Effective Watch Views)
    # -------------------------------------------------------------
    if "content_type" in df_feat.columns:
        c_raw = df_feat["content_type"]
        c_is_na = c_raw.isna().to_numpy(dtype=bool)
        c_str_lower = c_raw.astype(str).str.strip().str.lower()
        is_video = (~c_is_na) & c_str_lower.isin({"reel", "short", "video", "short video", "short_video"}).to_numpy(dtype=bool)
    else:
        c_is_na = np.zeros(n, dtype=bool)
        c_str_lower = pd.Series(["unknown"] * n, index=df_feat.index)
        is_video = np.zeros(n, dtype=bool)

    eq_comp = is_video & has_v & has_ret
    eq_series = pd.Series(pd.NA, index=df_feat.index, dtype="Float64")
    if eq_comp.any():
        eq_series.iloc[eq_comp] = np.round(v_arr[eq_comp] * ret_arr[eq_comp], 1)
    equiv_full_watch_status = np.select(
        [eq_comp, ~is_video],
        ["computed", "not_applicable_non_video"],
        default="unmeasured_missing_views_or_retention",
    )

    # -------------------------------------------------------------
    # 6. Availability-Aware Data Completeness Audit (Grouped by pair)
    # -------------------------------------------------------------
    if "platform" in df_feat.columns:
        p_raw = df_feat["platform"]
        p_is_na = p_raw.isna().to_numpy(dtype=bool)
        p_str_lower = p_raw.astype(str).str.lower()
    else:
        p_is_na = np.zeros(n, dtype=bool)
        p_str_lower = pd.Series(["unknown"] * n, index=df_feat.index)

    # Match row.get("col") / pd.notna per-column presence map cleanly
    col_notna: Dict[str, np.ndarray] = {}
    candidate_fields: Set[str] = set(_AUDIT_FIELDS) | {"follower_count"}
    for fname in candidate_fields:
        if fname in df_feat.columns:
            col_notna[fname] = df_feat[fname].notna().to_numpy(dtype=bool)
        else:
            col_notna[fname] = np.zeros(n, dtype=bool)

    # Precompute 8-bit missing mask across the audited fields (1 = missing/None/NaN)
    audit_missing_bits = np.zeros(n, dtype=np.int16)
    for bit_idx, fname in enumerate(_AUDIT_FIELDS):
        audit_missing_bits |= (~col_notna[fname]).astype(np.int16) << bit_idx

    pair_keys_df = pd.DataFrame(
        {
            "p": np.where(p_is_na, "__NA__", p_str_lower.to_numpy(dtype=object)),
            "c": np.where(c_is_na, "__NA__", c_str_lower.to_numpy(dtype=object)),
        },
        index=np.arange(n),
    )

    completeness_arr = np.full(n, 100.0, dtype="float64")
    unavail_audit_arr = np.empty(n, dtype=object)

    for (p_key, c_key), grp in pair_keys_df.groupby(["p", "c"], sort=False):
        idx = grp.index.to_numpy()
        p_arg = None if p_key == "__NA__" else p_key
        c_arg = None if c_key == "__NA__" else c_key
        p_norm = "unknown" if p_key == "__NA__" else p_key
        pair_is_video = (c_key != "__NA__") and (c_key in {"reel", "short", "video", "short video", "short_video"})

        applicable_set = get_applicable_metrics(p_arg, c_arg, access_tier=access_tier)
        if applicable_set:
            obs_counts = np.zeros(len(idx), dtype=np.int32)
            for m in applicable_set:
                if m in df_feat.columns:
                    obs_counts += col_notna[m][idx].astype(np.int32)
            completeness_arr[idx] = np.round((obs_counts / float(len(applicable_set))) * 100.0, 1)

        field_tags: List[str] = []
        for fname in _AUDIT_FIELDS:
            if fname not in applicable_set:
                if fname == "saves" and p_norm == "youtube":
                    field_tags.append("saves:not_applicable_youtube")
                elif fname in {"views", "retention_rate"} and not pair_is_video:
                    field_tags.append(f"{fname}:not_applicable_static_media")
                else:
                    field_tags.append(f"{fname}:unavailable_in_{access_tier}")
            else:
                field_tags.append(f"{fname}:missing_applicable")

        grp_bits = audit_missing_bits[idx]
        unique_bits = np.unique(grp_bits)
        bit_to_str: Dict[int, str] = {}
        for mask_int in unique_bits:
            m_int = int(mask_int)
            reasons = [field_tags[b] for b in range(len(_AUDIT_FIELDS)) if (m_int & (1 << b))]
            bit_to_str[m_int] = "; ".join(reasons) if reasons else "all_applicable_observed"

        if len(bit_to_str) == 1:
            only_str = next(iter(bit_to_str.values()))
            unavail_audit_arr[idx] = only_str
        else:
            # Map strictly via a tiny 256-slot lookup array
            lut = np.empty(256, dtype=object)
            for k_bit, v_str in bit_to_str.items():
                lut[k_bit] = v_str
            unavail_audit_arr[idx] = lut[grp_bits]

    # Assign columns with appropriate nullable types
    df_feat["high_intent_actions"] = high_intent_series
    df_feat["high_intent_completeness"] = pd.Series(high_intent_completeness, dtype="string", index=df_feat.index)
    df_feat["high_intent_contributing_metrics"] = pd.Series(high_intent_contrib, dtype="string", index=df_feat.index)

    df_feat["save_to_share_ratio"] = ssr_series
    df_feat["save_share_ratio_status"] = pd.Series(save_share_status, dtype="string", index=df_feat.index)

    df_feat["intent_to_passive_ratio"] = ipr_series
    df_feat["intent_passive_ratio_status"] = pd.Series(intent_passive_status, dtype="string", index=df_feat.index)

    df_feat["views_engagement_rate"] = views_eng_series
    df_feat["views_engagement_denominator"] = pd.Series("views", dtype="string", index=df_feat.index)
    df_feat["views_engagement_contributing_metrics"] = pd.Series(views_eng_contrib, dtype="string", index=df_feat.index)

    df_feat["reach_engagement_rate"] = reach_eng_series
    df_feat["reach_engagement_denominator"] = pd.Series("reach", dtype="string", index=df_feat.index)
    df_feat["reach_engagement_contributing_metrics"] = pd.Series(reach_eng_contrib, dtype="string", index=df_feat.index)

    # Backwards-compatible overall_engagement_rate alias
    df_feat["overall_engagement_rate"] = df_feat["views_engagement_rate"]

    # Equivalent full-watch views
    df_feat["equivalent_full_watch_views"] = eq_series
    df_feat["effective_watch_views"] = df_feat["equivalent_full_watch_views"]  # Backwards compatibility alias
    df_feat["equivalent_full_watch_status"] = pd.Series(equiv_full_watch_status, dtype="string", index=df_feat.index)

    # Data completeness percentage (availability-aware) & audit log
    df_feat["data_completeness_pct"] = pd.Series(completeness_arr, dtype="Float64", index=df_feat.index)
    df_feat["metric_unavailability_audit"] = pd.Series(unavail_audit_arr, dtype="string", index=df_feat.index)

    return df_feat
