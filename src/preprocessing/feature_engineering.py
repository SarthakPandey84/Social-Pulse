"""Feature engineering module extracting high-intent signals and engagement ratios.

Handles nullable fields cleanly without assuming missing metrics are zero.
Safely guards against division-by-zero across all calculated ratios.
Audits metric availability, partial vs complete derivations, and explicit zero handling.
"""

from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd
from src.data_ingestion.schemas import UNIFIED_POST_FIELDS, get_applicable_metrics


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

    # Trackers for derived features and their audit metadata
    high_intent = []
    high_intent_completeness = []
    high_intent_contrib = []

    save_share_ratios = []
    save_share_status = []

    intent_passive_ratios = []
    intent_passive_status = []

    views_eng_rates = []
    views_eng_contrib = []

    reach_eng_rates = []
    reach_eng_contrib = []

    equiv_full_watches = []
    equiv_full_watch_status = []

    completeness_pcts = []
    unavailability_audits = []

    for _, row in df_feat.iterrows():
        platform = str(row.get("platform", "unknown")).lower()
        content_type = str(row.get("content_type", "unknown")).lower()

        s = row.get("shares")
        b = row.get("saves")
        l = row.get("likes")
        c = row.get("comments_count")
        v = row.get("views")
        r = row.get("reach")
        ret = row.get("retention_rate")

        s_val = float(s) if pd.notna(s) else None
        b_val = float(b) if pd.notna(b) else None
        l_val = float(l) if pd.notna(l) else None
        c_val = float(c) if pd.notna(c) else None
        v_val = float(v) if pd.notna(v) else None
        r_val = float(r) if pd.notna(r) else None
        ret_val = float(ret) if pd.notna(ret) else None

        # -------------------------------------------------------------
        # 1. High-Intent Actions & Completeness Tracking
        # -------------------------------------------------------------
        if s_val is not None and b_val is not None:
            high_intent.append(int(s_val + b_val))
            high_intent_completeness.append("complete")
            high_intent_contrib.append("shares,saves")
        elif s_val is not None:
            high_intent.append(int(s_val))
            high_intent_completeness.append("partial_shares_only")
            high_intent_contrib.append("shares")
        elif b_val is not None:
            high_intent.append(int(b_val))
            high_intent_completeness.append("partial_saves_only")
            high_intent_contrib.append("saves")
        else:
            high_intent.append(pd.NA)
            high_intent_completeness.append("unmeasured")
            high_intent_contrib.append("none")

        # -------------------------------------------------------------
        # 2. Save-to-Share Ratio (Zero-Share Explicit Handling)
        # -------------------------------------------------------------
        if s_val is not None and b_val is not None:
            if s_val > 0:
                save_share_ratios.append(round(b_val / s_val, 2))
                save_share_status.append("valid_ratio")
            elif s_val == 0 and b_val > 0:
                # Mathematical division by zero: undefined / infinite ratio
                save_share_ratios.append(pd.NA)
                save_share_status.append("undefined_zero_shares")
            elif s_val == 0 and b_val == 0:
                save_share_ratios.append(0.0)
                save_share_status.append("zero_both")
            else:
                save_share_ratios.append(pd.NA)
                save_share_status.append("undefined")
        else:
            save_share_ratios.append(pd.NA)
            save_share_status.append("unmeasured")

        # -------------------------------------------------------------
        # 3. Intent-to-Passive Ratio (Zero-Likes Explicit Handling)
        # -------------------------------------------------------------
        hi_val = high_intent[-1]
        hi_num = float(hi_val) if pd.notna(hi_val) else None

        if hi_num is not None and l_val is not None:
            if l_val > 0:
                intent_passive_ratios.append(round(hi_num / l_val, 3))
                intent_passive_status.append("valid_ratio")
            elif l_val == 0 and hi_num > 0:
                # High intent occurred with zero likes: undefined ratio
                intent_passive_ratios.append(pd.NA)
                intent_passive_status.append("undefined_zero_likes")
            elif l_val == 0 and hi_num == 0:
                intent_passive_ratios.append(0.0)
                intent_passive_status.append("zero_both")
            else:
                intent_passive_ratios.append(pd.NA)
                intent_passive_status.append("undefined")
        else:
            intent_passive_ratios.append(pd.NA)
            intent_passive_status.append("unmeasured")

        # -------------------------------------------------------------
        # 4. Overall Engagement Rate (Views vs Reach Strictly Separated)
        # -------------------------------------------------------------
        # Interactions sum
        observed_interactions = []
        interaction_sum = 0.0
        if l_val is not None:
            interaction_sum += l_val
            observed_interactions.append("likes")
        if c_val is not None:
            interaction_sum += c_val
            observed_interactions.append("comments")
        if s_val is not None:
            interaction_sum += s_val
            observed_interactions.append("shares")
        if b_val is not None:
            interaction_sum += b_val
            observed_interactions.append("saves")

        contrib_str = ",".join(observed_interactions) if observed_interactions else "none"

        # Views engagement rate: strictly views denominator
        if v_val is not None and v_val > 0 and observed_interactions:
            views_eng_rates.append(round(interaction_sum / v_val, 4))
            views_eng_contrib.append(contrib_str)
        else:
            views_eng_rates.append(pd.NA)
            views_eng_contrib.append("none")

        # Reach engagement rate: strictly reach denominator
        if r_val is not None and r_val > 0 and observed_interactions:
            reach_eng_rates.append(round(interaction_sum / r_val, 4))
            reach_eng_contrib.append(contrib_str)
        else:
            reach_eng_rates.append(pd.NA)
            reach_eng_contrib.append("none")

        # -------------------------------------------------------------
        # 5. Equivalent Full-Watch Views (Effective Watch Views)
        # -------------------------------------------------------------
        is_video = content_type in {"reel", "short", "video", "short video"}
        if is_video and v_val is not None and ret_val is not None:
            equiv_full_watches.append(round(v_val * ret_val, 1))
            equiv_full_watch_status.append("computed")
        elif not is_video:
            equiv_full_watches.append(pd.NA)
            equiv_full_watch_status.append("not_applicable_non_video")
        else:
            equiv_full_watches.append(pd.NA)
            equiv_full_watch_status.append("unmeasured_missing_views_or_retention")

        # -------------------------------------------------------------
        # 6. Availability-Aware Data Completeness Audit
        # -------------------------------------------------------------
        applicable_set = get_applicable_metrics(platform, content_type, access_tier=access_tier)
        observed_count = sum(1 for m in applicable_set if m in row.index and pd.notna(row[m]))
        completeness = (observed_count / len(applicable_set)) * 100.0 if applicable_set else 100.0
        completeness_pcts.append(round(completeness, 1))

        # Record why unobserved fields are null
        unavail_reasons = []
        for field_name in ["views", "impressions", "reach", "likes", "comments_count", "shares", "saves", "retention_rate"]:
            val = row.get(field_name)
            if pd.isna(val) or val is None:
                if field_name not in applicable_set:
                    if field_name == "saves" and platform == "youtube":
                        unavail_reasons.append("saves:not_applicable_youtube")
                    elif field_name in {"views", "retention_rate"} and not is_video:
                        unavail_reasons.append(f"{field_name}:not_applicable_static_media")
                    else:
                        unavail_reasons.append(f"{field_name}:unavailable_in_{access_tier}")
                else:
                    unavail_reasons.append(f"{field_name}:missing_applicable")
        unavailability_audits.append("; ".join(unavail_reasons) if unavail_reasons else "all_applicable_observed")

    # Assign columns with appropriate nullable types
    df_feat["high_intent_actions"] = pd.Series(high_intent, dtype="Int64", index=df_feat.index)
    df_feat["high_intent_completeness"] = pd.Series(high_intent_completeness, dtype="string", index=df_feat.index)
    df_feat["high_intent_contributing_metrics"] = pd.Series(high_intent_contrib, dtype="string", index=df_feat.index)

    df_feat["save_to_share_ratio"] = pd.Series(save_share_ratios, dtype="Float64", index=df_feat.index)
    df_feat["save_share_ratio_status"] = pd.Series(save_share_status, dtype="string", index=df_feat.index)

    df_feat["intent_to_passive_ratio"] = pd.Series(intent_passive_ratios, dtype="Float64", index=df_feat.index)
    df_feat["intent_passive_ratio_status"] = pd.Series(intent_passive_status, dtype="string", index=df_feat.index)

    df_feat["views_engagement_rate"] = pd.Series(views_eng_rates, dtype="Float64", index=df_feat.index)
    df_feat["views_engagement_denominator"] = pd.Series(["views"] * len(df_feat), dtype="string", index=df_feat.index)
    df_feat["views_engagement_contributing_metrics"] = pd.Series(views_eng_contrib, dtype="string", index=df_feat.index)

    df_feat["reach_engagement_rate"] = pd.Series(reach_eng_rates, dtype="Float64", index=df_feat.index)
    df_feat["reach_engagement_denominator"] = pd.Series(["reach"] * len(df_feat), dtype="string", index=df_feat.index)
    df_feat["reach_engagement_contributing_metrics"] = pd.Series(reach_eng_contrib, dtype="string", index=df_feat.index)

    # Backwards-compatible overall_engagement_rate alias
    df_feat["overall_engagement_rate"] = df_feat["views_engagement_rate"]

    # Equivalent full-watch views
    df_feat["equivalent_full_watch_views"] = pd.Series(equiv_full_watches, dtype="Float64", index=df_feat.index)
    df_feat["effective_watch_views"] = df_feat["equivalent_full_watch_views"]  # Backwards compatibility alias
    df_feat["equivalent_full_watch_status"] = pd.Series(equiv_full_watch_status, dtype="string", index=df_feat.index)

    # Data completeness percentage (availability-aware) & audit log
    df_feat["data_completeness_pct"] = pd.Series(completeness_pcts, dtype="Float64", index=df_feat.index)
    df_feat["metric_unavailability_audit"] = pd.Series(unavailability_audits, dtype="string", index=df_feat.index)

    return df_feat
