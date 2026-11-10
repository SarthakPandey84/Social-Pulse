"""CLI runner for real-data empirical validation across Modules 1–7."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.real_data_benchmarks import run_all_real_data_benchmarks


def _update_build_provenance(provenance_path: Path, key: str, value: str) -> None:
    """Persist runtime benchmark timestamp into gitignored build provenance."""
    payload: Dict[str, Any] = {}
    if provenance_path.is_file():
        try:
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload[key] = value
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(provenance_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    out_path = PROJECT_ROOT / "models" / "real_data_benchmarks.json"
    summary = run_all_real_data_benchmarks(
        processed_external_root=PROJECT_ROOT / "data" / "processed" / "external",
        output_path=out_path,
        include_volatile_timestamps=False,
    )
    _update_build_provenance(
        PROJECT_ROOT / "models" / "build_provenance.json",
        "real_data_benchmarks_generated_at_utc",
        datetime.now(timezone.utc).isoformat(),
    )
    print(f"Real-data benchmarks written to: {out_path}")
    m2 = summary["modules"].get("module2", {})
    for key, val in m2.items():
        b_name = val.get("best_model")
        best = val.get("models", {}).get(b_name, {})
        print(
            f"[M2 {key}] total={val.get('total_evaluated_rows')} train={val.get('train_rows')} test={val.get('test_rows')} "
            f"overlap={val.get('creator_overlap_count')} best={b_name} R2={best.get('r2_log1p')} "
            f"Spearman={best.get('spearman_rho')} Cov90={best.get('empirical_90pct_test_coverage')} "
            f"lift={val.get('rmse_lift_over_dummy_pct')}%"
        )
    m3 = summary["modules"].get("module3", {})
    if "instagram_sentiment_heldout" in m3:
        ig = m3["instagram_sentiment_heldout"]
        tf_full = ig["full_heldout_test"]["tfidf_logreg"]
        tf_uniq = ig["strict_unique_heldout_test"]["tfidf_logreg"]
        lex = ig["full_heldout_test"]["lexicon_baseline"]
        print(
            f"[M3 IG Sentiment Heldout N_full={ig['test_samples_full']} N_unique={ig['test_samples_strict_unique']}] "
            f"TFIDF_full acc={tf_full['accuracy']} macro_f1={tf_full['macro_f1']} ece={tf_full['ece']} | "
            f"TFIDF_uniq acc={tf_uniq['accuracy']} macro_f1={tf_uniq['macro_f1']} | "
            f"Lexicon acc_eval={lex['accuracy_on_evaluated']} ({lex['evaluated_count']}/{ig['test_samples_full']})"
        )
    if "lokananta_indonesian_guardrail" in m3:
        lk = m3["lokananta_indonesian_guardrail"]
        print(
            f"[M3 Lokananta N={lk['total_comments']}] auto_unsupported={lk['auto_unsupported_language_pct']}% "
            f"auto_abstained={lk['auto_abstained_total_pct']}% hinted_rejected={lk['hinted_unsupported_language_count']}"
        )
    m4 = summary["modules"].get("module4", {})
    if "sponsored_unsponsored_observational" in m4:
        sp = m4["sponsored_unsponsored_observational"]
        cl_list = sp.get("creator_clustered_comparisons", [])
        cl0 = cl_list[0] if cl_list else {}
        print(
            f"[M4 Sponsored IG] posts={sp.get('n_posts_total')} creators={sp.get('n_creators_total')} "
            f"clustered_likes p_adj={cl0.get('p_value_adjusted')} method={cl0.get('clustering_method')} "
            f"is_causal={sp.get('is_causal')}"
        )
    m7 = summary["modules"].get("module7", {})
    for col, res in m7.items():
        print(
            f"[M7 {col}] status={res['status']} pts={res['history_points']} ({res['history_start']}..{res['history_end']}) "
            f"champ={res['champion_model']} origins={res['validation_origins_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
