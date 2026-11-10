"""CLI utility for reproducible acquisition, integrity verification, and raw quality auditing of external datasets.

Usage examples:
    python scripts/download_external_datasets.py --list
    python scripts/download_external_datasets.py --all-accessible
    python scripts/download_external_datasets.py --dataset tiktok_engagement_200k
    python scripts/download_external_datasets.py --dataset youtube_trending_kaggle --report-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_ingestion.external_datasets import (
    DATASET_REGISTRY,
    download_all_accessible,
    download_dataset,
    generate_raw_quality_report,
    list_dataset_specs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproducibly download, verify, and audit external SocialPulse datasets."
    )
    parser.add_argument("--list", action="store_true", help="List all registered external datasets and exit.")
    parser.add_argument("--dataset", action="append", default=[], help="Download/inspect specific dataset_id (repeatable).")
    parser.add_argument("--all-accessible", action="store_true", help="Download all accessible datasets and record Kaggle status.")
    parser.add_argument(
        "--include-pending-review",
        action="store_true",
        default=False,
        help="Explicitly opt in to downloading datasets whose license_status is PENDING_REVIEW (excluded by default).",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip Kaggle-gated dataset specs when running --all-accessible.",
    )
    parser.add_argument("--report-only", action="store_true", help="Re-run raw quality report on existing files only.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if verified files exist.")
    parser.add_argument("--root-dir", type=Path, default=ROOT_DIR, help="Workspace root directory.")
    args = parser.parse_args(argv)

    if args.list or (not args.dataset and not args.all_accessible):
        summary = [spec.to_summary_dict() for spec in list_dataset_specs(include_kaggle=True)]
        print(json.dumps(summary, indent=2))
        return 0

    include_pending = bool(args.include_pending_review)

    if args.report_only:
        targets = args.dataset or list(DATASET_REGISTRY.keys())
        reports = {ds_id: generate_raw_quality_report(ds_id, root_dir=args.root_dir) for ds_id in targets}
        print(json.dumps({k: len(v["tables"]) for k, v in reports.items()}, indent=2))
        return 0

    if args.all_accessible:
        results = download_all_accessible(
            root_dir=args.root_dir,
            include_pending_review=include_pending,
            attempt_kaggle=not args.skip_kaggle,
            force=args.force,
        )
        print(json.dumps({k: v.get("status") for k, v in results.items()}, indent=2))
        return 0

    results = {}
    for ds_id in args.dataset:
        results[ds_id] = download_dataset(
            ds_id,
            root_dir=args.root_dir,
            include_pending_review=include_pending,
            force=args.force,
        )
    print(json.dumps({k: v.get("status") for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

