"""CLI script to normalize and clean all downloaded external datasets into Parquet artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion.external_adapters import prepare_all_installed_external_datasets
from src.data_ingestion.external_datasets import generate_raw_quality_report, list_dataset_specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize downloaded external datasets through the 12-stage quality pipeline."
    )
    parser.add_argument("--refresh-raw-reports", action="store_true", help="Regenerate raw quality reports first.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary to stdout.")
    args = parser.parse_args(argv)

    root = PROJECT_ROOT
    if args.refresh_raw_reports:
        for spec in list_dataset_specs(include_kaggle=False):
            generate_raw_quality_report(spec.dataset_id, root_dir=root)

    results = prepare_all_installed_external_datasets(root_dir=root)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for ds_id, info in results.items():
            status = info.get("status", "prepared")
            rows = info.get("row_counts", {})
            print(f"- {ds_id}: status={status} rows={rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

