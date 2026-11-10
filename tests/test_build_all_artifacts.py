"""Tests for scripts/build_all_artifacts.py: clean build, idempotency, and strict failure guards."""

import concurrent.futures
from contextlib import closing
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading
import time

import pandas as pd
import pytest

from scripts.build_all_artifacts import (
    build_all_artifacts,
    main as build_cli_main,
    validate_source_inputs,
    verify_generated_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _copy_sample_inputs(target_root: Path) -> None:
    """Copy the 4 read-only sample CSVs into an isolated temporary workspace."""
    sample_dest = target_root / "data" / "sample"
    sample_dest.mkdir(parents=True, exist_ok=True)
    for csv_path in (REPO_ROOT / "data" / "sample").glob("*.csv"):
        shutil.copy2(csv_path, sample_dest / csv_path.name)


def _hash_sample_dir(target_root: Path) -> dict:
    return {
        p.name: hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted((target_root / "data" / "sample").glob("*.csv"))
    }


def test_build_all_artifacts_succeeds_and_preserves_sample_inputs(tmp_path: Path):
    """Verify full build from only data/sample/ produces all artifacts and leaves sample inputs untouched."""
    _copy_sample_inputs(tmp_path)
    before_hashes = _hash_sample_dir(tmp_path)

    result = build_all_artifacts(root_dir=tmp_path)
    after_hashes = _hash_sample_dir(tmp_path)

    assert before_hashes == after_hashes, "Sample input CSVs must never be overwritten during build."
    assert result["mode"] == "build"

    v = result["verification"]
    assert v["processed_posts_rows"] == 121
    assert v["sqlite_posts_rows"] == 121
    assert v["sqlite_latest_posts_rows"] == 120
    assert v["processed_comments_rows"] == 500
    assert v["processed_ab_tests_rows"] == 380
    assert v["virality_mature_posts"] >= 50

    expected_files = [
        tmp_path / "data" / "processed" / "processed_posts.parquet",
        tmp_path / "data" / "processed" / "processed_posts.csv",
        tmp_path / "data" / "processed" / "processed_comments.parquet",
        tmp_path / "data" / "processed" / "processed_comments.csv",
        tmp_path / "data" / "processed" / "processed_ab_tests.parquet",
        tmp_path / "data" / "processed" / "processed_ab_tests.csv",
        tmp_path / "database" / "social_analytics.db",
        tmp_path / "models" / "virality_predictor.joblib",
        tmp_path / "models" / "virality_predictor_metadata.json",
        tmp_path / "models" / "relatability_classifier.joblib",
        tmp_path / "models" / "relatability_metadata.json",
    ]
    for path in expected_files:
        assert path.exists() and path.stat().st_size > 0, f"Missing or empty output: {path}"

    # Verify raw directory was not polluted with snapshot files
    raw_dir = tmp_path / "data" / "raw"
    assert not raw_dir.exists() or len(list(raw_dir.glob("*.parquet"))) == 0


def test_build_all_artifacts_is_idempotent_across_repeated_runs(tmp_path: Path):
    """Verify running build_all_artifacts repeatedly produces identical counts without SQLite duplicates."""
    _copy_sample_inputs(tmp_path)

    first = build_all_artifacts(root_dir=tmp_path)
    second = build_all_artifacts(root_dir=tmp_path)
    validated = build_all_artifacts(root_dir=tmp_path, validate_only=True)

    assert first["verification"] == second["verification"] == validated["verification"]

    with closing(sqlite3.connect(tmp_path / "database" / "social_analytics.db")) as conn:
        posts_count = int(conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0])
        latest_count = int(conn.execute("SELECT COUNT(*) FROM latest_posts").fetchone()[0])
    assert posts_count == 121
    assert latest_count == 120


@pytest.mark.parametrize(
    "missing_filename",
    [
        "sample_posts.csv",
        "sample_comments.csv",
        "sample_experiments.csv",
        "sample_trends.csv",
    ],
)
def test_build_fails_fast_when_required_sample_input_is_missing(tmp_path: Path, missing_filename: str):
    """Verify missing any source sample CSV raises FileNotFoundError before writing outputs."""
    _copy_sample_inputs(tmp_path)
    (tmp_path / "data" / "sample" / missing_filename).unlink()

    with pytest.raises(FileNotFoundError, match=missing_filename):
        build_all_artifacts(root_dir=tmp_path)

    assert not (tmp_path / "data" / "processed").exists()
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "database").exists()


def test_build_rejects_corrupted_sample_posts(tmp_path: Path):
    """Verify missing schema columns or negative metrics in sample_posts.csv are rejected upfront."""
    _copy_sample_inputs(tmp_path)
    posts_path = tmp_path / "data" / "sample" / "sample_posts.csv"

    # Case 1: Negative views
    df_bad = pd.read_csv(posts_path)
    df_bad.loc[0, "views"] = -500
    df_bad.to_csv(posts_path, index=False)
    with pytest.raises(ValueError, match="negative|validation failed"):
        validate_source_inputs(tmp_path)

    # Case 2: Missing modeling context column
    df_missing_col = df_bad.drop(columns=["topic"])
    df_missing_col.to_csv(posts_path, index=False)
    with pytest.raises(ValueError, match="topic"):
        validate_source_inputs(tmp_path)

    # Case 3: Insufficient mature rows (< 50)
    _copy_sample_inputs(tmp_path)
    df_short = pd.read_csv(posts_path).head(20)
    df_short.to_csv(posts_path, index=False)
    with pytest.raises(ValueError, match="mature unique posts"):
        build_all_artifacts(root_dir=tmp_path)


def test_build_rejects_corrupted_comments_experiments_and_trends(tmp_path: Path):
    """Verify corrupted comments, experiments, or trends CSVs fail loudly before mutating outputs."""
    # Corrupted trends: negative volume
    _copy_sample_inputs(tmp_path)
    trends_path = tmp_path / "data" / "sample" / "sample_trends.csv"
    df_tr = pd.read_csv(trends_path)
    df_tr.loc[0, "mention_volume"] = -10
    df_tr.to_csv(trends_path, index=False)
    with pytest.raises(ValueError, match="negative mention_volume"):
        validate_source_inputs(tmp_path)

    # Corrupted experiments: invalid experiment_type
    _copy_sample_inputs(tmp_path)
    exp_path = tmp_path / "data" / "sample" / "sample_experiments.csv"
    df_exp = pd.read_csv(exp_path)
    df_exp.loc[0, "experiment_type"] = "invalid_type"
    df_exp.to_csv(exp_path, index=False)
    with pytest.raises(ValueError, match="experiment_type"):
        validate_source_inputs(tmp_path)

    # Corrupted comments: single class only (cannot stratify)
    _copy_sample_inputs(tmp_path)
    cmt_path = tmp_path / "data" / "sample" / "sample_comments.csv"
    df_cmt = pd.read_csv(cmt_path)
    df_cmt["ground_truth_relatability"] = "Relatable"
    df_cmt.to_csv(cmt_path, index=False)
    with pytest.raises(ValueError, match="Relatable.*Neutral|stratified"):
        validate_source_inputs(tmp_path)


def test_post_build_verifier_catches_missing_or_corrupt_outputs(tmp_path: Path):
    """Verify post-build verifier catches missing, empty, or schema-broken artifacts."""
    _copy_sample_inputs(tmp_path)
    build_all_artifacts(root_dir=tmp_path)

    # Corrupt processed_comments.parquet by dropping a required column
    cmt_pq = tmp_path / "data" / "processed" / "processed_comments.parquet"
    df_cmt = pd.read_parquet(cmt_pq)
    df_cmt.drop(columns=["sentiment_score"]).to_parquet(cmt_pq, index=False)
    with pytest.raises(RuntimeError, match="sentiment_score"):
        verify_generated_artifacts(tmp_path)

    # Restore cleanly then delete a model artifact
    build_all_artifacts(root_dir=tmp_path)
    model_file = tmp_path / "models" / "relatability_classifier.joblib"
    model_file.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        verify_generated_artifacts(tmp_path)

    # Empty 0-byte artifact detection
    model_file.write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty"):
        verify_generated_artifacts(tmp_path)


def test_build_cli_exit_codes(tmp_path: Path):
    """Verify CLI returns 0 on valid build / --validate-only and 1 on missing inputs."""
    _copy_sample_inputs(tmp_path)
    assert build_cli_main(["--root", str(tmp_path)]) == 0
    assert build_cli_main(["--root", str(tmp_path), "--validate-only"]) == 0

    (tmp_path / "data" / "sample" / "sample_trends.csv").unlink()
    assert build_cli_main(["--root", str(tmp_path), "--validate-only"]) == 1


def _copy_clean_app_tree(target_root: Path) -> None:
    """Copy clean tracked runtime files (without processed/DB/model outputs) into tmp_path."""
    for rel_dir in ["src", "scripts", "app", "utils", "assets", ".streamlit"]:
        shutil.copytree(
            REPO_ROOT / rel_dir,
            target_root / rel_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    shutil.copy2(REPO_ROOT / "streamlit_app.py", target_root / "streamlit_app.py")
    _copy_sample_inputs(target_root)


def test_build_runs_before_ui_and_resolves_page3_dependency(tmp_path: Path):
    """Prove build_all_artifacts runs before any Streamlit page opens and unblocks locked Page 03."""
    _copy_clean_app_tree(tmp_path)
    page3_file = tmp_path / "app" / "pages" / "03_Sentiment_Analysis.py"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)

    # 1. Pre-build: opening Page 03 before artifacts exist must fail loudly (not masked)
    prebuild_probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "from streamlit.testing.v1 import AppTest; "
                f"at = AppTest.from_file({repr(str(page3_file))}, default_timeout=30); "
                "at.run(); "
                "assert len(at.exception) > 0, 'Expected KeyError on pre-build sample fallback'; "
                "assert 'relatability_label' in str(at.exception[0].message)"
            ),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert prebuild_probe.returncode == 0, prebuild_probe.stderr or prebuild_probe.stdout

    # 2. Build BEFORE any Streamlit page succeeds: must exit 0 and generate Module 3 artifacts
    build_run = subprocess.run(
        [
            sys.executable,
            "-B",
            str(tmp_path / "scripts" / "build_all_artifacts.py"),
            "--root",
            str(tmp_path),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert build_run.returncode == 0, build_run.stderr or build_run.stdout

    cmt_pq = tmp_path / "data" / "processed" / "processed_comments.parquet"
    assert cmt_pq.exists()
    assert (tmp_path / "models" / "relatability_classifier.joblib").exists()
    assert (tmp_path / "models" / "relatability_metadata.json").exists()
    df_cmt = pd.read_parquet(cmt_pq)
    for required_col in [
        "relatability_label",
        "relatability_confidence",
        "sentiment_label",
        "sentiment_score",
        "classification_status",
    ]:
        assert required_col in df_cmt.columns

    # 3. Post-build: locked Page 03 now renders cleanly without exceptions
    postbuild_probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "from streamlit.testing.v1 import AppTest; "
                f"at = AppTest.from_file({repr(str(page3_file))}, default_timeout=30); "
                "at.run(); "
                "assert len(at.exception) == 0, f'Unexpected Page 03 exceptions: {[e.message for e in at.exception]}'"
            ),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert postbuild_probe.returncode == 0, postbuild_probe.stderr or postbuild_probe.stdout


def test_tracked_model_metadata_is_byte_reproducible_while_preserving_truthful_provenance(tmp_path: Path):
    """Repeated builds must produce identical tracked JSONs while keeping truthful runtime timestamps separate."""
    _copy_sample_inputs(tmp_path)

    res1 = build_all_artifacts(root_dir=tmp_path)
    vir_meta_path = tmp_path / "models" / "virality_predictor_metadata.json"
    nlp_meta_path = tmp_path / "models" / "relatability_metadata.json"
    prov_path = tmp_path / "models" / "build_provenance.json"

    vir_bytes_1 = vir_meta_path.read_bytes()
    nlp_bytes_1 = nlp_meta_path.read_bytes()
    prov_1 = json.loads(prov_path.read_text(encoding="utf-8"))

    # Also confirm in-memory metrics/metadata still retain truthful runtime timestamps
    assert "candidate_metrics" in res1["outputs"]["virality_metrics"] or "mae" in res1["outputs"]["virality_metrics"]

    time.sleep(0.05)
    build_all_artifacts(root_dir=tmp_path)

    vir_bytes_2 = vir_meta_path.read_bytes()
    nlp_bytes_2 = nlp_meta_path.read_bytes()
    prov_2 = json.loads(prov_path.read_text(encoding="utf-8"))

    # Tracked JSON metadata artifacts must be byte-for-byte identical with LF endings
    assert vir_bytes_1 == vir_bytes_2
    assert nlp_bytes_1 == nlp_bytes_2
    assert b"\r\n" not in vir_bytes_1 and vir_bytes_1.endswith(b"\n")
    assert b"\r\n" not in nlp_bytes_1 and nlp_bytes_1.endswith(b"\n")

    vir_meta = json.loads(vir_bytes_1.decode("utf-8"))
    nlp_meta = json.loads(nlp_bytes_1.decode("utf-8"))
    assert "trained_at" not in vir_meta
    assert "trained_at" not in nlp_meta
    assert "trained_at" in vir_meta["metadata_reproducibility"]["intentionally_variable_fields_excluded"]
    assert "trained_at" in nlp_meta["metadata_reproducibility"]["intentionally_variable_fields_excluded"]

    # Truthful provenance timestamps must reflect actual distinct build runs
    t1 = datetime.fromisoformat(prov_1["virality_model_trained_at_utc"])
    t2 = datetime.fromisoformat(prov_2["virality_model_trained_at_utc"])
    assert t2 > t1
    assert datetime.fromisoformat(prov_2["nlp_model_trained_at_utc"]) > datetime.fromisoformat(prov_1["nlp_model_trained_at_utc"])
    assert prov_1["source_dataset_sha256"] == prov_2["source_dataset_sha256"]
    assert "sample_posts.csv" in prov_1["source_dataset_sha256"]


def _copy_tracked_clean_checkout(target_root: Path) -> None:
    """Copy clean tracked files mirroring a fresh GitHub checkout without generated artifacts."""
    _copy_clean_app_tree(target_root)
    models_dir = target_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for meta_file in (REPO_ROOT / "models").glob("*_metadata.json"):
        shutil.copy2(meta_file, models_dir / meta_file.name)


def test_clean_checkout_root_entrypoint_bootstrap_and_full_page_unlock(tmp_path: Path):
    """Root streamlit_app.py must auto-build artifacts once from clean checkout and unblock all pages."""
    _copy_tracked_clean_checkout(tmp_path)

    assert not (tmp_path / "data" / "processed").exists()
    assert not (tmp_path / "database" / "social_analytics.db").exists()
    assert not (tmp_path / "models" / "virality_predictor.joblib").exists()
    assert not (tmp_path / "models" / "relatability_classifier.joblib").exists()
    assert not (tmp_path / "models" / "build_provenance.json").exists()

    before_vir_meta = (tmp_path / "models" / "virality_predictor_metadata.json").read_bytes()
    before_nlp_meta = (tmp_path / "models" / "relatability_metadata.json").read_bytes()

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)

    probe_script = "\n".join([
        "from pathlib import Path",
        "from streamlit.testing.v1 import AppTest",
        "from scripts.build_all_artifacts import verify_generated_artifacts",
        "root = Path.cwd()",
        "at = AppTest.from_file(str(root / 'streamlit_app.py'), default_timeout=60)",
        "at.run()",
        "assert len(at.exception) == 0, [e.message for e in at.exception]",
        "assert len(at.error) == 0, [e.value for e in at.error]",
        "at.run()",
        "assert len(at.exception) == 0, [e.message for e in at.exception]",
        "verify_generated_artifacts(root)",
        "views = ['streamlit_app.py', 'app/streamlit_app.py'] + sorted(str(p.relative_to(root)) for p in (root / 'app' / 'pages').glob('*.py') if p.name != '__init__.py')",
        "assert len(views) == 9",
        "for rel in views:",
        "    v = AppTest.from_file(str(root / rel), default_timeout=45)",
        "    v.run()",
        "    assert len(v.exception) == 0, f'{rel} exceptions: {[e.message for e in v.exception]}'",
        "    assert len(v.error) == 0, f'{rel} errors: {[e.value for e in v.error]}'",
    ])

    probe = subprocess.run(
        [sys.executable, "-B", "-c", probe_script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr or probe.stdout

    assert verify_generated_artifacts(tmp_path)["sqlite_latest_posts_rows"] == 120
    assert (tmp_path / "models" / "virality_predictor_metadata.json").read_bytes() == before_vir_meta
    assert (tmp_path / "models" / "relatability_metadata.json").read_bytes() == before_nlp_meta


def test_ensure_artifacts_ready_idempotent_without_rewrites(tmp_path: Path, monkeypatch):
    """Repeated ensure_artifacts_ready() calls must build once and never rewrite valid outputs."""
    from app.components import data_loader

    _copy_tracked_clean_checkout(tmp_path)
    build_calls = []
    real_build = data_loader.build_all_artifacts

    def counting_build(*args, **kwargs):
        build_calls.append(kwargs.get("root_dir"))
        return real_build(*args, **kwargs)

    monkeypatch.setattr(data_loader, "build_all_artifacts", counting_build)

    first = data_loader.ensure_artifacts_ready(tmp_path)
    assert first["built"] is True
    assert first["status"] == "built"
    assert len(build_calls) == 1

    tracked_files = [
        tmp_path / "models" / "virality_predictor_metadata.json",
        tmp_path / "models" / "relatability_metadata.json",
    ]
    before_meta_bytes = {p: p.read_bytes() for p in tracked_files}

    monitored_paths = [
        tmp_path / rel for rel in data_loader.REQUIRED_ARTIFACT_RELATIVE_PATHS
    ] + [tmp_path / "models" / "build_provenance.json"]
    before_stats = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in monitored_paths}

    second = data_loader.ensure_artifacts_ready(tmp_path)
    third = data_loader.ensure_artifacts_ready(tmp_path)

    assert second["built"] is False
    assert second["status"] == "ready"
    assert third["built"] is False
    assert third["status"] == "ready"
    assert len(build_calls) == 1

    for p, expected_bytes in before_meta_bytes.items():
        assert p.read_bytes() == expected_bytes
    for p, expected_stat in before_stats.items():
        assert (p.stat().st_mtime_ns, p.stat().st_size) == expected_stat


def test_ensure_artifacts_ready_detects_missing_corrupt_and_failed_builds(tmp_path: Path, monkeypatch):
    """Missing/corrupted generated artifacts must trigger rebuild; failed builds must not cache ready."""
    from app.components import data_loader

    _copy_tracked_clean_checkout(tmp_path)
    build_calls = []
    real_build = data_loader.build_all_artifacts

    def counting_build(*args, **kwargs):
        build_calls.append("build")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(data_loader, "build_all_artifacts", counting_build)

    assert data_loader.ensure_artifacts_ready(tmp_path)["built"] is True
    assert len(build_calls) == 1

    # 1. Missing required model file -> detects and rebuilds
    nlp_model = tmp_path / "models" / "relatability_classifier.joblib"
    nlp_model.unlink()
    rebuilt_missing = data_loader.ensure_artifacts_ready(tmp_path)
    assert rebuilt_missing["built"] is True
    assert len(build_calls) == 2
    assert nlp_model.exists() and nlp_model.stat().st_size > 0

    # 2. Invalid/corrupted Parquet schema -> detects and repairs cleanly
    cmt_pq = tmp_path / "data" / "processed" / "processed_comments.parquet"
    df_bad = pd.read_parquet(cmt_pq).drop(columns=["sentiment_score"])
    df_bad.to_parquet(cmt_pq, index=False)
    rebuilt_corrupt = data_loader.ensure_artifacts_ready(tmp_path)
    assert rebuilt_corrupt["built"] is True
    assert len(build_calls) == 3
    assert "sentiment_score" in pd.read_parquet(cmt_pq).columns

    # 3. Failed build -> raises without marking workspace ready on repeated calls
    nlp_model.unlink()

    def failing_build(*args, **kwargs):
        raise RuntimeError("simulated build crash")

    monkeypatch.setattr(data_loader, "build_all_artifacts", failing_build)
    with pytest.raises(RuntimeError, match="simulated build crash"):
        data_loader.ensure_artifacts_ready(tmp_path)
    with pytest.raises(RuntimeError, match="simulated build crash"):
        data_loader.ensure_artifacts_ready(tmp_path)
    assert not nlp_model.exists()

    # Restoring builder completes cleanly
    monkeypatch.setattr(data_loader, "build_all_artifacts", counting_build)
    recovered = data_loader.ensure_artifacts_ready(tmp_path)
    assert recovered["built"] is True
    assert len(build_calls) == 4
    verify_generated_artifacts(tmp_path)


def test_ensure_artifacts_ready_concurrent_callers_execute_single_build(tmp_path: Path, monkeypatch):
    """Concurrent calls to ensure_artifacts_ready() against one root must serialize to one build."""
    from app.components import data_loader

    _copy_tracked_clean_checkout(tmp_path)
    real_build = data_loader.build_all_artifacts

    state_guard = threading.Lock()
    state = {"active": 0, "max_active": 0, "total": 0}

    def synchronized_counting_build(*args, **kwargs):
        with state_guard:
            state["active"] += 1
            state["total"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            return real_build(*args, **kwargs)
        finally:
            with state_guard:
                state["active"] -= 1

    monkeypatch.setattr(data_loader, "build_all_artifacts", synchronized_counting_build)

    worker_count = 6
    barrier = threading.Barrier(worker_count)

    def _worker():
        barrier.wait(timeout=10)
        return data_loader.ensure_artifacts_ready(tmp_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_worker) for _ in range(worker_count)]
        results = [fut.result(timeout=60) for fut in futures]

    assert len(results) == worker_count
    assert state["max_active"] == 1
    assert state["total"] == 1
    assert sum(1 for r in results if r["built"]) == 1
    assert sum(1 for r in results if not r["built"]) == worker_count - 1
    assert verify_generated_artifacts(tmp_path)["sqlite_posts_rows"] == 121
    with closing(sqlite3.connect(tmp_path / "database" / "social_analytics.db")) as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]) == 121
        assert int(conn.execute("SELECT COUNT(*) FROM latest_posts").fetchone()[0]) == 120

