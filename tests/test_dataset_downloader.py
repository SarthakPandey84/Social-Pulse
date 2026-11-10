"""Unit tests for the reproducible external dataset registry, integrity verifier, and downloader."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
import urllib.error
import zipfile

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_ingestion.external_datasets import (
    DATASET_REGISTRY,
    DatasetFileSpec,
    IntegrityVerificationError,
    KaggleCredentialsMissingError,
    LicenseReviewRequiredError,
    check_kaggle_credentials,
    download_dataset,
    download_file_resumable,
    download_kaggle_dataset,
    generate_raw_quality_report,
    get_dataset_spec,
    read_xlsx_sheet_to_dataframe,
    redact_secrets,
    verify_downloaded_file,
)
from src.data_ingestion.schemas import SourceProvenance


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


class _FakeResponse:
    def __init__(self, data: bytes, status: int = 200):
        self._buf = io.BytesIO(data)
        self.status = status

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_registry_covers_pinned_datasets_and_provenance_taxonomy():
    """Verify all expected dataset specs exist with pinned revisions, digests, and truthful provenance."""
    assert "tiktok_engagement_200k" in DATASET_REGISTRY
    assert "instagram_analytics_github_alt" in DATASET_REGISTRY
    assert "instagram_engagement_eda_hf" in DATASET_REGISTRY
    assert "instaart_human_ai" in DATASET_REGISTRY
    assert "instagram_comments_sentiment_hf" in DATASET_REGISTRY
    assert "social_instagram_marketing_hf" in DATASET_REGISTRY
    assert "zenodo_sponsored_unsponsored_ig" in DATASET_REGISTRY
    assert "zenodo_educational_reels_ig" in DATASET_REGISTRY
    assert "zenodo_lokananta_comments_ig" in DATASET_REGISTRY
    assert "youtube_trending_kaggle" in DATASET_REGISTRY

    # Pinned revisions / DOIs & large asset exclusions
    e_spec = get_dataset_spec("social_instagram_marketing_hf")
    assert e_spec.provenance == SourceProvenance.SYNTHETIC_DATA
    assert all(not f.local_filename.endswith(".png") for f in e_spec.files)

    b_spec = get_dataset_spec("instagram_engagement_eda_hf")
    assert b_spec.license_status == "PENDING_REVIEW"
    assert b_spec.provenance == SourceProvenance.EXTERNAL_BENCHMARK_UNVERIFIED
    assert all(not f.local_filename.endswith((".mp4", ".ipynb")) for f in b_spec.files)

    # Every downloadable file in non-Kaggle specs carries size + cryptographic checksum
    for ds_id, spec in DATASET_REGISTRY.items():
        if not spec.requires_kaggle_auth:
            assert len(spec.files) >= 1, ds_id
            for fspec in spec.files:
                assert fspec.expected_size and fspec.expected_size > 0
                assert any(
                    [fspec.expected_sha256, fspec.expected_md5, fspec.expected_git_sha1]
                ), f"Missing digest for {ds_id}/{fspec.local_filename}"


def test_resumable_download_retry_backoff_and_integrity_checks(tmp_path, monkeypatch):
    """Test transient retry, Range 206 resume from .part, SHA-256 verification, and tamper rejection."""
    import urllib.request

    payload = b"post_id,views,likes\np1,100,10\np2,250,25\n"
    sha256 = hashlib.sha256(payload).hexdigest()
    md5 = hashlib.md5(payload).hexdigest()
    gsha1 = _git_blob_sha1(payload)

    fspec = DatasetFileSpec(
        url="https://example.invalid/data.csv",
        local_filename="data.csv",
        expected_size=len(payload),
        expected_sha256=sha256,
        expected_md5=md5,
        expected_git_sha1=gsha1,
    )

    dest = tmp_path / "data.csv"
    part = tmp_path / "data.csv.part"
    split_at = 16
    part.write_bytes(payload[:split_at])

    calls = []
    sleeps = []

    def fake_urlopen(req, timeout=60):
        calls.append((req.full_url, req.headers.get("Range")))
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", None, None)
        assert req.headers.get("Range") == f"bytes={split_at}-"
        return _FakeResponse(payload[split_at:], status=206)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    res = download_file_resumable(
        fspec,
        dest,
        max_retries=2,
        backoff_base=0.5,
        sleep_fn=sleeps.append,
    )
    assert res["status"] == "downloaded_verified"
    assert res["resumed_from_bytes"] == split_at
    assert sleeps == [0.5]
    assert dest.read_bytes() == payload
    assert not part.exists()
    assert verify_downloaded_file(dest, fspec)["sha256"] == sha256

    # Cached fast-path check
    cached = download_file_resumable(fspec, dest)
    assert cached["status"] == "cached_verified"

    # Corrupt checksum must reject without promoting .part
    bad_spec = DatasetFileSpec(
        url="https://example.invalid/bad.csv",
        local_filename="bad.csv",
        expected_size=len(payload),
        expected_sha256="0" * 64,
    )
    bad_dest = tmp_path / "bad.csv"
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=60: _FakeResponse(payload, status=200))
    with pytest.raises(IntegrityVerificationError):
        download_file_resumable(bad_spec, bad_dest)
    assert not bad_dest.exists()
    assert not bad_dest.with_name("bad.csv.part").exists()


def test_license_pending_review_gate_and_kaggle_missing_creds_and_secret_redaction(tmp_path, monkeypatch):
    """Verify PENDING_REVIEW gating, clean Kaggle missing-credential failure, and secret redaction."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    with pytest.raises(LicenseReviewRequiredError):
        download_dataset("instagram_engagement_eda_hf", root_dir=tmp_path, include_pending_review=False)
    # Default-deny behavior: omitting include_pending_review must also block PENDING_REVIEW datasets
    with pytest.raises(LicenseReviewRequiredError):
        download_dataset("instagram_engagement_eda_hf", root_dir=tmp_path)
    with pytest.raises(LicenseReviewRequiredError):
        download_dataset("instagram_comments_sentiment_hf", root_dir=tmp_path)

    empty_kdir = tmp_path / "empty_kaggle"
    empty_kdir.mkdir()
    creds = check_kaggle_credentials(kaggle_dir=empty_kdir)
    assert creds["has_credentials"] is False

    yt_spec = get_dataset_spec("youtube_trending_kaggle")
    res = download_kaggle_dataset(yt_spec, tmp_path / "yt_raw", raise_on_missing_creds=False, kaggle_dir=empty_kdir)
    assert res["status"] == "BLOCKED_MISSING_CREDENTIALS"

    with pytest.raises(KaggleCredentialsMissingError):
        download_kaggle_dataset(yt_spec, tmp_path / "yt_raw", raise_on_missing_creds=True, kaggle_dir=empty_kdir)

    raw_msg = "Failed request with KAGGLE_KEY=super_secret_123 Bearer abc.def.ghi ?api_key=AIzaFakeKeyValue12345"
    safe = redact_secrets(raw_msg)
    assert "super_secret_123" not in safe
    assert "abc.def.ghi" not in safe
    assert "AIzaFakeKeyValue12345" not in safe
    assert "[REDACTED]" in safe


def test_stdlib_xlsx_reader_and_quality_report_generation(tmp_path):
    """Verify minimal stdlib XML .xlsx parser and raw quality/PII/leakage report writer."""
    # Build minimal valid OOXML workbook in temp directory
    xlsx_path = tmp_path / "mini_reels.xlsx"
    with zipfile.ZipFile(xlsx_path, "w") as zf:
        zf.writestr(
            "xl/sharedStrings.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="3" uniqueCount="3">'
            "<si><t>post_id</t></si><si><t>views</t></si><si><t>r_1</t></si></sst>",
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<sheetData>"
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>420</v></c></row>'
            "</sheetData></worksheet>",
        )

    parsed_df = read_xlsx_sheet_to_dataframe(xlsx_path)
    assert list(parsed_df.columns) == ["post_id", "views"]
    assert parsed_df.loc[0, "post_id"] == "r_1"
    assert int(parsed_df.loc[0, "views"]) == 420

    # Quality report test with PII and leakage columns on disk
    raw_dir = tmp_path / "data" / "external" / "instagram_comments_sentiment_hf" / "raw"
    raw_dir.mkdir(parents=True)
    pd.DataFrame([
        {"user_id": "123", "username": "@alice", "comment": "Love @bob mail me a@b.com", "sentiment": "pos", "performance_bucket_label": "high"}
    ]).to_csv(raw_dir / "train.csv", index=False)

    report = generate_raw_quality_report("instagram_comments_sentiment_hf", root_dir=tmp_path)
    assert len(report["tables"]) == 1
    tbl = report["tables"][0]
    assert "user_id" in tbl["pii_audit"]["flagged_identifier_columns"]
    assert "username" in tbl["pii_audit"]["flagged_identifier_columns"]
    assert tbl["pii_audit"]["sample_pattern_counts"]["handles"] >= 1
    assert "performance_bucket_label" in tbl["leakage_audit"]["flagged_columns"]
    assert (tmp_path / "data" / "external" / "reports" / "instagram_comments_sentiment_hf_quality.md").exists()

    # Verify nested ndarray columns (such as TikTok hashtag arrays) do not crash duplicate checks
    import numpy as np
    from src.data_ingestion.external_datasets import inspect_dataframe_quality
    arr_df = pd.DataFrame({
        "video_id": ["v1", "v1", "v2"],
        "hashtags": [np.array(["a", "b"]), np.array(["a", "b"]), np.array(["c"])],
    })
    arr_report = inspect_dataframe_quality(arr_df, "videos.parquet", get_dataset_spec("tiktok_engagement_200k"))
    assert arr_report["nested_array_columns"] == ["hashtags"]
    assert arr_report["duplicate_rows"] == 1


def test_download_all_default_denies_pending_review_and_allows_explicit_opt_in(tmp_path, monkeypatch):
    """Verify PENDING_REVIEW datasets are skipped by default in API & CLI unless explicitly opted in."""
    from scripts import download_external_datasets as dl_cli
    from src.data_ingestion import external_datasets as ed

    downloaded_ids = []

    def fake_download_file(fspec, target, **kwargs):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"col\n1\n")
        return {
            "filename": fspec.local_filename,
            "path": str(target),
            "url": fspec.url,
            "status": "downloaded_verified",
            "size": 6,
            "sha256": "0" * 64,
            "md5": "0" * 32,
            "git_sha1": "0" * 40,
        }

    monkeypatch.setattr(ed, "download_file_resumable", fake_download_file)
    monkeypatch.setattr(
        ed,
        "generate_raw_quality_report",
        lambda ds_id, root_dir=None: {"dataset_id": ds_id, "tables": []},
    )

    def _track_download_dataset(dataset_id, **kwargs):
        downloaded_ids.append(dataset_id)
        return ed.download_dataset.__wrapped__(dataset_id, **kwargs) if hasattr(ed.download_dataset, "__wrapped__") else _orig_dl(dataset_id, **kwargs)

    _orig_dl = ed.download_dataset
    # 1. Default API call skips PENDING_REVIEW without blocking VERIFIED / NON_COMMERCIAL / RESEARCH_ONLY
    default_results = ed.download_all_accessible(root_dir=tmp_path, attempt_kaggle=False)
    for pending_id in ("instagram_engagement_eda_hf", "instagram_comments_sentiment_hf"):
        assert default_results[pending_id]["status"] == "SKIPPED_PENDING_REVIEW"
        assert default_results[pending_id]["requires_explicit_opt_in"] is True

    # Approved public + non-commercial + research-only datasets remain enabled and labeled
    assert default_results["tiktok_engagement_200k"]["status"] == "downloaded_verified"
    assert default_results["tiktok_engagement_200k"]["governance_tier"] == "NON_COMMERCIAL_RESEARCH_ONLY"
    assert default_results["instaart_human_ai"]["status"] == "downloaded_verified"
    assert default_results["instaart_human_ai"]["governance_tier"] == "NON_COMMERCIAL_RESEARCH_ONLY"
    assert default_results["zenodo_sponsored_unsponsored_ig"]["status"] == "downloaded_verified"
    assert default_results["zenodo_sponsored_unsponsored_ig"]["governance_tier"] == "VERIFIED_COMMERCIAL_SAFE"

    # 2. CLI default excludes PENDING_REVIEW; --include-pending-review enables explicit opt-in
    cli_dir = tmp_path / "cli_run"
    assert dl_cli.main(["--all-accessible", "--skip-kaggle", "--root-dir", str(cli_dir)]) == 0
    assert not (cli_dir / "data" / "external" / "instagram_engagement_eda_hf").exists()
    assert not (cli_dir / "data" / "external" / "instagram_comments_sentiment_hf").exists()
    assert (cli_dir / "data" / "external" / "zenodo_lokananta_comments_ig").exists()

    assert dl_cli.main(["--all-accessible", "--skip-kaggle", "--include-pending-review", "--root-dir", str(cli_dir)]) == 0
    assert (cli_dir / "data" / "external" / "instagram_engagement_eda_hf" / "raw").exists()
    assert (cli_dir / "data" / "external" / "instagram_comments_sentiment_hf" / "raw").exists()
