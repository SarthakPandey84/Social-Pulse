"""Repository hygiene, structure, and safe-cleanup regression checks."""

from pathlib import Path
import re


ROOT_DIR = Path(__file__).resolve().parent.parent


EXPECTED_PAGES = [
    "01_Content_Performance.py",
    "02_Virality_Prediction.py",
    "03_Sentiment_Analysis.py",
    "04_AB_Testing.py",
    "05_Recommendations.py",
    "06_Growth_Dashboard.py",
    "07_Trend_Forecasting.py",
]

EXPECTED_NOTEBOOKS = [
    "01_eda_and_metrics.ipynb",
    "02_virality_modeling.ipynb",
    "03_nlp_sentiment_training.ipynb",
    "04_ab_testing_analysis.ipynb",
    "05_trend_forecasting.ipynb",
]


def test_expected_entrypoints_and_pages_present():
    assert (ROOT_DIR / "streamlit_app.py").is_file()
    assert (ROOT_DIR / "app" / "streamlit_app.py").is_file()
    pages_dir = ROOT_DIR / "app" / "pages"
    actual_pages = sorted(p.name for p in pages_dir.glob("*.py") if p.name != "__init__.py")
    assert actual_pages == EXPECTED_PAGES


def test_removed_obsolete_files_and_orphan_bytecode_absent():
    assert not (ROOT_DIR / "app" / "pages" / "06_Growth_Analytics.py").exists()
    assert not (ROOT_DIR / "scripts" / "audit_module3.py").exists()
    assert not (ROOT_DIR / "app" / "pages" / "__pycache__" / "06_Growth_Analytics.cpython-311.pyc").exists()
    assert not (ROOT_DIR / "scripts" / "__pycache__" / "audit_module3.cpython-311.pyc").exists()


def test_historical_academic_and_sample_assets_preserved():
    assert (ROOT_DIR / "Problem_Statement" / "Data_Science_Major_Project_December.pdf").is_file()
    for nb in EXPECTED_NOTEBOOKS:
        assert (ROOT_DIR / "notebooks" / nb).is_file(), f"Missing notebook: {nb}"
    for csv_name in [
        "sample_posts.csv",
        "sample_comments.csv",
        "sample_experiments.csv",
        "sample_trends.csv",
    ]:
        assert (ROOT_DIR / "data" / "sample" / csv_name).is_file()
    assert (ROOT_DIR / "utils" / "helpers.py").is_file()


def test_pages_have_no_redundant_import_fallbacks():
    for page_name in EXPECTED_PAGES:
        text = (ROOT_DIR / "app" / "pages" / page_name).read_text(encoding="utf-8")
        assert not re.search(r"from\s+components\.", text), f"Redundant fallback import in {page_name}"


def test_runtime_code_packages_have_clean_socialpulse_branding():
    legacy_brand = "un" + "lox"
    self_path = Path(__file__).resolve()
    for rel_dir in ["app", "src", "scripts", "utils", "tests"]:
        for py_file in (ROOT_DIR / rel_dir).rglob("*.py"):
            if "__pycache__" in py_file.parts or py_file.resolve() == self_path:
                continue
            text = py_file.read_text(encoding="utf-8")
            assert legacy_brand not in text.lower(), f"Stale branding reference in {py_file.relative_to(ROOT_DIR)}"
