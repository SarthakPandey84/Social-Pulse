"""Regression and parity tests for Phase 3 optimizations & cache invalidation safety."""

from pathlib import Path
import warnings
import pandas as pd

from src.data_ingestion.sample_ingestion import SampleDataIngestion
from src.nlp.text_preprocessor import TextPreprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_text_preprocessor_contractions_and_normalization_parity():
    """Verify compiled TextPreprocessor rules match expected contraction expansion and edge guards."""
    prep = TextPreprocessor()

    res = prep.preprocess("I don't think you're wrong!!! Can't wait???")
    assert res.is_valid is True
    assert res.cleaned_text == "I don't think you're wrong!! Can't wait??"
    assert "do not" in res.normalized_tokens
    assert "you are" in res.normalized_tokens
    assert "cannot" in res.normalized_tokens

    # Edge guards must still catch insufficient context and empty strings without regression
    assert prep.preprocess("ok").is_valid is False
    assert prep.preprocess("🔥🔥").is_valid is False
    assert prep.preprocess("   ").is_valid is False
    assert prep.preprocess(None).is_valid is False


def test_sample_ingestion_null_safe_impressions_parity(tmp_path):
    """Verify vectorized null-safe Int64 impressions calculation handles both valid integers and <NA>."""
    csv_file = tmp_path / "custom_posts.csv"
    pd.DataFrame([
        {
            "post_id": "P1",
            "timestamp": "2026-10-01T12:00:00Z",
            "content_type": "reel",
            "views": 100,
            "likes": 10,
            "comments_count": 2,
            "shares": 3,
            "saves": 4,
            "retention_rate": 0.6,
        },
        {
            "post_id": "P2",
            "timestamp": "2026-10-02T12:00:00Z",
            "content_type": "carousel",
            "views": pd.NA,
            "likes": 5,
            "comments_count": 1,
            "shares": 1,
            "saves": 2,
            "retention_rate": pd.NA,
        },
    ]).to_csv(csv_file, index=False)

    loader = SampleDataIngestion(sample_dir=tmp_path)
    df, _ = loader.load_sample_posts(filename="custom_posts.csv")
    assert str(df["impressions"].dtype) == "Int64"
    assert df.loc[0, "impressions"] == int(100 * 1.15)
    assert pd.isna(df.loc[1, "impressions"])


def test_page7_cache_invalidation_and_zero_deprecation_warnings():
    """Verify Page 7 responds accurately to parameter changes (no stale cache) and emits no use_container_width warnings."""
    from streamlit.testing.v1 import AppTest

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        at = AppTest.from_file(str(REPO_ROOT / "app" / "pages" / "07_Trend_Forecasting.py"), default_timeout=30)
        at.run()
        assert len(at.exception) == 0

        # Switch horizon slider 7 -> 10: forecast detail table must reflect 10 steps, not stale 7
        slider = next(s for s in at.sidebar.slider if "Horizon" in s.label)
        slider.set_value(10).run()
        assert len(at.exception) == 0
        forecast_df = next(d.value for d in at.dataframe if "Horizon Step" in d.value.columns)
        assert len(forecast_df) == 10

        # Switch model selector: Automatic -> Naive (Persistence) (distinct non-champion override)
        model_sb = next(sb for sb in at.sidebar.selectbox if "Model" in sb.label)
        model_sb.set_value("Naive (Persistence)").run()
        assert len(at.exception) == 0
        assert any("Naive (Persistence)" in w.value for w in at.warning)

        # Switch domain: Publication-Cohort Performance -> Struggle Keyword Trajectories
        domain_radio = next(r for r in at.sidebar.radio if "Domain" in r.label)
        domain_radio.set_value("Struggle Keyword Trajectories").run()
        assert len(at.exception) == 0
        assert any("Keyword Observed Calendar-Date Continuity" in elem.value for elem in at.info)

    ucw_warnings = [w for w in caught if "use_container_width" in str(w.message)]
    assert len(ucw_warnings) == 0, f"Unexpected use_container_width warnings: {ucw_warnings}"


def test_home_page_loads_cleanly_without_exceptions():
    """Verify cached Home entrypoint renders SocialPulse KPIs and 7 module cards without exceptions."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    rendered_html = "\n".join(m.value for m in at.markdown)
    assert "SocialPulse" in rendered_html
    assert "sp-module-card" in rendered_html
    assert rendered_html.count("MODULE 0") == 7
