"""UI/UX, shared component compatibility, and SocialPulse shell verification tests (Phase 4)."""

from pathlib import Path
import inspect
import plotly.graph_objects as go
import plotly.io as pio
import pytest
from streamlit.testing.v1 import AppTest

from app.components.ui_widgets import (
    SOCIALPULSE_PALETTE,
    apply_plotly_theme,
    configure_plotly_theme,
    render_footer,
    render_header,
    render_metric_card,
    render_section_header,
    render_status_badge,
)


def test_ui_widget_backward_compatible_signatures():
    """Ensure shared UI helpers retain exact positional/keyword backward compatibility with locked Pages 01-06."""
    header_params = list(inspect.signature(render_header).parameters.values())
    assert [p.name for p in header_params] == ["title", "subtitle"]
    assert header_params[1].default == ""

    metric_params = list(inspect.signature(render_metric_card).parameters.values())
    assert [p.name for p in metric_params] == ["label", "value", "delta"]
    assert metric_params[2].default == ""

    assert "title" in inspect.signature(render_section_header).parameters
    assert "note" in inspect.signature(render_footer).parameters


def test_ui_widget_xss_and_badge_escaping():
    """Verify HTML special characters inside dynamic text do not inject raw HTML tags."""
    badge = render_status_badge("<script>alert(1)</script>", tone="warning")
    assert "<script>" not in badge
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in badge
    assert "sp-badge-warning" in badge


def test_plotly_socialpulse_theme_registration():
    """Verify SocialPulse Plotly template registers cleanly and matches design palette."""
    configure_plotly_theme()
    assert "socialpulse" in pio.templates
    assert pio.templates.default == "socialpulse"

    fig = apply_plotly_theme(go.Figure(go.Scatter(x=[1, 2], y=[3, 4])), height=360)
    assert fig.layout.height == 360
    assert fig.layout.paper_bgcolor == "#FFFFFF"
    assert fig.layout.plot_bgcolor == "#FFFFFF"
    assert list(fig.layout.template.layout.colorway) == SOCIALPULSE_PALETTE["series"]


def test_streamlit_config_toml_palette_consistency():
    """Ensure .streamlit/config.toml defines the SocialPulse light workspace + dark slate sidebar tokens."""
    root = Path(__file__).resolve().parent.parent
    config_file = root / ".streamlit" / "config.toml"
    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert 'base = "light"' in content
    assert 'primaryColor = "#0284C7"' in content
    assert 'backgroundColor = "#F8FAFC"' in content
    assert "[theme.sidebar]" in content
    assert 'backgroundColor = "#0F172A"' in content


def test_root_socialpulse_entrypoint_and_clean_navigation():
    """Verify streamlit_app.py renders without exceptions and excludes the legacy 06_Growth_Analytics route."""
    root = Path(__file__).resolve().parent.parent
    entrypoint_src = (root / "streamlit_app.py").read_text(encoding="utf-8")
    assert "06_Growth_Analytics.py" not in entrypoint_src
    for idx_file in [
        "01_Content_Performance.py",
        "02_Virality_Prediction.py",
        "03_Sentiment_Analysis.py",
        "04_AB_Testing.py",
        "05_Recommendations.py",
        "06_Growth_Dashboard.py",
        "07_Trend_Forecasting.py",
    ]:
        assert idx_file in entrypoint_src

    at = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=30)
    at.run()
    assert len(at.exception) == 0


@pytest.mark.parametrize(
    "page_path",
    [
        "app/streamlit_app.py",
        "app/pages/01_Content_Performance.py",
        "app/pages/02_Virality_Prediction.py",
        "app/pages/03_Sentiment_Analysis.py",
        "app/pages/04_AB_Testing.py",
        "app/pages/05_Recommendations.py",
        "app/pages/06_Growth_Dashboard.py",
        "app/pages/07_Trend_Forecasting.py",
    ],
)
def test_all_eight_views_render_without_exceptions(page_path: str):
    """Ensure Home and all 7 analytical modules render cleanly under the upgraded UI system."""
    root = Path(__file__).resolve().parent.parent
    at = AppTest.from_file(str(root / page_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"{page_path} raised exceptions: {at.exception}"
