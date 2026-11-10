"""UI/UX, shared component compatibility, and SocialPulse shell verification tests (Phase 4)."""

from pathlib import Path
import inspect
import warnings
import plotly.graph_objects as go
import plotly.io as pio
import pytest
import streamlit.deprecation_util as st_deprecation_util
from streamlit.testing.v1 import AppTest

from app.components.ui_widgets import (
    SOCIALPULSE_PALETTE,
    apply_plotly_theme,
    build_ab_experiment_banner_html,
    build_ab_verdict_card_html,
    build_dashboard_experiment_card_html,
    build_dashboard_recommendation_card_html,
    build_nlp_diagnostic_card_html,
    build_recommendation_detail_card_html,
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
    """Verify streamlit_app.py renders without exceptions and the redundant 06_Growth_Analytics redirect is removed."""
    root = Path(__file__).resolve().parent.parent
    assert not (root / "app" / "pages" / "06_Growth_Analytics.py").exists()
    assert not (root / "config.yaml").exists()
    assert not (root / "scripts" / "audit_module3.py").exists()

    import utils
    from utils import get_project_root as root_fn_pkg
    from utils.helpers import get_project_root as root_fn_mod

    assert utils.__all__ == ["get_project_root"]
    assert not hasattr(utils, "load_config")
    assert root_fn_pkg is root_fn_mod
    assert root_fn_pkg() == root

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
def test_all_eight_views_render_without_exceptions(page_path: str, monkeypatch):
    """Ensure Home and all 7 analytical modules render cleanly with zero deprecation/syntax warnings."""
    root = Path(__file__).resolve().parent.parent
    st_deprecations = []
    orig_show_dep = st_deprecation_util.show_deprecation_warning

    def _spy_deprecation(message, *args, **kwargs):
        st_deprecations.append(str(message))
        return orig_show_dep(message, *args, **kwargs)

    monkeypatch.setattr(st_deprecation_util, "show_deprecation_warning", _spy_deprecation)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        at = AppTest.from_file(str(root / page_path), default_timeout=30)
        at.run()

    assert len(at.exception) == 0, f"{page_path} raised exceptions: {at.exception}"
    ucw_msgs = [m for m in st_deprecations if "use_container_width" in m]
    assert len(ucw_msgs) == 0, f"{page_path} emitted use_container_width deprecations: {ucw_msgs}"
    escape_warnings = [
        str(w.message) for w in caught if "invalid escape sequence" in str(w.message)
    ]
    assert len(escape_warnings) == 0, f"{page_path} emitted invalid escape warnings: {escape_warnings}"


XSS_PAYLOAD = '<script>alert("xss")</script><img src=x onerror=\'boom\'> & <b>bold</b>'


def test_dynamic_page_html_builders_escape_untrusted_payloads_and_keep_structure():
    """All dynamic HTML card builders must escape untrusted text while keeping structural HTML intact."""
    expected_escaped = "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;&lt;img src=x onerror=&#x27;boom&#x27;&gt; &amp; &lt;b&gt;bold&lt;/b&gt;"

    nlp_html = build_nlp_diagnostic_card_html(
        eyebrow=XSS_PAYLOAD,
        headline=XSS_PAYLOAD,
        metric_label="Detected Lang:",
        metric_value=XSS_PAYLOAD,
        accent_color="red; onclick=alert(1)",
        footnote=XSS_PAYLOAD,
    )
    assert "<script>" not in nlp_html
    assert "<img " not in nlp_html
    assert "onclick=" not in nlp_html
    assert "#64748B" in nlp_html  # invalid CSS color rejected to safe default
    assert expected_escaped in nlp_html
    assert '<div style="background-color: #F8FAFC;' in nlp_html

    banner_html = build_ab_experiment_banner_html(
        exp_id=XSS_PAYLOAD,
        exp_type_badge="Demo",
        hypothesis=XSS_PAYLOAD,
        platform=XSS_PAYLOAD,
        variant_a_label=XSS_PAYLOAD,
        variant_b_label=XSS_PAYLOAD,
    )
    assert "<script>" not in banner_html
    assert "<img " not in banner_html
    assert expected_escaped in banner_html
    assert "EXPERIMENT REGISTRATION:" in banner_html

    verdict_html = build_ab_verdict_card_html(
        badge_color="javascript:alert(1)",
        alpha_val=0.05,
        verdict=XSS_PAYLOAD,
        primary_method=XSS_PAYLOAD,
        test_statistic=1.2345,
        effect_size_type=XSS_PAYLOAD,
        effect_size=0.5,
        ci_low=0.1,
        ci_high=0.9,
        ci_method=XSS_PAYLOAD,
        unit_of_analysis=XSS_PAYLOAD,
        cluster_badge=XSS_PAYLOAD,
    )
    assert "<script>" not in verdict_html
    assert "javascript:" not in verdict_html
    assert expected_escaped in verdict_html

    rec_detail_html = build_recommendation_detail_card_html(
        idx=1,
        confidence_level="HIGH",
        is_synthetic_demonstration=True,
        title=XSS_PAYLOAD,
        category=XSS_PAYLOAD,
        platform_label=XSS_PAYLOAD,
        source_label=XSS_PAYLOAD,
        priority_score=0.8123,
        suggested_action=XSS_PAYLOAD,
        reasoning=XSS_PAYLOAD,
        supporting_evidence=XSS_PAYLOAD,
        sample_size=100,
        unit_of_analysis=XSS_PAYLOAD,
        adj_p_str="1.00e-03",
        ci_str="[0.10, 0.20]",
        expected_impact_direction=XSS_PAYLOAD,
        limitations_and_caveats=XSS_PAYLOAD,
        suggested_validation_method=XSS_PAYLOAD,
    )
    assert "<script>" not in rec_detail_html
    assert "<img " not in rec_detail_html
    assert expected_escaped in rec_detail_html
    assert "SYNTHETIC DEMO" in rec_detail_html
    assert "HIGH EVIDENCE (DEMO RCT)" in rec_detail_html

    dash_exp_html = build_dashboard_experiment_card_html(
        is_randomized=True,
        is_significant=True,
        experiment_id=XSS_PAYLOAD,
        name=XSS_PAYLOAD,
        platform=XSS_PAYLOAD,
        sample_size=60,
        unit_of_analysis=XSS_PAYLOAD,
        test_name=XSS_PAYLOAD,
        variant_a=XSS_PAYLOAD,
        value_a=0.1,
        variant_b=XSS_PAYLOAD,
        value_b=0.2,
        relative_lift_pct=100.0,
        effect_size=0.5,
        effect_size_type=XSS_PAYLOAD,
        raw_p_value=0.001,
        holm_adj_p_value=0.002,
        ci_low=0.01,
        ci_high=0.19,
        ci_method=XSS_PAYLOAD,
    )
    assert "<script>" not in dash_exp_html
    assert expected_escaped in dash_exp_html
    assert "RANDOMIZED RCT (STAT. SIG)" in dash_exp_html

    dash_rec_html = build_dashboard_recommendation_card_html(
        idx=1,
        confidence_level="LOW",
        title=XSS_PAYLOAD,
        priority_score=0.55,
        suggested_action=XSS_PAYLOAD,
        expected_impact_direction=XSS_PAYLOAD,
        suggested_validation_method=XSS_PAYLOAD,
    )
    assert "<script>" not in dash_rec_html
    assert expected_escaped in dash_rec_html


def test_pages_03_to_06_escape_malicious_payloads_end_to_end(monkeypatch):
    """Render Pages 03, 04, 05, and 06 through AppTest with injected XSS payloads to verify safe DOM HTML."""
    from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend
    from src.nlp.relatability_classifier import RelatabilityClassifier
    from src.nlp.sentiment_analyzer import SentimentAnalyzer
    from src.recommender.engine import OptimizationRecommender
    from src.recommender.schemas import (
        ConfidenceLevel,
        ContentRecommendation,
        EvidenceSource,
        RecommendationCategory,
        RecommendationReport,
    )
    from src.statistics.ab_testing import ABTestingFramework, ComparisonResult

    root = Path(__file__).resolve().parent.parent
    escaped_prefix = "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;"

    # Page 03: live diagnostic cards
    monkeypatch.setattr(
        RelatabilityClassifier,
        "classify_comment",
        lambda self, text, language_hint=None: {
            "raw_text": str(text),
            "cleaned_text": str(text),
            "language": XSS_PAYLOAD,
            "relatability_label": XSS_PAYLOAD,
            "relatability_confidence": 0.75,
            "classification_status": XSS_PAYLOAD,
            "rejection_reason": None,
        },
    )
    monkeypatch.setattr(
        SentimentAnalyzer,
        "analyze_text",
        lambda self, text, language_hint=None: {
            "raw_text": str(text),
            "cleaned_text": str(text),
            "polarity": 0.42,
            "subjectivity": 0.5,
            "sentiment_label": XSS_PAYLOAD,
            "sentiment_confidence": 0.80,
            "classification_status": "success",
            "language": XSS_PAYLOAD,
            "rejection_reason": None,
        },
    )
    at3 = AppTest.from_file(str(root / "app" / "pages" / "03_Sentiment_Analysis.py"), default_timeout=30)
    at3.run()
    assert len(at3.exception) == 0
    md3_all = "\n".join(m.value for m in at3.markdown if m.proto.allow_html)
    assert "<script>" not in md3_all
    assert "<img " not in md3_all
    assert escaped_prefix in md3_all

    # Page 04: statistical verdict card
    xss_ab_result = ComparisonResult(
        metric_name="share_rate",
        metric_type="continuous",
        variant_a_label=XSS_PAYLOAD,
        variant_b_label=XSS_PAYLOAD,
        sample_size_a=30,
        sample_size_b=30,
        value_a=0.03,
        value_b=0.05,
        absolute_difference=0.02,
        relative_lift_pct=66.7,
        test_statistic=2.5,
        test_name=XSS_PAYLOAD,
        p_value=0.015,
        confidence_interval=(0.005, 0.035),
        ci_level=0.95,
        effect_size=0.65,
        effect_size_type=XSS_PAYLOAD,
        is_statistically_significant=True,
        verdict=XSS_PAYLOAD,
        primary_method=XSS_PAYLOAD,
        ci_method=XSS_PAYLOAD,
        unit_of_analysis=XSS_PAYLOAD,
        clustering_handled=True,
        mann_whitney_p_value=0.018,
    )
    monkeypatch.setattr(
        ABTestingFramework,
        "compare_post_level_rates",
        lambda self, **kwargs: xss_ab_result,
    )
    monkeypatch.setattr(
        ABTestingFramework,
        "compare_continuous",
        lambda self, *args, **kwargs: xss_ab_result,
    )
    at4 = AppTest.from_file(str(root / "app" / "pages" / "04_AB_Testing.py"), default_timeout=30)
    at4.run()
    assert len(at4.exception) == 0
    md4_all = "\n".join(m.value for m in at4.markdown if m.proto.allow_html)
    assert "<script>" not in md4_all
    assert "<img " not in md4_all
    assert escaped_prefix in md4_all

    # Page 05 & Page 06: recommendation and experiment cards
    malicious_rec = ContentRecommendation(
        recommendation_id="REC_XSS",
        title=XSS_PAYLOAD,
        category=RecommendationCategory.CONTENT_FORMAT_STRUCTURE,
        suggested_action=XSS_PAYLOAD,
        reasoning=XSS_PAYLOAD,
        supporting_evidence=XSS_PAYLOAD,
        evidence_source=EvidenceSource.AB_TESTING,
        relevant_platform="instagram",
        expected_impact_direction=XSS_PAYLOAD,
        priority_score=0.9,
        confidence_level=ConfidenceLevel.HIGH,
        is_synthetic_demonstration=True,
        sample_size=60,
        unit_of_analysis=XSS_PAYLOAD,
        p_value_adjusted=0.001,
        confidence_interval=(0.01, 0.05),
        limitations_and_caveats=XSS_PAYLOAD,
        suggested_validation_method=XSS_PAYLOAD,
        scoring_breakdown={
            "evidence_score": 1.0,
            "impact_score": 0.9,
            "sample_score": 0.8,
            "relevance_score": 1.0,
            "recency_score": 1.0,
            "quality_score": 1.0,
            "conflict_penalty": 1.0,
        },
    )
    fake_report = RecommendationReport(
        total_recommendations=1,
        recommendations=[malicious_rec],
        high_confidence_count=1,
        medium_confidence_count=0,
        low_confidence_count=0,
        conflicting_signals=[],
        insufficient_data_warnings=[],
        generated_at="2026-11-10T00:00:00Z",
    )

    monkeypatch.setattr(
        OptimizationRecommender,
        "generate_recommendations",
        lambda self, **kwargs: fake_report,
    )

    at5 = AppTest.from_file(str(root / "app" / "pages" / "05_Recommendations.py"), default_timeout=30)
    at5.run()
    assert len(at5.exception) == 0
    md5_all = "\n".join(m.value for m in at5.markdown if m.proto.allow_html)
    assert "<script>" not in md5_all
    assert "<img " not in md5_all
    assert escaped_prefix in md5_all

    import pandas as pd

    monkeypatch.setattr(
        GrowthDashboardBackend,
        "calculate_recommendation_priorities",
        lambda self: fake_report,
    )
    monkeypatch.setattr(
        GrowthDashboardBackend,
        "calculate_experiment_performance",
        lambda self: {
            "has_data": True,
            "forest_plot_data": pd.DataFrame(),
            "experiments": [
                {
                    "experiment_id": XSS_PAYLOAD,
                    "name": XSS_PAYLOAD,
                    "platform": XSS_PAYLOAD,
                    "is_randomized": True,
                    "is_significant": True,
                    "sample_size": 80,
                    "unit_of_analysis": XSS_PAYLOAD,
                    "test_name": XSS_PAYLOAD,
                    "variant_a": XSS_PAYLOAD,
                    "value_a": 0.03,
                    "variant_b": XSS_PAYLOAD,
                    "value_b": 0.06,
                    "relative_lift_pct": 100.0,
                    "effect_size": 0.4,
                    "effect_size_type": XSS_PAYLOAD,
                    "raw_p_value": 0.001,
                    "holm_adj_p_value": 0.002,
                    "confidence_interval": (0.01, 0.05),
                    "ci_method": XSS_PAYLOAD,
                }
            ],
        },
    )

    at6 = AppTest.from_file(str(root / "app" / "pages" / "06_Growth_Dashboard.py"), default_timeout=30)
    at6.run()
    assert len(at6.exception) == 0
    md6_all = "\n".join(m.value for m in at6.markdown if m.proto.allow_html)
    assert "<script>" not in md6_all
    assert "<img " not in md6_all
    assert escaped_prefix in md6_all


def test_build_dashboard_experiment_card_html_handles_missing_and_non_finite_adj_p_value(monkeypatch):
    """Missing, non-numeric, bool, or non-finite holm_adj_p_value must render safely without float(None)."""
    import pandas as pd
    from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend
    from src.recommender.schemas import RecommendationReport

    base_kwargs = dict(
        is_randomized=True,
        is_significant=False,
        experiment_id="EXP_EDGE",
        name="Edge Adjusted P Test",
        platform="Instagram",
        sample_size=40,
        unit_of_analysis="Post",
        test_name="Welch t-test",
        variant_a="Control",
        value_a=0.10,
        variant_b="Treatment",
        value_b=0.12,
        relative_lift_pct=20.0,
        effect_size=0.15,
        effect_size_type="Hedges g",
        raw_p_value=0.045,
        ci_low=-0.02,
        ci_high=0.06,
        ci_method="Welch",
    )

    valid_html = build_dashboard_experiment_card_html(holm_adj_p_value=0.002, **base_kwargs)
    assert "Holm-Bonferroni Adj. p = 2.00e-03" in valid_html

    for invalid_val in [None, "not-a-number", '<script>alert("x")</script>', True, False, float("nan"), float("inf"), float("-inf"), object()]:
        safe_html = build_dashboard_experiment_card_html(holm_adj_p_value=invalid_val, **base_kwargs)
        assert "Holm-Bonferroni Adj. p = N/A (not calculated)" in safe_html
        assert "<script>" not in safe_html

    # End-to-end check on Page 06 when backend returns holm_adj_p_value=None
    root = Path(__file__).resolve().parent.parent
    empty_recs = RecommendationReport(
        total_recommendations=0,
        recommendations=[],
        high_confidence_count=0,
        medium_confidence_count=0,
        low_confidence_count=0,
        conflicting_signals=[],
        insufficient_data_warnings=[],
        generated_at="2026-11-10T00:00:00Z",
    )
    monkeypatch.setattr(GrowthDashboardBackend, "calculate_recommendation_priorities", lambda self: empty_recs)
    monkeypatch.setattr(
        GrowthDashboardBackend,
        "calculate_experiment_performance",
        lambda self: {
            "has_data": True,
            "forest_plot_data": pd.DataFrame(),
            "experiments": [
                {
                    **base_kwargs,
                    "holm_adj_p_value": None,
                    "confidence_interval": (-0.02, 0.06),
                }
            ],
        },
    )
    at6 = AppTest.from_file(str(root / "app" / "pages" / "06_Growth_Dashboard.py"), default_timeout=30)
    at6.run()
    assert len(at6.exception) == 0
    md6_html = "\n".join(m.value for m in at6.markdown if m.proto.allow_html)
    assert "Holm-Bonferroni Adj. p = N/A (not calculated)" in md6_html


def test_page3_unavailable_confidence_does_not_claim_model_probability(monkeypatch):
    """Page 03 interactive tester must never claim 'Model probability (uncalibrated)' when confidence is unavailable."""
    from src.nlp.relatability_classifier import RelatabilityClassifier
    from src.nlp.sentiment_analyzer import SentimentAnalyzer

    root = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(
        SentimentAnalyzer,
        "analyze_text",
        lambda self, text, language_hint=None: {
            "raw_text": str(text),
            "cleaned_text": str(text),
            "polarity": None,
            "subjectivity": None,
            "sentiment_label": None,
            "sentiment_confidence": None,
            "classification_status": "inference_failed",
            "language": "en",
            "rejection_reason": None,
        },
    )

    scenarios = [
        (
            {"relatability_label": None, "relatability_confidence": None, "classification_status": "inference_failed", "language": "en", "rejection_reason": "Backend error"},
            ["Inference Failed", "N/A", "Unavailable — statistical inference failed"],
            ["Model probability (uncalibrated)", "Heuristic rule weight"],
        ),
        (
            {"relatability_label": None, "relatability_confidence": None, "classification_status": "unsupported_language", "language": "fr", "rejection_reason": "Non-English"},
            ["Unavailable", "N/A", "Confidence unavailable (not scored)"],
            ["Model probability (uncalibrated)", "Heuristic rule weight"],
        ),
        (
            {"relatability_label": "Neutral", "relatability_confidence": 0.812, "classification_status": "success", "language": "en", "rejection_reason": None},
            ["Neutral", "81.2%", "Model probability (uncalibrated)"],
            ["Inference Failed", "Confidence unavailable"],
        ),
        (
            {"relatability_label": "Relatable", "relatability_confidence": 0.95, "classification_status": "success", "language": "en", "rejection_reason": None},
            ["Relatable", "95.0%", "Heuristic rule weight"],
            ["Model probability (uncalibrated)", "Inference Failed"],
        ),
    ]

    for diag_payload, must_contain, must_exclude in scenarios:
        monkeypatch.setattr(
            RelatabilityClassifier,
            "classify_comment",
            lambda self, text, language_hint=None, payload=diag_payload: {
                "raw_text": str(text),
                "cleaned_text": str(text),
                **payload,
            },
        )
        at3 = AppTest.from_file(str(root / "app" / "pages" / "03_Sentiment_Analysis.py"), default_timeout=30)
        at3.run()
        assert len(at3.exception) == 0
        # Isolate the RELATABILITY DIMENSION diagnostic card HTML specifically
        rel_cards = [m.value for m in at3.markdown if m.proto.allow_html and "RELATABILITY DIMENSION" in m.value]
        assert len(rel_cards) == 1
        rel_card_html = rel_cards[0]
        for expected in must_contain:
            assert expected in rel_card_html
        for forbidden in must_exclude:
            assert forbidden not in rel_card_html


def test_page3_counts_and_status_filter_handle_inference_failures_and_empty_text(monkeypatch):
    """Page 03 KPI counts, denominators, and Tab 3 filter must expose inference_failed/empty_text accurately without double counting."""
    import re
    import pandas as pd

    root = Path(__file__).resolve().parent.parent

    # 1. Verify existing processed dataset KPI parity first (500 total -> 203 Relatable, 239 Neutral, 58 Excluded, 45.9%)
    at3_prod = AppTest.from_file(str(root / "app" / "pages" / "03_Sentiment_Analysis.py"), default_timeout=30)
    at3_prod.run()
    assert len(at3_prod.exception) == 0
    prod_kpis = {}
    for m in at3_prod.markdown:
        if m.proto.allow_html and "sp-kpi-card" in m.value:
            lbl = re.search(r'<div class="sp-kpi-label">(.*?)</div>', m.value)
            val = re.search(r'<div class="sp-kpi-value">(.*?)</div>', m.value)
            if lbl and val:
                prod_kpis[lbl.group(1)] = val.group(1)
    assert prod_kpis["Total Ingested"] == "500"
    assert prod_kpis["Relatable Comments"] == "203"
    assert prod_kpis["Neutral Comments"] == "239"
    assert prod_kpis["Relatability Ratio"] == "45.9%"

    # 2. Inject controlled dataset containing inference_failed (including leaked labels) + empty_text + unsupported + uncertain
    synthetic_df = pd.DataFrame([
        {"comment_id": "C_OK_1", "post_id": "P1", "platform": "instagram", "comment_text": "Literally me", "language": "en", "relatability_label": "Relatable", "relatability_confidence": 0.95, "sentiment_label": "Positive", "sentiment_score": 0.5, "classification_status": "success"},
        {"comment_id": "C_OK_2", "post_id": "P1", "platform": "instagram", "comment_text": "Felt this deep", "language": "en", "relatability_label": "Relatable", "relatability_confidence": 0.85, "sentiment_label": "Negative", "sentiment_score": -0.5, "classification_status": "success"},
        {"comment_id": "C_OK_3", "post_id": "P2", "platform": "youtube", "comment_text": "Camera info", "language": "en", "relatability_label": "Neutral", "relatability_confidence": 0.92, "sentiment_label": "Neutral", "sentiment_score": 0.0, "classification_status": "success"},
        {"comment_id": "C_UNC_1", "post_id": "P2", "platform": "youtube", "comment_text": "Maybe idk", "language": "en", "relatability_label": "Uncertain", "relatability_confidence": 0.51, "sentiment_label": "Neutral", "sentiment_score": 0.0, "classification_status": "uncertain"},
        {"comment_id": "C_LANG_1", "post_id": "P3", "platform": "instagram", "comment_text": "Cest vrai", "language": "non-en", "relatability_label": None, "relatability_confidence": None, "sentiment_label": None, "sentiment_score": None, "classification_status": "unsupported_language"},
        {"comment_id": "C_INSUF_1", "post_id": "P3", "platform": "youtube", "comment_text": "lol", "language": "en", "relatability_label": None, "relatability_confidence": None, "sentiment_label": "Positive", "sentiment_score": 0.2, "classification_status": "insufficient_context"},
        {"comment_id": "C_EMPTY_1", "post_id": "P3", "platform": "youtube", "comment_text": "", "language": "unknown", "relatability_label": None, "relatability_confidence": None, "sentiment_label": None, "sentiment_score": None, "classification_status": "empty_text"},
        # Failed rows with intentionally polluted labels must still never enter successful counts or ratio denominator
        {"comment_id": "C_FAIL_1", "post_id": "P4", "platform": "instagram", "comment_text": "Error sample 1", "language": "en", "relatability_label": "Neutral", "relatability_confidence": None, "sentiment_label": "Neutral", "sentiment_score": 0.0, "classification_status": "inference_failed"},
        {"comment_id": "C_FAIL_2", "post_id": "P4", "platform": "youtube", "comment_text": "Error sample 2", "language": "en", "relatability_label": "Relatable", "relatability_confidence": None, "sentiment_label": "Neutral", "sentiment_score": 0.0, "classification_status": "inference_failed"},
    ])

    orig_read_parquet = pd.read_parquet

    def _patched_read_parquet(path, *args, **kwargs):
        if Path(path).name == "processed_comments.parquet":
            return synthetic_df.copy()
        return orig_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _patched_read_parquet)

    at3 = AppTest.from_file(str(root / "app" / "pages" / "03_Sentiment_Analysis.py"), default_timeout=30)
    at3.run()
    assert len(at3.exception) == 0

    kpis = {}
    for m in at3.markdown:
        if m.proto.allow_html and "sp-kpi-card" in m.value:
            lbl = re.search(r'<div class="sp-kpi-label">(.*?)</div>', m.value)
            val = re.search(r'<div class="sp-kpi-value">(.*?)</div>', m.value)
            if lbl and val:
                kpis[lbl.group(1)] = val.group(1)

    assert kpis["Total Ingested"] == "9"
    assert kpis["Relatable Comments"] == "2"
    assert kpis["Neutral Comments"] == "1"
    assert kpis["Uncertain / Unsupported / Failed"] == "6"
    # Ratio = 2 / 3 valid comments = 66.7% (NOT 2/9 and NOT 3/5)
    assert kpis["Relatability Ratio"] == "66.7%"

    caption_texts = "\n".join(c.value for c in at3.caption)
    assert "N = 3" in caption_texts
    assert "excluding 3 unsupported/insufficient/empty comments, 2 inference failures, and 1 ambiguous cases" in caption_texts

    status_boxes = [sb for sb in at3.selectbox if sb.label == "Status Filter"]
    assert len(status_boxes) == 1
    status_box = status_boxes[0]
    assert "inference_failed" in status_box.options
    assert "empty_text" in status_box.options

    status_box.set_value("inference_failed").run()
    assert len(at3.exception) == 0
    assert any("Showing **2** matching comments:" in m.value for m in at3.markdown)
    filtered_df_out = at3.dataframe[0].value
    assert set(filtered_df_out["comment_id"].tolist()) == {"C_FAIL_1", "C_FAIL_2"}


# =========================================================================
# BATCH 4: SHARED DATA LOADER & RESOURCE CACHING TESTS
# =========================================================================


def test_shared_loader_csv_and_parquet_caching_invalidation_and_isolation(tmp_path, monkeypatch):
    """CSV and Parquet loaders must hit cache on unchanged files, invalidate on disk updates, and isolate DataFrame copies."""
    import os
    import pandas as pd
    from app.components.data_loader import (
        file_signature,
        load_csv_table,
        load_parquet_table,
    )

    # Missing file signature must not crash and loaders return None cleanly
    missing_path = tmp_path / "does_not_exist.csv"
    assert file_signature(missing_path)[1] is False
    assert load_csv_table(missing_path) is None
    assert load_parquet_table(tmp_path / "does_not_exist.parquet") is None

    csv_file = tmp_path / "sample.csv"
    pq_file = tmp_path / "sample.parquet"
    pd.DataFrame({"post_id": ["P1", "P2"], "views": [100, 200]}).to_csv(csv_file, index=False)
    pd.DataFrame({"post_id": ["P1", "P2"], "views": [100, 200]}).to_parquet(pq_file, index=False)

    csv_calls = {"count": 0}
    pq_calls = {"count": 0}
    orig_read_csv = pd.read_csv
    orig_read_parquet = pd.read_parquet

    def _spy_csv(path, *args, **kwargs):
        if Path(path).resolve() == csv_file.resolve():
            csv_calls["count"] += 1
        return orig_read_csv(path, *args, **kwargs)

    def _spy_parquet(path, *args, **kwargs):
        if Path(path).resolve() == pq_file.resolve():
            pq_calls["count"] += 1
        return orig_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _spy_csv)
    monkeypatch.setattr(pd, "read_parquet", _spy_parquet)

    # 1. Cold load -> cache hit on repeated call
    df_csv_1 = load_csv_table(csv_file)
    df_csv_2 = load_csv_table(csv_file)
    assert csv_calls["count"] == 1
    assert len(df_csv_1) == 2 and len(df_csv_2) == 2

    df_pq_1 = load_parquet_table(pq_file)
    df_pq_2 = load_parquet_table(pq_file)
    assert pq_calls["count"] == 1
    assert len(df_pq_1) == 2 and len(df_pq_2) == 2

    # 2. DataFrame copy isolation across retrievals
    df_csv_1["views"] = -999
    df_csv_1["injected_col"] = True
    df_csv_3 = load_csv_table(csv_file)
    assert csv_calls["count"] == 1
    assert "injected_col" not in df_csv_3.columns
    assert df_csv_3["views"].tolist() == [100, 200]

    df_pq_1["views"] = -888
    df_pq_3 = load_parquet_table(pq_file)
    assert pq_calls["count"] == 1
    assert df_pq_3["views"].tolist() == [100, 200]

    # 3. File update on disk -> automatic invalidation and reload
    pd.DataFrame({"post_id": ["P1", "P2", "P3"], "views": [100, 200, 300]}).to_csv(csv_file, index=False)
    st_csv = csv_file.stat()
    os.utime(csv_file, ns=(st_csv.st_atime_ns, st_csv.st_mtime_ns + 1_000_000))
    df_csv_updated = load_csv_table(csv_file)
    assert csv_calls["count"] == 2
    assert len(df_csv_updated) == 3

    pd.DataFrame({"post_id": ["P1", "P2", "P3"], "views": [100, 200, 300]}).to_parquet(pq_file, index=False)
    st_pq = pq_file.stat()
    os.utime(pq_file, ns=(st_pq.st_atime_ns, st_pq.st_mtime_ns + 1_000_000))
    df_pq_updated = load_parquet_table(pq_file)
    assert pq_calls["count"] == 2
    assert len(df_pq_updated) == 3


def test_shared_loader_json_caching_and_invalidation(tmp_path):
    """JSON metadata loader must cache unchanged files and reload cleanly after updates."""
    import json
    import os
    from app.components.data_loader import load_json_artifact

    meta_file = tmp_path / "meta.json"
    assert load_json_artifact(meta_file) == {}

    meta_file.write_text(json.dumps({"version": 1, "accuracy": 0.91}), encoding="utf-8")
    first = load_json_artifact(meta_file)
    second = load_json_artifact(meta_file)
    assert first == {"version": 1, "accuracy": 0.91}
    assert second == first

    meta_file.write_text(json.dumps({"version": 2, "accuracy": 0.94, "note": "updated"}), encoding="utf-8")
    st_meta = meta_file.stat()
    os.utime(meta_file, ns=(st_meta.st_atime_ns, st_meta.st_mtime_ns + 1_000_000))
    updated = load_json_artifact(meta_file)
    assert updated == {"version": 2, "accuracy": 0.94, "note": "updated"}


def test_shared_resource_loaders_reuse_invalidate_and_stay_read_only(tmp_path, monkeypatch):
    """Cached ML/backend resources must reuse across unchanged signatures, invalidate on artifact updates, and remain read-only."""
    import os
    import shutil
    import pandas as pd
    from app.components.data_loader import (
        get_dashboard_experiment_performance,
        get_dashboard_recommendation_priorities,
        get_growth_dashboard_backend,
        get_optimization_recommender,
        get_relatability_classifier,
        get_trend_forecaster,
        get_virality_predictor,
        load_ab_observations_dataset,
        load_comments_dataset,
        load_experiments_dataset,
        load_posts_dataset,
    )
    from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend

    root = Path(__file__).resolve().parent.parent

    # 1. Reuse across unchanged artifact signatures + verify read-only query safety
    vp1 = get_virality_predictor(root_dir=root)
    vp2 = get_virality_predictor(root_dir=root)
    assert vp1 is vp2
    pred_before = vp1.predict_pre_publication(pd.DataFrame([{
        "platform": "instagram",
        "content_type": "Reel",
        "topic": "Social Anxiety",
        "hook_type": "Visual Hook",
        "duration_seconds": 30,
        "posting_hour": 18,
        "posting_day_of_week": "Sunday",
        "follower_count": 35000,
    }]))
    pred_after = vp2.predict_pre_publication(pd.DataFrame([{
        "platform": "instagram",
        "content_type": "Reel",
        "topic": "Social Anxiety",
        "hook_type": "Visual Hook",
        "duration_seconds": 30,
        "posting_hour": 18,
        "posting_day_of_week": "Sunday",
        "follower_count": 35000,
    }]))
    assert pred_before.equals(pred_after)

    rc1 = get_relatability_classifier(root_dir=root)
    rc2 = get_relatability_classifier(root_dir=root)
    assert rc1 is rc2
    c_res1 = rc1.classify_comment("This hits way too close to home.")
    c_res2 = rc2.classify_comment("This hits way too close to home.")
    assert c_res1 == c_res2

    tf1 = get_trend_forecaster(rolling_window=7)
    tf2 = get_trend_forecaster(rolling_window=7)
    assert tf1 is tf2

    posts_df, _ = load_posts_dataset(root_dir=root)
    comments_df, _ = load_comments_dataset(root_dir=root)
    exp_df = load_experiments_dataset(root_dir=root)
    obs_df = load_ab_observations_dataset(root_dir=root)

    rec1 = get_optimization_recommender(
        posts_df=posts_df,
        comments_df=comments_df,
        experiments_df=exp_df,
        observations_df=obs_df,
        root_dir=root,
    )
    rec2 = get_optimization_recommender(
        posts_df=posts_df,
        comments_df=comments_df,
        experiments_df=exp_df,
        observations_df=obs_df,
        root_dir=root,
    )
    assert rec1 is rec2
    posts_cols_before = list(rec1.posts_df.columns)
    rep_a = rec1.generate_recommendations(platform="instagram", max_recommendations=5)
    rep_b = rec2.generate_recommendations(platform="youtube", max_recommendations=5)
    assert list(rec1.posts_df.columns) == posts_cols_before
    assert rep_a.total_recommendations >= 0 and rep_b.total_recommendations >= 0

    # 2. Invalidation when underlying source artifacts change on disk
    isolated_root = tmp_path / "workspace"
    (isolated_root / "data" / "processed").mkdir(parents=True)
    (isolated_root / "data" / "sample").mkdir(parents=True)
    (isolated_root / "models").mkdir(parents=True)

    for rel in [
        "data/processed/processed_posts.parquet",
        "data/processed/processed_comments.parquet",
        "data/processed/processed_ab_tests.parquet",
        "data/sample/sample_posts.csv",
        "data/sample/sample_experiments.csv",
        "models/relatability_classifier.joblib",
    ]:
        shutil.copy2(root / rel, isolated_root / rel)

    iso_clf_1 = get_relatability_classifier(root_dir=isolated_root)
    iso_clf_2 = get_relatability_classifier(root_dir=isolated_root)
    assert iso_clf_1 is iso_clf_2

    iso_model = isolated_root / "models" / "relatability_classifier.joblib"
    st_m = iso_model.stat()
    os.utime(iso_model, ns=(st_m.st_atime_ns, st_m.st_mtime_ns + 2_000_000))
    iso_clf_3 = get_relatability_classifier(root_dir=isolated_root)
    assert iso_clf_3 is not iso_clf_1

    backend_1 = get_growth_dashboard_backend(root_dir=isolated_root)
    backend_2 = get_growth_dashboard_backend(root_dir=isolated_root)
    assert backend_1 is backend_2

    # Monkeypatching backend synthesis methods must bypass cached synthesis output
    real_exp = get_dashboard_experiment_performance(backend_1)
    assert len(real_exp["experiments"]) >= 1
    monkeypatch.setattr(
        GrowthDashboardBackend,
        "calculate_experiment_performance",
        lambda self: {"experiments": [], "forest_plot_data": pd.DataFrame()},
    )
    patched_exp = get_dashboard_experiment_performance(backend_1)
    assert len(patched_exp["experiments"]) == 0

    from dataclasses import replace

    real_recs = get_dashboard_recommendation_priorities(backend_1)
    assert real_recs.total_recommendations >= 1
    monkeypatch.setattr(
        GrowthDashboardBackend,
        "calculate_recommendation_priorities",
        lambda self: replace(real_recs, total_recommendations=0, recommendations=[]),
    )
    patched_recs = get_dashboard_recommendation_priorities(backend_1)
    assert patched_recs.total_recommendations == 0

    # Updating any source dataset invalidates GrowthDashboardBackend
    iso_exp_csv = isolated_root / "data" / "sample" / "sample_experiments.csv"
    st_e = iso_exp_csv.stat()
    os.utime(iso_exp_csv, ns=(st_e.st_atime_ns, st_e.st_mtime_ns + 2_000_000))
    backend_3 = get_growth_dashboard_backend(root_dir=isolated_root)
    assert backend_3 is not backend_1

    # Sequential distinct backend instances must never collide even if memory addresses are recycled
    empty_workspace = tmp_path / "empty_workspace"
    empty_workspace.mkdir()
    del backend_1, backend_2, backend_3
    empty_backend = GrowthDashboardBackend(root_dir=empty_workspace)
    assert get_dashboard_experiment_performance(empty_backend)["experiments"] == []
    assert get_dashboard_recommendation_priorities(empty_backend).total_recommendations == 0


def test_page06_synthesis_cache_key_isolates_workspaces_and_invalidates_cleanly(tmp_path, monkeypatch):
    """Page 06 synthesis caches must key on workspace/dataset/method signatures rather than id(backend)."""
    from dataclasses import replace
    import os
    import shutil
    import pandas as pd
    from app.components.data_loader import (
        _dashboard_backend_signature,
        get_dashboard_experiment_performance,
        get_dashboard_recommendation_priorities,
    )
    from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend

    root = Path(__file__).resolve().parent.parent

    def _make_workspace(name: str, subset_exp_id: str) -> Path:
        ws = tmp_path / name
        (ws / "data" / "processed").mkdir(parents=True)
        (ws / "data" / "sample").mkdir(parents=True)
        for rel in [
            "data/processed/processed_posts.parquet",
            "data/processed/processed_comments.parquet",
            "data/processed/processed_ab_tests.parquet",
            "data/sample/sample_posts.csv",
            "data/sample/sample_experiments.csv",
        ]:
            shutil.copy2(root / rel, ws / rel)
        exp_df = pd.read_csv(ws / "data" / "sample" / "sample_experiments.csv")
        exp_df[exp_df["experiment_id"] == subset_exp_id].to_csv(
            ws / "data" / "sample" / "sample_experiments.csv", index=False
        )
        return ws

    ws_a = _make_workspace("workspace_a", "EXP_001")
    ws_b = _make_workspace("workspace_b", "EXP_002")

    backend_a = GrowthDashboardBackend(root_dir=ws_a)
    exp_a = get_dashboard_experiment_performance(backend_a)
    recs_a = get_dashboard_recommendation_priorities(backend_a)
    assert [e["experiment_id"] for e in exp_a["experiments"]] == ["EXP_001"]
    assert recs_a.total_recommendations >= 1

    # Drop backend_a before creating backend_b so CPython may reuse memory addresses
    sig_a = _dashboard_backend_signature(backend_a)
    del backend_a

    backend_b = GrowthDashboardBackend(root_dir=ws_b)
    sig_b = _dashboard_backend_signature(backend_b)
    assert sig_a != sig_b

    exp_b = get_dashboard_experiment_performance(backend_b)
    recs_b = get_dashboard_recommendation_priorities(backend_b)
    assert [e["experiment_id"] for e in exp_b["experiments"]] == ["EXP_002"]
    assert recs_b.total_recommendations >= 1
    assert [r.title for r in recs_b.recommendations] != [r.title for r in recs_a.recommendations]

    # Equal-length in-memory DataFrames (when backing files are absent) must also produce distinct signatures
    mem_ws = tmp_path / "mem_workspace"
    mem_ws.mkdir()
    mem_backend_1 = GrowthDashboardBackend(root_dir=mem_ws)
    mem_backend_1.experiments_df = pd.DataFrame([{"experiment_id": "EXP_A", "val": 1}])
    mem_backend_2 = GrowthDashboardBackend(root_dir=mem_ws)
    mem_backend_2.experiments_df = pd.DataFrame([{"experiment_id": "EXP_B", "val": 2}])
    assert _dashboard_backend_signature(mem_backend_1) != _dashboard_backend_signature(mem_backend_2)

    # Updating a source dataset on disk in workspace_b invalidates cached synthesis results
    exp_csv_b = ws_b / "data" / "sample" / "sample_experiments.csv"
    full_exp_df = pd.read_csv(root / "data" / "sample" / "sample_experiments.csv")
    full_exp_df.iloc[:2].to_csv(exp_csv_b, index=False)
    st_b = exp_csv_b.stat()
    os.utime(exp_csv_b, ns=(st_b.st_atime_ns, st_b.st_mtime_ns + 2_000_000))

    backend_b_reloaded = GrowthDashboardBackend(root_dir=ws_b)
    exp_b_updated = get_dashboard_experiment_performance(backend_b_reloaded)
    assert len(exp_b_updated["experiments"]) == 2

    # Both instance-level and class-level method monkeypatching must invalidate synthesis caches
    monkeypatch.setattr(
        backend_b_reloaded,
        "calculate_experiment_performance",
        lambda: {"experiments": [{"experiment_id": "INSTANCE_PATCH"}], "forest_plot_data": pd.DataFrame()},
    )
    assert [e["experiment_id"] for e in get_dashboard_experiment_performance(backend_b_reloaded)["experiments"]] == ["INSTANCE_PATCH"]

    monkeypatch.setattr(
        GrowthDashboardBackend,
        "calculate_recommendation_priorities",
        lambda self: replace(recs_b, total_recommendations=0, recommendations=[]),
    )
    assert get_dashboard_recommendation_priorities(backend_b_reloaded).total_recommendations == 0


def test_virality_predictor_fallback_invalidates_when_fallback_data_changes(tmp_path):
    """When serialized model artifacts are absent, changing fallback posts data must train a fresh predictor."""
    import os
    import pandas as pd
    from app.components.data_loader import get_virality_predictor, load_posts_dataset

    root = Path(__file__).resolve().parent.parent
    base_posts = pd.read_parquet(root / "data" / "processed" / "processed_posts.parquet")

    missing_model = tmp_path / "absent_models" / "virality_predictor.joblib"
    missing_meta = tmp_path / "absent_models" / "virality_predictor_metadata.json"
    fallback_pq = tmp_path / "fallback_posts.parquet"
    base_posts.to_parquet(fallback_pq, index=False)

    df_fb_1, _ = load_posts_dataset(fallback_pq, sample_path=None)
    pred_1 = get_virality_predictor(
        model_path=missing_model,
        meta_path=missing_meta,
        fallback_posts_df=df_fb_1,
        fallback_posts_path=fallback_pq,
        save_if_trained=False,
    )
    # Same absent model files + unchanged fallback data -> reuses cached fallback predictor
    pred_2 = get_virality_predictor(
        model_path=missing_model,
        meta_path=missing_meta,
        fallback_posts_df=df_fb_1,
        fallback_posts_path=fallback_pq,
        save_if_trained=False,
    )
    assert pred_1 is pred_2
    assert not missing_model.exists()

    # Mutate fallback dataset on disk while model files remain absent -> must retrain fresh predictor
    modified_posts = base_posts.copy()
    modified_posts["views"] = modified_posts["views"] * 2
    modified_posts.to_parquet(fallback_pq, index=False)
    st_fb = fallback_pq.stat()
    os.utime(fallback_pq, ns=(st_fb.st_atime_ns, st_fb.st_mtime_ns + 2_000_000))

    df_fb_2, _ = load_posts_dataset(fallback_pq, sample_path=None)
    pred_3 = get_virality_predictor(
        model_path=missing_model,
        meta_path=missing_meta,
        fallback_posts_df=df_fb_2,
        fallback_posts_path=fallback_pq,
        save_if_trained=False,
    )
    assert pred_3 is not pred_1
    assert not missing_model.exists()


def test_home_and_pages_01_to_07_smoke_and_rerun_parity():
    """Home and Pages 01–07 must render cleanly and preserve exact outputs across widget reruns."""
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parent.parent

    # Home view
    at_home = AppTest.from_file(str(root / "app" / "streamlit_app.py"), default_timeout=30)
    at_home.run()
    assert len(at_home.exception) == 0

    # Page 01 + widget rerun
    at1 = AppTest.from_file(str(root / "app" / "pages" / "01_Content_Performance.py"), default_timeout=30)
    at1.run()
    assert len(at1.exception) == 0
    if at1.selectbox:
        opts = at1.selectbox[0].options
        if len(opts) > 1:
            at1.selectbox[0].set_value(opts[1]).run()
            assert len(at1.exception) == 0

    # Page 02
    at2 = AppTest.from_file(str(root / "app" / "pages" / "02_Virality_Prediction.py"), default_timeout=30)
    at2.run()
    assert len(at2.exception) == 0

    # Page 04
    at4 = AppTest.from_file(str(root / "app" / "pages" / "04_AB_Testing.py"), default_timeout=30)
    at4.run()
    assert len(at4.exception) == 0

    # Page 05 + filter rerun
    at5 = AppTest.from_file(str(root / "app" / "pages" / "05_Recommendations.py"), default_timeout=30)
    at5.run()
    assert len(at5.exception) == 0
    if at5.selectbox:
        opts5 = at5.selectbox[0].options
        if len(opts5) > 1:
            at5.selectbox[0].set_value(opts5[1]).run()
            assert len(at5.exception) == 0

    # Page 06 + sidebar platform filter rerun
    at6 = AppTest.from_file(str(root / "app" / "pages" / "06_Growth_Dashboard.py"), default_timeout=30)
    at6.run()
    assert len(at6.exception) == 0
    if at6.sidebar.selectbox:
        plat_opts = at6.sidebar.selectbox[0].options
        if len(plat_opts) > 1:
            at6.sidebar.selectbox[0].set_value(plat_opts[1]).run()
            assert len(at6.exception) == 0

    # Page 07 + horizon rerun parity
    at7 = AppTest.from_file(str(root / "app" / "pages" / "07_Trend_Forecasting.py"), default_timeout=30)
    at7.run()
    assert len(at7.exception) == 0
    at7.run()
    assert len(at7.exception) == 0


def test_root_entrypoint_clean_checkout_bootstrap_failure_surfaces_error_and_stops(tmp_path):
    """When source inputs are missing cleanly, streamlit_app.py must surface error/exception and stop pre-navigation."""
    import os
    import shutil
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    for rel_dir in ["src", "scripts", "app", "utils", "assets", ".streamlit"]:
        shutil.copytree(
            root / rel_dir,
            tmp_path / rel_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    shutil.copy2(root / "streamlit_app.py", tmp_path / "streamlit_app.py")
    (tmp_path / "data" / "sample").mkdir(parents=True, exist_ok=True)
    for csv_path in (root / "data" / "sample").glob("*.csv"):
        if csv_path.name != "sample_comments.csv":
            shutil.copy2(csv_path, tmp_path / "data" / "sample" / csv_path.name)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)

    fail_script = "\n".join([
        "from pathlib import Path",
        "from streamlit.testing.v1 import AppTest",
        "root = Path.cwd()",
        "at = AppTest.from_file(str(root / 'streamlit_app.py'), default_timeout=30)",
        "at.run()",
        "assert len(at.error) >= 1",
        "assert 'sample_comments.csv' in at.error[0].value",
        "assert len(at.exception) >= 1",
        "assert 'sample_comments.csv' in str(at.exception[0].message)",
        "rendered_html = '\\n'.join(m.value for m in at.markdown)",
        "assert 'MODULE 0' not in rendered_html",
        "assert 'Benchmark Dataset Overview' not in rendered_html",
        "assert not any('SocialPulse v1.0' in c.value for c in at.sidebar.caption)",
    ])

    probe = subprocess.run(
        [sys.executable, "-B", "-c", fail_script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr or probe.stdout
    assert not (tmp_path / "data" / "processed").exists()
    assert not (tmp_path / "database" / "social_analytics.db").exists()
    assert not (tmp_path / "models").exists()


def test_clean_checkout_bootstrapped_via_root_serves_all_nine_views_and_avoids_page2_fallback(tmp_path):
    """After clean-checkout bootstrap via streamlit_app.py, ViralityPredictor loads serialized model and all 9 views run cleanly."""
    import os
    import shutil
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    for rel_dir in ["src", "scripts", "app", "utils", "assets", ".streamlit"]:
        shutil.copytree(
            root / rel_dir,
            tmp_path / rel_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    shutil.copy2(root / "streamlit_app.py", tmp_path / "streamlit_app.py")
    (tmp_path / "data" / "sample").mkdir(parents=True, exist_ok=True)
    for csv_path in (root / "data" / "sample").glob("*.csv"):
        shutil.copy2(csv_path, tmp_path / "data" / "sample" / csv_path.name)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    for meta_path in (root / "models").glob("*_metadata.json"):
        shutil.copy2(meta_path, tmp_path / "models" / meta_path.name)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)

    smoke_script = "\n".join([
        "from pathlib import Path",
        "from streamlit.testing.v1 import AppTest",
        "root = Path.cwd()",
        "at_root = AppTest.from_file(str(root / 'streamlit_app.py'), default_timeout=60)",
        "at_root.run()",
        "assert len(at_root.exception) == 0, [e.message for e in at_root.exception]",
        "assert len(at_root.error) == 0, [e.value for e in at_root.error]",
        "vir_model = root / 'models' / 'virality_predictor.joblib'",
        "vir_meta = root / 'models' / 'virality_predictor_metadata.json'",
        "assert vir_model.exists() and vir_meta.exists()",
        "mtime_before = (vir_model.stat().st_mtime_ns, vir_meta.read_bytes())",
        "at_p2 = AppTest.from_file(str(root / 'app' / 'pages' / '02_Virality_Prediction.py'), default_timeout=30)",
        "at_p2.run()",
        "assert len(at_p2.exception) == 0, [e.message for e in at_p2.exception]",
        "assert (vir_model.stat().st_mtime_ns, vir_meta.read_bytes()) == mtime_before",
        "views = [",
        "    'app/streamlit_app.py',",
        "    'app/pages/01_Content_Performance.py',",
        "    'app/pages/03_Sentiment_Analysis.py',",
        "    'app/pages/04_AB_Testing.py',",
        "    'app/pages/05_Recommendations.py',",
        "    'app/pages/06_Growth_Dashboard.py',",
        "    'app/pages/07_Trend_Forecasting.py',",
        "]",
        "for rel in views:",
        "    v = AppTest.from_file(str(root / rel), default_timeout=30)",
        "    v.run()",
        "    assert len(v.exception) == 0, f'{rel}: {[e.message for e in v.exception]}'",
        "    assert len(v.error) == 0, f'{rel} errors: {[e.value for e in v.error]}'",
        "    assert all('not found' not in w.value.lower() and 'sample offline benchmark' not in w.value.lower() for w in v.warning), f'{rel} unexpected warning: {[w.value for w in v.warning]}'",
    ])

    probe = subprocess.run(
        [sys.executable, "-B", "-c", smoke_script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr or probe.stdout


def test_data_loader_in_memory_override_and_fallback_lock_regressions(tmp_path, monkeypatch):
    """Verify cache-key safety for in-memory DF overrides and lock-guarded deterministic fallback saves."""
    import json
    import shutil
    import pandas as pd
    import app.components.data_loader as dl
    from app.components.data_loader import (
        _dashboard_backend_signature,
        get_dashboard_experiment_performance,
        get_optimization_recommender,
        get_virality_predictor,
    )
    from src.dashboard.growth_dashboard_backend import GrowthDashboardBackend

    root = Path(__file__).resolve().parent.parent

    # 1. Passing distinct in-memory DataFrames to get_optimization_recommender while disk artifacts exist
    p_df = pd.read_parquet(root / "data" / "processed" / "processed_posts.parquet")
    c_df = pd.read_parquet(root / "data" / "processed" / "processed_comments.parquet")
    e_df = pd.read_csv(root / "data" / "sample" / "sample_experiments.csv")
    o_df = pd.read_parquet(root / "data" / "processed" / "processed_ab_tests.parquet")

    e_one = e_df[e_df["experiment_id"] == "EXP_001"].copy()
    e_two = e_df[e_df["experiment_id"] == "EXP_002"].copy()
    rec_1 = get_optimization_recommender(
        posts_df=p_df,
        comments_df=c_df,
        experiments_df=e_one,
        observations_df=o_df,
        root_dir=root,
    )
    rec_2 = get_optimization_recommender(
        posts_df=p_df,
        comments_df=c_df,
        experiments_df=e_two,
        observations_df=o_df,
        root_dir=root,
    )
    assert rec_1 is not rec_2
    assert set(rec_1.experiments_df["experiment_id"]) == {"EXP_001"}
    assert set(rec_2.experiments_df["experiment_id"]) == {"EXP_002"}

    # 2. Mutating backend.experiments_df in-memory when disk backing files exist must invalidate Page 06 cache
    ws = tmp_path / "disk_ws"
    (ws / "data" / "processed").mkdir(parents=True)
    (ws / "data" / "sample").mkdir(parents=True)
    for rel in [
        "data/processed/processed_posts.parquet",
        "data/processed/processed_comments.parquet",
        "data/processed/processed_ab_tests.parquet",
        "data/sample/sample_experiments.csv",
    ]:
        shutil.copy2(root / rel, ws / rel)

    backend = GrowthDashboardBackend(root_dir=ws)
    sig_before = _dashboard_backend_signature(backend)
    perf_before = get_dashboard_experiment_performance(backend)
    assert len(perf_before["experiments"]) >= 2

    backend.experiments_df = backend.experiments_df[backend.experiments_df["experiment_id"] == "EXP_001"].copy()
    sig_after = _dashboard_backend_signature(backend)
    assert sig_before != sig_after
    perf_after = get_dashboard_experiment_performance(backend)
    assert [e["experiment_id"] for e in perf_after["experiments"]] == ["EXP_001"]

    # 3. ViralityPredictor fallback save must acquire build lock and omit volatile timestamps
    lock_events = []
    orig_get_build_lock = dl.get_build_lock

    class _RecordingLock:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            lock_events.append("enter")
            return self._inner.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            lock_events.append("exit")
            return self._inner.__exit__(exc_type, exc_val, exc_tb)

    monkeypatch.setattr(dl, "get_build_lock", lambda r: _RecordingLock(orig_get_build_lock(r)))

    fb_ws = tmp_path / "fb_ws"
    m_path = fb_ws / "models" / "virality_predictor.joblib"
    j_path = fb_ws / "models" / "virality_predictor_metadata.json"
    _ = get_virality_predictor(
        model_path=m_path,
        meta_path=j_path,
        fallback_posts_df=p_df,
        save_if_trained=True,
    )
    assert lock_events == ["enter", "exit"]
    assert m_path.exists() and j_path.exists()
    saved_meta = json.loads(j_path.read_text(encoding="utf-8"))
    assert "trained_at" not in saved_meta
    assert saved_meta["metadata_reproducibility"]["artifact_type"] == "deterministic_model_metadata"


def test_page02_dynamic_residual_band_caption():
    """Verify Page 02 uncertainty caption displays dynamic empirical_residual_q95 from metadata."""
    import json
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parent.parent
    meta = json.loads((root / "models" / "virality_predictor_metadata.json").read_text(encoding="utf-8"))
    expected_q95 = f"±{meta['empirical_residual_q95']:.2f} points"

    at = AppTest.from_file(str(root / "app" / "pages" / "02_Virality_Prediction.py"), default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    assert len(at.button) >= 1
    at.button[0].click().run()
    assert len(at.exception) == 0
    captions = "\n".join(c.value for c in at.caption)
    assert "±50.2 points" not in captions
    assert expected_q95 in captions


def test_external_dataset_switcher_0_1_multiple_and_fail_loud_corruption(tmp_path):
    """Verify UI dataset discovery/switching across 0, 1, multiple datasets and fail-loud handling on corrupt inputs."""
    import json
    import shutil
    import pandas as pd
    import pytest
    from streamlit.testing.v1 import AppTest
    from app.components.data_loader import (
        DatasetLoadError,
        SYNTHETIC_DEMO_DATASET_ID,
        list_available_datasets,
        load_selected_posts_dataset,
    )

    root = Path(__file__).resolve().parent.parent

    # --- 1. Zero external datasets workspace ---
    ws0 = tmp_path / "ws_zero_ext"
    (ws0 / "data" / "sample").mkdir(parents=True)
    shutil.copy2(root / "data" / "sample" / "sample_posts.csv", ws0 / "data" / "sample" / "sample_posts.csv")
    found0 = list_available_datasets(root_dir=ws0)
    assert [d["dataset_id"] for d in found0] == [SYNTHETIC_DEMO_DATASET_ID]

    # --- 2. One external dataset workspace ---
    ws1 = tmp_path / "ws_one_ext"
    (ws1 / "data" / "sample").mkdir(parents=True)
    shutil.copy2(root / "data" / "sample" / "sample_posts.csv", ws1 / "data" / "sample" / "sample_posts.csv")
    alpha_dir = ws1 / "data" / "processed" / "external" / "alpha_corpus"
    alpha_dir.mkdir(parents=True)
    pd.DataFrame([
        {
            "post_id": "A1",
            "platform": "tiktok",
            "timestamp": "2026-09-01T10:00:00Z",
            "content_type": "Short Video",
            "topic": "Focus",
            "views": 1000,
            "likes": 100,
            "comments_count": 10,
            "shares": 5,
            "saves": pd.NA,
            "collect_count": 8,
            "provenance": "real_public_data",
        }
    ]).to_parquet(alpha_dir / "posts.parquet", index=False)
    (alpha_dir / "dataset_meta.json").write_text(
        json.dumps({
            "dataset_id": "alpha_corpus",
            "display_name": "Alpha Corpus",
            "provenance": "real_public_data",
            "platforms": ["tiktok"],
            "tables": ["posts"],
            "row_counts": {"posts": 1},
        }),
        encoding="utf-8",
    )
    found1 = list_available_datasets(root_dir=ws1)
    assert [d["dataset_id"] for d in found1] == [SYNTHETIC_DEMO_DATASET_ID, "alpha_corpus"]
    df_alpha, meta_alpha = load_selected_posts_dataset("alpha_corpus", root_dir=ws1)
    assert len(df_alpha) == 1
    assert meta_alpha["dataset_id"] == "alpha_corpus"

    # --- 3. Corrupt parquet & unknown dataset_id fail loudly without falling back ---
    bad_dir = ws1 / "data" / "processed" / "external" / "broken_corpus"
    bad_dir.mkdir(parents=True)
    (bad_dir / "posts.parquet").write_bytes(b"NOT_A_VALID_PARQUET_HEADER")
    (bad_dir / "dataset_meta.json").write_text(
        json.dumps({
            "dataset_id": "broken_corpus",
            "display_name": "Broken Corpus",
            "provenance": "real_public_data",
            "platforms": ["tiktok"],
            "tables": ["posts"],
            "row_counts": {"posts": 5},
        }),
        encoding="utf-8",
    )
    with pytest.raises(DatasetLoadError):
        load_selected_posts_dataset("broken_corpus", root_dir=ws1)
    with pytest.raises(DatasetLoadError):
        load_selected_posts_dataset("totally_unknown_dataset_id", root_dir=ws1)

    # Corrupt dataset selected in Home UI and subpages must surface st.error and halt without falling back to synthetic KPIs
    for bad_id in ("totally_unknown_dataset_id", "synthetic_demo/../etc"):
        at_bad = AppTest.from_file(str(root / "app" / "streamlit_app.py"), default_timeout=30)
        at_bad.session_state["socialpulse_dataset_id"] = bad_id
        at_bad.run()
        assert len(at_bad.exception) == 0
        assert len(at_bad.error) >= 1
        rendered_html = "\n".join(m.value for m in at_bad.markdown)
        assert "Benchmark Dataset Overview" not in rendered_html

    for rel_sub in (
        "app/pages/01_Content_Performance.py",
        "app/pages/02_Virality_Prediction.py",
        "app/pages/03_Sentiment_Analysis.py",
        "app/pages/04_AB_Testing.py",
        "app/pages/05_Recommendations.py",
        "app/pages/06_Growth_Dashboard.py",
        "app/pages/07_Trend_Forecasting.py",
    ):
        at_sub_bad = AppTest.from_file(str(root / rel_sub), default_timeout=30)
        at_sub_bad.session_state["socialpulse_dataset_id"] = "totally_unknown_dataset_id"
        at_sub_bad.run()
        assert len(at_sub_bad.exception) == 0, f"{rel_sub} threw uncaught exception on unknown dataset"
        assert len(at_sub_bad.error) >= 1, f"{rel_sub} failed to display st.error on unknown dataset"

    # --- 4. Multiple prepared external corpora on UI views if present in repo workspace ---
    ext_root = root / "data" / "processed" / "external"
    if (ext_root / "tiktok_engagement_200k" / "posts.parquet").is_file():
        for rel in (
            "app/streamlit_app.py",
            "app/pages/01_Content_Performance.py",
            "app/pages/02_Virality_Prediction.py",
            "app/pages/05_Recommendations.py",
            "app/pages/06_Growth_Dashboard.py",
            "app/pages/07_Trend_Forecasting.py",
        ):
            at_tk = AppTest.from_file(str(root / rel), default_timeout=45)
            at_tk.session_state["socialpulse_dataset_id"] = "tiktok_engagement_200k"
            at_tk.run()
            assert len(at_tk.exception) == 0, f"{rel} failed on tiktok_engagement_200k"
            assert len(at_tk.error) == 0, f"{rel} errored on tiktok_engagement_200k"

    if (ext_root / "zenodo_sponsored_unsponsored_ig" / "posts.parquet").is_file():
        at_ab = AppTest.from_file(str(root / "app" / "pages" / "04_AB_Testing.py"), default_timeout=45)
        at_ab.session_state["socialpulse_dataset_id"] = "zenodo_sponsored_unsponsored_ig"
        at_ab.run()
        assert len(at_ab.exception) == 0
        assert len(at_ab.error) == 0
        warn_text = "\n".join(w.value for w in at_ab.warning)
        assert "is_causal=False" in warn_text

        # Cross-sectional corpus on Page 07 must gracefully report unavailable time series without error
        at_ts = AppTest.from_file(str(root / "app" / "pages" / "07_Trend_Forecasting.py"), default_timeout=30)
        at_ts.session_state["socialpulse_dataset_id"] = "zenodo_sponsored_unsponsored_ig"
        at_ts.run()
        assert len(at_ts.exception) == 0
        assert len(at_ts.error) == 0
        assert len(at_ts.info) >= 1

    if (ext_root / "zenodo_lokananta_comments_ig" / "comments.parquet").is_file():
        at_nlp = AppTest.from_file(str(root / "app" / "pages" / "03_Sentiment_Analysis.py"), default_timeout=30)
        at_nlp.session_state["socialpulse_dataset_id"] = "zenodo_lokananta_comments_ig"
        at_nlp.run()
        assert len(at_nlp.exception) == 0
        assert len(at_nlp.error) == 0
