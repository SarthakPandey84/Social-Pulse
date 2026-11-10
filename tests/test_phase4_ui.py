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
