"""Reusable Streamlit UI widgets, design tokens, CSS styling, and Plotly theme helpers."""

from __future__ import annotations

from html import escape
from typing import Any, Dict, Optional

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

SOCIALPULSE_PALETTE: Dict[str, Any] = {
    "primary_bg": "#0F172A",
    "primary_accent": "#0284C7",
    "secondary_accent": "#0D9488",
    "main_canvas": "#F8FAFC",
    "card_surface": "#FFFFFF",
    "border": "#E2E8F0",
    "text_primary": "#0F172A",
    "text_secondary": "#64748B",
    "positive": "#16A34A",
    "warning": "#D97706",
    "negative": "#DC2626",
    "series": [
        "#0284C7",
        "#0D9488",
        "#16A34A",
        "#D97706",
        "#DC2626",
        "#4F46E5",
        "#64748B",
    ],
}

_PLOTLY_TEMPLATE_NAME = "socialpulse"
_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

_GLOBAL_CSS = f"""
<style id="socialpulse-global-css">
:root {{
    --sp-primary-bg: {SOCIALPULSE_PALETTE["primary_bg"]};
    --sp-primary-accent: {SOCIALPULSE_PALETTE["primary_accent"]};
    --sp-secondary-accent: {SOCIALPULSE_PALETTE["secondary_accent"]};
    --sp-canvas: {SOCIALPULSE_PALETTE["main_canvas"]};
    --sp-surface: {SOCIALPULSE_PALETTE["card_surface"]};
    --sp-border: {SOCIALPULSE_PALETTE["border"]};
    --sp-text-primary: {SOCIALPULSE_PALETTE["text_primary"]};
    --sp-text-secondary: {SOCIALPULSE_PALETTE["text_secondary"]};
    --sp-positive: {SOCIALPULSE_PALETTE["positive"]};
    --sp-warning: {SOCIALPULSE_PALETTE["warning"]};
    --sp-negative: {SOCIALPULSE_PALETTE["negative"]};
}}

/* Zero out empty layout gaps around pure <style> injection containers */
div[data-testid="stElementContainer"]:has(style#socialpulse-global-css),
div.element-container:has(style#socialpulse-global-css) {{
    display: none !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 0 !important;
}}

/* Global app typography & canvas spacing */
html, body, .stApp {{
    font-family: {_FONT_STACK};
    background-color: var(--sp-canvas);
    color: var(--sp-text-primary);
}}

[data-testid="stMainBlockContainer"],
.block-container {{
    padding-top: 3.9rem !important;
    padding-bottom: 2.75rem !important;
    max-width: 1420px;
}}

h1, h2, h3, h4 {{
    color: var(--sp-text-primary);
    letter-spacing: -0.015em;
}}

hr {{
    border: none;
    border-top: 1px solid var(--sp-border);
    margin: 1.25rem 0;
}}

/* Sidebar navigation & SaaS dark rail polish */
[data-testid="stSidebar"] {{
    background-color: var(--sp-primary-bg);
    border-right: 1px solid #1E293B;
}}

[data-testid="stSidebarNav"] {{
    padding-top: 0.25rem;
}}

[data-testid="stSidebarNav"] [data-testid="stSidebarNavSeparator"],
[data-testid="stSidebar"] hr {{
    border-color: #1E293B;
}}

[data-testid="stSidebarNav"] > ul > div > header,
[data-testid="stSidebarNav"] span[data-testid="stNavSectionHeader"] {{
    text-transform: uppercase;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #94A3B8;
}}

[data-testid="stSidebarNavLink"] {{
    border-radius: 8px;
    margin: 2px 8px;
    padding: 6px 10px;
    transition: background-color 0.15s ease, color 0.15s ease;
}}

[data-testid="stSidebarNavLink"]:hover {{
    background-color: rgba(148, 163, 184, 0.14);
}}

[data-testid="stSidebarNavLink"][aria-current="page"] {{
    background: rgba(2, 132, 199, 0.22);
    border-left: 3px solid #38BDF8;
    font-weight: 600;
}}

[data-testid="stSidebar"] .stButton > button {{
    background-color: #1E293B;
    color: #F8FAFC;
    border: 1px solid #334155;
    border-radius: 8px;
    width: 100%;
}}

[data-testid="stSidebar"] .stButton > button:hover {{
    background-color: #0284C7;
    border-color: #0284C7;
    color: #FFFFFF;
}}

/* SocialPulse Hero Header Banner (compatible across Home & Modules 1-7) */
.sp-hero {{
    background: linear-gradient(135deg, #0F172A 0%, #0C4A6E 62%, #0D9488 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
    color: #FFFFFF;
    box-shadow: 0 8px 20px -8px rgba(15, 23, 42, 0.28);
}}

.sp-hero-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.20);
    color: #BAE6FD;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}

.sp-hero-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #2DD4BF;
    display: inline-block;
}}

.sp-hero-title {{
    margin: 10px 0 4px 0 !important;
    padding: 0 !important;
    font-size: clamp(1.35rem, 2.2vw, 1.85rem) !important;
    font-weight: 700 !important;
    line-height: 1.24 !important;
    letter-spacing: -0.02em !important;
    color: #FFFFFF !important;
}}

.sp-hero-subtitle {{
    margin: 0 !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    color: #E2E8F0 !important;
    font-weight: 400 !important;
}}

/* Equal-height alignment for KPI card & Module card rows */
div[data-testid="stHorizontalBlock"]:has(.sp-kpi-card) {{
    align-items: stretch;
}}

div[data-testid="stHorizontalBlock"]:has(.sp-kpi-card) > div[data-testid="stColumn"] {{
    display: flex;
    flex-direction: column;
}}

div[data-testid="stHorizontalBlock"]:has(.sp-kpi-card) > div[data-testid="stColumn"] > div,
div[data-testid="stHorizontalBlock"]:has(.sp-kpi-card) > div[data-testid="stColumn"] [data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"]:has(.sp-kpi-card) [data-testid="stElementContainer"]:only-child,
div[data-testid="stHorizontalBlock"]:has(.sp-kpi-card) [data-testid="stElementContainer"]:only-child > div,
div[data-testid="stHorizontalBlock"]:has(.sp-kpi-card) [data-testid="stElementContainer"]:only-child [data-testid="stMarkdownContainer"] {{
    height: 100%;
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
}}

div[data-testid="stHorizontalBlock"]:has(.sp-kpi-card) [data-testid="stElementContainer"]:only-child .sp-kpi-card {{
    flex: 1 1 auto;
    height: 100%;
}}

/* Equal-height KPI Card System */
.sp-kpi-card {{
    background-color: var(--sp-surface);
    border: 1px solid var(--sp-border);
    border-top: 3px solid var(--sp-primary-accent);
    border-radius: 12px;
    padding: 15px 17px;
    min-height: 134px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    margin-bottom: 10px;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}

.sp-kpi-card:hover {{
    border-color: #CBD5E1;
    box-shadow: 0 4px 12px -4px rgba(15, 23, 42, 0.08);
}}

.sp-kpi-label {{
    color: var(--sp-text-secondary);
    font-size: 11.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.045em;
    line-height: 1.35;
}}

.sp-kpi-value {{
    color: var(--sp-text-primary);
    font-size: clamp(1.08rem, 1.42vw, 1.38rem);
    font-weight: 700;
    line-height: 1.25;
    margin-top: 6px;
    font-variant-numeric: tabular-nums;
    overflow-wrap: anywhere;
}}

.sp-kpi-delta {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: 8px;
    background-color: #DCFCE7;
    color: #15803D;
    border-radius: 999px;
    padding: 2px 9px;
    font-size: 11.5px;
    font-weight: 600;
    width: fit-content;
}}

/* Native st.metric card alignment (e.g., locked Module 2) */
[data-testid="stMetric"] {{
    background-color: var(--sp-surface);
    border: 1px solid var(--sp-border);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}}

/* Section Header Component */
.sp-section-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 10px;
    padding-bottom: 10px;
    margin: 22px 0 14px 0;
    border-bottom: 1px solid var(--sp-border);
}}

.sp-section-title {{
    margin: 0 !important;
    font-size: 1.22rem !important;
    font-weight: 700 !important;
    color: var(--sp-text-primary) !important;
    line-height: 1.3 !important;
}}

.sp-section-subtitle {{
    margin: 3px 0 0 0 !important;
    font-size: 0.88rem !important;
    color: var(--sp-text-secondary) !important;
    line-height: 1.45 !important;
}}

/* Semantic Badges & Banners */
.sp-badge {{
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 0.02em;
    white-space: nowrap;
}}

.sp-badge-accent {{ background: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }}
.sp-badge-teal {{ background: #CCFBF1; color: #0F766E; border: 1px solid #99F6E4; }}
.sp-badge-positive {{ background: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; }}
.sp-badge-warning {{ background: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }}
.sp-badge-negative {{ background: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }}
.sp-badge-neutral {{ background: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }}

.sp-banner {{
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 14px;
    border: 1px solid var(--sp-border);
    background-color: var(--sp-surface);
    color: var(--sp-text-primary);
}}

.sp-banner-info {{ border-left: 4px solid var(--sp-primary-accent); background-color: #F0F9FF; border-color: #BAE6FD; }}
.sp-banner-warning {{ border-left: 4px solid var(--sp-warning); background-color: #FFFBEB; border-color: #FDE68A; }}
.sp-banner-teal {{ border-left: 4px solid var(--sp-secondary-accent); background-color: #F0FDFA; border-color: #99F6E4; }}

/* Home Page Module Directory Cards */
div[data-testid="stHorizontalBlock"]:has(.sp-module-card) {{
    align-items: stretch;
}}

div[data-testid="stHorizontalBlock"]:has(.sp-module-card) > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"]:has(.sp-module-card) [data-testid="stVerticalBlock"] {{
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
    height: 100%;
}}

.sp-module-card {{
    background-color: var(--sp-surface);
    border: 1px solid var(--sp-border);
    border-top: 3px solid var(--sp-primary-accent);
    border-radius: 12px 12px 6px 6px;
    padding: 18px 18px 14px 18px;
    min-height: 196px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    margin-bottom: 6px;
}}

.sp-module-card.is-teal {{
    border-top-color: var(--sp-secondary-accent);
}}

.sp-module-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}}

.sp-module-title {{
    font-size: 1rem;
    font-weight: 700;
    color: var(--sp-text-primary);
    margin: 0 0 6px 0;
    line-height: 1.35;
}}

.sp-module-desc {{
    font-size: 0.875rem;
    color: #475569;
    line-height: 1.48;
    margin: 0;
}}

.sp-module-tags {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}}

.sp-tag {{
    font-size: 11px;
    font-weight: 600;
    color: #475569;
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 2px 7px;
}}

/* Home Pillar Cards */
.sp-pillar-card {{
    background-color: var(--sp-surface);
    border: 1px solid var(--sp-border);
    border-radius: 12px;
    padding: 16px 18px;
    min-height: 132px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}}

.sp-pillar-title {{
    font-size: 0.96rem;
    font-weight: 700;
    color: var(--sp-text-primary);
    margin-bottom: 6px;
}}

.sp-pillar-body {{
    font-size: 0.855rem;
    color: #475569;
    line-height: 1.5;
}}

/* Chart, Table, Tab, Alert, Expander card surfaces across all modules */
[data-testid="stPlotlyChart"] {{
    background-color: var(--sp-surface);
    border: 1px solid var(--sp-border);
    border-radius: 12px;
    padding: 12px 14px 14px 14px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    margin-bottom: 14px;
}}

[data-testid="stDataFrame"],
[data-testid="stTable"] {{
    background-color: var(--sp-surface);
    border: 1px solid var(--sp-border);
    border-radius: 10px;
    overflow: hidden;
    font-variant-numeric: tabular-nums;
}}

[data-testid="stTabs"] [role="tablist"] {{
    gap: 6px;
    border-bottom: 1px solid var(--sp-border);
    overflow-x: auto;
    scrollbar-width: thin;
}}

[data-testid="stTabs"] button[role="tab"] {{
    font-weight: 600;
    font-size: 0.89rem;
    padding: 8px 14px;
    border-radius: 8px 8px 0 0;
    white-space: nowrap;
}}

[data-testid="stExpander"] {{
    background-color: var(--sp-surface);
    border: 1px solid var(--sp-border);
    border-radius: 10px;
}}

[data-testid="stAlert"] {{
    border-radius: 10px;
}}

[data-testid="stPageLink-NavLink"] {{
    background-color: #FFFFFF;
    border: 1px solid var(--sp-border);
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: 600;
    color: var(--sp-primary-accent) !important;
    transition: background-color 0.15s ease, border-color 0.15s ease;
}}

[data-testid="stPageLink-NavLink"]:hover {{
    background-color: #F0F9FF;
    border-color: var(--sp-primary-accent);
}}

/* Safeguard readability inside inline HTML cards authored in locked Pages 03-06 */
.stMainBlockContainer div[style*="background-color: #F8FAFC"],
.stMainBlockContainer div[style*="background-color: #FFFFFF"],
.stMainBlockContainer div[style*="background-color: #F1F5F9"] {{
    color: var(--sp-text-primary);
    border-radius: 10px;
}}

.stMainBlockContainer div[style*="justify-content: space-between"] {{
    flex-wrap: wrap;
    gap: 8px;
}}

/* Footer */
.sp-footer {{
    margin-top: 28px;
    padding-top: 14px;
    border-top: 1px solid var(--sp-border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.8rem;
    color: var(--sp-text-secondary);
}}

/* Tablet & Mobile Responsive Rules */
@media (max-width: 1024px) {{
    [data-testid="stMainBlockContainer"],
    .block-container {{
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }}
}}

@media (min-width: 641px) and (max-width: 1024px) {{
    .sp-kpi-card {{
        min-height: 134px;
        padding: 14px 15px;
    }}
    .sp-module-card {{
        min-height: 268px;
        padding: 16px 16px 13px 16px;
    }}
}}

@media (max-width: 640px) {{
    [data-testid="stMainBlockContainer"],
    .block-container {{
        padding-top: 3.75rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
        padding-bottom: 2rem !important;
    }}
    h1 {{
        font-size: 1.45rem !important;
        line-height: 1.25 !important;
    }}
    h2 {{
        font-size: 1.24rem !important;
        line-height: 1.3 !important;
    }}
    h3 {{
        font-size: 1.08rem !important;
        line-height: 1.35 !important;
    }}
    .sp-hero {{
        padding: 16px 18px;
        border-radius: 12px;
    }}
    .sp-kpi-card {{
        min-height: auto;
        padding: 14px 15px;
    }}
    .sp-module-card,
    .sp-pillar-card {{
        min-height: auto;
        padding: 15px;
    }}
    [data-testid="stPlotlyChart"] {{
        padding: 6px;
    }}
    .katex-display {{
        overflow-x: auto;
        overflow-y: hidden;
        max-width: 100%;
        padding: 4px 0;
    }}
}}
</style>
"""


def configure_plotly_theme() -> None:
    """Register and activate the cohesive SocialPulse Plotly template (idempotent)."""
    template = go.layout.Template(pio.templates["plotly_white"])
    template.layout.paper_bgcolor = "#FFFFFF"
    template.layout.plot_bgcolor = "#FFFFFF"
    template.layout.colorway = SOCIALPULSE_PALETTE["series"]
    template.layout.font = dict(
        family=_FONT_STACK,
        size=12,
        color=SOCIALPULSE_PALETTE["text_primary"],
    )
    template.layout.title = dict(
        font=dict(size=15, color=SOCIALPULSE_PALETTE["text_primary"]),
        x=0.0,
        xanchor="left",
        pad=dict(b=10),
    )
    template.layout.hoverlabel = dict(
        bgcolor="#0F172A",
        bordercolor="#0284C7",
        font=dict(color="#F8FAFC", size=12, family=_FONT_STACK),
    )
    template.layout.xaxis = dict(
        showgrid=True,
        gridcolor="#E2E8F0",
        gridwidth=1,
        zeroline=False,
        linecolor="#CBD5E1",
        tickfont=dict(color="#64748B", size=11),
        title=dict(font=dict(color="#475569", size=12)),
        automargin=True,
    )
    template.layout.yaxis = dict(
        showgrid=True,
        gridcolor="#E2E8F0",
        gridwidth=1,
        zeroline=False,
        linecolor="#CBD5E1",
        tickfont=dict(color="#64748B", size=11),
        title=dict(font=dict(color="#475569", size=12)),
        automargin=True,
    )
    template.layout.margin = dict(l=52, r=28, t=60, b=58)
    pio.templates[_PLOTLY_TEMPLATE_NAME] = template
    pio.templates.default = _PLOTLY_TEMPLATE_NAME


def apply_plotly_theme(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    """Apply consistent SocialPulse typography, colors, gridlines, and spacing to a Plotly figure."""
    configure_plotly_theme()
    layout_kwargs: Dict[str, Any] = {
        "template": _PLOTLY_TEMPLATE_NAME,
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "font": dict(family=_FONT_STACK, size=12, color=SOCIALPULSE_PALETTE["text_primary"]),
    }
    if height is not None:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    return fig


def inject_global_styles() -> None:
    """Inject SocialPulse CSS tokens and activate unified Plotly defaults."""
    configure_plotly_theme()
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "") -> None:
    """Render consistent SocialPulse branded hero header across all pages.

    Signature stays 100% backward-compatible with Modules 1-6:
    `render_header(title: str, subtitle: str = "")`.
    """
    inject_global_styles()
    safe_title = escape(str(title))
    safe_subtitle = escape(str(subtitle)) if subtitle else ""
    subtitle_html = f'<p class="sp-hero-subtitle">{safe_subtitle}</p>' if safe_subtitle else ""
    st.markdown(
        f"""
        <div class="sp-hero">
            <div class="sp-hero-eyebrow"><span class="sp-hero-dot"></span>SocialPulse · Analytics &amp; Growth Intelligence</div>
            <h1 class="sp-hero-title">{safe_title}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, delta: str = "") -> None:
    """Render an equal-height SaaS KPI metric card.

    Signature stays 100% backward-compatible with Modules 1-6:
    `render_metric_card(label: str, value: str, delta: str = "")`.
    """
    safe_label = escape(str(label))
    safe_value = escape(str(value))
    delta_html = (
        f'<div class="sp-kpi-delta">↑ {escape(str(delta))}</div>'
        if delta
        else ""
    )
    st.markdown(
        f"""
        <div class="sp-kpi-card">
            <div>
                <div class="sp-kpi-label">{safe_label}</div>
                <div class="sp-kpi-value">{safe_value}</div>
            </div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(label: str, tone: str = "neutral") -> str:
    """Return an inline HTML status pill for headers, callouts, and cards."""
    allowed_tones = {"accent", "teal", "positive", "warning", "negative", "neutral"}
    tone_cls = tone if tone in allowed_tones else "neutral"
    return f'<span class="sp-badge sp-badge-{tone_cls}">{escape(str(label))}</span>'


def render_section_header(title: str, subtitle: str = "", badge: str = "", badge_tone: str = "accent") -> None:
    """Render a consistent section header with bottom divider and optional status badge."""
    safe_title = escape(str(title))
    subtitle_html = f'<p class="sp-section-subtitle">{escape(str(subtitle))}</p>' if subtitle else ""
    badge_html = render_status_badge(badge, tone=badge_tone) if badge else ""
    st.markdown(
        f"""
        <div class="sp-section-header">
            <div>
                <h2 class="sp-section-title">{safe_title}</h2>
                {subtitle_html}
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_banner(title: str, body: str, tone: str = "info") -> None:
    """Render a high-contrast informational/methodological banner card."""
    tone_cls = tone if tone in {"info", "warning", "teal"} else "info"
    st.markdown(
        f"""
        <div class="sp-banner sp-banner-{tone_cls}">
            <div style="font-size: 0.86rem; font-weight: 700; color: #0F172A; margin-bottom: 4px;">{escape(str(title))}</div>
            <div style="font-size: 0.855rem; color: #334155; line-height: 1.5;">{escape(str(body))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(note: str = "Data-Driven Social Engagement Initiative · Synthetic Benchmark Environment") -> None:
    """Render standardized SocialPulse application footer."""
    safe_note = escape(str(note))
    st.markdown(
        f"""
        <div class="sp-footer">
            <span><strong>SocialPulse</strong> — Intelligent Social Media Analytics &amp; Engagement Optimization Platform</span>
            <span>{safe_note}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
