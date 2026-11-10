"""Reusable Streamlit UI widgets, design tokens, CSS styling, and Plotly theme helpers."""

from __future__ import annotations

from html import escape
import math
import re
from typing import Any, Dict, List, Optional

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


_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _safe_hex_color(color: str, default: str = "#64748B") -> str:
    """Validate hex color token to prevent CSS style attribute injection."""
    candidate = str(color).strip()
    return candidate if _HEX_COLOR_RE.match(candidate) else default


def build_nlp_diagnostic_card_html(
    eyebrow: str,
    headline: str,
    metric_label: str,
    metric_value: str,
    accent_color: str,
    footnote: str = "",
) -> str:
    """Build escaped HTML card for Module 3 NLP diagnostic metrics."""
    safe_color = _safe_hex_color(accent_color)
    footnote_html = (
        f'<div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">{escape(str(footnote))}</div>'
        if footnote
        else ""
    )
    return f"""
                <div style="background-color: #F8FAFC; border-left: 5px solid {safe_color}; padding: 16px; border-radius: 8px;">
                    <div style="font-size: 13px; color: #64748B; font-weight: 600;">{escape(str(eyebrow))}</div>
                    <div style="font-size: 24px; font-weight: 700; color: {safe_color}; margin-top: 4px;">{escape(str(headline))}</div>
                    <div style="font-size: 13px; color: #475569; margin-top: 4px;">{escape(str(metric_label))} <b>{escape(str(metric_value))}</b></div>
                    {footnote_html}
                </div>
                """


def build_ab_experiment_banner_html(
    exp_id: Any,
    exp_type_badge: Any,
    hypothesis: Any,
    platform: Any,
    variant_a_label: Any,
    variant_b_label: Any,
) -> str:
    """Build escaped HTML experiment registration banner for Page 04."""
    return f"""
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 14px; margin-bottom: 16px;">
            <div style="font-size: 13px; font-weight: 700; color: #475569;">
                EXPERIMENT REGISTRATION: <span style="background-color: #E2E8F0; padding: 2px 8px; border-radius: 4px;">{escape(str(exp_id))}</span> | {escape(str(exp_type_badge))}
            </div>
            <div style="font-size: 14px; margin-top: 6px;"><b>Hypothesis:</b> <i>"{escape(str(hypothesis))}"</i></div>
            <div style="font-size: 13px; color: #64748B; margin-top: 4px;">
                Platform: <b>{escape(str(platform))}</b> |
                Control (A): <b>{escape(str(variant_a_label))}</b> |
                Treatment (B): <b>{escape(str(variant_b_label))}</b> |
                Observation Horizon: <b>7-day settled post observations (deduplicated)</b>
            </div>
        </div>
        """


def build_ab_verdict_card_html(
    badge_color: str,
    alpha_val: float,
    verdict: Any,
    primary_method: Any,
    test_statistic: float,
    effect_size_type: Any,
    effect_size: float,
    ci_low: float,
    ci_high: float,
    ci_method: Any,
    unit_of_analysis: Any,
    cluster_badge: Any,
) -> str:
    """Build escaped HTML statistical verdict card for Page 04."""
    safe_color = _safe_hex_color(badge_color)
    return f"""
        <div style="background-color: #F8FAFC; border-left: 5px solid {safe_color}; padding: 16px; border-radius: 8px; margin-top: 12px;">
            <div style="font-size: 13px; color: #64748B; font-weight: 600;">STATISTICAL VERDICT (α = {escape(str(alpha_val))})</div>
            <div style="font-size: 20px; font-weight: 700; color: {safe_color}; margin-top: 4px;">{escape(str(verdict))}</div>
            <div style="font-size: 13px; color: #475569; margin-top: 8px;">
                <b>Primary Method:</b> {escape(str(primary_method))} |
                <b>Test Statistic:</b> {escape(f"{float(test_statistic):.4f}")} |
                <b>Effect Size:</b> {escape(str(effect_size_type))} = {escape(f"{float(effect_size):+.3f}")} |
                <b>Confidence Interval:</b> [{escape(f"{float(ci_low):.4f}")}, {escape(f"{float(ci_high):.4f}")}] ({escape(str(ci_method))})<br>
                <b>Unit of Analysis:</b> {escape(str(unit_of_analysis))} |
                <b>Clustering Handled:</b> {escape(str(cluster_badge))} |
                <b>Data Provenance:</b> Synthetic Benchmark Fixture
            </div>
        </div>
        """


def build_ab_mann_whitney_card_html(p_value: float, mw_sig: Any) -> str:
    """Build escaped HTML Mann-Whitney secondary diagnostic card for Page 04."""
    return f"""
                <div style="background-color: #F1F5F9; border-radius: 6px; padding: 10px; margin-top: 10px; font-size: 13px;">
                    <b>Non-Parametric Diagnostic (Mann-Whitney U):</b> p = {escape(f"{float(p_value):.5f}")} ({escape(str(mw_sig))})<br>
                    <span style="color: #64748B;">Evaluates rank distribution stochastic dominance P(X &gt; Y) + 0.5P(X = Y) = 0.5; does not assume equal shapes or medians.</span>
                </div>
                """


def build_ab_fisher_secondary_card_html(z_stat: float, p_value: float) -> str:
    """Build escaped HTML secondary asymptotic z-test card for Page 04."""
    return f"""
                <div style="background-color: #F1F5F9; border-radius: 6px; padding: 10px; margin-top: 10px; font-size: 13px;">
                    <b>Secondary Diagnostic (Asymptotic z-test):</b> z = {escape(f"{float(z_stat):.3f}")}, p = {escape(f"{float(p_value):.5f}")}<br>
                    <span style="color: #64748B;">Retained for diagnostic reference; verdict was governed strictly by Fisher's Exact Test due to sparse cell count.</span>
                </div>
                """


def build_recommendation_detail_card_html(
    idx: int,
    confidence_level: Any,
    is_synthetic_demonstration: bool,
    title: Any,
    category: Any,
    platform_label: Any,
    source_label: Any,
    priority_score: float,
    suggested_action: Any,
    reasoning: Any,
    supporting_evidence: Any,
    sample_size: Any,
    unit_of_analysis: Any,
    adj_p_str: Any,
    ci_str: Any,
    expected_impact_direction: Any,
    limitations_and_caveats: Any,
    suggested_validation_method: Any,
) -> str:
    """Build escaped HTML recommendation card for Page 05."""
    conf_val = getattr(confidence_level, "value", str(confidence_level)).upper()
    if conf_val == "HIGH":
        border_color = "#10B981"
        conf_badge = '<span style="background-color: #D1FAE5; color: #065F46; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px;">🟢 HIGH EVIDENCE (DEMO RCT)</span>'
    elif conf_val == "MEDIUM":
        border_color = "#F59E0B"
        conf_badge = '<span style="background-color: #FEF3C7; color: #92400E; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px;">🟡 MEDIUM EVIDENCE</span>'
    else:
        border_color = "#94A3B8"
        conf_badge = '<span style="background-color: #F1F5F9; color: #475569; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px;">⚪ EXPLORATORY / LOW</span>'

    demo_pill = (
        '<span style="background-color: #EDE9FE; color: #5B21B6; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; margin-left: 6px;">SYNTHETIC DEMO</span>'
        if is_synthetic_demonstration
        else ""
    )
    cat_str = getattr(category, "value", str(category))

    return f"""
            <div style="background-color: #FFFFFF; border-left: 6px solid {border_color}; border: 1px solid #E2E8F0; border-left-width: 6px; border-radius: 8px; padding: 18px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 16px; font-weight: 700; color: #0F172A;">
                        #{escape(str(idx))}. {escape(str(title))} {demo_pill}
                    </div>
                    <div>{conf_badge}</div>
                </div>
                <div style="font-size: 12px; color: #64748B; margin-top: 4px;">
                    Category: <b>{escape(str(cat_str))}</b> |
                    Platform: <b>{escape(str(platform_label))}</b> |
                    Source: <b>{escape(str(source_label))}</b> |
                    Priority Score: <b>{escape(f"{float(priority_score):.4f}")}</b>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px; margin-top: 10px; font-size: 14px;">
                    <b>Suggested Action:</b><br>{escape(str(suggested_action))}
                </div>
                <div style="margin-top: 10px; font-size: 13px; color: #334155;">
                    <b>Reasoning:</b> {escape(str(reasoning))}
                </div>
                <div style="margin-top: 6px; font-size: 13px; color: #1E293B;">
                    <b>Supporting Evidence:</b> {escape(str(supporting_evidence))}
                </div>
                <div style="margin-top: 6px; font-size: 12px; color: #0284C7; background-color: #F0F9FF; padding: 6px 10px; border-radius: 4px;">
                    <b>Statistical Metrics:</b> Sample Size = <b>N={escape(str(sample_size))}</b> |
                    Unit = <b>{escape(str(unit_of_analysis))}</b> |
                    Holm Adj. p = <b>{escape(str(adj_p_str))}</b> |
                    95% CI = <b>{escape(str(ci_str))}</b>
                </div>
                <div style="margin-top: 6px; font-size: 13px; color: #0369A1;">
                    <b>Expected Impact:</b> <b>{escape(str(expected_impact_direction))}</b>
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: #B45309; background-color: #FFFBEB; padding: 8px; border-radius: 4px;">
                    <b>Limitations &amp; Caveats:</b> {escape(str(limitations_and_caveats))}
                </div>
                <div style="margin-top: 6px; font-size: 12px; color: #047857;">
                    <b>Recommended Validation:</b> {escape(str(suggested_validation_method))}
                </div>
            </div>
            """


def _format_p_value_sci(value: Any, unavailable: str = "N/A (not calculated)") -> str:
    """Format a p-value safely in scientific notation without calling float(None)."""
    if value is None or isinstance(value, bool):
        return unavailable
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return unavailable
    if not math.isfinite(numeric):
        return unavailable
    return f"{numeric:.2e}"


def build_dashboard_experiment_card_html(
    is_randomized: bool,
    is_significant: bool,
    experiment_id: Any,
    name: Any,
    platform: Any,
    sample_size: Any,
    unit_of_analysis: Any,
    test_name: Any,
    variant_a: Any,
    value_a: float,
    variant_b: Any,
    value_b: float,
    relative_lift_pct: float,
    effect_size: float,
    effect_size_type: Any,
    raw_p_value: float,
    holm_adj_p_value: Any,
    ci_low: float,
    ci_high: float,
    ci_method: Any,
) -> str:
    """Build escaped HTML experiment summary card for Page 06."""
    badge_color = "#10B981" if is_randomized and is_significant else ("#F59E0B" if not is_randomized else "#EF4444")
    badge_label = (
        "🟢 RANDOMIZED RCT (STAT. SIG)"
        if is_randomized and is_significant
        else ("🟡 OBSERVATIONAL COHORT (NON-RANDOM)" if not is_randomized else "🔴 INCONCLUSIVE")
    )
    holm_adj_str = _format_p_value_sci(holm_adj_p_value)
    return f"""
            <div style="background-color: #FFFFFF; border-left: 5px solid {badge_color}; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <b>{escape(str(experiment_id))}: {escape(str(name))}</b>
                    <span style="font-size: 11px; font-weight: 700; color: {badge_color};">{badge_label}</span>
                </div>
                <div style="font-size: 12px; color: #64748B; margin-top: 4px;">
                    Platform: <b>{escape(str(platform))}</b> |
                    Sample Size: <b>N = {escape(str(sample_size))} posts</b> |
                    Unit: <b>{escape(str(unit_of_analysis))}</b> |
                    Test: <b>{escape(str(test_name))}</b>
                </div>
                <div style="font-size: 13px; margin-top: 6px; color: #1E293B;">
                    • <b>Variant A:</b> {escape(str(variant_a))} (Mean: {escape(f"{float(value_a):.4f}")})<br>
                    • <b>Variant B:</b> {escape(str(variant_b))} (Mean: {escape(f"{float(value_b):.4f}")})<br>
                    • <b>Observed Relative Lift:</b> <b>{escape(f"{float(relative_lift_pct):+.2f}%")}</b> |
                    Effect Size: <b>{escape(f"{float(effect_size):.3f}")} ({escape(str(effect_size_type))})</b>
                </div>
                <div style="font-size: 12px; margin-top: 6px; color: #0284C7;">
                    <b>Statistical Inference:</b> Raw p = {escape(f"{float(raw_p_value):.2e}")} |
                    <b>Holm-Bonferroni Adj. p = {escape(holm_adj_str)}</b> |
                    95% CI: [{escape(f"{float(ci_low):,.2f}")}, {escape(f"{float(ci_high):,.2f}")}] ({escape(str(ci_method))})
                </div>
            </div>
            """


def build_dashboard_recommendation_card_html(
    idx: int,
    confidence_level: Any,
    title: Any,
    priority_score: float,
    suggested_action: Any,
    expected_impact_direction: Any,
    suggested_validation_method: Any,
) -> str:
    """Build escaped HTML recommendation summary card for Page 06."""
    conf_val = getattr(confidence_level, "value", str(confidence_level)).upper()
    c_badge = "🟢 HIGH EVIDENCE" if conf_val == "HIGH" else "⚪ EXPLORATORY / LOW"
    return f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between;">
                    <b>#{escape(str(idx))}. {escape(str(title))}</b>
                    <span style="font-size: 11px; font-weight: 700;">{c_badge} (Score: {escape(f"{float(priority_score):.4f}")})</span>
                </div>
                <div style="font-size: 13px; color: #334155; margin-top: 4px;"><b>Suggested Action:</b> {escape(str(suggested_action))}</div>
                <div style="font-size: 12px; color: #0284C7; margin-top: 4px;"><b>Expected Impact:</b> {escape(str(expected_impact_direction))} | <b>Recommended Validation:</b> {escape(str(suggested_validation_method))}</div>
            </div>
            """


_PROVENANCE_BADGE_MAP: Dict[str, tuple[str, str]] = {
    "real_public_data": ("Real Public Data", "teal"),
    "owner_authorized_data": ("Owner-Authorized Data", "positive"),
    "synthetic_data": ("Synthetic Benchmark", "warning"),
    "simulated_data": ("Simulated Benchmark", "warning"),
    "external_benchmark_unverified": ("External Benchmark (Unverified Origin)", "warning"),
    "weakly_labelled_data": ("Weakly Labelled Data", "warning"),
}

_GOVERNANCE_SHORT_TAG: Dict[str, str] = {
    "VERIFIED_COMMERCIAL_SAFE": "Commercial-Safe",
    "NON_COMMERCIAL_RESEARCH_ONLY": "Research-Only",
    "PENDING_REVIEW_WEAKLY_LABELLED": "Pending Review / Weak Labels",
    "EXTERNAL_BENCHMARK_UNVERIFIED": "Unverified Benchmark",
    "SYNTHETIC_BENCHMARK": "Synthetic Benchmark",
    "BLOCKED_MISSING_CREDENTIALS": "Gated / Credentials Missing",
}


def format_dataset_option_label(entry: Dict[str, Any]) -> str:
    """Return a concise, human-readable label for a dataset selector entry."""
    ds_id = str(entry.get("dataset_id", "synthetic_demo"))
    name = str(entry.get("display_name") or ds_id)
    prov = str(entry.get("provenance") or "synthetic_data").lower()
    prov_tag = _PROVENANCE_BADGE_MAP.get(prov, ("External Dataset", "neutral"))[0]
    gov_tier = str(entry.get("governance_tier") or "").strip()
    gov_tag = _GOVERNANCE_SHORT_TAG.get(gov_tier)
    tag_text = f"{prov_tag} · {gov_tag}" if (gov_tag and gov_tag.lower() not in prov_tag.lower()) else prov_tag
    if entry.get("meta_error"):
        return f"{name} [Metadata Error]"
    counts = entry.get("row_counts") or {}
    if isinstance(counts, dict) and counts:
        if "posts" in counts:
            size_hint = f" · {int(counts['posts']):,} posts"
        elif "comments" in counts:
            size_hint = f" · {int(counts['comments']):,} comments"
        else:
            size_hint = ""
    else:
        size_hint = ""
    return f"{name} [{tag_text}{size_hint}]"


def render_dataset_selector(
    available_datasets: List[Dict[str, Any]],
    all_dataset_ids: Optional[List[str]] = None,
    key: str = "sp_dataset_selector",
    label: str = "Active Dataset Corpus:",
    location: str = "sidebar",
    help_text: Optional[str] = None,
) -> str:
    """Render unified dataset switcher synchronized across pages via ``st.session_state['socialpulse_dataset_id']``."""
    option_map = {str(item["dataset_id"]): item for item in available_datasets if item.get("dataset_id")}
    option_ids = list(option_map.keys()) or ["synthetic_demo"]

    current_global = st.session_state.get("socialpulse_dataset_id", option_ids[0])
    known_universe = set(all_dataset_ids) if all_dataset_ids is not None else set(option_ids)

    # If caller/user explicitly requested an unknown dataset_id not registered anywhere,
    # surface that exact ID so loader raises DatasetLoadError instead of masking it.
    if current_global and current_global not in known_universe:
        return str(current_global)

    target_container = st.sidebar if location == "sidebar" else st
    if current_global not in option_ids:
        target_container.caption(
            f"Note: Active workspace dataset `{current_global}` does not provide this module's table; "
            f"displaying `{option_ids[0]}` for this view."
        )
        default_idx = 0
    else:
        default_idx = option_ids.index(current_global)

    selected_id = target_container.selectbox(
        label,
        options=option_ids,
        index=default_idx,
        format_func=lambda ds_id: format_dataset_option_label(option_map[ds_id]) if ds_id in option_map else str(ds_id),
        key=key,
        help=help_text or "Switch between the offline synthetic benchmark fixtures and prepared real/external corpora.",
    )
    if selected_id and (selected_id != current_global or current_global in option_ids):
        st.session_state["socialpulse_dataset_id"] = selected_id
    return str(selected_id)


def render_provenance_banner(meta: Dict[str, Any], extra_note: str = "") -> None:
    """Render an XSS-safe provenance, governance-tier, license, proxy-metric, and limitation disclosure banner."""
    prov = str(meta.get("provenance") or "synthetic_data").lower()
    badge_title, _ = _PROVENANCE_BADGE_MAP.get(prov, ("External Dataset", "info"))
    name = str(meta.get("display_name") or meta.get("dataset_id") or "Active Corpus")
    platforms = ", ".join(str(p).title() for p in (meta.get("platforms") or [])) or "Unspecified"
    license_name = str(meta.get("license_name") or "Unknown")
    gov_tier = str(meta.get("governance_tier") or "")
    revision = str(meta.get("pinned_revision") or "N/A")

    parts = [
        f"Source: {name} | Platforms: {platforms} | Provenance: {prov}"
        + (f" | Governance Tier: {gov_tier}" if gov_tier else "")
        + f" | License: {license_name} | Ref: {revision}."
    ]
    proxy_metrics = meta.get("proxy_metrics") or {}
    if isinstance(proxy_metrics, dict) and proxy_metrics:
        proxy_summary = "; ".join(f"{k}: {v}" for k, v in proxy_metrics.items())
        parts.append(f"Proxy metrics: {proxy_summary}")
    limitations = meta.get("limitations") or []
    if limitations:
        parts.append("Key caveats: " + " | ".join(str(x) for x in limitations[:3]))
    if extra_note:
        parts.append(str(extra_note))

    tone = (
        "teal"
        if gov_tier in {"VERIFIED_COMMERCIAL_SAFE", "NON_COMMERCIAL_RESEARCH_ONLY"} or prov in {"real_public_data", "owner_authorized_data"}
        else "info"
    )
    render_info_banner(
        title=f"Dataset Provenance: {badge_title}" + (f" [{_GOVERNANCE_SHORT_TAG.get(gov_tier, gov_tier)}]" if gov_tier else ""),
        body=" ".join(parts),
        tone=tone,
    )
