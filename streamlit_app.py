"""SocialPulse — Global Streamlit Application Entrypoint.

Provides unified multipage navigation and shared branding across all 7 modules:
- Module 1: Content Performance Tracker
- Module 2: Virality Prediction Engine
- Module 3: Audience Sentiment & Relatability Analyzer
- Module 4: A/B Testing Framework
- Module 5: Engagement Optimization Recommender
- Module 6: Growth Visualization Dashboard
- Module 7: Trend Forecasting Engine
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.components.data_loader import artifacts_present_and_nonempty, ensure_artifacts_ready
from app.components.ui_widgets import inject_global_styles

st.set_page_config(
    page_title="SocialPulse — Social Engagement & Growth Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)

# Activate global SaaS styling and Plotly theme across all views
inject_global_styles()

# Branded sidebar & top-left logo lockup
logo_path = root_dir / "assets" / "socialpulse_logo.svg"
icon_path = root_dir / "assets" / "socialpulse_icon.svg"
if logo_path.exists():
    st.logo(
        str(logo_path),
        size="large",
        icon_image=str(icon_path) if icon_path.exists() else None,
    )

# Ensure processed datasets, SQLite warehouse, and trained models are ready before navigation
try:
    if artifacts_present_and_nonempty(root_dir):
        ensure_artifacts_ready(root_dir)
    else:
        with st.spinner("Initializing SocialPulse analytical workspace and trained models..."):
            ensure_artifacts_ready(root_dir)
except Exception as exc:
    st.error(
        "SocialPulse failed to prepare required datasets and ML/NLP model artifacts before startup "
        f"({type(exc).__name__}: {exc}). Verify inputs in `data/sample/` or run "
        "`python scripts/build_all_artifacts.py` for diagnostics."
    )
    st.exception(exc)
    st.stop()

# Configure logically grouped multi-page navigation across Home and Modules 1–7
home_page = st.Page(
    "app/streamlit_app.py",
    title="SocialPulse Hub",
    icon=":material/space_dashboard:",
    default=True,
)
page_1 = st.Page(
    "app/pages/01_Content_Performance.py",
    title="01. Content Performance",
    icon=":material/monitoring:",
)
page_2 = st.Page(
    "app/pages/02_Virality_Prediction.py",
    title="02. Virality Prediction",
    icon=":material/rocket_launch:",
)
page_3 = st.Page(
    "app/pages/03_Sentiment_Analysis.py",
    title="03. Sentiment & Relatability",
    icon=":material/forum:",
)
page_4 = st.Page(
    "app/pages/04_AB_Testing.py",
    title="04. A/B Testing",
    icon=":material/science:",
)
page_5 = st.Page(
    "app/pages/05_Recommendations.py",
    title="05. Recommendations",
    icon=":material/lightbulb:",
)
page_6 = st.Page(
    "app/pages/06_Growth_Dashboard.py",
    title="06. Growth Dashboard",
    icon=":material/dashboard:",
)
page_7 = st.Page(
    "app/pages/07_Trend_Forecasting.py",
    title="07. Trend Forecasting",
    icon=":material/trending_up:",
)

navigation_structure = {
    "Overview": [home_page],
    "Performance & Modeling": [page_1, page_2],
    "Audience & Experimentation": [page_3, page_4, page_5],
    "Executive & Forecasting": [page_6, page_7],
}

# Expose registered Page routes for interactive navigation cards on Home
st.session_state["_socialpulse_pages"] = {
    "m1": page_1,
    "m2": page_2,
    "m3": page_3,
    "m4": page_4,
    "m5": page_5,
    "m6": page_6,
    "m7": page_7,
}

pg = st.navigation(navigation_structure)
active_ds = st.session_state.get("socialpulse_dataset_id", "synthetic_demo")
mode_label = "Synthetic Benchmark Mode" if active_ds == "synthetic_demo" else f"Dataset: {active_ds}"
st.sidebar.caption(f"SocialPulse v1.0 · {mode_label}")
pg.run()
