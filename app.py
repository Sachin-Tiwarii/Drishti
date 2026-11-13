import datetime
import html
import inspect
import math

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Drishti | दृष्टि",
    page_icon=":material/visibility:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# TRANSLATIONS (English / Hindi)
# ============================================================

TR = {
    "en": {
        "brand": "Drishti",
        "tagline": "Crime analytics and forecasting · Indore, Madhya Pradesh",
        "switch_to": "हिन्दी",
        "controls_title": "Forecast controls",
        "tag_controls": "CONTROLS",
        "tag_info": "INFORMATION",
        "controls_desc": "Choose the forecast week, model, and area coverage.",
        "overview_title": "Weekly overview",
        "overview_desc": "Forecast summary for the selected week and zone.",
        "forecast_week": "Forecast week",
        "model": "Forecast model",
        "zone": "Zone",
        "all_zones": "All zones",
        "show_names": "Show locality names",
        "week_of": "Week {w}, {y}",
        "kpi_localities": "Localities monitored",
        "kpi_total": "Forecast incidents",
        "kpi_total_note": "Total for selected week",
        "kpi_top": "Highest-load locality",
        "kpi_top_note": "{n} forecast incidents",
        "kpi_elev": "Elevated localities",
        "kpi_elev_note": "Top third by forecast load",
        "tab_map": "Hotspot Map",
        "tab_trend": "Weekly Trends",
        "tab_cat": "Crime Categories",
        "tab_perf": "Model Performance",
        "map_title": "Hotspot Map",
        "map_sub": "Forecast intensity by locality for the selected week.",
        "lvl_low": "Low",
        "lvl_mod": "Moderate",
        "lvl_high": "Elevated",
        "heat_low": "Lower intensity",
        "heat_high": "Higher intensity",
        "focus_title": "Locality Focus",
        "focus_sub": "Select a locality to compare model forecasts.",
        "locality": "Locality",
        "mean_fc": "Mean forecast",
        "per_week": "/ week",
        "consensus": "Model consensus",
        "area_n": "Area {n}",
        "zone_n": "{z} zone",
        "lat": "Latitude",
        "lon": "Longitude",
        "hist": "Historical total",
        "priority": "Priority Localities",
        "priority_sub": "Ranked by forecast incidents.",
        "dl_forecast": "Download forecast (CSV)",
        "dl_locality": "Download locality report (CSV)",
        "dl_category": "Download category data (CSV)",
        "tt_forecast": "Forecast",
        "tt_level": "Level",
        "trend_title": "Weekly Trend",
        "trend_sub": "Weekly incident counts for the most recent two years on record.",
        "records": "View weekly records",
        "year": "Year",
        "week": "Week",
        "incidents": "Incidents",
        "cat_title": "Crime Category Mix",
        "cat_sub": "Distribution of incidents by category.",
        "cat_table": "View category table",
        "category": "Category",
        "perf_title": "Model Performance",
        "perf_sub": "Validated on held-out 2025 records; trained on 2024 and earlier.",
        "best_model": "Best model (MAE)",
        "lowest_mae": "Lowest MAE",
        "mae_note": "Average absolute error",
        "rmse_note": "Penalises large errors",
        "r2_note": "Closer to 1 is better",
        "val_metrics": "Validation metrics",
        "err_compare": "Error comparison",
        "model_cfg": "Model configuration",
        "col_model": "Model",
        "col_cfg": "Configuration",
        "cfg_xgb": "140 estimators, max depth 4, learning rate 0.05",
        "cfg_rf": "140 trees, max depth 10, minimum leaf size 2",
        "cfg_knn": "StandardScaler, 7 neighbours, distance-weighted",
        "no_localities": "No localities available for the selected zone.",
        "footer": "Drishti · Crime Analytics & Forecasting · Indore, Madhya Pradesh",
    },
    "hi": {
        "brand": "दृष्टि",
        "tagline": "अपराध विश्लेषण एवं साप्ताहिक पूर्वानुमान · इंदौर, मध्य प्रदेश",
        "switch_to": "English",
        "controls_title": "पूर्वानुमान नियंत्रण",
        "tag_controls": "नियंत्रण",
        "tag_info": "जानकारी",
        "controls_desc": "पूर्वानुमान सप्ताह, मॉडल और क्षेत्र कवरेज चुनें।",
        "overview_title": "साप्ताहिक अवलोकन",
        "overview_desc": "चयनित सप्ताह और ज़ोन का पूर्वानुमान सारांश।",
        "forecast_week": "पूर्वानुमान सप्ताह",
        "model": "पूर्वानुमान मॉडल",
        "zone": "ज़ोन",
        "all_zones": "सभी ज़ोन",
        "show_names": "क्षेत्रों के नाम दिखाएँ",
        "week_of": "सप्ताह {w}, {y}",
        "kpi_localities": "निगरानी में क्षेत्र",
        "kpi_total": "अनुमानित घटनाएँ",
        "kpi_total_note": "चयनित सप्ताह का कुल",
        "kpi_top": "सर्वाधिक भार वाला क्षेत्र",
        "kpi_top_note": "{n} अनुमानित घटनाएँ",
        "kpi_elev": "उच्च स्तर के क्षेत्र",
        "kpi_elev_note": "अनुमानित भार के अनुसार शीर्ष एक-तिहाई",
        "tab_map": "हॉटस्पॉट मानचित्र",
        "tab_trend": "साप्ताहिक रुझान",
        "tab_cat": "अपराध श्रेणियाँ",
        "tab_perf": "मॉडल प्रदर्शन",
        "map_title": "हॉटस्पॉट मानचित्र",
        "map_sub": "चयनित सप्ताह के लिए क्षेत्रवार अनुमानित तीव्रता।",
        "lvl_low": "कम",
        "lvl_mod": "मध्यम",
        "lvl_high": "उच्च",
        "heat_low": "कम तीव्रता",
        "heat_high": "अधिक तीव्रता",
        "focus_title": "क्षेत्र विवरण",
        "focus_sub": "मॉडल पूर्वानुमानों की तुलना के लिए क्षेत्र चुनें।",
        "locality": "क्षेत्र",
        "mean_fc": "औसत पूर्वानुमान",
        "per_week": "/ सप्ताह",
        "consensus": "मॉडल सहमति",
        "area_n": "क्षेत्र {n}",
        "zone_n": "{z} ज़ोन",
        "lat": "अक्षांश",
        "lon": "देशांतर",
        "hist": "ऐतिहासिक कुल",
        "priority": "प्राथमिकता वाले क्षेत्र",
        "priority_sub": "अनुमानित घटनाओं के क्रम में।",
        "dl_forecast": "पूर्वानुमान डाउनलोड करें (CSV)",
        "dl_locality": "क्षेत्र रिपोर्ट डाउनलोड करें (CSV)",
        "dl_category": "श्रेणी डेटा डाउनलोड करें (CSV)",
        "tt_forecast": "पूर्वानुमान",
        "tt_level": "स्तर",
        "trend_title": "साप्ताहिक रुझान",
        "trend_sub": "उपलब्ध रिकॉर्ड के पिछले दो वर्षों की साप्ताहिक घटनाएँ।",
        "records": "साप्ताहिक रिकॉर्ड देखें",
        "year": "वर्ष",
        "week": "सप्ताह",
        "incidents": "घटनाएँ",
        "cat_title": "अपराध श्रेणी वितरण",
        "cat_sub": "श्रेणी के अनुसार घटनाओं का वितरण।",
        "cat_table": "श्रेणी तालिका देखें",
        "category": "श्रेणी",
        "perf_title": "मॉडल प्रदर्शन",
        "perf_sub": "2025 के अलग रखे गए रिकॉर्ड पर परीक्षण; 2024 तक के डेटा पर प्रशिक्षित।",
        "best_model": "सर्वश्रेष्ठ मॉडल (MAE)",
        "lowest_mae": "न्यूनतम MAE",
        "mae_note": "औसत निरपेक्ष त्रुटि",
        "rmse_note": "बड़ी त्रुटियों को अधिक महत्व देता है",
        "r2_note": "1 के जितना निकट, उतना बेहतर",
        "val_metrics": "सत्यापन मीट्रिक",
        "err_compare": "त्रुटि तुलना",
        "model_cfg": "मॉडल विन्यास",
        "col_model": "मॉडल",
        "col_cfg": "विन्यास",
        "cfg_xgb": "140 अनुमानक, अधिकतम गहराई 4, लर्निंग रेट 0.05",
        "cfg_rf": "140 वृक्ष, अधिकतम गहराई 10, न्यूनतम पर्ण आकार 2",
        "cfg_knn": "StandardScaler, 7 निकटतम पड़ोसी, दूरी-भारित",
        "no_localities": "चयनित ज़ोन के लिए कोई क्षेत्र उपलब्ध नहीं है।",
        "footer": "दृष्टि · अपराध विश्लेषण एवं पूर्वानुमान · इंदौर, मध्य प्रदेश",
    },
}

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
LANG = st.session_state["lang"]


def t(key, **kw):
    text = TR[LANG][key]
    return text.format(**kw) if kw else text


def toggle_language():
    st.session_state["lang"] = "hi" if st.session_state.get("lang", "en") == "en" else "en"


def stretch(fn):
    """Full-width kwarg that works across Streamlit versions."""
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


# ============================================================
# DESIGN SYSTEM
# Deep navy base · one shared navy surface (#1A2533) · saffron accent
# ============================================================

P = {
    "BG": "#0F141B",
    "SURFACE": "#1A2533",     # single shared surface colour (cards, KPIs, panels, tabs)
    "SURFACE2": "#233245",
    "BORDER": "#2B394D",
    "TEXT": "#ECEFF4",
    "MUTED": "#9AA7BA",
    "ACCENT": "#FF9F1C",      # orange kept exactly as before
    "ALERT": "#EF4444",
}
BG, SURFACE, SURFACE2, BORDER = P["BG"], P["SURFACE"], P["SURFACE2"], P["BORDER"]
TEXT, MUTED, ACCENT, ALERT = P["TEXT"], P["MUTED"], P["ACCENT"], P["ALERT"]

LEVEL_COLOR = {"Low": "#64748B", "Moderate": "#E2B93B", "Elevated": ALERT}
LEVEL_KEY = {"Low": "lvl_low", "Moderate": "lvl_mod", "Elevated": "lvl_high"}

HEAT_RANGE = [
    [255, 255, 178], [254, 217, 118], [254, 178, 76],
    [253, 141, 60], [240, 59, 32], [189, 0, 38],
]

ICONS = {
    "pin": '<path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0"/><circle cx="12" cy="10" r="3"/>',
    "calendar": '<path d="M8 2v4M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/>',
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 12 10 5 10-5"/><path d="m2 17 10 5 10-5"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4M12 17h.01"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9"/><path d="M15 2v2M15 20v2M9 2v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "trend": '<path d="M16 7h6v6"/><path d="m22 7-8.5 8.5-5-5L2 17"/>',
    "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
}

# Logo: navy badge, white almond-shaped eye (drishti = vision), saffron iris with scan ticks.
LOGO_SVG_SRC = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" fill="none" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M24 5 40 11v12c0 10-7 17-16 21C15 40 8 33 8 23V11z" stroke="#FFFFFF" stroke-width="2.6"/>'
    '<path d="M13.5 24C17 18.5 20.6 16.3 24 16.3s7 2.2 10.5 7.7C31 29.5 27.4 31.7 24 31.7s-7-2.2-10.5-7.7z" '
    'stroke="#FFFFFF" stroke-width="2.2"/>'
    '<circle cx="24" cy="24" r="4.6" fill="#FF9F1C" stroke="none"/>'
    '<circle cx="24" cy="24" r="1.9" fill="#0F141B" stroke="none"/>'
    '</svg>'
)
from urllib.parse import quote
LOGO_URI = "data:image/svg+xml;utf8," + quote(LOGO_SVG_SRC, safe="/:=")


def icon(name, size=18, color=ACCENT):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-3px;flex-shrink:0">{ICONS[name]}</svg>'
    )


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, button, input, select, textarea {
    font-family: 'Inter', 'Noto Sans Devanagari', sans-serif !important;
}
.stApp { background: %BG%; color: %TEXT%; }
header[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }

/* ---------- compact layout: no dead space ---------- */
.block-container { padding: 1.2rem 2rem 1rem; max-width: 1560px; }
[data-testid="stVerticalBlock"] { gap: .85rem; }
[data-testid="stHorizontalBlock"] { gap: .85rem; }
[data-testid="stElementContainer"]:has(> .stHtml:empty) { display: none; }

h1, h2, h3, h4 { color: %TEXT%; letter-spacing: -.01em; font-weight: 600; }
hr { border-color: %BORDER% !important; }

/* ---------- brand bar ---------- */
.brandbar { display: flex; align-items: center; gap: 14px; }
.logo-mark { width: 54px; height: 54px; flex: 0 0 54px; border-radius: 14px;
             display: flex; align-items: center; justify-content: center;
             background: url("%LOGO_URI%") center / 36px 36px no-repeat,
                         linear-gradient(135deg, #2F5080 0%, #16263B 100%);
             border: 1px solid rgba(255,255,255,.16); box-shadow: 0 6px 18px rgba(0,0,0,.35); }
.logo-mark svg { display: block; }
.brand-name { font-size: 1.75rem; font-weight: 700; letter-spacing: .01em; line-height: 1.1; color: %TEXT%; }
.brand-tag { color: %MUTED%; font-size: .84rem; margin-top: 3px; line-height: 1.4; }

/* ---------- inputs ---------- */
label p, [data-testid="stWidgetLabel"] p { color: %MUTED% !important; font-size: .76rem !important; font-weight: 500; }
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
[data-testid="stDateInput"] div[data-baseweb="input"] {
    background-color: %BG% !important; border-color: %BORDER% !important;
    border-radius: 10px !important; color: %TEXT% !important;
}
div[data-baseweb="select"] span, div[data-baseweb="input"] input { color: %TEXT% !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li { background: %SURFACE2% !important; color: %TEXT% !important; }
.stButton > button, .stDownloadButton > button {
    background: %SURFACE%; color: %TEXT%; border: 1px solid %BORDER%;
    border-radius: 10px; font-weight: 500; min-height: 40px;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: %ACCENT%; color: %ACCENT%; }
[data-testid="stDateInput"]:focus-within, div[data-baseweb="select"]:focus-within {
    outline: 2px solid #5B7391; outline-offset: 2px; border-radius: 10px;
}

/* ---------- control panel ---------- */
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .control-panel-heading) {
    background: %SURFACE% !important; border-radius: 16px !important;
    padding: 18px 18px 16px !important; gap: .7rem;
    border: 1px solid #34465E;
    background-image: linear-gradient(180deg, #223248 0%, #1A2533 55%) !important;
    box-shadow: 0 10px 28px rgba(0,0,0,.28);
}
.panel-heading > svg { box-sizing: content-box; padding: 8px; border-radius: 10px; background: #2A3B52; }
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .weekly-overview-heading) {
    background: #131B26 !important; border-radius: 16px !important;
    padding: 18px 18px 16px !important; border: 1px solid %BORDER%; gap: .8rem;
}
.panel-tag { margin-left: auto; font-size: .62rem; letter-spacing: .12em; font-weight: 700;
             color: %MUTED%; border: 1px solid %BORDER%; border-radius: 99px; padding: 2px 9px; }
.panel-heading { display: flex; align-items: center; gap: 9px; color: %TEXT%; font-size: .98rem; font-weight: 700; }
.panel-description { color: %MUTED%; font-size: .79rem; line-height: 1.45; margin-top: 2px; }

/* ---------- KPI ---------- */
.kpi { background: %SURFACE%; border: 1px solid %BORDER%; border-radius: 14px; padding: 14px 16px; height: 100%; min-height: 112px; }
.kpi-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.kpi-label { color: %MUTED%; font-size: .7rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; line-height: 1.4; }
.kpi-value { color: %TEXT%; font-size: 1.7rem; font-weight: 700; margin-top: 8px; overflow-wrap: anywhere; line-height: 1.15; }
.kpi-value.sm { font-size: 1.2rem; padding: 4px 0; }
.kpi-note { color: %MUTED%; font-size: .78rem; margin-top: 4px; line-height: 1.4; }

/* ---------- cards ---------- */
.sec-title { display: flex; align-items: center; gap: 9px; font-size: 1.02rem; font-weight: 600; color: %TEXT%; }
.sec-sub { color: %MUTED%; font-size: .82rem; margin: 3px 0 0; }
.sec-sub:empty { display: none; }
.card { background: %SURFACE%; border-radius: 14px; padding: 16px 18px; }
.card-name { font-size: 1.15rem; font-weight: 700; color: %TEXT%; }
.card-meta { color: %MUTED%; font-size: .8rem; margin-top: 2px; }
.big { font-size: 2.3rem; font-weight: 700; color: %ACCENT%; line-height: 1; }
.big small { font-size: .85rem; color: %MUTED%; font-weight: 500; margin-left: 6px; }
.lbl { color: %MUTED%; font-size: .7rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }
.val { color: %TEXT%; font-size: .95rem; font-weight: 600; margin-top: 2px; }
.bar { height: 6px; background: %BG%; border-radius: 99px; overflow: hidden; margin-top: 6px; }
.bar > div { height: 100%; background: %ACCENT%; border-radius: 99px; }
.mrow { display: flex; align-items: center; gap: 10px; margin-top: 10px; font-size: .82rem; color: %MUTED%; }
.mrow .nm { width: 128px; flex-shrink: 0; }
.mrow .bar { flex: 1; margin-top: 0; }
.mrow .num { width: 36px; text-align: right; color: %TEXT%; font-weight: 600; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; padding-top: 12px; border-top: 1px solid %BORDER%; }

.rank { display: flex; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid %BORDER%; }
.rank:last-child { border-bottom: 0; }
.rank-no { color: %MUTED%; font-size: .76rem; font-weight: 600; width: 20px; }
.rank-main { flex: 1; min-width: 0; }
.rank-name { color: %TEXT%; font-size: .88rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rank-val { color: %TEXT%; font-weight: 700; font-size: 1rem; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }

.legend { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; color: %MUTED%; font-size: .78rem; }
.heatbar { width: 150px; height: 8px; border-radius: 99px;
           background: linear-gradient(90deg, #FFFFB2, #FED976, #FEB24C, #FD8D3C, #F03B20, #BD0026); }

.map-snapshot { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.map-snapshot-item { background: %SURFACE%; border-radius: 12px; padding: 10px 12px; min-width: 0; }
.map-snapshot-label { color: %MUTED%; font-size: .68rem; line-height: 1.35; }
.map-snapshot-value { color: %TEXT%; font-size: .98rem; font-weight: 700; margin-top: 4px; overflow-wrap: anywhere; }

.tbl-wrap { max-height: 380px; overflow: auto; border: 1px solid %BORDER%; border-radius: 12px; background: %SURFACE%; }
.tbl { width: 100%; border-collapse: collapse; font-size: .85rem; }
.tbl th { position: sticky; top: 0; background: %SURFACE2%; color: %MUTED%; text-align: left;
          font-size: .72rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; padding: 10px 14px; }
.tbl td { padding: 10px 14px; color: %TEXT%; border-top: 1px solid %BORDER%; }

.footer { text-align: center; color: %MUTED%; font-size: .76rem; padding: 12px 0 0; border-top: 1px solid %BORDER%; margin-top: 10px; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    display: flex; gap: 6px; background: %BG%; border: 1px solid %BORDER%;
    border-radius: 14px; padding: 5px; width: 100%; max-width: 100%; margin: 0 0 6px;
}
.stTabs [data-baseweb="tab"] {
    flex: 1; justify-content: center; min-height: 42px; height: auto; border-radius: 10px;
    padding: 8px 12px; color: %MUTED%; font-weight: 600; border: 0; background: transparent;
    transition: background .18s ease, color .18s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: %TEXT%; background: %SURFACE%; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: %TEXT%; background: #1A2533; box-shadow: inset 0 -2px 0 %ACCENT%;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-testid="stTabContent"] { padding-top: 6px; }

[data-testid="stExpander"] { border: 1px solid %BORDER%; border-radius: 12px; background: %SURFACE%; }

@media (max-width: 900px) {
    .block-container { padding: 1rem .9rem; }
    .brand-name { font-size: 1.45rem; }
    .brand-tag { font-size: .75rem; }
    .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] { flex: 1 1 42%; }
    .map-snapshot { grid-template-columns: 1fr; }
}
</style>
"""
for _k, _v in P.items():
    CSS = CSS.replace(f"%{_k}%", _v)
CSS = CSS.replace("%LOGO_URI%", LOGO_URI)
st.html(CSS)


# ---------- small HTML helpers ----------

def esc(x):
    return html.escape(str(x))


def kpi(label, value, note="", icon_name="activity", small=False):
    cls = "kpi-value sm" if small else "kpi-value"
    return (
        f'<div class="kpi"><div class="kpi-top"><span class="kpi-label">{esc(label)}</span>'
        f'<span>{icon(icon_name, 17, "#9DB5D2")}</span></div>'
        f'<div class="{cls}">{esc(value)}</div>'
        f'<div class="kpi-note">{esc(note)}</div></div>'
    )


def section(icon_name, title, sub=""):
    sub_html = f'<div class="sec-sub">{esc(sub)}</div>' if sub else ""
    st.html(
        f'<div class="sec-title">{icon(icon_name, 18)}<span>{esc(title)}</span></div>{sub_html}'
    )


def html_table(df):
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False, name=None)
    )
    return (
        f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def style_chart(chart):
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor=BORDER, domainColor=BORDER,
            tickColor=BORDER, labelFont="Inter", titleFont="Inter", labelFontSize=11,
        )
        .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFont="Inter")
        .configure_header(labelColor=TEXT, titleColor=MUTED, labelFont="Inter")
    )


# ============================================================
# LOAD + VALIDATE DATA
# ============================================================

@st.cache_data
def load_data():
    areas = pd.read_csv("dataset/area_reference.csv")
    weekly = pd.read_csv("dataset/grouped_week.csv")
    categories = pd.read_csv("dataset/synthetic_category_weekly.csv")
    return areas, weekly, categories


try:
    areas, weekly, category_weekly = load_data()
except Exception as exc:
    st.error(f"Could not load the CSV files: {exc}")
    st.info("Run this app from the folder that contains app.py and the dataset folder.")
    st.stop()

required_area_columns = {"AREA", "AREA NAME", "ZONE", "LAT", "LON"}
required_week_columns = {
    "AREA", "AREA NAME", "LAT", "LON", "iso_year", "iso_week", "crime_count",
}

if not required_area_columns.issubset(areas.columns):
    st.error("area_reference.csv is missing columns: " +
             ", ".join(sorted(required_area_columns - set(areas.columns))))
    st.stop()

if not required_week_columns.issubset(weekly.columns):
    st.error("grouped_week.csv is missing columns: " +
             ", ".join(sorted(required_week_columns - set(weekly.columns))))
    st.stop()

for col in ["AREA", "LAT", "LON"]:
    areas[col] = pd.to_numeric(areas[col], errors="coerce")
for col in ["AREA", "LAT", "LON", "iso_year", "iso_week", "crime_count"]:
    weekly[col] = pd.to_numeric(weekly[col], errors="coerce")

areas = areas.dropna(subset=["AREA", "LAT", "LON", "AREA NAME", "ZONE"]).copy()
weekly = weekly.dropna(
    subset=["AREA", "LAT", "LON", "iso_year", "iso_week", "crime_count"]
).copy()

areas["AREA"] = areas["AREA"].astype(int)
weekly["AREA"] = weekly["AREA"].astype(int)
weekly["iso_year"] = weekly["iso_year"].astype(int)
weekly["iso_week"] = weekly["iso_week"].astype(int)

weekly["sin_week"] = np.sin(2 * np.pi * weekly["iso_week"] / 52)
weekly["cos_week"] = np.cos(2 * np.pi * weekly["iso_week"] / 52)

FEATURES = ["AREA", "LAT", "LON", "iso_week", "sin_week", "cos_week"]


# ============================================================
# MODEL TRAINING + HELD-OUT EVALUATION
# ============================================================

@st.cache_resource
def fit_models(data):
    data = data.copy()
    train = data[data["iso_year"] <= 2024]
    test = data[data["iso_year"] == 2025]

    if train.empty:
        raise ValueError("No training rows from 2024 or earlier were found.")
    if test.empty:
        raise ValueError("No held-out 2025 rows were found.")

    model_specs = {
        "XGBoost": XGBRegressor(
            n_estimators=140, max_depth=4, learning_rate=0.05,
            objective="reg:squarederror", random_state=42, n_jobs=2,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=140, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=2,
        ),
        "Scaled KNN Pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        ]),
    }

    metrics = {}
    for name, model in model_specs.items():
        model.fit(train[FEATURES], train["crime_count"])
        test_pred = np.maximum(0, model.predict(test[FEATURES]))
        metrics[name] = {
            "MAE": mean_absolute_error(test["crime_count"], test_pred),
            "RMSE": np.sqrt(mean_squared_error(test["crime_count"], test_pred)),
            "R²": r2_score(test["crime_count"], test_pred),
        }
        model.fit(data[FEATURES], data["crime_count"])

    return model_specs, metrics


try:
    models, metrics = fit_models(weekly)
except Exception as exc:
    st.error(f"Could not train the forecasting models: {exc}")
    st.stop()

MODEL_NAMES = ["XGBoost", "Random Forest", "Scaled KNN Pipeline"]


# ============================================================
# HEADER (brand + language switch)
# ============================================================

hb, hl = st.columns([7, 1.15], vertical_alignment="center")

with hb:
    st.html(
        f"""
        <div class="brandbar">
          <div class="logo-mark" role="img" aria-label="Drishti logo"></div>
          <div>
            <div class="brand-name">{esc(t("brand"))}</div>
            <div class="brand-tag">{esc(t("tagline"))}</div>
          </div>
        </div>
        """
    )

with hl:
    st.button(
        t("switch_to"), key="lang_btn", on_click=toggle_language,
        icon=":material/translate:", **stretch(st.button),
    )


# ============================================================
# LEFT = CONTROLS  |  RIGHT = INFORMATION (clearly separated panels)
# ============================================================

control_col, summary_col = st.columns([1, 2.1], gap="large", vertical_alignment="top")

with control_col:
    with st.container():
        st.html(
            f'<div class="panel-heading control-panel-heading">{icon("sliders", 18, "#9DB5D2")}'
            f'<span>{esc(t("controls_title"))}</span><span class="panel-tag">{esc(t("tag_controls"))}</span></div>'
            f'<div class="panel-description">{esc(t("controls_desc"))}</div>'
        )
        selected_date = st.date_input(t("forecast_week"), value=datetime.date.today(), key="week_date")
        model_name = st.selectbox(t("model"), MODEL_NAMES, key="model_sel")
        zone_options = ["__all__"] + sorted(areas["ZONE"].astype(str).unique().tolist())
        zone = st.selectbox(
            t("zone"), zone_options, key="zone_sel",
            format_func=lambda z: t("all_zones") if z == "__all__" else z,
        )
        show_names = st.toggle(t("show_names"), value=True, key="show_names")

with summary_col:
    with st.container():
        st.html(
            f'<div class="panel-heading weekly-overview-heading">{icon("activity", 18, "#9DB5D2")}'
            f'<span>{esc(t("overview_title"))}</span><span class="panel-tag">{esc(t("tag_info"))}</span></div>'
            f'<div class="panel-description">{esc(t("overview_desc"))}</div>'
        )
        k1, k2 = st.columns(2, gap="medium")
        k3, k4 = st.columns(2, gap="medium")

year, week, _ = selected_date.isocalendar()


# ============================================================
# MODEL OUTPUTS FOR EVERY AREA
# ============================================================

feat = areas[["AREA", "LAT", "LON"]].copy()
feat["iso_week"] = int(week)
feat["sin_week"] = np.sin(2 * np.pi * week / 52)
feat["cos_week"] = np.cos(2 * np.pi * week / 52)
feat = feat[FEATURES]

raw = {
    name: np.maximum(0.0, np.asarray(model.predict(feat), dtype=float))
    for name, model in models.items()
}
stack = np.vstack([raw[n] for n in MODEL_NAMES])
mean_pred = stack.mean(axis=0)
spread = stack.max(axis=0) - stack.min(axis=0)
safe_mean = np.where(mean_pred > 0, mean_pred, 1.0)
consensus_pct = np.where(
    mean_pred > 0,
    np.maximum(0.0, 100.0 * (1.0 - spread / safe_mean)),
    np.where(spread == 0, 100.0, 0.0),
)

base = areas[["AREA", "AREA NAME", "ZONE", "LAT", "LON"]].reset_index(drop=True).copy()
base["AREA NAME"] = base["AREA NAME"].astype(str)
base["ZONE"] = base["ZONE"].astype(str)

consensus = base.copy()
for name in MODEL_NAMES:
    consensus[name] = np.rint(raw[name]).astype(int)
consensus["Mean forecast"] = mean_pred
consensus["Consensus (%)"] = consensus_pct

forecast = base.copy()
forecast["forecast_count"] = np.rint(raw[model_name]).astype(int)

view = (
    forecast.copy() if zone == "__all__"
    else forecast[forecast["ZONE"] == zone].copy()
).reset_index(drop=True)

view["level"] = "Moderate"
if len(view) >= 3:
    ranks = view["forecast_count"].rank(method="first")
    view.loc[ranks <= len(view) / 3, "level"] = "Low"
    view.loc[ranks > 2 * len(view) / 3, "level"] = "Elevated"

if not view.empty:
    top = view.sort_values("forecast_count", ascending=False).iloc[0]
    top_name, top_count = str(top["AREA NAME"]), int(top["forecast_count"])
    elevated_n = int((view["level"] == "Elevated").sum())
else:
    top_name, top_count, elevated_n = "—", 0, 0


# ============================================================
# KPI VALUES (shown in the right-hand information panel)
# ============================================================

k1.html(kpi(t("kpi_localities"), f"{len(view)}", t("week_of", w=week, y=year), "pin"))
k2.html(kpi(t("kpi_total"), f'{int(view["forecast_count"].sum()):,}', t("kpi_total_note"), "activity"))
k3.html(kpi(t("kpi_top"), top_name, t("kpi_top_note", n=f"{top_count:,}"), "target", small=True))
k4.html(kpi(t("kpi_elev"), f"{elevated_n}", t("kpi_elev_note"), "alert"))


tab_map, tab_trend, tab_cat, tab_perf = st.tabs([
    f":material/map: {t('tab_map')}",
    f":material/trending_up: {t('tab_trend')}",
    f":material/category: {t('tab_cat')}",
    f":material/analytics: {t('tab_perf')}",
])


def fit_zoom(lat, lon):
    lat_span = max(float(lat.max() - lat.min()), 0.01)
    lon_span = max(float(lon.max() - lon.min()), 0.01)
    cos_lat = math.cos(math.radians(float(lat.mean())))
    z_lon = math.log2(492.0 / lon_span)
    z_lat = math.log2(352.0 * cos_lat / lat_span)
    return float(min(14.0, max(8.0, min(z_lon, z_lat) - 0.35)))


# ============================================================
# TAB 1: HOTSPOT MAP
# ============================================================

with tab_map:
    if view.empty:
        st.info(t("no_localities"))
    else:
        names = view["AREA NAME"].tolist()
        if st.session_state.get("focus_loc") not in names:
            st.session_state["focus_loc"] = names[0]

        left, right = st.columns([2.15, 1], gap="large")

        # ---- right column first, so the focus locality is known for the map ----
        with right:
            section("layers", t("focus_title"), t("focus_sub"))
            focus_name = st.selectbox(
                t("locality"), names, key="focus_loc", label_visibility="collapsed",
            )
            detail = areas[areas["AREA NAME"].astype(str) == focus_name].iloc[0]
            detail_id = int(detail["AREA"])
            sel = consensus[consensus["AREA"] == detail_id].iloc[0]
            history_total = int(weekly.loc[weekly["AREA"] == detail_id, "crime_count"].sum())

            vmax = max(1, int(max(sel[n] for n in MODEL_NAMES)))
            mrows = "".join(
                f'<div class="mrow"><span class="nm">{esc(n)}</span>'
                f'<div class="bar"><div style="width:{int(sel[n]) / vmax * 100:.0f}%"></div></div>'
                f'<span class="num">{int(sel[n]):,}</span></div>'
                for n in MODEL_NAMES
            )
            st.html(
                f"""
                <div class="card">
                  <div class="card-name">{esc(focus_name)}</div>
                  <div class="card-meta">{esc(t("zone_n", z=detail["ZONE"]))} · {esc(t("area_n", n=detail_id))}</div>
                  <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:14px;margin-top:14px">
                    <div>
                      <div class="lbl">{esc(t("mean_fc"))}</div>
                      <div class="big" style="margin-top:6px">{sel["Mean forecast"]:.0f}<small>{esc(t("per_week"))}</small></div>
                    </div>
                    <div style="width:46%">
                      <div class="lbl">{esc(t("consensus"))} · {sel["Consensus (%)"]:.0f}%</div>
                      <div class="bar"><div style="width:{sel["Consensus (%)"]:.0f}%"></div></div>
                    </div>
                  </div>
                  <div style="margin-top:4px">{mrows}</div>
                  <div class="grid3">
                    <div><div class="lbl">{esc(t("lat"))}</div><div class="val">{float(detail["LAT"]):.4f}</div></div>
                    <div><div class="lbl">{esc(t("lon"))}</div><div class="val">{float(detail["LON"]):.4f}</div></div>
                    <div><div class="lbl">{esc(t("hist"))}</div><div class="val">{history_total:,}</div></div>
                  </div>
                </div>
                """
            )

            section("alert", t("priority"), t("priority_sub"))
            ranked = view.sort_values("forecast_count", ascending=False).head(6)
            rmax = max(1, int(ranked["forecast_count"].max()))
            rows = ""
            for i, (_, row) in enumerate(ranked.iterrows(), start=1):
                lvl = str(row["level"])
                rows += (
                    f'<div class="rank"><div class="rank-no">{i:02d}</div>'
                    f'<div class="rank-main"><div class="rank-name">{esc(row["AREA NAME"])}</div>'
                    f'<div class="bar"><div style="width:{int(row["forecast_count"]) / rmax * 100:.0f}%;'
                    f'background:{LEVEL_COLOR.get(lvl, MUTED)}"></div></div></div>'
                    f'<div class="rank-val">{int(row["forecast_count"]):,}</div></div>'
                )
            st.html(f'<div class="card" style="padding-top:6px;padding-bottom:6px">{rows}</div>')

            dl1, dl2 = st.columns(2)
            export_view = view.drop(columns=["level"]).rename(columns={"forecast_count": "forecast"})
            dl1.download_button(
                t("dl_forecast"),
                data=export_view.to_csv(index=False).encode("utf-8"),
                file_name="drishti_forecast.csv", mime="text/csv",
                icon=":material/download:", key="dl_fc", **stretch(st.download_button),
            )
            export = pd.DataFrame([{
                "Locality": detail["AREA NAME"],
                "Area Code": f"AREA {detail_id}",
                "Zone": detail["ZONE"],
                "Latitude": float(detail["LAT"]),
                "Longitude": float(detail["LON"]),
                "Forecast Week": week,
                "Forecast Year": year,
                "XGBoost": int(sel["XGBoost"]),
                "Random Forest": int(sel["Random Forest"]),
                "Scaled KNN Pipeline": int(sel["Scaled KNN Pipeline"]),
                "Mean Forecast": round(float(sel["Mean forecast"]), 2),
                "Model Consensus (%)": round(float(sel["Consensus (%)"]), 2),
                "Historical Total": history_total,
            }])
            dl2.download_button(
                t("dl_locality"),
                data=export.to_csv(index=False).encode("utf-8"),
                file_name="drishti_locality_report.csv", mime="text/csv",
                icon=":material/download:", key="dl_loc", **stretch(st.download_button),
            )

        # ---- map ----
        with left:
            section("pin", t("map_title"), t("map_sub"))

            map_data = pd.DataFrame({
                "name": view["AREA NAME"],
                "zone": view["ZONE"],
                "LAT": view["LAT"],
                "LON": view["LON"],
                "forecast_count": view["forecast_count"],
                "level_label": view["level"].map(lambda v: t(LEVEL_KEY[v])),
            })
            focus_data = map_data[map_data["name"] == focus_name].head(1)

            layers = [
                pdk.Layer(
                    "HeatmapLayer", data=map_data,
                    get_position="[LON, LAT]", get_weight="forecast_count",
                    radius_pixels=42, intensity=1, threshold=0.05, opacity=0.9,
                    color_range=HEAT_RANGE, pickable=False,
                ),
                pdk.Layer(
                    "ScatterplotLayer", data=map_data,
                    get_position="[LON, LAT]", get_radius=60,
                    radius_min_pixels=3, radius_max_pixels=5,
                    get_fill_color="[236, 239, 244, 150]",
                    pickable=True, auto_highlight=True,
                ),
                pdk.Layer(
                    "ScatterplotLayer", data=focus_data,
                    get_position="[LON, LAT]", get_radius=80,
                    radius_min_pixels=7, radius_max_pixels=9,
                    get_fill_color="[56, 189, 248, 255]",
                    get_line_color="[255, 255, 255, 255]",
                    stroked=True, line_width_min_pixels=2, pickable=False,
                ),
            ]
            if show_names:
                layers.append(
                    pdk.Layer(
                        "TextLayer", data=map_data,
                        get_position="[LON, LAT]", get_text="name", get_size=13,
                        get_color="[236, 239, 244, 255]",
                        get_alignment_baseline="'top'", get_text_anchor="'middle'",
                        get_pixel_offset="[0, 12]",
                        font_family="'Inter', 'Noto Sans Devanagari', sans-serif",
                        font_weight=600, outline_width=4,
                        outline_color=[14, 17, 22, 255],
                        font_settings={"sdf": True}, billboard=True, pickable=False,
                    )
                )

            deck = pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(
                    latitude=float(map_data["LAT"].mean()),
                    longitude=float(map_data["LON"].mean()),
                    zoom=fit_zoom(map_data["LAT"], map_data["LON"]),
                    pitch=0, bearing=0,
                ),
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                tooltip={
                    "html": (
                        "<b>{name}</b><br/>{zone}<br/>"
                        + t("tt_forecast") + ": <b>{forecast_count}</b><br/>"
                        + t("tt_level") + ": {level_label}"
                    ),
                    "style": {
                        "backgroundColor": SURFACE2, "color": TEXT,
                        "border": f"1px solid {BORDER}", "borderRadius": "8px",
                        "fontFamily": "Inter, 'Noto Sans Devanagari', sans-serif",
                        "fontSize": "12px",
                    },
                },
            )
            st.pydeck_chart(deck, height=520, use_container_width=True)

            lvl_html = "".join(
                f'<span><span class="dot" style="background:{LEVEL_COLOR[k]}"></span>{esc(t(LEVEL_KEY[k]))}</span>'
                for k in ("Low", "Moderate", "Elevated")
            )
            st.html(
                f"""
                <div class="legend">
                  <span style="display:flex;align-items:center;gap:8px">
                    {esc(t("heat_low"))}<span class="heatbar"></span>{esc(t("heat_high"))}
                  </span>
                  {lvl_html}
                </div>
                <div class="map-snapshot" style="margin-top:12px">
                  <div class="map-snapshot-item">
                    <div class="map-snapshot-label">{esc(t("forecast_week"))}</div>
                    <div class="map-snapshot-value">{esc(t("week_of", w=week, y=year))}</div>
                  </div>
                  <div class="map-snapshot-item">
                    <div class="map-snapshot-label">{esc(t("kpi_localities"))}</div>
                    <div class="map-snapshot-value">{len(view):,}</div>
                  </div>
                  <div class="map-snapshot-item">
                    <div class="map-snapshot-label">{esc(t("kpi_top"))}</div>
                    <div class="map-snapshot-value">{esc(top_name)} · {top_count:,}</div>
                  </div>
                </div>
                """
            )


# ============================================================
# TAB 2: WEEKLY TRENDS
# ============================================================

with tab_trend:
    section("trend", t("trend_title"), t("trend_sub"))

    if view.empty:
        st.info(t("no_localities"))
    else:
        t_names = view["AREA NAME"].tolist()
        if st.session_state.get("trend_loc") not in t_names:
            st.session_state["trend_loc"] = t_names[0]
        tc, _ = st.columns([1, 2])
        chosen = tc.selectbox(t("locality"), t_names, key="trend_loc")
        chosen_id = int(view.loc[view["AREA NAME"] == chosen, "AREA"].iloc[0])

        td = weekly[weekly["AREA"] == chosen_id].copy()
        td["date"] = pd.to_datetime(
            td["iso_year"].astype(str) + "-" + td["iso_week"].astype(str).str.zfill(2) + "-1",
            format="%G-%V-%u", errors="coerce",
        )
        td = td.dropna(subset=["date"]).sort_values("date").tail(104)

        if not td.empty:
            chart = (
                alt.Chart(td)
                .mark_area(
                    color=ACCENT, opacity=0.14, interpolate="monotone",
                    line={"color": ACCENT, "strokeWidth": 2},
                )
                .encode(
                    x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", grid=False)),
                    y=alt.Y("crime_count:Q", title=None, scale=alt.Scale(zero=True)),
                    tooltip=[
                        alt.Tooltip("date:T", title=t("week"), format="%d %b %Y"),
                        alt.Tooltip("crime_count:Q", title=t("incidents")),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(style_chart(chart), theme=None, **stretch(st.altair_chart))

            with st.expander(t("records")):
                rec = td.sort_values("date", ascending=False)[["iso_year", "iso_week", "crime_count"]]
                rec.columns = [t("year"), t("week"), t("incidents")]
                st.html(html_table(rec))


# ============================================================
# TAB 3: CRIME CATEGORIES
# ============================================================

with tab_cat:
    section("layers", t("cat_title"), t("cat_sub"))

    if {"crime_category", "crime_count"}.issubset(category_weekly.columns):
        summary = (
            category_weekly.groupby("crime_category", as_index=False)["crime_count"]
            .sum().sort_values("crime_count", ascending=False)
        )
        chart = (
            alt.Chart(summary)
            .mark_bar(color=ACCENT, cornerRadiusEnd=4, size=22)
            .encode(
                y=alt.Y("crime_category:N", sort="-x", title=None,
                        axis=alt.Axis(labelLimit=260, grid=False)),
                x=alt.X("crime_count:Q", title=None),
                tooltip=[
                    alt.Tooltip("crime_category:N", title=t("category")),
                    alt.Tooltip("crime_count:Q", title=t("incidents"), format=","),
                ],
            )
            .properties(height=max(220, 38 * len(summary)))
        )
        st.altair_chart(style_chart(chart), theme=None, **stretch(st.altair_chart))

        with st.expander(t("cat_table")):
            ct = summary.copy()
            ct["crime_count"] = ct["crime_count"].map(lambda v: f"{int(v):,}")
            ct.columns = [t("category"), t("incidents")]
            st.html(html_table(ct))

        st.download_button(
            t("dl_category"),
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="drishti_categories.csv", mime="text/csv",
            icon=":material/download:", key="dl_cat",
        )
    else:
        st.error("Expected crime_category and crime_count columns in the category dataset.")


# ============================================================
# TAB 4: MODEL PERFORMANCE
# ============================================================

with tab_perf:
    section("cpu", t("perf_title"), t("perf_sub"))

    metric_df = pd.DataFrame(metrics).T.reset_index(names="Model")
    best = metric_df.sort_values("MAE").iloc[0]

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.html(kpi(t("best_model"), best["Model"], "", "cpu", small=True))
    m2.html(kpi(t("lowest_mae"), f'{best["MAE"]:.2f}', t("mae_note"), "target"))
    m3.html(kpi("RMSE", f'{best["RMSE"]:.2f}', t("rmse_note"), "activity"))
    m4.html(kpi("R²", f'{best["R²"]:.3f}', t("r2_note"), "trend"))

    a, b = st.columns([1, 1.1], gap="large")
    with a:
        st.html(f'<div class="sec-title"><span>{esc(t("val_metrics"))}</span></div>')
        shown = metric_df.copy()
        for c in ["MAE", "RMSE", "R²"]:
            shown[c] = shown[c].map(lambda v: f"{v:.3f}")
        shown = shown.rename(columns={"Model": t("col_model")})
        st.html(html_table(shown))
    with b:
        st.html(f'<div class="sec-title"><span>{esc(t("err_compare"))}</span></div>')
        long_df = metric_df.melt(id_vars="Model", value_vars=["MAE", "RMSE"],
                                 var_name="Metric", value_name="Value")
        chart = (
            alt.Chart(long_df)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                column=alt.Column("Model:N", title=None, header=alt.Header(labelFontSize=11)),
                x=alt.X("Metric:N", title=None, axis=None),
                y=alt.Y("Value:Q", title=None),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(domain=["MAE", "RMSE"], range=[ACCENT, "#64748B"]),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=["Model:N", "Metric:N", alt.Tooltip("Value:Q", format=".3f")],
            )
            .properties(width=70, height=260)
        )
        st.altair_chart(style_chart(chart), theme=None)

    st.html(f'<div class="sec-title"><span>{esc(t("model_cfg"))}</span></div>')
    cfg = pd.DataFrame([
        {t("col_model"): "XGBoost", t("col_cfg"): t("cfg_xgb")},
        {t("col_model"): "Random Forest", t("col_cfg"): t("cfg_rf")},
        {t("col_model"): "Scaled KNN Pipeline", t("col_cfg"): t("cfg_knn")},
    ])
    st.html(html_table(cfg))


# ============================================================
# FOOTER
# ============================================================

st.html(f'<div class="footer">{esc(t("footer"))}</div>')