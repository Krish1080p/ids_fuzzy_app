"""
styles.py - everything about the look & feel of the application.

Contents (in order):
  1. Neon colour palette
  2. Global CSS injected into the Streamlit page
  3. Small HTML helpers (hero banner, glass card)
  4. Plotly helpers (dark neon theme, gauge, confusion matrix, membership plots)
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fuzzy_logic import OUT_MF, UNIVERSE, VARIABLES, membership


# =============================================================================
# 1. COLOUR PALETTE
# =============================================================================


# Neon palette
CYAN, BLUE, PURPLE = "#00f5ff", "#3b82f6", "#a855f7"
GREEN, AMBER, RED = "#22ffaa", "#ffb020", "#ff3b6b"
CLASS_COLORS = {"Normal": GREEN, "Suspicious": AMBER, "Malicious": RED}


# =============================================================================
# 2. GLOBAL CSS
# =============================================================================


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', Roboto, sans-serif; }
.stApp { background: radial-gradient(circle at 12% 8%, rgba(59,130,246,0.18), transparent 38%),
                     radial-gradient(circle at 90% 15%, rgba(168,85,247,0.18), transparent 40%),
                     linear-gradient(160deg, #02040c 0%, #070d22 55%, #0a1030 100%); color: #dff3ff; }
.block-container { max-width: 1500px; padding-top: 1.6rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(5,10,28,0.97), rgba(10,14,40,0.97));
        border-right: 1px solid rgba(0,245,255,0.25); box-shadow: 0 0 30px rgba(0,245,255,0.08); }
h1, h2, h3 { color: #eaf8ff; letter-spacing: 0.3px; }
.hero { padding: 22px 28px; border-radius: 20px; margin-bottom: 18px;
        background: linear-gradient(120deg, rgba(0,245,255,0.10), rgba(168,85,247,0.14));
        border: 1px solid rgba(0,245,255,0.35); box-shadow: 0 0 28px rgba(0,245,255,0.15);
        backdrop-filter: blur(14px); }
.hero h1 { margin: 0; font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #00f5ff, #3b82f6, #a855f7); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; }
.hero p { margin: 6px 0 0 0; color: #a9c7e8; }
.glass { background: rgba(255,255,255,0.045); border: 1px solid rgba(120,200,255,0.25); border-radius: 16px;
         padding: 16px 20px; margin: 8px 0 14px 0; backdrop-filter: blur(12px);
         box-shadow: 0 6px 24px rgba(0,0,0,0.45), inset 0 0 18px rgba(0,245,255,0.04); }
.glass b { color: #00f5ff; }
.sim { border-left: 4px solid #ffb020; }
.bad { border-left: 4px solid #ff3b6b; }
.good { border-left: 4px solid #22ffaa; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.05); border: 1px solid rgba(0,245,255,0.3);
        border-radius: 14px; padding: 12px 16px; box-shadow: 0 0 16px rgba(0,245,255,0.10); }
div[data-testid="stMetricValue"] { color: #00f5ff; }
div[data-testid="stMetricValue"] > div { font-size: 1.6rem; overflow: visible; text-overflow: clip; white-space: nowrap; }
div[data-testid="stMetricLabel"] p { font-size: 0.8rem; white-space: normal; }
div[data-testid="stPlotlyChart"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(120,160,255,0.22);
        border-radius: 16px; padding: 6px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
.stButton > button, .stDownloadButton > button { background: linear-gradient(90deg, #0891b2, #3b82f6, #7c3aed);
        color: white; border: 1px solid rgba(0,245,255,0.5); border-radius: 12px; font-weight: 600;
        box-shadow: 0 0 14px rgba(0,245,255,0.3); }
.stButton > button:hover { box-shadow: 0 0 24px rgba(0,245,255,0.7); border-color: #00f5ff; }
button[data-baseweb="tab"] { color: #9fc4e8; }
button[data-baseweb="tab"][aria-selected="true"] { color: #00f5ff; }
div[data-testid="stExpander"] { background: rgba(255,255,255,0.035); border: 1px solid rgba(120,200,255,0.2);
        border-radius: 14px; }
.tag { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem; margin-right:6px;
       border:1px solid rgba(0,245,255,0.5); color:#00f5ff; }
</style>
"""


# =============================================================================
# 3. HTML HELPERS
# =============================================================================


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(html, kind=""):
    st.markdown(f'<div class="glass {kind}">{html}</div>', unsafe_allow_html=True)


# =============================================================================
# 4. PLOTTING HELPERS (neon dark theme)
# =============================================================================


def style(fig, title, height=380, xtitle=None, ytitle=None):
    fig.update_layout(
        template="plotly_dark", title=dict(text=title, font=dict(size=16, color="#e6f7ff")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,18,40,0.35)", height=height,
        margin=dict(l=50, r=20, t=55, b=45), font=dict(color="#cfe9ff"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor="rgba(120,160,255,0.12)", title=xtitle)
    fig.update_yaxes(gridcolor="rgba(120,160,255,0.12)", title=ytitle)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


def gauge(value, title, t_normal, t_mal):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), title={"text": title, "font": {"size": 16}},
        number={"suffix": " /100", "font": {"color": "#e6f7ff"}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": CYAN},
               "bgcolor": "rgba(0,0,0,0)",
               "steps": [{"range": [0, t_normal], "color": "rgba(34,255,170,0.35)"},
                         {"range": [t_normal, t_mal], "color": "rgba(255,176,32,0.35)"},
                         {"range": [t_mal, 100], "color": "rgba(255,59,107,0.40)"}]}))
    fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cfe9ff"),
                      margin=dict(l=25, r=25, t=60, b=10))
    return fig


def confusion_figure(c, title):
    z = [[c["TN"], c["FP"]], [c["FN"], c["TP"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted Normal", "Predicted Attack"], y=["Actual Normal", "Actual Attack"],
                               colorscale=[[0, "#0b1230"], [1, CYAN]], showscale=False,
                               text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}",
                               textfont=dict(size=20, color="white")))
    fig.update_yaxes(autorange="reversed")
    return style(fig, title, 330)


def membership_figure(var, mfs, marker=None):
    cfg = VARIABLES[var]
    xs = np.linspace(0, 1, 400)
    xr = xs * (np.log10(1 + cfg["cap"]) if cfg["log"] else cfg["cap"])
    fig = go.Figure()
    for (name, (k, p)), col in zip(mfs[var].items(), [CYAN, PURPLE, AMBER, GREEN]):
        fig.add_trace(go.Scatter(x=xr, y=membership(k, p, xs), name=f"{name} ({k})", mode="lines",
                                 line=dict(color=col, width=3), fill="tozeroy"))
    if marker is not None:
        fig.add_vline(x=marker, line_dash="dash", line_color="#ffffff")
    unit = "log10(1 + bytes)" if cfg["log"] else "raw value"
    return style(fig, f"Membership functions: {cfg['label']}", 330, xtitle=f"{cfg['label']} ({unit})",
                 ytitle="Membership degree")


def output_figure():
    fig = go.Figure()
    for (name, (k, p)), col in zip(OUT_MF.items(), [GREEN, AMBER, RED]):
        fig.add_trace(go.Scatter(x=UNIVERSE, y=membership(k, p, UNIVERSE), name=f"{name} ({k})",
                                 line=dict(color=col, width=3), fill="tozeroy"))
    return style(fig, "Mamdani output sets (risk score)", 330, xtitle="Risk score", ytitle="Membership degree")
