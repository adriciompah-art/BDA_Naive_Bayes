# =========================================================
# INSTALL:  pip install streamlit openpyxl scikit-learn
#           pandas numpy seaborn matplotlib pyngrok
# RUN      :  streamlit run ona_new.py
# COLAB    :  Salin isi file ini ke ona.py di Colab, lalu
#             jalankan dengan pyngrok seperti semula.
# =========================================================

import io
import base64

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Penentuan Keluarga Miskin – Naive Bayes",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS  —  Warm Earth & Terracotta Theme
#               (layout & feature pattern dari contoh.py)
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Nunito:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── COLOUR TOKENS ──────────────────────────────────── */
    :root {
        --bg:           #FDF6EE;
        --surface:      #FFFFFF;
        --surface2:     #FEF9F4;
        --surface3:     #FAF0E6;
        --border:       #E8D5C0;
        --border2:      #D4B896;

        --brown:        #3D1C02;
        --brown-mid:    #6B3A1F;
        --brown-light:  #9B5E2E;
        --terra:        #C2570A;
        --terra-light:  #E07030;
        --terra-pale:   #FDECD8;
        --gold:         #C49A1A;
        --gold-pale:    #FEF3C7;
        --sage:         #3D7054;
        --sage-pale:    #D4EDDA;
        --rose:         #C0392B;
        --rose-pale:    #FDECEA;

        --text:         #1A0A00;
        --text-body:    #3D2B1F;
        --text-dim:     #7A5C44;
        --text-muted:   #A88B78;

        --shadow-sm:    0 1px 3px rgba(61,28,2,.08), 0 1px 2px rgba(61,28,2,.06);
        --shadow-md:    0 4px 16px rgba(61,28,2,.10), 0 2px 6px rgba(61,28,2,.07);
        --shadow-lg:    0 10px 40px rgba(61,28,2,.13), 0 4px 12px rgba(61,28,2,.08);
        --radius:       10px;
        --radius-lg:    16px;
        --radius-xl:    22px;
    }

    /* ── BASE ──────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
        background-color: var(--bg);
        color: var(--text-body);
        font-size: 14px;
        line-height: 1.65;
    }
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 100% !important;
    }

    /* ── SIDEBAR ───────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(170deg, #1A0A00 0%, #3D1C02 40%, #2A0E00 100%) !important;
        border-right: 1px solid rgba(194,87,10,0.20) !important;
        box-shadow: 4px 0 32px rgba(26,10,0,0.40) !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 0 !important; background: transparent !important; }
    section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.82) !important; font-family: 'Nunito', sans-serif !important; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(194,87,10,0.08) !important;
        border: 2px dashed rgba(224,112,48,0.45) !important;
        border-radius: 12px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(224,112,48,0.80) !important;
        background: rgba(194,87,10,0.14) !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #C2570A, #E07030) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        font-size: 0.82rem !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 4px 16px rgba(194,87,10,0.40) !important;
        transition: all 0.22s !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #E07030, #FF9050) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(194,87,10,0.55) !important;
    }

    /* ── SIDEBAR LOGO ───────────────────────────────────── */
    .sb-logo {
        padding: 1.6rem 1.4rem 1.2rem;
        background: linear-gradient(135deg, rgba(194,87,10,0.25), rgba(61,28,2,0.12));
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.5rem;
    }
    .sb-logo-row { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.6rem; }
    .sb-logo-icon {
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #3D1C02, #C2570A);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 16px rgba(194,87,10,0.40);
        flex-shrink: 0;
    }
    .sb-logo-title { font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #FFFFFF !important; line-height: 1.1; }
    .sb-logo-title span { color: #E07030 !important; }
    .sb-logo-desc { font-size: 0.68rem !important; color: rgba(255,255,255,0.38) !important; letter-spacing: 0.3px; }

    /* ── SIDEBAR SECTION LABEL ──────────────────────────── */
    .sb-sec {
        font-size: 0.62rem !important;
        font-weight: 800 !important;
        letter-spacing: 2.2px !important;
        text-transform: uppercase !important;
        color: #E07030 !important;
        margin: 1.4rem 0 0.6rem !important;
        display: block;
        padding-left: 0.1rem;
    }

    /* ── PIPELINE STEPS ─────────────────────────────────── */
    .pipeline { display: flex; flex-direction: column; gap: 0.3rem; }
    .pipeline-step {
        display: flex; align-items: center; gap: 0.65rem;
        padding: 0.55rem 0.9rem;
        border-radius: 10px;
        border: 1px solid transparent;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: rgba(255,255,255,0.28) !important;
        transition: all 0.2s;
    }
    .pipeline-step.done {
        background: rgba(194,87,10,0.18) !important;
        border-color: rgba(224,112,48,0.40) !important;
        color: #E07030 !important;
    }
    .pipeline-step.active {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.18) !important;
        color: rgba(255,255,255,0.85) !important;
    }
    .step-num {
        width: 22px; height: 22px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.68rem; font-weight: 800;
        flex-shrink: 0;
        background: rgba(255,255,255,0.08);
        color: rgba(255,255,255,0.30) !important;
    }
    .step-num.done { background: #C2570A; color: #fff !important; box-shadow: 0 0 10px rgba(194,87,10,0.55); }
    .step-num.active { background: rgba(224,112,48,0.45); color: #fff !important; }

    /* ── INLINE PAGE TITLE BAR ──────────────────────────── */
    .page-titlebar {
        display: flex; align-items: center; gap: 1rem;
        background: linear-gradient(110deg, #1A0A00 0%, #3D1C02 55%, #2A0E00 100%);
        border-radius: var(--radius-lg);
        padding: 1rem 1.6rem;
        margin-bottom: 1.4rem;
        border: 1px solid rgba(194,87,10,0.25);
        box-shadow: 0 4px 24px rgba(26,10,0,0.22), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative; overflow: hidden;
    }
    .page-titlebar::before {
        content: '';
        position: absolute; inset: 0;
        background: radial-gradient(ellipse 50% 100% at 90% 50%, rgba(194,87,10,0.12) 0%, transparent 65%);
        pointer-events: none;
    }
    .ptb-badge {
        width: 42px; height: 42px; flex-shrink: 0;
        background: linear-gradient(135deg, #3D1C02, #C2570A);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
        box-shadow: 0 3px 14px rgba(194,87,10,0.42);
        position: relative; z-index: 1;
    }
    .ptb-text { position: relative; z-index: 1; flex: 1; }
    .ptb-eyebrow {
        font-size: 0.58rem; font-weight: 800; letter-spacing: 2.5px;
        text-transform: uppercase; color: #E07030; margin-bottom: 0.15rem;
    }
    .ptb-title {
        font-family: 'Playfair Display', serif; font-size: 1.22rem; font-weight: 700;
        color: #FFFFFF; line-height: 1.1; letter-spacing: -0.2px;
    }
    .ptb-title .hl { color: #FFAD55; font-style: italic; }
    .ptb-pills {
        margin-left: auto; display: flex; gap: 0.4rem; align-items: center;
        position: relative; z-index: 1; flex-wrap: wrap;
    }
    .hpill {
        border-radius: 999px; padding: 4px 12px;
        font-size: 0.68rem; font-weight: 800; letter-spacing: 0.3px;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .hpill-terra  { background: rgba(194,87,10,0.28); color: #FFAD55 !important; }
    .hpill-brown  { background: rgba(61,28,2,0.50);   color: #D4A06A !important; }
    .hpill-gold   { background: rgba(196,154,26,0.28); color: #FFD97A !important; }

    /* ── NOTIFICATION BANNERS ───────────────────────────── */
    .notif-box {
        border-radius: var(--radius);
        padding: 1rem 1.3rem;
        margin: 0.8rem 0 1.2rem;
        display: flex; align-items: flex-start; gap: 0.9rem;
        border: 1px solid;
        animation: slideIn 0.35s ease;
    }
    @keyframes slideIn {
        from { opacity:0; transform: translateY(-8px); }
        to   { opacity:1; transform: translateY(0); }
    }
    .notif-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 0.1rem; }
    .notif-content { flex: 1; }
    .notif-title { font-weight: 800; font-size: 0.88rem; margin-bottom: 0.2rem; }
    .notif-body { font-size: 0.8rem; line-height: 1.6; }
    .notif-stats { display: flex; gap: 0.6rem; margin-top: 0.6rem; flex-wrap: wrap; }
    .notif-stat {
        border-radius: 8px; padding: 4px 12px;
        font-size: 0.73rem; font-weight: 700;
        display: inline-flex; align-items: center; gap: 5px;
    }
    .notif-upload { background: linear-gradient(135deg, #FEF9F4, #FEF0E0); border-color: rgba(194,87,10,0.22); }
    .notif-upload .notif-title { color: #8B3A00; }
    .notif-upload .notif-stat { background: rgba(194,87,10,0.10); color: #8B3A00; }
    .notif-prep { background: linear-gradient(135deg, #F0FAF4, #E2F8EC); border-color: rgba(61,112,84,0.28); }
    .notif-prep .notif-title { color: #2A5C3A; }
    .notif-prep .notif-stat { background: rgba(61,112,84,0.10); color: #2A5C3A; }
    .notif-nb { background: linear-gradient(135deg, #FEF6EC, #FEE9D4); border-color: rgba(196,154,26,0.28); }
    .notif-nb .notif-title { color: #6B4A00; }
    .notif-nb .notif-stat { background: rgba(196,154,26,0.12); color: #6B4A00; }
    .notif-warn { background: linear-gradient(135deg, #FFF8F0, #FFF0E0); border-color: rgba(192,57,43,0.22); }
    .notif-warn .notif-title { color: #7A1500; }
    .notif-warn .notif-stat { background: rgba(192,57,43,0.10); color: #7A1500; }

    /* ── METRIC CARDS ──────────────────────────────────── */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 140px;
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: 1.3rem 1.4rem 1.1rem;
        position: relative; overflow: hidden;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border);
        transition: transform 0.22s, box-shadow 0.22s;
    }
    .metric-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-md); }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0;
        height: 4px; border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }
    .mc-brown::before   { background: linear-gradient(90deg, #3D1C02, #6B3A1F); }
    .mc-terra::before   { background: linear-gradient(90deg, #C2570A, #E07030); }
    .mc-gold::before    { background: linear-gradient(90deg, #C49A1A, #E8B528); }
    .mc-sage::before    { background: linear-gradient(90deg, #3D7054, #4E9068); }
    .mc-violet::before  { background: linear-gradient(90deg, #7C3AED, #9F67FF); }
    .mc-rose::before    { background: linear-gradient(90deg, #C0392B, #E74C3C); }

    .mc-icon { font-size: 1.3rem; margin-bottom: 0.55rem; display: block; }
    .mc-label { font-size: 0.64rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700; margin-bottom: 0.3rem; }
    .mc-value { font-family: 'Playfair Display', serif; font-size: 2.3rem; line-height: 1; }
    .cv-brown  { color: #3D1C02; }
    .cv-terra  { color: #C2570A; }
    .cv-gold   { color: #C49A1A; }
    .cv-sage   { color: #3D7054; }
    .cv-violet { color: #7C3AED; }
    .cv-rose   { color: #C0392B; }

    /* ── SECTION HEADER ─────────────────────────────────── */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem; font-weight: 700;
        color: var(--brown);
        padding: 0.4rem 0 0.4rem 1rem;
        border-left: 4px solid var(--terra);
        margin-bottom: 1.2rem; line-height: 1.2;
        background: linear-gradient(90deg, rgba(194,87,10,0.06), transparent);
        border-radius: 0 8px 8px 0;
    }

    /* ── INFO BOX ──────────────────────────────────────── */
    .info-box {
        background: linear-gradient(135deg, #FEF9F4, #FEF0E4);
        border: 1px solid rgba(194,87,10,0.16);
        border-left: 3px solid #C2570A;
        border-radius: var(--radius);
        padding: 0.9rem 1.2rem; margin-bottom: 1.1rem;
        font-size: 0.83rem; color: #3D2B1F; line-height: 1.7;
    }

    /* ── FORMULA BOX ────────────────────────────────────── */
    .formula-box {
        background: linear-gradient(135deg, #FEF9F4, #FEF3E0);
        border: 1px solid rgba(196,154,26,0.22);
        border-left: 3px solid #C49A1A;
        border-radius: var(--radius);
        padding: 1rem 1.3rem; margin: 0.9rem 0;
        font-size: 0.82rem; color: var(--text-body); line-height: 1.9;
    }
    .formula-box code {
        font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;
        background: rgba(196,154,26,0.12); border-radius: 5px;
        padding: 0.12rem 0.45rem; color: #7A5000;
    }
    .formula-title {
        font-weight: 800; font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 1.8px; color: #C49A1A; margin-bottom: 0.6rem;
    }

    /* ── SPLIT CONFIG BOX ───────────────────────────────── */
    .split-config-box {
        background: linear-gradient(135deg, #1A0A00, #2A1000);
        border: 1px solid rgba(224,112,48,0.25);
        border-radius: var(--radius-lg);
        padding: 1.4rem 1.6rem; margin: 1rem 0 1.4rem;
        position: relative; overflow: hidden;
    }
    .split-config-box::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #3D1C02, #C2570A, #C49A1A);
    }
    .split-config-title { font-family: 'Playfair Display', serif; font-size: 1rem; color: #FFFFFF; margin-bottom: 0.3rem; }
    .split-config-desc { font-size: 0.74rem; color: rgba(255,255,255,0.45); margin-bottom: 1.1rem; }
    .split-bar-wrap { background: rgba(255,255,255,0.06); border-radius: 999px; height: 10px; overflow: hidden; margin-bottom: 0.5rem; }
    .split-bar-train { height: 100%; border-radius: 999px; transition: width 0.4s ease; }
    .split-bar-labels { display: flex; justify-content: space-between; font-size: 0.68rem; font-weight: 700; }
    .split-label-train { color: #E07030; }
    .split-label-test  { color: #C49A1A; }

    /* ── TABS ──────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface); border-bottom: 2px solid var(--border);
        gap: 0; padding: 0 0.5rem;
        border-radius: var(--radius) var(--radius) 0 0;
        box-shadow: var(--shadow-sm);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: var(--text-muted);
        font-family: 'Nunito', sans-serif; font-weight: 700;
        font-size: 0.76rem; padding: 0.9rem 1.1rem;
        border: none; border-bottom: 3px solid transparent;
        margin-bottom: -2px; transition: color 0.2s, border-color 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #3D1C02; }
    .stTabs [aria-selected="true"] {
        background: transparent !important; color: #C2570A !important;
        border-bottom: 3px solid #C2570A !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: var(--surface); border: 1px solid var(--border);
        border-top: none; border-radius: 0 0 var(--radius) var(--radius);
        padding: 1.8rem 2rem;
    }

    /* ── DATAFRAME ─────────────────────────────────────── */
    .stDataFrame { border-radius: var(--radius) !important; overflow: hidden !important; border: 1px solid var(--border) !important; box-shadow: var(--shadow-sm) !important; }
    .stDataFrame thead th { background: linear-gradient(90deg, #1A0A00, #3D1C02) !important; color: rgba(255,255,255,0.92) !important; font-weight: 700 !important; font-size: 0.73rem !important; letter-spacing: 0.5px !important; }
    .stDataFrame tbody tr:nth-child(even) td { background: rgba(194,87,10,0.03) !important; }
    .stDataFrame tbody tr:hover td { background: rgba(61,28,2,0.06) !important; }

    /* ── MAIN BUTTONS ───────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #3D1C02, #6B3A1F);
        color: #fff; border: none; border-radius: var(--radius);
        font-family: 'Nunito', sans-serif; font-weight: 800;
        font-size: 0.84rem; padding: 0.65rem 1.5rem;
        cursor: pointer; transition: all 0.22s; width: 100%;
        box-shadow: 0 4px 16px rgba(61,28,2,0.38); letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6B3A1F, #C2570A);
        box-shadow: 0 8px 28px rgba(194,87,10,0.48); transform: translateY(-2px);
    }

    /* ── PRED RESULT ─────────────────────────────────────── */
    .pred-result-box {
        border-radius: var(--radius-lg); padding: 1.8rem 2rem;
        margin-top: 1rem; text-align: center; border: 2px solid;
        position: relative; overflow: hidden;
    }
    .pred-result-box::before {
        content: ''; position: absolute; inset: 0;
        background-image: radial-gradient(rgba(255,255,255,0.5) 1px, transparent 1px);
        background-size: 20px 20px; pointer-events: none; opacity: 0.3;
    }
    .pred-miskin { background: linear-gradient(135deg, #FEF9F4, #FDECD8); border-color: rgba(194,87,10,0.40); }
    .pred-tidak  { background: linear-gradient(135deg, #F0FAF4, #E2F8EC); border-color: rgba(61,112,84,0.35); }
    .pred-label  { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; position: relative; }
    .pred-sub    { font-size: 0.79rem; color: var(--text-dim); margin-top: 0.5rem; position: relative; }

    /* ── DOWNLOAD LINK ─────────────────────────────────── */
    .dl-btn {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: var(--surface3); color: #3D1C02 !important;
        text-decoration: none; border: 1.5px solid var(--border2);
        border-radius: var(--radius); padding: 0.48rem 1.2rem;
        font-family: 'Nunito', sans-serif; font-weight: 700;
        font-size: 0.78rem; transition: all 0.2s; margin-top: 0.4rem;
    }
    .dl-btn:hover { border-color: #C2570A; background: #FDECD8; color: #8B3A00 !important; }

    /* ── MISC ──────────────────────────────────────────── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--surface2); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--terra); }
    [data-testid="stAlert"] { border-radius: var(--radius) !important; font-family: 'Nunito', sans-serif !important; }
    footer { display: none !important; }
    #MainMenu { visibility: hidden; }
    .stSelectbox label, .stNumberInput label, .stTextInput label, .stRadio label {
        color: var(--text-body) !important; font-weight: 700 !important; font-size: 0.81rem !important;
    }
    .stTextInput input {
        background: var(--surface) !important; border: 1.5px solid var(--border2) !important;
        border-radius: 8px !important; color: var(--text) !important;
        font-family: 'Nunito', sans-serif !important;
    }
    .stTextInput input:focus { border-color: #C2570A !important; box-shadow: 0 0 0 3px rgba(194,87,10,0.14) !important; }
    .stSlider > div > div > div { background: linear-gradient(90deg, #C2570A, #3D1C02) !important; }
    .stSlider label { color: var(--text-body) !important; font-weight: 700 !important; font-size: 0.81rem !important; }
    button[kind="header"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SPLASH SCREEN (WELCOME PAGE)
# =========================================================

if "page_mode" not in st.session_state:
    st.session_state.page_mode = "welcome"

if st.session_state.page_mode == "welcome":
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] { display: none !important; }
        header[data-testid="stHeader"] { display: none !important; }
        .main .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 1100px !important; }
        .stApp { background: linear-gradient(135deg, #FEF6EE 0%, #FAF0E6 100%); }
        .welcome-container {
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; min-height: 85vh; text-align: center; padding: 2rem;
        }
        .hero-card {
            background: rgba(255,255,255,0.88); backdrop-filter: blur(12px);
            border-radius: 48px; padding: 2.5rem 3rem;
            box-shadow: 0 25px 60px rgba(61,28,2,0.16), inset 0 1px 0 rgba(255,255,255,0.6);
            border: 1px solid rgba(194,87,10,0.12); max-width: 820px; width: 100%;
            transition: transform 0.3s ease;
        }
        .hero-card:hover { transform: translateY(-5px); }
        .icon-pulse {
            background: linear-gradient(135deg, #3D1C02, #C2570A);
            width: 100px; height: 100px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            margin: 0 auto 1.5rem;
            box-shadow: 0 15px 40px rgba(194,87,10,0.35);
            animation: float 3s ease-in-out infinite;
        }
        @keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-12px); } }
        .icon-pulse span { font-size: 3.5rem; filter: drop-shadow(2px 4px 6px rgba(0,0,0,0.25)); }
        .hero-title {
            font-family: 'Playfair Display', serif; font-size: 2.5rem; font-weight: 700;
            background: linear-gradient(135deg, #1A0A00, #C2570A, #C49A1A);
            background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
            color: transparent; margin-bottom: 0.5rem; animation: shine 6s linear infinite;
        }
        @keyframes shine { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
        .hero-sub { font-family: 'Playfair Display', serif; font-size: 1.7rem; font-weight: 600; color: #3D1C02; margin-bottom: 1rem; }
        .hero-desc { font-size: 1rem; color: #7A5C44; max-width: 580px; margin: 0 auto 1.8rem; line-height: 1.65; }
        .pill-group { display: flex; flex-wrap: wrap; gap: 0.8rem; justify-content: center; margin-bottom: 2rem; }
        .pill {
            background: rgba(194,87,10,0.10); backdrop-filter: blur(4px);
            padding: 0.5rem 1.4rem; border-radius: 60px; font-weight: 600;
            color: #8B3A00; border: 1px solid rgba(194,87,10,0.22);
            transition: all 0.2s; font-size: 0.85rem;
        }
        .pill:hover { background: rgba(61,112,84,0.14); color: #2A5C3A; border-color: rgba(61,112,84,0.35); transform: translateY(-2px); }
        .welcome-footer { margin-top: 2rem; font-size: 0.75rem; color: #A88B78; }
        </style>
        <div class="welcome-container">
            <div class="hero-card">
                <div class="icon-pulse"><span>🏠</span></div>
                <div class="hero-title">Sistem Pendukung Keputusan</div>
                <div class="hero-sub">Penentuan Keluarga Miskin</div>
                <div class="hero-desc">
                    Menggunakan algoritma <strong>Gaussian Naive Bayes</strong> untuk menentukan kelayakan keluarga penerima bantuan sosial secara objektif — Desa Nansean, Kabupaten TTU.
                </div>
                <div class="pill-group">
                    <div class="pill">📊 Akurasi Tinggi</div>
                    <div class="pill">🤖 Machine Learning</div>
                    <div class="pill">🎯 Tepat Sasaran</div>
                    <div class="pill">📈 Evaluasi Otomatis</div>
                    <div class="pill">⚡ Real-time</div>
                </div>
                <div style="margin-top: 0.5rem;">
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if st.button("🚀 Mulai Klasifikasi", use_container_width=True, key="start_btn"):
            st.session_state.page_mode = "main"
            st.rerun()

    st.markdown(
        """
                </div>
                <div class="welcome-footer">© 2025 · Kemiskinan Classifier · Gaussian Naive Bayes · Desa Nansean, TTU</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# =========================================================
# TITLE BAR (hanya tampil di halaman utama)
# =========================================================

st.markdown(
    """
    <div class="page-titlebar">
        <div class="ptb-badge">🏠</div>
        <div class="ptb-text">
            <div class="ptb-eyebrow">Sistem Pendukung Keputusan · Gaussian Naive Bayes</div>
            <div class="ptb-title">Penentuan <span class="hl">Keluarga Miskin</span></div>
        </div>
        <div class="ptb-pills">
            <div class="hpill hpill-terra">🤖 ML</div>
            <div class="hpill hpill-brown">📊 Gaussian NB</div>
            <div class="hpill hpill-gold">🏡 Desa Nansean</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE
# =========================================================

for key in ["raw", "df_clean", "nb_results", "data_tambahan",
            "notif_upload", "notif_prep", "notif_nb", "split_ratio"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state["data_tambahan"] is None:
    st.session_state["data_tambahan"] = pd.DataFrame(
        columns=["NAMA KEPALA KELUARGA", "PEKERJAAN", "PENGHASILAN",
                 "TANGGUNGAN", "STATUS KEPEMILIKAN RUMAH", "KETERANGAN (PREDIKSI)"]
    )

if st.session_state["split_ratio"] is None:
    st.session_state["split_ratio"] = 80

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def make_excel_download(df: pd.DataFrame, filename: str, label: str, judul: str = "") -> str:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        if judul:
            title_df = pd.DataFrame([[judul]], columns=[""])
            title_df.to_excel(w, index=False, startrow=0)
            df.to_excel(w, index=False, startrow=2)
        else:
            df.to_excel(w, index=False)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<a class="dl-btn" href="data:application/vnd.openxmlformats-officedocument'
        f'.spreadsheetml.sheet;base64,{b64}" download="{filename}">{label}</a>'
    )


def make_png_download(fig, filename: str, label: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<a class="dl-btn" href="data:image/png;base64,{b64}" download="{filename}">{label}</a>'
    )


def notif_html(variant: str, icon: str, title: str, body: str, stats: list = None) -> str:
    stats_html = ""
    if stats:
        stats_html = '<div class="notif-stats">'
        for s in stats:
            stats_html += f'<span class="notif-stat">{s}</span>'
        stats_html += '</div>'
    return (
        f'<div class="notif-box notif-{variant}">'
        f'<div class="notif-icon">{icon}</div>'
        f'<div class="notif-content">'
        f'<div class="notif-title">{title}</div>'
        f'<div class="notif-body">{body}</div>'
        f'{stats_html}'
        f'</div></div>'
    )


# =========================================================
# GAUSSIAN PROBABILITY
# =========================================================

def gaussian_probability(x, mean, std):
    if std == 0 or pd.isna(std):
        std = 1e-9
    exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


# =========================================================
# LOAD DATA
# =========================================================

def load_data(file) -> tuple:
    try:
        if file.name.endswith(".csv"):
            raw = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            raw = pd.read_excel(file)
        else:
            return None, "❌ Format file tidak didukung (gunakan .csv atau .xlsx)", {}
        info = {
            "rows":     len(raw),
            "cols":     len(raw.columns),
            "filename": file.name,
            "size_kb":  round(file.size / 1024, 1) if hasattr(file, "size") else "?",
        }
        return raw, "ok", info
    except Exception as e:
        return None, f"❌ Error membaca file: {e}", {}


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_data(raw: pd.DataFrame) -> tuple:
    try:
        data = raw.copy()
        data.columns = data.columns.str.strip().str.upper()

        # Bersihkan PENGHASILAN
        data["PENGHASILAN"] = (
            data["PENGHASILAN"].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("RP", "", regex=False)
            .str.strip()
        )
        data["PENGHASILAN"] = pd.to_numeric(data["PENGHASILAN"], errors="coerce")
        data["TANGGUNGAN"]  = pd.to_numeric(data["TANGGUNGAN"],  errors="coerce")

        before_na = len(data)
        data = data.dropna(subset=[
            "PEKERJAAN", "PENGHASILAN", "TANGGUNGAN",
            "STATUS KEPEMILIKAN RUMAH", "KETERANGAN"
        ])
        after_count = len(data)
        dropped     = before_na - after_count

        miskin_count = int((data["KETERANGAN"].astype(str).str.lower().str.strip() == "ya").sum())
        tidak_count  = int((data["KETERANGAN"].astype(str).str.lower().str.strip() == "tidak").sum())

        info = {
            "before":  before_na,
            "after":   after_count,
            "dropped": dropped,
            "miskin":  miskin_count,
            "tidak":   tidak_count,
        }
        return data, "ok", info
    except Exception as e:
        return None, f"❌ Error preprocessing: {e}", {}


# =========================================================
# NAIVE BAYES PROCESS
# =========================================================

def naive_bayes_process(df: pd.DataFrame, train_pct: int = 80) -> dict:
    data = df.copy()

    le_pekerjaan = LabelEncoder()
    le_rumah     = LabelEncoder()
    le_target    = LabelEncoder()

    data["PEKERJAAN"]                = le_pekerjaan.fit_transform(data["PEKERJAAN"].astype(str))
    data["STATUS KEPEMILIKAN RUMAH"] = le_rumah.fit_transform(data["STATUS KEPEMILIKAN RUMAH"].astype(str))
    data["KETERANGAN"]               = le_target.fit_transform(data["KETERANGAN"].astype(str))

    fitur  = ["PEKERJAAN", "PENGHASILAN", "TANGGUNGAN", "STATUS KEPEMILIKAN RUMAH"]
    target = "KETERANGAN"

    X, y = data[fitur], data[target]

    if train_pct >= 100:
        X_train, X_test = X.copy(), X.copy()
        y_train, y_test = y.copy(), y.copy()
        split_mode = "full_train"
    else:
        test_size = (100 - train_pct) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        split_mode = "split"

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    classes = np.unique(y_train)
    prior, mean_d, std_d = {}, {}, {}
    for c in classes:
        prior[c]  = len(y_train[y_train == c]) / len(y_train)
        data_c    = X_train[y_train == c]
        mean_d[c] = data_c.mean()
        std_d[c]  = data_c.std()

    tabel_prior = pd.DataFrame({
        "Kelas":               [le_target.inverse_transform([c])[0] for c in classes],
        "Kode Kelas":          list(classes),
        "Prior Probability":   [prior[c] for c in classes],
    })

    hasil_manual = []
    for i in range(len(X_test)):
        row   = X_test.iloc[i]
        probs = {}
        for c in classes:
            probs[c] = prior[c]
            for f in fitur:
                probs[c] *= gaussian_probability(row[f], mean_d[c][f], std_d[c][f])
        prediksi     = max(probs, key=probs.get)
        label_pred   = le_target.inverse_transform([prediksi])[0]
        label_actual = le_target.inverse_transform([y_test.iloc[i]])[0]
        hasil_manual.append({
            "Data Ke": i + 1,
            **{f"P({le_target.inverse_transform([c])[0]})": probs.get(c, 0) for c in classes},
            "Prediksi":  label_pred,
            "Aktual":    label_actual,
            "Benar?":    "✅" if prediksi == y_test.iloc[i] else "❌",
        })

    hasil_manual_df = pd.DataFrame(hasil_manual)

    # Prediksi seluruh data
    y_pred_all       = model.predict(X)
    y_pred_all_label = le_target.inverse_transform(y_pred_all)
    y_actual_label   = le_target.inverse_transform(y)

    hasil_semua = df[["NAMA KEPALA KELUARGA"]].copy() if "NAMA KEPALA KELUARGA" in df.columns else pd.DataFrame()
    hasil_semua["PEKERJAAN"]                = le_pekerjaan.inverse_transform(data["PEKERJAAN"])
    hasil_semua["PENGHASILAN"]              = df["PENGHASILAN"].values
    hasil_semua["TANGGUNGAN"]               = df["TANGGUNGAN"].values
    hasil_semua["STATUS KEPEMILIKAN RUMAH"] = le_rumah.inverse_transform(data["STATUS KEPEMILIKAN RUMAH"])
    hasil_semua["AKTUAL"]                   = y_actual_label
    hasil_semua["PREDIKSI"]                 = y_pred_all_label
    posterior_all = model.predict_proba(X)
    for idx, c in enumerate(model.classes_):
        lbl = le_target.inverse_transform([c])[0]
        hasil_semua[f"P({lbl})"] = posterior_all[:, idx]

    cm       = confusion_matrix(le_target.inverse_transform(y_test), le_target.inverse_transform(y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(
        classification_report(
            le_target.inverse_transform(y_test),
            le_target.inverse_transform(y_pred),
            output_dict=True, zero_division=0,
        )
    ).transpose()

    jumlah_miskin       = int(np.sum(y_pred_all_label == "ya"))
    jumlah_tidak_miskin = int(np.sum(y_pred_all_label == "tidak"))

    benar = int((hasil_manual_df["Benar?"] == "✅").sum())
    salah = int((hasil_manual_df["Benar?"] == "❌").sum())

    # ── Chart palette ──────────────────────────────────
    BG     = "#FDF6EE"
    SURFACE = "#FFFFFF"
    BROWN  = "#3D1C02"
    TERRA  = "#C2570A"
    TERRA_L= "#E07030"
    GOLD   = "#C49A1A"
    SAGE   = "#3D7054"
    MUTED  = "#7A5C44"
    BORDER = "#E8D5C0"
    TEXT   = "#1A0A00"
    ROSE   = "#C0392B"

    # Confusion Matrix
    cm_labels = le_target.classes_
    n_cls     = len(cm_labels)

    fig_cm, ax = plt.subplots(figsize=(7, 5.5))
    fig_cm.patch.set_facecolor(SURFACE)
    ax.set_facecolor(BG)

    cell_colors = [[SAGE, ROSE], [ROSE, SAGE]] if n_cls == 2 else None
    cell_labels_map = [["TN", "FP"], ["FN", "TP"]] if n_cls == 2 else None

    for i in range(n_cls):
        for j in range(n_cls):
            val      = cm[i, j]
            bg_color = (cell_colors[i][j] if cell_colors else (TERRA if i == j else ROSE))
            lbl      = (cell_labels_map[i][j] if cell_labels_map else ("Benar" if i == j else "Salah"))
            rect = plt.Rectangle((j+0.06, i+0.06), 0.88, 0.88, color=bg_color, alpha=0.10, zorder=0)
            ax.add_patch(rect)
            brect = plt.Rectangle((j+0.06, i+0.06), 0.88, 0.88, fill=False, edgecolor=bg_color, alpha=0.45, linewidth=2.2, zorder=1)
            ax.add_patch(brect)
            ax.text(j+0.5, i+0.4,  str(val), ha="center", va="center", fontsize=52, fontweight="bold", color=bg_color, zorder=2, fontfamily="monospace")
            ax.text(j+0.5, i+0.74, lbl,      ha="center", va="center", fontsize=12, fontweight="700",  color=bg_color, alpha=0.70, zorder=2)

    for k in range(1, n_cls): ax.axhline(k, color=BORDER, linewidth=1.5, zorder=3)
    for k in range(1, n_cls): ax.axvline(k, color=BORDER, linewidth=1.5, zorder=3)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER); spine.set_linewidth(1.5)

    ax.set_xlim(0, n_cls); ax.set_ylim(0, n_cls); ax.invert_yaxis()
    ax.set_xticks([i+0.5 for i in range(n_cls)])
    ax.set_yticks([i+0.5 for i in range(n_cls)])
    ax.set_xticklabels(cm_labels, color=TEXT, fontsize=11, fontweight="600")
    ax.set_yticklabels(cm_labels, color=TEXT, fontsize=11, fontweight="600", rotation=0, va="center")
    ax.set_xlabel("Prediksi", color=MUTED, labelpad=14, fontsize=12)
    ax.set_ylabel("Aktual",   color=MUTED, labelpad=14, fontsize=12)
    split_label = f"Training {train_pct}% / Testing {100-train_pct}%"
    ax.set_title(f"Confusion Matrix — Hasil Evaluasi Model\n({split_label})",
                 color=BROWN, fontsize=13, fontweight="bold", pad=22, fontfamily="DejaVu Serif")
    ax.tick_params(colors=BORDER, length=0)
    legend_items = [
        mpatches.Patch(color=SAGE, label="Prediksi Benar (TN / TP)"),
        mpatches.Patch(color=ROSE, label="Prediksi Salah (FP / FN)"),
    ]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=10)
    plt.tight_layout()

    st.session_state["le_pekerjaan"] = le_pekerjaan
    st.session_state["le_rumah"]     = le_rumah
    st.session_state["le_target"]    = le_target

    return {
        "prior":               tabel_prior,
        "manual":              hasil_manual_df,
        "hasil_semua":         hasil_semua,
        "report":              report_df,
        "accuracy":            accuracy,
        "fig_cm":              fig_cm,
        "train_size":          len(X_train),
        "test_size":           len(X_test),
        "train_pct":           train_pct,
        "test_pct":            100 - train_pct,
        "split_mode":          split_mode,
        "classes":             classes.tolist(),
        "classes_label":       le_target.classes_.tolist(),
        "model":               model,
        "cm":                  cm,
        "jumlah_miskin":       jumlah_miskin,
        "jumlah_tidak_miskin": jumlah_tidak_miskin,
        "total_data":          len(y),
        "le_pekerjaan":        le_pekerjaan,
        "le_rumah":            le_rumah,
        "le_target":           le_target,
        "mean_d":              mean_d,
        "std_d":               std_d,
        "fitur":               fitur,
        "benar":               benar,
        "salah":               salah,
    }


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sb-logo">
            <div class="sb-logo-row">
                <div class="sb-logo-icon">🏠</div>
                <div>
                    <div class="sb-logo-title">Kemiskinan <span>Classifier</span></div>
                    <div class="sb-logo-desc">Gaussian Naive Bayes</div>
                </div>
            </div>
            <div class="sb-logo-desc">Sistem Penentuan Keluarga Miskin — Desa Nansean, Kab. TTU</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<span class="sb-sec">📂 Unggah Dataset</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Pilih file CSV atau Excel", type=["csv", "xlsx"],
        label_visibility="collapsed", key="uploader_main"
    )

    if uploaded:
        if st.button("⬆ Upload Dataset", key="btn_upload"):
            raw, msg, info = load_data(uploaded)
            if raw is not None:
                st.session_state["raw"]          = raw
                st.session_state["df_clean"]     = None
                st.session_state["nb_results"]   = None
                st.session_state["notif_upload"] = info
                st.session_state["notif_prep"]   = None
                st.session_state["notif_nb"]     = None
            else:
                st.error(msg)

    st.markdown('<span class="sb-sec">⚙ Preprocessing Data</span>', unsafe_allow_html=True)
    if st.session_state["raw"] is not None:
        if st.button("🔧 Jalankan Preprocessing", key="btn_preprocess"):
            df, msg, info = preprocess_data(st.session_state["raw"])
            if df is not None:
                st.session_state["df_clean"]   = df
                st.session_state["nb_results"] = None
                st.session_state["notif_prep"] = info
                st.session_state["notif_nb"]   = None
            else:
                st.error(msg)

    # Pipeline status
    st.markdown('<span class="sb-sec">📋 Status Pipeline</span>', unsafe_allow_html=True)
    raw_ok   = st.session_state["raw"]        is not None
    clean_ok = st.session_state["df_clean"]   is not None
    nb_ok    = st.session_state["nb_results"] is not None

    steps = [
        ("1", "Upload Data",   raw_ok,   raw_ok and not clean_ok),
        ("2", "Preprocessing", clean_ok, clean_ok and not nb_ok),
        ("3", "Naive Bayes",   nb_ok,    False),
        ("4", "Evaluasi",      nb_ok,    False),
    ]
    pipe_html = '<div class="pipeline">'
    for num, label_s, done, active in steps:
        if done:   cls = "done";   num_cls = "done";   symbol = "✓"
        elif active: cls = "active"; num_cls = "active"; symbol = num
        else:      cls = "";       num_cls = "";        symbol = num
        pipe_html += (
            f'<div class="pipeline-step {cls}">'
            f'<span class="step-num {num_cls}">{symbol}</span>{label_s}'
            f'</div>'
        )
    pipe_html += '</div>'
    st.markdown(pipe_html, unsafe_allow_html=True)

    # Mini stats jika model sudah dilatih
    if nb_ok:
        nb_side = st.session_state["nb_results"]
        st.markdown('<span class="sb-sec">📊 Ringkasan Model</span>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:rgba(194,87,10,0.14);border:1px solid rgba(224,112,48,0.32);
                        border-radius:12px;padding:1rem 1.1rem;margin-top:0.2rem;">
                <div style="font-size:0.65rem;color:rgba(255,255,255,0.40);text-transform:uppercase;letter-spacing:1.8px;font-weight:700;margin-bottom:0.7rem;">Performa</div>
                <div style="font-family:'Playfair Display',serif;font-size:2.6rem;color:#FFAD55;line-height:1;text-align:center;margin-bottom:0.3rem;">{nb_side['accuracy']*100:.1f}%</div>
                <div style="text-align:center;font-size:0.68rem;color:rgba(255,255,255,0.38);margin-bottom:0.8rem;">Akurasi Model</div>
                <div style="text-align:center;font-size:0.63rem;color:rgba(255,255,255,0.30);margin-bottom:0.8rem;">Split: {nb_side['train_pct']}% Train / {nb_side['test_pct']}% Test</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
                    <div style="background:rgba(194,87,10,0.18);border-radius:8px;padding:0.5rem;text-align:center;">
                        <div style="font-size:1.1rem;font-weight:800;color:#FFAD55;">{nb_side['train_size']}</div>
                        <div style="font-size:0.58rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:1px;">Train</div>
                    </div>
                    <div style="background:rgba(196,154,26,0.18);border-radius:8px;padding:0.5rem;text-align:center;">
                        <div style="font-size:1.1rem;font-weight:800;color:#FFD97A;">{nb_side['test_size']}</div>
                        <div style="font-size:0.58rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:1px;">Test</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# METRIC CARDS
# =========================================================

raw      = st.session_state["raw"]
df_clean = st.session_state["df_clean"]
nb       = st.session_state["nb_results"]

total_raw   = len(raw)      if raw      is not None else "–"
total_clean = len(df_clean) if df_clean is not None else "–"
akurasi_val = f"{nb['accuracy']*100:.1f}%" if nb else "–"
train_size  = nb["train_size"] if nb else "–"
test_size   = nb["test_size"]  if nb else "–"

if nb:
    split_badge_train = f"{nb['train_pct']}%"
    split_badge_test  = f"{nb['test_pct']}%"
else:
    split_ratio_now   = st.session_state.get("split_ratio", 80)
    split_badge_train = f"{split_ratio_now}%"
    split_badge_test  = f"{100 - split_ratio_now}%"

# ── Global Notifications ────────────────────────────────
notif_upload = st.session_state.get("notif_upload")
notif_prep   = st.session_state.get("notif_prep")
notif_nb_    = st.session_state.get("notif_nb")

if notif_upload:
    st.markdown(
        notif_html("upload", "📂", "✅ Dataset Berhasil Diunggah!",
                   f"File <strong>{notif_upload['filename']}</strong> berhasil dibaca dan siap diproses.",
                   [f"📋 Total Baris: {notif_upload['rows']}", f"📊 Kolom: {notif_upload['cols']}", f"💾 Ukuran: {notif_upload['size_kb']} KB"]),
        unsafe_allow_html=True,
    )

if notif_prep:
    st.markdown(
        notif_html("prep", "🔧", "✅ Preprocessing Selesai!",
                   f"Data berhasil dibersihkan. <strong>{notif_prep['dropped']}</strong> baris dihapus karena nilai tidak valid.",
                   [f"📥 Input: {notif_prep['before']} baris", f"✅ Output: {notif_prep['after']} baris bersih",
                    f"🏠 Keluarga Miskin: {notif_prep['miskin']}", f"📋 Tidak Miskin: {notif_prep['tidak']}",
                    f"🗑 Dihapus: {notif_prep['dropped']} baris"]),
        unsafe_allow_html=True,
    )

if notif_nb_:
    st.markdown(
        notif_html("nb", "🤖", "✅ Model Naive Bayes Selesai Dilatih!",
                   f"Model dilatih dengan split <strong>{notif_nb_['train_pct']}%/{notif_nb_['test_pct']}%</strong>. Akurasi: <strong>{notif_nb_['acc']*100:.2f}%</strong>.",
                   [f"🏋 Training: {notif_nb_['train']} data ({notif_nb_['train_pct']}%)",
                    f"🧪 Testing: {notif_nb_['test']} data ({notif_nb_['test_pct']}%)",
                    f"✅ Prediksi Benar: {notif_nb_['benar']}", f"❌ Prediksi Salah: {notif_nb_['salah']}",
                    f"🏅 Akurasi: {notif_nb_['acc']*100:.2f}%"]),
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-card mc-brown">
            <span class="mc-icon">📋</span>
            <div class="mc-label">Total Data Awal</div>
            <div class="mc-value cv-brown">{total_raw}</div>
        </div>
        <div class="metric-card mc-terra">
            <span class="mc-icon">✅</span>
            <div class="mc-label">Setelah Preprocessing</div>
            <div class="mc-value cv-terra">{total_clean}</div>
        </div>
        <div class="metric-card mc-violet">
            <span class="mc-icon">🏋</span>
            <div class="mc-label">Data Training ({split_badge_train})</div>
            <div class="mc-value cv-violet">{train_size}</div>
        </div>
        <div class="metric-card mc-rose">
            <span class="mc-icon">🧪</span>
            <div class="mc-label">Data Testing ({split_badge_test})</div>
            <div class="mc-value cv-rose">{test_size}</div>
        </div>
        <div class="metric-card mc-gold">
            <span class="mc-icon">🏅</span>
            <div class="mc-label">Akurasi Model</div>
            <div class="mc-value cv-gold">{akurasi_val}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📁 Data Awal",
    "🔧 Preprocessing",
    "📊 Prior & Manual",
    "🎯 Hasil Prediksi",
    "🔲 Confusion Matrix",
    "📈 Laporan Klasifikasi",
    "➕ Input Data Baru",
])

# ─────────────────────────────────────────────────────────
# TAB 1 — Dataset Awal
# ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📁 Dataset Mentah</div>', unsafe_allow_html=True)
    if raw is not None:
        if notif_upload:
            st.markdown(
                notif_html("upload", "💡", f"Dataset dimuat: {notif_upload['rows']} baris",
                           "Data di bawah adalah tampilan mentah sebelum preprocessing. Pastikan kolom yang dibutuhkan tersedia.",
                           [f"📋 Total baris file: {notif_upload['rows']}", f"📊 Kolom: {notif_upload['cols']}"]),
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="info-box">💡 Ini adalah data asli sebelum preprocessing. '
            'Kolom yang dibutuhkan: <strong>NAMA KEPALA KELUARGA, PEKERJAAN, PENGHASILAN, '
            'TANGGUNGAN, STATUS KEPEMILIKAN RUMAH, KETERANGAN</strong> (ya = miskin / tidak = tidak miskin).</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(raw, use_container_width=True, height=420)
        st.markdown("---")
        col_j1, col_dl1 = st.columns([2, 1])
        with col_j1:
            judul_raw = st.text_input("Judul untuk file unduhan", value="Dataset Awal Keluarga Miskin", key="judul_raw")
        with col_dl1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(make_excel_download(raw, f"{judul_raw.replace(' ','_')}.xlsx", f"⬇ Unduh {judul_raw} (.xlsx)", judul_raw), unsafe_allow_html=True)
    else:
        st.info("👈 Unggah dataset terlebih dahulu melalui panel kiri untuk memulai.")
        st.markdown("""
        **Format file yang diharapkan:**
        | NAMA KEPALA KELUARGA | PEKERJAAN | PENGHASILAN | TANGGUNGAN | STATUS KEPEMILIKAN RUMAH | KETERANGAN |
        |---|---|---|---|---|---|
        | Wilfridus Emanuel Nesi | Petani | 250000 | 5 | Milik Sendiri | ya |

        *Keterangan: `ya` = miskin, `tidak` = tidak miskin*
        """)

# ─────────────────────────────────────────────────────────
# TAB 2 — Preprocessing + KONFIGURASI SPLIT + TOMBOL NB
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🔧 Hasil Preprocessing & Konfigurasi Model</div>', unsafe_allow_html=True)
    if df_clean is not None:
        if notif_prep:
            st.markdown(
                notif_html("prep", "🔧", "Hasil Preprocessing Dataset",
                           f"Berhasil memproses <strong>{notif_prep['after']}</strong> data bersih dari <strong>{notif_prep['before']}</strong> data awal.",
                           [f"✅ Data bersih: {notif_prep['after']}", f"🗑 Dihapus: {notif_prep['dropped']}",
                            f"🏠 Miskin: {notif_prep['miskin']}", f"📋 Tidak Miskin: {notif_prep['tidak']}"]),
                unsafe_allow_html=True,
            )

        st.markdown(
            """<div class="info-box">
            💡 Data telah dibersihkan dan siap diproses model.<br>
            <strong>PENGHASILAN</strong> → nilai numerik (Rp) &nbsp;|&nbsp;
            <strong>TANGGUNGAN</strong> → bilangan bulat &nbsp;|&nbsp;
            <strong>PEKERJAAN & STATUS RUMAH</strong> → di-encode saat training (LabelEncoder) &nbsp;|&nbsp;
            <strong>KETERANGAN</strong> → target (ya = miskin, tidak = tidak miskin)
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="formula-box">
                <div class="formula-title">📐 Catatan Rumus Encoding & Preprocessing</div>
                <strong>1. PENGHASILAN</strong> — Dibersihkan dari karakter non-numerik (titik, koma, Rp), lalu dikonversi ke <code>float</code><br>
                <strong>2. TANGGUNGAN</strong> — Dikonversi ke <code>int</code><br>
                <strong>3. PEKERJAAN & STATUS KEPEMILIKAN RUMAH</strong> — Label Encoding otomatis saat training<br>
                <strong>4. KETERANGAN</strong> — Label Encoding: <code>tidak</code> → 0, <code>ya</code> → 1 (atau sesuai urutan alfabet)
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(df_clean, use_container_width=True, height=320)
        st.markdown("---")

        # ── Konfigurasi Split Ratio ──────────────────────
        st.markdown('<div class="section-header">⚖️ Konfigurasi Pembagian Data (Train/Test Split)</div>', unsafe_allow_html=True)

        total_data = len(df_clean)
        PRESET_OPTIONS = {
            "90% / 10%": 90,
            "80% / 20%": 80,
            "70% / 30%": 70,
            "60% / 40%": 60,
            "50% / 50%": 50,
            "100% / 0%": 100,
        }
        current_ratio = st.session_state.get("split_ratio", 80)

        col_preset, col_slider = st.columns([1.2, 1.8])
        with col_preset:
            st.markdown('<p style="font-size:.72rem;color:#7A5C44;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.6rem;">🎛 Pilih Preset Rasio</p>', unsafe_allow_html=True)
            preset_labels = list(PRESET_OPTIONS.keys())
            default_preset_idx = 1
            for idx, (lbl_, val_) in enumerate(PRESET_OPTIONS.items()):
                if val_ == current_ratio:
                    default_preset_idx = idx
                    break
            selected_preset = st.radio("Pilih preset rasio", options=preset_labels, index=default_preset_idx, key="preset_split", label_visibility="collapsed")
            st.session_state["split_ratio"] = PRESET_OPTIONS[selected_preset]

        with col_slider:
            st.markdown('<p style="font-size:.72rem;color:#7A5C44;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.6rem;">🔢 Atau Atur Manual dengan Slider (% Data Training)</p>', unsafe_allow_html=True)
            slider_val = st.slider("Persentase data training", min_value=50, max_value=100, value=st.session_state["split_ratio"], step=5, format="%d%%", key="slider_split", label_visibility="collapsed")
            if slider_val != st.session_state["split_ratio"]:
                st.session_state["split_ratio"] = slider_val

        final_ratio = st.session_state["split_ratio"]
        train_count = round(total_data * final_ratio / 100)
        test_count  = total_data - train_count

        st.markdown(
            f"""
            <div class="split-config-box">
                <div class="split-config-title">⚖️ Rasio Pembagian Data yang Dipilih</div>
                <div class="split-config-desc">Konfigurasi ini akan digunakan saat tombol <strong>🤖 Proses & Klasifikasi Naive Bayes</strong> di bawah ditekan.</div>
                <div class="split-bar-wrap">
                    <div class="split-bar-train" style="width:{final_ratio}%;background:linear-gradient(90deg,#C2570A,#3D1C02);"></div>
                </div>
                <div class="split-bar-labels">
                    <span class="split-label-train">🏋 Training — {final_ratio}% ({train_count} data)</span>
                    <span class="split-label-test">🧪 Testing — {100-final_ratio}% ({test_count} data)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if final_ratio == 100:
            st.markdown(
                notif_html("warn", "⚠️", "Mode 100% Training — Perhatian!",
                           "Seluruh data digunakan untuk training. Evaluasi dilakukan pada data training itu sendiri (in-sample), sehingga akurasi cenderung lebih tinggi namun tidak mencerminkan performa pada data baru.",
                           ["📌 In-sample evaluation", "⚠️ Risiko overfitting", "💡 Gunakan untuk eksplorasi"]),
                unsafe_allow_html=True,
            )

        st.markdown("---")
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("🤖 Proses & Klasifikasi Naive Bayes", use_container_width=True):
                with st.spinner("Melatih model dan melakukan klasifikasi..."):
                    nb_res = naive_bayes_process(st.session_state["df_clean"], train_pct=final_ratio)
                    st.session_state["nb_results"] = nb_res
                    st.session_state["notif_nb"]   = {
                        "train":     nb_res["train_size"],
                        "test":      nb_res["test_size"],
                        "acc":       nb_res["accuracy"],
                        "benar":     nb_res["benar"],
                        "salah":     nb_res["salah"],
                        "train_pct": nb_res["train_pct"],
                        "test_pct":  nb_res["test_pct"],
                    }
                    st.success(f"✅ Klasifikasi selesai! Akurasi: {nb_res['accuracy']*100:.2f}%")
        with col_btn2:
            if st.button("🗑 Reset Model & Data", use_container_width=True):
                for k in ["nb_results", "notif_nb", "df_clean", "raw", "notif_upload", "notif_prep"]:
                    st.session_state[k] = None
                st.session_state["data_tambahan"] = pd.DataFrame(
                    columns=["NAMA KEPALA KELUARGA", "PEKERJAAN", "PENGHASILAN",
                             "TANGGUNGAN", "STATUS KEPEMILIKAN RUMAH", "KETERANGAN (PREDIKSI)"]
                )
                st.rerun()

        # Distribusi label
        st.markdown("---")
        miskin_n = int((df_clean["KETERANGAN"].astype(str).str.lower().str.strip() == "ya").sum())
        tidak_n  = len(df_clean) - miskin_n
        st.markdown(
            f"""
            <div style="display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap;">
                <div style="flex:1;min-width:160px;background:linear-gradient(135deg,#FEF9F4,#FDECD8);
                            border:1.5px solid rgba(194,87,10,0.28);border-radius:14px;padding:1.1rem 1.3rem;text-align:center;">
                    <div style="font-size:.62rem;color:#8B3A00;text-transform:uppercase;letter-spacing:1.5px;font-weight:800;margin-bottom:.4rem;">🏠 Keluarga Miskin</div>
                    <div style="font-family:'Playfair Display',serif;font-size:2.4rem;color:#C2570A;line-height:1;font-weight:700;">{miskin_n:,}</div>
                    <div style="font-size:.75rem;color:#7A5C44;margin-top:.3rem;font-weight:600;">{miskin_n/len(df_clean)*100:.1f}% dari total data</div>
                </div>
                <div style="flex:1;min-width:160px;background:linear-gradient(135deg,#F0FAF4,#E2F8EC);
                            border:1.5px solid rgba(61,112,84,0.25);border-radius:14px;padding:1.1rem 1.3rem;text-align:center;">
                    <div style="font-size:.62rem;color:#2A5C3A;text-transform:uppercase;letter-spacing:1.5px;font-weight:800;margin-bottom:.4rem;">📋 Tidak Miskin</div>
                    <div style="font-family:'Playfair Display',serif;font-size:2.4rem;color:#3D7054;line-height:1;font-weight:700;">{tidak_n:,}</div>
                    <div style="font-size:.75rem;color:#5A7A6A;margin-top:.3rem;font-weight:600;">{tidak_n/len(df_clean)*100:.1f}% dari total data</div>
                </div>
                <div style="flex:1;min-width:160px;background:linear-gradient(135deg,#FEF9F4,#FEF0E4);
                            border:1.5px solid rgba(61,28,2,0.18);border-radius:14px;padding:1.1rem 1.3rem;text-align:center;">
                    <div style="font-size:.62rem;color:#3D1C02;text-transform:uppercase;letter-spacing:1.5px;font-weight:800;margin-bottom:.4rem;">👥 Total Keluarga</div>
                    <div style="font-family:'Playfair Display',serif;font-size:2.4rem;color:#3D1C02;line-height:1;font-weight:700;">{len(df_clean):,}</div>
                    <div style="font-size:.75rem;color:#7A5C44;margin-top:.3rem;font-weight:600;">Data bersih siap diproses</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_j2, col_dl2 = st.columns([2, 1])
        with col_j2:
            judul_prep = st.text_input("Judul file", value="Hasil Preprocessing Keluarga Miskin", key="judul_prep")
        with col_dl2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(make_excel_download(df_clean, f"{judul_prep.replace(' ','_')}.xlsx", f"⬇ Unduh {judul_prep} (.xlsx)", judul_prep), unsafe_allow_html=True)
    else:
        st.info("👈 Klik **Jalankan Preprocessing** di panel kiri setelah mengunggah data.")

# ─────────────────────────────────────────────────────────
# TAB 3 — Prior & Manual
# ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📊 Prior Probability & Perhitungan Manual</div>', unsafe_allow_html=True)
    if nb is not None:
        if notif_nb_:
            st.markdown(
                notif_html("nb", "📊", "Model Telah Dilatih — Data Prior & Likelihood Tersedia",
                           f"Dihitung dari <strong>{notif_nb_['train']}</strong> data training ({notif_nb_['train_pct']}%).",
                           [f"🏋 Training: {notif_nb_['train']} ({notif_nb_['train_pct']}%)", f"📊 Kelas: {len(nb['classes'])}"]),
                unsafe_allow_html=True,
            )

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<p style="font-size:.68rem;color:#7A5C44;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.45rem;">Tabel Prior Probability</p>', unsafe_allow_html=True)
            st.dataframe(nb["prior"], use_container_width=True)

            fig_prior, ax_pr = plt.subplots(figsize=(3.5, 2.8))
            fig_prior.patch.set_facecolor("#FFFFFF")
            ax_pr.set_facecolor("#FDF6EE")
            bar_col  = ["#C2570A", "#3D7054"]
            bar_lbls = nb["prior"]["Kelas"].tolist()
            bar_vals = nb["prior"]["Prior Probability"].tolist()
            bars = ax_pr.bar(bar_lbls, bar_vals, color=bar_col[:len(bar_lbls)], width=0.5, edgecolor="#fff", linewidth=1.5)
            ax_pr.set_ylim(0, 1)
            ax_pr.set_ylabel("Probabilitas", color="#7A5C44", fontsize=9)
            ax_pr.tick_params(colors="#7A5C44", labelsize=8)
            for spine in ax_pr.spines.values(): spine.set_edgecolor("#E8D5C0")
            for bar in bars:
                ax_pr.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.015, f"{bar.get_height():.3f}", ha="center", va="bottom", color="#1A0A00", fontsize=9, fontweight="bold")
            ax_pr.axhline(0.5, color="#C49A1A", linewidth=1, linestyle="--", alpha=.6)
            ax_pr.set_title("Prior per Kelas", fontsize=10, color="#3D1C02", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_prior)

        with c2:
            st.markdown('<p style="font-size:.68rem;color:#7A5C44;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.45rem;">Perhitungan Manual Naive Bayes (per data uji)</p>', unsafe_allow_html=True)
            st.dataframe(nb["manual"], use_container_width=True, height=380)

        st.markdown(
            """
            <div class="formula-box">
                <div class="formula-title">📐 Rumus Perhitungan Naive Bayes</div>
                <strong>1. Prior Probability:</strong><br>
                &nbsp;&nbsp;&nbsp;<code>P(Kelas) = Jumlah data kelas / Total data training</code><br><br>
                <strong>2. Gaussian Probability (Likelihood):</strong><br>
                &nbsp;&nbsp;&nbsp;<code>P(x|Kelas) = (1 / √(2π × σ²)) × exp(-(x-μ)² / (2σ²))</code><br><br>
                <strong>3. Posterior Probability:</strong><br>
                &nbsp;&nbsp;&nbsp;<code>P(Kelas|X) = P(Kelas) × P(x₁|Kelas) × P(x₂|Kelas) × ... × P(xₙ|Kelas)</code><br><br>
                <strong>4. Keputusan:</strong><br>
                &nbsp;&nbsp;&nbsp;<code>Prediksi = argmax P(Kelas|X)</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        col_j3, col_dl3 = st.columns([2, 1])
        with col_j3:
            judul_manual = st.text_input("Judul untuk file unduhan", value="Perhitungan Manual Naive Bayes", key="judul_manual")
        with col_dl3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(make_excel_download(nb["manual"], f"{judul_manual.replace(' ','_')}.xlsx", f"⬇ Unduh {judul_manual} (.xlsx)", judul_manual), unsafe_allow_html=True)
    else:
        st.info("👈 Setelah preprocessing, atur rasio split di Tab 'Preprocessing' lalu klik **Proses & Klasifikasi Naive Bayes**.")

# ─────────────────────────────────────────────────────────
# TAB 4 — Hasil Prediksi
# ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🎯 Hasil Prediksi Seluruh Data</div>', unsafe_allow_html=True)
    if nb is not None:
        miskin     = nb["jumlah_miskin"]
        tdk_miskin = nb["jumlah_tidak_miskin"]
        total      = nb["total_data"]
        benar      = nb["benar"]
        salah      = nb["salah"]

        st.markdown(
            notif_html("nb", "🎯",
                       f"Prediksi Selesai — {benar} dari {benar+salah} data uji diprediksi dengan benar",
                       f"Split: <strong>{nb['train_pct']}% Training / {nb['test_pct']}% Testing</strong>.",
                       [f"✅ Benar: {benar}", f"❌ Salah: {salah}", f"📊 Total uji: {benar+salah}",
                        f"🏅 Akurasi: {(benar/(benar+salah)*100) if (benar+salah)>0 else 0:.2f}%"]),
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="metric-row">
            <div class="metric-card mc-terra"><span class="mc-icon">🏠</span><div class="mc-label">Prediksi Miskin</div><div class="mc-value cv-terra">{miskin}</div></div>
            <div class="metric-card mc-sage"><span class="mc-icon">✅</span><div class="mc-label">Prediksi Tidak Miskin</div><div class="mc-value cv-sage">{tdk_miskin}</div></div>
            <div class="metric-card mc-brown"><span class="mc-icon">📋</span><div class="mc-label">Total Seluruh Data</div><div class="mc-value cv-brown">{total}</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

        col_t, col_p = st.columns([2, 1])
        with col_t:
            st.dataframe(nb["hasil_semua"], use_container_width=True, height=400)
        with col_p:
            fig_pie2, ax_p2 = plt.subplots(figsize=(4, 4))
            fig_pie2.patch.set_facecolor("#FFFFFF")
            ax_p2.set_facecolor("#FFFFFF")
            pie_vals   = [miskin, tdk_miskin]
            pie_labels = ["Miskin", "Tidak Miskin"]
            pie_colors = ["#C2570A", "#3D7054"]
            wedges2, texts2, autotexts2 = ax_p2.pie(
                pie_vals, labels=pie_labels, autopct="%1.1f%%", colors=pie_colors, startangle=90,
                wedgeprops={"edgecolor": "#fff", "linewidth": 2.5},
                textprops={"color": "#1A0A00", "fontsize": 10, "fontweight": "700"},
            )
            for at in autotexts2: at.set_color("white"); at.set_fontweight("bold")
            centre2 = plt.Circle((0, 0), 0.55, fc="#ffffff")
            ax_p2.add_patch(centre2)
            ax_p2.text(0, 0, str(total), ha="center", va="center", fontsize=20, fontweight="bold", color="#3D1C02")
            ax_p2.set_title("Distribusi Kemiskinan\nSeluruh Data", fontsize=10, color="#7A5C44", pad=6)
            plt.tight_layout()
            st.pyplot(fig_pie2)

        st.markdown("---")
        col_j4, col_dl4 = st.columns([2, 1])
        with col_j4:
            judul_pred = st.text_input("Judul untuk file unduhan", value="Hasil Prediksi Naive Bayes", key="judul_pred")
        with col_dl4:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(make_excel_download(nb["hasil_semua"], f"{judul_pred.replace(' ','_')}.xlsx", f"⬇ Unduh {judul_pred} (.xlsx)", judul_pred), unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="formula-box">
                <div class="formula-title">📐 Rumus Akurasi</div>
                <code>Akurasi = (Jumlah Prediksi Benar / Total Data Uji) × 100%</code><br>
                &nbsp;&nbsp;&nbsp;= ({benar} / {benar+salah}) × 100% = <strong>{(benar/(benar+salah)*100) if (benar+salah)>0 else 0:.2f}%</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("👈 Setelah preprocessing, atur rasio split di Tab 'Preprocessing' lalu klik **Proses & Klasifikasi Naive Bayes**.")

# ─────────────────────────────────────────────────────────
# TAB 5 — Confusion Matrix
# ─────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
    if nb is not None:
        cm_raw = nb["cm"]
        tn = cm_raw[0, 0] if cm_raw.shape[0] > 1 else 0
        fp = cm_raw[0, 1] if cm_raw.shape[1] > 1 else 0
        fn = cm_raw[1, 0] if cm_raw.shape[0] > 1 else 0
        tp = cm_raw[1, 1] if cm_raw.shape[1] > 1 else 0

        st.markdown(
            notif_html("nb", "🔲", f"Confusion Matrix — Split {nb['train_pct']}%/{nb['test_pct']}%",
                       f"TN: <strong>{tn}</strong> | TP: <strong>{tp}</strong> | FP: <strong>{fp}</strong> | FN: <strong>{fn}</strong>",
                       [f"✅ TN: {tn}", f"✅ TP: {tp}", f"⚠️ FP: {fp}", f"⚠️ FN: {fn}", f"🏅 Akurasi: {nb['accuracy']*100:.2f}%"]),
            unsafe_allow_html=True,
        )

        col_img, col_exp = st.columns([1, 1])
        with col_img:
            st.pyplot(nb["fig_cm"])
            col_j5, col_dl5 = st.columns([2, 1])
            with col_j5:
                judul_cm = st.text_input("Judul file", value="Confusion Matrix Keluarga Miskin", key="judul_cm")
            with col_dl5:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(make_png_download(nb["fig_cm"], f"{judul_cm.replace(' ','_')}.png", f"⬇ Unduh {judul_cm} (.png)"), unsafe_allow_html=True)

        with col_exp:
            st.markdown(
                """<div class="info-box">
                <strong>Cara Membaca Confusion Matrix</strong><br><br>
                <strong style="color:#3D7054;">True Negative (TN)</strong> — Prediksi <em>Tidak Miskin</em>, Aktual <em>Tidak Miskin</em> ✅<br><br>
                <strong style="color:#C0392B;">False Positive (FP)</strong> — Prediksi <em>Miskin</em>, Aktual <em>Tidak Miskin</em> ❌<br><br>
                <strong style="color:#C0392B;">False Negative (FN)</strong> — Prediksi <em>Tidak Miskin</em>, Aktual <em>Miskin</em> ❌<br><br>
                <strong style="color:#3D7054;">True Positive (TP)</strong> — Prediksi <em>Miskin</em>, Aktual <em>Miskin</em> ✅
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-bottom:.9rem">
                    <div style="background:#fff;border:1.5px solid #E8D5C0;border-top:3px solid #3D7054;border-radius:12px;padding:.9rem 1rem;text-align:center;">
                        <div style="font-size:.6rem;color:#A88B78;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.3rem;font-weight:700;">True Negative</div>
                        <div style="font-family:'Playfair Display',serif;font-size:2.4rem;color:#3D7054;line-height:1">{tn}</div>
                    </div>
                    <div style="background:#fff;border:1.5px solid #E8D5C0;border-top:3px solid #C0392B;border-radius:12px;padding:.9rem 1rem;text-align:center;">
                        <div style="font-size:.6rem;color:#A88B78;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.3rem;font-weight:700;">False Positive</div>
                        <div style="font-family:'Playfair Display',serif;font-size:2.4rem;color:#C0392B;line-height:1">{fp}</div>
                    </div>
                    <div style="background:#fff;border:1.5px solid #E8D5C0;border-top:3px solid #C0392B;border-radius:12px;padding:.9rem 1rem;text-align:center;">
                        <div style="font-size:.6rem;color:#A88B78;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.3rem;font-weight:700;">False Negative</div>
                        <div style="font-family:'Playfair Display',serif;font-size:2.4rem;color:#C0392B;line-height:1">{fn}</div>
                    </div>
                    <div style="background:#fff;border:1.5px solid #E8D5C0;border-top:3px solid #3D7054;border-radius:12px;padding:.9rem 1rem;text-align:center;">
                        <div style="font-size:.6rem;color:#A88B78;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.3rem;font-weight:700;">True Positive</div>
                        <div style="font-family:'Playfair Display',serif;font-size:2.4rem;color:#3D7054;line-height:1">{tp}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            acc = nb["accuracy"]
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#1A0A00,#3D1C02);border-radius:16px;
                            padding:1.4rem 1.6rem;text-align:center;border-bottom:4px solid #C2570A;
                            box-shadow:0 10px 36px rgba(26,10,0,0.28);">
                    <div style="font-size:.62rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:2px;margin-bottom:.4rem;font-weight:700;">Akurasi Model</div>
                    <div style="font-family:'Playfair Display',serif;font-size:3.8rem;color:#FFAD55;line-height:1">{acc*100:.2f}%</div>
                    <div style="font-size:.78rem;color:rgba(255,255,255,0.35);margin-top:.5rem;">{tn+tp} prediksi benar dari {tn+tp+fn+fp} data uji</div>
                    <div style="font-size:.68rem;color:rgba(224,112,48,0.55);margin-top:.3rem;">Split: {nb['train_pct']}% Train / {nb['test_pct']}% Test</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="formula-box" style="margin-top:.9rem;">
                    <div class="formula-title">📐 Rumus Metrik dari Confusion Matrix</div>
                    <code>Akurasi  = (TP + TN) / (TP + TN + FP + FN)</code><br>
                    <code>Presisi  = TP / (TP + FP)</code><br>
                    <code>Recall   = TP / (TP + FN)</code><br>
                    <code>F1-Score = 2 × (Presisi × Recall) / (Presisi + Recall)</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("👈 Setelah preprocessing, atur rasio split di Tab 'Preprocessing' lalu klik **Proses & Klasifikasi Naive Bayes**.")

# ─────────────────────────────────────────────────────────
# TAB 6 — Classification Report
# ─────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">📈 Laporan Klasifikasi Lengkap</div>', unsafe_allow_html=True)
    if nb is not None:
        report         = nb["report"].copy()
        report_display = report.round(4)

        st.dataframe(report_display, use_container_width=True)

        valid_classes = [c for c in nb["classes_label"] if c in report.index]
        if valid_classes:
            metrics_df = report.loc[valid_classes, ["precision", "recall", "f1-score"]]

            fig_bar, ax_b = plt.subplots(figsize=(6, 3.5))
            fig_bar.patch.set_facecolor("#FFFFFF")
            ax_b.set_facecolor("#FDF6EE")
            x = np.arange(len(metrics_df))
            w = 0.25
            bar_palette = ["#3D1C02", "#C2570A", "#C49A1A"]
            for idx, (col, color) in enumerate(zip(metrics_df.columns, bar_palette)):
                ax_b.bar(x + idx*w, metrics_df[col], w, label=col.capitalize(), color=color, edgecolor="#ffffff", linewidth=1.5)
            ax_b.set_xticks(x + w)
            ax_b.set_xticklabels([f"Kelas '{c}'" for c in valid_classes], color="#1A0A00", fontsize=10)
            ax_b.set_ylim(0, 1.2)
            ax_b.tick_params(colors="#7A5C44", labelsize=9)
            ax_b.legend(facecolor="#FFFFFF", edgecolor="#E8D5C0", labelcolor="#1A0A00", fontsize=9)
            for spine in ax_b.spines.values(): spine.set_edgecolor("#E8D5C0")
            ax_b.axhline(1.0, color="#E8D5C0", linewidth=0.8, linestyle="--")
            ax_b.set_title(f"Perbandingan Metrik Evaluasi per Kelas (Split {nb['train_pct']}%/{nb['test_pct']}%)",
                           color="#3D1C02", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_bar)

            # Interpretasi per kelas
            try:
                def interpret(val):
                    if val >= 0.90:   return ("Sangat Baik", "#3D7054")
                    elif val >= 0.75: return ("Baik", "#C2570A")
                    elif val >= 0.60: return ("Cukup", "#C49A1A")
                    else:             return ("Perlu Ditingkatkan", "#C0392B")

                rows_html = ""
                for cls in valid_classes:
                    prec_ = float(report.loc[cls, "precision"])
                    rec_  = float(report.loc[cls, "recall"])
                    f1_   = float(report.loc[cls, "f1-score"])
                    ip, cp = interpret(prec_)
                    ir, cr = interpret(rec_)
                    if_, cf = interpret(f1_)
                    label_display = "Miskin (ya)" if str(cls).lower() == "ya" else "Tidak Miskin (tidak)"
                    border_color  = "#C2570A" if str(cls).lower() == "ya" else "#3D7054"
                    rows_html += f"""
                    <div style="flex:1;min-width:260px;background:#fff;border:1.5px solid #E8D5C0;
                                border-top:4px solid {border_color};border-radius:14px;padding:1.1rem 1.3rem;
                                box-shadow:0 2px 10px rgba(61,28,2,0.07);">
                        <div style="font-size:.7rem;font-weight:800;color:{border_color};text-transform:uppercase;letter-spacing:1.2px;margin-bottom:.7rem;">📋 Kelas — {label_display}</div>
                        <div style="display:flex;flex-direction:column;gap:.45rem;">
                            <div style="display:flex;justify-content:space-between;align-items:center;background:#FDF6EE;border-radius:8px;padding:.45rem .8rem;">
                                <span style="font-size:.8rem;color:#3D2B1F;font-weight:600;">🎯 Precision</span>
                                <span style="font-size:.88rem;font-weight:800;color:{cp};">{prec_:.2f} — {ip}</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;align-items:center;background:#FDF6EE;border-radius:8px;padding:.45rem .8rem;">
                                <span style="font-size:.8rem;color:#3D2B1F;font-weight:600;">🔍 Recall</span>
                                <span style="font-size:.88rem;font-weight:800;color:{cr};">{rec_:.2f} — {ir}</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;align-items:center;background:#FDF6EE;border-radius:8px;padding:.45rem .8rem;">
                                <span style="font-size:.8rem;color:#3D2B1F;font-weight:600;">⚖️ F1-Score</span>
                                <span style="font-size:.88rem;font-weight:800;color:{cf};">{f1_:.2f} — {if_}</span>
                            </div>
                        </div>
                    </div>
                    """

                st.markdown(
                    f'<div style="margin-top:1.2rem;">'
                    f'<p style="font-size:.68rem;color:#7A5C44;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.8rem;">💬 Interpretasi Diagram — Apa Artinya?</p>'
                    f'<div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:.8rem;">{rows_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        st.markdown(
            """
            <div class="formula-box">
                <div class="formula-title">📐 Keterangan Metrik Evaluasi</div>
                <strong>Precision</strong> — Dari semua yang diprediksi positif, berapa yang benar-benar positif:<br>
                &nbsp;&nbsp;&nbsp;<code>Precision = TP / (TP + FP)</code><br><br>
                <strong>Recall</strong> — Dari semua data positif aktual, berapa yang berhasil dideteksi:<br>
                &nbsp;&nbsp;&nbsp;<code>Recall = TP / (TP + FN)</code><br><br>
                <strong>F1-Score</strong> — Harmonic mean antara Precision dan Recall:<br>
                &nbsp;&nbsp;&nbsp;<code>F1 = 2 × (Precision × Recall) / (Precision + Recall)</code><br><br>
                <strong>Support</strong> — Jumlah data aktual per kelas dalam data uji
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        col_dl_a, col_dl_b = st.columns(2)
        with col_dl_a:
            judul_rpt = st.text_input("Judul file laporan", value="Laporan Klasifikasi Keluarga Miskin", key="judul_rpt")
            st.markdown(make_excel_download(report_display, f"{judul_rpt.replace(' ','_')}.xlsx", f"⬇ Unduh {judul_rpt} (.xlsx)", judul_rpt), unsafe_allow_html=True)
        with col_dl_b:
            st.markdown("<br>", unsafe_allow_html=True)
            if "fig_bar" in dir():
                st.markdown(make_png_download(fig_bar, f"{judul_rpt.replace(' ','_')}_chart.png", f"⬇ Unduh Grafik (.png)"), unsafe_allow_html=True)
    else:
        st.info("👈 Setelah preprocessing, atur rasio split di Tab 'Preprocessing' lalu klik **Proses & Klasifikasi Naive Bayes**.")

# ─────────────────────────────────────────────────────────
# TAB 7 — Input Data Baru
# ─────────────────────────────────────────────────────────
with tab7:
    st.markdown('<div class="section-header">➕ Input & Prediksi Data Keluarga Baru</div>', unsafe_allow_html=True)

    if nb is None:
        st.markdown(
            notif_html("warn", "⚠️", "Model Belum Dilatih",
                       "Harap jalankan proses Naive Bayes terlebih dahulu. Ikuti langkah: Upload Data → Preprocessing → Atur Split → Klik tombol **Proses & Klasifikasi Naive Bayes** di Tab Preprocessing.",
                       ["1️⃣ Upload Data", "2️⃣ Preprocessing", "3️⃣ Atur Split di Tab Preprocessing", "4️⃣ Klik tombol Proses NB"]),
            unsafe_allow_html=True,
        )
    else:
        le_pekerjaan = nb["le_pekerjaan"]
        le_rumah     = nb["le_rumah"]
        le_target    = nb["le_target"]

        mode = st.radio(
            "Pilih metode input data:",
            ["📝 Input Manual (Satu per Satu)", "📂 Upload File Banyak Data (CSV/Excel)"],
            horizontal=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── MODE 1 — Input Manual ─────────────────────────
        if mode == "📝 Input Manual (Satu per Satu)":
            st.markdown(
                '<div class="info-box">🏠 Masukkan data atribut kepala keluarga di bawah ini. <strong>Status kemiskinan akan diprediksi secara otomatis</strong> oleh model Gaussian Naive Bayes yang telah dilatih.</div>',
                unsafe_allow_html=True,
            )

            pekerjaan_options = list(le_pekerjaan.classes_)
            rumah_options     = list(le_rumah.classes_)

            with st.form("form_tambah_keluarga", clear_on_submit=True):
                st.markdown('<p style="font-size:.72rem;color:#7A5C44;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.8rem;">📝 Identitas Kepala Keluarga</p>', unsafe_allow_html=True)
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    inp_nama       = st.text_input("Nama Kepala Keluarga", placeholder="Masukkan nama lengkap...")
                    inp_pekerjaan  = st.selectbox("Pekerjaan", options=pekerjaan_options)
                    inp_penghasilan = st.number_input("Penghasilan (Rp)", min_value=0, value=500000, step=50000)
                with col_f2:
                    inp_tanggungan = st.number_input("Jumlah Tanggungan", min_value=0, max_value=20, value=2, step=1)
                    inp_rumah      = st.selectbox("Status Kepemilikan Rumah", options=rumah_options)

                submitted = st.form_submit_button("🔍 Prediksi & Tambahkan ke Riwayat", use_container_width=True)

            if submitted:
                if not inp_nama.strip():
                    st.markdown(notif_html("warn", "⚠️", "Nama Kosong", "Nama kepala keluarga tidak boleh kosong.", None), unsafe_allow_html=True)
                else:
                    pk_enc  = int(le_pekerjaan.transform([inp_pekerjaan])[0])
                    rmh_enc = int(le_rumah.transform([inp_rumah])[0])
                    fitur_input = pd.DataFrame([{
                        "PEKERJAAN":                pk_enc,
                        "PENGHASILAN":              inp_penghasilan,
                        "TANGGUNGAN":               inp_tanggungan,
                        "STATUS KEPEMILIKAN RUMAH": rmh_enc,
                    }])
                    label_pred     = nb["model"].predict(fitur_input)[0]
                    label_text_raw = le_target.inverse_transform([label_pred])[0]
                    is_miskin      = label_text_raw.lower().strip() == "ya"
                    label_text     = "Termasuk Keluarga Miskin" if is_miskin else "Tidak Termasuk Keluarga Miskin"
                    css_class      = "pred-miskin" if is_miskin else "pred-tidak"
                    icon_pred      = "🏠" if is_miskin else "✅"
                    color_pred     = "#C2570A" if is_miskin else "#3D7054"

                    st.markdown(
                        notif_html(
                            "upload" if is_miskin else "prep",
                            icon_pred,
                            f"Hasil Prediksi: {inp_nama.strip()}",
                            f"Kepala keluarga <strong>{inp_nama.strip()}</strong> diprediksi: <strong style='color:{color_pred}'>{label_text}</strong>",
                            [f"Pekerjaan: {inp_pekerjaan}", f"Penghasilan: Rp{inp_penghasilan:,.0f}",
                             f"Tanggungan: {inp_tanggungan}", f"Status Rumah: {inp_rumah}"],
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="pred-result-box {css_class}">'
                        f'<div class="pred-label" style="color:{color_pred}">{icon_pred} {label_text}</div>'
                        f'<div class="pred-sub">Diprediksi oleh Gaussian Naive Bayes untuk <strong>{inp_nama.strip()}</strong></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    new_row = pd.DataFrame([{
                        "NAMA KEPALA KELUARGA":    inp_nama.strip(),
                        "PEKERJAAN":               inp_pekerjaan,
                        "PENGHASILAN":             inp_penghasilan,
                        "TANGGUNGAN":              inp_tanggungan,
                        "STATUS KEPEMILIKAN RUMAH": inp_rumah,
                        "KETERANGAN (PREDIKSI)":   label_text_raw,
                    }])
                    st.session_state["data_tambahan"] = pd.concat(
                        [st.session_state["data_tambahan"], new_row], ignore_index=True
                    )

        # ── MODE 2 — Upload File ──────────────────────────
        else:
            st.markdown(
                """<div class="info-box">
                📂 Upload file CSV atau Excel berisi data kepala keluarga. File harus memiliki kolom:<br>
                <strong>NAMA KEPALA KELUARGA</strong>, <strong>PEKERJAAN</strong> (sesuai data training),
                <strong>PENGHASILAN</strong> (angka Rp), <strong>TANGGUNGAN</strong> (angka),
                <strong>STATUS KEPEMILIKAN RUMAH</strong> (sesuai data training)
                </div>""",
                unsafe_allow_html=True,
            )

            template_df = pd.DataFrame([{
                "NAMA KEPALA KELUARGA": "Wilfridus Emanuel Nesi",
                "PEKERJAAN":           "Petani",
                "PENGHASILAN":         250000,
                "TANGGUNGAN":          5,
                "STATUS KEPEMILIKAN RUMAH": "Milik Sendiri",
            }])
            st.markdown(make_excel_download(template_df, "template_input_keluarga.xlsx", "⬇ Unduh Template File (.xlsx)"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            uploaded_bulk = st.file_uploader("Upload file data keluarga (CSV atau Excel)", type=["csv", "xlsx"], key="uploader_bulk")

            if uploaded_bulk is not None:
                try:
                    if uploaded_bulk.name.endswith(".csv"):
                        df_bulk = pd.read_csv(uploaded_bulk)
                    else:
                        df_bulk = pd.read_excel(uploaded_bulk)

                    df_bulk.columns = df_bulk.columns.str.upper().str.strip()
                    required_cols   = ["PEKERJAAN", "PENGHASILAN", "TANGGUNGAN", "STATUS KEPEMILIKAN RUMAH"]
                    missing_cols    = [c for c in required_cols if c not in df_bulk.columns]

                    if missing_cols:
                        st.markdown(notif_html("warn", "❌", "Kolom Tidak Ditemukan",
                                               f"Kolom berikut tidak ada dalam file: {', '.join(missing_cols)}", None), unsafe_allow_html=True)
                    else:
                        st.markdown(
                            notif_html("upload", "📂", f"File Berhasil Dibaca — {len(df_bulk)} baris data ditemukan",
                                       f"File <strong>{uploaded_bulk.name}</strong> siap diproses.",
                                       [f"📋 Total baris: {len(df_bulk)}", f"📊 Kolom: {len(df_bulk.columns)}"]),
                            unsafe_allow_html=True,
                        )
                        st.dataframe(df_bulk.head(10), use_container_width=True)

                        if st.button("🔍 Prediksi Semua Data dari File", use_container_width=False):
                            df_bulk["PEKERJAAN"] = df_bulk["PEKERJAAN"].astype(str)
                            df_bulk["STATUS KEPEMILIKAN RUMAH"] = df_bulk["STATUS KEPEMILIKAN RUMAH"].astype(str)
                            df_bulk["PENGHASILAN"] = pd.to_numeric(df_bulk["PENGHASILAN"].astype(str).str.replace(".", "").str.replace(",", "").str.replace("RP", "").str.strip(), errors="coerce").fillna(0)
                            df_bulk["TANGGUNGAN"]  = pd.to_numeric(df_bulk["TANGGUNGAN"],  errors="coerce").fillna(0).astype(int)

                            known_pekerjaan = set(le_pekerjaan.classes_)
                            known_rumah     = set(le_rumah.classes_)
                            df_bulk["PEKERJAAN"] = df_bulk["PEKERJAAN"].apply(lambda x: x if x in known_pekerjaan else le_pekerjaan.classes_[0])
                            df_bulk["STATUS KEPEMILIKAN RUMAH"] = df_bulk["STATUS KEPEMILIKAN RUMAH"].apply(lambda x: x if x in known_rumah else le_rumah.classes_[0])

                            pk_enc  = le_pekerjaan.transform(df_bulk["PEKERJAAN"])
                            rmh_enc = le_rumah.transform(df_bulk["STATUS KEPEMILIKAN RUMAH"])

                            X_bulk = pd.DataFrame({
                                "PEKERJAAN":                pk_enc,
                                "PENGHASILAN":              df_bulk["PENGHASILAN"].values,
                                "TANGGUNGAN":               df_bulk["TANGGUNGAN"].values,
                                "STATUS KEPEMILIKAN RUMAH": rmh_enc,
                            })
                            labels_pred = le_target.inverse_transform(nb["model"].predict(X_bulk)).tolist()

                            hasil_bulk = df_bulk.copy()
                            hasil_bulk["KETERANGAN (PREDIKSI)"] = labels_pred
                            if "NAMA KEPALA KELUARGA" not in hasil_bulk.columns:
                                hasil_bulk.insert(0, "NAMA KEPALA KELUARGA", "–")

                            cols_order = ["NAMA KEPALA KELUARGA", "PEKERJAAN", "PENGHASILAN", "TANGGUNGAN", "STATUS KEPEMILIKAN RUMAH", "KETERANGAN (PREDIKSI)"]
                            cols_exist = [c for c in cols_order if c in hasil_bulk.columns]
                            hasil_bulk = hasil_bulk[cols_exist]

                            st.session_state["data_tambahan"] = pd.concat(
                                [st.session_state["data_tambahan"], hasil_bulk], ignore_index=True
                            )

                            miskin_count = sum(1 for l in labels_pred if str(l).lower() == "ya")
                            tidak_count  = len(labels_pred) - miskin_count

                            st.markdown(
                                notif_html("prep", "✅", f"Prediksi Selesai — {len(df_bulk)} data berhasil diproses!",
                                           "Semua data dari file telah diprediksi dan ditambahkan ke riwayat.",
                                           [f"🏠 Miskin: {miskin_count}", f"📋 Tidak Miskin: {tidak_count}", f"📊 Total: {len(df_bulk)}"]),
                                unsafe_allow_html=True,
                            )
                except Exception as e:
                    st.markdown(notif_html("warn", "❌", "Error Membaca File", f"Terjadi error: {e}", None), unsafe_allow_html=True)

        # ── Riwayat Prediksi ──────────────────────────────
        st.markdown("---")
        dt = st.session_state["data_tambahan"]

        col_hd, col_hapus = st.columns([3, 1])
        with col_hd:
            st.markdown(
                f'<div class="section-header">📋 Riwayat Prediksi Keluarga '
                f'<span style="color:#A88B78;font-size:.85rem;font-family:Nunito,sans-serif;font-weight:500;">({len(dt)} entri)</span></div>',
                unsafe_allow_html=True,
            )
        with col_hapus:
            if st.button("🗑 Hapus Semua Riwayat", key="hapus_riwayat"):
                st.session_state["data_tambahan"] = pd.DataFrame(
                    columns=["NAMA KEPALA KELUARGA", "PEKERJAAN", "PENGHASILAN",
                             "TANGGUNGAN", "STATUS KEPEMILIKAN RUMAH", "KETERANGAN (PREDIKSI)"]
                )
                st.rerun()

        if len(dt) > 0:
            dt_display = dt.copy()
            dt_display.insert(0, "No", range(1, len(dt_display)+1))
            st.dataframe(dt_display, use_container_width=True, height=320)

            label_dist = dt["KETERANGAN (PREDIKSI)"].astype(str).str.lower().str.strip().value_counts()
            miskin_n   = int(label_dist.get("ya", 0))
            tidak_n    = int(label_dist.get("tidak", 0))

            st.markdown(
                notif_html("prep", "📊", f"Ringkasan Riwayat — {len(dt)} keluarga telah diprediksi",
                           "Berikut distribusi kemiskinan dari semua data yang telah dimasukkan.",
                           [f"🏠 Miskin: {miskin_n}", f"📋 Tidak Miskin: {tidak_n}", f"👥 Total: {len(dt)}"]),
                unsafe_allow_html=True,
            )

            st.markdown("---")
            col_dl_a, col_dl_b, col_dl_c = st.columns(3)
            with col_dl_a:
                judul_tambahan = st.text_input("Judul file riwayat", value="Riwayat Prediksi Keluarga Miskin", key="judul_tambahan")
            with col_dl_b:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(make_excel_download(dt, f"{judul_tambahan.replace(' ','_')}.xlsx", f"⬇ Unduh {judul_tambahan} (.xlsx)", judul_tambahan), unsafe_allow_html=True)
            with col_dl_c:
                if df_clean is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    dt_gabung = dt.rename(columns={"KETERANGAN (PREDIKSI)": "KETERANGAN"})
                    gabungan  = pd.concat([df_clean, dt_gabung], ignore_index=True)
                    st.markdown(make_excel_download(gabungan, f"{judul_tambahan.replace(' ','_')}_Gabungan.xlsx", f"⬇ Unduh Data Gabungan (.xlsx)", judul_tambahan+" (Gabungan)"), unsafe_allow_html=True)

            # Bar chart distribusi
            st.markdown('<p style="font-size:.68rem;color:#7A5C44;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.65rem;margin-top:1rem;">Distribusi Kemiskinan — Data Riwayat</p>', unsafe_allow_html=True)
            fig_dt, ax_dt = plt.subplots(figsize=(4, 2.4))
            fig_dt.patch.set_facecolor("#FFFFFF")
            ax_dt.set_facecolor("#FDF6EE")
            label_dist_renamed = label_dist.rename({"ya": "Miskin", "tidak": "Tidak Miskin"})
            bar_c_dt = ["#C2570A" if k == "ya" else "#3D7054" for k in label_dist.index]
            bar_label_dt = ["Miskin" if k == "ya" else "Tidak Miskin" for k in label_dist.index]
            bars_dt = ax_dt.bar(bar_label_dt, label_dist.values, color=bar_c_dt, edgecolor="#ffffff", width=0.5, linewidth=1.5)
            ax_dt.tick_params(colors="#7A5C44")
            ax_dt.set_ylabel("Jumlah Keluarga", color="#7A5C44", fontsize=9)
            for spine in ax_dt.spines.values(): spine.set_edgecolor("#E8D5C0")
            for bar in bars_dt:
                ax_dt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, str(int(bar.get_height())), ha="center", va="bottom", color="#1A0A00", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_dt)
        else:
            st.markdown(
                '<div style="text-align:center;color:#A88B78;padding:2.5rem 2rem;'
                'border:2px dashed #E8D5C0;border-radius:16px;margin-top:1rem;background:#FDF6EE;">'
                '<div style="font-size:2.2rem;margin-bottom:.7rem;">🏠</div>'
                '<div style="font-weight:800;color:#7A5C44;margin-bottom:.3rem;">Belum Ada Data yang Dimasukkan</div>'
                '<div style="font-size:.8rem;">Isi form manual atau upload file di atas, lalu klik prediksi.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#A88B78;font-size:.78rem;padding:.5rem 0 1rem;
                letter-spacing:.3px;font-family:'Nunito',sans-serif;">
        🏠 Sistem Penentuan Keluarga Miskin &nbsp;·&nbsp;
        Desa Nansean, Kabupaten TTU &nbsp;·&nbsp;
        Algoritma Gaussian Naive Bayes &nbsp;·&nbsp;
        Dibangun dengan <span style="color:#C2570A;">♥</span> menggunakan Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)