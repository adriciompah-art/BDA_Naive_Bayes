# =========================================================
# INSTALL:  pip install streamlit openpyxl scikit-learn
#           pandas numpy seaborn matplotlib
# RUN    :  streamlit run klasifikasi_bsm.py
# =========================================================

import io
import base64

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
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
    page_title="Klasifikasi BSM – Naive Bayes",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS — Clean Sage & Warm Cream Theme
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=DM+Mono:wght@400;500&display=swap');

    /* ── COLOUR TOKENS ──────────────────────────────────── */
    :root {
        /* Backgrounds — warm off-white with sage tones */
        --bg:           #F7F6F1;
        --bg2:          #EFEDE6;
        --surface:      #FFFFFF;
        --surface2:     #F9F8F4;
        --surface3:     #F0EEE8;

        /* Borders */
        --border:       #DDD9CF;
        --border2:      #C8C3B6;

        /* Primary — deep forest green */
        --green:        #2D5A3D;
        --green-mid:    #3D7A54;
        --green-light:  #5A9E70;
        --green-pale:   #D4EDDA;
        --green-glow:   rgba(45,90,61,0.15);

        /* Accent — warm terracotta */
        --terra:        #C4622D;
        --terra-light:  #E07848;
        --terra-pale:   #FAE8DC;
        --terra-glow:   rgba(196,98,45,0.15);

        /* Accent 2 — dusty gold */
        --gold:         #B89A3E;
        --gold-light:   #D4B85A;
        --gold-pale:    #F5EDD0;
        --gold-glow:    rgba(184,154,62,0.15);

        /* Accent 3 — slate blue */
        --slate:        #3A5470;
        --slate-light:  #5278A0;
        --slate-pale:   #DDE5EF;
        --slate-glow:   rgba(58,84,112,0.15);

        /* Accent 4 — dusty rose */
        --rose:         #A85060;
        --rose-light:   #C4687A;
        --rose-pale:    #F5DDE2;
        --rose-glow:    rgba(168,80,96,0.15);

        /* Text */
        --text:         #1C1A16;
        --text-body:    #2E2C26;
        --text-dim:     #5C5748;
        --text-muted:   #8C8578;

        /* Shadows */
        --shadow-sm:    0 1px 4px rgba(28,26,22,0.08), 0 2px 8px rgba(28,26,22,0.04);
        --shadow-md:    0 4px 16px rgba(28,26,22,0.10), 0 2px 6px rgba(28,26,22,0.06);
        --shadow-lg:    0 8px 32px rgba(28,26,22,0.12), 0 4px 12px rgba(28,26,22,0.08);

        --radius:       10px;
        --radius-lg:    16px;
        --radius-xl:    22px;
    }

    /* ── RESET BASE ─────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text-body);
        font-size: 14px;
        line-height: 1.65;
    }

    /* ── SIDEBAR ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1C2E20 0%, #24382A 55%, #1A2C20 100%) !important;
        border-right: 1px solid rgba(90,158,112,0.18) !important;
        box-shadow: 3px 0 24px rgba(20,30,22,0.30) !important;
        width: 295px !important;
        min-width: 295px !important;
        max-width: 295px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0 !important;
        background: transparent !important;
    }
    section[data-testid="stSidebar"] * {
        color: rgba(255,252,240,0.80) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFCF0 !important;
    }

    /* Sidebar file uploader */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(90,158,112,0.08) !important;
        border: 1.5px dashed rgba(90,158,112,0.45) !important;
        border-radius: 10px !important;
        transition: all 0.25s !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(90,158,112,0.80) !important;
        background: rgba(90,158,112,0.14) !important;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #3D7A54, #5A9E70) !important;
        color: #0E1A12 !important;
        border: none !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        font-size: 0.80rem !important;
        padding: 0.58rem 1rem !important;
        box-shadow: 0 3px 12px rgba(45,90,61,0.35) !important;
        transition: all 0.22s !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #5A9E70, #7FBE8F) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(45,90,61,0.42) !important;
    }

    /* ── SIDEBAR LOGO AREA ──────────────────────────────── */
    .sb-logo {
        padding: 1.5rem 1.3rem 1.1rem;
        background: linear-gradient(135deg, rgba(45,90,61,0.35), rgba(196,98,45,0.12));
        border-bottom: 1px solid rgba(255,252,240,0.08);
        margin-bottom: 0.4rem;
    }
    .sb-logo-row {
        display: flex; align-items: center; gap: 0.85rem; margin-bottom: 0.55rem;
    }
    .sb-logo-icon {
        width: 46px; height: 46px;
        background: linear-gradient(135deg, #2D5A3D, #C4622D);
        border-radius: 13px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.45rem;
        box-shadow: 0 3px 14px rgba(45,90,61,0.40);
        flex-shrink: 0;
    }
    .sb-logo-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.22rem;
        color: #FFFCF0 !important;
        line-height: 1.15;
        font-weight: 600;
    }
    .sb-logo-title span { color: #9FD4AE !important; }
    .sb-logo-desc {
        font-size: 0.66rem !important;
        color: rgba(255,252,240,0.35) !important;
        letter-spacing: 0.3px;
    }

    /* ── SIDEBAR SECTION LABEL ──────────────────────────── */
    .sb-sec {
        font-size: 0.60rem !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: #9FD4AE !important;
        margin: 1.3rem 0 0.5rem !important;
        display: block;
        padding-left: 0.1rem;
    }

    /* ── PIPELINE STEPS ─────────────────────────────────── */
    .pipeline { display: flex; flex-direction: column; gap: 0.25rem; }
    .pipeline-step {
        display: flex; align-items: center; gap: 0.6rem;
        padding: 0.5rem 0.85rem;
        border-radius: 9px;
        border: 1px solid transparent;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        color: rgba(255,252,240,0.28) !important;
        transition: all 0.2s;
    }
    .pipeline-step.done {
        background: rgba(45,90,61,0.22) !important;
        border-color: rgba(90,158,112,0.38) !important;
        color: #9FD4AE !important;
    }
    .pipeline-step.active {
        background: rgba(196,98,45,0.16) !important;
        border-color: rgba(196,98,45,0.35) !important;
        color: rgba(255,252,240,0.85) !important;
    }
    .step-num {
        width: 20px; height: 20px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.66rem; font-weight: 700;
        flex-shrink: 0;
        background: rgba(255,255,255,0.07);
        color: rgba(255,255,255,0.28) !important;
    }
    .step-num.done {
        background: #3D7A54;
        color: #FFFCF0 !important;
        box-shadow: 0 0 8px rgba(45,90,61,0.50);
    }
    .step-num.active {
        background: rgba(196,98,45,0.55);
        color: #fff !important;
    }

    /* ── MAIN CONTENT ───────────────────────────────────── */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 100% !important;
    }

    /* ── PAGE TITLE BAR ─────────────────────────────────── */
    .page-titlebar {
        display: flex;
        align-items: center;
        gap: 1rem;
        background: linear-gradient(110deg, #1C2E20 0%, #24382A 60%, #1C2E20 100%);
        border-radius: var(--radius-lg);
        padding: 1rem 1.6rem;
        margin-bottom: 1.4rem;
        border: 1px solid rgba(90,158,112,0.20);
        box-shadow: var(--shadow-md), inset 0 1px 0 rgba(255,255,255,0.04);
        position: relative; overflow: hidden;
    }
    .page-titlebar::before {
        content: '';
        position: absolute; inset: 0;
        background: radial-gradient(ellipse 45% 100% at 88% 50%, rgba(196,98,45,0.10) 0%, transparent 65%);
        pointer-events: none;
    }
    .ptb-badge {
        width: 42px; height: 42px; flex-shrink: 0;
        background: linear-gradient(135deg, #2D5A3D, #C4622D);
        border-radius: 11px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 3px 12px rgba(45,90,61,0.40);
        position: relative; z-index: 1;
    }
    .ptb-text { position: relative; z-index: 1; flex: 1; }
    .ptb-eyebrow {
        font-size: 0.56rem; font-weight: 700; letter-spacing: 2.2px;
        text-transform: uppercase; color: #9FD4AE; margin-bottom: 0.12rem;
    }
    .ptb-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.18rem; font-weight: 700;
        color: #FFFCF0; line-height: 1.15; letter-spacing: -0.1px;
    }
    .ptb-title .hl { color: #F0BC7A; font-style: italic; }
    .ptb-pills {
        margin-left: auto; display: flex; gap: 0.35rem; align-items: center;
        position: relative; z-index: 1; flex-wrap: wrap;
    }
    .hpill {
        border-radius: 999px; padding: 3px 11px;
        font-size: 0.65rem; font-weight: 700; letter-spacing: 0.2px;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .hpill-green { background: rgba(45,90,61,0.40); color: #9FD4AE !important; }
    .hpill-terra { background: rgba(196,98,45,0.30); color: #F0A070 !important; }
    .hpill-gold  { background: rgba(184,154,62,0.28); color: #F0D080 !important; }

    /* ── NOTIFICATION BANNERS ───────────────────────────── */
    .notif-box {
        border-radius: var(--radius);
        padding: 0.9rem 1.2rem;
        margin: 0.7rem 0 1.1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.85rem;
        border: 1px solid;
        animation: slideIn 0.30s ease;
    }
    @keyframes slideIn {
        from { opacity:0; transform: translateY(-6px); }
        to   { opacity:1; transform: translateY(0); }
    }
    .notif-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 0.1rem; }
    .notif-content { flex: 1; }
    .notif-title { font-weight: 700; font-size: 0.86rem; margin-bottom: 0.18rem; }
    .notif-body { font-size: 0.78rem; line-height: 1.6; }
    .notif-stats { display: flex; gap: 0.5rem; margin-top: 0.55rem; flex-wrap: wrap; }
    .notif-stat {
        border-radius: 7px; padding: 3px 11px;
        font-size: 0.71rem; font-weight: 700;
        display: inline-flex; align-items: center; gap: 4px;
    }

    .notif-upload {
        background: linear-gradient(135deg, #EEF4F0, #E8F2EA);
        border-color: rgba(45,90,61,0.22);
    }
    .notif-upload .notif-title { color: #2D5A3D; }
    .notif-upload .notif-stat { background: rgba(45,90,61,0.10); color: #2D5A3D; }

    .notif-prep {
        background: linear-gradient(135deg, #FBF5EE, #FAF0E6);
        border-color: rgba(196,98,45,0.22);
    }
    .notif-prep .notif-title { color: #9A4020; }
    .notif-prep .notif-stat { background: rgba(196,98,45,0.10); color: #9A4020; }

    .notif-nb {
        background: linear-gradient(135deg, #F0F3F8, #E8EDF5);
        border-color: rgba(58,84,112,0.22);
    }
    .notif-nb .notif-title { color: #2D4060; }
    .notif-nb .notif-stat { background: rgba(58,84,112,0.10); color: #2D4060; }

    .notif-warn {
        background: linear-gradient(135deg, #FBF5E8, #FAF0D8);
        border-color: rgba(184,154,62,0.28);
    }
    .notif-warn .notif-title { color: #7A6010; }
    .notif-warn .notif-stat { background: rgba(184,154,62,0.12); color: #7A6010; }

    /* ── METRIC CARDS ───────────────────────────────────── */
    .metric-row { display: flex; gap: 0.85rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 135px;
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: 1.2rem 1.3rem 1rem;
        position: relative; overflow: hidden;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border);
        transition: transform 0.20s, box-shadow 0.20s;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
    .metric-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }
    .metric-card::after {
        content: '';
        position: absolute; bottom: -18px; right: -18px;
        width: 72px; height: 72px; border-radius: 50%;
        opacity: 0.05;
    }
    .mc-green::before  { background: linear-gradient(90deg, #2D5A3D, #5A9E70); }
    .mc-green::after   { background: #2D5A3D; }
    .mc-terra::before  { background: linear-gradient(90deg, #C4622D, #E07848); }
    .mc-terra::after   { background: #C4622D; }
    .mc-gold::before   { background: linear-gradient(90deg, #B89A3E, #D4B85A); }
    .mc-gold::after    { background: #B89A3E; }
    .mc-slate::before  { background: linear-gradient(90deg, #3A5470, #5278A0); }
    .mc-slate::after   { background: #3A5470; }
    .mc-rose::before   { background: linear-gradient(90deg, #A85060, #C4687A); }
    .mc-rose::after    { background: #A85060; }

    .mc-icon { font-size: 1.2rem; margin-bottom: 0.5rem; display: block; }
    .mc-label {
        font-size: 0.61rem; color: var(--text-muted);
        text-transform: uppercase; letter-spacing: 1.2px;
        font-weight: 700; margin-bottom: 0.28rem;
    }
    .mc-value { font-family: 'Playfair Display', serif; font-size: 2.2rem; line-height: 1; }
    .cv-green { color: #2D5A3D; }
    .cv-terra { color: #C4622D; }
    .cv-gold  { color: #B89A3E; }
    .cv-slate { color: #3A5470; }
    .cv-rose  { color: #A85060; }

    /* ── SECTION HEADER ─────────────────────────────────── */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.22rem;
        font-weight: 600;
        color: var(--text);
        padding: 0.35rem 0 0.35rem 0.95rem;
        border-left: 3px solid var(--green-mid);
        margin-bottom: 1.1rem;
        line-height: 1.25;
        background: linear-gradient(90deg, rgba(45,90,61,0.05), transparent);
        border-radius: 0 7px 7px 0;
    }

    /* ── INFO BOX ───────────────────────────────────────── */
    .info-box {
        background: linear-gradient(135deg, #EEF4F0, #F4F8F0);
        border: 1px solid rgba(45,90,61,0.14);
        border-left: 3px solid #5A9E70;
        border-radius: var(--radius);
        padding: 0.85rem 1.15rem;
        margin-bottom: 1rem;
        font-size: 0.82rem;
        color: var(--text-body);
        line-height: 1.7;
    }

    /* ── FORMULA BOX ────────────────────────────────────── */
    .formula-box {
        background: linear-gradient(135deg, #FBF5EE, #FDF8F2);
        border: 1px solid rgba(196,98,45,0.18);
        border-left: 3px solid #C4622D;
        border-radius: var(--radius);
        padding: 0.95rem 1.2rem;
        margin: 0.85rem 0;
        font-size: 0.81rem;
        color: var(--text-body);
        line-height: 1.9;
    }
    .formula-box code {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        background: rgba(196,98,45,0.10);
        border-radius: 5px;
        padding: 0.1rem 0.4rem;
        color: #9A4020;
    }
    .formula-title {
        font-weight: 700;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1.6px;
        color: #C4622D;
        margin-bottom: 0.55rem;
    }

    /* ── SPLIT CONFIG BOX ───────────────────────────────── */
    .split-config-box {
        background: linear-gradient(135deg, #1C2E20, #24382A);
        border: 1px solid rgba(90,158,112,0.22);
        border-radius: var(--radius-lg);
        padding: 1.3rem 1.5rem;
        margin: 0.9rem 0 1.3rem;
        position: relative; overflow: hidden;
    }
    .split-config-box::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #2D5A3D, #5A9E70, #C4622D);
    }
    .split-config-title {
        font-family: 'Playfair Display', serif;
        font-size: 0.98rem; color: #FFFCF0;
        margin-bottom: 0.28rem;
        display: flex; align-items: center; gap: 0.45rem;
    }
    .split-config-desc {
        font-size: 0.72rem; color: rgba(255,252,240,0.42); margin-bottom: 1rem;
    }
    .split-bar-wrap {
        background: rgba(255,255,255,0.08);
        border-radius: 999px; height: 8px; overflow: hidden; margin-bottom: 0.45rem;
    }
    .split-bar-train { height: 100%; border-radius: 999px; transition: width 0.4s ease; }
    .split-bar-labels {
        display: flex; justify-content: space-between;
        font-size: 0.66rem; font-weight: 700;
    }
    .split-label-train { color: #9FD4AE; }
    .split-label-test  { color: #F0A070; }

    /* ── TABS ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border-bottom: 2px solid var(--border);
        gap: 0; padding: 0 0.4rem;
        border-radius: var(--radius) var(--radius) 0 0;
        box-shadow: var(--shadow-sm);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-muted);
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.74rem;
        padding: 0.85rem 1rem;
        border: none;
        border-bottom: 3px solid transparent;
        margin-bottom: -2px;
        letter-spacing: 0.2px;
        transition: color 0.2s, border-color 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--green); }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: var(--terra) !important;
        border-bottom: 3px solid var(--terra) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 var(--radius) var(--radius);
        padding: 1.7rem 1.9rem;
    }

    /* ── DATAFRAME ──────────────────────────────────────── */
    .stDataFrame {
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        border: 1px solid var(--border) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stDataFrame thead th {
        background: linear-gradient(90deg, #1C2E20, #2D5A3D) !important;
        color: #FFFCF0 !important;
        font-weight: 700 !important;
        font-size: 0.71rem !important;
        letter-spacing: 0.5px !important;
    }
    .stDataFrame tbody tr:nth-child(even) td {
        background: rgba(45,90,61,0.03) !important;
    }
    .stDataFrame tbody tr:hover td {
        background: rgba(196,98,45,0.05) !important;
    }

    /* ── MAIN BUTTONS ───────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #2D5A3D, #3D7A54);
        color: #FFFCF0;
        border: none;
        border-radius: var(--radius);
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 0.62rem 1.4rem;
        cursor: pointer;
        transition: all 0.22s;
        width: 100%;
        box-shadow: 0 3px 14px rgba(45,90,61,0.30);
        letter-spacing: 0.2px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3D7A54, #5A9E70);
        box-shadow: 0 6px 22px rgba(45,90,61,0.38);
        transform: translateY(-1px);
    }

    /* ── PRED RESULT ─────────────────────────────────────── */
    .pred-result-box {
        border-radius: var(--radius-lg);
        padding: 1.6rem 1.9rem;
        margin-top: 0.9rem;
        text-align: center;
        border: 1.5px solid;
        position: relative; overflow: hidden;
    }
    .pred-layak {
        background: linear-gradient(135deg, #EBF5EE, #D8EEE0);
        border-color: rgba(45,90,61,0.30);
    }
    .pred-tidak {
        background: linear-gradient(135deg, #F8EEE8, #F2E0D4);
        border-color: rgba(196,98,45,0.30);
    }
    .pred-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.85rem; font-weight: 700; position: relative;
    }
    .pred-sub { font-size: 0.77rem; color: var(--text-dim); margin-top: 0.45rem; position: relative; }

    /* ── DOWNLOAD LINK ──────────────────────────────────── */
    .dl-btn {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: var(--surface3);
        color: var(--green) !important;
        text-decoration: none;
        border: 1.5px solid var(--border2);
        border-radius: var(--radius);
        padding: 0.45rem 1.1rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 0.76rem;
        transition: all 0.2s;
        margin-top: 0.35rem;
    }
    .dl-btn:hover {
        border-color: var(--green-mid);
        background: var(--green-pale);
        color: var(--green) !important;
    }

    /* ── MISC ───────────────────────────────────────────── */
    hr { border-color: var(--border) !important; margin: 1.4rem 0; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--surface2); }
    ::-webkit-scrollbar-thumb { background: #C8C3B6; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--green-mid); }

    [data-testid="stAlert"] {
        border-radius: var(--radius) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    footer { display: none !important; }
    #MainMenu { visibility: hidden; }

    /* ── SELECT/NUMBER/INPUTS ──────────────────────────── */
    .stSelectbox label, .stNumberInput label,
    .stTextInput label, .stRadio label {
        color: var(--text-body) !important;
        font-weight: 600 !important;
        font-size: 0.80rem !important;
    }
    .stTextInput input {
        background: var(--surface) !important;
        border: 1.5px solid var(--border2) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
    }
    .stTextInput input:focus {
        border-color: var(--green-mid) !important;
        box-shadow: 0 0 0 3px var(--green-glow) !important;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #2D5A3D, #5A9E70) !important;
    }
    .stSlider label {
        color: var(--text-body) !important;
        font-weight: 600 !important;
        font-size: 0.80rem !important;
    }

    /* Hide sidebar collapse */
    button[kind="header"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    button[data-testid="baseButton-header"] { display: none !important; }

    /* ── WELCOME PAGE ───────────────────────────────────── */
    .welcome-bg {
        position: fixed; inset: 0; z-index: 0; pointer-events: none;
        background:
            radial-gradient(ellipse 60% 60% at 15% 20%, rgba(45,90,61,0.12) 0%, transparent 65%),
            radial-gradient(ellipse 50% 50% at 85% 75%, rgba(196,98,45,0.10) 0%, transparent 65%),
            linear-gradient(160deg, #F5F3EC 0%, #EEF0E8 55%, #F5F0E8 100%);
    }
    .welcome-wrap {
        min-height: 88vh; display: flex; align-items: center;
        justify-content: center; padding: 2rem; position: relative; z-index: 1;
    }
    .hero-card {
        background: rgba(255,255,255,0.78);
        backdrop-filter: blur(14px);
        border-radius: 28px;
        padding: 2.8rem 3.2rem;
        box-shadow: 0 20px 60px rgba(28,26,22,0.14), 0 4px 16px rgba(28,26,22,0.06), inset 0 1px 0 rgba(255,255,255,0.70);
        border: 1px solid rgba(255,255,255,0.50);
        max-width: 820px; width: 100%;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .hero-card:hover { transform: translateY(-4px); }
    .hero-icon-wrap {
        width: 90px; height: 90px; margin: 0 auto 1.6rem;
        background: linear-gradient(135deg, #2D5A3D, #C4622D);
        border-radius: 26px;
        display: flex; align-items: center; justify-content: center;
        font-size: 3rem;
        box-shadow: 0 12px 32px rgba(45,90,61,0.28);
        animation: heroFloat 3.5s ease-in-out infinite;
    }
    @keyframes heroFloat {
        0%,100% { transform: translateY(0px); }
        50%      { transform: translateY(-10px); }
    }
    .hero-eyebrow {
        font-size: 0.62rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2.5px;
        color: #5A9E70; margin-bottom: 0.5rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem; font-weight: 700; line-height: 1.15;
        color: #1C2E20; margin-bottom: 0.4rem;
    }
    .hero-title span { color: #C4622D; font-style: italic; }
    .hero-subtitle {
        font-family: 'Playfair Display', serif;
        font-size: 1.55rem; color: #3D5040;
        margin-bottom: 0.9rem; font-weight: 400;
    }
    .hero-desc {
        font-size: 0.95rem; color: #5C5748;
        max-width: 560px; margin: 0 auto 1.8rem; line-height: 1.7;
    }
    .hero-pills {
        display: flex; flex-wrap: wrap; gap: 0.65rem;
        justify-content: center; margin-bottom: 2rem;
    }
    .hero-pill {
        background: rgba(45,90,61,0.08);
        padding: 0.42rem 1.25rem; border-radius: 999px;
        font-weight: 600; color: #2D5A3D;
        border: 1px solid rgba(45,90,61,0.18);
        font-size: 0.83rem; transition: all 0.2s;
    }
    .hero-pill:hover {
        background: rgba(196,98,45,0.10);
        color: #9A4020;
        border-color: rgba(196,98,45,0.25);
    }
    .hero-footer {
        margin-top: 1.6rem; font-size: 0.72rem;
        color: #9C9488; text-align: center;
    }
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
        .main .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            max-width: 1100px !important;
        }
        </style>
        <div class="welcome-bg"></div>
        <div class="welcome-wrap">
            <div class="hero-card">
                <div class="hero-icon-wrap">🎓</div>
                <div class="hero-eyebrow">Sistem Pendukung Keputusan · Machine Learning</div>
                <div class="hero-title">Klasifikasi <span>Beasiswa</span></div>
                <div class="hero-subtitle">Siswa Miskin (BSM)</div>
                <div class="hero-desc">
                    Menggunakan algoritma <strong>Gaussian Naive Bayes</strong> untuk menentukan
                    kelayakan penerima beasiswa secara objektif dan akurat berdasarkan
                    data ekonomi keluarga siswa.
                </div>
                <div class="hero-pills">
                    <div class="hero-pill">📊 Akurasi Tinggi</div>
                    <div class="hero-pill">🤖 Machine Learning</div>
                    <div class="hero-pill">🎯 Tepat Sasaran</div>
                    <div class="hero-pill">📈 Evaluasi Otomatis</div>
                    <div class="hero-pill">⚡ Real-time</div>
                </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if st.button("🚀  Mulai Klasifikasi", use_container_width=True, key="start_btn"):
            st.session_state.page_mode = "main"
            st.rerun()

    st.markdown(
        """
                <div class="hero-footer">
                    © 2025 · BSM Classifier · Gaussian Naive Bayes
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# =========================================================
# PAGE TITLE BAR
# =========================================================

st.markdown(
    """
    <div class="page-titlebar">
        <div class="ptb-badge">🎓</div>
        <div class="ptb-text">
            <div class="ptb-eyebrow">Sistem Pendukung Keputusan · Gaussian Naive Bayes</div>
            <div class="ptb-title">Klasifikasi <span class="hl">Beasiswa Siswa Miskin</span></div>
        </div>
        <div class="ptb-pills">
            <div class="hpill hpill-green">🤖 ML</div>
            <div class="hpill hpill-terra">📊 Gaussian NB</div>
            <div class="hpill hpill-gold">🎒 BSM</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE
# =========================================================

for key in ["raw","df_clean","nb_results","data_tambahan","notif_upload","notif_prep","notif_nb","split_ratio"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state["data_tambahan"] is None:
    st.session_state["data_tambahan"] = pd.DataFrame(
        columns=["NAMA SISWA","KELAS","NIS","PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL (PREDIKSI)"]
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
# NORMALISASI
# =========================================================

def kategori_pendapatan(x):
    x = str(x).lower().replace(".", "").replace("rp", "").strip()
    angka = "".join(filter(str.isdigit, x))
    if angka == "": return np.nan
    angka = int(angka)
    if 500_000 <= angka <= 1_000_000: return 0
    if 3_000_000 <= angka <= 4_000_000: return 1
    if 5_000_000 <= angka <= 7_000_000: return 2
    return np.nan

def kategori_pekerjaan(x):
    mapping = {"petani": 0, "pns": 1, "polisi": 2, "wiraswasta": 3}
    return mapping.get(str(x).lower().strip(), np.nan)

def kategori_rumah(x):
    x = str(x).lower().strip()
    if x == "milik sendiri": return 1
    if x in ["kontrak", "sewa", "kontrak/sewa"]: return 0
    return np.nan

def kategori_label(x):
    x = str(x).lower().strip()
    if x == "ya": return 1
    if x == "tidak": return 0
    return np.nan


# =========================================================
# LOAD DATA
# =========================================================

def load_data(file) -> tuple:
    try:
        if file.name.endswith(".csv"):
            raw = pd.read_csv(file, header=None)
        elif file.name.endswith(".xlsx"):
            raw = pd.read_excel(file, header=None)
        else:
            return None, "❌ Format file tidak didukung (gunakan .csv atau .xlsx)", {}
        info = {
            "rows": len(raw),
            "cols": len(raw.columns),
            "data_rows": max(0, len(raw) - 3),
            "filename": file.name,
            "size_kb": round(file.size / 1024, 1) if hasattr(file, 'size') else "?",
        }
        return raw, "ok", info
    except Exception as e:
        return None, f"❌ Error membaca file: {e}", {}


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_data(raw: pd.DataFrame) -> tuple:
    try:
        before_count = max(0, len(raw) - 3)
        header = raw.iloc[2]
        df = raw.iloc[3:].copy()
        df.columns = header
        df.reset_index(drop=True, inplace=True)
        df.columns = df.columns.astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()

        col_map = {}
        for col in df.columns:
            if "PENDAPATAN" in col: col_map["pendapatan"] = col
            if "PEKERJAAN"  in col: col_map["pekerjaan"]  = col
            if "TANGGUNGAN" in col: col_map["tanggungan"]  = col
            if "RUMAH"      in col: col_map["rumah"]       = col
            if "LABEL"      in col: col_map["label"]       = col

        before_na = len(df)
        df[col_map["pendapatan"]] = df[col_map["pendapatan"]].apply(kategori_pendapatan)
        df[col_map["pekerjaan"]]  = df[col_map["pekerjaan"]].apply(kategori_pekerjaan)
        df[col_map["tanggungan"]] = pd.to_numeric(df[col_map["tanggungan"]], errors="coerce")
        df[col_map["rumah"]]      = df[col_map["rumah"]].apply(kategori_rumah)
        df[col_map["label"]]      = df[col_map["label"]].apply(kategori_label)

        df.rename(columns={
            col_map["pendapatan"]: "PENDAPATAN ORANG TUA",
            col_map["pekerjaan"]:  "PEKERJAAN ORANG TUA",
            col_map["tanggungan"]: "JUMLAH TANGGUNGAN",
            col_map["rumah"]:      "STATUS RUMAH",
            col_map["label"]:      "LABEL",
        }, inplace=True)

        df.dropna(subset=["PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL"], inplace=True)
        after_count  = len(df)
        dropped      = before_na - after_count
        layak_count  = int((df["LABEL"] == 1).sum())
        tidak_count  = int((df["LABEL"] == 0).sum())

        info = {
            "before": before_count,
            "after": after_count,
            "dropped": dropped,
            "layak": layak_count,
            "tidak": tidak_count,
        }
        return df, "ok", info
    except Exception as e:
        return None, f"❌ Error preprocessing: {e}", {}


# =========================================================
# GAUSSIAN PROBABILITY & NAIVE BAYES
# =========================================================

def gaussian_probability(x, mean, std):
    if std == 0 or pd.isna(std): std = 0.0001
    exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


def naive_bayes_process(df: pd.DataFrame, train_pct: int = 80) -> dict:
    fitur  = ["PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH"]
    target = "LABEL"
    X, y = df[fitur], df[target]

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
    prior, mean, std = {}, {}, {}
    for c in classes:
        prior[c] = len(y_train[y_train == c]) / len(y_train)
        data_c   = X_train[y_train == c]
        mean[c]  = data_c.mean()
        std[c]   = data_c.std()

    tabel_prior = pd.DataFrame({"Kelas": list(prior.keys()), "Prior Probability": list(prior.values())})

    hasil_manual = []
    for i in range(len(X_test)):
        row = X_test.iloc[i]
        probs = {}
        for c in classes:
            probs[c] = prior[c]
            for f in fitur:
                probs[c] *= gaussian_probability(row[f], mean[c][f], std[c][f])
        prediksi = max(probs, key=probs.get)
        hasil_manual.append({
            "Data Ke": i + 1,
            "P(Tidak Layak=0)": probs.get(0, 0),
            "P(Layak=1)":       probs.get(1, 0),
            "Prediksi":         prediksi,
            "Aktual":           y_test.iloc[i],
            "Benar?":           "✅" if prediksi == y_test.iloc[i] else "❌",
        })

    hasil_manual_df = pd.DataFrame(hasil_manual)
    cm       = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).transpose()

    # ── Chart colours matching new theme ─────────────────
    BG     = "#F7F6F1"
    DARK   = "#1C2E20"
    GREEN  = "#2D5A3D"
    GREEN2 = "#5A9E70"
    TERRA  = "#C4622D"
    TERRA2 = "#E07848"
    GOLD   = "#B89A3E"
    MUTED  = "#8C8578"
    BORDER = "#DDD9CF"
    TEXT   = "#1C1A16"

    # Confusion Matrix chart
    fig_cm, ax = plt.subplots(figsize=(6.8, 5.2))
    fig_cm.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor(BG)

    n_rows, n_cols = cm.shape
    cell_colors = [[GREEN, TERRA], [TERRA, GREEN]]
    cell_labels  = [["TN", "FP"], ["FN", "TP"]]

    for i in range(n_rows):
        for j in range(n_cols):
            val      = cm[i, j]
            bg_color = cell_colors[i][j]
            rect = plt.Rectangle((j+0.07, i+0.07), 0.86, 0.86, color=bg_color, alpha=0.10, zorder=0)
            ax.add_patch(rect)
            border_rect = plt.Rectangle((j+0.07, i+0.07), 0.86, 0.86, fill=False,
                                         edgecolor=bg_color, alpha=0.40, linewidth=2, zorder=1)
            ax.add_patch(border_rect)
            ax.text(j+0.5, i+0.40, str(val), ha="center", va="center",
                    fontsize=48, fontweight="bold", color=bg_color, zorder=2, fontfamily="monospace")
            ax.text(j+0.5, i+0.72, cell_labels[i][j], ha="center", va="center",
                    fontsize=11, fontweight="700", color=bg_color, alpha=0.65, zorder=2)

    for k in range(1, n_rows): ax.axhline(k, color=BORDER, linewidth=1.5, zorder=3)
    for k in range(1, n_cols): ax.axvline(k, color=BORDER, linewidth=1.5, zorder=3)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER); spine.set_linewidth(1.2)

    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows); ax.invert_yaxis()
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Tidak Layak (0)", "Layak (1)"], color=TEXT, fontsize=10.5, fontweight="600")
    ax.set_yticklabels(["Tidak Layak (0)", "Layak (1)"], color=TEXT, fontsize=10.5, fontweight="600", rotation=0, va="center")
    ax.set_xlabel("Prediksi", color=MUTED, labelpad=12, fontsize=11)
    ax.set_ylabel("Aktual",   color=MUTED, labelpad=12, fontsize=11)
    split_label = f"Training {train_pct}% / Testing {100-train_pct}%"
    ax.set_title(f"Confusion Matrix — Evaluasi Model\n({split_label})", color=DARK,
                 fontsize=12.5, fontweight="bold", pad=20, fontfamily="DejaVu Serif")
    ax.tick_params(colors=BORDER, length=0)
    legend_items = [
        mpatches.Patch(color=GREEN, label="Prediksi Benar  (TN / TP)"),
        mpatches.Patch(color=TERRA, label="Prediksi Salah  (FP / FN)"),
    ]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=2, facecolor="#FFFFFF", edgecolor=BORDER, labelcolor=TEXT, fontsize=10)
    plt.tight_layout()

    benar = int((hasil_manual_df["Benar?"] == "✅").sum())
    salah = int((hasil_manual_df["Benar?"] == "❌").sum())

    return {
        "prior":      tabel_prior,
        "manual":     hasil_manual_df,
        "report":     report_df,
        "accuracy":   accuracy,
        "fig_cm":     fig_cm,
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "train_pct":  train_pct,
        "test_pct":   100 - train_pct,
        "split_mode": split_mode,
        "classes":    classes.tolist(),
        "model":      model,
        "cm":         cm,
        "mean":       mean,
        "std":        std,
        "prior_dict": prior,
        "benar":      benar,
        "salah":      salah,
    }


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sb-logo">
            <div class="sb-logo-row">
                <div class="sb-logo-icon">🎓</div>
                <div>
                    <div class="sb-logo-title">BSM <span>Classifier</span></div>
                    <div class="sb-logo-desc">Gaussian Naive Bayes</div>
                </div>
            </div>
            <div class="sb-logo-desc">Sistem Klasifikasi Kelayakan Beasiswa Siswa Miskin</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<span class="sb-sec">📂 Unggah Dataset</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Pilih file CSV atau Excel", type=["csv","xlsx"],
                                 label_visibility="collapsed", key="uploader_main")

    if uploaded:
        if st.button("⬆  Upload Dataset", key="btn_upload"):
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
        if st.button("🔧  Jalankan Preprocessing", key="btn_preprocess"):
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
        if done:
            cls = "done"; num_cls = "done"; symbol = "✓"
        elif active:
            cls = "active"; num_cls = "active"; symbol = num
        else:
            cls = ""; num_cls = ""; symbol = num
        pipe_html += (
            f'<div class="pipeline-step {cls}">'
            f'<span class="step-num {num_cls}">{symbol}</span>'
            f'{label_s}</div>'
        )
    pipe_html += '</div>'
    st.markdown(pipe_html, unsafe_allow_html=True)

    # Mini stats
    if nb_ok:
        nb_side = st.session_state["nb_results"]
        st.markdown('<span class="sb-sec">📊 Ringkasan Model</span>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:rgba(45,90,61,0.18);border:1px solid rgba(90,158,112,0.28);
                        border-radius:12px;padding:1rem 1.1rem;margin-top:0.2rem;">
                <div style="font-size:0.63rem;color:rgba(255,252,240,0.38);text-transform:uppercase;
                            letter-spacing:1.6px;font-weight:700;margin-bottom:0.6rem;">Performa</div>
                <div style="font-family:'Playfair Display',serif;font-size:2.5rem;color:#9FD4AE;
                            line-height:1;text-align:center;margin-bottom:0.3rem;">
                    {nb_side['accuracy']*100:.1f}%</div>
                <div style="text-align:center;font-size:0.66rem;color:rgba(255,252,240,0.36);margin-bottom:0.7rem;">
                    Akurasi Model</div>
                <div style="text-align:center;font-size:0.61rem;color:rgba(255,252,240,0.28);margin-bottom:0.7rem;">
                    Split: {nb_side['train_pct']}% Train / {nb_side['test_pct']}% Test</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.45rem;">
                    <div style="background:rgba(45,90,61,0.18);border-radius:8px;padding:0.48rem;text-align:center;">
                        <div style="font-size:1.05rem;font-weight:700;color:#9FD4AE;">{nb_side['train_size']}</div>
                        <div style="font-size:0.56rem;color:rgba(255,252,240,0.32);text-transform:uppercase;letter-spacing:1px;">Train</div>
                    </div>
                    <div style="background:rgba(196,98,45,0.16);border-radius:8px;padding:0.48rem;text-align:center;">
                        <div style="font-size:1.05rem;font-weight:700;color:#F0A070;">{nb_side['test_size']}</div>
                        <div style="font-size:0.56rem;color:rgba(255,252,240,0.32);text-transform:uppercase;letter-spacing:1px;">Test</div>
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

total_raw   = max(0, len(raw) - 3) if raw is not None else "–"
total_clean = len(df_clean) if df_clean is not None else "–"
akurasi_val = f"{nb['accuracy']*100:.1f}%" if nb else "–"
train_size  = nb["train_size"] if nb else "–"
test_size   = nb["test_size"]  if nb else "–"

# ── GLOBAL NOTIFICATIONS ─────────────────────────────────
notif_upload = st.session_state.get("notif_upload")
notif_prep   = st.session_state.get("notif_prep")
notif_nb_val = st.session_state.get("notif_nb")

if notif_upload:
    st.markdown(
        notif_html(
            "upload", "📂",
            "✅ Dataset Berhasil Diunggah!",
            f"File <strong>{notif_upload['filename']}</strong> berhasil dibaca dan siap diproses.",
            [
                f"📋 Total Baris: {notif_upload['rows']}",
                f"📊 Kolom: {notif_upload['cols']}",
                f"🗃 Data Siswa: {notif_upload['data_rows']} baris",
                f"💾 Ukuran: {notif_upload['size_kb']} KB",
            ]
        ),
        unsafe_allow_html=True,
    )

if notif_prep:
    st.markdown(
        notif_html(
            "prep", "🔧",
            "✅ Preprocessing Selesai!",
            f"Data berhasil dinormalisasi. <strong>{notif_prep['dropped']}</strong> baris dihapus karena nilai tidak valid.",
            [
                f"📥 Input: {notif_prep['before']} baris",
                f"✅ Output: {notif_prep['after']} baris bersih",
                f"🎓 Layak BSM: {notif_prep['layak']} siswa",
                f"📋 Tidak Layak: {notif_prep['tidak']} siswa",
                f"🗑 Dihapus: {notif_prep['dropped']} baris",
            ]
        ),
        unsafe_allow_html=True,
    )

if notif_nb_val:
    st.markdown(
        notif_html(
            "nb", "🤖",
            "✅ Model Naive Bayes Selesai Dilatih!",
            f"Split <strong>{notif_nb_val['train_pct']}%/{notif_nb_val['test_pct']}%</strong>. Akurasi: <strong>{notif_nb_val['acc']*100:.2f}%</strong>.",
            [
                f"🏋 Training: {notif_nb_val['train']} ({notif_nb_val['train_pct']}%)",
                f"🧪 Testing: {notif_nb_val['test']} ({notif_nb_val['test_pct']}%)",
                f"✅ Benar: {notif_nb_val['benar']}",
                f"❌ Salah: {notif_nb_val['salah']}",
                f"🏅 Akurasi: {notif_nb_val['acc']*100:.2f}%",
            ]
        ),
        unsafe_allow_html=True,
    )

# Split badge
if nb:
    split_badge_train = f"{nb['train_pct']}%"
    split_badge_test  = f"{nb['test_pct']}%"
else:
    split_ratio_now   = st.session_state.get("split_ratio", 80)
    split_badge_train = f"{split_ratio_now}%"
    split_badge_test  = f"{100 - split_ratio_now}%"

st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-card mc-green">
            <span class="mc-icon">📋</span>
            <div class="mc-label">Total Data Awal</div>
            <div class="mc-value cv-green">{total_raw}</div>
        </div>
        <div class="metric-card mc-terra">
            <span class="mc-icon">✅</span>
            <div class="mc-label">Setelah Preprocessing</div>
            <div class="mc-value cv-terra">{total_clean}</div>
        </div>
        <div class="metric-card mc-slate">
            <span class="mc-icon">🏋</span>
            <div class="mc-label">Data Training ({split_badge_train})</div>
            <div class="mc-value cv-slate">{train_size}</div>
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
    "🔧 Normalisasi",
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
        data_rows = max(0, len(raw) - 3)
        st.markdown(
            notif_html(
                "upload", "💡",
                f"Dataset dimuat — {data_rows} baris data siswa",
                "Header tabel dimulai dari baris ke-3 (index 2) pada file sumber. Data di bawah adalah tampilan mentah sebelum preprocessing.",
                [f"📋 Total baris file: {len(raw)}", f"🗃 Estimasi data: {data_rows}", f"📊 Kolom: {len(raw.columns)}"]
            ),
            unsafe_allow_html=True,
        )
        st.dataframe(raw, use_container_width=True, height=420)
        st.markdown("---")
        col_judul1, col_dl1 = st.columns([2, 1])
        with col_judul1:
            judul_raw = st.text_input("Judul untuk file unduhan", value="Dataset Awal BSM", key="judul_raw")
        with col_dl1:
            st.markdown("<br>", unsafe_allow_html=True)
            filename_raw = f"{judul_raw.replace(' ', '_')}.xlsx"
            st.markdown(make_excel_download(raw, filename_raw, f"⬇ Unduh {judul_raw} (.xlsx)", judul_raw), unsafe_allow_html=True)
    else:
        st.info("👈 Unggah dataset terlebih dahulu melalui panel kiri untuk memulai.")

# ─────────────────────────────────────────────────────────
# TAB 2 — Normalisasi + Split Config + Tombol NB
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🔧 Hasil Normalisasi & Encoding</div>', unsafe_allow_html=True)
    if df_clean is not None:
        if notif_prep:
            st.markdown(
                notif_html(
                    "prep", "🔧",
                    "Hasil Preprocessing Dataset",
                    f"Berhasil memproses <strong>{notif_prep['after']}</strong> data bersih dari <strong>{notif_prep['before']}</strong> data awal.",
                    [
                        f"✅ Bersih: {notif_prep['after']}",
                        f"🗑 Dihapus: {notif_prep['dropped']}",
                        f"🎓 Layak: {notif_prep['layak']}",
                        f"📋 Tidak Layak: {notif_prep['tidak']}",
                    ]
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            """<div class="info-box">
            💡 Setiap atribut diubah ke bentuk numerik agar dapat diproses model.<br>
            <strong>Pendapatan</strong> → 0 (Rendah) / 1 (Menengah) / 2 (Tinggi) &nbsp;·&nbsp;
            <strong>Pekerjaan</strong> → Petani=0, PNS=1, Polisi=2, Wiraswasta=3 &nbsp;·&nbsp;
            <strong>Rumah</strong> → Kontrak=0, Milik=1 &nbsp;·&nbsp;
            <strong>Label</strong> → Tidak=0, Ya=1
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="formula-box">
                <div class="formula-title">📐 Catatan Encoding & Normalisasi</div>
                <strong>1. Pendapatan Orang Tua</strong><br>
                &nbsp;&nbsp;&nbsp;• Rp500.000–Rp1.000.000 → <code>0 (Rendah)</code> &nbsp;·&nbsp;
                Rp3.000.000–Rp4.000.000 → <code>1 (Menengah)</code> &nbsp;·&nbsp;
                Rp5.000.000–Rp7.000.000 → <code>2 (Tinggi)</code><br>
                <strong>2. Pekerjaan Orang Tua</strong> — Petani=<code>0</code>, PNS=<code>1</code>, Polisi=<code>2</code>, Wiraswasta=<code>3</code><br>
                <strong>3. Status Rumah</strong> — Kontrak/Sewa=<code>0</code>, Milik Sendiri=<code>1</code><br>
                <strong>4. Label Target</strong> — Tidak Layak=<code>0</code>, Layak=<code>1</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(df_clean, use_container_width=True, height=300)
        st.markdown("---")

        # ── Split Config ─────────────────────────────────
        st.markdown('<div class="section-header">⚖️ Konfigurasi Train/Test Split</div>', unsafe_allow_html=True)

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
            st.markdown(
                '<p style="font-size:.70rem;color:#8C8578;text-transform:uppercase;'
                'letter-spacing:1.2px;font-weight:700;margin-bottom:.55rem;">Pilih Preset</p>',
                unsafe_allow_html=True,
            )
            preset_labels = list(PRESET_OPTIONS.keys())
            default_preset_idx = 1
            for idx, (label, val) in enumerate(PRESET_OPTIONS.items()):
                if val == current_ratio:
                    default_preset_idx = idx
                    break
            selected_preset = st.radio("Pilih preset rasio", options=preset_labels,
                                        index=default_preset_idx, key="preset_split",
                                        label_visibility="collapsed")
            preset_val = PRESET_OPTIONS[selected_preset]
            st.session_state["split_ratio"] = preset_val

        with col_slider:
            st.markdown(
                '<p style="font-size:.70rem;color:#8C8578;text-transform:uppercase;'
                'letter-spacing:1.2px;font-weight:700;margin-bottom:.55rem;">Atau Atur Manual (% Training)</p>',
                unsafe_allow_html=True,
            )
            slider_val = st.slider("Training %", min_value=50, max_value=100,
                                    value=st.session_state["split_ratio"], step=5,
                                    format="%d%%", key="slider_split", label_visibility="collapsed")
            if slider_val != st.session_state["split_ratio"]:
                st.session_state["split_ratio"] = slider_val

        final_ratio = st.session_state["split_ratio"]
        train_count = round(total_data * final_ratio / 100)
        test_count  = total_data - train_count

        st.markdown(
            f"""
            <div class="split-config-box">
                <div class="split-config-title">⚖️ Rasio yang Dipilih</div>
                <div class="split-config-desc">
                    Konfigurasi ini digunakan saat menekan tombol Proses Naive Bayes di bawah.
                </div>
                <div class="split-bar-wrap">
                    <div class="split-bar-train"
                         style="width:{final_ratio}%;background:linear-gradient(90deg,#2D5A3D,#5A9E70);"></div>
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
                notif_html(
                    "warn", "⚠️",
                    "Mode 100% Training",
                    "Seluruh data digunakan untuk training. Evaluasi dilakukan secara in-sample, sehingga akurasi cenderung lebih tinggi namun tidak mencerminkan performa pada data baru.",
                    ["📌 In-sample evaluation", "⚠️ Risiko overfitting", "💡 Gunakan untuk eksplorasi"]
                ),
                unsafe_allow_html=True,
            )

        st.markdown("---")
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("🤖  Proses & Klasifikasi Naive Bayes", use_container_width=True):
                with st.spinner("Melatih model dan melakukan klasifikasi..."):
                    nb_res = naive_bayes_process(st.session_state["df_clean"], train_pct=final_ratio)
                    st.session_state["nb_results"] = nb_res
                    st.session_state["notif_nb"] = {
                        "train": nb_res["train_size"], "test": nb_res["test_size"],
                        "acc": nb_res["accuracy"], "benar": nb_res["benar"],
                        "salah": nb_res["salah"], "train_pct": nb_res["train_pct"],
                        "test_pct": nb_res["test_pct"],
                    }
                    st.success(f"✅ Klasifikasi selesai! Akurasi: {nb_res['accuracy']*100:.2f}%")
        with col_btn2:
            if st.button("🗑  Reset Semua Data & Model", use_container_width=True):
                st.session_state["nb_results"]   = None
                st.session_state["notif_nb"]     = None
                st.session_state["df_clean"]     = None
                st.session_state["raw"]          = None
                st.session_state["notif_upload"] = None
                st.session_state["notif_prep"]   = None
                st.session_state["data_tambahan"] = pd.DataFrame(
                    columns=["NAMA SISWA","KELAS","NIS","PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL (PREDIKSI)"]
                )
                st.rerun()

        st.markdown("---")
        layak_n = int(df_clean["LABEL"].sum())
        tidak_n = len(df_clean) - layak_n

        st.markdown(
            f"""
            <div style="display:flex;gap:0.85rem;margin-bottom:1rem;flex-wrap:wrap;">
                <div style="flex:1;min-width:150px;background:linear-gradient(135deg,#EBF5EE,#D8EEE0);
                            border:1px solid rgba(45,90,61,0.22);border-radius:12px;
                            padding:1rem 1.2rem;text-align:center;">
                    <div style="font-size:.60rem;color:#2D5A3D;text-transform:uppercase;
                                letter-spacing:1.4px;font-weight:800;margin-bottom:.35rem;">🎓 Layak BSM</div>
                    <div style="font-family:'Playfair Display',serif;font-size:2.3rem;
                                color:#2D5A3D;line-height:1;font-weight:700;">{layak_n:,}</div>
                    <div style="font-size:.73rem;color:#4A7060;margin-top:.28rem;font-weight:600;">
                        {layak_n/len(df_clean)*100:.1f}% dari total</div>
                </div>
                <div style="flex:1;min-width:150px;background:linear-gradient(135deg,#FBF2EC,#F5E4D4);
                            border:1px solid rgba(196,98,45,0.20);border-radius:12px;
                            padding:1rem 1.2rem;text-align:center;">
                    <div style="font-size:.60rem;color:#9A4020;text-transform:uppercase;
                                letter-spacing:1.4px;font-weight:800;margin-bottom:.35rem;">📋 Tidak Layak</div>
                    <div style="font-family:'Playfair Display',serif;font-size:2.3rem;
                                color:#C4622D;line-height:1;font-weight:700;">{tidak_n:,}</div>
                    <div style="font-size:.73rem;color:#8A5040;margin-top:.28rem;font-weight:600;">
                        {tidak_n/len(df_clean)*100:.1f}% dari total</div>
                </div>
                <div style="flex:1;min-width:150px;background:linear-gradient(135deg,#EDF0F8,#E0E8F2);
                            border:1px solid rgba(58,84,112,0.18);border-radius:12px;
                            padding:1rem 1.2rem;text-align:center;">
                    <div style="font-size:.60rem;color:#3A5470;text-transform:uppercase;
                                letter-spacing:1.4px;font-weight:800;margin-bottom:.35rem;">👥 Total Siswa</div>
                    <div style="font-family:'Playfair Display',serif;font-size:2.3rem;
                                color:#3A5470;line-height:1;font-weight:700;">{len(df_clean):,}</div>
                    <div style="font-size:.73rem;color:#3A5070;margin-top:.28rem;font-weight:600;">
                        Data bersih siap proses</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_dl_wrap, _ = st.columns([1, 2])
        with col_dl_wrap:
            judul_norm = st.text_input("Judul file", value="Hasil Normalisasi BSM", key="judul_norm")
            filename_norm = f"{judul_norm.replace(' ', '_')}.xlsx"
            st.markdown(make_excel_download(df_clean, filename_norm, f"⬇ Unduh {judul_norm} (.xlsx)", judul_norm), unsafe_allow_html=True)
    else:
        st.info("👈 Klik **Jalankan Preprocessing** di panel kiri setelah mengunggah data.")

# ─────────────────────────────────────────────────────────
# TAB 3 — Prior & Manual
# ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📊 Prior Probability & Perhitungan Manual</div>', unsafe_allow_html=True)
    if nb is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<p style="font-size:.66rem;color:#8C8578;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.4rem;">Tabel Prior Probability</p>', unsafe_allow_html=True)
            st.dataframe(nb["prior"], use_container_width=True)

            # Prior chart — warm theme
            BG2 = "#F7F6F1"
            fig_prior, ax_pr = plt.subplots(figsize=(3.5, 2.8))
            fig_prior.patch.set_facecolor("#FFFFFF")
            ax_pr.set_facecolor(BG2)
            bar_col = ["#C4622D", "#2D5A3D"]
            bars = ax_pr.bar(
                ["Tidak Layak (0)", "Layak (1)"],
                nb["prior"]["Prior Probability"],
                color=bar_col, width=0.48, edgecolor="#ffffff", linewidth=1.5,
            )
            ax_pr.set_ylim(0, 1)
            ax_pr.set_ylabel("Probabilitas", color="#8C8578", fontsize=9)
            ax_pr.tick_params(colors="#8C8578", labelsize=8)
            for spine in ax_pr.spines.values(): spine.set_edgecolor("#DDD9CF")
            for bar in bars:
                ax_pr.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.015, f"{bar.get_height():.3f}",
                           ha="center", va="bottom", color="#1C1A16", fontsize=9, fontweight="bold")
            ax_pr.axhline(0.5, color="#B89A3E", linewidth=1, linestyle="--", alpha=.55)
            ax_pr.set_title("Prior per Kelas", fontsize=10, color="#1C2E20", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_prior)

        with c2:
            st.markdown('<p style="font-size:.66rem;color:#8C8578;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.4rem;">Perhitungan Manual Naive Bayes (per data uji)</p>', unsafe_allow_html=True)
            st.dataframe(nb["manual"], use_container_width=True, height=380)

        st.markdown(
            """
            <div class="formula-box">
                <div class="formula-title">📐 Rumus Perhitungan Naive Bayes</div>
                <strong>1. Prior Probability:</strong><br>
                &nbsp;&nbsp;&nbsp;<code>P(Kelas) = Jumlah data kelas / Total data training</code><br><br>
                <strong>2. Gaussian Likelihood:</strong><br>
                &nbsp;&nbsp;&nbsp;<code>P(x|Kelas) = (1 / √(2π × σ²)) × exp(-(x-μ)² / (2σ²))</code><br><br>
                <strong>3. Posterior:</strong><br>
                &nbsp;&nbsp;&nbsp;<code>P(Kelas|X) = P(Kelas) × P(x₁|Kelas) × ... × P(xₙ|Kelas)</code><br><br>
                <strong>4. Keputusan:</strong>
                &nbsp;&nbsp;&nbsp;<code>Prediksi = argmax P(Kelas|X)</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        col_j3, col_d3 = st.columns([2, 1])
        with col_j3:
            judul_manual = st.text_input("Judul untuk file unduhan", value="Perhitungan Manual Naive Bayes", key="judul_manual")
        with col_d3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(make_excel_download(nb["manual"], f"{judul_manual.replace(' ','_')}.xlsx", f"⬇ Unduh {judul_manual} (.xlsx)", judul_manual), unsafe_allow_html=True)
    else:
        st.info("👈 Jalankan preprocessing dan klik **Proses & Klasifikasi Naive Bayes** di Tab Normalisasi.")

# ─────────────────────────────────────────────────────────
# TAB 4 — Hasil Prediksi
# ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🎯 Hasil Prediksi vs Data Aktual</div>', unsafe_allow_html=True)
    if nb is not None:
        prediksi_df = nb["manual"][["Data Ke","Prediksi","Aktual","Benar?"]]
        benar = nb["benar"]
        salah = nb["salah"]

        st.markdown(
            notif_html(
                "nb", "🎯",
                f"Prediksi Selesai — {benar} dari {benar+salah} data uji benar",
                f"Split: <strong>{nb['train_pct']}% Training / {nb['test_pct']}% Testing</strong> · Akurasi: <strong>{(benar/(benar+salah)*100) if (benar+salah)>0 else 0:.2f}%</strong>",
                [
                    f"✅ Benar: {benar}", f"❌ Salah: {salah}",
                    f"📊 Total: {benar+salah}",
                    f"🏅 Akurasi: {(benar/(benar+salah)*100) if (benar+salah)>0 else 0:.2f}%",
                ]
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="metric-row">
            <div class="metric-card mc-green"><span class="mc-icon">✅</span><div class="mc-label">Prediksi Benar</div><div class="mc-value cv-green">{benar}</div></div>
            <div class="metric-card mc-terra"><span class="mc-icon">❌</span><div class="mc-label">Prediksi Salah</div><div class="mc-value cv-terra">{salah}</div></div>
            <div class="metric-card mc-slate"><span class="mc-icon">🧑‍🎓</span><div class="mc-label">Total Data Uji</div><div class="mc-value cv-slate">{benar+salah}</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.dataframe(prediksi_df, use_container_width=True, height=400)

        st.markdown(
            f"""
            <div class="formula-box">
                <div class="formula-title">📐 Rumus Akurasi</div>
                <code>Akurasi = (Prediksi Benar / Total Data Uji) × 100%</code><br>
                &nbsp;&nbsp;&nbsp;= ({benar} / {benar+salah}) × 100% = <strong>{(benar/(benar+salah)*100) if (benar+salah)>0 else 0:.2f}%</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        col_j4, col_d4 = st.columns([2, 1])
        with col_j4:
            judul_pred = st.text_input("Judul untuk file unduhan", value="Hasil Prediksi Naive Bayes", key="judul_pred")
        with col_d4:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(make_excel_download(prediksi_df, f"{judul_pred.replace(' ','_')}.xlsx", f"⬇ Unduh {judul_pred} (.xlsx)", judul_pred), unsafe_allow_html=True)
    else:
        st.info("👈 Jalankan preprocessing dan klik **Proses & Klasifikasi Naive Bayes** di Tab Normalisasi.")

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
            notif_html(
                "nb", "🔲",
                f"Confusion Matrix — Split {nb['train_pct']}% / {nb['test_pct']}%",
                f"TP: <strong>{tp}</strong> · TN: <strong>{tn}</strong> · FP: <strong>{fp}</strong> · FN: <strong>{fn}</strong>",
                [f"✅ TN: {tn}", f"✅ TP: {tp}", f"⚠️ FP: {fp}", f"⚠️ FN: {fn}", f"🏅 Akurasi: {nb['accuracy']*100:.2f}%"]
            ),
            unsafe_allow_html=True,
        )

        col_img, col_exp = st.columns([1, 1])
        with col_img:
            st.pyplot(nb["fig_cm"])
            col_j5, col_d5 = st.columns([2, 1])
            with col_j5:
                judul_cm = st.text_input("Judul file", value="Confusion Matrix BSM", key="judul_cm")
            with col_d5:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(make_png_download(nb["fig_cm"], f"{judul_cm.replace(' ','_')}.png", f"⬇ Unduh {judul_cm} (.png)"), unsafe_allow_html=True)

        with col_exp:
            st.markdown(
                """
                <div class="info-box">
                <strong>Cara Membaca Confusion Matrix</strong><br><br>
                <strong style="color:#2D5A3D;">True Negative (TN)</strong> — Prediksi <em>Tidak Layak</em>, Aktual <em>Tidak Layak</em> ✅<br><br>
                <strong style="color:#C4622D;">False Positive (FP)</strong> — Prediksi <em>Layak</em>, Aktual <em>Tidak Layak</em> ❌<br><br>
                <strong style="color:#C4622D;">False Negative (FN)</strong> — Prediksi <em>Tidak Layak</em>, Aktual <em>Layak</em> ❌<br><br>
                <strong style="color:#2D5A3D;">True Positive (TP)</strong> — Prediksi <em>Layak</em>, Aktual <em>Layak</em> ✅
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin-bottom:.85rem">
                    <div style="background:#fff;border:1px solid #DDD9CF;border-top:3px solid #2D5A3D;border-radius:11px;padding:.85rem;text-align:center;box-shadow:0 2px 6px rgba(45,90,61,0.08)">
                        <div style="font-size:.58rem;color:#8C8578;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.28rem;font-weight:700;">True Negative</div>
                        <div style="font-family:'Playfair Display',serif;font-size:2.3rem;color:#2D5A3D;line-height:1">{tn}</div>
                    </div>
                    <div style="background:#fff;border:1px solid #DDD9CF;border-top:3px solid #C4622D;border-radius:11px;padding:.85rem;text-align:center;box-shadow:0 2px 6px rgba(196,98,45,0.08)">
                        <div style="font-size:.58rem;color:#8C8578;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.28rem;font-weight:700;">False Positive</div>
                        <div style="font-family:'Playfair Display',serif;font-size:2.3rem;color:#C4622D;line-height:1">{fp}</div>
                    </div>
                    <div style="background:#fff;border:1px solid #DDD9CF;border-top:3px solid #C4622D;border-radius:11px;padding:.85rem;text-align:center;box-shadow:0 2px 6px rgba(196,98,45,0.08)">
                        <div style="font-size:.58rem;color:#8C8578;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.28rem;font-weight:700;">False Negative</div>
                        <div style="font-family:'Playfair Display',serif;font-size:2.3rem;color:#C4622D;line-height:1">{fn}</div>
                    </div>
                    <div style="background:#fff;border:1px solid #DDD9CF;border-top:3px solid #2D5A3D;border-radius:11px;padding:.85rem;text-align:center;box-shadow:0 2px 6px rgba(45,90,61,0.08)">
                        <div style="font-size:.58rem;color:#8C8578;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.28rem;font-weight:700;">True Positive</div>
                        <div style="font-family:'Playfair Display',serif;font-size:2.3rem;color:#2D5A3D;line-height:1">{tp}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            acc = nb["accuracy"]
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#1C2E20,#24382A);
                            border-radius:14px;padding:1.3rem 1.5rem;text-align:center;
                            border-bottom:3px solid #3D7A54;box-shadow:0 8px 28px rgba(20,30,22,0.22);">
                    <div style="font-size:.60rem;color:rgba(255,252,240,0.35);text-transform:uppercase;
                                letter-spacing:2px;margin-bottom:.38rem;font-weight:700;">Akurasi Model</div>
                    <div style="font-family:'Playfair Display',serif;font-size:3.6rem;
                                color:#9FD4AE;line-height:1">{acc*100:.2f}%</div>
                    <div style="font-size:.76rem;color:rgba(255,252,240,0.34);margin-top:.45rem;">
                        {tn+tp} prediksi benar dari {tn+tp+fn+fp} data uji
                    </div>
                    <div style="font-size:.66rem;color:rgba(159,212,174,0.45);margin-top:.28rem;">
                        Split {nb['train_pct']}% Train / {nb['test_pct']}% Test
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="formula-box" style="margin-top:.85rem;">
                    <div class="formula-title">📐 Rumus dari Confusion Matrix</div>
                    <code>Akurasi  = (TP + TN) / (TP + TN + FP + FN)</code><br>
                    <code>Presisi  = TP / (TP + FP)</code><br>
                    <code>Recall   = TP / (TP + FN)</code><br>
                    <code>F1-Score = 2 × (Presisi × Recall) / (Presisi + Recall)</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("👈 Jalankan preprocessing dan klik **Proses & Klasifikasi Naive Bayes** di Tab Normalisasi.")

# ─────────────────────────────────────────────────────────
# TAB 6 — Classification Report
# ─────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">📈 Laporan Klasifikasi Lengkap</div>', unsafe_allow_html=True)
    if nb is not None:
        report         = nb["report"].copy()
        report_display = report.round(4)

        st.dataframe(report_display, use_container_width=True)

        # Bar chart — warm theme
        metrics_df = report.loc[["0","1"], ["precision","recall","f1-score"]]
        BG2 = "#F7F6F1"
        fig_bar, ax_b = plt.subplots(figsize=(6, 3.5))
        fig_bar.patch.set_facecolor("#FFFFFF")
        ax_b.set_facecolor(BG2)

        x = np.arange(len(metrics_df))
        w = 0.24
        bar_palette = ["#2D5A3D", "#C4622D", "#B89A3E"]
        for idx, (col, color) in enumerate(zip(metrics_df.columns, bar_palette)):
            ax_b.bar(x + idx*w, metrics_df[col], w, label=col.capitalize(),
                     color=color, edgecolor="#ffffff", linewidth=1.3)

        ax_b.set_xticks(x + w)
        ax_b.set_xticklabels(["Kelas 0 — Tidak Layak", "Kelas 1 — Layak"], color="#1C1A16", fontsize=10)
        ax_b.set_ylim(0, 1.2)
        ax_b.tick_params(colors="#8C8578", labelsize=9)
        ax_b.legend(facecolor="#FFFFFF", edgecolor="#DDD9CF", labelcolor="#1C1A16", fontsize=9)
        for spine in ax_b.spines.values(): spine.set_edgecolor("#DDD9CF")
        ax_b.axhline(1.0, color="#DDD9CF", linewidth=0.8, linestyle="--")
        ax_b.set_title(f"Perbandingan Metrik Evaluasi per Kelas (Split {nb['train_pct']}%/{nb['test_pct']}%)",
                       color="#1C2E20", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_bar)

        try:
            prec0 = float(report.loc["0", "precision"]); rec0 = float(report.loc["0", "recall"]); f1_0 = float(report.loc["0", "f1-score"])
            prec1 = float(report.loc["1", "precision"]); rec1 = float(report.loc["1", "recall"]); f1_1 = float(report.loc["1", "f1-score"])

            def interpret(val):
                if val >= 0.90: return ("Sangat Baik", "#2D5A3D")
                elif val >= 0.75: return ("Baik", "#3A5470")
                elif val >= 0.60: return ("Cukup", "#B89A3E")
                else: return ("Perlu Ditingkatkan", "#C4622D")

            ip0,cp0 = interpret(prec0); ir0,cr0 = interpret(rec0); if0_,cf0 = interpret(f1_0)
            ip1,cp1 = interpret(prec1); ir1,cr1 = interpret(rec1); if1_,cf1 = interpret(f1_1)

            st.markdown(
                f"""
                <div style="margin-top:1.1rem;">
                <p style="font-size:.66rem;color:#8C8578;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.75rem;">💬 Interpretasi Metrik</p>
                <div style="display:flex;gap:.75rem;flex-wrap:wrap;">

                <div style="flex:1;min-width:250px;background:#fff;border:1px solid #DDD9CF;
                            border-top:3px solid #C4622D;border-radius:12px;padding:1rem 1.2rem;
                            box-shadow:0 2px 8px rgba(196,98,45,0.07);">
                    <div style="font-size:.68rem;font-weight:700;color:#C4622D;text-transform:uppercase;
                                letter-spacing:1.2px;margin-bottom:.6rem;">📋 Kelas 0 — Tidak Layak BSM</div>
                    <div style="display:flex;flex-direction:column;gap:.38rem;">
                        <div style="display:flex;justify-content:space-between;align-items:center;
                                    background:#F7F6F1;border-radius:7px;padding:.4rem .75rem;">
                            <span style="font-size:.78rem;color:#5C5748;font-weight:600;">🎯 Precision</span>
                            <span style="font-size:.85rem;font-weight:800;color:{cp0};">{prec0:.2f} — {ip0}</span></div>
                        <div style="display:flex;justify-content:space-between;align-items:center;
                                    background:#F7F6F1;border-radius:7px;padding:.4rem .75rem;">
                            <span style="font-size:.78rem;color:#5C5748;font-weight:600;">🔍 Recall</span>
                            <span style="font-size:.85rem;font-weight:800;color:{cr0};">{rec0:.2f} — {ir0}</span></div>
                        <div style="display:flex;justify-content:space-between;align-items:center;
                                    background:#F7F6F1;border-radius:7px;padding:.4rem .75rem;">
                            <span style="font-size:.78rem;color:#5C5748;font-weight:600;">⚖️ F1-Score</span>
                            <span style="font-size:.85rem;font-weight:800;color:{cf0};">{f1_0:.2f} — {if0_}</span></div>
                    </div>
                </div>

                <div style="flex:1;min-width:250px;background:#fff;border:1px solid #DDD9CF;
                            border-top:3px solid #2D5A3D;border-radius:12px;padding:1rem 1.2rem;
                            box-shadow:0 2px 8px rgba(45,90,61,0.07);">
                    <div style="font-size:.68rem;font-weight:700;color:#2D5A3D;text-transform:uppercase;
                                letter-spacing:1.2px;margin-bottom:.6rem;">🎓 Kelas 1 — Layak BSM</div>
                    <div style="display:flex;flex-direction:column;gap:.38rem;">
                        <div style="display:flex;justify-content:space-between;align-items:center;
                                    background:#F7F6F1;border-radius:7px;padding:.4rem .75rem;">
                            <span style="font-size:.78rem;color:#5C5748;font-weight:600;">🎯 Precision</span>
                            <span style="font-size:.85rem;font-weight:800;color:{cp1};">{prec1:.2f} — {ip1}</span></div>
                        <div style="display:flex;justify-content:space-between;align-items:center;
                                    background:#F7F6F1;border-radius:7px;padding:.4rem .75rem;">
                            <span style="font-size:.78rem;color:#5C5748;font-weight:600;">🔍 Recall</span>
                            <span style="font-size:.85rem;font-weight:800;color:{cr1};">{rec1:.2f} — {ir1}</span></div>
                        <div style="display:flex;justify-content:space-between;align-items:center;
                                    background:#F7F6F1;border-radius:7px;padding:.4rem .75rem;">
                            <span style="font-size:.78rem;color:#5C5748;font-weight:600;">⚖️ F1-Score</span>
                            <span style="font-size:.85rem;font-weight:800;color:{cf1};">{f1_1:.2f} — {if1_}</span></div>
                    </div>
                </div>

                </div></div>
                """,
                unsafe_allow_html=True,
            )
        except:
            pass

        st.markdown(
            """
            <div class="formula-box">
                <div class="formula-title">📐 Keterangan Metrik Evaluasi</div>
                <strong>Precision</strong> — Dari semua prediksi positif, berapa yang benar:
                &nbsp;<code>TP / (TP + FP)</code><br>
                <strong>Recall</strong> — Dari semua aktual positif, berapa yang terdeteksi:
                &nbsp;<code>TP / (TP + FN)</code><br>
                <strong>F1-Score</strong> — Harmonic mean Precision & Recall:
                &nbsp;<code>2 × (P × R) / (P + R)</code><br>
                <strong>Support</strong> — Jumlah data aktual per kelas dalam data uji
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        col_dla, col_dlb = st.columns(2)
        with col_dla:
            judul_rpt = st.text_input("Judul file laporan", value="Laporan Klasifikasi BSM", key="judul_rpt")
            st.markdown(make_excel_download(report_display, f"{judul_rpt.replace(' ','_')}.xlsx", f"⬇ Unduh {judul_rpt} (.xlsx)", judul_rpt), unsafe_allow_html=True)
        with col_dlb:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(make_png_download(fig_bar, f"{judul_rpt.replace(' ','_')}_chart.png", f"⬇ Unduh Grafik (.png)"), unsafe_allow_html=True)
    else:
        st.info("👈 Jalankan preprocessing dan klik **Proses & Klasifikasi Naive Bayes** di Tab Normalisasi.")

# ─────────────────────────────────────────────────────────
# TAB 7 — Input Data Baru
# ─────────────────────────────────────────────────────────
with tab7:
    st.markdown('<div class="section-header">➕ Input & Prediksi Data Siswa Baru</div>', unsafe_allow_html=True)

    if nb is None:
        st.markdown(
            notif_html(
                "warn", "⚠️",
                "Model Belum Dilatih",
                "Harap jalankan proses Naive Bayes terlebih dahulu. Ikuti langkah: Upload Data → Preprocessing → Atur Split → Klik Proses di Tab Normalisasi.",
                ["1️⃣ Upload Data", "2️⃣ Preprocessing", "3️⃣ Atur Split", "4️⃣ Proses NB"]
            ),
            unsafe_allow_html=True,
        )
    else:
        mode = st.radio(
            "Pilih metode input data:",
            ["📝 Input Manual (Satu per Satu)", "📂 Upload File Banyak Data (CSV/Excel)"],
            horizontal=True,
            label_visibility="visible",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── MODE 1 — Input Manual ─────────────────────────
        if mode == "📝 Input Manual (Satu per Satu)":
            st.markdown(
                '<div class="info-box">🧑‍🎓 Isi data atribut siswa. Label kelayakan akan <strong>diprediksi otomatis</strong> oleh model Gaussian Naive Bayes yang telah dilatih.</div>',
                unsafe_allow_html=True,
            )
            with st.form("form_tambah_data", clear_on_submit=True):
                st.markdown('<p style="font-size:.70rem;color:#8C8578;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.7rem;">📝 Identitas Siswa</p>', unsafe_allow_html=True)
                col_id1, col_id2, col_id3 = st.columns(3)
                with col_id1: inp_nama  = st.text_input("Nama Siswa", placeholder="Contoh: Ahmad Fauzi")
                with col_id2: inp_kelas = st.text_input("Kelas", placeholder="Contoh: X IPA 1")
                with col_id3: inp_nis   = st.text_input("NIS / No. Induk", placeholder="Contoh: 2024001")

                st.markdown('<p style="font-size:.70rem;color:#8C8578;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin:.7rem 0;">📋 Data Atribut Ekonomi</p>', unsafe_allow_html=True)
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    inp_pendapatan = st.selectbox("Pendapatan Orang Tua", options=[0,1,2],
                        format_func=lambda x: {0:"Rendah — Rp500.000–Rp1.000.000",1:"Menengah — Rp3.000.000–Rp4.000.000",2:"Tinggi — Rp5.000.000–Rp7.000.000"}[x])
                    inp_pekerjaan = st.selectbox("Pekerjaan Orang Tua", options=[0,1,2,3],
                        format_func=lambda x: {0:"Petani",1:"PNS",2:"Polisi / TNI",3:"Wiraswasta"}[x])
                with col_f2:
                    inp_tanggungan = st.number_input("Jumlah Tanggungan", min_value=0, max_value=20, value=1, step=1)
                    inp_rumah      = st.selectbox("Status Kepemilikan Rumah", options=[0,1],
                        format_func=lambda x: "Kontrak / Sewa" if x == 0 else "Milik Sendiri")

                submitted = st.form_submit_button("🔍  Prediksi & Tambahkan ke Riwayat", use_container_width=True)

            if submitted:
                if not inp_nama.strip():
                    st.markdown(notif_html("warn","⚠️","Nama Siswa Kosong","Nama siswa tidak boleh kosong.",None), unsafe_allow_html=True)
                else:
                    fitur_input = pd.DataFrame([{
                        "PENDAPATAN ORANG TUA": int(inp_pendapatan),
                        "PEKERJAAN ORANG TUA":  int(inp_pekerjaan),
                        "JUMLAH TANGGUNGAN":    int(inp_tanggungan),
                        "STATUS RUMAH":         int(inp_rumah),
                    }])
                    label_pred = int(nb["model"].predict(fitur_input)[0])
                    label_text = "Layak Menerima Beasiswa" if label_pred == 1 else "Tidak Layak Menerima Beasiswa"
                    css_class  = "pred-layak" if label_pred == 1 else "pred-tidak"
                    icon       = "🎓" if label_pred == 1 else "📋"
                    color      = "#2D5A3D" if label_pred == 1 else "#C4622D"

                    st.markdown(
                        notif_html(
                            "prep" if label_pred == 1 else "warn",
                            icon,
                            f"Hasil Prediksi: {inp_nama.strip()}",
                            f"Siswa <strong>{inp_nama.strip()}</strong> diprediksi: <strong style='color:{color}'>{label_text}</strong>",
                            [
                                f"Pendapatan: {['Rendah','Menengah','Tinggi'][int(inp_pendapatan)]}",
                                f"Pekerjaan: {['Petani','PNS','Polisi','Wiraswasta'][int(inp_pekerjaan)]}",
                                f"Tanggungan: {inp_tanggungan}",
                                f"Rumah: {'Milik Sendiri' if inp_rumah==1 else 'Kontrak/Sewa'}",
                            ]
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="pred-result-box {css_class}">'
                        f'<div class="pred-label" style="color:{color}">{icon} {label_text}</div>'
                        f'<div class="pred-sub">Diprediksi oleh Gaussian Naive Bayes untuk <strong>{inp_nama.strip()}</strong></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    new_row = pd.DataFrame([{
                        "NAMA SISWA":           inp_nama.strip(),
                        "KELAS":                inp_kelas.strip(),
                        "NIS":                  inp_nis.strip(),
                        "PENDAPATAN ORANG TUA": int(inp_pendapatan),
                        "PEKERJAAN ORANG TUA":  int(inp_pekerjaan),
                        "JUMLAH TANGGUNGAN":    int(inp_tanggungan),
                        "STATUS RUMAH":         int(inp_rumah),
                        "LABEL (PREDIKSI)":     label_pred,
                    }])
                    st.session_state["data_tambahan"] = pd.concat(
                        [st.session_state["data_tambahan"], new_row], ignore_index=True
                    )

        # ── MODE 2 — Upload File ──────────────────────────
        else:
            st.markdown(
                """<div class="info-box">
                📂 Upload file CSV atau Excel dengan kolom:
                <strong>NAMA SISWA</strong>, <strong>KELAS</strong>, <strong>NIS</strong>,
                <strong>PENDAPATAN ORANG TUA</strong> (0/1/2),
                <strong>PEKERJAAN ORANG TUA</strong> (0/1/2/3),
                <strong>JUMLAH TANGGUNGAN</strong>,
                <strong>STATUS RUMAH</strong> (0/1)
                </div>""",
                unsafe_allow_html=True,
            )
            template_df = pd.DataFrame([{"NAMA SISWA":"Ahmad Fauzi","KELAS":"X IPA 1","NIS":"2024001","PENDAPATAN ORANG TUA":0,"PEKERJAAN ORANG TUA":0,"JUMLAH TANGGUNGAN":4,"STATUS RUMAH":0}])
            st.markdown(make_excel_download(template_df, "template_input_data.xlsx", "⬇ Unduh Template File (.xlsx)"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            uploaded_bulk = st.file_uploader("Upload file data siswa", type=["csv","xlsx"], key="uploader_bulk")
            if uploaded_bulk is not None:
                try:
                    if uploaded_bulk.name.endswith(".csv"):
                        df_bulk = pd.read_csv(uploaded_bulk)
                    else:
                        df_bulk = pd.read_excel(uploaded_bulk)
                    df_bulk.columns = df_bulk.columns.str.upper().str.strip()
                    required_cols   = ["PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH"]
                    missing_cols    = [c for c in required_cols if c not in df_bulk.columns]

                    if missing_cols:
                        st.markdown(notif_html("warn","❌","Kolom Tidak Ditemukan",f"Kolom tidak ada: {', '.join(missing_cols)}",None), unsafe_allow_html=True)
                    else:
                        st.markdown(
                            notif_html("upload","📂",f"File Siap — {len(df_bulk)} baris ditemukan",
                                       f"<strong>{uploaded_bulk.name}</strong> siap diproses.",
                                       [f"📋 Total: {len(df_bulk)}", f"📊 Kolom: {len(df_bulk.columns)}"]),
                            unsafe_allow_html=True,
                        )
                        st.dataframe(df_bulk.head(10), use_container_width=True)

                        if st.button("🔍  Prediksi Semua Data dari File", use_container_width=False):
                            fitur_cols  = ["PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH"]
                            X_bulk      = df_bulk[fitur_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
                            labels_pred = nb["model"].predict(X_bulk).tolist()
                            hasil_bulk  = df_bulk.copy()
                            hasil_bulk["LABEL (PREDIKSI)"] = labels_pred
                            for col_id in ["NAMA SISWA","KELAS","NIS"]:
                                if col_id not in hasil_bulk.columns:
                                    hasil_bulk[col_id] = "-"
                            cols_order = ["NAMA SISWA","KELAS","NIS","PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL (PREDIKSI)"]
                            hasil_bulk = hasil_bulk[[c for c in cols_order if c in hasil_bulk.columns]]
                            st.session_state["data_tambahan"] = pd.concat(
                                [st.session_state["data_tambahan"], hasil_bulk], ignore_index=True
                            )
                            layak_count = sum(1 for l in labels_pred if l == 1)
                            tidak_count = sum(1 for l in labels_pred if l == 0)
                            st.markdown(
                                notif_html("prep","✅",f"Prediksi Selesai — {len(df_bulk)} data diproses",
                                           "Data telah diprediksi dan ditambahkan ke riwayat.",
                                           [f"🎓 Layak: {layak_count}", f"📋 Tidak Layak: {tidak_count}", f"📊 Total: {len(df_bulk)}"]),
                                unsafe_allow_html=True,
                            )
                except Exception as e:
                    st.markdown(notif_html("warn","❌","Error Membaca File",f"Terjadi error: {e}",None), unsafe_allow_html=True)

        # ── Riwayat ───────────────────────────────────────
        st.markdown("---")
        dt = st.session_state["data_tambahan"]
        col_hd, col_hapus = st.columns([3, 1])
        with col_hd:
            st.markdown(
                f'<div class="section-header">📋 Riwayat Prediksi '
                f'<span style="color:#8C8578;font-size:.82rem;font-family:DM Sans,sans-serif;font-weight:500;">({len(dt)} siswa)</span></div>',
                unsafe_allow_html=True,
            )
        with col_hapus:
            if st.button("🗑  Hapus Semua Riwayat", key="hapus_riwayat"):
                st.session_state["data_tambahan"] = pd.DataFrame(
                    columns=["NAMA SISWA","KELAS","NIS","PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL (PREDIKSI)"]
                )
                st.rerun()

        if len(dt) > 0:
            dt_display = dt.copy()
            dt_display.insert(0, "No", range(1, len(dt_display)+1))
            label_map  = {0:"Tidak Layak (0)", 1:"Layak (1)"}
            rumah_map  = {0:"Kontrak/Sewa (0)", 1:"Milik Sendiri (1)"}
            kerja_map  = {0:"Petani (0)", 1:"PNS (1)", 2:"Polisi (2)", 3:"Wiraswasta (3)"}
            income_map = {0:"Rendah (0)", 1:"Menengah (1)", 2:"Tinggi (2)"}
            for col_name, mapping in [("LABEL (PREDIKSI)", label_map), ("STATUS RUMAH", rumah_map),
                                       ("PEKERJAAN ORANG TUA", kerja_map), ("PENDAPATAN ORANG TUA", income_map)]:
                if col_name in dt_display.columns:
                    dt_display[col_name] = dt_display[col_name].apply(lambda x: mapping.get(int(x), x) if pd.notna(x) else x)
            st.dataframe(dt_display, use_container_width=True, height=300)

            layak_n = int(dt["LABEL (PREDIKSI)"].sum())
            tidak_n = len(dt) - layak_n
            st.markdown(
                notif_html("prep","📊",f"Ringkasan — {len(dt)} siswa diprediksi",
                           "Distribusi kelayakan dari semua data yang dimasukkan.",
                           [f"🎓 Layak BSM: {layak_n}", f"📋 Tidak Layak: {tidak_n}", f"👥 Total: {len(dt)}"]),
                unsafe_allow_html=True,
            )

            st.markdown("---")
            col_dla, col_dlb, col_dlc = st.columns(3)
            with col_dla:
                judul_tambahan = st.text_input("Judul file riwayat", value="Riwayat Prediksi BSM", key="judul_tambahan")
            with col_dlb:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(make_excel_download(dt, f"{judul_tambahan.replace(' ','_')}.xlsx", f"⬇ Unduh {judul_tambahan} (.xlsx)", judul_tambahan), unsafe_allow_html=True)
            with col_dlc:
                if df_clean is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    dt_gabung = dt[["PENDAPATAN ORANG TUA","PEKERJAAN ORANG TUA","JUMLAH TANGGUNGAN","STATUS RUMAH","LABEL (PREDIKSI)"]].copy()
                    dt_gabung = dt_gabung.rename(columns={"LABEL (PREDIKSI)":"LABEL"})
                    gabungan  = pd.concat([df_clean, dt_gabung], ignore_index=True)
                    st.markdown(make_excel_download(gabungan, f"{judul_tambahan.replace(' ','_')}_Gabungan.xlsx", f"⬇ Unduh Data Gabungan (.xlsx)", judul_tambahan+" (Gabungan)"), unsafe_allow_html=True)

            # Distribution chart
            label_dist = dt["LABEL (PREDIKSI)"].value_counts().rename({0:"Tidak Layak", 1:"Layak"})
            fig_dt, ax_dt = plt.subplots(figsize=(4, 2.4))
            fig_dt.patch.set_facecolor("#FFFFFF")
            ax_dt.set_facecolor("#F7F6F1")
            bar_c_dt = ["#C4622D" if "Tidak" in str(k) else "#2D5A3D" for k in label_dist.index]
            bars_dt  = ax_dt.bar(label_dist.index, label_dist.values, color=bar_c_dt,
                                  edgecolor="#ffffff", width=0.48, linewidth=1.5)
            ax_dt.tick_params(colors="#8C8578")
            ax_dt.set_ylabel("Jumlah Siswa", color="#8C8578", fontsize=9)
            for spine in ax_dt.spines.values(): spine.set_edgecolor("#DDD9CF")
            for bar in bars_dt:
                ax_dt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.04,
                           str(int(bar.get_height())), ha="center", va="bottom",
                           color="#1C1A16", fontsize=11, fontweight="bold")
            ax_dt.set_title("Distribusi Kelayakan — Riwayat", color="#1C2E20", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_dt)
        else:
            st.markdown(
                '<div style="text-align:center;color:#8C8578;padding:2.3rem 2rem;'
                'border:1.5px dashed #DDD9CF;border-radius:14px;margin-top:0.9rem;background:#F7F6F1;">'
                '<div style="font-size:2rem;margin-bottom:.6rem;">🧑‍🎓</div>'
                '<div style="font-weight:700;color:#5C5748;margin-bottom:.28rem;">Belum Ada Data</div>'
                '<div style="font-size:.78rem;">Isi form manual atau upload file di atas, lalu klik prediksi.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#8C8578;font-size:.76rem;padding:.5rem 0 1rem;
                letter-spacing:.3px;font-family:'DM Sans',sans-serif;">
        🎓 Sistem Klasifikasi Beasiswa Siswa Miskin &nbsp;·&nbsp;
        Algoritma Gaussian Naive Bayes &nbsp;·&nbsp;
        Dibangun dengan <span style="color:#C4622D;">♥</span> menggunakan Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)