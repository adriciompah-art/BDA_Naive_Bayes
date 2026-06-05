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
# CUSTOM CSS  —  Professional Education Theme
# Palet: Biru Indigo (#1E3A5F) + Hijau Teal (#0D9488) + Abu & Putih Bersih
# Tipografi: DM Sans (body) + DM Serif Display (heading)
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── COLOUR TOKENS ──────────────────────────────────── */
    :root {
        /* Backgrounds */
        --bg:           #F0F4F8;
        --surface:      #FFFFFF;
        --surface2:     #F7F9FC;
        --surface3:     #EEF2F7;
        --border:       #D9E2EE;
        --border2:      #C4D0E3;

        /* Brand */
        --navy:         #1E3A5F;
        --navy-mid:     #2D5282;
        --navy-light:   #3B6CB0;
        --teal:         #0D9488;
        --teal-light:   #14B8A6;
        --teal-pale:    #CCFBF1;
        --amber:        #D97706;
        --amber-pale:   #FEF3C7;
        --rose:         #E11D48;
        --rose-pale:    #FFE4E6;
        --indigo:       #4338CA;
        --indigo-pale:  #E0E7FF;

        /* Text */
        --text:         #0F172A;
        --text-body:    #334155;
        --text-dim:     #64748B;
        --text-muted:   #94A3B8;

        /* Effects */
        --shadow-sm:    0 1px 3px rgba(15,23,42,.08), 0 1px 2px rgba(15,23,42,.06);
        --shadow-md:    0 4px 16px rgba(15,23,42,.10), 0 2px 6px rgba(15,23,42,.07);
        --shadow-lg:    0 10px 40px rgba(15,23,42,.13), 0 4px 12px rgba(15,23,42,.08);
        --radius:       10px;
        --radius-lg:    16px;
        --radius-xl:    22px;
    }

    /* ── BASE ──────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text-body);
        font-size: 15px;
        line-height: 1.6;
    }

    .main .block-container {
        padding: 0 2.5rem 4rem;
        max-width: 1500px;
    }

    /* ── HEADER BANNER ─────────────────────────────────── */
    .header-banner {
        background: linear-gradient(135deg, #0F2444 0%, #1E3A5F 40%, #2D5282 75%, #1B4B82 100%);
        margin: 0 -2.5rem 2.5rem;
        padding: 0;
        position: relative;
        overflow: hidden;
        border-bottom: 3px solid var(--teal);
    }

    /* Subtle dot-grid pattern */
    .header-banner::before {
        content: '';
        position: absolute; inset: 0;
        background-image: radial-gradient(rgba(255,255,255,.06) 1px, transparent 1px);
        background-size: 28px 28px;
        pointer-events: none;
    }

    /* Teal glow bottom-right */
    .header-banner::after {
        content: '';
        position: absolute; bottom: -30px; right: -30px;
        width: 340px; height: 340px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(13,148,136,.22) 0%, transparent 70%);
        pointer-events: none;
    }

    .header-inner {
        position: relative;
        z-index: 2;
        padding: 3rem 3.5rem 2.8rem;
        display: flex;
        align-items: center;
        gap: 2.2rem;
    }

    /* Icon badge */
    .header-badge {
        width: 76px; height: 76px;
        background: linear-gradient(145deg, var(--teal), var(--teal-light));
        border-radius: 20px;
        display: flex; align-items: center; justify-content: center;
        font-size: 2.2rem;
        flex-shrink: 0;
        box-shadow: 0 8px 24px rgba(13,148,136,.35);
    }

    .header-eyebrow {
        font-size: .72rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--teal-light);
        margin-bottom: .55rem;
    }

    .header-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        font-weight: 400;
        color: #FFFFFF;
        margin: 0 0 .4rem;
        line-height: 1.1;
        letter-spacing: -.5px;
    }
    .header-title .accent { color: var(--teal-light); font-style: italic; }

    .header-sub {
        font-size: .9rem;
        color: rgba(255,255,255,.5);
        font-weight: 400;
    }

    /* Right-side decorative pills */
    .header-pills {
        margin-left: auto;
        display: flex; flex-direction: column; gap: .5rem; align-items: flex-end;
    }
    .header-pill {
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 100px;
        padding: .3rem .85rem;
        font-size: .75rem;
        color: rgba(255,255,255,.65);
        letter-spacing: .5px;
        font-weight: 500;
    }
    .header-pill.highlight {
        background: rgba(13,148,136,.25);
        border-color: rgba(13,148,136,.45);
        color: var(--teal-light);
    }

    /* ── METRIC CARDS ──────────────────────────────────── */
    .metric-row {
        display: flex; gap: 1.1rem;
        margin-bottom: 2.2rem;
        flex-wrap: wrap;
    }
    .metric-card {
        flex: 1; min-width: 155px;
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: 1.4rem 1.6rem 1.2rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border);
        transition: transform .22s, box-shadow .22s;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-md);
    }
    /* Top colour bar */
    .metric-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0;
        height: 4px;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }
    .metric-card.navy::before   { background: linear-gradient(90deg, var(--navy), var(--navy-light)); }
    .metric-card.teal::before   { background: linear-gradient(90deg, var(--teal), var(--teal-light)); }
    .metric-card.amber::before  { background: linear-gradient(90deg, var(--amber), #F59E0B); }
    .metric-card.indigo::before { background: linear-gradient(90deg, var(--indigo), #6366F1); }

    /* Soft background blob */
    .metric-card::after {
        content: attr(data-icon);
        position: absolute; right: 1.2rem; bottom: .6rem;
        font-size: 3rem;
        opacity: .06;
        pointer-events: none;
    }

    .metric-card .mc-icon {
        font-size: 1.5rem;
        margin-bottom: .7rem;
        display: block;
    }
    .metric-card .mc-label {
        font-size: .72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: .35rem;
    }
    .metric-card .mc-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        line-height: 1;
        color: var(--text);
    }
    .mc-navy   { color: var(--navy)   !important; }
    .mc-teal   { color: var(--teal)   !important; }
    .mc-amber  { color: var(--amber)  !important; }
    .mc-indigo { color: var(--indigo) !important; }

    /* ── SIDEBAR ───────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0F2444;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: rgba(255,255,255,.80) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .sidebar-logo {
        padding: .8rem 1rem 1.4rem;
        border-bottom: 1px solid rgba(255,255,255,.08);
        margin-bottom: 1.4rem;
    }
    .sidebar-logo-top {
        display: flex; align-items: center; gap: .8rem;
        margin-bottom: .6rem;
    }
    .sidebar-logo-icon {
        font-size: 1.8rem;
        background: rgba(13,148,136,.25);
        border-radius: 10px;
        padding: .3rem .5rem;
    }
    .sidebar-logo-name {
        font-family: 'DM Serif Display', serif;
        font-size: 1.2rem;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .sidebar-logo-name span { color: var(--teal-light); }
    .sidebar-logo-desc {
        font-size: .72rem;
        color: rgba(255,255,255,.35);
        letter-spacing: .5px;
    }

    .sidebar-section-label {
        font-size: .67rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: rgba(20,184,166,.65);
        margin: 1.4rem 0 .6rem;
        padding-left: .1rem;
    }

    /* ── PIPELINE ──────────────────────────────────────── */
    .pipeline { display: flex; flex-direction: column; gap: .4rem; }
    .pipeline-step {
        display: flex; align-items: center; gap: .7rem;
        padding: .55rem .85rem;
        border-radius: 8px;
        border: 1px solid transparent;
        font-size: .82rem;
        font-weight: 500;
        color: rgba(255,255,255,.4);
        transition: all .2s;
    }
    .pipeline-step.done {
        background: rgba(13,148,136,.12);
        border-color: rgba(13,148,136,.3);
        color: var(--teal-light);
    }
    .pipeline-step.active {
        background: rgba(255,255,255,.06);
        border-color: rgba(255,255,255,.15);
        color: rgba(255,255,255,.8);
    }
    .step-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .dot-done   { background: var(--teal-light); }
    .dot-active { background: rgba(255,255,255,.55); }
    .dot-idle   { background: rgba(255,255,255,.15); }

    /* ── SECTION HEADER ────────────────────────────────── */
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.45rem;
        font-weight: 400;
        color: var(--navy);
        padding: .4rem 0 .4rem 1.1rem;
        border-left: 4px solid var(--teal);
        margin-bottom: 1.3rem;
        line-height: 1.2;
    }

    /* ── INFO BOX ──────────────────────────────────────── */
    .info-box {
        background: linear-gradient(135deg, #EFF6FF, #F0FDFA);
        border: 1px solid rgba(13,148,136,.25);
        border-left: 3px solid var(--teal);
        border-radius: var(--radius);
        padding: .9rem 1.2rem;
        margin-bottom: 1.2rem;
        font-size: .875rem;
        color: var(--navy);
        line-height: 1.65;
    }

    /* ── PRED RESULT ───────────────────────────────────── */
    .pred-result-box {
        border-radius: var(--radius-lg);
        padding: 1.8rem 2rem;
        margin-top: 1rem;
        text-align: center;
        border: 2px solid;
    }
    .pred-layak {
        background: linear-gradient(135deg, #F0FDF9, #ECFDF5);
        border-color: rgba(13,148,136,.35);
    }
    .pred-tidak {
        background: linear-gradient(135deg, #FFF1F2, #FFF5F5);
        border-color: rgba(225,29,72,.3);
    }
    .pred-label {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        font-weight: 400;
    }
    .pred-sub { font-size: .83rem; color: var(--text-dim); margin-top: .5rem; }

    /* ── TABS ──────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border-bottom: 2px solid var(--border);
        gap: 0; padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-dim);
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: .845rem;
        padding: .9rem 1.4rem;
        border: none;
        border-bottom: 3px solid transparent;
        margin-bottom: -2px;
        text-transform: none;
        letter-spacing: .2px;
        transition: color .2s, border-color .2s;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--navy); }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: var(--teal) !important;
        border-bottom: 3px solid var(--teal) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 var(--radius) var(--radius);
        padding: 2rem 2.2rem;
    }

    /* ── DATAFRAME ─────────────────────────────────────── */
    .stDataFrame {
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        border: 1px solid var(--border) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stDataFrame thead th {
        background: var(--navy) !important;
        color: rgba(255,255,255,.9) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: .80rem !important;
        letter-spacing: .6px !important;
    }
    .stDataFrame tbody tr:nth-child(even) td {
        background: rgba(13,148,136,.03) !important;
    }
    .stDataFrame tbody tr:hover td {
        background: rgba(30,58,95,.05) !important;
    }

    /* ── BUTTONS ───────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, var(--navy), var(--navy-light));
        color: #fff;
        border: none;
        border-radius: var(--radius);
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: .9rem;
        padding: .65rem 1.5rem;
        cursor: pointer;
        transition: all .22s;
        width: 100%;
        box-shadow: var(--shadow-sm);
        letter-spacing: .2px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--navy-mid), var(--indigo));
        box-shadow: 0 6px 20px rgba(30,58,95,.3);
        transform: translateY(-1px);
    }

    /* ── DOWNLOAD LINK ─────────────────────────────────── */
    .dl-btn {
        display: inline-flex; align-items: center; gap: .45rem;
        background: var(--surface3);
        color: var(--navy) !important;
        text-decoration: none;
        border: 1.5px solid var(--border2);
        border-radius: var(--radius);
        padding: .5rem 1.2rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: .845rem;
        transition: all .2s;
        margin-top: .5rem;
    }
    .dl-btn:hover {
        border-color: var(--teal);
        background: var(--teal-pale);
        color: var(--teal) !important;
        box-shadow: 0 0 12px rgba(13,148,136,.15);
    }

    /* ── FILE UPLOADER ─────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.04);
        border: 1.5px dashed rgba(20,184,166,.3);
        border-radius: var(--radius);
        padding: .85rem;
    }
    [data-testid="stFileUploader"]:hover { border-color: rgba(20,184,166,.55); }

    /* ── MISC ──────────────────────────────────────────── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0; }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--surface3); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--teal); }

    [data-testid="stAlert"] {
        border-radius: var(--radius) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    footer { display: none !important; }

    /* ── FORM LABEL VISIBILITY ─────────────────────────── */
    .stSelectbox label, .stNumberInput label { color: var(--text-body) !important; font-weight: 500 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE
# =========================================================

for key in ["raw", "df_clean", "nb_results", "data_tambahan"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state["data_tambahan"] is None:
    st.session_state["data_tambahan"] = pd.DataFrame(
        columns=[
            "PENDAPATAN ORANG TUA",
            "PEKERJAAN ORANG TUA",
            "JUMLAH TANGGUNGAN",
            "STATUS RUMAH",
            "LABEL (PREDIKSI)",
        ]
    )

# =========================================================
# HELPER: DOWNLOAD LINK
# =========================================================

def make_excel_download(df: pd.DataFrame, filename: str, label: str) -> str:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = (
        f'<a class="dl-btn" href="data:application/vnd.openxmlformats-officedocument'
        f'.spreadsheetml.sheet;base64,{b64}" download="{filename}">⬇ {label}</a>'
    )
    return href


def make_png_download(fig, filename: str, label: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = (
        f'<a class="dl-btn" href="data:image/png;base64,{b64}" download="{filename}">'
        f'⬇ {label}</a>'
    )
    return href


# =========================================================
# NORMALISASI FUNCTIONS
# =========================================================

def kategori_pendapatan(x):
    x = str(x).lower().replace(".", "").replace("rp", "").strip()
    angka = "".join(filter(str.isdigit, x))
    if angka == "":
        return np.nan
    angka = int(angka)
    if 500_000 <= angka <= 1_000_000:
        return 0
    if 3_000_000 <= angka <= 4_000_000:
        return 1
    if 5_000_000 <= angka <= 7_000_000:
        return 2
    return np.nan


def kategori_pekerjaan(x):
    mapping = {"petani": 0, "pns": 1, "polisi": 2, "wiraswasta": 3}
    return mapping.get(str(x).lower().strip(), np.nan)


def kategori_rumah(x):
    x = str(x).lower().strip()
    if x == "milik sendiri":
        return 1
    if x in ["kontrak", "sewa", "kontrak/sewa"]:
        return 0
    return np.nan


def kategori_label(x):
    x = str(x).lower().strip()
    if x == "ya":
        return 1
    if x == "tidak":
        return 0
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
            return None, "❌ Format file tidak didukung (gunakan .csv atau .xlsx)"
        return raw, "✅ Data berhasil diupload"
    except Exception as e:
        return None, f"❌ Error membaca file: {e}"


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_data(raw: pd.DataFrame) -> tuple:
    try:
        header = raw.iloc[2]
        df = raw.iloc[3:].copy()
        df.columns = header
        df.reset_index(drop=True, inplace=True)
        df.columns = (
            df.columns.astype(str)
            .str.upper()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        col_map = {}
        for col in df.columns:
            if "PENDAPATAN" in col:
                col_map["pendapatan"] = col
            if "PEKERJAAN" in col:
                col_map["pekerjaan"] = col
            if "TANGGUNGAN" in col:
                col_map["tanggungan"] = col
            if "RUMAH" in col:
                col_map["rumah"] = col
            if "LABEL" in col:
                col_map["label"] = col

        df[col_map["pendapatan"]] = df[col_map["pendapatan"]].apply(kategori_pendapatan)
        df[col_map["pekerjaan"]] = df[col_map["pekerjaan"]].apply(kategori_pekerjaan)
        df[col_map["tanggungan"]] = pd.to_numeric(df[col_map["tanggungan"]], errors="coerce")
        df[col_map["rumah"]]      = df[col_map["rumah"]].apply(kategori_rumah)
        df[col_map["label"]]      = df[col_map["label"]].apply(kategori_label)

        df.rename(
            columns={
                col_map["pendapatan"]: "PENDAPATAN ORANG TUA",
                col_map["pekerjaan"]: "PEKERJAAN ORANG TUA",
                col_map["tanggungan"]: "JUMLAH TANGGUNGAN",
                col_map["rumah"]: "STATUS RUMAH",
                col_map["label"]: "LABEL",
            },
            inplace=True,
        )

        df.dropna(
            subset=[
                "PENDAPATAN ORANG TUA",
                "PEKERJAAN ORANG TUA",
                "JUMLAH TANGGUNGAN",
                "STATUS RUMAH",
                "LABEL",
            ],
            inplace=True,
        )

        return df, "✅ Preprocessing berhasil"
    except Exception as e:
        return None, f"❌ Error preprocessing: {e}"


# =========================================================
# GAUSSIAN PROBABILITY
# =========================================================

def gaussian_probability(x, mean, std):
    if std == 0 or pd.isna(std):
        std = 0.0001
    exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


# =========================================================
# NAIVE BAYES — Professional Education colour palette
# =========================================================

def naive_bayes_process(df: pd.DataFrame) -> dict:
    fitur = [
        "PENDAPATAN ORANG TUA",
        "PEKERJAAN ORANG TUA",
        "JUMLAH TANGGUNGAN",
        "STATUS RUMAH",
    ]
    target = "LABEL"

    X, y = df[fitur], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    classes = np.unique(y_train)
    prior, mean, std = {}, {}, {}
    for c in classes:
        prior[c] = len(y_train[y_train == c]) / len(y_train)
        data_c = X_train[y_train == c]
        mean[c] = data_c.mean()
        std[c]  = data_c.std()

    tabel_prior = pd.DataFrame(
        {"Kelas": list(prior.keys()), "Prior Probability": list(prior.values())}
    )

    hasil_manual = []
    for i in range(len(X_test)):
        row = X_test.iloc[i]
        probs = {}
        for c in classes:
            probs[c] = prior[c]
            for f in fitur:
                probs[c] *= gaussian_probability(row[f], mean[c][f], std[c][f])
        prediksi = max(probs, key=probs.get)
        hasil_manual.append(
            {
                "Data Ke": i + 1,
                "P(Tidak Layak=0)": probs.get(0, 0),
                "P(Layak=1)":       probs.get(1, 0),
                "Prediksi":         prediksi,
                "Aktual":           y_test.iloc[i],
                "Benar?":           "✅" if prediksi == y_test.iloc[i] else "❌",
            }
        )

    hasil_manual_df = pd.DataFrame(hasil_manual)
    cm       = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).transpose()

    # ─── Chart colour palette (professional) ───────────
    BG       = "#F7F9FC"
    SURFACE  = "#FFFFFF"
    NAVY     = "#1E3A5F"
    NAVY_MID = "#2D5282"
    TEAL     = "#0D9488"
    TEAL_L   = "#14B8A6"
    AMBER    = "#D97706"
    ROSE     = "#E11D48"
    MUTED    = "#64748B"
    BORDER   = "#D9E2EE"
    TEXT     = "#0F172A"

    # ── Confusion Matrix ────────────────────────────────
    fig_cm, ax = plt.subplots(figsize=(7, 6))
    fig_cm.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    n_rows, n_cols = cm.shape
    cell_colors = [
        [TEAL,  ROSE],
        [ROSE,  TEAL],
    ]
    cell_labels = [["TN", "FP"], ["FN", "TP"]]

    for i in range(n_rows):
        for j in range(n_cols):
            val = cm[i, j]
            bg_color = cell_colors[i][j]

            rect = plt.Rectangle(
                (j + .06, i + .06), .88, .88,
                color=bg_color,
                alpha=.10,
                zorder=0,
            )
            ax.add_patch(rect)

            border_rect = plt.Rectangle(
                (j + .06, i + .06), .88, .88,
                fill=False,
                edgecolor=bg_color,
                alpha=.40,
                linewidth=1.8,
                zorder=1,
            )
            ax.add_patch(border_rect)

            ax.text(
                j + 0.5, i + 0.4, str(val),
                ha="center", va="center",
                fontsize=48, fontweight="bold",
                color=bg_color,
                zorder=2,
                fontfamily="monospace",
            )
            ax.text(
                j + 0.5, i + 0.74,
                cell_labels[i][j],
                ha="center", va="center",
                fontsize=11, fontweight="700",
                color=bg_color,
                alpha=.70,
                zorder=2,
            )

    for k in range(1, n_rows):
        ax.axhline(k, color=BORDER, linewidth=1.5, zorder=3)
    for k in range(1, n_cols):
        ax.axvline(k, color=BORDER, linewidth=1.5, zorder=3)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1.5)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    ax.set_xticks([.5, 1.5])
    ax.set_yticks([.5, 1.5])
    ax.set_xticklabels(["Tidak Layak (0)", "Layak (1)"], color=TEXT, fontsize=11, fontweight="600")
    ax.set_yticklabels(["Tidak Layak (0)", "Layak (1)"], color=TEXT, fontsize=11, fontweight="600", rotation=0, va="center")

    ax.set_xlabel("Prediksi", color=MUTED, labelpad=14, fontsize=12)
    ax.set_ylabel("Aktual",   color=MUTED, labelpad=14, fontsize=12)
    ax.set_title(
        "Confusion Matrix — Hasil Evaluasi Model",
        color=NAVY, fontsize=14, fontweight="bold", pad=22,
        fontfamily="DejaVu Serif",
    )
    ax.tick_params(colors=BORDER, length=0)

    legend_items = [
        mpatches.Patch(color=TEAL, label="Prediksi Benar  (TN / TP)"),
        mpatches.Patch(color=ROSE, label="Prediksi Salah  (FP / FN)"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        facecolor=SURFACE,
        edgecolor=BORDER,
        labelcolor=TEXT,
        fontsize=10,
    )
    plt.tight_layout()

    return {
        "prior":      tabel_prior,
        "manual":     hasil_manual_df,
        "report":     report_df,
        "accuracy":   accuracy,
        "fig_cm":     fig_cm,
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "classes":    classes.tolist(),
        "model":      model,
        "cm":         cm,
    }


# =========================================================
# HEADER BANNER
# =========================================================

st.markdown(
    """
    <div class="header-banner">
        <div class="header-inner">
            <div class="header-badge">🎓</div>
            <div class="header-text">
                <div class="header-eyebrow">Sistem Pendukung Keputusan · Gaussian Naive Bayes</div>
                <h1 class="header-title">Klasifikasi <span class="accent">Beasiswa Siswa Miskin</span></h1>
                <p class="header-sub">Analisis kelayakan penerima BSM menggunakan algoritma Machine Learning</p>
            </div>
            <div class="header-pills">
                <div class="header-pill highlight">✦ Machine Learning</div>
                <div class="header-pill">📚 Gaussian Naive Bayes</div>
                <div class="header-pill">🏫 Data Siswa</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="sidebar-logo-top">
                <div class="sidebar-logo-icon">🎓</div>
                <div class="sidebar-logo-name">BSM<br><span>Classifier</span></div>
            </div>
            <div class="sidebar-logo-desc">Klasifikasi Kelayakan Beasiswa Siswa Miskin</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">📂 Unggah Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Pilih file CSV atau Excel",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⬆ Upload Dataset"):
            raw, msg = load_data(uploaded)
            if raw is not None:
                st.session_state["raw"]        = raw
                st.session_state["df_clean"]   = None
                st.session_state["nb_results"] = None
                st.success(msg)
            else:
                st.error(msg)

    st.markdown('<div class="sidebar-section-label">⚙ Jalankan Proses</div>', unsafe_allow_html=True)

    if st.session_state["raw"] is not None:
        if st.button("🔧 Jalankan Preprocessing"):
            df, msg = preprocess_data(st.session_state["raw"])
            if df is not None:
                st.session_state["df_clean"]   = df
                st.session_state["nb_results"] = None
                st.success(msg)
            else:
                st.error(msg)

    if st.session_state["df_clean"] is not None:
        if st.button("🤖 Proses Naive Bayes"):
            with st.spinner("Melatih model..."):
                st.session_state["nb_results"] = naive_bayes_process(
                    st.session_state["df_clean"]
                )
            st.success("✅ Model selesai dilatih!")

    # Pipeline status
    st.markdown('<div class="sidebar-section-label">📋 Status Pipeline</div>', unsafe_allow_html=True)
    raw_ok   = st.session_state["raw"]        is not None
    clean_ok = st.session_state["df_clean"]   is not None
    nb_ok    = st.session_state["nb_results"] is not None

    steps = [
        ("Upload Data",    raw_ok,   raw_ok and not clean_ok),
        ("Preprocessing",  clean_ok, clean_ok and not nb_ok),
        ("Naive Bayes",    nb_ok,    False),
    ]

    pipeline_html = '<div class="pipeline">'
    for label, done, active in steps:
        if done:
            cls = "done"; dot_cls = "dot-done"
        elif active:
            cls = "active"; dot_cls = "dot-active"
        else:
            cls = ""; dot_cls = "dot-idle"
        icon = "✓" if done else ("▶" if active else "○")
        pipeline_html += (
            f'<div class="pipeline-step {cls}">'
            f'<span class="step-dot {dot_cls}"></span>{icon} {label}'
            f'</div>'
        )
    pipeline_html += "</div>"
    st.markdown(pipeline_html, unsafe_allow_html=True)


# =========================================================
# METRIC CARDS
# =========================================================

raw      = st.session_state["raw"]
df_clean = st.session_state["df_clean"]
nb       = st.session_state["nb_results"]

total_raw   = len(raw)      if raw      is not None else "–"
total_clean = len(df_clean) if df_clean is not None else "–"
akurasi_val = f"{nb['accuracy']*100:.1f}%" if nb else "–"
test_size   = nb["test_size"]              if nb else "–"

st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-card navy" data-icon="📋">
            <span class="mc-icon">📋</span>
            <div class="mc-label">Total Data Awal</div>
            <div class="mc-value mc-navy">{total_raw}</div>
        </div>
        <div class="metric-card teal" data-icon="✅">
            <span class="mc-icon">✅</span>
            <div class="mc-label">Setelah Preprocessing</div>
            <div class="mc-value mc-teal">{total_clean}</div>
        </div>
        <div class="metric-card indigo" data-icon="🧪">
            <span class="mc-icon">🧪</span>
            <div class="mc-label">Data Testing</div>
            <div class="mc-value mc-indigo">{test_size}</div>
        </div>
        <div class="metric-card amber" data-icon="🏅">
            <span class="mc-icon">🏅</span>
            <div class="mc-label">Akurasi Model</div>
            <div class="mc-value mc-amber">{akurasi_val}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📁 Data Awal",
        "🔧 Normalisasi",
        "📊 Prior & Manual",
        "🎯 Hasil Prediksi",
        "🔲 Confusion Matrix",
        "📈 Laporan Klasifikasi",
        "➕ Input Data Baru",
    ]
)

# ─────────────────────────────────────────────────────────
# TAB 1 — Dataset Awal
# ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📁 Dataset Mentah</div>', unsafe_allow_html=True)
    if raw is not None:
        st.markdown(
            '<div class="info-box">💡 Ini adalah data asli sebelum preprocessing. Header tabel dimulai dari baris ke-3 (index 2) pada file sumber.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(raw, use_container_width=True, height=420)
        st.markdown(make_excel_download(raw, "dataset_awal.xlsx", "Unduh Dataset Awal (.xlsx)"), unsafe_allow_html=True)
    else:
        st.info("👈 Unggah dataset terlebih dahulu melalui panel kiri untuk memulai.")

# ─────────────────────────────────────────────────────────
# TAB 2 — Normalisasi
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🔧 Hasil Normalisasi & Encoding</div>', unsafe_allow_html=True)
    if df_clean is not None:
        st.markdown(
            """<div class="info-box">
            💡 Setiap atribut diubah ke bentuk numerik agar dapat diproses model.<br>
            <strong>Pendapatan</strong> → 0 (Rendah) / 1 (Menengah) / 2 (Tinggi) &nbsp;|&nbsp;
            <strong>Pekerjaan</strong> → Petani=0, PNS=1, Polisi=2, Wiraswasta=3 &nbsp;|&nbsp;
            <strong>Rumah</strong> → Kontrak/Sewa=0, Milik Sendiri=1 &nbsp;|&nbsp;
            <strong>Label</strong> → Tidak Layak=0, Layak=1
            </div>""",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.dataframe(df_clean, use_container_width=True, height=420)
        with col_b:
            st.markdown(
                '<p style="font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.7rem;">Distribusi Label Siswa</p>',
                unsafe_allow_html=True,
            )
            label_counts = df_clean["LABEL"].value_counts().rename({0: "Tidak Layak", 1: "Layak"})

            fig_pie, ax_p = plt.subplots(figsize=(3, 3))
            fig_pie.patch.set_facecolor("#FFFFFF")
            ax_p.set_facecolor("#FFFFFF")
            colors = ["#E11D48", "#0D9488"]
            wedges, texts, autotexts = ax_p.pie(
                label_counts,
                labels=label_counts.index,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                wedgeprops={"edgecolor": "#fff", "linewidth": 2.5},
                textprops={"color": "#0F172A", "fontsize": 9, "fontweight": "600"},
            )
            for at in autotexts:
                at.set_color("white")
                at.set_fontweight("bold")
                at.set_fontsize(9)
            centre_circle = plt.Circle((0, 0), 0.55, fc="#ffffff")
            ax_p.add_patch(centre_circle)
            ax_p.text(0, 0, str(len(df_clean)), ha="center", va="center",
                      fontsize=18, fontweight="bold", color="#1E3A5F")
            ax_p.set_title("Total Siswa", fontsize=9, color="#64748B", pad=4)
            st.pyplot(fig_pie)

        st.markdown(make_excel_download(df_clean, "hasil_normalisasi_bsm.xlsx", "Unduh Hasil Normalisasi (.xlsx)"), unsafe_allow_html=True)
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
            st.markdown(
                '<p style="font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.5rem;">Tabel Prior Probability</p>',
                unsafe_allow_html=True,
            )
            st.dataframe(nb["prior"], use_container_width=True)

            fig_prior, ax_pr = plt.subplots(figsize=(3.5, 2.8))
            fig_prior.patch.set_facecolor("#FFFFFF")
            ax_pr.set_facecolor("#F7F9FC")
            bar_colors = ["#E11D48", "#0D9488"]
            bars = ax_pr.bar(
                ["Tidak Layak (0)", "Layak (1)"],
                nb["prior"]["Prior Probability"],
                color=bar_colors,
                width=0.5,
                edgecolor="#ffffff",
                linewidth=1.5,
            )
            ax_pr.set_ylim(0, 1)
            ax_pr.set_ylabel("Probabilitas", color="#64748B", fontsize=9)
            ax_pr.tick_params(colors="#64748B", labelsize=8)
            for spine in ax_pr.spines.values():
                spine.set_edgecolor("#D9E2EE")
            for bar in bars:
                ax_pr.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + .015,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom",
                    color="#0F172A", fontsize=9, fontweight="bold",
                )
            ax_pr.axhline(0.5, color="#D97706", linewidth=1, linestyle="--", alpha=.6)
            ax_pr.set_title("Prior per Kelas", fontsize=10, color="#1E3A5F", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_prior)

        with c2:
            st.markdown(
                '<p style="font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.5rem;">Perhitungan Manual Naive Bayes (per data uji)</p>',
                unsafe_allow_html=True,
            )
            st.dataframe(nb["manual"], use_container_width=True, height=400)
            st.markdown(
                make_excel_download(nb["manual"], "perhitungan_manual_nb.xlsx", "Unduh Tabel Manual (.xlsx)"),
                unsafe_allow_html=True,
            )
    else:
        st.info("👈 Klik **Proses Naive Bayes** setelah preprocessing selesai.")

# ─────────────────────────────────────────────────────────
# TAB 4 — Hasil Prediksi
# ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🎯 Hasil Prediksi vs Data Aktual</div>', unsafe_allow_html=True)
    if nb is not None:
        prediksi_df = nb["manual"][["Data Ke", "Prediksi", "Aktual", "Benar?"]]

        benar = (prediksi_df["Benar?"] == "✅").sum()
        salah = (prediksi_df["Benar?"] == "❌").sum()
        st.markdown(
            f"""<div class="metric-row">
            <div class="metric-card teal" data-icon="✅">
                <span class="mc-icon">✅</span>
                <div class="mc-label">Prediksi Benar</div>
                <div class="mc-value mc-teal">{benar}</div>
            </div>
            <div class="metric-card navy" data-icon="❌">
                <span class="mc-icon">❌</span>
                <div class="mc-label">Prediksi Salah</div>
                <div class="mc-value mc-navy">{salah}</div>
            </div>
            <div class="metric-card indigo" data-icon="🧑‍🎓">
                <span class="mc-icon">🧑‍🎓</span>
                <div class="mc-label">Total Data Uji</div>
                <div class="mc-value mc-indigo">{benar+salah}</div>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.dataframe(prediksi_df, use_container_width=True, height=400)
        st.markdown(
            make_excel_download(prediksi_df, "hasil_prediksi.xlsx", "Unduh Hasil Prediksi (.xlsx)"),
            unsafe_allow_html=True,
        )
    else:
        st.info("👈 Klik **Proses Naive Bayes** terlebih dahulu.")

# ─────────────────────────────────────────────────────────
# TAB 5 — Confusion Matrix
# ─────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
    if nb is not None:
        col_img, col_exp = st.columns([1, 1])
        with col_img:
            st.pyplot(nb["fig_cm"])
            st.markdown(
                make_png_download(nb["fig_cm"], "confusion_matrix.png", "Unduh Confusion Matrix (.png)"),
                unsafe_allow_html=True,
            )

        with col_exp:
            st.markdown(
                """
                <div class="info-box">
                <strong>Cara Membaca Confusion Matrix</strong><br><br>
                <strong style="color:#0D9488;">True Negative (TN)</strong> — Model memprediksi <em>Tidak Layak</em>, data asli <em>Tidak Layak</em> — Prediksi benar ✅<br><br>
                <strong style="color:#E11D48;">False Positive (FP)</strong> — Model memprediksi <em>Layak</em>, data asli <em>Tidak Layak</em> — Prediksi salah ❌<br><br>
                <strong style="color:#E11D48;">False Negative (FN)</strong> — Model memprediksi <em>Tidak Layak</em>, data asli <em>Layak</em> — Prediksi salah ❌<br><br>
                <strong style="color:#0D9488;">True Positive (TP)</strong> — Model memprediksi <em>Layak</em>, data asli <em>Layak</em> — Prediksi benar ✅
                </div>
                """,
                unsafe_allow_html=True,
            )

            cm_raw = nb["cm"]
            tn = cm_raw[0, 0] if cm_raw.shape[0] > 1 else 0
            fp = cm_raw[0, 1] if cm_raw.shape[1] > 1 else 0
            fn = cm_raw[1, 0] if cm_raw.shape[0] > 1 else 0
            tp = cm_raw[1, 1] if cm_raw.shape[1] > 1 else 0

            st.markdown(
                f"""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-bottom:1rem">
                    <div style="background:#fff;border:1.5px solid #D9E2EE;border-top:3px solid #0D9488;border-radius:12px;padding:1rem 1.1rem;text-align:center;box-shadow:0 2px 8px rgba(15,23,42,.06)">
                        <div style="font-size:.65rem;color:#94A3B8;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.4rem;font-weight:700;">True Negative</div>
                        <div style="font-family:'DM Serif Display',serif;font-size:2.6rem;color:#0D9488;line-height:1">{tn}</div>
                    </div>
                    <div style="background:#fff;border:1.5px solid #D9E2EE;border-top:3px solid #E11D48;border-radius:12px;padding:1rem 1.1rem;text-align:center;box-shadow:0 2px 8px rgba(15,23,42,.06)">
                        <div style="font-size:.65rem;color:#94A3B8;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.4rem;font-weight:700;">False Positive</div>
                        <div style="font-family:'DM Serif Display',serif;font-size:2.6rem;color:#E11D48;line-height:1">{fp}</div>
                    </div>
                    <div style="background:#fff;border:1.5px solid #D9E2EE;border-top:3px solid #E11D48;border-radius:12px;padding:1rem 1.1rem;text-align:center;box-shadow:0 2px 8px rgba(15,23,42,.06)">
                        <div style="font-size:.65rem;color:#94A3B8;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.4rem;font-weight:700;">False Negative</div>
                        <div style="font-family:'DM Serif Display',serif;font-size:2.6rem;color:#E11D48;line-height:1">{fn}</div>
                    </div>
                    <div style="background:#fff;border:1.5px solid #D9E2EE;border-top:3px solid #0D9488;border-radius:12px;padding:1rem 1.1rem;text-align:center;box-shadow:0 2px 8px rgba(15,23,42,.06)">
                        <div style="font-size:.65rem;color:#94A3B8;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:.4rem;font-weight:700;">True Positive</div>
                        <div style="font-family:'DM Serif Display',serif;font-size:2.6rem;color:#0D9488;line-height:1">{tp}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            acc = nb["accuracy"]
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#0F2444,#1E3A5F);
                            border-radius:14px;padding:1.5rem 1.6rem;text-align:center;
                            border-bottom:4px solid #0D9488;box-shadow:0 8px 28px rgba(15,23,42,.18);">
                    <div style="font-size:.68rem;color:rgba(255,255,255,.45);text-transform:uppercase;
                                letter-spacing:2px;margin-bottom:.45rem;font-weight:700;">Akurasi Model</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:3.8rem;
                                color:#14B8A6;line-height:1">{acc*100:.2f}%</div>
                    <div style="font-size:.82rem;color:rgba(255,255,255,.4);margin-top:.6rem;">
                        {tn+tp} prediksi benar dari {tn+tp+fn+fp} data uji
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("👈 Klik **Proses Naive Bayes** terlebih dahulu.")

# ─────────────────────────────────────────────────────────
# TAB 6 — Classification Report
# ─────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">📈 Laporan Klasifikasi Lengkap</div>', unsafe_allow_html=True)
    if nb is not None:
        report         = nb["report"].copy()
        report_display = report.round(4)
        st.dataframe(report_display, use_container_width=True)

        metrics_df = report.loc[["0", "1"], ["precision", "recall", "f1-score"]]

        fig_bar, ax_b = plt.subplots(figsize=(6, 3.5))
        fig_bar.patch.set_facecolor("#FFFFFF")
        ax_b.set_facecolor("#F7F9FC")

        x = np.arange(len(metrics_df))
        w = 0.25
        bar_palette = ["#1E3A5F", "#0D9488", "#D97706"]
        for idx, (col, color) in enumerate(zip(metrics_df.columns, bar_palette)):
            ax_b.bar(
                x + idx * w,
                metrics_df[col],
                w,
                label=col.capitalize(),
                color=color,
                edgecolor="#ffffff",
                linewidth=1.2,
            )

        ax_b.set_xticks(x + w)
        ax_b.set_xticklabels(["Kelas 0 — Tidak Layak", "Kelas 1 — Layak"], color="#0F172A", fontsize=10)
        ax_b.set_ylim(0, 1.2)
        ax_b.tick_params(colors="#64748B", labelsize=9)
        ax_b.legend(
            facecolor="#FFFFFF",
            edgecolor="#D9E2EE",
            labelcolor="#0F172A",
            fontsize=9,
        )
        for spine in ax_b.spines.values():
            spine.set_edgecolor("#D9E2EE")
        ax_b.axhline(1.0, color="#D9E2EE", linewidth=.8, linestyle="--")
        ax_b.set_title(
            "Perbandingan Metrik Evaluasi per Kelas",
            color="#1E3A5F", fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        st.pyplot(fig_bar)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.markdown(
                make_excel_download(report_display, "classification_report.xlsx", "Unduh Laporan (.xlsx)"),
                unsafe_allow_html=True,
            )
        with col_dl2:
            st.markdown(
                make_png_download(fig_bar, "metrics_chart.png", "Unduh Grafik (.png)"),
                unsafe_allow_html=True,
            )
    else:
        st.info("👈 Klik **Proses Naive Bayes** terlebih dahulu.")

# ─────────────────────────────────────────────────────────
# TAB 7 — Tambah Data Baru
# ─────────────────────────────────────────────────────────
with tab7:
    st.markdown('<div class="section-header">➕ Input Data Siswa Baru — Prediksi Otomatis</div>', unsafe_allow_html=True)

    if nb is None:
        st.warning("⚠️ Harap **Proses Naive Bayes** terlebih dahulu agar prediksi label dapat dijalankan secara otomatis.")
    else:
        st.markdown(
            '<div class="info-box">'
            '🧑‍🎓 Masukkan data atribut siswa di bawah ini. <strong>Label kelayakan akan diprediksi secara otomatis</strong> '
            'oleh model Gaussian Naive Bayes yang telah dilatih dari dataset.'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.form("form_tambah_data", clear_on_submit=True):
            st.markdown(
                '<p style="font-size:.75rem;color:#64748B;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.9rem;">📝 Data Atribut Siswa</p>',
                unsafe_allow_html=True,
            )
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                inp_pendapatan = st.selectbox(
                    "Pendapatan Orang Tua",
                    options=[0, 1, 2],
                    format_func=lambda x: {
                        0: "Rendah — Rp500.000 hingga Rp1.000.000",
                        1: "Menengah — Rp3.000.000 hingga Rp4.000.000",
                        2: "Tinggi — Rp5.000.000 hingga Rp7.000.000",
                    }[x],
                )
                inp_pekerjaan = st.selectbox(
                    "Pekerjaan Orang Tua",
                    options=[0, 1, 2, 3],
                    format_func=lambda x: {
                        0: "Petani",
                        1: "Pegawai Negeri Sipil (PNS)",
                        2: "Polisi / TNI",
                        3: "Wiraswasta",
                    }[x],
                )

            with col_f2:
                inp_tanggungan = st.number_input(
                    "Jumlah Tanggungan Keluarga",
                    min_value=0, max_value=20, value=1, step=1,
                )
                inp_rumah = st.selectbox(
                    "Status Kepemilikan Rumah",
                    options=[0, 1],
                    format_func=lambda x: "Kontrak / Sewa" if x == 0 else "Milik Sendiri",
                )

            submitted = st.form_submit_button("🔍 Prediksi & Tambahkan ke Riwayat", use_container_width=True)

        if submitted:
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
            color      = "#0D9488" if label_pred == 1 else "#E11D48"

            st.markdown(
                f"""
                <div class="pred-result-box {css_class}">
                    <div style="font-size:.68rem;color:#94A3B8;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.5rem;font-weight:700;">
                        Hasil Prediksi Model
                    </div>
                    <div class="pred-label" style="color:{color}">{icon} {label_text}</div>
                    <div class="pred-sub">Diprediksi secara otomatis oleh algoritma Gaussian Naive Bayes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            new_row = pd.DataFrame([{
                "PENDAPATAN ORANG TUA":  int(inp_pendapatan),
                "PEKERJAAN ORANG TUA":   int(inp_pekerjaan),
                "JUMLAH TANGGUNGAN":     int(inp_tanggungan),
                "STATUS RUMAH":          int(inp_rumah),
                "LABEL (PREDIKSI)":      label_pred,
            }])
            st.session_state["data_tambahan"] = pd.concat(
                [st.session_state["data_tambahan"], new_row],
                ignore_index=True,
            )

        # ── Tabel Riwayat ──────────────────────────────────
        st.markdown("---")
        dt = st.session_state["data_tambahan"]

        col_hd, col_hapus = st.columns([3, 1])
        with col_hd:
            st.markdown(
                f'<div class="section-header">Riwayat Prediksi Siswa '
                f'<span style="color:#94A3B8;font-size:.9rem;font-family:DM Sans,sans-serif;font-weight:400;">'
                f'({len(dt)} entri)</span></div>',
                unsafe_allow_html=True,
            )
        with col_hapus:
            if st.button("🗑 Hapus Semua Riwayat", use_container_width=True):
                st.session_state["data_tambahan"] = pd.DataFrame(
                    columns=[
                        "PENDAPATAN ORANG TUA",
                        "PEKERJAAN ORANG TUA",
                        "JUMLAH TANGGUNGAN",
                        "STATUS RUMAH",
                        "LABEL (PREDIKSI)",
                    ]
                )
                st.rerun()

        if len(dt) > 0:
            dt_display = dt.copy()
            dt_display.insert(0, "No", range(1, len(dt_display) + 1))
            dt_display["LABEL (PREDIKSI)"] = dt_display["LABEL (PREDIKSI)"].map(
                {0: "Tidak Layak (0)", 1: "Layak (1)"}
            )
            dt_display["STATUS RUMAH"] = dt_display["STATUS RUMAH"].astype(int).map(
                {0: "Kontrak/Sewa (0)", 1: "Milik Sendiri (1)"}
            )
            dt_display["PEKERJAAN ORANG TUA"] = dt_display["PEKERJAAN ORANG TUA"].astype(int).map(
                {0: "Petani (0)", 1: "PNS (1)", 2: "Polisi (2)", 3: "Wiraswasta (3)"}
            )
            dt_display["PENDAPATAN ORANG TUA"] = dt_display["PENDAPATAN ORANG TUA"].astype(int).map(
                {0: "Rendah (0)", 1: "Menengah (1)", 2: "Tinggi (2)"}
            )

            st.dataframe(dt_display, use_container_width=True, height=320)

            col_dl_a, col_dl_b = st.columns(2)
            with col_dl_a:
                st.markdown(
                    make_excel_download(dt, "data_tambahan_bsm.xlsx", "Unduh Data Tambahan (.xlsx)"),
                    unsafe_allow_html=True,
                )
            with col_dl_b:
                if df_clean is not None:
                    dt_gabung = dt.rename(columns={"LABEL (PREDIKSI)": "LABEL"})
                    gabungan  = pd.concat([df_clean, dt_gabung], ignore_index=True)
                    st.markdown(
                        make_excel_download(
                            gabungan,
                            "data_gabungan_bsm.xlsx",
                            "Unduh Data Gabungan (.xlsx)",
                        ),
                        unsafe_allow_html=True,
                    )

            # Bar chart distribusi label tambahan
            st.markdown("---")
            st.markdown(
                '<p style="font-size:.72rem;color:#64748B;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:.7rem;">Distribusi Kelayakan — Data yang Dimasukkan</p>',
                unsafe_allow_html=True,
            )
            label_dist = dt["LABEL (PREDIKSI)"].value_counts().rename({0: "Tidak Layak", 1: "Layak"})

            fig_dt, ax_dt = plt.subplots(figsize=(4, 2.5))
            fig_dt.patch.set_facecolor("#FFFFFF")
            ax_dt.set_facecolor("#F7F9FC")
            bar_c = ["#E11D48" if "Tidak" in str(k) else "#0D9488" for k in label_dist.index]
            bars_dt = ax_dt.bar(
                label_dist.index,
                label_dist.values,
                color=bar_c,
                edgecolor="#ffffff",
                width=0.5,
                linewidth=1.5,
            )
            ax_dt.tick_params(colors="#64748B")
            ax_dt.set_ylabel("Jumlah Siswa", color="#64748B", fontsize=9)
            for spine in ax_dt.spines.values():
                spine.set_edgecolor("#D9E2EE")
            for bar in bars_dt:
                ax_dt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(int(bar.get_height())),
                    ha="center", va="bottom",
                    color="#0F172A", fontsize=11, fontweight="bold",
                )
            plt.tight_layout()
            st.pyplot(fig_dt)

        else:
            st.markdown(
                '<div style="text-align:center;color:#94A3B8;padding:2.5rem 2rem;'
                'border:1.5px dashed #D9E2EE;border-radius:14px;margin-top:1rem;'
                'background:#F7F9FC;">'
                '<div style="font-size:2.2rem;margin-bottom:.8rem;">🧑‍🎓</div>'
                '<div style="font-weight:600;color:#64748B;margin-bottom:.3rem;">Belum Ada Data yang Dimasukkan</div>'
                '<div style="font-size:.83rem;">Isi form di atas lalu klik <strong style="color:#0D9488;">Prediksi & Tambahkan</strong>.</div>'
                '</div>',
                unsafe_allow_html=True,
            )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#94A3B8;font-size:.8rem;padding:.6rem 0 1rem;letter-spacing:.4px;font-family:'DM Sans',sans-serif;">
        🎓 Sistem Klasifikasi Beasiswa Siswa Miskin &nbsp;·&nbsp;
        Algoritma Gaussian Naive Bayes &nbsp;·&nbsp;
        Dibangun dengan <span style="color:#E11D48;">♥</span> menggunakan Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)