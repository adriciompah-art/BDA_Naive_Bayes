# ==========================================================
# INSTALL LIBRARY
# ==========================================================

# !pip install gradio pandas openpyxl graphviz tabulate scikit-learn -q
# !apt-get install graphviz -y -qq

# ==========================================================
# IMPORT LIBRARY
# ==========================================================

import gradio as gr
import pandas as pd
import math
from collections import Counter
from graphviz import Digraph
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# ==========================================================
# GLOBAL DATASET
# ==========================================================

dataset_global = pd.DataFrame()

# ==========================================================
# FUNGSI ENTROPY
# ==========================================================

def entropy_manual(data):
    total = len(data)
    jumlah = Counter(data)
    hasil = 0
    for kelas in jumlah:
        p = jumlah[kelas] / total
        if p != 0:
            hasil -= p * math.log2(p)
    return hasil

# ==========================================================
# HITUNG GAIN
# ==========================================================

def hitung_gain(df, kolom, target):
    entropy_total = entropy_manual(df[target])
    total_bobot = 0
    detail = ""
    for nilai in df[kolom].unique():
        subset = df[df[kolom] == nilai]
        ent = entropy_manual(subset[target])
        bobot = len(subset) / len(df)
        total_bobot += bobot * ent
        detail += f"""
Nilai : {nilai}
Jumlah Data : {len(subset)}
Entropy : {round(ent,4)}
Bobot x Entropy : {round(bobot * ent,4)}
-------------------------------------
"""
    gain = entropy_total - total_bobot
    return gain, detail

# ==========================================================
# NODE
# ==========================================================

class Node:
    def __init__(self, attribute=None, result=None):
        self.attribute = attribute
        self.result = result
        self.children = {}

# ==========================================================
# BUILD TREE ID3
# ==========================================================

def build_tree(df):
    target = df.columns[-1]
    if len(df[target].unique()) == 1:
        return Node(result=df[target].iloc[0])
    if len(df.columns) == 1:
        return Node(result=df[target].mode()[0])
    fitur = list(df.columns[:-1])
    gains = {}
    for f in fitur:
        g, _ = hitung_gain(df, f, target)
        gains[f] = g
    best = max(gains, key=gains.get)
    node = Node(attribute=best)
    for val in df[best].unique():
        subset = df[df[best] == val].copy()
        subset = subset.drop(columns=[best])
        if len(subset) == 0:
            node.children[val] = Node(result=df[target].mode()[0])
        else:
            node.children[val] = build_tree(subset)
    return node

# ==========================================================
# VISUALISASI TREE
# ==========================================================

def draw_tree(node, dot=None):
    if dot is None:
        dot = Digraph()
        dot.attr(
            rankdir="TB",
            bgcolor="#0F1724",
            fontname="Helvetica",
            pad="0.5",
            nodesep="0.6",
            ranksep="0.8"
        )
        dot.attr(
            "node",
            fontname="Helvetica",
            fontsize="11",
            penwidth="1.5"
        )
        dot.attr(
            "edge",
            fontname="Helvetica",
            fontsize="10",
            color="#4ECDC4",
            fontcolor="#A8D8E8",
            penwidth="1.5"
        )

    uid = str(id(node))

    if node.result is not None:
        dot.node(
            uid,
            f"✓ {node.result}",
            shape="box",
            style="filled,rounded",
            fillcolor="#1B4D3E",
            fontcolor="#52D9A4",
            color="#52D9A4"
        )
    else:
        dot.node(
            uid,
            node.attribute,
            shape="ellipse",
            style="filled",
            fillcolor="#1A2744",
            fontcolor="#7EB8F7",
            color="#3D7EE8"
        )
        for val, child in node.children.items():
            cid = str(id(child))
            draw_tree(child, dot)
            dot.edge(uid, cid, label=f" {val} ")

    return dot

# ==========================================================
# RULES
# ==========================================================

def extract_rules(node, path="", hasil=None):
    if hasil is None:
        hasil = []
    if node.result is not None:
        hasil.append([
            len(hasil) + 1,
            path.strip(" | "),
            node.result
        ])
        return hasil
    for val, child in node.children.items():
        teks = path + f"{node.attribute} = {val} | "
        extract_rules(child, teks, hasil)
    return hasil

# ==========================================================
# UPLOAD FILE
# ==========================================================

def upload_file(file):
    global dataset_global
    if file is None:
        return "⚠️ Silakan upload file terlebih dahulu", None
    if file.name.endswith(".csv"):
        dataset_global = pd.read_csv(file.name)
    else:
        dataset_global = pd.read_excel(file.name)
    if "No" in dataset_global.columns:
        dataset_global = dataset_global.drop(columns=["No"])
    df = dataset_global.copy()
    df.insert(0, "No", range(1, len(df)+1))
    total_data = len(dataset_global)
    total_fitur = len(dataset_global.columns) - 1
    target_col = dataset_global.columns[-1]
    kelas = dataset_global[target_col].unique().tolist()
    status = f"✅ Dataset berhasil dimuat  |  {total_data} data  |  {total_fitur} fitur  |  Kelas: {', '.join(map(str, kelas))}"
    return status, df

# ==========================================================
# HITUNG MANUAL
# ==========================================================

def hitung_manual():
    global dataset_global
    if dataset_global.empty:
        return "⚠️ Dataset belum dimuat. Silakan upload dataset terlebih dahulu."
    df = dataset_global.copy()
    target = df.columns[-1]
    fitur = list(df.columns[:-1])
    teks = ""
    entropy_total = entropy_manual(df[target])

    teks += "╔══════════════════════════════════════════╗\n"
    teks += "║           ENTROPY TOTAL (S)              ║\n"
    teks += "╚══════════════════════════════════════════╝\n"
    teks += f"  Entropy(S) = {round(entropy_total, 6)}\n\n"

    gain_all = {}
    for kolom in fitur:
        gain, detail = hitung_gain(df, kolom, target)
        gain_all[kolom] = gain
        teks += f"┌──────────────────────────────────────────┐\n"
        teks += f"│  ATRIBUT : {kolom:<30} │\n"
        teks += f"└──────────────────────────────────────────┘\n"
        teks += detail
        teks += f"  ➤  Gain({kolom}) = {round(gain, 6)}\n\n"

    root = max(gain_all, key=gain_all.get)
    teks += "╔══════════════════════════════════════════╗\n"
    teks += "║         🏆 ROOT NODE TERBAIK             ║\n"
    teks += "╚══════════════════════════════════════════╝\n"
    teks += f"  Atribut Terpilih : {root}\n"
    teks += f"  Nilai Gain       : {round(gain_all[root], 6)}\n"
    teks += "\n  Ranking Gain Semua Atribut:\n"
    sorted_gains = sorted(gain_all.items(), key=lambda x: x[1], reverse=True)
    for i, (k, v) in enumerate(sorted_gains, 1):
        bar = "█" * int(v * 30)
        marker = " ← ROOT" if k == root else ""
        teks += f"  {i}. {k:<20} {round(v,4):.4f}  {bar}{marker}\n"

    return teks

# ==========================================================
# TAMPIL TREE
# ==========================================================

def tampil_tree():
    global dataset_global
    if dataset_global.empty:
        return None
    tree = build_tree(dataset_global.copy())
    dot = draw_tree(tree)
    dot.render("tree_stunting", format="png", cleanup=True)
    return "tree_stunting.png"

# ==========================================================
# RULE TABLE
# ==========================================================

def tampil_rules():
    global dataset_global
    if dataset_global.empty:
        return pd.DataFrame()
    tree = build_tree(dataset_global.copy())
    rules = extract_rules(tree)
    return pd.DataFrame(rules, columns=["No", "Aturan (IF ... THEN ...)", "Hasil / Label"])

# ==========================================================
# PREDIKSI
# ==========================================================

def prediksi(jenis_kelamin, kategori_umur, bb_lahir, tb_u, bb_u):
    global dataset_global
    if dataset_global.empty:
        return "⚠️ Dataset belum dimuat!"
    df = dataset_global.copy()
    target = df.columns[-1]
    hasil = df[
        (df["JENIS KELAMIN"] == jenis_kelamin) &
        (df["KATEGORI UMUR"] == kategori_umur) &
        (df["BB_Lahir"] == bb_lahir) &
        (df["TB/U"] == tb_u) &
        (df["BB/U"] == bb_u)
    ]
    if len(hasil) == 0:
        return "🔴 Tidak Terdeteksi Stunting"
    prediksi_hasil = hasil[target].mode()[0]
    if "ya" in str(prediksi_hasil).lower() or "stunting" in str(prediksi_hasil).lower():
        return f"🔴 Hasil Prediksi: {prediksi_hasil}"
    return f"🟢 Hasil Prediksi: {prediksi_hasil}"

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def confusion_manual():
    global dataset_global
    if dataset_global.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = dataset_global.copy()
    target = df.columns[-1]
    fitur = list(df.columns[:-1])
    y_true = []
    y_pred = []
    for i in range(len(df)):
        row = df.iloc[i]
        aktual = row[target]
        subset = df.copy()
        for f in fitur:
            subset = subset[subset[f].astype(str) == str(row[f])]
        if len(subset) == 0:
            hasil = "Tidak"
        else:
            hasil = subset[target].mode()[0]
        y_true.append(aktual)
        y_pred.append(hasil)

    labels = list(df[target].unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=["Aktual: " + x for x in labels],
        columns=["Prediksi: " + x for x in labels]
    )
    acc = accuracy_score(y_true, y_pred)
    laporan_df = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True)
    ).transpose().round(4)
    laporan_df.loc["accuracy", "precision"] = str(round(acc * 100, 2)) + " %"
    return cm_df, laporan_df

# ==========================================================
# SIMPAN DATASET
# ==========================================================

def simpan():
    global dataset_global
    dataset_global.to_csv("dataset_stunting_baru.csv", index=False)
    return "dataset_stunting_baru.csv"

# ==========================================================
# CUSTOM CSS - TEMA GELAP ELEGAN
# ==========================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --c-bg:        #F6F8FB;
    --c-surface:   #FFFFFF;
    --c-surface2:  #F0F4FA;
    --c-border:    #D8E2F0;
    --c-border2:   #B8CCE8;
    --c-primary:   #1B6EF3;
    --c-primary-h: #1557D0;
    --c-primary-s: #EBF2FF;
    --c-teal:      #00B894;
    --c-teal-s:    #E0FAF4;
    --c-purple:    #7C3AED;
    --c-purple-s:  #F0EBFF;
    --c-amber:     #D97706;
    --c-amber-s:   #FFF8E6;
    --c-red:       #DC2626;
    --c-red-s:     #FFF0F0;
    --c-text:      #0F172A;
    --c-text2:     #475569;
    --c-text3:     #94A3B8;
    --r:           10px;
    --r-lg:        16px;
    --font:        'Plus Jakarta Sans', sans-serif;
    --mono:        'JetBrains Mono', monospace;
    --shadow-sm:   0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.05);
    --shadow-md:   0 4px 16px rgba(15,23,42,0.10), 0 2px 6px rgba(15,23,42,0.06);
    --shadow-btn:  0 2px 8px rgba(27,110,243,0.30);
}

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: var(--font) !important;
    background: var(--c-bg) !important;
    color: var(--c-text) !important;
    min-height: 100vh;
}

/* ── HERO ── */
.hero-header {
    background: linear-gradient(135deg, #1B6EF3 0%, #0EA5E9 55%, #00B894 100%);
    border-radius: var(--r-lg);
    padding: 2.8rem 2.5rem 2.4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(27,110,243,0.28);
}

.hero-header::after {
    content: '';
    position: absolute;
    inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    pointer-events: none;
}

.hero-header h1 {
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    margin: 0 0 0.5rem !important;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.hero-header p {
    color: rgba(255,255,255,0.88) !important;
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.35);
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    backdrop-filter: blur(4px);
}

/* ── TABS ── */
.tabs > .tab-nav {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-lg) !important;
    padding: 5px !important;
    gap: 3px !important;
    box-shadow: var(--shadow-sm) !important;
    margin-bottom: 1.2rem !important;
}

.tabs > .tab-nav > button {
    font-family: var(--font) !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    color: var(--c-text2) !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    transition: all 0.18s ease !important;
}

.tabs > .tab-nav > button:hover {
    color: var(--c-primary) !important;
    background: var(--c-primary-s) !important;
}

.tabs > .tab-nav > button.selected {
    color: #ffffff !important;
    background: var(--c-primary) !important;
    box-shadow: var(--shadow-btn) !important;
}

/* ── TAB CONTENT PANEL ── */
.tabitem {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-lg) !important;
    padding: 1.8rem !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── BUTTONS ── */
button, .gr-button {
    font-family: var(--font) !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    border-radius: var(--r) !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
}

.btn-teal, button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, #1B6EF3, #0EA5E9) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: var(--shadow-btn) !important;
    letter-spacing: 0.2px !important;
}

.btn-teal:hover, button.primary:hover, button[variant="primary"]:hover {
    background: linear-gradient(135deg, #1557D0, #0284C7) !important;
    box-shadow: 0 4px 16px rgba(27,110,243,0.40) !important;
    transform: translateY(-1px) !important;
}

.btn-teal:active, button.primary:active {
    transform: translateY(0) !important;
    box-shadow: var(--shadow-btn) !important;
}

/* ── TEXTBOX ── */
textarea, input[type="text"] {
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    background: var(--c-surface2) !important;
    color: var(--c-text) !important;
    border: 1.5px solid var(--c-border) !important;
    border-radius: var(--r) !important;
    padding: 12px 14px !important;
    line-height: 1.75 !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--c-primary) !important;
    box-shadow: 0 0 0 3px rgba(27,110,243,0.12) !important;
    outline: none !important;
    background: #ffffff !important;
}

/* ── STATUS BOX ── */
.status-box textarea {
    background: var(--c-teal-s) !important;
    border-color: #00B89455 !important;
    color: #007A62 !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

/* ── RESULT BOX ── */
.result-box textarea {
    font-family: var(--font) !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    background: var(--c-primary-s) !important;
    border-color: #1B6EF344 !important;
    color: var(--c-primary) !important;
    padding: 20px !important;
}

/* ── LABELS ── */
label, .gr-label, span.svelte-1gfkfd6 {
    font-family: var(--font) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: var(--c-text2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    margin-bottom: 5px !important;
}

/* ── DROPDOWN ── */
select {
    font-family: var(--font) !important;
    background: var(--c-surface2) !important;
    color: var(--c-text) !important;
    border: 1.5px solid var(--c-border) !important;
    border-radius: var(--r) !important;
    padding: 10px 14px !important;
    transition: border-color 0.15s !important;
    font-size: 0.875rem !important;
}

select:focus {
    border-color: var(--c-primary) !important;
    box-shadow: 0 0 0 3px rgba(27,110,243,0.12) !important;
    outline: none !important;
    background: #fff !important;
}

/* ── DATAFRAME / TABLE ── */
.dataframe, table {
    font-family: var(--font) !important;
    font-size: 0.83rem !important;
    background: var(--c-surface) !important;
    border-radius: var(--r) !important;
    border: 1px solid var(--c-border) !important;
    overflow: hidden !important;
    width: 100% !important;
    box-shadow: var(--shadow-sm) !important;
}

.dataframe th, thead th {
    background: var(--c-primary) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.74rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    padding: 11px 16px !important;
    border-bottom: none !important;
}

.dataframe td, tbody td {
    color: var(--c-text) !important;
    padding: 10px 16px !important;
    border-bottom: 1px solid var(--c-border) !important;
    line-height: 1.5 !important;
}

.dataframe tr:last-child td { border-bottom: none !important; }

.dataframe tr:hover td {
    background: var(--c-primary-s) !important;
}

.dataframe tr:nth-child(even) td {
    background: var(--c-surface2);
}

.dataframe tr:nth-child(even):hover td {
    background: var(--c-primary-s) !important;
}

/* ── FILE UPLOAD ── */
.gr-file, input[type="file"] {
    background: var(--c-surface2) !important;
    border: 2px dashed var(--c-border2) !important;
    border-radius: var(--r-lg) !important;
    transition: border-color 0.2s, background 0.2s !important;
}

.gr-file:hover {
    border-color: var(--c-primary) !important;
    background: var(--c-primary-s) !important;
}

/* ── MARKDOWN ── */
.gr-markdown h3 {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: var(--c-text) !important;
    margin: 0 0 0.4rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid var(--c-primary-s) !important;
}

.gr-markdown p {
    color: var(--c-text2) !important;
    font-size: 0.875rem !important;
    line-height: 1.7 !important;
    margin: 0.3rem 0 !important;
}

.gr-markdown strong { color: var(--c-text) !important; }

.gr-markdown em { color: var(--c-primary) !important; font-style: normal !important; font-weight: 500 !important; }

.gr-markdown code {
    font-family: var(--mono) !important;
    background: var(--c-primary-s) !important;
    color: var(--c-primary) !important;
    padding: 2px 7px !important;
    border-radius: 5px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

.gr-markdown blockquote {
    border-left: 3px solid var(--c-primary) !important;
    padding-left: 12px !important;
    margin: 8px 0 !important;
    color: var(--c-text2) !important;
}

/* ── IMAGE ── */
img {
    border-radius: var(--r) !important;
    border: 1px solid var(--c-border) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--c-surface2); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: var(--c-border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--c-primary); }

/* ── FADE IN ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.gradio-container > * { animation: fadeUp 0.35s ease both; }
"""

# ==========================================================
# INTERFACE GRADIO - REDESIGNED
# ==========================================================

with gr.Blocks(title="Decision Tree ID3 — Prediksi Stunting") as demo:

    # ---- HERO HEADER ----
    gr.HTML("""
    <div class="hero-header">
        <div class="hero-badge">🌿 Machine Learning &nbsp;·&nbsp; Decision Tree</div>
        <h1>🌳 Algoritma Decision Tree ID3</h1>
        <p>Sistem Prediksi Risiko Stunting berbasis Information Gain &amp; Entropy</p>
    </div>
    """)

    with gr.Tabs(elem_classes="main-tabs"):

        # ============================================
        # TAB 1 — UPLOAD DATASET
        # ============================================
        with gr.Tab("📂  Dataset"):
            gr.Markdown("""
            ### Upload Dataset
            Mendukung format **CSV** dan **Excel (.xlsx)**. Pastikan kolom terakhir adalah **label/target**.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    file = gr.File(
                        label="📂  Pilih File CSV / XLSX",
                        file_types=[".csv", ".xlsx"]
                    )
                    btn_upload = gr.Button(
                        "⬆️  Muat Dataset",
                        variant="primary",
                        size="lg",
                        elem_classes="btn-teal"
                    )
                with gr.Column(scale=3):
                    status = gr.Textbox(
                        label="📋  Status",
                        lines=2,
                        interactive=False,
                        elem_classes="status-box"
                    )
                    tabel = gr.Dataframe(
                        label="📊  Preview Dataset",
                        wrap=True,
                        max_height=400
                    )

            btn_upload.click(
                upload_file,
                inputs=file,
                outputs=[status, tabel]
            )

        # ============================================
        # TAB 2 — PERHITUNGAN ID3
        # ============================================
        with gr.Tab("🧮  Perhitungan"):
            gr.Markdown("""
            ### Perhitungan Entropy & Information Gain
            Menampilkan detail kalkulasi **entropy** setiap atribut dan pemilihan **root node** terbaik.
            """)
            btn_hitung = gr.Button(
                "🧮  Hitung Entropy & Information Gain",
                variant="primary",
                size="lg",
                elem_classes="btn-teal"
            )
            hasil = gr.Textbox(
                label="📐  Hasil Perhitungan Detail",
                lines=40,
                interactive=False
            )
            btn_hitung.click(hitung_manual, outputs=hasil)

        # ============================================
        # TAB 3 — POHON KEPUTUSAN
        # ============================================
        with gr.Tab("🌳  Pohon Keputusan"):
            gr.Markdown("""
            ### Visualisasi Pohon Keputusan
            Struktur tree yang dibangun menggunakan algoritma **ID3** berbasis nilai *Information Gain* tertinggi.
            """)
            btn_tree = gr.Button(
                "🌳  Generate Pohon Keputusan",
                variant="primary",
                size="lg",
                elem_classes="btn-teal"
            )
            img = gr.Image(
                label="🖼️  Decision Tree"
            )
            btn_tree.click(tampil_tree, outputs=img)

        # ============================================
        # TAB 4 — RULES ID3
        # ============================================
        with gr.Tab("📜  Rules"):
            gr.Markdown("""
            ### Aturan Keputusan (IF ... THEN ...)
            Setiap baris merepresentasikan **satu jalur** dari root node ke leaf node pada decision tree.
            """)
            btn_rule = gr.Button(
                "📜  Ekstrak Rules dari Tree",
                variant="primary",
                size="lg",
                elem_classes="btn-teal"
            )
            tabel_rule = gr.Dataframe(
                label="📋  Daftar Aturan Keputusan",
                wrap=True,
                max_height=500
            )
            btn_rule.click(tampil_rules, outputs=tabel_rule)

        # ============================================
        # TAB 5 — PREDIKSI
        # ============================================
        with gr.Tab("🔍  Prediksi"):
            gr.Markdown("""
            ### Prediksi Risiko Stunting
            Masukkan data pasien untuk mengetahui hasil prediksi berdasarkan model Decision Tree ID3.
            """)
            with gr.Row():
                with gr.Column():
                    gr.HTML('<p style="color:#475569;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;margin:0 0 8px;padding-bottom:8px;border-bottom:2px solid #EBF2FF;">🧒 Data Pasien</p>')
                    in1 = gr.Dropdown(
                        choices=["Laki-laki", "Perempuan"],
                        label="Jenis Kelamin",
                        info="Pilih jenis kelamin pasien"
                    )
                    in2 = gr.Dropdown(
                        choices=["Bayi", "Balita"],
                        label="Kategori Umur",
                        info="Bayi: 0–12 bulan | Balita: 13–60 bulan"
                    )
                    in3 = gr.Dropdown(
                        choices=["Normal", "Tidak Normal"],
                        label="Berat Badan Lahir (BB_Lahir)",
                        info="Normal: ≥ 2500 gram"
                    )

                with gr.Column():
                    gr.HTML('<p style="color:#475569;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;margin:0 0 8px;padding-bottom:8px;border-bottom:2px solid #EBF2FF;">📏 Indikator Antropometri</p>')
                    in4 = gr.Dropdown(
                        choices=["Pendek", "Sangat Pendek"],
                        label="Tinggi Badan / Umur (TB/U)",
                        info="Indikator status stunting"
                    )
                    in5 = gr.Dropdown(
                        choices=["Normal", "Kurang", "Sangat Kurang"],
                        label="Berat Badan / Umur (BB/U)",
                        info="Indikator gizi balita"
                    )
                    gr.HTML('<br/>')
                    btn_prediksi = gr.Button(
                        "🔍  Jalankan Prediksi",
                        variant="primary",
                        size="lg",
                        elem_classes="btn-teal"
                    )

            gr.HTML('<hr style="border:none;border-top:1px solid #D8E2F0;margin:1.5rem 0"/>')

            hasil_prediksi = gr.Textbox(
                label="🎯  Hasil Prediksi Risiko Stunting",
                lines=3,
                interactive=False,
                elem_classes="result-box"
            )

            btn_prediksi.click(
                prediksi,
                inputs=[in1, in2, in3, in4, in5],
                outputs=hasil_prediksi
            )

        # ============================================
        # TAB 6 — EVALUASI / CONFUSION MATRIX
        # ============================================
        with gr.Tab("📊  Evaluasi"):
            gr.Markdown("""
            ### Evaluasi Model (Confusion Matrix & Classification Report)
            Mengukur performa model menggunakan **akurasi**, **precision**, **recall**, dan **F1-score**.
            """)
            btn_cm = gr.Button(
                "📊  Hitung Evaluasi Model",
                variant="primary",
                size="lg",
                elem_classes="btn-teal"
            )
            with gr.Row():
                with gr.Column():
                    tabel_cm = gr.Dataframe(
                        label="🗃️  Confusion Matrix",
                        wrap=True
                    )
                with gr.Column():
                    laporan_cm = gr.Dataframe(
                        label="📈  Classification Report",
                        wrap=True
                    )
            btn_cm.click(
                confusion_manual,
                outputs=[tabel_cm, laporan_cm]
            )

        # ============================================
        # TAB 7 — SIMPAN DATASET
        # ============================================
        with gr.Tab("💾  Simpan"):
            gr.Markdown("""
            ### Ekspor Dataset
            Simpan dataset yang sedang dimuat ke dalam format **CSV** untuk digunakan kembali.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    btn_save = gr.Button(
                        "💾  Ekspor ke CSV",
                        variant="primary",
                        size="lg",
                        elem_classes="btn-teal"
                    )
                    gr.Markdown("""
                    > **Catatan:** File akan disimpan sebagai `dataset_stunting_baru.csv`
                    """)
                with gr.Column(scale=2):
                    file_download = gr.File(
                        label="📥  Download Dataset",
                    )
            btn_save.click(simpan, outputs=file_download)

    # ---- FOOTER ----
    gr.HTML("""
    <div style="
        margin-top: 2rem;
        padding: 1rem 1.5rem;
        background: #ffffff;
        border: 1px solid #D8E2F0;
        border-radius: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.06);
    ">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.3rem;">🌳</span>
            <span style="color:#0F172A;font-weight:700;font-size:0.9rem;">Decision Tree ID3</span>
            <span style="color:#1B6EF3;font-size:0.72rem;background:#EBF2FF;padding:3px 10px;border-radius:20px;border:1px solid #B8D4FF;font-weight:600;">Stunting Predictor</span>
        </div>
        <div style="color:#94A3B8;font-size:0.75rem;font-family:'Plus Jakarta Sans',sans-serif;">
            Built with Gradio &nbsp;·&nbsp; Scikit-learn &nbsp;·&nbsp; Graphviz
        </div>
    </div>
    """)

# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":
    demo.launch(
        share=True,
        css=custom_css,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "sans-serif"]
        )
    )