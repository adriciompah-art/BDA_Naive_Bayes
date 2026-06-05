# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.upload import upload_file
from modules.preprocessing import preprocessing_data
from modules.training import training_model
from modules.evaluasi import confusion_manual, tampil_akurasi
from modules.prediksi import prediksi_data
from modules.export_laporan import export_laporan

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Naive Bayes BSM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# INISIALISASI SESSION STATE
# ==========================================================
if "status_upload" not in st.session_state:
    st.session_state.status_upload = "Belum ada file diupload"
if "preview_df" not in st.session_state:
    st.session_state.preview_df = None
if "preprocessing_log" not in st.session_state:
    st.session_state.preprocessing_log = ""
if "preprocessed_df" not in st.session_state:
    st.session_state.preprocessed_df = None
if "training_summary" not in st.session_state:
    st.session_state.training_summary = ""
if "training_params_df" not in st.session_state:
    st.session_state.training_params_df = None
if "confusion_df" not in st.session_state:
    st.session_state.confusion_df = None
if "class_report_df" not in st.session_state:
    st.session_state.class_report_df = None
if "accuracy" not in st.session_state:
    st.session_state.accuracy = ""

# ==========================================================
# SIDEBAR - NAVIGASI
# ==========================================================
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio(
    "Pilih Tahapan",
    ["📂 Upload Data", "⚙️ Preprocessing", "🧮 Training Model",
     "📊 Evaluasi Model", "🔍 Prediksi Individual", "📥 Laporan Akhir"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Aplikasi untuk prediksi kelayakan Bantuan Siswa Miskin (BSM) "
    "menggunakan algoritma **Naive Bayes**."
)

# ==========================================================
# BAGIAN UPLOAD DATA
# ==========================================================
if menu == "📂 Upload Data":
    st.header("📂 Upload Data Excel")
    uploaded_file = st.file_uploader("Pilih file XLSX", type=["xlsx"])
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Upload Sekarang", use_container_width=True):
            if uploaded_file is not None:
                # Fungsi upload_file membutuhkan path file, kita simpan dulu
                with open("temp_upload.xlsx", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                status, df = upload_file("temp_upload.xlsx")
                st.session_state.status_upload = status
                st.session_state.preview_df = df
                st.success("Upload berhasil!")
            else:
                st.warning("Pilih file terlebih dahulu.")
    with col2:
        st.text_area("📋 Status", value=st.session_state.status_upload, height=100, disabled=True)

    if st.session_state.preview_df is not None:
        st.subheader("📊 Pratinjau Data")
        st.dataframe(st.session_state.preview_df, use_container_width=True)

# ==========================================================
# BAGIAN PREPROCESSING
# ==========================================================
elif menu == "⚙️ Preprocessing":
    st.header("⚙️ Preprocessing Data")
    if st.button("🔄 Jalankan Preprocessing", use_container_width=True):
        with st.spinner("Memproses data..."):
            log, df_pre = preprocessing_data()
            st.session_state.preprocessing_log = log
            st.session_state.preprocessed_df = df_pre
            st.success("Preprocessing selesai!")

    if st.session_state.preprocessing_log:
        st.text_area("📝 Log Preprocessing", value=st.session_state.preprocessing_log, height=300, disabled=True)

    if st.session_state.preprocessed_df is not None:
        st.subheader("📋 Data Hasil Preprocessing")
        st.dataframe(st.session_state.preprocessed_df, use_container_width=True)

        # GRAFIK: distribusi fitur numerik
        st.subheader("📈 Distribusi Fitur Numerik")
        numeric_cols = st.session_state.preprocessed_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5*len(numeric_cols), 4))
            if len(numeric_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, numeric_cols):
                sns.histplot(st.session_state.preprocessed_df[col], kde=True, ax=ax, color='#6366f1')
                ax.set_title(f"Distribusi {col}")
                ax.set_xlabel(col)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Tidak ada kolom numerik untuk ditampilkan.")

# ==========================================================
# BAGIAN TRAINING
# ==========================================================
elif menu == "🧮 Training Model":
    st.header("🧮 Training Model Naive Bayes")
    if st.button("⚡ Latih Model", use_container_width=True):
        with st.spinner("Melatih model..."):
            summary, params_df = training_model()
            st.session_state.training_summary = summary
            st.session_state.training_params_df = params_df
            st.success("Training selesai!")

    if st.session_state.training_summary:
        st.text_area("📊 Ringkasan Training", value=st.session_state.training_summary, height=200, disabled=True)

    if st.session_state.training_params_df is not None:
        st.subheader("📈 Parameter Model")
        st.dataframe(st.session_state.training_params_df, use_container_width=True)

# ==========================================================
# BAGIAN EVALUASI + GRAFIK
# ==========================================================
elif menu == "📊 Evaluasi Model":
    st.header("📊 Evaluasi Performa Model")
    if st.button("🎯 Evaluasi Performa", use_container_width=True):
        with st.spinner("Menghitung metrik evaluasi..."):
            cm_df, report_df = confusion_manual()
            acc = tampil_akurasi()
            st.session_state.confusion_df = cm_df
            st.session_state.class_report_df = report_df
            st.session_state.accuracy = acc
            st.success("Evaluasi selesai!")

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.confusion_df is not None:
            st.subheader("📉 Confusion Matrix (Tabel)")
            st.dataframe(st.session_state.confusion_df, use_container_width=True)

            # GRAFIK: Heatmap Confusion Matrix
            st.subheader("🔥 Heatmap Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(st.session_state.confusion_df.astype(int), annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

    with col2:
        if st.session_state.class_report_df is not None:
            st.subheader("📋 Classification Report")
            st.dataframe(st.session_state.class_report_df, use_container_width=True)

    if st.session_state.accuracy:
        st.metric(label="✅ Akurasi Akhir", value=f"{st.session_state.accuracy}")

    # GRAFIK TAMBAHAN: Pie chart target jika ada di data preprocessing
    if st.session_state.preprocessed_df is not None:
        st.subheader("🥧 Distribusi Kelas Target")
        # Asumsikan kolom target bernama 'Kelayakan' atau 'Target' – sesuaikan dengan data Anda
        target_col = None
        possible_targets = ['Kelayakan', 'Target', 'label', 'Class']
        for col in possible_targets:
            if col in st.session_state.preprocessed_df.columns:
                target_col = col
                break
        if target_col:
            target_counts = st.session_state.preprocessed_df[target_col].value_counts()
            fig, ax = plt.subplots()
            ax.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90, colors=['#10b981','#ef4444'])
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("Tidak dapat menemukan kolom target untuk pie chart.")

# ==========================================================
# BAGIAN PREDIKSI INDIVIDUAL
# ==========================================================
elif menu == "🔍 Prediksi Individual":
    st.header("🔍 Prediksi Kelayakan Individual")
    with st.form("pred_form"):
        col1, col2 = st.columns(2)
        with col1:
            pendapatan = st.text_input("💰 PENDAPATAN ORANG TUA", placeholder="Contoh: 1500000")
            pekerjaan = st.text_input("💼 PEKERJAAN ORANG TUA", placeholder="Contoh: Petani, Buruh, Wiraswasta")
        with col2:
            tanggungan = st.text_input("👨‍👩‍👧 JUMLAH TANGGUNGAN", placeholder="Contoh: 3")
            status_rumah = st.text_input("🏠 STATUS RUMAH", placeholder="Contoh: Milik Sendiri, Sewa, Kontrak")

        submitted = st.form_submit_button("🔮 Prediksi", use_container_width=True)

    if submitted:
        if pendapatan and pekerjaan and tanggungan and status_rumah:
            with st.spinner("Menghitung prediksi..."):
                hasil = prediksi_data(pendapatan, pekerjaan, tanggungan, status_rumah)
            st.success(f"🎯 Hasil Prediksi: **{hasil}**")
        else:
            st.warning("Harap isi semua field!")

# ==========================================================
# BAGIAN LAPORAN AKHIR
# ==========================================================
elif menu == "📥 Laporan Akhir":
    st.header("📥 Download Laporan Excel")
    if st.button("📄 Generate Laporan Excel", use_container_width=True):
        with st.spinner("Membuat file laporan..."):
            excel_file = export_laporan()
        if excel_file:
            with open(excel_file, "rb") as f:
                st.download_button(
                    label="💾 Download Laporan",
                    data=f,
                    file_name="laporan_bsm.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.error("Gagal membuat laporan. Pastikan semua tahapan sudah dijalankan.")

# ==========================================================
# FOOTER (opsional)
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.caption("Aplikasi Naive Bayes BSM | Dibangun dengan Streamlit")