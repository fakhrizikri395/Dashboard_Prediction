# Dashboard_Final.py
# by MFZ + bre 😄

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO, BytesIO

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="Final Project Dashboard", layout="wide")
st.markdown("<h1 style='margin-top:-10px'>🏁 Final Project Dashboard</h1>", unsafe_allow_html=True)

# =========================
# --- GANTI INI SESUAI NAMA FILE CSV HASIL PREDIKSI ---
# Contoh umum:
# - Klasifikasi: kolom minimal ["Index","Prediction","Probability"(opsional)]
# - Regresi: kolom minimal ["Index","Prediction"]
CSV_PATH = "hasil_prediksi_TitanicDataset.csv"
# ======================================================

# =========================
# UTIL & LOAD DATA (aman dataset besar)
# =========================
@st.cache_data(show_spinner="Memuat CSV…")
def load_csv_safely(path: str) -> pd.DataFrame:
    # Pakai low_memory + engine c buat efisien
    df = pd.read_csv(path, low_memory=True, engine="c")
    return df

def detect_task_type(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    if "probability" in cols:
        return "classifier"
    # Kalau kolom Prediction ada & jumlah nilai unik relatif kecil → klasifikasi
    if "prediction" in cols:
        pred = df[df.columns[cols.index("prediction")]]
        # Kalau numeric & unik <= 10 dan integer-ish
        if pd.api.types.is_numeric_dtype(pred):
            unique_vals = pd.unique(pred.dropna())
            if len(unique_vals) <= 10 and np.all(np.isclose(unique_vals, np.round(unique_vals))):
                return "classifier"
        # Kalau tipe object dengan kategori terbatas
        if pred.dtype == "object" and pred.nunique(dropna=True) <= 20:
            return "classifier"
    return "regressor"

def safe_cols(df: pd.DataFrame, include=None):
    return [c for c in df.columns if (include is None or c in include)]

def try_to_image(fig) -> bytes | None:
    # Export PNG tanpa memaksa kaleido
    try:
        import plotly.io as pio  # noqa
        return fig.to_image(format="png")
    except Exception:
        return None

# =========================
# LOAD
# =========================
try:
    df_raw = load_csv_safely(CSV_PATH)
except Exception as e:
    st.error(f"❌ Gagal memuat CSV: {e}")
    st.stop()

df = df_raw.copy()

# Info ringkas dataset
with st.expander("ℹ️ Ringkasan Dataset"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Baris", f"{len(df):,}")
    c2.metric("Kolom", f"{df.shape[1]:,}")
    mem_mb = df.memory_usage(index=True, deep=True).sum() / (1024**2)
    c3.metric("Perkiraan Memori", f"{mem_mb:,.2f} MB")
    st.dataframe(df.head(50), use_container_width=True, height=260)

# Deteksi tipe tugas
task_type = detect_task_type(df)
st.info(f"🔎 Deteksi otomatis: **{task_type.title()}**")

# =========================
# SIDEBAR: FILTER GENERIK
# =========================
st.sidebar.header("🔍 Filter Data")

# Filter kategorikal
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
for col in cat_cols:
    uniq = df[col].dropna().unique().tolist()
    if len(uniq) > 0:
        chosen = st.sidebar.multiselect(f"{col}", uniq, default=uniq)
        df = df[df[col].isin(chosen)]

# Filter numerik (slider NaN-safe)
num_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
for col in num_cols:
    series = df[col]
    if series.dropna().empty:
        continue
    vmin, vmax = float(series.min(skipna=True)), float(series.max(skipna=True))
    r = st.sidebar.slider(f"{col}", min_value=vmin, max_value=vmax, value=(vmin, vmax))
    df = df[series.between(r[0], r[1]) | series.isna()]

# Tombol download data terfilter
csv_buf = StringIO()
df.to_csv(csv_buf, index=False)
st.sidebar.download_button(
    "💾 Download Data Terfilter (CSV)",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="filtered_results.csv",
    mime="text/csv",
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Filter aman terhadap **NaN** dan **dataset besar**.")

# =========================
# TABS
# =========================
t1, t2, t3, t4 = st.tabs(["📋 Data", "📈 Visualisasi", "📊 Analitik", "ℹ️ Info"])

# =========================
# TAB 1 – DATA
# =========================
with t1:
    st.subheader("📋 Data Terfilter")
    st.dataframe(df, use_container_width=True, height=420)

# =========================
# TAB 2 – VISUALISASI (AUTO-SMART)
# =========================
with t2:
    st.subheader("📈 Visualisasi Otomatis")

    # Pilihan kolom X/Y disesuaikan
    all_cols = df.columns.tolist()
    default_x = "Index" if "Index" in all_cols else all_cols[0]
    x_col = st.selectbox("Kolom X", options=all_cols, index=all_cols.index(default_x))

    # Untuk classifier: Y bisa "Prediction" atau numerik lain
    # Untuk regresi: Y default "Prediction" kalau ada
    y_candidates = all_cols.copy()
    if "Prediction" in all_cols:
        y_default = "Prediction"
    else:
        # fallback: cari numerik lain
        y_nums = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        y_default = y_nums[0] if y_nums else all_cols[0]
    y_col = st.selectbox("Kolom Y", options=y_candidates, index=y_candidates.index(y_default))

    color_col = st.selectbox("Kolom Warna (opsional)", options=[None] + all_cols)

    # Downsampling opsional agar grafis ringan untuk dataset sangat besar
    max_plot_points = st.number_input("Batas titik untuk visualisasi (downsample)", min_value=1000, max_value=200_000, value=20_000, step=1000)
    plot_df = df
    if len(df) > max_plot_points:
        plot_df = df.sample(max_plot_points, random_state=42)

    # Auto pilih tipe chart yang cocok
    def is_numeric(col):
        return pd.api.types.is_numeric_dtype(plot_df[col])

    if task_type == "classifier":
        # Default: kalau Y kategorikal → bar, kalau numerik → scatter
        if not is_numeric(y_col):
            fig = px.histogram(plot_df, x=y_col, color=color_col, barmode="group", title=f"Distribusi {y_col}")
        else:
            fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
    else:
        # Regressor: tampilkan scatter + trend sederhana (tanpa fit)
        if is_numeric(x_col) and is_numeric(y_col):
            fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
        else:
            # fallback: histogram untuk kolom Y
            fig = px.histogram(plot_df, x=y_col, color=color_col, title=f"Distribusi {y_col}")

    st.plotly_chart(fig, use_container_width=True)

    # Download gambar
    img = try_to_image(fig)
    if img:
        st.download_button("📷 Download Grafik (PNG)", data=img, file_name="visualisasi.png", mime="image/png")
    else:
        st.caption("💡 Ingin export gambar? Install sekali saja: `pip install -U kaleido`.")

# =========================
# TAB 3 – ANALITIK (KHUSUS SESUAI TIPE)
# =========================
with t3:
    if task_type == "classifier":
        st.subheader("📊 Analitik Klasifikasi")
        cols_lower = [c.lower() for c in df.columns]
        # Distribusi label prediksi
        if "prediction" in cols_lower:
            pred_col = df.columns[cols_lower.index("prediction")]
            vc = df[pred_col].value_counts(dropna=False)
            st.write("Distribusi Prediction")
            st.bar_chart(vc)

        # Histogram Probability (kalau ada)
        if "probability" in cols_lower:
            prob_col = df.columns[cols_lower.index("probability")]
            st.write("Sebaran Probability")
            figp = px.histogram(df, x=prob_col, nbins=30, title="Histogram Probability")
            st.plotly_chart(figp, use_container_width=True)

            # Thresholding opsional untuk bikin label 0/1 dari probability
            with st.expander("⚙️ Buat Label dari Probability (threshold)"):
                th = st.slider("Threshold positif", 0.0, 1.0, 0.5, 0.01)
                pos_label = st.text_input("Nama label positif", "1")
                neg_label = st.text_input("Nama label negatif", "0")
                label = np.where(df[prob_col] >= th, pos_label, neg_label)
                tmp = df.copy()
                tmp["Label_by_threshold"] = label
                st.dataframe(tmp.head(30), use_container_width=True)
                out = StringIO()
                tmp.to_csv(out, index=False)
                st.download_button("💾 Download dengan Label (CSV)", data=out.getvalue().encode("utf-8"), file_name="with_threshold_label.csv", mime="text/csv")

        st.caption("Catatan: tanpa ground truth, metrik seperti akurasi/ROC tidak ditampilkan.")

    else:
        st.subheader("📊 Analitik Regresi")
        cols_lower = [c.lower() for c in df.columns]
        if "prediction" in cols_lower:
            pred_col = df.columns[cols_lower.index("prediction")]
            c1, c2, c3, c4 = st.columns(4)
            s = df[pred_col].describe()
            c1.metric("Mean", f"{s['mean']:.4f}")
            c2.metric("Std", f"{s['std']:.4f}")
            c3.metric("Min", f"{s['min']:.4f}")
            c4.metric("Max", f"{s['max']:.4f}")

            figh = px.histogram(df, x=pred_col, nbins=40, title=f"Distribusi {pred_col}")
            st.plotly_chart(figh, use_container_width=True)

            # Top/Bottom values
            with st.expander("🔎 Observasi Ekstrem (Top/Bottom)"):
                n = st.slider("Tampilkan", 5, 50, 10)
                st.write("Terbesar")
                st.dataframe(df.nlargest(n, pred_col), use_container_width=True, height=260)
                st.write("Terkecil")
                st.dataframe(df.nsmallest(n, pred_col), use_container_width=True, height=260)
        else:
            st.warning("Kolom 'Prediction' tidak ditemukan. Tampilkan analitik umum.")
            st.write(df.describe())

# =========================
# TAB 4 – INFO (untuk klien)
# =========================
with t4:
    st.markdown("""
### ℹ️ Panduan Penggunaan (Untuk Klien)
Dashboard ini menampilkan **hasil akhir modelling** (prediksi) dan memudahkan analisis:
- **Auto-detect** proyek Anda: *Classifier* atau *Regressor*.
- **Filter data** kategori & angka, aman untuk data besar (berkala di-downsample saat plot).
- **Visualisasi otomatis** menyesuaikan tipe data (bar/histogram/scatter).
- **Export** data terfilter (CSV) & grafik (PNG).

    ---
### ⚡ Cara Pakai Cepat
1. Klik link yang sudah disediakan, maka akan automatis dibawa ke website streamlit cloud dimana dataset sudah siap di akses 
2. Gunakan filter di sidebar untuk menyaring data.  
3. Pilih jenis visualisasi → hasil otomatis menyesuaikan tipe data.  
4. Download data atau grafik sesuai kebutuhan.  

    ---
### 💡 Tips
- Tidak perlu repot pilih tipe visualisasi manual — sistem sudah otomatis mendeteksi dan menyesuaikan.  
- Untuk dataset besar, pastikan file CSV sudah hasil *preparation* agar proses lancar.  
- Instal **kaleido** sekali saja untuk ekspor grafik ke PNG.  
- Santai saja, semua fitur sudah siap — tinggal **lihat hasilnya**.  

    """)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("© 2025 Final Dashboard — Muhammad Fahri Zikri")
