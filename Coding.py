import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(page_title="Dashboard Mini Analisis", layout="wide")

st.title("📊 Mini Dashboard Analisis Data Siswa")

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
    return df

df = load_data()

# Hapus kolom responden jika ada
if "Responden" in df.columns:
    data = df.drop(columns=["Responden"])
else:
    data = df.copy()

# Ambil numerik saja
data = data.select_dtypes(include=np.number)

# Tambah total skor
data["Total_Skor"] = data.sum(axis=1)

# ==================================================
# SIDEBAR REGRESI
# ==================================================
st.sidebar.header("⚙️ Pengaturan Regresi")

target = st.sidebar.selectbox(
    "Pilih Variabel Target (Y)",
    data.columns
)

opsi_fitur = [col for col in data.columns if col != target]
default_fitur = opsi_fitur[:min(3, len(opsi_fitur))]

fitur = st.sidebar.multiselect(
    "Pilih Variabel Prediktor (X)",
    opsi_fitur,
    default=default_fitur
)

# ==================================================
# TAB MENU
# ==================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Tabel",
    "📊 Diagram Batang",
    "📈 Grafik",
    "📉 Distribusi",
    "🥧 Diagram Lingkaran",
    "📊 Korelasi & Regresi"
])

# ==================================================
# TAB 1 — TABEL
# ==================================================
with tab1:
    st.subheader("Tabel Data")
    st.dataframe(data, use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(data.describe(), use_container_width=True)

# ==================================================
# TAB 2 — DIAGRAM BATANG
# ==================================================
with tab2:
    st.subheader("Rata-rata Tiap Soal")

    mean_values = data.mean()

    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(mean_values.index, mean_values.values)
    ax.tick_params(axis='x', rotation=90)

    st.pyplot(fig)

# ==================================================
# TAB 3 — GRAFIK
# ==================================================
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("Sumbu X", data.columns)

    with col2:
        y_axis = st.selectbox("Sumbu Y", data.columns, index=1)

    fig2, ax2 = plt.subplots()
    ax2.scatter(data[x_axis], data[y_axis])
    ax2.set_xlabel(x_axis)
    ax2.set_ylabel(y_axis)

    st.pyplot(fig2)

    st.line_chart(data[[x_axis, y_axis]])

# ==================================================
# TAB 4 — DISTRIBUSI
# ==================================================
with tab4:
    st.subheader("Histogram Total Skor")

    fig3, ax3 = plt.subplots()
    ax3.hist(data["Total_Skor"], bins=10)
    st.pyplot(fig3)

# ==================================================
# TAB 5 — PIE CHART
# ==================================================
with tab5:
    st.subheader("Kategori Total Skor")

    kategori = pd.cut(
        data["Total_Skor"],
        bins=3,
        labels=["Rendah", "Sedang", "Tinggi"]
    )

    kategori_count = kategori.value_counts()

    fig5, ax5 = plt.subplots()
    ax5.pie(
        kategori_count,
        labels=kategori_count.index,
        autopct='%1.1f%%',
        startangle=90
    )

    st.pyplot(fig5)

# ==================================================
# TAB 6 — KORELASI & REGRESI
# ==================================================
with tab6:

    # ---------- KORELASI ----------
    st.subheader("🔥 Matriks Korelasi")

    corr = data.corr()

    fig6, ax6 = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", ax=ax6)

    st.pyplot(fig6)

    # ---------- REGRESI ----------
    st.subheader("📈 Analisis Regresi Linear")

    if len(fitur) > 0:

        X = data[fitur]
        y = data[target]

        model = LinearRegression()
        model.fit(X, y)

        prediksi = model.predict(X)

        # Koefisien
        coef_df = pd.DataFrame({
            "Variabel": fitur,
            "Koefisien": model.coef_
        })

        st.write("### Koefisien Regresi")
        st.dataframe(coef_df, use_container_width=True)

        st.write("Intercept:", round(model.intercept_,4))

        # R2
        r2 = model.score(X, y)
        st.success(f"R² Score = {round(r2,4)}")

        # Grafik prediksi
        fig7, ax7 = plt.subplots()
        ax7.scatter(y, prediksi)
        ax7.set_xlabel("Nilai Aktual")
        ax7.set_ylabel("Nilai Prediksi")
        ax7.set_title("Aktual vs Prediksi")

        st.pyplot(fig7)

    else:
        st.warning("Pilih minimal satu variabel X.")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("Dashboard Analisis Lengkap • Diagram • Korelasi • Regresi")
