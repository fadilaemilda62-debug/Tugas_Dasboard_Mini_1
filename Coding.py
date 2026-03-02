import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(page_title="Dashboard Analisis Lengkap", layout="wide")

st.title("📊 Dashboard Analisis Data Penelitian")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
    return df

df = load_data()

# preprocessing
if "Responden" in df.columns:
    data = df.drop(columns=["Responden"])
else:
    data = df.copy()

data = data.select_dtypes(include=np.number)

# total skor
data["Total_Skor"] = data.sum(axis=1)

# =====================================================
# SIDEBAR (REGRESI)
# =====================================================
st.sidebar.header("⚙️ Pengaturan Analisis")

target = st.sidebar.selectbox("Variabel Target (Y)", data.columns)

opsi_fitur = [c for c in data.columns if c != target]
default_fitur = opsi_fitur[:min(3, len(opsi_fitur))]

fitur = st.sidebar.multiselect(
    "Variabel Prediktor (X)",
    opsi_fitur,
    default=default_fitur
)

# =====================================================
# TAB DASHBOARD
# =====================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Tabel",
    "📊 Statistik",
    "📈 Grafik",
    "📉 Distribusi",
    "🥧 Diagram Lingkaran",
    "🔥 Korelasi & Regresi",
    "🧠 Kesimpulan"
])

# =====================================================
# TAB 1 — TABEL
# =====================================================
with tab1:
    st.subheader("Tabel Data")
    st.dataframe(data, use_container_width=True)

# =====================================================
# TAB 2 — STATISTIK
# =====================================================
with tab2:
    st.subheader("Statistik Deskriptif")
    stats = data.describe()
    st.dataframe(stats, use_container_width=True)

    mean_values = data.mean()

    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(mean_values.index, mean_values.values)
    ax.set_title("Rata-rata Variabel")
    ax.tick_params(axis='x', rotation=90)
    st.pyplot(fig)

# =====================================================
# TAB 3 — GRAFIK
# =====================================================
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

# =====================================================
# TAB 4 — DISTRIBUSI
# =====================================================
with tab4:
    st.subheader("Distribusi Total Skor")

    fig3, ax3 = plt.subplots()
    ax3.hist(data["Total_Skor"], bins=10)
    ax3.set_xlabel("Total Skor")
    ax3.set_ylabel("Frekuensi")

    st.pyplot(fig3)

# =====================================================
# TAB 5 — DIAGRAM LINGKARAN
# =====================================================
with tab5:

    st.subheader("Diagram Lingkaran Kategori Skor")

    kategori = pd.cut(
        data["Total_Skor"],
        bins=3,
        labels=["Rendah", "Sedang", "Tinggi"]
    )

    kategori_count = kategori.value_counts()

    fig4, ax4 = plt.subplots()
    ax4.pie(
        kategori_count,
        labels=kategori_count.index,
        autopct='%1.1f%%',
        startangle=90
    )

    st.pyplot(fig4)

# =====================================================
# TAB 6 — KORELASI & REGRESI
# =====================================================
with tab6:

    st.subheader("Heatmap Korelasi")

    corr = data.corr()

    fig5, ax5 = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)

    mean_corr = corr.abs().mean().mean()
    st.info(f"Rata-rata kekuatan korelasi: {round(mean_corr,3)}")

    # REGRESI
    st.subheader("Regresi Linear")

    if len(fitur) > 0:

        X = data[fitur]
        y = data[target]

        model = LinearRegression()
        model.fit(X, y)

        prediksi = model.predict(X)

        coef_df = pd.DataFrame({
            "Variabel": fitur,
            "Koefisien": model.coef_
        })

        st.dataframe(coef_df, use_container_width=True)

        r2 = model.score(X, y)
        st.success(f"R² Score = {round(r2,4)}")

        fig6, ax6 = plt.subplots()
        ax6.scatter(y, prediksi)
        ax6.set_xlabel("Aktual")
        ax6.set_ylabel("Prediksi")

        st.pyplot(fig6)

    else:
        st.warning("Pilih variabel prediktor.")

# =====================================================
# TAB 7 — KESIMPULAN OTOMATIS
# =====================================================
with tab7:

    st.subheader("Kesimpulan Analisis")

    mean_total = data["Total_Skor"].mean()

    if mean_total < data["Total_Skor"].quantile(0.33):
        kategori_text = "rendah"
    elif mean_total < data["Total_Skor"].quantile(0.66):
        kategori_text = "sedang"
    else:
        kategori_text = "tinggi"

    if mean_corr < 0.3:
        korelasi_text = "lemah"
    elif mean_corr < 0.6:
        korelasi_text = "sedang"
    else:
        korelasi_text = "kuat"

    if len(fitur) > 0:
        if r2 < 0.3:
            regresi_text = "rendah"
        elif r2 < 0.7:
            regresi_text = "cukup baik"
        else:
            regresi_text = "sangat baik"
    else:
        regresi_text = "belum dianalisis"

    kesimpulan = f"""
    • Rata-rata kemampuan siswa berada pada kategori **{kategori_text}**.  
    • Hubungan antar variabel menunjukkan korelasi **{korelasi_text}**.  
    • Model regresi memiliki kemampuan prediksi **{regresi_text}**.  

    Secara umum data menunjukkan adanya hubungan antar variabel yang dapat
    digunakan untuk menjelaskan variasi pada variabel target penelitian.
    """

    st.success(kesimpulan)

# =====================================================
st.markdown("---")
st.caption("Dashboard Analisis Lengkap • Statistik • Korelasi • Regresi")
