import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# =====================================================
# KONFIGURASI
# =====================================================
st.set_page_config(page_title="Dashboard Analisis Penelitian", layout="wide")

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
# SIDEBAR REGRESI
# =====================================================
st.sidebar.header("⚙️ Pengaturan Analisis")

target = st.sidebar.selectbox(
    "Variabel Target (Y)",
    data.columns
)

opsi_fitur = [c for c in data.columns if c != target]
default_fitur = opsi_fitur[:min(3, len(opsi_fitur))]

fitur = st.sidebar.multiselect(
    "Variabel Prediktor (X)",
    opsi_fitur,
    default=default_fitur
)

# =====================================================
# TAB
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Statistik",
    "🔥 Korelasi",
    "📈 Regresi",
    "🧠 Kesimpulan"
])

# =====================================================
# TAB 1 — STATISTIK
# =====================================================
with tab1:

    st.subheader("Statistik Deskriptif")
    st.dataframe(data.describe(), use_container_width=True)

    fig1, ax1 = plt.subplots()
    ax1.hist(data["Total_Skor"], bins=10)
    ax1.set_title("Distribusi Total Skor")

    st.pyplot(fig1)

# =====================================================
# TAB 2 — KORELASI
# =====================================================
with tab2:

    st.subheader("Matriks Korelasi")

    corr = data.corr()

    fig2, ax2 = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", ax=ax2)

    st.pyplot(fig2)

    # korelasi rata-rata
    mean_corr = corr.abs().mean().mean()

    st.info(f"Rata-rata kekuatan korelasi = {round(mean_corr,3)}")

# =====================================================
# TAB 3 — REGRESI
# =====================================================
with tab3:

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

        st.subheader("Koefisien Regresi")
        st.dataframe(coef_df, use_container_width=True)

        intercept = model.intercept_
        r2 = model.score(X, y)

        st.write("Intercept:", round(intercept,4))
        st.success(f"R² Score = {round(r2,4)}")

        # grafik prediksi
        fig3, ax3 = plt.subplots()
        ax3.scatter(y, prediksi)
        ax3.set_xlabel("Aktual")
        ax3.set_ylabel("Prediksi")
        ax3.set_title("Aktual vs Prediksi")

        st.pyplot(fig3)

    else:
        st.warning("Pilih minimal satu variabel prediktor.")

# =====================================================
# TAB 4 — KESIMPULAN OTOMATIS
# =====================================================
with tab4:

    st.subheader("Kesimpulan Analisis Otomatis")

    # Statistik
    mean_total = data["Total_Skor"].mean()

    if mean_total < data["Total_Skor"].quantile(0.33):
        kategori = "rendah"
    elif mean_total < data["Total_Skor"].quantile(0.66):
        kategori = "sedang"
    else:
        kategori = "tinggi"

    # Korelasi interpretasi
    if mean_corr < 0.3:
        korelasi_text = "lemah"
    elif mean_corr < 0.6:
        korelasi_text = "sedang"
    else:
        korelasi_text = "kuat"

    # Regresi interpretasi
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
    Berdasarkan hasil analisis data:

    1. Rata-rata total skor siswa berada pada kategori **{kategori}**.
    2. Hubungan antar variabel menunjukkan tingkat korelasi **{korelasi_text}**.
    3. Model regresi memiliki kemampuan prediksi **{regresi_text}** dengan nilai R² sebesar {round(r2,3) if len(fitur)>0 else "-"}.
    
    Secara umum, data menunjukkan adanya hubungan antar variabel yang dapat digunakan
    untuk menjelaskan variasi pada variabel target penelitian.
    """

    st.success(kesimpulan)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("Dashboard Analisis Penelitian Otomatis")
