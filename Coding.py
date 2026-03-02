import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================
# KONFIGURASI DASHBOARD
# ======================================
st.set_page_config(page_title="Dashboard Mini", layout="wide")

st.title("📊 Dashboard Analisis Data")

# ======================================
# LOAD DATA
# ======================================
@st.cache_data
def load_data():
    df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
    return df

df = load_data()

# hapus kolom responden jika ada
if "Responden" in df.columns:
    data = df.drop(columns=["Responden"])
else:
    data = df.copy()

# ambil hanya numerik
data = data.select_dtypes(include=np.number)

# tambah total skor
data["Total_Skor"] = data.sum(axis=1)

# ======================================
# MENU TAB
# ======================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Tabel",
    "📊 Diagram",
    "📈 Grafik",
    "📉 Distribusi"
])

# ======================================
# TAB 1 — TABEL
# ======================================
with tab1:

    st.subheader("Tabel Data")
    st.dataframe(data, use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(data.describe(), use_container_width=True)

# ======================================
# TAB 2 — DIAGRAM (BAR CHART)
# ======================================
with tab2:

    st.subheader("Diagram Rata-rata Tiap Soal")

    mean_values = data.mean()

    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(mean_values.index, mean_values.values)
    ax.set_xlabel("Variabel")
    ax.set_ylabel("Rata-rata")
    ax.tick_params(axis='x', rotation=90)

    st.pyplot(fig)

# ======================================
# TAB 3 — GRAFIK
# ======================================
with tab3:

    st.subheader("Grafik Interaktif")

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("Pilih Sumbu X", data.columns)

    with col2:
        y_axis = st.selectbox("Pilih Sumbu Y", data.columns, index=1)

    # Scatter plot
    fig2, ax2 = plt.subplots()
    ax2.scatter(data[x_axis], data[y_axis])
    ax2.set_xlabel(x_axis)
    ax2.set_ylabel(y_axis)

    st.pyplot(fig2)

    # Line chart
    st.subheader("Line Chart")
    st.line_chart(data[[x_axis, y_axis]])

# ======================================
# TAB 4 — DISTRIBUSI (HISTOGRAM + HEATMAP)
# ======================================
with tab4:

    st.subheader("Histogram Total Skor")

    fig3, ax3 = plt.subplots()
    ax3.hist(data["Total_Skor"], bins=10)
    ax3.set_xlabel("Total Skor")
    ax3.set_ylabel("Frekuensi")

    st.pyplot(fig3)

    st.subheader("Heatmap Korelasi")

    corr = data.corr()

    fig4, ax4 = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", ax=ax4)

    st.pyplot(fig4)

# ======================================
# FOOTER
# ======================================
st.markdown("---")
st.caption("Dashboard Mini • Diagram • Grafik • Tabel")
