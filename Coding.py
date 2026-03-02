import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

st.set_page_config(layout="wide")

# =====================================================
# HEADER
# =====================================================
st.title("📊 Mini Dashboard Analisis Data Siswa")
st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================
file = "data_simulasi_50_siswa_20_soal.xlsx"
df = pd.read_excel(file)

soal_cols = [c for c in df.columns if "Soal" in c]

# =====================================================
# DATA OTOMATIS
# =====================================================
df["Total_Nilai"] = df[soal_cols].sum(axis=1)
df["Rata_siswa"] = df[soal_cols].mean(axis=1)
rata_soal = df[soal_cols].mean()

# =====================================================
# TAB MENU (SEMUA DI BAWAH JUDUL)
# =====================================================
tabs = st.tabs([
"📋 Data Identitas",
"📊 Skor & Nilai Siswa",
"📈 Statistik Deskriptif",
"🧪 Analisis Butir Soal",
"🔗 Korelasi",
"📉 Regresi Linear",
"📊 Distribusi Data",
"🥧 Diagram Lingkaran",
"📊 Grafik Analisis",
"🧠 Kesimpulan",
"🎓 Validitas Butir",
"🧮 Reliabilitas (Cronbach Alpha)",
"📏 Indeks Kesukaran",
"⚖️ Daya Pembeda"
])

# =====================================================
# 1 DATA IDENTITAS
# =====================================================
with tabs[0]:
    st.subheader("Data Identitas")
    st.dataframe(df.select_dtypes(include="object"))

# =====================================================
# 2 SKOR SISWA
# =====================================================
with tabs[1]:
    st.subheader("Skor & Nilai Siswa")
    st.dataframe(df[soal_cols + ["Total_Nilai","Rata_siswa"]])

# =====================================================
# 3 STATISTIK
# =====================================================
with tabs[2]:
    stat = df[soal_cols].agg(
        ["mean","median","std","min","max"]
    ).T
    stat["modus"] = df[soal_cols].mode().iloc[0]
    st.dataframe(stat)

# =====================================================
# 4 ANALISIS BUTIR
# =====================================================
with tabs[3]:
    st.dataframe(rata_soal)

    fig, ax = plt.subplots()
    rata_soal.plot(kind="bar", ax=ax)
    st.pyplot(fig)

# =====================================================
# 5 KORELASI
# =====================================================
with tabs[4]:
    corr = df[soal_cols].corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =====================================================
# 6 REGRESI
# =====================================================
with tabs[5]:
    X = df[["Rata_siswa"]]
    y = df["Total_Nilai"]

    model = LinearRegression().fit(X,y)
    pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X,y)
    ax.plot(X,pred)
    st.pyplot(fig)

    st.write("Koefisien:", model.coef_[0])
    st.write("R²:", model.score(X,y))

# =====================================================
# 7 DISTRIBUSI
# =====================================================
with tabs[6]:
    fig, ax = plt.subplots()
    ax.hist(df["Total_Nilai"], bins=10)
    st.pyplot(fig)

# =====================================================
# 8 PIE CHART
# =====================================================
with tabs[7]:
    kategori = pd.cut(df["Total_Nilai"],3,
                      labels=["Rendah","Sedang","Tinggi"])

    fig, ax = plt.subplots()
    kategori.value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    st.pyplot(fig)

# =====================================================
# 9 GRAFIK
# =====================================================
with tabs[8]:

    col1,col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        rata_soal.plot(kind="bar", ax=ax1)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(df["Total_Nilai"])
        st.pyplot(fig2)

# =====================================================
# 10 KESIMPULAN
# =====================================================
with tabs[9]:

    mean_total = df["Total_Nilai"].mean()
    max_total = df["Total_Nilai"].max()

    if mean_total > 0.7*max_total:
        st.success("Performa siswa tinggi")
    elif mean_total > 0.4*max_total:
        st.warning("Performa siswa sedang")
    else:
        st.error("Performa siswa rendah")

# =====================================================
# 11 VALIDITAS
# =====================================================
with tabs[10]:

    total = df["Total_Nilai"]
    hasil = []

    for col in soal_cols:
        r,_ = pearsonr(df[col], total)
        hasil.append(r)

    validitas = pd.DataFrame({
        "Soal":soal_cols,
        "r_hitung":hasil
    })

    st.dataframe(validitas)

# =====================================================
# 12 RELIABILITAS
# =====================================================
with tabs[11]:

    k = len(soal_cols)
    var_item = df[soal_cols].var(axis=0, ddof=1)
    var_total = df[soal_cols].sum(axis=1).var(ddof=1)

    alpha = (k/(k-1))*(1-(var_item.sum()/var_total))

    st.success(f"Cronbach Alpha = {alpha:.3f}")

# =====================================================
# 13 INDEKS KESUKARAN
# =====================================================
with tabs[12]:

    indeks = df[soal_cols].mean()
    st.dataframe(indeks)

# =====================================================
# 14 DAYA PEMBEDA
# =====================================================
with tabs[13]:

    df_sorted = df.sort_values("Total_Nilai")
    n = int(len(df)*0.27)

    bawah = df_sorted.head(n)
    atas = df_sorted.tail(n)

    dp = atas[soal_cols].mean() - bawah[soal_cols].mean()
    st.dataframe(dp)
