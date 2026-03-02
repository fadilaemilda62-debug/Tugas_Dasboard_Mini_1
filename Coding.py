import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

st.set_page_config(layout="wide")
st.title("🎓 Dashboard Analisis Instrumen Tes (Level Penelitian)")

# =====================================================
# LOAD DATA
# =====================================================
file = "data_simulasi_50_siswa_20_soal.xlsx"
df = pd.read_excel(file)

soal_cols = [c for c in df.columns if "Soal" in c]

# =====================================================
# FILTER INTERAKTIF
# =====================================================
st.sidebar.header("🎛️ Filter Data")

if "Jenis_Kelamin" in df.columns:
    gender = st.sidebar.multiselect(
        "Filter Jenis Kelamin",
        df["Jenis_Kelamin"].unique(),
        default=df["Jenis_Kelamin"].unique()
    )
    df = df[df["Jenis_Kelamin"].isin(gender)]

# =====================================================
# DATA OTOMATIS
# =====================================================
df["Total_Nilai"] = df[soal_cols].sum(axis=1)
df["Rata_siswa"] = df[soal_cols].mean(axis=1)
rata_soal = df[soal_cols].mean()

# =====================================================
# SIDEBAR MENU
# =====================================================
menu = st.sidebar.radio("Menu Dashboard",[
"Data Identitas",
"Skor & Nilai Siswa",
"Statistik Deskriptif",
"Analisis Butir Soal",
"Korelasi",
"Regresi Linear",
"Distribusi Data",
"Diagram Lingkaran",
"Grafik Analisis",
"Kesimpulan",
"Validitas Butir",
"Reliabilitas (Cronbach Alpha)",
"Indeks Kesukaran",
"Daya Pembeda"
])

# =====================================================
# DATA IDENTITAS
# =====================================================
if menu == "Data Identitas":
    st.dataframe(df.select_dtypes(include="object"))

# =====================================================
# SKOR SISWA
# =====================================================
elif menu == "Skor & Nilai Siswa":
    st.dataframe(df[["Total_Nilai","Rata_siswa"] + soal_cols])

# =====================================================
# STATISTIK DESKRIPTIF
# =====================================================
elif menu == "Statistik Deskriptif":

    stat = df[soal_cols].agg(
        ["mean","median","std","min","max"]
    ).T

    stat["modus"] = df[soal_cols].mode().iloc[0]

    st.dataframe(stat)

# =====================================================
# ANALISIS BUTIR
# =====================================================
elif menu == "Analisis Butir Soal":

    st.dataframe(rata_soal)

    fig, ax = plt.subplots()
    rata_soal.plot(kind="bar", ax=ax)
    st.pyplot(fig)

# =====================================================
# KORELASI
# =====================================================
elif menu == "Korelasi":

    corr = df[soal_cols].corr()

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =====================================================
# REGRESI
# =====================================================
elif menu == "Regresi Linear":

    X = df[["Rata_siswa"]]
    y = df["Total_Nilai"]

    model = LinearRegression().fit(X,y)

    pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X,y)
    ax.plot(X,pred)
    st.pyplot(fig)

    st.write("Koefisien:",model.coef_[0])
    st.write("R²:",model.score(X,y))

# =====================================================
# DISTRIBUSI
# =====================================================
elif menu == "Distribusi Data":

    fig, ax = plt.subplots()
    ax.hist(df["Total_Nilai"], bins=10)
    st.pyplot(fig)

# =====================================================
# PIE CHART
# =====================================================
elif menu == "Diagram Lingkaran":

    kategori = pd.cut(df["Total_Nilai"],3,
                      labels=["Rendah","Sedang","Tinggi"])

    fig, ax = plt.subplots()
    kategori.value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    st.pyplot(fig)

# =====================================================
# GRAFIK
# =====================================================
elif menu == "Grafik Analisis":

    col1,col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(df["Total_Nilai"])
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        rata_soal.plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

# =====================================================
# VALIDITAS BUTIR
# =====================================================
elif menu == "Validitas Butir":

    hasil = []

    total = df["Total_Nilai"]

    for col in soal_cols:
        r,_ = pearsonr(df[col], total)
        hasil.append(r)

    validitas = pd.DataFrame({
        "Soal":soal_cols,
        "r_hitung":hasil
    })

    st.dataframe(validitas)

# =====================================================
# CRONBACH ALPHA
# =====================================================
elif menu == "Reliabilitas (Cronbach Alpha)":

    k = len(soal_cols)
    var_item = df[soal_cols].var(axis=0, ddof=1)
    var_total = df[soal_cols].sum(axis=1).var(ddof=1)

    alpha = (k/(k-1))*(1-(var_item.sum()/var_total))

    st.success(f"Cronbach Alpha = {alpha:.3f}")

# =====================================================
# INDEKS KESUKARAN
# =====================================================
elif menu == "Indeks Kesukaran":

    indeks = df[soal_cols].mean()

    kategori = pd.cut(
        indeks,
        bins=[0,0.3,0.7,1],
        labels=["Sulit","Sedang","Mudah"]
    )

    hasil = pd.DataFrame({
        "Indeks":indeks,
        "Kategori":kategori
    })

    st.dataframe(hasil)

# =====================================================
# DAYA PEMBEDA
# =====================================================
elif menu == "Daya Pembeda":

    df_sorted = df.sort_values("Total_Nilai")

    n = int(len(df)*0.27)

    bawah = df_sorted.head(n)
    atas = df_sorted.tail(n)

    dp = atas[soal_cols].mean() - bawah[soal_cols].mean()

    st.dataframe(dp)

# =====================================================
# KESIMPULAN
# =====================================================
elif menu == "Kesimpulan":

    mean_total = df["Total_Nilai"].mean()
    max_total = df["Total_Nilai"].max()

    if mean_total > 0.7*max_total:
        st.success("Kemampuan siswa tinggi.")
    elif mean_total > 0.4*max_total:
        st.warning("Kemampuan siswa sedang.")
    else:
        st.error("Kemampuan siswa rendah.")
