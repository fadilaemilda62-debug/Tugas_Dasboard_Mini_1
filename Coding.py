import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Dashboard Analisis Siswa",
    layout="wide"
)

st.title("📊 Dashboard Analisis Data Siswa")

# ===============================
# LOAD DATA
# ===============================
file_path = "data_simulasi_50_siswa_20_soal.xlsx"
df = pd.read_excel(file_path)

# Ambil kolom soal saja
kolom_soal = df.columns[1:]

# ===============================
# HITUNG STATISTIK
# ===============================
df["Rata-rata Siswa"] = df[kolom_soal].mean(axis=1)
rata_soal = df[kolom_soal].mean()

# ===============================
# METRICS
# ===============================
st.subheader("📌 Statistik Umum")

col1, col2, col3 = st.columns(3)

col1.metric("Jumlah Responden", len(df))
col2.metric("Jumlah Soal", len(kolom_soal))
col3.metric("Rata-rata Keseluruhan", round(df["Rata-rata Siswa"].mean(), 2))

# ===============================
# GRAFIK RATA-RATA PER SOAL
# ===============================
st.subheader("📈 Rata-rata Nilai Tiap Soal")

fig1, ax1 = plt.subplots()
ax1.bar(rata_soal.index, rata_soal.values)
plt.xticks(rotation=90)
ax1.set_ylabel("Nilai Rata-rata")
st.pyplot(fig1)

# ===============================
# DISTRIBUSI NILAI SISWA
# ===============================
st.subheader("📊 Distribusi Rata-rata Nilai Siswa")

fig2, ax2 = plt.subplots()
ax2.hist(df["Rata-rata Siswa"], bins=10)
ax2.set_xlabel("Rata-rata Nilai")
ax2.set_ylabel("Jumlah Siswa")
st.pyplot(fig2)

# ===============================
# TABEL DATA
# ===============================
st.subheader("📋 Data Lengkap")

st.dataframe(df, use_container_width=True)

# ===============================
# FILTER SISWA
# ===============================
st.subheader("🔎 Filter Berdasarkan Responden")

nama = st.selectbox("Pilih Responden", df["Responden"])

data_siswa = df[df["Responden"] == nama]

st.write("Detail Nilai:")
st.dataframe(data_siswa)
