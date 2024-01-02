import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd

def generate_plots(df):
    # Plot histogram
    st.subheader("Histogram Distribusi Fitur")
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        if df[column].dtype != 'O':  # Plot hanya untuk fitur numerik
            sns.histplot(df[column], bins=20, kde=True)
            st.pyplot()
            st.write(f"### Distribusi dari {column}")

    # Plot correlation heatmap
    st.subheader("Heatmap Korelasi Fitur")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    st.pyplot()

# Load dataset
st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data():
    # Load dataset sesuai kebutuhan
    df = pd.read_csv('Penyakit_Paru.csv')
    return df

def app():

    # Judul halaman aplikasi
    st.title("APLIKASI PREDIKSI TERKENA PENYAKIT PARU-PARU")
    
    # Load dataset
    df = load_data()

    # Tampilkan gambar
    st.image('paru.jpg', use_column_width=True)

    # Tampilkan keterangan
    st.write("Penyakit paru-paru adalah salah satu penyebab kematian tertinggi di dunia. Penyakit ini dapat disebabkan oleh berbagai faktor. Aplikasi ini dapat digunakan untuk memperkirakan risiko seseorang terkena penyakit paru-paru berdasarkan faktor-faktor risiko yang dapat diukur. aplikasi ini sudah melakukan riset kepada 30rb orang yang diliat dari beberapa faktor yaitu : Usia, Jenis_Kelamin, Merokok, Bekerja, Rumah_Tangga, Aktivitas_Begadang, Aktivitas_Olahraga, Asuransi, Penyakit_Bawaan,Hasil.")
             
    # Tampilkan grafik
    generate_plots(df)
