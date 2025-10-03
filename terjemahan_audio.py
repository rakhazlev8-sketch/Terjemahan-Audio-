import streamlit as st
import librosa
import numpy as np
import random

# --- KONFIGURASI DAN KAMUS DATA ---

# Kamus untuk menyimpan kalimat Sunda dan terjemahan Inggrisnya.
TRANSLATION_MAP = {
    "Abdi hoyong tuang": "I want to eat",
    "Abdi geulis": "I am beautiful",
    "Abdi bade ka pasar": "I am going to the market", # Koreksi terjemahan agar lebih sesuai
    "Hatur nuhun": "Thank you",
    "Nuju naon": "What are you doing",
    "Kamana": "Where to",
    "Namina saha": "What is the name",
    "Bade meli naon": "What do you want to buy",
    "Sampurasun": "Excuse me",
    "Kumaha damang": "How are you"
}

# Daftar kalimat untuk digunakan dalam simulasi model
SUNDANESE_SENTENCES = list(TRANSLATION_MAP.keys())

# --- FUNGSI UTAMA ---

def extract_mfcc(audio_file, max_pad_len=174):
    """
    Mengekstrak fitur MFCC dari file audio.
    Fungsi ini memuat file audio, menghitung MFCC, dan memastikan
    output memiliki panjang yang konsisten dengan padding atau pemotongan.

    Args:
        audio_file: File audio yang diunggah Streamlit.
        max_pad_len (int): Panjang maksimum untuk padding/pemotongan.

    Returns:
        np.ndarray: Array numpy dari fitur MFCC.
    """
    try:
        # Muat file audio dengan librosa
        # sr=None berarti menggunakan sample rate asli file audio
        audio, sample_rate = librosa.load(audio_file, sr=None)

        # Hitung MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Lakukan padding atau pemotongan agar panjangnya seragam
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        return mfccs
    except Exception as e:
        st.error(f"Gagal memproses file audio: {e}")
        return None

def predict_sundanese_sentence(mfcc_features):
    """
    Fungsi placeholder/simulasi untuk model DNN.
    Fungsi ini tidak benar-benar menggunakan fitur MFCC, tetapi hanya
    secara acak memilih salah satu kalimat Sunda untuk mensimulasikan
    proses prediksi model.

    Args:
        mfcc_features (np.ndarray): Fitur MFCC (tidak digunakan dalam simulasi ini).

    Returns:
        str: String kalimat Sunda yang "diprediksi".
    """
    # Simulasi: Pilih kalimat secara acak dari daftar yang ada
    predicted_sentence = random.choice(SUNDANESE_SENTENCES)
    return predicted_sentence

# --- TAMPILAN ANTARMUKA (UI) STREAMLIT ---

# Mengatur judul dan header aplikasi
st.set_page_config(page_title="Penerjemah Suara Sunda", layout="centered")
st.title("🎤 Penerjemah Suara Sunda ke Inggris")
st.markdown("---")

# Menampilkan instruksi untuk pengguna
st.info(
    "**Cara Penggunaan:**\n"
    "1. Unggah file audio Anda dalam format `.wav`.\n"
    "2. Klik tombol 'Proses Audio' untuk memulai pengenalan suara.\n"
    "3. Hasil pengenalan dan terjemahan akan muncul di bawah."
)

# Widget untuk mengunggah file audio
uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav"])

# Tombol untuk memproses audio
if st.button("Proses Audio", use_container_width=True):
    if uploaded_file is not None:
        # Tampilkan status saat pemrosesan berjalan
        with st.spinner("Menganalisis audio... 🔊"):
            # 1. Ekstraksi Fitur MFCC
            mfcc_features = extract_mfcc(uploaded_file)

            if mfcc_features is not None:
                # 2. Simulasi Prediksi Model DNN
                recognized_sentence = predict_sundanese_sentence(mfcc_features)

                # 3. Dapatkan Terjemahan dari Kamus
                translation = TRANSLATION_MAP.get(recognized_sentence, "Terjemahan tidak ditemukan.")

                # 4. Tampilkan Hasil
                st.markdown("---")
                st.subheader("✅ Hasil Pengenalan")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Kalimat Dikenali (Sunda)", value=recognized_sentence)
                with col2:
                    st.metric(label="Terjemahan (Inggris)", value=translation)
    else:
        # Tampilkan pesan error jika tidak ada file yang diunggah
        st.error("⚠️ Harap unggah file audio terlebih dahulu.")