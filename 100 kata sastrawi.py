import streamlit as st
import librosa
import numpy as np
import random

# --- KONFIGURASI DAN KAMUS DATA ---

# Kamus 100 Kata Sastrawi Sunda-Inggris
NEW_TRANSLATION_MAP = {
    # Perasaan & Sifat
    "Anjeun": "You (poetic/formal)", "Tresna": "Deep Love", "Asih": "Affection/Love",
    "Kalbu": "Heart/Soul", "Lirih": "Soft/Gentle (voice)", "Balaka": "Honest/Frank",
    "Handeueul": "Regretful", "Jempling": "Silent/Serene", "Someah": "Friendly/Welcoming",
    "Geten": "Diligent", "Sauyunan": "In Harmony", "Langlaung": "Melancholic",
    "Kamelang": "Worried", "Deudeuh": "Compassion", "Bagja": "Happy/Blessed",
    "Bingah": "Joyful", "Amis": "Sweet", "Geulis": "Beautiful", "Kasep": "Handsome",
    
    # Alam & Lingkungan
    "Mega": "Cloud", "Angkeub": "Gloomy/Overcast", "Cangra": "Bright/Clear (sky)",
    "Layung": "Dusk/Twilight", "Galura": "Wave (water)", "Cai": "Water",
    "Seuneu": "Fire", "Angin": "Wind", "Langit": "Sky", "Bentang": "Star",
    "Soca": "Eye (poetic)", "Gunung": "Mountain", "Leuweung": "Forest",
    "Walungan": "River", "Tatangkalan": "Trees", "Kembang": "Flower",
    "Hujan": "Rain", "Basisir": "Coast/Beach", "Ibuh": "Mist/Fog",
    "Lamping": "Slope/Hillside", "Sawah": "Paddy Field", "Parak": "Orchard/Garden",

    # Waktu & Keadaan
    "Wanci": "Time/Period", "Kamari": "Yesterday", "Pageto": "Day after tomorrow",
    "Isuk": "Morning", "Beurang": "Daytime", "Peuting": "Night",
    "Reumbeuy": "Drizzling", "Girimis": "Light Rain", "Lalakon": "Life Story/Fate",
    "Ringkang": "Step/Footstep", "Lawas": "Old/Ancient", "Buhun": "Ancient/Traditional",
    "Kiwari": "Present day", "Salawasna": "Forever", "Sakedap": "A moment",

    # Benda & Konsep
    "Sajak": "Poem/Verse", "Sora": "Voice/Sound", "Carita": "Story",
    "Pakarang": "Weapon/Tool", "Waruga": "Physical Body", "Batin": "Inner self",
    "Imah": "House/Home", "Lembur": "Village/Hometown", "Nagara": "Country/State",
    "Jagat": "Universe", "Alam": "Nature", "Ruhay": "Blazing (fire)",
    "Panceg": "Steadfast/Firm", "Ajeg": "Upright/Stable", "Lestari": "Everlasting/Sustainable",
    "Jatnika": "Careful/Prudent",

    # Kata Kerja & Lainnya
    "Neuteup": "To Gaze", "Nganti": "To Wait", "Miang": "To Depart/Go",
    "Mulang": "To Return Home", "Seuri": "To Laugh", "Ceungceurikan": "To Weep",
    "Ngadangu": "To Hear (formal)", "Ningali": "To See (formal)", "Nyarios": "To Speak (formal)",
    "Leumpang": "To Walk", "Lumpat": "To Run", "Cicing": "To be quiet/stay",
    "Neang": "To Search for", "Manggih": "To Find", "Leungit": "Lost",
    "Hirup": "Life/Alive", "Paeh": "Dead", "Caang": "Bright/Light", "Poek": "Dark"
}

# Daftar kata untuk digunakan dalam simulasi model
SUNDANESE_WORDS = list(NEW_TRANSLATION_MAP.keys())

# --- FUNGSI UTAMA ---

def extract_mfcc(audio_file, max_pad_len=174):
    """
    Mengekstrak fitur MFCC dari file audio.
    """
    try:
        audio, sample_rate = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return mfccs
    except Exception as e:
        st.error(f"Gagal memproses file audio: {e}")
        return None

def predict_sundanese_word(mfcc_features):
    """
    Fungsi placeholder/simulasi untuk model DNN.
    """
    predicted_word = random.choice(SUNDANESE_WORDS)
    return predicted_word

# --- TAMPILAN ANTARMUKA (UI) STREAMLIT ---

st.set_page_config(page_title="Penerjemah Suara Sunda", layout="centered")
st.title("🎤 Penerjemah Suara Kata Sunda ke Inggris")
st.markdown("---")

st.info(
    "**Cara Penggunaan:**\n"
    "1. Unggah file audio `.wav` berisi **satu kata** Sunda.\n"
    "2. Nama file dan pemutar audio akan muncul secara otomatis.\n"
    "3. Klik tombol 'Proses Audio' untuk memulai prediksi AI."
)

uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav"])

# --- PERUBAHAN UTAMA DIMULAI DARI SINI ---

# Cek apakah file sudah diunggah
if uploaded_file is not None:
    
    # BARU: Tampilkan nama file yang diunggah
    st.markdown(f"**Nama File:** `{uploaded_file.name}`")
    
    # BARU: Tampilkan widget pemutar audio
    st.audio(uploaded_file, format='audio/wav')
    
    st.markdown("---")

    # Tombol proses audio sekarang hanya muncul setelah file diunggah
    if st.button("Proses Audio & Prediksi Kata", use_container_width=True):
        with st.spinner("Menganalisis audio... 🔊"):
            mfcc_features = extract_mfcc(uploaded_file)
            
            if mfcc_features is not None:
                recognized_word = predict_sundanese_word(mfcc_features)
                translation = NEW_TRANSLATION_MAP.get(recognized_word, "Terjemahan tidak ditemukan.")
                
                st.subheader("✅ Hasil Prediksi")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Kata Dikenali (Sunda)", value=recognized_word)
                with col2:
                    st.metric(label="Terjemahan (Inggris)", value=translation)