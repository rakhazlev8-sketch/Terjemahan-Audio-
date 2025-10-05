import streamlit as st
import librosa
import numpy as np
import random
import os
import time

# --- KONFIGURASI ---
CONFIDENCE_THRESHOLD = 0.5 # Ambang batas kepercayaan (50%).

# --- KAMUS DATA 1 & 2 ---
KALIMAT_MAP = {
    "Abdi hoyong tuang": "I want to eat", "Abdi geulis": "I am beautiful",
    "Abdi bade ka pasar": "I am going to the market", "Hatur nuhun": "Thank you",
    "Nuju naon": "What are you doing", "Kamana": "Where to",
    "Namina saha": "What is the name", "Bade meli naon": "What do you want to buy",
    "Sampurasun": "Excuse me", "Kumaha damang": "How are you"
}

SASTRAWI_WORD_MAP = {
    "Anjeun": "You (poetic/formal)", "Tresna": "Deep Love", "Asih": "Affection/Love", "Kalbu": "Heart/Soul", 
    "Lirih": "Soft/Gentle (voice)", "Balaka": "Honest/Frank", "Handeueul": "Regretful", "Jempling": "Silent/Serene", 
    "Someah": "Friendly/Welcoming", "Geten": "Diligent", "Sauyunan": "In Harmony", "Langlaung": "Melancholic", 
    "Kamelang": "Worried", "Deudeuh": "Compassion", "Bagja": "Happy/Blessed", "Bingah": "Joyful", "Amis": "Sweet", 
    "Geulis": "Beautiful", "Kasep": "Handsome", "Mega": "Cloud", "Angkeub": "Gloomy/Overcast", 
    "Cangra": "Bright/Clear (sky)", "Layung": "Dusk/Twilight", "Galura": "Wave (water)", "Cai": "Water", 
    "Seuneu": "Fire", "Angin": "Wind", "Langit": "Sky", "Bentang": "Star", "Soca": "Eye (poetic)", 
    "Gunung": "Mountain", "Leuweung": "Forest", "Walungan": "River", "Tatangkalan": "Trees", "Kembang": "Flower", 
    "Hujan": "Rain", "Basisir": "Coast/Beach", "Ibuh": "Mist/Fog", "Lamping": "Slope/Hillside", 
    "Sawah": "Paddy Field", "Parak": "Orchard/Garden", "Wanci": "Time/Period", "Kamari": "Yesterday", 
    "Pageto": "Day after tomorrow", "Isuk": "Morning", "Beurang": "Daytime", "Peuting": "Night", 
    "Reumbeuy": "Drizzling", "Girimis": "Light Rain", "Lalakon": "Life Story/Fate", "Ringkang": "Step/Footstep", 
    "Lawas": "Old/Ancient", "Buhun": "Ancient/Traditional", "Kiwari": "Present day", "Salawasna": "Forever", 
    "Sakedap": "A moment", "Sajak": "Poem/Verse", "Sora": "Voice/Sound", "Carita": "Story", 
    "Pakarang": "Weapon/Tool", "Waruga": "Physical Body", "Batin": "Inner self", "Imah": "House/Home", 
    "Lembur": "Village/Hometown", "Nagara": "Country/State", "Jagat": "Universe", "Alam": "Nature", 
    "Ruhay": "Blazing (fire)", "Panceg": "Steadfast/Firm", "Ajeg": "Upright/Stable", "Lestari": "Everlasting/Sustainable", 
    "Jatnika": "Careful/Prudent", "Neuteup": "To Gaze", "Nganti": "To Wait", "Miang": "To Depart/Go", 
    "Mulang": "To Return Home", "Seuri": "To Laugh", "Ceungceurikan": "To Weep", "Ngadangu": "To Hear (formal)", 
    "Ningali": "To See (formal)", "Nyarios": "To Speak (formal)", "Leumpang": "To Walk", "Lumpat": "To Run", 
    "Cicing": "To be quiet/stay", "Neang": "To Search for", "Manggih": "To Find", "Leungit": "Lost", 
    "Hirup": "Life/Alive", "Paeh": "Dead", "Caang": "Bright/Light", "Poek": "Dark"
}

# --- KUMPULAN FUNGSI UTAMA ---

def extract_mfcc(audio_file, max_pad_len=174):
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

def simulate_audio_prediction(target_list):
    predicted_item = random.choice(target_list)
    confidence = random.random()
    return predicted_item, confidence

def predict_from_filename(filename):
    base_name = os.path.splitext(filename)[0]
    # Mengganti spasi dengan underscore untuk file kalimat, lalu split
    parts = base_name.replace(" ", "_").split('_')
    # Menggabungkan kembali jika nama file kalimat mengandung spasi
    keyword_base = parts[0]
    if keyword_base in ["Abdi", "Hatur", "Nuju", "Bade", "Namina", "Kumaha"]:
         # Heuristik sederhana untuk kalimat, bisa disempurnakan
         keyword = " ".join(base_name.split('_')[0].split()[:3]).capitalize()
    else:
         keyword = keyword_base.capitalize()
    
    # Untuk kasus khusus kalimat yang lebih dari 1 kata
    full_sentence_guess = " ".join(base_name.split('_'))
    if full_sentence_guess in KALIMAT_MAP:
        return full_sentence_guess
    
    # Mencocokan keyword yang sudah di proses
    for key in list(KALIMAT_MAP.keys()) + list(SASTRAWI_WORD_MAP.keys()):
        if key.startswith(keyword):
            return key
            
    return keyword

# --- TAMPILAN ANTARMUKA (UI) STREAMLIT ---

st.set_page_config(page_title="Penerjemah Suara Sunda", layout="centered")
st.title("🎤 Penerjemah Suara Sunda ke Inggris")
st.markdown("---")

app_mode = st.radio(
    "Pilih Mode Deteksi:",
    ("Deteksi Kalimat Umum", "Deteksi Kata Sastrawi"),
    horizontal=True
)
st.markdown("---")

if app_mode == "Deteksi Kalimat Umum":
    TARGET_
