import streamlit as st
import librosa
import numpy as np
import random
import os

# --- KAMUS DATA 1: 10 KALIMAT UMUM ---
KALIMAT_MAP = {
    "Abdi hoyong tuang": "I want to eat", "Abdi geulis": "I am beautiful",
    "Abdi bade ka pasar": "I am going to the market", "Hatur nuhun": "Thank you",
    "Nuju naon": "What are you doing", "Kamana": "Where to",
    "Namina saha": "What is the name", "Bade meli naon": "What do you want to buy",
    "Sampurasun": "Excuse me", "Kumaha damang": "How are you"
}

# --- KAMUS DATA 2: 100 KATA SASTRAWI ---
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

def simulate_prediction(target_list):
    return random.choice(target_list)

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

# Pilihan 1: Mode Deteksi (Kata atau Kalimat)
app_mode = st.radio(
    "1. Pilih Mode Deteksi:",
    ("Deteksi Kalimat Umum", "Deteksi Kata Sastrawi"),
    horizontal=True
)

# Pilihan 2: Metode Prediksi
prediction_method = st.radio(
    "2. Pilih Metode Prediksi:",
    ("Simulasi Acak (dari Audio)", "Tebak dari Nama File"),
    horizontal=True
)
st.markdown("---")

# Logika untuk menentukan kamus dan tipe item
if app_mode == "Deteksi Kalimat Umum":
    TARGET_MAP = KALIMAT_MAP
    TARGET_LIST = list(KALIMAT_MAP.keys())
    item_type = "Kalimat"
    instruction_item = "satu kalimat"
else:
    TARGET_MAP = SASTRAWI_WORD_MAP
    TARGET_LIST = list(SASTRAWI_WORD_MAP.keys())
    item_type = "Kata"
    instruction_item = "satu kata"

st.info(
    f"**Mode Aktif:** `{app_mode}` | **Metode:** `{prediction_method}`\n\n"
    f"**Cara Penggunaan:**\n"
    f"1. Unggah file audio `.wav` berisi **{instruction_item}** Sunda.\n"
    f"2. Klik tombol di bawah untuk memulai prediksi."
)

uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    st.markdown(f"**Nama File:** `{uploaded_file.name}`")
    st.audio(uploaded_file, format='audio/wav')
    st.markdown("---")

    if st.button(f"Jalankan Prediksi ({item_type})", use_container_width=True):
        recognized_item = None
        
        # --- LOGIKA UTAMA YANG DIGABUNGKAN ---
        if prediction_method == "Simulasi Acak (dari Audio)":
            with st.spinner("Menganalisis audio & simulasi acak... 🔊"):
                mfcc_features = extract_mfcc(uploaded_file)
                if mfcc_features is not None:
                    recognized_item = simulate_prediction(TARGET_LIST)
        else: # "Tebak dari Nama File"
            with st.spinner("Membaca nama file... 📝"):
                recognized_item = predict_from_filename(uploaded_file.name)
        
        # --- Menampilkan Hasil ---
        if recognized_item:
            translation = TARGET_MAP.get(recognized_item, f"{item_type} '{recognized_item}' tidak ada dalam kamus.")
            st.subheader("✅ Hasil Prediksi")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"{item_type} Dikenali (Sunda)", value=recognized_item)
            with col2:
                st.metric(label="Terjemahan (Inggris)", value=translation)
        elif prediction_method == "Simulasi Acak (dari Audio)":
             st.error("Gagal memproses audio untuk simulasi.")
