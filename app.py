import os
import io
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import pandas as pd
import gdown

# ============================================================
# DOWNLOAD MODEL DARI GOOGLE DRIVE (JIKA BELUM ADA LOKAL)
# ============================================================
MODEL_DRIVE_IDS = {
    "model_tomat_resnetlike_scratch.keras": "1nFwZqbeNJp-ciDNXTljDzvtL_M4BOEvP",
    "model_tomat_vgglike_scratch.keras"    : "1PYcvj4Nkpb6IZFy4VJTkE6p9tq182EfN",
}

def ensure_model_from_drive(file_name):
    """Jika file model belum ada, download dari Google Drive."""
    if os.path.exists(file_name):
        return True
    file_id = MODEL_DRIVE_IDS.get(file_name)
    if not file_id:
        return False
    url = f"https://drive.google.com/uc?id={file_id}"
    with st.spinner(f"⬇️ Mengunduh {file_name} dari Google Drive (sekitar 200 MB)..."):
        try:
            gdown.download(url, file_name, quiet=False)
            return os.path.exists(file_name)
        except Exception:
            return False

# Konfigurasi deterministik agar hasil prediksi konsisten
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

st.set_page_config(
    page_title="Tomat AI Diagnosis",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={}
)

st.markdown("""
<style>
    [data-testid="stSidebar"]        { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stToolbar"]        { display: none !important; }
    #MainMenu                        { visibility: hidden; }
    footer                           { visibility: hidden; }
    [data-testid="stHeader"]         { display: none !important; }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main, .main .block-container    { background: #0d0d0d !important; }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
        margin: 0 auto;
    }

    html, body { color: #e0e0e0; }
    h1, h2, h3, h4, h5, h6 { color: #f0f0f0 !important; }
    p, span, label, div     { color: #e0e0e0; }

    .hero {
        background: linear-gradient(135deg, #c0392b 0%, #7b241c 100%);
        border-radius: 18px;
        padding: 2.5rem 2rem 2.2rem;
        text-align: center;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 32px rgba(192,57,43,0.45);
    }
    .hero h1 {
        font-size: 2.4rem; font-weight: 800;
        margin: 0 0 .5rem; color: #fff !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .hero p { font-size: 1rem; color: #ffd5d0 !important; margin: 0; opacity: .9; }

    /* MODEL SELECTOR */
    .model-selector {
        background: #111;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.4rem;
    }
    .model-selector h4 {
        color: #ddd !important;
        font-size: .95rem;
        margin: 0 0 .8rem;
    }
    .model-badge-vgg {
        display: inline-block;
        background: #1a1000;
        color: #f0a500 !important;
        border: 1px solid #f0a50033;
        border-radius: 20px;
        padding: .2rem .9rem;
        font-size: .78rem;
        font-weight: 700;
    }
    .model-badge-res {
        display: inline-block;
        background: #001a10;
        color: #37d68a !important;
        border: 1px solid #37d68a33;
        border-radius: 20px;
        padding: .2rem .9rem;
        font-size: .78rem;
        font-weight: 700;
    }

    .chips { display:flex; flex-wrap:wrap; gap:.7rem; margin-bottom:1.2rem; }
    .chip  {
        background:#1a1a1a; border:1px solid #2a2a2a;
        border-radius:8px; padding:.4rem .9rem;
        font-size:.82rem; color:#888 !important;
    }
    .chip b { color:#e0e0e0 !important; }

    .section-title {
        font-size: 1rem; font-weight: 700;
        color: #ddd !important; margin: 0 0 1rem;
    }

    [data-testid="stFileUploader"] {
        background: #111 !important;
        border: 1px dashed #333 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploader"]:hover { border-color: #555 !important; }
    [data-testid="stFileUploaderDropzone"] { background: #1e1e24 !important; border-radius: 8px !important; }
    [data-testid="stFileUploader"] button {
        background: #16161c !important;
        color: #fff !important;
        border: 1px solid #333 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: #2b2b36 !important;
        border-color: #555 !important;
    }
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span { color: #888 !important; }

    /* Radio button styling */
    [data-testid="stRadio"] label { color: #ccc !important; }
    [data-testid="stRadio"] div   { gap: .5rem !important; }

    .result-card {
        background: #141414; border: 1px solid #252525;
        border-radius: 14px; padding: 1.4rem 1.5rem; margin-bottom: 1.1rem;
    }
    .result-card.danger  { border-left: 5px solid #c0392b; }
    .result-card.success { border-left: 5px solid #37a66b; }
    .result-disease {
        font-size: 1.75rem; font-weight: 800;
        color: #f2f2f2 !important; margin: .4rem 0 .2rem;
    }
    .result-conf { font-size: .92rem; color: #888 !important; margin: 0; }
    .badge-d { display:inline-block; background:#1a0a0a; color:#cc4c3c; border-radius:20px; padding:.18rem .85rem; font-size:.76rem; font-weight:700; border:1px solid #e74c3c22; }
    .badge-s { display:inline-block; background:#0a1a10; color:#52d68a; border-radius:20px; padding:.18rem .85rem; font-size:.76rem; font-weight:700; border:1px solid #27ae6022; }

    .rec-card {
        background: #1a1a1a; border: 1px solid #2a2a2a; border-left: 4px solid #4f5b6b;
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1.1rem;
    }
    .rec-card h4 { color: #f0f0f0 !important; font-size: .88rem; margin: 0 0 .4rem; }
    .rec-card p  { color: #aaa !important; font-size: .86rem; margin: 0; line-height: 1.65; }

    .top3-card { background: #141414; border: 1px solid #252525; border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 1.2rem; }
    .top3-title { font-size: .95rem; font-weight: 700; color: #ddd !important; margin-bottom: .9rem; }
    .p-row  { display:flex; align-items:center; gap:.85rem; margin-bottom:.6rem; }
    .p-medal{ width:24px; font-size:.9rem; }
    .p-name { width:200px; font-weight:600; color:#ccc !important; font-size:.86rem; }
    .p-bg   { flex:1; height:8px; background:#222; border-radius:8px; overflow:hidden; }
    .p-fill { height:8px; border-radius:8px; }
    .p-pct  { width:50px; text-align:right; font-size:.84rem; color:#888 !important; font-weight:600; }

    [data-testid="stExpander"] { background: #141414 !important; border: 1px solid #252525 !important; border-radius: 12px !important; }
    [data-testid="stExpander"] summary { color: #ccc !important; }

    hr { border-color: #222 !important; margin: 1rem 0 !important; }

    .empty { text-align: center; padding: 4rem 2rem; background: #111; border: 2px dashed #252525; border-radius: 18px; margin-top: .5rem; }
    .empty-icon { font-size: 3.5rem; margin-bottom: .8rem; }
    .empty h3   { color: #666 !important; font-size: 1.25rem; margin: 0 0 .4rem; }
    .empty p    { color: #444 !important; font-size: .9rem; margin: 0; }

    .footer-wrap { background: #0f0f0f; border: 1px solid #1e1e1e; border-radius: 12px; padding: 1.2rem; text-align: center; margin-top: 2rem; }
    .footer-wrap p   { color: #444 !important; margin: 0; font-size: .82rem; }
    .footer-wrap p+p { margin-top: .25rem; }
    .footer-wrap b   { color: #e84545 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONFIGURASI
# ============================================================
CLASS_NAMES = [
    'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
    'Septoria Leaf Spot', 'Spider Mites', 'Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Healthy'
]

DISEASE_INFO = {
    'Bacterial Spot': "Disebabkan oleh bakteri <i>Xanthomonas</i>. Gejala berupa bercak kecil, gelap, berair pada daun. Gunakan fungisida tembaga dan hindari penyiraman dari atas.",
    'Early Blight': "Disebabkan oleh jamur <i>Alternaria solani</i>. Bercak coklat dengan pola lingkaran konsentris. Buang daun terinfeksi, gunakan fungisida mankozeb.",
    'Late Blight': "Disebabkan oleh <i>Phytophthora infestans</i>. Sangat merusak dan menyebar cepat. Gunakan fungisida sistemik, jaga sirkulasi udara.",
    'Leaf Mold': "Disebabkan oleh jamur <i>Passalora fulva</i>. Bercak kuning di permukaan atas daun. Kurangi kelembapan dan perbaiki ventilasi.",
    'Septoria Leaf Spot': "Disebabkan oleh <i>Septoria lycopersici</i>. Bercak kecil berbatas jelas, pusat abu-abu. Hindari penyiraman daun, fungisida rutin.",
    'Spider Mites': "Hama tungau kecil menyebabkan daun kuning berbintik. Semprot air atau gunakan akarisida.",
    'Target Spot': "Disebabkan jamur <i>Corynespora cassiicola</i>. Bercak coklat pola sasaran. Gunakan fungisida, jaga jarak tanam.",
    'Tomato Yellow Leaf Curl Virus': "Virus dari kutu kebul (<i>Bemisia tabaci</i>). Daun mengerut dan menguning. Gunakan insektisida kendalikan vektor.",
    'Tomato Mosaic Virus': "Virus menyebar via kontak atau alat. Daun bermotif mosaik, bergelombang. Buang tanaman terinfeksi, disinfeksi alat.",
    'Healthy': "Tanaman tomat Anda <b>sehat!</b> Pertahankan praktik perawatan yang baik.",
}

# ============================================================
# LOAD METADATA AKURASI (model_info.json)
# ============================================================
MODEL_INFO = {}
if os.path.exists("model_info.json"):
    try:
        with open("model_info.json", "r", encoding="utf-8") as f:
            MODEL_INFO = json.load(f)
    except Exception:
        pass

def get_model_accuracy(file_name, default="N/A"):
    return MODEL_INFO.get(file_name, {}).get("accuracy", default)

# ============================================================
# KONFIGURASI 2 MODEL
# ============================================================
MODEL_OPTIONS = {
    "🟡 VGG-like CNN (From Scratch)": {
        "file"     : "model_tomat_vgglike_scratch.keras",
        "name"     : "VGG-like CNN",
        "accuracy" : get_model_accuracy("model_tomat_vgglike_scratch.keras", "94.80%"),
        "layers"   : "34 Layer",
        "badge"    : "<span class='model-badge-vgg'>VGG-like CNN</span>",
        "color"    : "#f0a500"
    },
    "🟢 ResNet-like CNN (From Scratch)": {
        "file"     : "model_tomat_resnetlike_scratch.keras",
        "name"     : "ResNet-like CNN",
        "accuracy" : get_model_accuracy("model_tomat_resnetlike_scratch.keras", "N/A"),
        "layers"   : "~40 Layer",
        "badge"    : "<span class='model-badge-res'>ResNet-like CNN</span>",
        "color"    : "#37d68a"
    }
}

BAR_COLORS_MONO = ['#e0e0e0', '#cccccc', '#aaaaaa']

# ============================================================
# LOAD MODEL (CACHED PER FILE)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model_by_path(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None

# ============================================================
# HERO SECTION
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🍅 Tomat AI: Smart Diagnosis</h1>
    <p>Deteksi penyakit daun tomat secara instan menggunakan CNN from scratch</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODEL SELECTOR
# ============================================================
st.markdown("<div class='model-selector'><h4>🧠 Pilih Model AI</h4>", unsafe_allow_html=True)

selected_label = st.radio(
    label="Pilih Model",
    options=list(MODEL_OPTIONS.keys()),
    horizontal=True,
    label_visibility="collapsed"
)

selected = MODEL_OPTIONS[selected_label]
st.markdown("</div>", unsafe_allow_html=True)

# Info chips model yang dipilih
st.markdown(f"""
<div class="chips">
    <div class="chip">🧠 Model: <b>{selected['name']}</b></div>
    <div class="chip">🏷️ Kelas: <b>10 Kelas</b></div>
    <div class="chip">📐 Input: <b>224 × 224 px</b></div>
    <div class="chip">📊 Akurasi Val: <b>{selected['accuracy']}</b></div>
    <div class="chip">🔢 Layer: <b>{selected['layers']}</b></div>
    <div class="chip">⚙️ Metode: <b>From Scratch</b></div>
</div>
""", unsafe_allow_html=True)

# Pastikan model tersedia (download dari Drive jika perlu)
if selected['file'] in MODEL_DRIVE_IDS:
    if not ensure_model_from_drive(selected['file']):
        st.error(f"⚠️ Gagal mengunduh **{selected['name']}** dari Google Drive. Periksa koneksi internet atau ID file Drive.")
        st.stop()

# Load model yang dipilih
with st.spinner(f"⏳ Memuat {selected['name']}..."):
    model = load_model_by_path(selected['file'])

if model is None:
    st.error(f"⚠️ Model **{selected['name']}** tidak berhasil dimuat. Pastikan file `{selected['file']}` ada di folder yang sama dengan `app.py`.")
    st.stop()
else:
    st.success(f"✅ {selected['name']} berhasil dimuat dan siap digunakan.")

# ============================================================
# FILE UPLOADER
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="Pilih file gambar (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
)

# ============================================================
# PREDIKSI
# ============================================================
if uploaded_file is not None:
    try:
        img_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"❌ Gagal membaca gambar: {e}")
        st.stop()

    st.markdown("<hr>", unsafe_allow_html=True)
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**🖼️ Gambar yang Diunggah**")
        st.image(image, use_container_width=True)
        st.caption(f"📄 `{uploaded_file.name}` · {uploaded_file.size / 1024:.1f} KB")

    with st.spinner("🔍 Menganalisis gambar…"):
        try:
            # Preprocessing IDENTIK dengan Google Colab (Keras load_img default=nearest)
            buf     = io.BytesIO(img_bytes)
            img     = load_img(buf, target_size=(224, 224), interpolation='nearest')
            img_arr = img_to_array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            predictions = model.predict(img_arr, verbose=0)
        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")
            st.stop()

    pred_idx   = int(np.argmax(predictions[0]))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(predictions[0][pred_idx]) * 100
    is_healthy = pred_class == 'Healthy'

    ctype      = "success" if is_healthy else "danger"
    emoji      = "🎉" if is_healthy else "⚠️"
    badge      = "<span class='badge-s'>✅ Sehat</span>" if is_healthy else "<span class='badge-d'>⚠️ Penyakit Terdeteksi</span>"
    info_title = "🌿 Status Tanaman" if is_healthy else "💊 Rekomendasi Penanganan"
    info_text  = DISEASE_INFO.get(pred_class, "Informasi tidak tersedia.")
    bar_color  = "#37a66b" if is_healthy else selected['color']

    with col_res:
        st.markdown(f"**🔬 Hasil Diagnosis — {selected['badge']}**", unsafe_allow_html=True)

        result_card = (
            f"<div class='result-card {ctype}'>"
            f"{badge}"
            f"<div class='result-disease'>{emoji} {pred_class}</div>"
            f"<p class='result-conf'>Tingkat Keyakinan AI: <strong>{confidence:.2f}%</strong></p>"
            f"<div style='width:100%;background:#222;border-radius:8px;height:10px;margin-top:15px;overflow:hidden;'>"
            f"<div style='width:{confidence}%;background:{bar_color};height:100%;border-radius:8px;'></div>"
            f"</div></div>"
        )
        st.markdown(result_card, unsafe_allow_html=True)

        rec_card = (
            f"<div class='rec-card'>"
            f"<h4>{info_title}</h4>"
            f"<p>{info_text}</p>"
            f"</div>"
        )
        st.markdown(rec_card, unsafe_allow_html=True)

    # TOP 3
    st.markdown("<hr>", unsafe_allow_html=True)
    top3   = np.argsort(predictions[0])[::-1][:3]
    medals = ["🥇", "🥈", "🥉"]
    rows   = ""

    for rank, idx in enumerate(top3):
        name  = CLASS_NAMES[idx]
        pct   = float(predictions[0][idx]) * 100
        color = BAR_COLORS_MONO[rank]
        rows += (
            f"<div class='p-row'>"
            f"<div class='p-medal'>{medals[rank]}</div>"
            f"<div class='p-name'>{name}</div>"
            f"<div class='p-bg'><div class='p-fill' style='width:{pct:.1f}%;background:{color};'></div></div>"
            f"<div class='p-pct'>{pct:.1f}%</div>"
            f"</div>"
        )

    st.markdown(
        f"<div class='top3-card'><div class='top3-title'>🏆 Top 3 Prediksi — {selected['name']}</div>{rows}</div>",
        unsafe_allow_html=True
    )

    with st.expander("📊 Probabilitas Semua Kelas"):
        sdata = dict(sorted(
            {CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
             for i in range(len(CLASS_NAMES))}.items(),
            key=lambda x: x[1], reverse=True
        ))
        df = pd.DataFrame(list(sdata.items()), columns=["Penyakit", "Probabilitas (%)"])
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("🏷️ Daftar Kelas Penyakit"):
        chips = "".join(
            f"<span style='display:inline-block;background:#1a1a1a;border:1px solid {'#27ae6044' if c == 'Healthy' else '#2a2a2a'};border-radius:6px;padding:.28rem .7rem;font-size:.78rem;margin:.25rem;color:{'#52d68a' if c == 'Healthy' else '#aaa'};'>{'✅ ' if c == 'Healthy' else '🔴 '}{c}</span>"
            for c in CLASS_NAMES
        )
        st.markdown(f"<div style='line-height:2.4;'>{chips}</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty">
        <div class="empty-icon">🍃</div>
        <h3>Belum ada gambar yang diunggah</h3>
        <p>Upload foto daun tomat di atas untuk memulai diagnosis AI.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-wrap">
    <p>🍅 <b>Tomat AI</b> — Sistem Deteksi Penyakit Daun Tomat berbasis CNN From Scratch</p>
    <p>© 2026 Dibuat dengan ❤️ oleh Raid &nbsp;|&nbsp; BINUS University</p>
</div>
""", unsafe_allow_html=True)