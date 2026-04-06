import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array
import pandas as pd

# ─────────────────────────────────────────────
# Page Config — HARUS PERTAMA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tomat AI Diagnosis",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={}
)

# ─────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Sidebar & header bawaan hilang ── */
    [data-testid="stSidebar"]        { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stToolbar"]        { display: none !important; }
    #MainMenu                        { visibility: hidden; }
    footer                           { visibility: hidden; }
    [data-testid="stHeader"]         { display: none !important; } /* Sembunyikan header total agar rapi */

    /* ── Dark base ── */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main, .main .block-container    { background: #0d0d0d !important; }

    /* ── Block container ── */
    .block-container {
        padding-top: 1.5rem !important; /* Disesuaikan agar hero pas di atas */
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
        margin: 0 auto;
    }

    /* ── Teks global ── */
    html, body { color: #e0e0e0; }
    h1, h2, h3, h4, h5, h6 { color: #f0f0f0 !important; }
    p, span, label, div     { color: #e0e0e0; }

    /* ── Hero ── */
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

    /* ── Info chips ── */
    .chips { display:flex; flex-wrap:wrap; gap:.7rem; margin-bottom:1.2rem; }
    .chip  {
        background:#1a1a1a; border:1px solid #2a2a2a;
        border-radius:8px; padding:.4rem .9rem;
        font-size:.82rem; color:#888 !important;
    }
    .chip b { color:#e84545 !important; }

    /* ── Section card ── */
    .section-title {
        font-size: 1rem; font-weight: 700;
        color: #ddd !important; margin: 0 0 1rem;
    }

    /* ── File uploader styling ── */
    [data-testid="stFileUploader"] {
        background: #1a1a1a !important;
        border: 1.5px dashed #333 !important;
        border-radius: 10px !important;
        padding: .6rem 1rem !important;
    }
    [data-testid="stFileUploader"]:hover { border-color: #e84545 !important; }
    [data-testid="stFileUploaderDropzone"] { background: transparent !important; }
    [data-testid="stFileUploader"] button {
        background: #e84545 !important; color: #fff !important;
        border: none !important; border-radius: 7px !important; font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button:hover { background: #c0392b !important; }
    [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: #777 !important; }

    /* ── Result card ── */
    .result-card {
        background: #141414; border: 1px solid #252525;
        border-radius: 14px; padding: 1.4rem 1.5rem; margin-bottom: 1.1rem;
    }
    .result-card.danger  { border-left: 5px solid #e74c3c; }
    .result-card.success { border-left: 5px solid #27ae60; }
    .result-disease {
        font-size: 1.75rem; font-weight: 800;
        color: #f2f2f2 !important; margin: .4rem 0 .2rem;
    }
    .result-conf { font-size: .92rem; color: #888 !important; margin: 0; }
    .badge-d { display:inline-block; background:#2e0f0f; color:#ff7070; border-radius:20px; padding:.18rem .85rem; font-size:.76rem; font-weight:700; border:1px solid #e74c3c44; }
    .badge-s { display:inline-block; background:#0a2419; color:#52d68a; border-radius:20px; padding:.18rem .85rem; font-size:.76rem; font-weight:700; border:1px solid #27ae6044; }

    /* ── Recommendation card ── */
    .rec-card {
        background: #161500; border: 1px solid #3d3410; border-left: 4px solid #e6a817;
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1.1rem;
    }
    .rec-card h4 { color: #e6c140 !important; font-size: .88rem; margin: 0 0 .4rem; }
    .rec-card p  { color: #c9b96a !important; font-size: .86rem; margin: 0; line-height: 1.65; }

    /* ── Top-3 ── */
    .top3-card { background: #141414; border: 1px solid #252525; border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 1.2rem; }
    .top3-title { font-size: .95rem; font-weight: 700; color: #ddd !important; margin-bottom: .9rem; }
    .p-row  { display:flex; align-items:center; gap:.85rem; margin-bottom:.6rem; }
    .p-medal{ width:24px; font-size:.9rem; }
    .p-name { width:200px; font-weight:600; color:#ccc !important; font-size:.86rem; }
    .p-bg   { flex:1; height:8px; background:#222; border-radius:8px; overflow:hidden; }
    .p-fill { height:8px; border-radius:8px; }
    .p-pct  { width:50px; text-align:right; font-size:.84rem; color:#888 !important; font-weight:600; }

    /* ── Expander ── */
    [data-testid="stExpander"] { background: #141414 !important; border: 1px solid #252525 !important; border-radius: 12px !important; }
    [data-testid="stExpander"] summary { color: #ccc !important; }

    /* ── Divider ── */
    hr { border-color: #222 !important; margin: 1rem 0 !important; }

    /* ── Empty state ── */
    .empty { text-align: center; padding: 4rem 2rem; background: #111; border: 2px dashed #252525; border-radius: 18px; margin-top: .5rem; }
    .empty-icon { font-size: 3.5rem; margin-bottom: .8rem; }
    .empty h3   { color: #666 !important; font-size: 1.25rem; margin: 0 0 .4rem; }
    .empty p    { color: #444 !important; font-size: .9rem; margin: 0; }

    /* ── Footer ── */
    .footer-wrap { background: #0f0f0f; border: 1px solid #1e1e1e; border-radius: 12px; padding: 1.2rem; text-align: center; margin-top: 2rem; }
    .footer-wrap p   { color: #444 !important; margin: 0; font-size: .82rem; }
    .footer-wrap p+p { margin-top: .25rem; }
    .footer-wrap b   { color: #e84545 !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
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
BAR_COLORS = ['#e84545', '#ff7043', '#ffa726']

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Memuat model AI…")
def load_model():
    try:
        return tf.keras.models.load_model('model_tomat_v2_pintar.keras')
    except Exception as e:
        return None

model = load_model()

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🍅 Tomat AI: Smart Diagnosis</h1>
    <p>Deteksi penyakit daun tomat secara instan menggunakan kecerdasan buatan berbasis MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Info chips
# ─────────────────────────────────────────────
st.markdown("""
<div class="chips">
    <div class="chip">🧠 Model: <b>MobileNetV2</b></div>
    <div class="chip">🏷️ Kelas: <b>10 Kelas</b></div>
    <div class="chip">📐 Input: <b>224 × 224 px</b></div>
    <div class="chip">📊 Format: <b>.keras</b></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model status
# ─────────────────────────────────────────────
if model is None:
    st.error("⚠️ Model tidak berhasil dimuat. Pastikan file `model_tomat_v2_pintar.keras` tersedia di folder yang sama.")
    st.stop()
else:
    st.success("✅ Model AI berhasil dimuat dan siap digunakan.")

# ─────────────────────────────────────────────
# Upload Container
# ─────────────────────────────────────────────
with st.container():
    st.markdown("<div class='section-title'>📂 Upload Gambar Daun Tomat</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Pilih gambar daun tomat (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload foto daun tomat yang ingin didiagnosis",
    )

# ─────────────────────────────────────────────
# Prediction Flow
# ─────────────────────────────────────────────
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
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
            img_arr = img_to_array(image.resize((224, 224))) / 255.0
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
    bar_color  = "#27ae60" if is_healthy else "#e74c3c"

    with col_res:
        st.markdown("**🔬 Hasil Diagnosis**")

        # Perbaikan: Gabungkan f-string tanpa spasi indentasi berlebih
        result_card = (
            f"<div class='result-card {ctype}'>"
            f"{badge}"
            f"<div class='result-disease'>{emoji} {pred_class}</div>"
            f"<p class='result-conf'>Tingkat Keyakinan AI: <strong>{confidence:.2f}%</strong></p>"
            f"<div style='width: 100%; background-color: #222; border-radius: 8px; height: 10px; margin-top: 15px; overflow: hidden;'>"
            f"<div style='width: {confidence}%; background-color: {bar_color}; height: 100%; border-radius: 8px;'></div>"
            f"</div></div>"
        )
        st.markdown(result_card, unsafe_allow_html=True)

        # Rec card
        rec_card = (
            f"<div class='rec-card'>"
            f"<h4>{info_title}</h4>"
            f"<p>{info_text}</p>"
            f"</div>"
        )
        st.markdown(rec_card, unsafe_allow_html=True)

    # ── Top-3 ──
    st.markdown("<hr>", unsafe_allow_html=True)
    top3 = np.argsort(predictions[0])[::-1][:3]
    medals = ["🥇", "🥈", "🥉"]

    rows = ""
    for rank, idx in enumerate(top3):
        name  = CLASS_NAMES[idx]
        pct   = float(predictions[0][idx]) * 100
        color = BAR_COLORS[rank]
        
        # Perbaikan: Gabungkan f-string per komponen
        rows += (
            f"<div class='p-row'>"
            f"<div class='p-medal'>{medals[rank]}</div>"
            f"<div class='p-name'>{name}</div>"
            f"<div class='p-bg'><div class='p-fill' style='width: {pct:.1f}%; background: {color};'></div></div>"
            f"<div class='p-pct'>{pct:.1f}%</div>"
            f"</div>"
        )

    st.markdown(f"<div class='top3-card'><div class='top3-title'>🏆 Top 3 Prediksi AI</div>{rows}</div>", unsafe_allow_html=True)
    # ── Expanders ──
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

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-wrap">
    <p>🍅 <b>Tomat AI</b> — Sistem Deteksi Penyakit Daun Tomat berbasis Deep Learning</p>
    <p>© 2026 Dibuat dengan ❤️ oleh Raid &nbsp;|&nbsp; BINUS University</p>
</div>
""", unsafe_allow_html=True)