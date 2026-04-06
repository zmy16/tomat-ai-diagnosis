import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array

# ─────────────────────────────────────────────
# Page Config  (HARUS dipanggil PERTAMA)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tomat AI Diagnosis",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Dark Base ── */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main .block-container {
        background: #0d0d0d !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111111 0%, #1a1a1a 100%) !important;
        border-right: 1px solid #2a2a2a;
    }
    /* Streamlit native text ke light */
    html, body, [class*="css"], p, span, label, div {
        color: #e0e0e0;
    }
    h1, h2, h3, h4 { color: #f0f0f0 !important; }

    /* Streamlit widget overrides */
    [data-testid="stFileUploader"] {
        background: #1c1c1c;
        border: 2px dashed #3a3a3a;
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #e84545;
    }
    .stAlert, [data-testid="stAlert"] {
        background: #1c1c1c !important;
        border-radius: 10px;
    }
    /* Expander dark */
    [data-testid="stExpander"] {
        background: #1c1c1c;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
    }
    /* Dataframe dark */
    [data-testid="stDataFrame"] {
        background: #1c1c1c;
    }
    /* Divider */
    hr { border-color: #2a2a2a !important; }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #c0392b 0%, #922b21 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(192,57,43,0.4);
        border: 1px solid #e74c3c33;
    }
    .hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0 0 .5rem; color: white !important; }
    .hero p  { font-size: 1.1rem; opacity: .88; margin: 0; color: #ffd0cc !important; }

    /* ── Upload Section ── */
    .upload-section {
        background: #161616;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 1.5rem 2rem 0.5rem;
        margin-bottom: 1.5rem;
    }
    .upload-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e0e0e0 !important;
        margin-bottom: .8rem;
    }

    /* ── Result Card ── */
    .result-card {
        background: #161616;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 1.6rem;
        margin-bottom: 1.5rem;
    }
    .result-card.danger  { border-left: 5px solid #e74c3c; }
    .result-card.success { border-left: 5px solid #27ae60; }
    .result-label { font-size: .8rem; font-weight: 600; letter-spacing: 1px;
                    text-transform: uppercase; color: #777; margin-bottom: .5rem; }
    .result-disease { font-size: 1.9rem; font-weight: 800; color: #f0f0f0 !important; margin-bottom: .3rem; }
    .result-conf    { font-size: 1rem; color: #aaa !important; }

    .badge-danger  { display:inline-block; background:#3d1515; color:#ff6b6b;
                     border-radius:20px; padding:.2rem .9rem; font-size:.82rem; font-weight:700;
                     border: 1px solid #e74c3c55; }
    .badge-success { display:inline-block; background:#0d2e1a; color:#58d68d;
                     border-radius:20px; padding:.2rem .9rem; font-size:.82rem; font-weight:700;
                     border: 1px solid #27ae6055; }

    /* ── Info Card ── */
    .info-card {
        background: #1a1a0a;
        border: 1px solid #4a3f1a;
        border-left: 4px solid #f39c12;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.5rem;
    }
    .info-card h4 { color: #f0c040 !important; margin: 0 0 .5rem; font-size: .95rem; }
    .info-card p  { color: #c8b87a !important; margin: 0; font-size: .92rem; line-height: 1.6; }

    /* ── Top-3 Bar ── */
    .top3-wrapper {
        background: #161616;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.5rem;
    }
    .top3-title { font-size: 1rem; font-weight: 700; color: #e0e0e0 !important;
                  margin-bottom: 1.2rem; }
    .pred-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: .75rem;
    }
    .pred-rank { width: 24px; font-weight: 800; color: #555; font-size: .9rem; }
    .pred-name { width: 230px; font-weight: 600; color: #d0d0d0 !important; font-size: .92rem; }
    .pred-bar-bg { flex: 1; height: 9px; background: #2a2a2a; border-radius: 10px; overflow: hidden; }
    .pred-bar    { height: 9px; border-radius: 10px; }
    .pred-pct   { width: 56px; text-align: right; font-size: .88rem;
                  color: #aaa !important; font-weight: 600; }

    /* ── Sidebar Cards ── */
    .sb-card {
        background: #1c1c1c;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        margin-bottom: .9rem;
    }
    .sb-card h4 { margin: 0 0 .6rem; font-size: .95rem; color: #ff8a80 !important; }
    .sb-card p  { margin: .28rem 0; font-size: .85rem; color: #b0b0b0 !important; }
    .sb-divider { border: none; border-top: 1px solid #2a2a2a; margin: .9rem 0; }

    /* ── Empty State ── */
    .empty-state {
        text-align: center;
        padding: 3.5rem 2rem;
        background: #141414;
        border: 2px dashed #2a2a2a;
        border-radius: 20px;
        margin-top: 1rem;
    }
    .empty-state h3 { color: #666 !important; margin: .6rem 0 .3rem; }
    .empty-state p  { color: #444 !important; }

    /* ── Footer ── */
    .footer {
        background: #111111;
        border: 1px solid #222;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: #555 !important;
        font-size: .85rem;
        margin-top: 2.5rem;
    }
    .footer strong { color: #e84545 !important; }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stToolbar"] { visibility: hidden; }
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
    'Bacterial Spot':
        "Disebabkan oleh bakteri <i>Xanthomonas</i>. Gejalanya berupa bercak kecil, gelap, dan berair pada daun. "
        "Gunakan fungisida tembaga dan hindari penyiraman dari atas.",
    'Early Blight':
        "Disebabkan oleh jamur <i>Alternaria solani</i>. Muncul bercak coklat dengan pola lingkaran konsentris. "
        "Buang daun yang terinfeksi dan gunakan fungisida berbahan mankozeb.",
    'Late Blight':
        "Disebabkan oleh <i>Phytophthora infestans</i>. Sangat merusak dan menyebar cepat. "
        "Gunakan fungisida sistemik dan pastikan sirkulasi udara yang baik.",
    'Leaf Mold':
        "Disebabkan oleh jamur <i>Passalora fulva</i>. Muncul bercak kuning di permukaan atas daun. "
        "Kurangi kelembapan dan perbaiki ventilasi.",
    'Septoria Leaf Spot':
        "Disebabkan oleh <i>Septoria lycopersici</i>. Bercak kecil berbatas jelas dengan pusat abu-abu. "
        "Hindari penyiraman daun dan gunakan fungisida secara rutin.",
    'Spider Mites':
        "Hama tungau kecil yang menyebabkan daun menjadi kuning dan berbintik. "
        "Semprot dengan air atau gunakan akarisida yang sesuai.",
    'Target Spot':
        "Disebabkan oleh jamur <i>Corynespora cassiicola</i>. Bercak coklat dengan pola sasaran. "
        "Gunakan fungisida dan jaga jarak tanam agar sirkulasi udara baik.",
    'Tomato Yellow Leaf Curl Virus':
        "Virus yang disebarkan oleh kutu kebul (<i>Bemisia tabaci</i>). Daun mengerut dan menguning. "
        "Gunakan insektisida untuk mengendalikan vektornya.",
    'Tomato Mosaic Virus':
        "Virus yang menyebar melalui kontak atau alat pertanian. Daun bermotif mosaik dan bergelombang. "
        "Tidak ada obat; buang tanaman yang terinfeksi dan disinfeksi alat tanam.",
    'Healthy':
        "Tanaman tomat Anda terlihat <strong>sehat</strong>! Pertahankan praktik perawatan yang baik.",
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
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <span style='font-size:3rem;'>🍅</span>
        <h2 style='color:white; margin:.3rem 0 0;'>Tomat AI</h2>
        <p style='color:#ff8a80; font-size:.85rem; margin:0;'>Smart Disease Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='sb-divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class='sb-card'>
        <h4>⚙️ Info Model</h4>
        <p>🧠 <strong>Arsitektur :</strong> MobileNetV2</p>
        <p>🏷️ <strong>Jumlah Kelas :</strong> 10</p>
        <p>📐 <strong>Input Size :</strong> 224 × 224 px</p>
        <p>📊 <strong>Format :</strong> .keras</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='sb-card'>
        <h4>📋 Kelas Penyakit</h4>
    """ + "".join([f"<p>{'✅' if c == 'Healthy' else '🔴'} {c}</p>" for c in CLASS_NAMES]) + """
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='sb-card'>
        <h4>📖 Cara Penggunaan</h4>
        <p>1️⃣ Upload foto daun tomat</p>
        <p>2️⃣ Tunggu analisis AI</p>
        <p>3️⃣ Baca hasil & rekomendasi</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🍅 Tomat AI: Smart Diagnosis</h1>
    <p>Deteksi penyakit daun tomat secara instan menggunakan kecerdasan buatan berbasis MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Status Banner
# ─────────────────────────────────────────────
if model is None:
    st.error("⚠️ Model tidak berhasil dimuat. Pastikan file `model_tomat_v2_pintar.keras` tersedia di direktori yang sama.")
    st.stop()
else:
    st.success("✅ Model AI berhasil dimuat dan siap digunakan.")

# ─────────────────────────────────────────────
# Upload Section
# ─────────────────────────────────────────────
st.markdown("<div class='upload-section'><div class='upload-title'>📂 Upload Gambar Daun Tomat</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="Pilih file gambar (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload foto daun tomat yang ingin didiagnosis",
)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Prediction Flow
# ─────────────────────────────────────────────
if uploaded_file is not None:

    # ── Open & convert image safely ──
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Gagal membaca gambar: {e}")
        st.stop()

    # ── Layout: gambar kiri | hasil kanan ──
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### 🖼️ Gambar yang Diunggah")
        st.image(image, use_container_width=True)
        st.caption(f"Nama file: `{uploaded_file.name}` | Ukuran: {uploaded_file.size / 1024:.1f} KB")

    # ── Predict ──
    with st.spinner("🔍 Menganalisis gambar…"):
        try:
            img_resized  = image.resize((224, 224))
            img_array    = img_to_array(img_resized) / 255.0
            img_array    = np.expand_dims(img_array, axis=0)
            predictions  = model.predict(img_array, verbose=0)
        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")
            st.stop()

    predicted_idx   = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(predictions[0][predicted_idx]) * 100
    is_healthy      = predicted_class == 'Healthy'

    with col_res:
        st.markdown("#### 🔬 Hasil Diagnosis")

        card_type  = "success" if is_healthy else "danger"
        emoji      = "🎉" if is_healthy else "⚠️"
        badge_html = (f"<span class='badge-success'>Sehat</span>"
                      if is_healthy else f"<span class='badge-danger'>Penyakit Terdeteksi</span>")

        st.markdown(f"""
        <div class='result-card {card_type}'>
            <div class='result-label'>{badge_html}</div>
            <div class='result-disease'>{emoji} {predicted_class}</div>
            <div class='result-conf'>Tingkat Keyakinan AI: <strong>{confidence:.2f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence progress bar
        st.progress(confidence / 100)

        # Disease info box
        info_text = DISEASE_INFO.get(predicted_class, "Informasi tidak tersedia.")
        st.markdown(f"""
        <div class='info-card'>
            <h4>{'🌿 Status Tanaman' if is_healthy else '💊 Rekomendasi Penanganan'}</h4>
            <p>{info_text}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Top-3 Predictions ──
    st.markdown("---")
    top3_indices = np.argsort(predictions[0])[::-1][:3]
    medals = ["🥇", "🥈", "🥉"]

    # Bangun seluruh HTML sebagai satu string — hindari f-string bersarang
    top3_html_parts = [
        "<div class='top3-wrapper'>",
        "<div class='top3-title'>🏆 Top 3 Prediksi AI</div>",
    ]
    for rank, idx in enumerate(top3_indices):
        name  = CLASS_NAMES[idx]
        pct   = float(predictions[0][idx]) * 100
        color = BAR_COLORS[rank]
        top3_html_parts.append(
            "<div class='pred-row'>"
            + "<div class='pred-rank'>" + medals[rank] + "</div>"
            + "<div class='pred-name'>" + name + "</div>"
            + "<div class='pred-bar-bg'>"
            + "<div class='pred-bar' style='width:" + str(round(pct, 1)) + "%; background:" + color + ";'></div>"
            + "</div>"
            + "<div class='pred-pct'>" + str(round(pct, 1)) + "%</div>"
            + "</div>"
        )
    top3_html_parts.append("</div>")

    st.markdown("\n".join(top3_html_parts), unsafe_allow_html=True)

    # ── All probabilities expander ──
    with st.expander("📊 Lihat Probabilitas Semua Kelas"):
        chart_data = {CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
                      for i in range(len(CLASS_NAMES))}
        sorted_data = dict(sorted(chart_data.items(), key=lambda x: x[1], reverse=True))
        import pandas as pd
        df = pd.DataFrame(list(sorted_data.items()), columns=["Penyakit", "Probabilitas (%)"])
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    # ── Empty State ──
    st.markdown("""
    <div class='empty-state'>
        <div style='font-size:4rem;'>🍃</div>
        <h3>Belum ada gambar yang diunggah</h3>
        <p>Upload foto daun tomat di atas untuk memulai diagnosis AI.</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🍅 <strong>Tomat AI</strong> — Sistem Deteksi Penyakit Daun Tomat berbasis Deep Learning<br>
    <span style='font-size:.8rem;'>© 2026 Dibuat dengan ❤️ &nbsp;|&nbsp; BINUS University</span>
</div>
""", unsafe_allow_html=True)