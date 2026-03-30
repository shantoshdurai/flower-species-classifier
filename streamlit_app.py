import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import io

# Page Config
st.set_page_config(
    page_title="Flower Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Auto-download model from Hugging Face if on deployment
try:
    from huggingface_hub import hf_hub_download
    if not os.path.exists("my_flower_cnn.h5"):
        hf_hub_download(repo_id="Santoshp123/flower-species-classifier", filename="my_flower_cnn.h5", local_dir=".")
except:
    pass

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #F7F7F5; }

/* limit max width for readability */
.block-container { max-width: 960px !important; padding: 1.5rem 2rem !important; }

.hero {
    background: #FFD600;
    border: 3px solid #000;
    box-shadow: 6px 6px 0 #000;
    padding: 1.4rem 2rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-text h1 { font-size: 2rem; font-weight: 800; margin: 0; line-height: 1.1; color: #000; }
.hero-text p  { font-size: 0.88rem; font-weight: 500; margin: 6px 0 0; color: #333; }
.badge   {
    display: inline-block; background: #000; color: #FFD600;
    padding: 3px 10px; font-size: 0.68rem; font-weight: 700;
    margin-right: 6px; margin-top: 10px; text-transform: uppercase; letter-spacing: 0.5px;
}

/* Stats */
.stat-card {
    background: #fff; border: 2.5px solid #000;
    box-shadow: 4px 4px 0 #000; padding: 0.8rem 1rem; text-align: center;
}
.stat-card .val { font-size: 1.4rem; font-weight: 800; color: #000; }
.stat-card .lbl { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; color: #666; letter-spacing: 1px; margin-top: 2px; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #fff !important;
    border: 2.5px solid #000 !important;
    box-shadow: 5px 5px 0 #000;
    padding: 0.5rem !important;
    margin-bottom: 1.2rem;
}
[data-testid="stFileUploader"] section { background: #fff !important; }
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small { color: #333 !important; }
[data-testid="stFileUploader"] button {
    background: #000 !important;
    color: #FFD600 !important;
    border: 2px solid #000 !important;
    border-radius: 0 !important;
    font-weight: 700 !important;
}

/* Section heading */
.sec { font-size: 0.68rem; font-weight: 900; text-transform: uppercase;
       letter-spacing: 2px; color: #000; border-bottom: 2.5px solid #000;
       padding-bottom: 4px; margin-bottom: 12px; }


/* Prediction cards */
.pred {
    display: flex; justify-content: space-between; align-items: center;
    background: #fff; border: 2.5px solid #000; box-shadow: 3px 3px 0 #000;
    padding: 10px 14px; margin-bottom: 6px; font-weight: 700; font-size: 0.9rem;
    color: #000 !important;
}
.pred.top { background: #FFD600; }
.pred .pct { font-size: 0.85rem; }
.pbar-bg { height: 6px; background: #e5e5e5; border: 1.5px solid #000; margin-top: -4px; margin-bottom: 10px; }

.alert-ok  { background: #B5FFD1; border: 2.5px solid #000; box-shadow: 3px 3px 0 #000; padding: 10px 14px; font-weight: 700; font-size: 0.85rem; margin-bottom: 12px; color: #000 !important; }
.alert-low { background: #FFD6CC; border: 2.5px solid #000; box-shadow: 3px 3px 0 #000; padding: 10px 14px; font-weight: 700; font-size: 0.85rem; margin-bottom: 12px; color: #000 !important; }

.footer { text-align: center; padding: 1.5rem 0 0.5rem; font-size: 0.72rem; font-weight: 600; color: #aaa; border-top: 2px solid #ddd; margin-top: 2rem; text-transform: uppercase; letter-spacing: 1px; }

/* Idle flower grid */
.fl-grid-item {
    background: #fff; border: 2.5px solid #000; box-shadow: 3px 3px 0 #000;
    padding: 0.8rem 0.5rem; text-align: center; margin-bottom: 0.5rem;
}
.fl-grid-item .fl-emoji { font-size: 1.8rem; line-height: 1; }
.fl-grid-item .fl-name  { font-size: 0.72rem; font-weight: 700; margin-top: 6px; color: #000; }
</style>
""", unsafe_allow_html=True)

# Constants
FLOWER_SPECIES = [
    'Tulips', 'Bougainvillea', 'Daisies', 'Garden Roses', 'Gardenias',
    'Hibiscus', 'Hydrangeas', 'Lilies', 'Orchids', 'Peonies'
]
EMOJIS = {
    'Tulips': '🌷', 'Bougainvillea': '🌺', 'Daisies': '🌼',
    'Garden Roses': '🌹', 'Gardenias': '🤍', 'Hibiscus': '🌺',
    'Hydrangeas': '💜', 'Lilies': '🤍', 'Orchids': '🌸', 'Peonies': '🌸'
}

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return tf.keras.models.load_model('my_flower_cnn.h5')
    except Exception as e:
        st.session_state['model_error'] = str(e)
        return None

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="font-size:3rem;line-height:1">🌸</div>
  <div class="hero-text">
    <h1>Flower Species Classifier</h1>
    <p>Upload a photo and find out what flower it is.</p>
    <div style="margin-top:10px">
      <span class="badge">MobileNetV2</span>
      <span class="badge">TensorFlow</span>
      <span class="badge">10 Species</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

model = load_model()
if model is None:
    err = st.session_state.get('model_error', 'Unknown error')
    st.error(f"❌ Could not load model: {err}")
    st.stop()

uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

# ── Idle state ───────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown('<div class="sec">Supported Flowers</div>', unsafe_allow_html=True)
    row1 = st.columns(5)
    row2 = st.columns(5)
    for i, flower in enumerate(FLOWER_SPECIES):
        col = row1[i] if i < 5 else row2[i - 5]
        col.markdown(f"""
        <div class="fl-grid-item">
          <div class="fl-emoji">{EMOJIS.get(flower,'🌸')}</div>
          <div class="fl-name">{flower}</div>
        </div>""", unsafe_allow_html=True)

else:
    # ── Run inference ─────────────────────────────────────────────────────────
    img = Image.open(uploaded_file).convert('RGB')
    img_arr = np.expand_dims(np.array(img.resize((224, 224))), axis=0).astype('float32')

    # Convert original image to bytes for display (avoids PIL rendering issues)
    img_buf = io.BytesIO()
    img.save(img_buf, format='PNG')
    img_buf.seek(0)

    with st.spinner("Classifying..."):
        preds = model.predict(img_arr, verbose=0)[0]

    top_idx   = np.argsort(preds)[::-1]
    best      = FLOWER_SPECIES[top_idx[0]]
    best_conf = float(preds[top_idx[0]])
    rel_color = "#27AE60" if best_conf > 0.7 else ("#E67E22" if best_conf > 0.5 else "#E74C3C")
    rel_label = "High" if best_conf > 0.7 else ("Medium" if best_conf > 0.5 else "Low")

    # ── Stat row ──────────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4, gap="small")
    s1.markdown(f'<div class="stat-card"><div class="val">{best_conf*100:.1f}%</div><div class="lbl">Confidence</div></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="stat-card"><div class="val">{EMOJIS.get(best,"🌸")} {best}</div><div class="lbl">Best Match</div></div>', unsafe_allow_html=True)
    s3.markdown(f'<div class="stat-card"><div class="val">10</div><div class="lbl">Species</div></div>', unsafe_allow_html=True)
    s4.markdown(f'<div class="stat-card"><div class="val" style="color:{rel_color}">{rel_label}</div><div class="lbl">Reliability</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

    # ── 3-column: image | predictions | chart ────────────────────────────────
    col_img, col_pred, col_chart = st.columns([1, 1.1, 1.3], gap="medium")

    with col_img:
        st.markdown('<div class="sec">Image</div>', unsafe_allow_html=True)
        st.image(img_buf, use_container_width=True)

    with col_pred:
        st.markdown('<div class="sec">Predictions</div>', unsafe_allow_html=True)
        if best_conf < 0.5:
            st.markdown('<div class="alert-low">⚠️ Low confidence — might not match any trained species.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-ok">✅ {best} — {best_conf*100:.1f}%</div>', unsafe_allow_html=True)

        for i in range(5):
            name = FLOWER_SPECIES[top_idx[i]]
            pct  = float(preds[top_idx[i]]) * 100
            st.markdown(f"""
            <div class="pred {'top' if i==0 else ''}">
              <span>{EMOJIS.get(name,'🌸')} {name}</span>
              <span class="pct">{pct:.1f}%</span>
            </div>
            <div class="pbar-bg">
              <div style="height:100%;width:{pct:.1f}%;background:{'#FFD600' if i==0 else '#555'}"></div>
            </div>""", unsafe_allow_html=True)

    with col_chart:
        st.markdown('<div class="sec">All Species</div>', unsafe_allow_html=True)
        chart_rows = ""
        for j in range(len(top_idx)):
            name = FLOWER_SPECIES[top_idx[j]]
            val  = float(preds[top_idx[j]])
            pct  = val * 100
            bar_color = '#FFD600' if j == 0 else '#DDD'
            chart_rows += f"""
            <div style="margin-bottom:7px">
              <div style="display:flex;justify-content:space-between;font-size:0.75rem;font-weight:700;margin-bottom:3px">
                <span>{EMOJIS.get(name,'🌸')} {name}</span>
                <span style="color:#555">{pct:.1f}%</span>
              </div>
              <div style="height:8px;background:#e5e5e5;border:1.5px solid #000">
                <div style="height:100%;width:{pct:.2f}%;background:{bar_color};border-right:{'1.5px solid #000' if pct > 1 else 'none'}"></div>
              </div>
            </div>"""
        st.markdown(chart_rows, unsafe_allow_html=True)

st.markdown('<div class="footer">Made with ❤️ · Streamlit · TensorFlow · MobileNetV2</div>', unsafe_allow_html=True)
