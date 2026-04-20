import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Page Config
st.set_page_config(
    page_title="Flower Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

*, html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}
/* Restore Material Symbols font so icons render as glyphs, not raw text */
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons {
    font-family: 'Material Symbols Rounded' !important;
}
/* Safe text color inherited by the app without overriding hidden elements */
.stApp { background: #F7F7F5; color: #000; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown li {
    color: #000;
}
#MainMenu, footer, header { visibility: hidden; }

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

/* Hide the radio widget label ("Input mode") — keep only the options */
[data-testid="stRadio"] > div:first-child > label {
    display: none !important;
}

/* Radio options as toggle buttons */
[data-testid="stRadio"] [role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 8px !important;
    margin-bottom: 0.8rem !important;
}
[data-testid="stRadio"] [role="radiogroup"] label {
    background: #fff !important;
    border: 2px solid #000 !important;
    box-shadow: 2px 2px 0 #000 !important;
    padding: 9px 14px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    cursor: pointer !important;
    border-radius: 0 !important;
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
}
[data-testid="stRadio"] [role="radiogroup"] label > div,
[data-testid="stRadio"] [role="radiogroup"] label span,
[data-testid="stRadio"] [role="radiogroup"] label p {
    color: #000 !important;
    font-size: 0.85rem !important;
    white-space: nowrap !important;
}
/* Hide ONLY the dot indicator, not the whole option */
[data-testid="stRadio"] [data-baseweb="radio"] {
    display: none !important;
}
[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}

/* File uploader outer box */
[data-testid="stFileUploader"] {
    background: #fff !important;
    border: 2.5px solid #000 !important;
    box-shadow: 5px 5px 0 #000 !important;
    padding: 0.5rem !important;
    margin-bottom: 1.2rem;
}
[data-testid="stFileUploader"] section {
    background: #fff !important;
}
/* Dropzone layout */
[data-testid="stFileUploaderDropzone"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 1rem !important;
    padding: 0.6rem 0.8rem !important;
    background: #fff !important;
    border: none !important;
    box-shadow: none !important;
}
/* Hide native file input (positioned behind button; causes double-text) */
[data-testid="stFileUploaderDropzoneInput"] {
    display: none !important;
}
/* The Upload button */
[data-testid="stFileUploaderDropzone"] button {
    background: #FFD600 !important;
    color: #000 !important;
    border: 2px solid #000 !important;
    border-radius: 0 !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    box-shadow: 3px 3px 0 #000 !important;
    padding: 8px 18px !important;
    flex-shrink: 0 !important;
    white-space: nowrap !important;
}
/* Hide icon glyph — renders as raw text "upload" because Space Grotesk overrides the font */
[data-testid="stFileUploaderDropzone"] button span.material-symbols-rounded,
[data-testid="stFileUploaderDropzone"] button span[class*="material"] {
    display: none !important;
}
/* File size / type hint text */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p:not(button p) {
    color: #555 !important;
    font-size: 0.75rem !important;
}

/* Mobile */
@media (max-width: 768px) {
    .block-container { padding: 0.8rem !important; }
    .hero { flex-direction: column; text-align: center; gap: 0.5rem; padding: 0.8rem 1rem; }
    .hero-text h1 { font-size: 1.25rem; }
    .hero-text p { font-size: 0.78rem; }
    .stat-card .val { font-size: 1rem; }
    [data-testid="stFileUploader"] { padding: 0.25rem !important; }
    [data-testid="stFileUploaderDropzone"] {
        flex-direction: row !important;
        padding: 0.5rem 0.7rem !important;
        gap: 0.7rem !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        width: auto !important;
        padding: 8px 14px !important;
        font-size: 0.85rem !important;
        flex-shrink: 0 !important;
    }
    .flower-grid-container {
        grid-template-columns: repeat(2, 1fr) !important;
    }
}

/* Desktop — hide toggle, only show file uploader */
@media (min-width: 769px) {
    [data-testid="stRadio"] { display: none !important; }
}

.flower-grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
}
div.stButton > button {
    background: #FFD600 !important;
    color: #000 !important;
    border: 3px solid #000 !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0 #000 !important;
    font-weight: 800 !important;
    font-size: 1.2rem !important;
    padding: 0.5rem 2rem !important;
    margin-top: 1rem !important;
    width: 100% !important;
}
div.stButton > button:hover {
    background: #000 !important;
    color: #FFD600 !important;
    box-shadow: none !important;
    transform: translate(2px, 2px) !important;
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

# Hide only the material icon span (not the label text) inside the upload button
import streamlit.components.v1 as components
components.html("""
<script>
(function(){
  function fix(){
    var doc = window.parent.document;
    doc.querySelectorAll('[data-testid="stFileUploaderDropzone"] button').forEach(function(btn){
      btn.querySelectorAll('span').forEach(function(s){
        var t = (s.innerText || s.textContent || '').trim();
        // Icon names are all-lowercase single words like "upload", "photo_camera"
        // Label text like "Upload" or "Upload Image" has capitals or spaces — keep those
        if (t.length > 0 && t === t.toLowerCase() && t.indexOf(' ') === -1) {
          s.style.display = 'none';
        }
      });
      // Fallback: if button is empty after hiding, write the label directly
      var p = btn.querySelector('p');
      if (p && p.innerText.trim() === '') {
        p.style.cssText = 'font-size:0.9rem;font-weight:700;color:#000';
        p.textContent = 'Upload Image';
      }
    });
  }
  fix();
  new MutationObserver(fix).observe(window.parent.document.body,{childList:true,subtree:true});
})();
</script>
""", height=0)

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
    return tf.keras.models.load_model('my_flower_cnn.h5', compile=False)

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

input_mode = st.radio("Input mode", ["📁 Upload Image", "📸 Take Photo"], horizontal=True, label_visibility="collapsed")

if input_mode == "📁 Upload Image":
    image_source = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
else:
    image_source = st.camera_input("Take a photo", label_visibility="collapsed")

# ── Idle state ───────────────────────────────────────────────────────────────
if image_source is None:
    st.markdown('<div class="sec">Supported Flowers</div>', unsafe_allow_html=True)
    grid_html = '<div class="flower-grid-container">'
    for flower in FLOWER_SPECIES:
        grid_html += f"""
        <div class="fl-grid-item">
          <div class="fl-emoji">{EMOJIS.get(flower,'🌸')}</div>
          <div class="fl-name">{flower}</div>
        </div>"""
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

else:
    import io
    img_bytes = image_source.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown('<div class="sec">Photo Preview</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    with col2:
        st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)
        st.markdown("### Ready to check?")
        st.markdown("Review your photo on the left. If it looks good, click the button below to proceed.")
        proceed = st.button("🔍 Classify Flower")
        
    if proceed:
        img_arr = np.expand_dims(np.array(img.resize((224, 224))), axis=0).astype('float32')

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

        # ── 2-column: predictions | chart ────────────────────────────────
        col_pred, col_chart = st.columns([1, 1.2], gap="medium")

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
