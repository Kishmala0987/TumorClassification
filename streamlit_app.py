"""
streamlit_app.py
Run with:  streamlit run streamlit_app.py
Requires:  backend.py  +  model.h5  in the same directory.
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan · MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d14;
    color: #c8c8e8;
}
[data-testid="stSidebar"] { background:#10101c; border-right:1px solid #1e1e34; }
[data-testid="stSidebar"] * { color:#c8c8e8 !important; }

.neuro-header { text-align:center; padding:2rem 0 1.2rem; }
.neuro-header h1 {
    font-family:'Space Mono',monospace; font-size:2.5rem; letter-spacing:.06em;
    background:linear-gradient(135deg,#a78bfa,#60a5fa,#34d399);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;
}
.neuro-header p { color:#555577; font-size:.82rem; margin-top:.4rem;
                  letter-spacing:.1em; text-transform:uppercase; }

.verdict-tumor  { background:linear-gradient(135deg,#1a0a0a,#2a0f0f);
                  border:1px solid #ff6b6b55; border-left:4px solid #ff6b6b;
                  border-radius:10px; padding:1rem 1.5rem; margin-bottom:1rem; }
.verdict-notumor{ background:linear-gradient(135deg,#0a1a0f,#0f2a18);
                  border:1px solid #6bff9e55; border-left:4px solid #6bff9e;
                  border-radius:10px; padding:1rem 1.5rem; margin-bottom:1rem; }
.verdict-title  { font-family:'Space Mono',monospace; font-size:1.25rem;
                  font-weight:700; margin:0 0 .3rem; }
.verdict-sub    { font-size:.8rem; color:#888899; letter-spacing:.05em; }

.metric-row { display:flex; gap:.8rem; margin-bottom:1rem; }
.metric-card { flex:1; background:#12121e; border:1px solid #1e1e34;
               border-radius:10px; padding:.8rem 1rem; text-align:center; }
.metric-card .val { font-family:'Space Mono',monospace; font-size:1.4rem;
                    font-weight:700; color:#a78bfa; }
.metric-card .lbl { font-size:.7rem; color:#444466; text-transform:uppercase;
                    letter-spacing:.1em; margin-top:.2rem; }

.section-title { font-family:'Space Mono',monospace; font-size:.72rem;
                 letter-spacing:.14em; text-transform:uppercase; color:#444466;
                 border-bottom:1px solid #1e1e34; padding-bottom:.4rem;
                 margin:1.2rem 0 .8rem; }

.prob-table { width:100%; border-collapse:collapse; font-size:.86rem; }
.prob-table th { color:#555577; padding:.4rem .6rem; text-align:left;
                 border-bottom:1px solid #1e1e34; font-weight:600; }
.prob-table td { padding:.45rem .6rem; border-bottom:1px solid #12121e; }
.prob-table tr.predicted td { background:#1a1e2a; }

.stButton > button {
    background:linear-gradient(135deg,#7c3aed,#2563eb);
    color:white; border:none; border-radius:8px;
    font-family:'Space Mono',monospace; font-size:.82rem;
    letter-spacing:.05em; padding:.55rem 1.2rem;
    width:100%; transition:opacity .2s;
}
.stButton > button:hover { opacity:.82; }

[data-testid="stFileUploader"] {
    background:#12121e; border:1.5px dashed #2e2e55; border-radius:12px; padding:1.2rem;
}
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS  (mirrored from backend for display)
# ─────────────────────────────────────────────
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
RISK_MAP     = {"glioma":"High ⚠", "meningioma":"Moderate", "pituitary":"Moderate", "notumor":"None ✓"}
RISK_COLORS  = {"High ⚠":"#ff6b6b", "Moderate":"#ffd166", "None ✓":"#6bff9e"}

# ─────────────────────────────────────────────
# CACHED MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_backend():
    from backend import get_model, predict as _predict
    model = get_model()
    return model, _predict

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="neuro-header">
  <h1>🧠 NeuroScan</h1>
  <p>MRI Brain Tumor Classifier · LIME + Grad-CAM++  Explainability</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conv_layer = st.selectbox(
        "Grad-CAM layer",
        ["block5_conv3", "block5_conv2", "block5_conv1"],
        index=0,
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.74rem; color:#444466; line-height:1.8'>
    <b style='color:#666688'>Classes</b><br>
    🔴 Glioma · High risk<br>
    🟡 Meningioma · Moderate<br>
    🟡 Pituitary · Moderate<br>
    🟢 No Tumor · None<br><br>
    <b style='color:#666688'>Model</b><br>
    VGG16 fine-tuned<br>
    Input: 128×128 RGB
    </div>
    """, unsafe_allow_html=True)
    use_lime = st.toggle("Enable LIME (slower)", value=True)

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an MRI image here  ·  JPG / PNG / BMP",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded is None:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div style='text-align:center; padding:5rem 0; color:#222244'>
          <div style='font-size:4rem'>🧠</div>
          <div style='font-family:Space Mono,monospace; font-size:.85rem;
                      letter-spacing:.1em; margin-top:.8rem'>
            UPLOAD AN MRI TO BEGIN
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# PREVIEW + RUN BUTTON
# ─────────────────────────────────────────────
image = Image.open(uploaded).convert("RGB")
w, h  = image.size

col_img, col_meta = st.columns([1, 2])
with col_img:
    st.markdown("<div class='section-title'>Input Image</div>", unsafe_allow_html=True)
    st.image(image, use_container_width=True)

with col_meta:
    st.markdown("<div class='section-title'>File Info</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='metric-row'>
      <div class='metric-card'><div class='val'>{w}×{h}</div><div class='lbl'>Resolution</div></div>
      <div class='metric-card'><div class='val'>{uploaded.size//1024} KB</div><div class='lbl'>File Size</div></div>
      <div class='metric-card'><div class='val'>RGB</div><div class='lbl'>Mode</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"`{uploaded.name}`")
    run = st.button("🔬  Analyse MRI", use_container_width=True)

if not run:
    st.stop()

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
try:
    model, predict_fn = load_backend()
except Exception as e:
    st.error(f"❌ Could not load model: {e}\n\nMake sure `model.h5` is in the same directory as `streamlit_app.py`.")
    st.stop()

with st.spinner("Running inference + Grad-CAM++ + LIME…"):
    result, confidence, cam_img, lime_img, fused_img, pred_class, preds = predict_fn(
        image,
        model,
        last_conv_layer=conv_layer,
        run_lime_explain=use_lime
    )

class_name = CLASS_LABELS[pred_class]
risk_label = RISK_MAP.get(class_name, "—")
risk_color = RISK_COLORS.get(risk_label, "#cccccc")
is_tumor   = class_name != "notumor"

# ─────────────────────────────────────────────
# VERDICT BANNER
# ─────────────────────────────────────────────
verdict_cls  = "verdict-tumor" if is_tumor else "verdict-notumor"
verdict_icon = "🔴" if is_tumor else "🟢"

st.markdown(f"""
<div class='{verdict_cls}'>
  <div class='verdict-title'>{verdict_icon}  {result}</div>
  <div class='verdict-sub'>
    Confidence: <b style='color:#e0e0f0'>{confidence*100:.1f}%</b>
    &nbsp;·&nbsp;
    Risk: <b style='color:{risk_color}'>{risk_label}</b>
    &nbsp;·&nbsp;
    Model: VGG16 fine-tuned
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
st.markdown(f"""
<div class='metric-row'>
  <div class='metric-card'>
    <div class='val' style='color:{"#ff6b6b" if is_tumor else "#6bff9e"}'>{class_name.capitalize()}</div>
    <div class='lbl'>Predicted Class</div>
  </div>
  <div class='metric-card'>
    <div class='val'>{confidence*100:.1f}%</div>
    <div class='lbl'>Confidence</div>
  </div>
  <div class='metric-card'>
    <div class='val' style='color:{risk_color}'>{risk_label}</div>
    <div class='lbl'>Risk Level</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDE-BY-SIDE: ORIGINAL  |  GRAD-CAM | LIME
# ─────────────────────────────────────────────
st.markdown("<div class='section-title'>Explainability</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.image(image, caption="Original MRI", use_container_width=True)

with c2:
    st.image(
        cam_img,
        caption=f"Grad-CAM++ | Focus Regions ({class_name}, {confidence * 100:.1f}%)",
        use_container_width=True
    )
with c3:
    if lime_img is not None:
        st.image(lime_img, caption="LIME Superpixels", use_container_width=True)
    else:
        st.info("LIME disabled")
if fused_img is not None:
    st.markdown("<div class='section-title'>Combined Explanation</div>", unsafe_allow_html=True)
    resized = cv2.resize(fused_img, (400, 400))  # width, height
    st.image(
        resized,
        caption="Combined Explanation (Model + Local + Spatial)",
        use_container_width=True,
    )
# ─────────────────────────────────────────────
# CLASS PROBABILITIES TABLE
# ─────────────────────────────────────────────
st.markdown("<div class='section-title'>Class Probabilities</div>", unsafe_allow_html=True)

order = np.argsort(preds)[::-1]

tbl_html = """
<table class='prob-table'>
  <tr><th>#</th><th>Class</th><th>Probability</th><th>Risk</th></tr>
"""
for rank, idx in enumerate(order, 1):
    lbl   = CLASS_LABELS[idx]
    prob  = preds[idx] * 100
    risk  = RISK_MAP.get(lbl, "—")
    rc    = RISK_COLORS.get(risk, "#cccccc")
    row_cls = "predicted" if idx == pred_class else ""
    conf_bar_width = int(prob)
    tbl_html += f"""
  <tr class='{row_cls}'>
    <td style='color:#555577'>{rank}</td>
    <td style='color:#c8c8e8; font-weight:{"600" if idx==pred_class else "400"}'>{lbl.capitalize()}</td>
    <td>
      <div style='display:flex; align-items:center; gap:.6rem'>
        <div style='flex:1; height:6px; background:#1a1a2e; border-radius:3px; overflow:hidden'>
          <div style='width:{conf_bar_width}%; height:100%;
                      background:{"#ff6b6b" if idx==pred_class else "#4dabf7"};
                      border-radius:3px'></div>
        </div>
        <span style='font-family:Space Mono,monospace; font-size:.8rem; color:#a78bfa;
                     min-width:4.5rem; text-align:right'>{prob:.2f}%</span>
      </div>
    </td>
    <td><span style='color:{rc}; font-weight:600'>{risk}</span></td>
  </tr>
"""
tbl_html += "</table>"
st.markdown(tbl_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem; color:#222244;
            font-size:.72rem; font-family:Space Mono,monospace'>
  NeuroScan · For research use only · Not a clinical diagnostic tool
</div>
""", unsafe_allow_html=True)