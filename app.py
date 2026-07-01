"""
SmartBin — live recycling classifier
=====================================
Take a photo of a waste item (works on an iPhone browser camera) and the app
predicts which bin it belongs in, in real time.

Run locally:        streamlit run app.py
Use from an iPhone: deploy to Streamlit Community Cloud (see README), or open the
                    "Network URL" Streamlit prints, from a phone on the same Wi-Fi.

Files required next to this script (produced by the training notebook):
    - garbage_model.keras
    - class_names.json
"""

import json
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# ----------------------------------------------------------------------------- #
# Page config
# ----------------------------------------------------------------------------- #
st.set_page_config(page_title="SmartBin", page_icon="♻️", layout="centered")

# Per-class look: bin colour + emoji. Drives the result card.
BIN_STYLE = {
    "cardboard": {"color": "#1f6feb", "emoji": "📦", "bin": "Paper & card"},
    "paper":     {"color": "#1f6feb", "emoji": "📄", "bin": "Paper & card"},
    "glass":     {"color": "#2da44e", "emoji": "🍾", "bin": "Glass"},
    "plastic":   {"color": "#d4a72c", "emoji": "🥤", "bin": "Plastics"},
    "metal":     {"color": "#8b949e", "emoji": "🥫", "bin": "Metal"},
    "trash":     {"color": "#57606a", "emoji": "🗑️", "bin": "General waste"},
}
FALLBACK = {"color": "#57606a", "emoji": "🗑️", "bin": "General waste"}

# ----------------------------------------------------------------------------- #
# Styling — one signature element: a big result card that takes the bin's colour
# ----------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .block-container { max-width: 560px; padding-top: 1.2rem; }
      h1, h2, h3 { font-family: -apple-system, "Segoe UI", system-ui, sans-serif; }
      .tagline { color: #6e7781; margin-top: -0.6rem; font-size: 0.95rem; }
      .result-card {
          border-radius: 18px; padding: 1.4rem 1.5rem; margin: 0.4rem 0 1rem 0;
          color: #fff; box-shadow: 0 6px 22px rgba(0,0,0,0.14);
      }
      .result-card .label { font-size: 2.0rem; font-weight: 800; line-height: 1.1; }
      .result-card .bin   { font-size: 1.05rem; opacity: 0.92; margin-top: 0.15rem; }
      .result-card .conf  { font-size: 0.95rem; opacity: 0.85; margin-top: 0.55rem; }
      .verdict { display:inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
                 font-weight: 700; font-size: 0.85rem; margin-top: 0.7rem;
                 background: rgba(255,255,255,0.22); }
      .bar-row { display:flex; align-items:center; gap:0.6rem; margin:0.18rem 0; }
      .bar-name { width: 84px; font-size: 0.86rem; color:#57606a; text-transform:capitalize; }
      .bar-track { flex:1; background:#eaeef2; border-radius:999px; height:10px; overflow:hidden; }
      .bar-fill { height:100%; border-radius:999px; }
      .bar-pct { width: 46px; text-align:right; font-size:0.82rem; color:#57606a; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("♻️ SmartBin")
st.markdown('<p class="tagline">Snap a photo of any item — get the right bin instantly.</p>',
            unsafe_allow_html=True)

# ----------------------------------------------------------------------------- #
# Load model + config (cached so it loads once)
# ----------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def load_assets():
    model = tf.keras.models.load_model("garbage_model_finetuned.keras")
    with open("class_names.json") as f:
        cfg = json.load(f)
    return model, cfg

try:
    model, cfg = load_assets()
except Exception as e:
    st.error(
        "Couldn't load the model. Make sure **garbage_model.keras** and "
        "**class_names.json** sit next to app.py (run the training notebook first)."
    )
    st.stop()

class_names = cfg["class_names"]
recyclable = set(cfg.get("recyclable", []))
# Read the exact input size from the model so preprocessing always matches training
_, H, W, _ = model.input_shape
H, W = int(H), int(W)


def preprocess(pil_img: Image.Image) -> np.ndarray:
    """Resize → rescale to [0,1] → add batch dim. Mirrors the notebook exactly."""
    img = ImageOps.exif_transpose(pil_img).convert("RGB").resize((W, H))
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def render_prediction(pil_img: Image.Image):
    probs = model.predict(preprocess(pil_img), verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top_i = int(order[0])
    label = class_names[top_i]
    conf = float(probs[top_i])
    style = BIN_STYLE.get(label, FALLBACK)
    is_rec = label in recyclable
    verdict = "♻️ Recyclable" if is_rec else "🗑️ General waste"

    st.image(pil_img, use_container_width=True)

    st.markdown(
        f"""
        <div class="result-card" style="background:{style['color']};">
          <div class="label">{style['emoji']} {label.capitalize()}</div>
          <div class="bin">Put it in: <b>{style['bin']}</b></div>
          <div class="conf">Confidence: {conf*100:.1f}%</div>
          <span class="verdict">{verdict}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability breakdown for the top classes
    st.caption("How sure the model is")
    for i in order[:5]:
        name = class_names[int(i)]
        pct = float(probs[int(i)]) * 100
        c = BIN_STYLE.get(name, FALLBACK)["color"]
        st.markdown(
            f"""
            <div class="bar-row">
              <div class="bar-name">{name}</div>
              <div class="bar-track"><div class="bar-fill"
                   style="width:{pct:.0f}%; background:{c};"></div></div>
              <div class="bar-pct">{pct:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if conf < 0.55:
        st.info("Low confidence — try a closer shot with the item centred and good lighting.")


# ----------------------------------------------------------------------------- #
# Input: camera first (opens the iPhone camera in Safari/Chrome), upload as backup
# ----------------------------------------------------------------------------- #
tab_cam, tab_upload = st.tabs(["📸 Camera", "🖼️ Upload"])

with tab_cam:
    shot = st.camera_input("Point at the item and take the photo")
    if shot is not None:
        render_prediction(Image.open(shot))

with tab_upload:
    up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if up is not None:
        render_prediction(Image.open(up))

st.divider()
st.caption(
    f"Model input {W}×{H}px · classes: {', '.join(class_names)}. "
    "Educational MVP — pair with human oversight before any real recycling decision."
)
