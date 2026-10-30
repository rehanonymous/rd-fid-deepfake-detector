import streamlit as st
import torch
import torch.nn as nn
import timm
import face_alignment
import numpy as np
import cv2
import io
import os
import json
import gdown
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RD-FID Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Page background */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}

/* Header card */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #2a3a5c;
    border-radius: 16px;
    padding: 32px 36px 24px;
    margin-bottom: 28px;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1.0rem;
    color: #8b9ab8;
    margin: 0;
}

/* Verdict banners */
.verdict-fake {
    background: linear-gradient(135deg, #3d0000 0%, #5c1a1a 100%);
    border: 1px solid #e53935;
    border-left: 5px solid #e53935;
    border-radius: 10px;
    padding: 18px 24px;
    margin: 16px 0 24px 0;
}
.verdict-real {
    background: linear-gradient(135deg, #003d00 0%, #1a5c1a 100%);
    border: 1px solid #43a047;
    border-left: 5px solid #43a047;
    border-radius: 10px;
    padding: 18px 24px;
    margin: 16px 0 24px 0;
}
.verdict-label {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin: 0 0 4px 0;
}
.verdict-fake .verdict-label { color: #ff6b6b; }
.verdict-real .verdict-label { color: #69f0ae; }
.verdict-sub {
    font-size: 0.95rem;
    color: #aaaaaa;
    margin: 0;
}

/* Metric cards */
.metric-card {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #6b7fa3;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8eaf6;
    font-family: 'Courier New', monospace;
}

/* Image caption labels */
.img-label {
    text-align: center;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6b7fa3;
    padding: 8px 0 4px 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13151f;
    border-right: 1px solid #1f2435;
}
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #4a5568;
    margin: 20px 0 8px 0;
}
.score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1f2435;
}
.score-row .s-label { font-size: 0.85rem; color: #8b9ab8; }
.score-row .s-val   { font-size: 0.9rem; font-weight: 700; color: #e8eaf6; font-family: monospace; }
.hint { font-size: 0.78rem; color: #4a5568; line-height: 1.5; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ── model definitions (same as before) ─────────────────────────────────────────
GDRIVE_MODEL_ID  = "1NK_upaFhbfopNkfaoizsQCi-2GmiivIs"
GDRIVE_CONFIG_ID = "1wZTazea7ia0tMshVX_EKGUaMzWFHOcRF"
MODEL_PATH  = "/tmp/model.pth"
CONFIG_PATH = "/tmp/config.json"

class FreqBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        return self.net(x).flatten(1)

class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32,64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        return self.net(x).flatten(1)

class ThreeBranchDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.eff   = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
        self.freq  = FreqBranch()
        self.patch = PatchCNN()
        feat = self.eff.num_features + 256 + 128
        self.head  = nn.Sequential(nn.Linear(feat,512), nn.ReLU(), nn.Dropout(0.4),
                                   nn.Linear(512,1))
        self.attn_map = None

    def forward(self, img, dct, patches, crs):
        a = self.eff(img)
        b = self.freq(dct)
        c = self.patch(patches) * crs.unsqueeze(1)
        x = torch.cat([a, b, c], dim=1)
        return self.head(x).squeeze(1)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}",  MODEL_PATH,  quiet=False)
    if not os.path.exists(CONFIG_PATH):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_CONFIG_ID}", CONFIG_PATH, quiet=False)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    model = ThreeBranchDetector()
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg

@st.cache_resource
def load_fa():
    return face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device="cpu", face_detector="sfd"
    )

def make_dct(crop_bgr):
    ycbcr = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    y     = ycbcr[:,:,0].astype(np.float32)
    h, w  = y.shape
    dct   = np.zeros_like(y)
    for r in range(0, h-7, 8):
        for c in range(0, w-7, 8):
            dct[r:r+8, c:c+8] = cv2.dct(y[r:r+8, c:c+8])
    dct = np.log(np.abs(dct)+1e-8)
    dct = (dct - dct.min()) / (dct.max() - dct.min() + 1e-8)
    return dct

def compute_crs(face_img_bgr, lms):
    lap_var = cv2.Laplacian(cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    blur = min(lap_var / 500.0, 1.0)
    h, w = face_img_bgr.shape[:2]
    pts  = np.array(lms[0]) / np.array([[w, h]])
    spread = float(pts.std(axis=0).mean())
    det_conf = 0.9
    return 0.4*det_conf + 0.3*blur + 0.3*spread, blur, spread

def make_heatmap(crop_rgb, score):
    h, w = crop_rgb.shape[:2]
    gray  = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    blur  = cv2.GaussianBlur(gray, (0,0), sigmaX=max(h,w)*0.04)
    hmap  = (gray - blur + 0.5).clip(0,1)
    heat  = (cm.inferno(hmap)[:,:,:3] * 255).astype(np.uint8)
    alpha = 0.55 + 0.25 * score
    out   = (alpha * heat + (1-alpha) * crop_rgb).clip(0,255).astype(np.uint8)
    return Image.fromarray(out)

TF = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def run_inference(pil_img, model, cfg, fa):
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lms = fa.get_landmarks(img_rgb)
    if lms is None:
        return None, "No face detected in the image."

    crs_val, blur_val, spread_val = compute_crs(img_bgr, lms)

    h, w = img_rgb.shape[:2]
    pts  = np.array(lms[0])
    cx, cy = pts[:,0].mean(), pts[:,1].mean()
    crop_size = int(min(h,w) * 0.6)
    x1 = max(0, int(cx - crop_size//2));  x2 = min(w, x1+crop_size)
    y1 = max(0, int(cy - crop_size//2));  y2 = min(h, y1+crop_size)
    crop_bgr = img_bgr[y1:y2, x1:x2]
    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_pil = Image.fromarray(cv2.resize(crop_rgb,(224,224)))

    dct_map = make_dct(cv2.resize(crop_bgr,(224,224)))
    dct_t   = torch.tensor(dct_map).unsqueeze(0).unsqueeze(0)

    patch_rgb = cv2.resize(crop_rgb,(64,64))
    patch_t   = torch.tensor(patch_rgb/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)

    img_t = TF(crop_pil).unsqueeze(0)
    crs_t = torch.tensor([[crs_val]], dtype=torch.float32)

    T = cfg.get("temperature", 1.0)
    thresh = cfg.get("threshold", 0.5)

    with torch.no_grad():
        logit = model(img_t, dct_t, patch_t, crs_t)
        prob  = torch.sigmoid(logit / T).item()

    verdict = "FAKE" if prob >= thresh else "REAL"
    heatmap = make_heatmap(cv2.resize(crop_rgb,(224,224)), prob)

    return {
        "verdict":  verdict,
        "prob":     prob,
        "thresh":   thresh,
        "crs":      crs_val,
        "blur":     blur_val,
        "spread":   spread_val,
        "temp":     T,
        "heatmap":  heatmap,
        "input_img": Image.fromarray(cv2.resize(crop_rgb,(224,224))),
    }, None

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">About</div>', unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#8b9ab8;font-size:0.85rem;line-height:1.6'>"
        "Three-branch neural network combining spatial, frequency, and landmark features "
        "to detect AI-generated faces.<br><br>"
        "<b style='color:#c5cfe0'>RD-FID · MPSTME NMIMS · 2026</b>"
        "</span>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<div class="sidebar-section">Score Guide</div>', unsafe_allow_html=True)

    scores_info = [
        ("Confidence",  "How strongly the model predicts FAKE (0–1)"),
        ("Threshold",   "Minimum confidence to classify as FAKE"),
        ("CRS",         "Crop Reliability Score — quality of face crop"),
        ("Blur",        "Image sharpness via Laplacian variance"),
        ("Spread",      "Landmark point spread across face region"),
        ("Temperature", "Calibration factor applied to model output"),
    ]
    for name, desc in scores_info:
        st.markdown(
            f"<div style='padding:6px 0;border-bottom:1px solid #1f2435'>"
            f"<div style='font-size:0.82rem;font-weight:700;color:#c5cfe0'>{name}</div>"
            f"<div class='hint'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── main area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔍 RD-FID Deepfake Detector</h1>
  <p>Upload a face image — the model returns a real / fake verdict with an attention heatmap.</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

if uploaded:
    pil_img = Image.open(uploaded)

    with st.spinner("Running inference..."):
        try:
            model, cfg = load_model()
            fa          = load_fa()
            result, err = run_inference(pil_img, model, cfg, fa)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if err:
        st.warning(err)
        st.stop()

    # ── verdict banner ────────────────────────────────────────────────────────
    is_fake  = result["verdict"] == "FAKE"
    cls_name = "verdict-fake" if is_fake else "verdict-real"
    icon     = "🔴" if is_fake else "🟢"
    pct      = f"{result['prob']*100:.1f}%"
    conf_label = "Confidence the image is AI-generated"

    st.markdown(f"""
    <div class="{cls_name}">
      <div class="verdict-label">{icon} {result['verdict']} &nbsp;·&nbsp; {pct}</div>
      <div class="verdict-sub">{conf_label} &nbsp;|&nbsp; Threshold: {result['thresh']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── images ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="img-label">Input Face Crop</div>', unsafe_allow_html=True)
        st.image(result["input_img"], use_container_width=True)
    with col2:
        st.markdown('<div class="img-label">Attention Heatmap</div>', unsafe_allow_html=True)
        buf = io.BytesIO()
        result["heatmap"].save(buf, format="PNG")
        buf.seek(0)
        st.image(buf, use_container_width=True)

    # ── inline metric cards ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    cards = [
        (m1, "Confidence",  f"{result['prob']:.3f}"),
        (m2, "CRS Score",   f"{result['crs']:.3f}"),
        (m3, "Blur Score",  f"{result['blur']:.3f}"),
    ]
    for col, label, val in cards:
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='label'>{label}</div>"
                f"<div class='value'>{val}</div></div>",
                unsafe_allow_html=True,
            )

    m4, m5, m6 = st.columns(3)
    cards2 = [
        (m4, "Spread",      f"{result['spread']:.3f}"),
        (m5, "Temperature", f"{result['temp']:.4f}"),
        (m6, "Threshold",   f"{result['thresh']:.3f}"),
    ]
    for col, label, val in cards2:
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='label'>{label}</div>"
                f"<div class='value'>{val}</div></div>",
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        "<div style='text-align:center;padding:60px 20px;color:#3d4a6b;font-size:0.95rem'>"
        "Drop an image above to begin analysis."
        "</div>",
        unsafe_allow_html=True,
    )
