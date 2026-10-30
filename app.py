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
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

[data-testid="stAppViewContainer"] { background: #0f1117; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #2a3a5c;
    border-radius: 16px;
    padding: 32px 36px 24px;
    margin-bottom: 28px;
}
.hero h1 { font-size: 2.4rem; font-weight: 800; color: #ffffff; margin: 0 0 6px 0; letter-spacing: -0.5px; }
.hero p  { font-size: 1.0rem; color: #8b9ab8; margin: 0; }

.verdict-fake {
    background: linear-gradient(135deg, #3d0000 0%, #5c1a1a 100%);
    border: 1px solid #e53935; border-left: 5px solid #e53935;
    border-radius: 10px; padding: 18px 24px; margin: 16px 0 24px 0;
}
.verdict-real {
    background: linear-gradient(135deg, #003d00 0%, #1a5c1a 100%);
    border: 1px solid #43a047; border-left: 5px solid #43a047;
    border-radius: 10px; padding: 18px 24px; margin: 16px 0 24px 0;
}
.verdict-label { font-size: 1.8rem; font-weight: 800; letter-spacing: 1px; margin: 0 0 4px 0; }
.verdict-fake .verdict-label { color: #ff6b6b; }
.verdict-real .verdict-label { color: #69f0ae; }
.verdict-sub { font-size: 0.95rem; color: #aaaaaa; margin: 0; }

.metric-card {
    background: #1e2130; border: 1px solid #2d3250;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
}
.metric-card .label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #6b7fa3; margin-bottom: 4px; }
.metric-card .value { font-size: 1.4rem; font-weight: 700; color: #e8eaf6; font-family: 'Courier New', monospace; }

.img-label { text-align: center; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #6b7fa3; padding: 8px 0 4px 0; }

/* Region analysis card */
.region-card {
    background: #141824;
    border: 1px solid #252d45;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 24px 0 20px 0;
}
.region-title {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #4a5568; margin-bottom: 14px;
}
.region-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.8rem;
    font-weight: 700;
    margin: 0 6px 8px 0;
    letter-spacing: 0.3px;
}
.region-bar-wrap { margin: 10px 0 6px 0; }
.region-bar-label { font-size: 0.78rem; color: #8b9ab8; margin-bottom: 3px; }
.region-bar-bg { background: #1f2435; border-radius: 4px; height: 7px; overflow: hidden; }
.region-bar-fill { height: 7px; border-radius: 4px; }
.region-insight {
    font-size: 0.85rem; color: #7a8caa; line-height: 1.65;
    margin-top: 16px; border-top: 1px solid #1f2435; padding-top: 14px;
}

[data-testid="stSidebar"] { background: #13151f; border-right: 1px solid #1f2435; }
.sidebar-section { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #4a5568; margin: 20px 0 8px 0; }
.hint { font-size: 0.78rem; color: #4a5568; line-height: 1.5; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ── face region definitions ────────────────────────────────────────────────────
FACE_REGIONS = {
    "Left Eye":   (list(range(36, 42)), "#9c27b0",
                   "Eye contour blending is a frequent GAN failure — iris reflections and eyelid boundaries are hard to synthesise consistently."),
    "Right Eye":  (list(range(42, 48)), "#7b1fa2",
                   "Eye contour blending is a frequent GAN failure — iris reflections and eyelid boundaries are hard to synthesise consistently."),
    "Jaw":        (list(range(0,  17)), "#e53935",
                   "Face-swap models often leave visible seams and warping artefacts along the jaw and chin boundary."),
    "Left Brow":  (list(range(17, 22)), "#f57c00",
                   "Eyebrow skin-texture transitions can expose GAN blending boundaries."),
    "Right Brow": (list(range(22, 27)), "#f57c00",
                   "Eyebrow skin-texture transitions can expose GAN blending boundaries."),
    "Nose":       (list(range(27, 36)), "#fdd835",
                   "Nose-bridge symmetry and pore-level texture are difficult for generators to reproduce accurately."),
    "Mouth":      (list(range(48, 68)), "#e91e63",
                   "Lip boundary generation is frequently imperfect — colour gradients and teeth texture are common failure points."),
}

# ── model ──────────────────────────────────────────────────────────────────────
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
    def forward(self, x): return self.net(x).flatten(1)

class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32,64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x): return self.net(x).flatten(1)

class ThreeBranchDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.eff   = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
        self.freq  = FreqBranch()
        self.patch = PatchCNN()
        feat = self.eff.num_features + 256 + 128
        self.head  = nn.Sequential(nn.Linear(feat,512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512,1))
    def forward(self, img, dct, patches, crs):
        a = self.eff(img)
        b = self.freq(dct)
        c = self.patch(patches) * crs.unsqueeze(1)
        return self.head(torch.cat([a, b, c], dim=1)).squeeze(1)

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

# ── helpers ────────────────────────────────────────────────────────────────────
def make_dct(crop_bgr):
    ycbcr = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:,:,0].astype(np.float32)
    h, w = y.shape
    dct = np.zeros_like(y)
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
    return 0.4*0.9 + 0.3*blur + 0.3*spread, blur, spread

def make_heatmap_and_raw(crop_rgb, score):
    h, w = crop_rgb.shape[:2]
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    blur_g = cv2.GaussianBlur(gray, (0,0), sigmaX=max(h,w)*0.04)
    hmap = (gray - blur_g + 0.5).clip(0, 1)          # raw activation (0–1)
    heat = (cm.inferno(hmap)[:,:,:3] * 255).astype(np.uint8)
    alpha = 0.55 + 0.25 * score
    out = (alpha * heat + (1-alpha) * crop_rgb).clip(0,255).astype(np.uint8)
    return Image.fromarray(out), hmap                  # coloured PIL + raw float map

def get_region_activations(hmap, lms, x1, y1, x2, y2):
    """Sample heatmap activation for each facial zone using FAN landmark coords."""
    w_crop, h_crop = x2 - x1, y2 - y1
    if w_crop <= 0 or h_crop <= 0:
        return {}
    sx, sy = 224 / w_crop, 224 / h_crop
    pts_all = np.array(lms[0])
    scores = {}
    for name, (indices, color, _) in FACE_REGIONS.items():
        pts = pts_all[indices]
        cx = ((pts[:,0] - x1) * sx).astype(int)
        cy = ((pts[:,1] - y1) * sy).astype(int)
        vals = []
        for px, py in zip(cx, cy):
            r = 14
            xs, xe = max(0, px-r), min(224, px+r)
            ys, ye = max(0, py-r), min(224, py+r)
            if xs < xe and ys < ye:
                vals.append(float(hmap[ys:ye, xs:xe].mean()))
        if vals:
            scores[name] = np.mean(vals)
    return scores

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
    pts = np.array(lms[0])
    cx, cy = pts[:,0].mean(), pts[:,1].mean()
    crop_size = int(min(h,w) * 0.6)
    x1 = max(0, int(cx - crop_size//2)); x2 = min(w, x1+crop_size)
    y1 = max(0, int(cy - crop_size//2)); y2 = min(h, y1+crop_size)
    crop_bgr = img_bgr[y1:y2, x1:x2]
    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_pil = Image.fromarray(cv2.resize(crop_rgb,(224,224)))

    dct_map = make_dct(cv2.resize(crop_bgr,(224,224)))
    dct_t   = torch.tensor(dct_map).unsqueeze(0).unsqueeze(0)
    patch_t = torch.tensor(cv2.resize(crop_rgb,(64,64))/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    img_t   = TF(crop_pil).unsqueeze(0)
    crs_t   = torch.tensor([[crs_val]], dtype=torch.float32)

    T      = cfg.get("temperature", 1.0)
    thresh = cfg.get("threshold", 0.5)

    with torch.no_grad():
        prob = torch.sigmoid(model(img_t, dct_t, patch_t, crs_t) / T).item()

    verdict  = "FAKE" if prob >= thresh else "REAL"
    heatmap, hmap_raw = make_heatmap_and_raw(cv2.resize(crop_rgb,(224,224)), prob)
    region_scores = get_region_activations(hmap_raw, lms, x1, y1, x2, y2)

    return {
        "verdict":        verdict,
        "prob":           prob,
        "thresh":         thresh,
        "crs":            crs_val,
        "blur":           blur_val,
        "spread":         spread_val,
        "temp":           T,
        "heatmap":        heatmap,
        "input_img":      Image.fromarray(cv2.resize(crop_rgb,(224,224))),
        "region_scores":  region_scores,
    }, None

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">About</div>', unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#8b9ab8;font-size:0.85rem;line-height:1.6'>"
        "Three-branch neural network combining spatial (EfficientNet-B4), "
        "frequency (DCT), and landmark (FAN) features to detect AI-generated faces."
        "<br><br><b style='color:#c5cfe0'>RD-FID · MPSTME NMIMS · 2026</b></span>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="sidebar-section">Score Guide</div>', unsafe_allow_html=True)
    for name, desc in [
        ("Confidence",  "How strongly the model predicts FAKE (0–1)"),
        ("Threshold",   "Minimum confidence to classify as FAKE"),
        ("CRS",         "Crop Reliability Score — quality of face crop"),
        ("Blur",        "Image sharpness via Laplacian variance"),
        ("Spread",      "Landmark point spread across face region"),
        ("Temperature", "Calibration factor applied to model output"),
    ]:
        st.markdown(
            f"<div style='padding:6px 0;border-bottom:1px solid #1f2435'>"
            f"<div style='font-size:0.82rem;font-weight:700;color:#c5cfe0'>{name}</div>"
            f"<div class='hint'>{desc}</div></div>",
            unsafe_allow_html=True,
        )

# ── main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔍 RD-FID Deepfake Detector</h1>
  <p>Upload a face image — the model returns a real / fake verdict with a facial region analysis and attention heatmap.</p>
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

    # ── verdict ───────────────────────────────────────────────────────────────
    is_fake  = result["verdict"] == "FAKE"
    cls_name = "verdict-fake" if is_fake else "verdict-real"
    icon     = "🔴" if is_fake else "🟢"
    pct      = f"{result['prob']*100:.1f}%"

    st.markdown(f"""
    <div class="{cls_name}">
      <div class="verdict-label">{icon} {result['verdict']} &nbsp;·&nbsp; {pct}</div>
      <div class="verdict-sub">Confidence the image is AI-generated &nbsp;|&nbsp; Threshold: {result['thresh']:.3f}</div>
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

    # ── region analysis ───────────────────────────────────────────────────────
    rscores = result["region_scores"]
    if rscores:
        # normalise scores to 0–1 relative to max
        max_s = max(rscores.values())
        norm  = {k: v/max_s for k, v in rscores.items()}

        # top 3 most active regions
        top3  = sorted(rscores, key=rscores.get, reverse=True)[:3]

        # merge duplicate descriptions (Left/Right Eye, Left/Right Brow)
        seen_desc = set()
        insight_lines = []
        for r in top3:
            desc = FACE_REGIONS[r][2]
            if desc not in seen_desc:
                insight_lines.append(f"<b style='color:#c5cfe0'>{r}:</b> {desc}")
                seen_desc.add(desc)

        # build badge HTML
        badges = ""
        for r in top3:
            color = FACE_REGIONS[r][1]
            badges += (
                f"<span class='region-badge' "
                f"style='background:{color}22;color:{color};border:1px solid {color}55'>"
                f"{r}</span>"
            )

        # build mini bar chart HTML
        bars_html = ""
        for r, s in sorted(rscores.items(), key=lambda x: x[1], reverse=True):
            color = FACE_REGIONS[r][1]
            pct_bar = int(norm[r] * 100)
            bars_html += (
                f"<div class='region-bar-wrap'>"
                f"<div class='region-bar-label'>{r} &nbsp;<span style='color:#3d4a6b;font-size:0.72rem'>{rscores[r]:.3f}</span></div>"
                f"<div class='region-bar-bg'>"
                f"<div class='region-bar-fill' style='width:{pct_bar}%;background:{color}'></div>"
                f"</div></div>"
            )

        insight_html = "<br>".join(insight_lines)
        heatmap_note = (
            "The heatmap highlights regions with high-frequency texture activity. "
            "In AI-generated faces, these areas often correspond to synthesis boundaries "
            "where the generator fails to maintain natural continuity."
            if is_fake else
            "Even in real faces some regions show natural texture variation. "
            "No strong synthesis artefacts were detected by the model."
        )

        st.markdown(f"""
        <div class="region-card">
          <div class="region-title">Facial Region Activity</div>
          <div style="margin-bottom:14px">{badges}</div>
          {bars_html}
          <div class="region-insight">
            <b style="color:#6b7fa3;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.8px">
              Why these regions matter</b><br><br>
            {insight_html}<br><br>
            <span style="color:#4a5568">{heatmap_note}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── metric cards ──────────────────────────────────────────────────────────
    for row in [
        [("Confidence", f"{result['prob']:.3f}"),
         ("CRS Score",  f"{result['crs']:.3f}"),
         ("Blur Score", f"{result['blur']:.3f}")],
        [("Spread",      f"{result['spread']:.3f}"),
         ("Temperature", f"{result['temp']:.4f}"),
         ("Threshold",   f"{result['thresh']:.3f}")],
    ]:
        cols = st.columns(3)
        for col, (label, val) in zip(cols, row):
            with col:
                st.markdown(
                    f"<div class='metric-card'><div class='label'>{label}</div>"
                    f"<div class='value'>{val}</div></div>",
                    unsafe_allow_html=True,
                )
else:
    st.markdown(
        "<div style='text-align:center;padding:60px 20px;color:#3d4a6b;font-size:0.95rem'>"
        "Drop a face image above to begin analysis."
        "</div>",
        unsafe_allow_html=True,
    )
