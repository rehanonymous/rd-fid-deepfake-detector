import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import math
import json
import io
import os
import gdown
import timm
import matplotlib.pyplot as plt
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RD-FID Deepfake Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Download model from Google Drive ──────────────────────────────────────────
MODEL_PATH  = "best_final.pt"
CONFIG_PATH = "config.json"

GDRIVE_MODEL_ID  = "1NK_upaFhbfopNkfaoizsQCi-2GmiivIs"
GDRIVE_CONFIG_ID = "1wZTazea7ia0tMshVX_EKGUaMzWFHOcRF"

@st.cache_resource
def download_weights():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}",
                           MODEL_PATH, quiet=False)
    if not os.path.exists(CONFIG_PATH):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_CONFIG_ID}",
                       CONFIG_PATH, quiet=False)

download_weights()

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

T_CAL       = cfg['temperature']
THRESHOLD   = cfg['threshold']
IMG_SIZE    = cfg['img_size']
PATCH_SIZE  = 48
DROPOUT     = 0.4
DEVICE      = torch.device('cpu')

# ── Model architecture ────────────────────────────────────────────────────────
def rgb_to_ycbcr(x):
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    y  =  0.299*r  + 0.587*g  + 0.114*b
    cb = -0.16875*r - 0.33126*g + 0.500*b  + 0.5
    cr =  0.500*r  - 0.41869*g - 0.08131*b + 0.5
    return torch.cat([y, cb, cr], dim=1)

class BlockDCT(nn.Module):
    def __init__(self, block_size=8):
        super().__init__()
        self.bs = block_size
        N = block_size
        D = torch.zeros(N, N)
        for k in range(N):
            for n in range(N):
                D[k, n] = (math.sqrt(1.0/N) if k == 0
                           else math.sqrt(2.0/N) * math.cos(math.pi*k*(2*n+1)/(2*N)))
        self.register_buffer('D', D)

    def forward(self, x):
        B, C, H, W = x.shape
        bs = self.bs
        ph = (bs - H % bs) % bs
        pw = (bs - W % bs) % bs
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        _, _, H2, W2 = x.shape
        x   = x.unfold(2, bs, bs).unfold(3, bs, bs)
        dct = self.D @ x @ self.D.t()
        dct = torch.log1p(dct.abs())
        dct = dct.permute(0,1,2,4,3,5).contiguous().view(B, C, H2, W2)
        return dct[:, :, :H, :W]

class FreqBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.dct = BlockDCT()
        self.net = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3,   32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, out_dim),
        )
    def forward(self, x_raw):
        return self.net(self.dct(rgb_to_ycbcr(x_raw)))

class SpatialNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        self.freq     = FreqBranch(out_dim=256)
        self.head     = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(self.backbone.num_features + 256, 1),
        )
    def forward(self, x_norm, x_raw):
        a = self.backbone(x_norm)
        b = self.freq(x_raw)
        return self.head(torch.cat([a, b], dim=1)).squeeze(1)

class PatchCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(256, out_dim)
    def forward(self, x):
        return self.proj(self.net(x))

class ThreeBranchDetector(nn.Module):
    def __init__(self, stage1_backbone, stage1_freq, patch_cnn):
        super().__init__()
        self.branch_a = stage1_backbone
        self.branch_b = stage1_freq
        self.branch_c = patch_cnn
        feat_a, feat_b, feat_c = 1792, 256, 128
        self.seam_attn = nn.Sequential(
            nn.Conv2d(feat_a, 64, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1), nn.Sigmoid(),
        )
        self.head_a = nn.Linear(feat_a, 1)
        self.head_b = nn.Linear(feat_b, 1)
        self.head_c = nn.Linear(feat_c, 1)
        self.fusion = nn.Sequential(
            nn.Linear(feat_a + feat_b + feat_c, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x_norm, x_raw, patches, blur_scores, landmark_spreads):
        feat_map = self.branch_a.forward_features(x_norm)
        attn_map = self.seam_attn(feat_map)
        feat_a   = (feat_map * attn_map).mean(dim=[2, 3])
        logit_a  = self.head_a(feat_a).squeeze(1)
        with torch.no_grad():
            conf_d = torch.sigmoid(logit_a.detach())
        feat_b  = self.branch_b(x_raw)
        logit_b = self.head_b(feat_b).squeeze(1)
        blur    = blur_scores.to(x_norm.device)
        spread  = landmark_spreads.to(x_norm.device)
        crs     = (0.4 * conf_d + 0.3 * blur + 0.3 * spread).clamp(0.0, 1.0)
        patch_feats = []
        for i in range(3):
            pf = self.branch_c(patches[:, i].to(x_norm.device))
            patch_feats.append(pf)
        feat_c  = torch.stack(patch_feats, dim=1).mean(dim=1)
        logit_c = self.head_c(feat_c).squeeze(1)
        fused   = torch.cat([feat_a, feat_b, feat_c * crs.unsqueeze(1)], dim=1)
        logit   = self.fusion(fused).squeeze(1)
        return dict(logit=logit, logit_a=logit_a, logit_b=logit_b,
                    logit_c=logit_c, attn_map=attn_map, crs=crs)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    s1 = SpatialNetV3()
    pc = PatchCNN(out_dim=128)
    m  = ThreeBranchDetector(s1.backbone, s1.freq, pc).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    m.load_state_dict(ckpt)
    m.eval()
    return m

model = load_model()

# ── Helpers ───────────────────────────────────────────────────────────────────
import face_alignment as _fa_lib

LEFT_EYE_IDX  = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))
MOUTH_IDX     = list(range(48, 68))
PAD_FRAC      = 0.30

@st.cache_resource
def get_fa():
    return _fa_lib.FaceAlignment(
        _fa_lib.LandmarksType.TWO_D, device='cpu', flip_input=False)

def extract_patch(img_np, lm, indices):
    h, w  = img_np.shape[:2]
    pts   = lm[indices]
    x1,y1 = pts.min(axis=0)
    x2,y2 = pts.max(axis=0)
    pw = (x2-x1)*PAD_FRAC; ph = (y2-y1)*PAD_FRAC
    x1 = max(0,int(x1-pw)); y1 = max(0,int(y1-ph))
    x2 = min(w,int(x2+pw)); y2 = min(h,int(y2+ph))
    if x2<=x1 or y2<=y1:
        return np.zeros((PATCH_SIZE,PATCH_SIZE,3),dtype=np.uint8)
    return cv2.resize(img_np[y1:y2,x1:x2],(PATCH_SIZE,PATCH_SIZE))

def get_patches(pil_img):
    fa  = get_fa()
    rgb = np.array(pil_img.convert('RGB'))
    res = fa.get_landmarks(rgb)
    if res is None or len(res)==0:
        return torch.zeros(3,3,PATCH_SIZE,PATCH_SIZE), 0.0
    lm      = res[0]
    h,w     = rgb.shape[:2]
    lm_norm = lm/np.array([w,h])
    spread  = float(min(lm_norm.std()/0.20,1.0))
    to_t    = T.ToTensor()
    patches = torch.stack([
        to_t(Image.fromarray(extract_patch(rgb,lm,LEFT_EYE_IDX))),
        to_t(Image.fromarray(extract_patch(rgb,lm,RIGHT_EYE_IDX))),
        to_t(Image.fromarray(extract_patch(rgb,lm,MOUTH_IDX))),
    ])
    return patches, spread

def blur_score(pil_img, max_var=500.0):
    gray = np.array(pil_img.convert('L'))
    return float(min(cv2.Laplacian(gray,cv2.CV_64F).var()/max_var,1.0))

_norm = T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

def preprocess(pil_img):
    img  = pil_img.resize((IMG_SIZE,IMG_SIZE),Image.BILINEAR)
    raw  = T.ToTensor()(img)
    norm = _norm(raw.clone())
    return norm, raw, img

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(pil_img):
    norm_t, _, img_r = preprocess(pil_img)
    img_tensor = norm_t.unsqueeze(0)
    img_raw    = TF.to_tensor(img_r).unsqueeze(0)

    spread_val = 0.0
    try:
        patches, spread_val = get_patches(img_r)
        patches = patches.unsqueeze(0)
        spread  = torch.tensor([spread_val],dtype=torch.float32)
    except:
        patches = torch.zeros(1,3,3,PATCH_SIZE,PATCH_SIZE)
        spread  = torch.zeros(1)

    blur_val = blur_score(img_r)
    blur     = torch.tensor([blur_val],dtype=torch.float32)

    with torch.no_grad():
        out  = model(img_tensor, img_raw, patches, blur, spread)
    prob = torch.sigmoid(out['logit'].float()/T_CAL).item()
    crs  = out['crs'].item()
    attn = out['attn_map']

    label      = 'FAKE' if prob >= THRESHOLD else 'REAL'
    confidence = prob if label=='FAKE' else 1.0-prob

    # heatmap
    attn_up = F.interpolate(attn.cpu().float(),
                            size=(IMG_SIZE,IMG_SIZE),
                            mode='bilinear',align_corners=False).squeeze().numpy()
    attn_up = (attn_up-attn_up.min())/(attn_up.max()-attn_up.min()+1e-8)
    img_np  = np.array(img_r)/255.0
    heatmap = plt.cm.inferno(attn_up)[:,:,:3]
    overlay = np.clip(0.45*img_np+0.55*heatmap,0,1)

    fig, axes = plt.subplots(1,2,figsize=(10,5))
    fig.patch.set_facecolor('#1C1C1C')
    axes[0].imshow(img_np); axes[0].set_title('Input',color='#CCC',fontsize=13); axes[0].axis('off')
    axes[1].imshow(overlay); axes[1].set_title('Attention Map',color='#CCC',fontsize=13); axes[1].axis('off')
    color = '#FF5555' if label=='FAKE' else '#55FF88'
    icon  = '🔴' if label=='FAKE' else '🟢'
    fig.suptitle(f'{icon} {label}  |  Confidence: {confidence*100:.1f}%  |  CRS: {crs:.3f}  |  Blur: {blur_val:.3f}  |  Spread: {spread_val:.3f}',
                 color=color,fontsize=11,fontweight='bold',y=0.04)
    plt.tight_layout(rect=[0,0.06,1,1])
    buf = io.BytesIO()
    plt.savefig(buf,format='png',dpi=120,bbox_inches='tight',facecolor='#1C1C1C')
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf), label, confidence, prob, crs, blur_val, spread_val

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 RD-FID — Deepfake Detector")
st.markdown("Upload a face image to detect whether it is **real** or **AI-generated**.")
st.markdown("---")

uploaded = st.file_uploader("Choose a face image", type=["jpg","jpeg","png","webp"])

if uploaded:
    pil_img = Image.open(uploaded).convert('RGB')
    with st.spinner("Analysing..."):
        result_img, label, conf, prob, crs, blur, spread = predict(pil_img)

    if label == 'FAKE':
        st.error(f"🔴 FAKE — Confidence: {conf*100:.1f}%")
    else:
        st.success(f"🟢 REAL — Confidence: {conf*100:.1f}%")

    st.image(result_img, use_column_width=True)

    with st.expander("Detailed Scores"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability", f"{prob:.4f}")
            st.metric("Threshold", f"{THRESHOLD}")
            st.metric("CRS Score", f"{crs:.4f}")
        with col2:
            st.metric("Blur Score", f"{blur:.4f}")
            st.metric("Spread", f"{spread:.4f}")
            st.metric("Temperature", f"{T_CAL}")

st.markdown("---")
st.caption("RD-FID · MPSTME NMIMS · 2026")
