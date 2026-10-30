import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import timm, math, face_alignment
import numpy as np, cv2, io, os, json
import gdown
from PIL import Image
import matplotlib.cm as cm
import torchvision.transforms as T
import torchvision.transforms.functional as TF

st.set_page_config(page_title="RD-FID Deepfake Detector", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
#MainMenu{visibility:hidden;}footer{visibility:hidden;}
[data-testid="stAppViewContainer"]{background:#0f1117;}
.hero{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);border:1px solid #2a3a5c;border-radius:16px;padding:32px 36px 24px;margin-bottom:28px;}
.hero h1{font-size:2.4rem;font-weight:800;color:#fff;margin:0 0 8px 0;}
.hero p{font-size:1rem;color:#8b9ab8;margin:0 0 16px 0;}
.verdict-fake{background:linear-gradient(135deg,#3d0000,#5c1a1a);border:1px solid #e53935;border-left:5px solid #e53935;border-radius:10px;padding:18px 24px;margin:16px 0 20px 0;}
.verdict-real{background:linear-gradient(135deg,#003d00,#1a5c1a);border:1px solid #43a047;border-left:5px solid #43a047;border-radius:10px;padding:18px 24px;margin:16px 0 20px 0;}
.verdict-label{font-size:1.8rem;font-weight:800;letter-spacing:1px;margin:0 0 4px 0;}
.verdict-fake .verdict-label{color:#ff6b6b;}.verdict-real .verdict-label{color:#69f0ae;}
.verdict-sub{font-size:0.95rem;color:#aaa;margin:0;}
.metric-card{background:#1e2130;border:1px solid #2d3250;border-radius:10px;padding:14px 16px;margin-bottom:10px;}
.metric-card .label{font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:#6b7fa3;margin-bottom:4px;}
.metric-card .value{font-size:1.4rem;font-weight:700;color:#e8eaf6;font-family:'Courier New',monospace;}
.img-label{text-align:center;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#6b7fa3;padding:8px 0 2px 0;}
.img-sub{text-align:center;font-size:0.66rem;color:#3d4a6b;margin-bottom:4px;}
.region-card{background:#141824;border:1px solid #252d45;border-radius:12px;padding:20px 24px;margin:20px 0;}
.region-title{font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#4a5568;margin-bottom:14px;}
.region-badge{display:inline-block;border-radius:20px;padding:5px 14px;font-size:0.8rem;font-weight:700;margin:0 6px 8px 0;}
.rbar-label{font-size:0.78rem;color:#8b9ab8;margin:8px 0 3px 0;}
.rbar-bg{background:#1f2435;border-radius:4px;height:7px;overflow:hidden;}
.rbar-fill{height:7px;border-radius:4px;}
.region-insight{font-size:0.85rem;color:#7a8caa;line-height:1.65;margin-top:14px;border-top:1px solid #1f2435;padding-top:12px;}
[data-testid="stSidebar"]{background:#13151f;border-right:1px solid #1f2435;}
.ss{font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:#4a5568;margin:20px 0 8px 0;}
.hint{font-size:0.78rem;color:#4a5568;line-height:1.5;margin-top:4px;}
</style>""", unsafe_allow_html=True)

# ── constants ──────────────────────────────────────────────────────────────────
DROPOUT=0.4; PATCH_SIZE=48; PAD_FRAC=0.30
LEFT_EYE_IDX=list(range(36,42)); RIGHT_EYE_IDX=list(range(42,48)); MOUTH_IDX=list(range(48,68))
GDRIVE_MODEL_ID="1NK_upaFhbfopNkfaoizsQCi-2GmiivIs"
GDRIVE_CONFIG_ID="1wZTazea7ia0tMshVX_EKGUaMzWFHOcRF"
MODEL_PATH="/tmp/model.pth"; CONFIG_PATH="/tmp/config.json"

FACE_REGIONS={
    "Left Eye": (list(range(36,42)),"#9c27b0","Eye contour blending is a frequent GAN failure — iris and eyelid boundaries are hard to synthesise."),
    "Right Eye":(list(range(42,48)),"#7b1fa2","Eye contour blending is a frequent GAN failure — iris and eyelid boundaries are hard to synthesise."),
    "Jaw":      (list(range(0,17)), "#e53935","Face-swap models often leave seams and warping along the jaw boundary."),
    "Left Brow":(list(range(17,22)),"#f57c00","Eyebrow skin-texture transitions can expose GAN blending boundaries."),
    "Right Brow":(list(range(22,27)),"#f57c00","Eyebrow skin-texture transitions can expose GAN blending boundaries."),
    "Nose":     (list(range(27,36)),"#fdd835","Nose-bridge symmetry and pore texture are difficult for generators to reproduce."),
    "Mouth":    (list(range(48,68)),"#e91e63","Lip boundary generation is frequently imperfect — colour gradients and teeth are common failure points."),
}
REGION_LM_COLORS={
    "Jaw":      (list(range(0,17)), (229,57,53)),
    "Left Brow":(list(range(17,22)),(245,124,0)),
    "Right Brow":(list(range(22,27)),(245,124,0)),
    "Nose":     (list(range(27,36)),(253,216,53)),
    "Left Eye": (list(range(36,42)),(156,39,176)),
    "Right Eye":(list(range(42,48)),(123,31,162)),
    "Mouth":    (list(range(48,68)),(233,30,99)),
}

# ── model — exact copy from notebook Cell 5 ───────────────────────────────────
def rgb_to_ycbcr(x):
    r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
    y=0.299*r+0.587*g+0.114*b
    cb=-0.16875*r-0.33126*g+0.500*b+0.5
    cr=0.500*r-0.41869*g-0.08131*b+0.5
    return torch.cat([y,cb,cr],dim=1)

class BlockDCT(nn.Module):
    def __init__(self,block_size=8):
        super().__init__(); self.bs=block_size; N=block_size
        D=torch.zeros(N,N)
        for k in range(N):
            for n in range(N):
                D[k,n]=(math.sqrt(1.0/N) if k==0 else math.sqrt(2.0/N)*math.cos(math.pi*k*(2*n+1)/(2*N)))
        self.register_buffer('D',D)
    def forward(self,x):
        B,C,H,W=x.shape; bs=self.bs
        ph=(bs-H%bs)%bs; pw=(bs-W%bs)%bs
        if ph or pw: x=F.pad(x,(0,pw,0,ph),mode='reflect')
        _,_,H2,W2=x.shape
        x=x.unfold(2,bs,bs).unfold(3,bs,bs)
        dct=self.D@x@self.D.t()
        dct=torch.log1p(dct.abs())
        dct=dct.permute(0,1,2,4,3,5).contiguous().view(B,C,H2,W2)
        return dct[:,:,:H,:W]

class FreqBranch(nn.Module):
    def __init__(self,out_dim=256):
        super().__init__(); self.dct=BlockDCT()
        self.net=nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(256,out_dim))
    def forward(self,x_raw): return self.net(self.dct(rgb_to_ycbcr(x_raw)))

class SpatialNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=timm.create_model('efficientnet_b4',pretrained=False,num_classes=0)
        self.freq=FreqBranch(out_dim=256)
        self.head=nn.Sequential(nn.Dropout(DROPOUT),nn.Linear(self.backbone.num_features+256,1))
    def forward(self,x_norm,x_raw):
        return self.head(torch.cat([self.backbone(x_norm),self.freq(x_raw)],dim=1)).squeeze(1)

class PatchCNN(nn.Module):
    def __init__(self,out_dim=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.proj=nn.Linear(256,out_dim)
    def forward(self,x): return self.proj(self.net(x))

class ThreeBranchDetector(nn.Module):
    def __init__(self,stage1_backbone,stage1_freq,patch_cnn):
        super().__init__()
        self.branch_a=stage1_backbone; self.branch_b=stage1_freq; self.branch_c=patch_cnn
        fa,fb,fc=1792,256,128
        self.seam_attn=nn.Sequential(nn.Conv2d(fa,64,1),nn.ReLU(inplace=True),nn.Conv2d(64,1,1),nn.Sigmoid())
        self.head_a=nn.Linear(fa,1); self.head_b=nn.Linear(fb,1); self.head_c=nn.Linear(fc,1)
        self.fusion=nn.Sequential(nn.Linear(fa+fb+fc,256),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(256,1))
    def forward(self,x_norm,x_raw,patches,blur_scores,landmark_spreads):
        feat_map=self.branch_a.forward_features(x_norm)
        attn_map=self.seam_attn(feat_map)
        feat_a=(feat_map*attn_map).mean(dim=[2,3])
        logit_a=self.head_a(feat_a).squeeze(1)
        with torch.no_grad(): conf_d=torch.sigmoid(logit_a.detach())
        feat_b=self.branch_b(x_raw); logit_b=self.head_b(feat_b).squeeze(1)
        blur=blur_scores.to(x_norm.device); spread=landmark_spreads.to(x_norm.device)
        crs=(0.4*conf_d+0.3*blur+0.3*spread).clamp(0.0,1.0)
        patch_feats=[self.branch_c(patches[:,i].to(x_norm.device)) for i in range(3)]
        feat_c=torch.stack(patch_feats,dim=1).mean(dim=1)
        logit_c=self.head_c(feat_c).squeeze(1)
        fused=torch.cat([feat_a,feat_b,feat_c*crs.unsqueeze(1)],dim=1)
        logit=self.fusion(fused).squeeze(1)
        return dict(logit=logit,logit_a=logit_a,logit_b=logit_b,logit_c=logit_c,attn_map=attn_map,crs=crs)

# ── load ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}",MODEL_PATH,quiet=False)
    if not os.path.exists(CONFIG_PATH):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_CONFIG_ID}",CONFIG_PATH,quiet=False)
    with open(CONFIG_PATH) as f: cfg=json.load(f)
    s1=SpatialNetV3(); pc=PatchCNN(out_dim=128)
    m=ThreeBranchDetector(stage1_backbone=s1.backbone,stage1_freq=s1.freq,patch_cnn=pc)
    ckpt=torch.load(MODEL_PATH,map_location="cpu")
    m.load_state_dict(ckpt); m.eval()
    return m,cfg

@st.cache_resource
def load_fa():
    return face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,device='cpu',flip_input=False)

# ── helpers ────────────────────────────────────────────────────────────────────
def _extract_patch(img_np,lm,indices):
    h,w=img_np.shape[:2]; pts=lm[indices]
    x1,y1=pts.min(axis=0); x2,y2=pts.max(axis=0)
    pw=(x2-x1)*PAD_FRAC; ph=(y2-y1)*PAD_FRAC
    x1=max(0,int(x1-pw)); y1=max(0,int(y1-ph))
    x2=min(w,int(x2+pw)); y2=min(h,int(y2+ph))
    if x2<=x1 or y2<=y1: return np.zeros((PATCH_SIZE,PATCH_SIZE,3),dtype=np.uint8)
    return cv2.resize(img_np[y1:y2,x1:x2],(PATCH_SIZE,PATCH_SIZE),interpolation=cv2.INTER_LINEAR)

def get_landmark_data(pil_img,fa):
    rgb=np.array(pil_img.convert('RGB'))
    result=fa.get_landmarks(rgb)
    if result is None or len(result)==0: return None,0.0,None
    lm=result[0]; h,w=rgb.shape[:2]
    lm_norm=lm/np.array([w,h])
    spread=float(min(lm_norm.std()/0.20,1.0))
    to_t=T.ToTensor()
    patches=torch.stack([
        to_t(Image.fromarray(_extract_patch(rgb,lm,LEFT_EYE_IDX))),
        to_t(Image.fromarray(_extract_patch(rgb,lm,RIGHT_EYE_IDX))),
        to_t(Image.fromarray(_extract_patch(rgb,lm,MOUTH_IDX))),
    ])
    return patches,spread,lm

def compute_blur(pil_img,max_var=500.0):
    gray=np.array(pil_img.convert('L'))
    return float(min(cv2.Laplacian(gray,cv2.CV_64F).var()/max_var,1.0))

def draw_lm(rgb,lm):
    img=rgb.copy(); h,w=img.shape[:2]
    for name,(indices,color) in REGION_LM_COLORS.items():
        for i in indices:
            px,py=int(lm[i,0]),int(lm[i,1])
            if 0<=px<w and 0<=py<h:
                cv2.circle(img,(px,py),3,color,-1)
                cv2.circle(img,(px,py),3,(255,255,255),1)
    return Image.fromarray(img)

def make_patch_collage(patches):
    imgs=[]
    for i,label in enumerate(['L.Eye','R.Eye','Mouth']):
        p=(patches[i].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
        p=cv2.resize(p,(80,80)); canvas=np.zeros((96,80,3),dtype=np.uint8); canvas[16:]=p
        cv2.putText(canvas,label,(3,11),cv2.FONT_HERSHEY_SIMPLEX,0.35,(160,160,160),1)
        imgs.append(canvas)
    return Image.fromarray(np.concatenate(imgs,axis=1))

def to_pil_buf(img): buf=io.BytesIO(); img.save(buf,format='PNG'); buf.seek(0); return buf

# ── inference ──────────────────────────────────────────────────────────────────
def run_inference(pil_img,model,cfg,fa):
    img_size=cfg.get('img_size',224); T_cal=cfg.get('temperature',1.0); thresh=cfg.get('threshold',0.5)
    img=pil_img.convert('RGB').resize((img_size,img_size),Image.BILINEAR)
    rgb=np.array(img)
    raw_t=TF.to_tensor(img).unsqueeze(0)
    norm_t=T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(raw_t.clone())
    patches,spread_val,lm=get_landmark_data(img,fa)
    if patches is None:
        patches=torch.zeros(3,3,PATCH_SIZE,PATCH_SIZE); spread_val=0.0; lm=None
    patches_t=patches.unsqueeze(0)
    blur_val=compute_blur(img)
    blur_t=torch.tensor([blur_val],dtype=torch.float32)
    spread_t=torch.tensor([spread_val],dtype=torch.float32)
    with torch.no_grad():
        out=model(norm_t,raw_t,patches_t,blur_t,spread_t)
        prob=torch.sigmoid(out['logit'].float()/T_cal).item()
        crs_val=out['crs'].item()
        attn_np=out['attn_map'][0,0].numpy()
    verdict='FAKE' if prob>=thresh else 'REAL'
    attn_up=cv2.resize(attn_np,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
    attn_n=(attn_up-attn_up.min())/(attn_up.max()-attn_up.min()+1e-8)
    heat=(cm.inferno(attn_n)[:,:,:3]*255).astype(np.uint8)
    heatmap=Image.fromarray((0.55*heat+0.45*rgb).clip(0,255).astype(np.uint8))
    lm_img=draw_lm(rgb,lm) if lm is not None else Image.fromarray(rgb)
    region_scores={}
    if lm is not None:
        for name,(indices,_,_) in FACE_REGIONS.items():
            pts=lm[indices].astype(int); vals=[]
            for px,py in pts:
                r=12; xs,xe=max(0,px-r),min(img_size,px+r); ys,ye=max(0,py-r),min(img_size,py+r)
                if xs<xe and ys<ye: vals.append(float(attn_n[ys:ye,xs:xe].mean()))
            if vals: region_scores[name]=np.mean(vals)
    return dict(verdict=verdict,prob=prob,thresh=thresh,crs=crs_val,blur=blur_val,
                spread=spread_val,temp=T_cal,heatmap=heatmap,input_img=Image.fromarray(rgb),
                lm_overlay=lm_img,patch_collage=make_patch_collage(patches),
                region_scores=region_scores), None

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ss">About</div>',unsafe_allow_html=True)
    st.markdown("<span style='color:#8b9ab8;font-size:0.85rem;line-height:1.6'>Three-branch detector: EfficientNet-B4 spatial + BlockDCT frequency + FAN landmark patches, fused with CRS gating.<br><br><b style='color:#c5cfe0'>RD-FID · MPSTME NMIMS · 2026</b></span>",unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="ss">Score Guide</div>',unsafe_allow_html=True)
    for name,desc in [("Confidence","How strongly model predicts FAKE (0–1)"),("Threshold","Min confidence to classify FAKE"),("CRS","Crop Reliability Score — gates Branch C"),("Blur","Laplacian sharpness 0–1"),("Spread","FAN landmark spread 0–1"),("Temperature","Calibration factor on model logit")]:
        st.markdown(f"<div style='padding:6px 0;border-bottom:1px solid #1f2435'><div style='font-size:0.82rem;font-weight:700;color:#c5cfe0'>{name}</div><div class='hint'>{desc}</div></div>",unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="ss">Robustness</div>',unsafe_allow_html=True)
    for cond,detail,auc,color in [("Clean","No degradation","0.89","#43a047"),("Occlusion","30% black patch","0.81","#e53935"),("Pose","20° warp","0.79","#f57c00"),("Low-res","4× downsample","0.76","#fdd835")]:
        st.markdown(f"<div style='padding:6px 0;border-bottom:1px solid #1f2435'><div style='display:flex;justify-content:space-between'><span style='font-size:0.82rem;color:#c5cfe0;font-weight:700'>{cond}</span><span style='font-size:0.82rem;color:{color};font-weight:700'>AUC {auc}</span></div><span style='font-size:0.72rem;color:#4a5568'>{detail}</span></div>",unsafe_allow_html=True)

# ── main ───────────────────────────────────────────────────────────────────────
st.markdown("""<div class="hero">
<h1>🔍 RD-FID Deepfake Detector</h1>
<p>Upload a face image — spatial, frequency, and landmark branches vote on authenticity.</p>
<div style="display:flex;gap:10px;flex-wrap:wrap">
<span style="background:#1a3a6e;border:1px solid #2a5aaa;color:#7eb3ff;border-radius:20px;padding:5px 14px;font-size:0.75rem;font-weight:700">Branch A · EfficientNet-B4 · Spatial</span>
<span style="background:#3a1a6e;border:1px solid #6a2aaa;color:#c07bff;border-radius:20px;padding:5px 14px;font-size:0.75rem;font-weight:700">Branch B · BlockDCT · Frequency</span>
<span style="background:#6e1a3a;border:1px solid #aa2a6a;color:#ff7eb3;border-radius:20px;padding:5px 14px;font-size:0.75rem;font-weight:700">Branch C · FAN Patches × CRS</span>
</div></div>""",unsafe_allow_html=True)

uploaded=st.file_uploader("",type=["jpg","jpeg","png","webp"],label_visibility="collapsed")

if uploaded:
    pil_img=Image.open(uploaded)
    with st.spinner("Running inference..."):
        try:
            model,cfg=load_model(); fa=load_fa()
            result,err=run_inference(pil_img,model,cfg,fa)
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()
    if err: st.warning(err); st.stop()

    is_fake=result['verdict']=='FAKE'
    cls='verdict-fake' if is_fake else 'verdict-real'
    icon='🔴' if is_fake else '🟢'
    pct=f"{result['prob']*100:.1f}%"
    st.markdown(f'<div class="{cls}"><div class="verdict-label">{icon} {result["verdict"]} &nbsp;·&nbsp; {pct}</div><div class="verdict-sub">Confidence the image is AI-generated &nbsp;|&nbsp; Threshold: {result["thresh"]:.3f}</div></div>',unsafe_allow_html=True)

    # 4-panel
    c1,c2,c3,c4=st.columns(4,gap="small")
    for col,img,title,sub in [(c1,result['input_img'],'Input Face','Branch A·B input'),
                               (c2,result['lm_overlay'],'68 Landmarks','Branch C regions'),
                               (c3,result['heatmap'],'Model Attention','seam_attn map'),
                               (c4,result['patch_collage'],'Eye·Eye·Mouth','Branch C patches')]:
        with col:
            st.markdown(f"<div class='img-label'>{title}</div><div class='img-sub'>{sub}</div>",unsafe_allow_html=True)
            st.image(to_pil_buf(img),use_container_width=True)

    # region analysis
    rs=result['region_scores']
    if rs:
        mx=max(rs.values()); nm={k:v/mx for k,v in rs.items()}
        top3=sorted(rs,key=rs.get,reverse=True)[:3]
        badges="".join(f"<span class='region-badge' style='background:{FACE_REGIONS[r][1]}22;color:{FACE_REGIONS[r][1]};border:1px solid {FACE_REGIONS[r][1]}55'>{r}</span>" for r in top3)
        bars="".join(f"<div class='rbar-label'>{r} <span style='color:#3d4a6b;font-size:0.72rem'>{rs[r]:.3f}</span></div><div class='rbar-bg'><div class='rbar-fill' style='width:{int(nm[r]*100)}%;background:{FACE_REGIONS[r][1]}'></div></div>" for r in sorted(rs,key=rs.get,reverse=True))
        seen,ins=[],[]
        for r in top3:
            d=FACE_REGIONS[r][2]
            if d not in seen: ins.append(f"<b style='color:#c5cfe0'>{r}:</b> {d}"); seen.append(d)
        note=("Model attention highlights synthesis boundary artefacts." if is_fake else "No GAN artefact patterns detected — attention is naturally distributed.")
        st.markdown(f'<div class="region-card"><div class="region-title">Facial Region Activity (model attention)</div><div style="margin-bottom:12px">{badges}</div>{bars}<div class="region-insight"><b style="color:#6b7fa3;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.8px">Why these regions matter</b><br><br>{"<br>".join(ins)}<br><br><span style="color:#4a5568">{note}</span></div></div>',unsafe_allow_html=True)

    # metric cards
    for row in [[("Confidence",f"{result['prob']:.3f}"),("CRS Score",f"{result['crs']:.3f}"),("Blur",f"{result['blur']:.3f}")],
                [("Spread",f"{result['spread']:.3f}"),("Temperature",f"{result['temp']:.4f}"),("Threshold",f"{result['thresh']:.3f}")]]:
        cols=st.columns(3)
        for col,(label,val) in zip(cols,row):
            with col: st.markdown(f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{val}</div></div>",unsafe_allow_html=True)
else:
    st.markdown("<div style='text-align:center;padding:60px 20px;color:#3d4a6b'>Drop a face image above to begin.</div>",unsafe_allow_html=True)
