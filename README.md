# RD-FID — Deepfake Face Detection

A three-branch neural network that detects AI-generated face images using spatial texture, frequency-domain analysis, and facial landmark patches — with a novel Crop Reliability Score (CRS) gating mechanism.

**Live demo:** [rd-fid-deepfake-detector.streamlit.app](https://rd-fid-deepfake-detector-6q8nn5cawip2je28384373.streamlit.app)

---

## How It Works

The model runs three parallel branches on every image:

| Branch | Module | What it captures |
|--------|--------|-----------------|
| A | EfficientNet-B4 + seam attention | Spatial texture inconsistencies |
| B | BlockDCT + FreqBranch CNN | Frequency-domain GAN artefacts |
| C | FAN 68-point landmarks + PatchCNN | Eye and mouth region anomalies |

**Crop Reliability Score (CRS)** — our novel contribution — dynamically gates how much Branch C contributes before fusion:

CRS = 0.4 × D  +  0.3 × B  +  0.3 × S

where D = Branch A confidence, B = Laplacian sharpness, S = landmark spread. If the face crop is unreliable, Branches A and B carry more weight automatically.

Final prediction uses temperature-scaled sigmoid with a tuned threshold learned on held-out data.

---

## Model Performance

| Condition | AUC |
|-----------|-----|
| Clean (no degradation) | 0.89 |
| Occlusion (30% patch) | 0.81 |
| Pose change (20° warp) | 0.79 |
| Low resolution (4× downsample) | 0.76 |

Trained on GAN140 + DFDC datasets.

---

## Web App Features

- Upload any face image (JPG / PNG / WEBP)
- Four-panel visualisation: input · 68 landmarks · model attention map · eye and mouth patches
- Per-region activity analysis mapped to the model's seam attention output
- CRS breakdown showing the three reliability components
- All six numerical scores displayed (confidence, threshold, CRS, blur, spread, temperature)

---

## Repository Structure

app.py              Streamlit web application
requirements.txt    Python dependencies
packages.txt        System packages (libgl1, libglib2.0-0)

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py

The model weights (~100 MB) and config are downloaded automatically from Google Drive on first run.

---
Architecture

Input image
    │
    ├─── Branch A: EfficientNet-B4 → seam_attn → 1792-d features
    │
    ├─── Branch B: YCbCr → BlockDCT → FreqCNN → 256-d features
    │
    └─── Branch C: FAN landmarks → 3 patches (eye·eye·mouth)
                   → PatchCNN → 128-d features × CRS

    Concatenate [A · B · C×CRS] → Fusion head → Temperature scaling → Verdict
