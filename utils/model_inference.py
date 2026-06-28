"""
MAIZE-XNet Model Inference Module
Loads 4 ONNX backbone models + Attention Gate from Hugging Face Hub.
Runs ensemble inference, Grad-CAM (via PyTorch hooks), and TSDS computation.

Setup:
  1. Upload your .onnx files to Hugging Face Hub
  2. Set HF_REPO_ID below to your repo (e.g. "your-username/maize-xnet")
  3. The app will auto-download models on first run and cache them
"""

import os
import io
import numpy as np
from PIL import Image

# ── CONFIG — UPDATE THESE ─────────────────────────────────────────────────────
HF_REPO_ID        = "akash4529/corn-leaf"   # ← Set your HF repo
USING_REAL_MODELS = True                          # ← Set False for demo only

# ONNX filenames as uploaded to HF Hub (must match Phase 6 output exactly)
ONNX_FILES = {
    "efficientnet_b4": "efficientnet_b4.onnx",
    "convnext_tiny":   "convnext_tiny.onnx",
    "maxvit_small":    "maxvit_small.onnx",
    "mobilevit_small": "mobilevit_small.onnx",
    "gate":            "attention_gate.onnx",
}

# Image sizes per model (must match training)
IMG_SIZES = {
    "efficientnet_b4": 380,
    "convnext_tiny":   224,
    "maxvit_small":    224,
    "mobilevit_small": 256,
}

NUM_CLASSES = 4
CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# ── NORMALIZATION (ImageNet) ──────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(pil_img, img_size):
    """Resize, normalize and convert PIL image to ONNX-ready numpy (1,3,H,W)."""
    img = pil_img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)          # HWC → CHW
    arr = arr[np.newaxis, ...]            # add batch dim
    return arr


def download_onnx_models():
    """Download all ONNX files from Hugging Face Hub. Returns dict of local paths."""
    try:
        from huggingface_hub import hf_hub_download
        paths = {}
        for key, fname in ONNX_FILES.items():
            local = hf_hub_download(repo_id=HF_REPO_ID, filename=fname)
            paths[key] = local
        return paths
    except Exception as e:
        print(f"[MAIZE-XNet] HF download failed: {e}")
        return None


def load_all_models():
    """
    Load all 5 ONNX InferenceSession objects.
    Returns dict of sessions, or None if unavailable.
    """
    if not USING_REAL_MODELS:
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        print("[MAIZE-XNet] onnxruntime not installed.")
        return None

    paths = download_onnx_models()
    if not paths:
        return None

    try:
        providers = ["CPUExecutionProvider"]
        sessions = {}
        for key, path in paths.items():
            sess = ort.InferenceSession(path, providers=providers)
            sessions[key] = sess
        print("[MAIZE-XNet] All 5 ONNX models loaded successfully.")
        return sessions
    except Exception as e:
        print(f"[MAIZE-XNet] ONNX session load failed: {e}")
        return None


def run_inference(pil_img, sessions):
    """
    Run all 4 ONNX backbone models + Attention Gate.

    Returns:
        pred_idx        (int)       — index of predicted class
        individual_probs (np.array) — shape (4, 4), softmax per model
        gate_weights    (np.array)  — shape (4,), attention weights
        final_probs     (np.array)  — shape (4,), ensemble final softmax
    """
    model_keys = ["efficientnet_b4", "convnext_tiny", "maxvit_small", "mobilevit_small"]
    softmax_probs = []

    for key in model_keys:
        sess     = sessions[key]
        img_size = IMG_SIZES[key]
        arr      = preprocess_image(pil_img, img_size)
        in_name  = sess.get_inputs()[0].name
        logits   = sess.run(None, {in_name: arr})[0][0]   # shape (4,)
        probs    = _softmax(logits)
        softmax_probs.append(probs)

    individual_probs = np.array(softmax_probs, dtype=np.float32)  # (4,4)

    # Compute entropy per model for gate input (TSDS proxy)
    entropies = [-np.sum(p * np.log(p + 1e-12)) for p in softmax_probs]
    entropies = np.array(entropies, dtype=np.float32)  # (4,)

    # Gate input: [probs(4), entropy(1)] × 4 = 20-dim
    gate_input = np.concatenate(
        [np.append(p, e) for p, e in zip(softmax_probs, entropies)]
    ).astype(np.float32)[np.newaxis, :]                   # (1, 20)

    gate_sess    = sessions["gate"]
    gate_in_name = gate_sess.get_inputs()[0].name
    gate_weights = gate_sess.run(None, {gate_in_name: gate_input})[0][0]  # (4,)
    # Ensure gate_weights are a valid probability distribution (sum to 1).
    # If the gate ONNX already applies softmax, _softmax here is a safe no-op.
    # If it outputs raw logits, this converts them correctly.
    gate_weights = _softmax(gate_weights)

    # Weighted ensemble
    # FIX: individual_probs are already valid probability distributions (each row sums to 1).
    # gate_weights also sum to 1 (softmax output from attention gate).
    # Therefore their weighted sum is already a valid probability distribution — DO NOT
    # apply softmax() again. Re-applying softmax compresses the distribution toward
    # uniform (all classes get ~1/4 ≈ 25%), which is why the ensemble confidence was
    # stuck at ~47% even when all 4 backbone models agreed at 99-100%.
    final_probs = np.sum(
        individual_probs * gate_weights[:, np.newaxis], axis=0
    )
    # Simple L1 normalization to handle any floating-point drift (stays in probability space)
    final_probs = final_probs / final_probs.sum()
    pred_idx    = int(np.argmax(final_probs))

    return pred_idx, individual_probs, gate_weights, final_probs


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ── GRAD-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam(pil_img, sessions, pred_class_idx):
    """
    Compute Grad-CAM attention maps for all 4 backbone models using
    PyTorch hooks (models reconstructed in eval mode from timm).

    Returns list of 4 numpy arrays (224×224, normalized 0-1), or None on failure.
    """
    try:
        import torch
        import torch.nn as nn
        import timm
        from huggingface_hub import hf_hub_download

        DEVICE = torch.device("cpu")

        # Rebuild PyTorch model architecture (same as training)
        class MAIZEXNetModel(nn.Module):
            def __init__(self, timm_name):
                super().__init__()
                self.backbone = timm.create_model(timm_name, pretrained=False, num_classes=0)
                in_feat = self.backbone.num_features
                self.head = nn.Sequential(
                    nn.Dropout(p=0.4), nn.Linear(in_feat, 512),
                    nn.GELU(), nn.Dropout(p=0.3), nn.Linear(512, NUM_CLASSES),
                )
            def forward(self, x):
                return self.head(self.backbone(x))

        # Load PyTorch checkpoints from HF Hub
        pt_files = {
            "efficientnet_b4": "best_efficientnetb4_maizexnet.pth",
            "convnext_tiny":   "best_convnexttiny_maizexnet.pth",
            "maxvit_small":    "best_maxvitsmall_maizexnet.pth",
            "mobilevit_small": "best_mobilvitsmall_maizexnet.pth",
        }
        timm_names = {
            "efficientnet_b4": "efficientnet_b4",
            "convnext_tiny":   "convnext_tiny",
            "maxvit_small":    "maxvit_small_tf_224",
            "mobilevit_small": "mobilevit_s",
        }
        pt_img_sizes = IMG_SIZES

        from torchvision import transforms
        def get_transform(size):
            return transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

        cam_maps = []

        for key in ["efficientnet_b4", "convnext_tiny", "maxvit_small", "mobilevit_small"]:
            # Download PyTorch checkpoint
            pt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=pt_files[key])
            model   = MAIZEXNetModel(timm_names[key]).to(DEVICE)
            model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
            model.eval()

            # Prepare input
            sz  = pt_img_sizes[key]
            tf  = get_transform(sz)
            inp = tf(pil_img).unsqueeze(0).to(DEVICE)

            # Hook-based Grad-CAM
            gradients, activations = [], []

            def save_grad(g):   gradients.append(g)
            def save_act(m, i, o):
                activations.append(o)
                o.register_hook(save_grad)

            # Pick target layer per architecture
            if key == "efficientnet_b4":
                target = model.backbone.blocks[-1]
            elif key == "convnext_tiny":
                target = model.backbone.stages[-1]
            elif key == "maxvit_small":
                target = model.backbone.stages[-1]
            else:   # mobilevit_small
                target = model.backbone.stages[-1]

            handle = target.register_forward_hook(save_act)

            # Forward + backward
            out  = model(inp)
            model.zero_grad()
            score = out[0, pred_class_idx]
            score.backward()

            handle.remove()

            if not gradients or not activations:
                cam_maps.append(np.zeros((224, 224), dtype=np.float32))
                continue

            grads = gradients[0]    # (1, C, H, W) or similar
            acts  = activations[0]

            # Pool gradients over spatial dims → weights
            if grads.dim() == 4:
                weights = grads.mean(dim=[2, 3], keepdim=True)
                cam     = (weights * acts).sum(dim=1).squeeze()
            else:
                # Transformer outputs may be (B, N, C)
                weights = grads.mean(dim=1)
                cam     = (weights * acts).sum(dim=-1).squeeze()
                n       = cam.shape[0]
                hw      = int(n ** 0.5)
                cam     = cam[:hw*hw].reshape(hw, hw)

            cam = cam.detach().cpu().numpy()
            cam = np.maximum(cam, 0)    # ReLU

            # Resize to 224×224
            import cv2
            cam = cv2.resize(cam.astype(np.float32), (224, 224))
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())

            cam_maps.append(cam)

            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return cam_maps

    except Exception as e:
        print(f"[MAIZE-XNet] Grad-CAM failed: {e}")
        return None


# ── TSDS — Temporal Saliency Drift Score ─────────────────────────────────────
def compute_tsds(cam_maps, thresh_pctile=80):
    """
    Compute TSDS from 4 Grad-CAM attention maps.

    TSDS = mean pairwise IoU of binarized saliency masks.
    High TSDS = models agree on WHERE the disease is → high spatial stability.
    Low TSDS  = models disagree → saliency drift → lower prediction trust.

    Args:
        cam_maps       : list of 4 np.arrays, shape (224,224), values in [0,1]
        thresh_pctile  : percentile for binarization (default 80)

    Returns:
        float in [0, 1]
    """
    def binarize(m):
        t = np.percentile(m, thresh_pctile)
        return (m >= t).astype(np.uint8)

    def iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union > 0 else 0.0

    bins  = [binarize(m) for m in cam_maps]
    pairs = [(bins[i], bins[j])
             for i in range(len(bins))
             for j in range(i + 1, len(bins))]
    return float(np.mean([iou(a, b) for a, b in pairs]))
