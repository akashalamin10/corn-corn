"""MAIZE-XNet — How It Works"""
import streamlit as st
import textwrap

st.set_page_config(page_title="How It Works | MAIZE-XNet", page_icon="🌽", layout="wide")

def md(html): st.markdown(textwrap.dedent(html), unsafe_allow_html=True)

with open("static/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

md("""
<div class="app-header">
  <div class="header-left">
    <div class="logo-wordmark">
      <span class="logo-maize">MAIZE</span><span class="logo-x">-X</span><span class="logo-net">Net</span>
    </div>
    <p class="app-subtitle">Technical Pipeline — How MAIZE-XNet Works</p>
  </div>
</div>

<div class="section-card">
  <h2 class="section-title">Inference Pipeline</h2>
  <div class="pipeline-bar" style="flex-direction:column;align-items:flex-start;gap:8px;">
    <div><span class="pipe-model" style="color:#34e0ff;">Step 1</span> <span class="pipe-sep">→</span> Upload corn leaf image</div>
    <div><span class="pipe-model" style="color:#39ff88;">Step 2</span> <span class="pipe-sep">→</span> Resize to 4 different resolutions: 380px (EfficientNet), 224px (ConvNeXt/MaxViT), 256px (MobileViT)</div>
    <div><span class="pipe-model" style="color:#ff3b5c;">Step 3</span> <span class="pipe-sep">→</span> Run 4 ONNX models independently → 4 softmax probability vectors (4 classes each)</div>
    <div><span class="pipe-model" style="color:#ff3df0;">Step 4</span> <span class="pipe-sep">→</span> Compute entropy per model → concatenate [probs(4) + entropy(1)] × 4 = 20-dim gate input</div>
    <div><span class="pipe-model" style="color:#ffe033;">Step 5</span> <span class="pipe-sep">→</span> Attention Gate produces 4 dynamic weights (sum to 1)</div>
    <div><span class="pipe-model" style="color:#00ffcc;">Step 6</span> <span class="pipe-sep">→</span> Weighted sum → final 4-class ensemble prediction</div>
    <div><span class="pipe-model" style="color:#39ff88;">Step 7</span> <span class="pipe-sep">→</span> Grad-CAM generates 4 attention heatmaps</div>
    <div><span class="pipe-model" style="color:#ffe033;">Step 8</span> <span class="pipe-sep">→</span> TSDS computes mean pairwise IoU of binarized attention masks → trust certificate</div>
  </div>
</div>

<div class="section-card">
  <h2 class="section-title">The 4 Base Models</h2>
  <div class="how-grid">
    <div class="how-step">
      <div class="step-num" style="color:#34e0ff;">B4</div>
      <h4 class="step-title" style="color:#34e0ff;">EfficientNet-B4</h4>
      <p class="step-desc">380px input. Compound scaling for global-to-local texture discrimination.
      Specialized for large-scale disease patterns like Northern Leaf Blight's elongated lesions.</p>
    </div>
    <div class="how-step">
      <div class="step-num" style="color:#39ff88;">CN</div>
      <h4 class="step-title" style="color:#39ff88;">ConvNeXt-Tiny</h4>
      <p class="step-desc">224px input. Modernized pure-CNN with depthwise separable convolutions.
      Captures fine-grained LOCAL lesion textures — specialized for Common Rust's small pustules.</p>
    </div>
    <div class="how-step">
      <div class="step-num" style="color:#ff3b5c;">MX</div>
      <h4 class="step-title" style="color:#ff3b5c;">MaxViT-Small</h4>
      <p class="step-desc">224px input. Multi-axis ViT with joint local window + global dilated attention.
      Captures both fine textures and full-leaf disease distribution simultaneously.</p>
    </div>
    <div class="how-step">
      <div class="step-num" style="color:#ff3df0;">MV</div>
      <h4 class="step-title" style="color:#ff3df0;">MobileViT-Small</h4>
      <p class="step-desc">256px input. Lightweight hybrid CNN-Transformer for efficient mobile inference.
      Smallest model — fastest inference path for deployment on low-resource devices.</p>
    </div>
  </div>
</div>

<div class="section-card tsds-card">
  <h2 class="section-title">TSDS — What Makes It Novel</h2>
  <p class="section-desc">
    The Temporal Saliency Drift Score is the central novel contribution of MAIZE-XNet.
    All 15 prior corn disease papers apply Grad-CAM as a static single-pass visualization.
    None measure whether saliency regions remain spatially stable when the same image is
    slightly augmented (rotated, brightness-shifted, flipped, noised).
  </p>
  <br/>
  <p class="section-desc">
    <strong style="color:#ffe033;">TSDS computation:</strong> For each test image, T augmented passes
    generate T Grad-CAM maps per model. TSDS = mean pairwise IoU of binarized attention masks
    (thresholded at 80th percentile activation). High TSDS = model consistently attends to the
    same leaf region = high diagnostic trust. Low TSDS = saliency drift = uncertain prediction.
  </p>
</div>
""")
