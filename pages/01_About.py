"""MAIZE-XNet — About Page"""
import streamlit as st
import textwrap

st.set_page_config(page_title="About | MAIZE-XNet", page_icon="🌽", layout="wide")

def md(html):
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)

with open("static/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

md("""
<div class="app-header">
  <div class="header-left">
    <div class="logo-wordmark">
      <span class="logo-maize">MAIZE</span><span class="logo-x">-X</span><span class="logo-net">Net</span>
    </div>
    <p class="app-subtitle">About the System &mdash; MSc Thesis Research</p>
  </div>
</div>
""")

md("""
<div class="section-card about-section">
  <h2 class="section-title">Research Overview</h2>
  <p class="section-desc">
    MAIZE-XNet addresses three critical gaps unresolved across 15 prior corn disease papers:
    no cross-architecture multi-scale ensemble, no augmentation-driven saliency stability
    metric, and no offline PWA deployment.
  </p>
</div>

<div class="section-card">
  <h3 class="section-title">The 4 Novel Contributions</h3>
  <div class="how-grid">
    <div class="how-step">
      <div class="step-num">01</div>
      <h4 class="step-title">Multi-Scale Ensemble</h4>
      <p class="step-desc">4 architectures spanning CNN and Transformer families,
      each specialized at a different spatial granularity for corn disease diagnosis.</p>
    </div>
    <div class="how-step">
      <div class="step-num">02</div>
      <h4 class="step-title">Attention Gate</h4>
      <p class="step-desc">Learned cross-architecture gate that dynamically weights
      each model's contribution per input image — replacing fixed soft voting.</p>
    </div>
    <div class="how-step">
      <div class="step-num">03</div>
      <h4 class="step-title">TSDS Trust Metric</h4>
      <p class="step-desc">Temporal Saliency Drift Score — first augmentation-driven
      saliency stability certificate in the corn disease classification literature.</p>
    </div>
    <div class="how-step">
      <div class="step-num">04</div>
      <h4 class="step-title">PWA Deployment</h4>
      <p class="step-desc">First offline-capable Progressive Web App for corn disease
      with ONNX inference, Grad-CAM, TSDS, and PDF reports — no internet required.</p>
    </div>
  </div>
</div>

<div class="section-card contact-card">
  <h3 class="section-title">Developer Contact</h3>
  <div class="contact-row">
    <div class="contact-item">
      <span class="contact-label">Developer</span>
      <span class="contact-val">Your Name — MSc Researcher</span>
    </div>
    <div class="contact-item">
      <span class="contact-label">Email</span>
      <span class="contact-val">your@email.com</span>
    </div>
    <div class="contact-item">
      <span class="contact-label">Institution</span>
      <span class="contact-val">Your University, Department of CSE</span>
    </div>
    <div class="contact-item">
      <span class="contact-label">Target Journal</span>
      <span class="contact-val">Computers and Electronics in Agriculture (Q1, Elsevier)</span>
    </div>
  </div>
</div>
""")
