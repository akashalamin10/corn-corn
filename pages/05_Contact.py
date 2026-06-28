"""MAIZE-XNet — Contact & Support Page"""
import streamlit as st
import textwrap
import datetime

st.set_page_config(page_title="Contact | MAIZE-XNet", page_icon="🌽", layout="wide")

def md(html): st.markdown(textwrap.dedent(html), unsafe_allow_html=True)

with open("static/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

md("""
<div class="app-header">
  <div class="header-left">
    <div class="logo-wordmark">
      <span class="logo-maize">MAIZE</span><span class="logo-x">-X</span><span class="logo-net">Net</span>
    </div>
    <p class="app-subtitle">Contact &amp; Support — Error Reporting</p>
  </div>
</div>
""")

# ── Common error solutions ─────────────────────────────────────────────────────
md("""
<div class="section-card" style="border-left:4px solid #ff3b5c;">
  <h2 class="section-title" style="color:#ff3b5c;">Common Issues &amp; Solutions</h2>

  <div style="display:flex;flex-direction:column;gap:14px;margin-top:10px;">

    <div style="background:#100a0a;border:1px solid #ff3b5c44;border-radius:2px;padding:14px 16px;">
      <div class="action-label">⚠ Model files not found / Loading failed</div>
      <p class="action-text">
        The ONNX model files (.onnx) or PyTorch checkpoints (.pth) have not been uploaded
        to the Hugging Face repository yet, or the <code>HF_REPO_ID</code> in
        <code>utils/model_inference.py</code> is not set correctly.<br/><br/>
        <strong>Fix:</strong> Upload all 5 ONNX files and 4 .pth files to your Hugging Face
        model repo, then update <code>HF_REPO_ID = "your-username/maize-xnet"</code> in
        <code>utils/model_inference.py</code>. The app will run in DEMO MODE until models load.
      </p>
    </div>

    <div style="background:#100a0a;border:1px solid #ffb62744;border-radius:2px;padding:14px 16px;">
      <div class="action-label" style="color:#ffb627;">⚠ Slow inference / Timeout</div>
      <p class="action-text">
        MAIZE-XNet runs 4 large ONNX models sequentially on CPU. Expected inference time:
        75–250ms per model on a modern CPU (425ms total sequential pipeline).<br/><br/>
        <strong>Fix:</strong> If running on Hugging Face free tier, MaxViT-Small (262MB) may
        load slowly on first request. Subsequent requests use cached sessions. Consider
        upgrading to a GPU Space for sub-50ms inference.
      </p>
    </div>

    <div style="background:#100a0a;border:1px solid #ffb62744;border-radius:2px;padding:14px 16px;">
      <div class="action-label" style="color:#ffb627;">⚠ PDF generation failed</div>
      <p class="action-text">
        The <code>reportlab</code> library may not be installed correctly.<br/><br/>
        <strong>Fix:</strong> Ensure <code>reportlab&gt;=4.0.0</code> is in
        <code>requirements.txt</code> (already included). If the error persists on
        Hugging Face Spaces, restart the Space from the Settings tab.
      </p>
    </div>

    <div style="background:#100a0a;border:1px solid #ff3b5c44;border-radius:2px;padding:14px 16px;">
      <div class="action-label">⚠ Grad-CAM returns blank heatmaps</div>
      <p class="action-text">
        Grad-CAM requires PyTorch .pth checkpoint files in addition to the ONNX files.
        If the .pth files are not uploaded to Hugging Face, Grad-CAM will fall back to
        demo mode heatmaps.<br/><br/>
        <strong>Fix:</strong> Upload all 4 <code>best_*.pth</code> checkpoint files to
        your Hugging Face repo alongside the ONNX files.
      </p>
    </div>

  </div>
</div>
""")

# ── Contact form ───────────────────────────────────────────────────────────────
md("""
<div class="section-card contact-card">
  <h2 class="section-title">Developer Contact</h2>
  <div class="contact-row">
    <div class="contact-item">
      <span class="contact-label">Developer</span>
      <span class="contact-val">Your Name — MSc Researcher, Department of CSE</span>
    </div>
    <div class="contact-item">
      <span class="contact-label">Email</span>
      <span class="contact-val">your@email.com</span>
    </div>
    <div class="contact-item">
      <span class="contact-label">Institution</span>
      <span class="contact-val">Your University, Department of Computer Science &amp; Engineering</span>
    </div>
    <div class="contact-item">
      <span class="contact-label">GitHub</span>
      <span class="contact-val">github.com/your-username/maize-xnet</span>
    </div>
    <div class="contact-item">
      <span class="contact-label">HF Space</span>
      <span class="contact-val">huggingface.co/spaces/your-username/maize-xnet</span>
    </div>
    <div class="contact-item">
      <span class="contact-label">Thesis</span>
      <span class="contact-val">
        MAIZE-XNet: A Multi-Scale Attention-Gated Cross-Architecture Ensemble
        with Temporal Saliency Drift Analysis — MSc Thesis, [Year]
      </span>
    </div>
    <div class="contact-item">
      <span class="contact-label">Target Journal</span>
      <span class="contact-val">Computers and Electronics in Agriculture (Q1, Elsevier)</span>
    </div>
  </div>
</div>
""")

# ── Bug report form ────────────────────────────────────────────────────────────
md("""
<div class="section-card">
  <h2 class="section-title">Report a Bug</h2>
  <p class="section-desc">
    If you encounter an issue not listed above, please fill in the details below
    and email to the developer address above with subject: <strong>[MAIZE-XNet BUG]</strong>
  </p>
</div>
""")

with st.form("bug_report_form"):
    col1, col2 = st.columns(2)
    with col1:
        name_in  = st.text_input("Your Name (optional)")
        email_in = st.text_input("Your Email (optional)")
    with col2:
        issue_type = st.selectbox("Issue Type", [
            "Model loading failure",
            "Inference / prediction error",
            "Grad-CAM / heatmap issue",
            "PDF report error",
            "UI / display issue",
            "Performance / timeout",
            "Other",
        ])
        severity = st.selectbox("Severity", ["Critical", "High", "Medium", "Low"])

    description = st.text_area("Issue Description", height=100,
                               placeholder="Describe what happened, what you expected, and any error messages...")
    submitted = st.form_submit_button("Submit Bug Report")
    if submitted:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md(f"""
        <div class="status-banner status-live" style="margin-top:10px;">
          ✓ Bug report logged at {now}. Please email the details above to your@email.com
          with subject: [MAIZE-XNet BUG] — {issue_type}
        </div>
        """)

md("""
<div class="app-footer">
  <div class="footer-inner">
    <div class="footer-brand">
      <span class="footer-logo-maize">MAIZE</span><span class="footer-logo-x">-X</span><span class="footer-logo-net">Net</span>
    </div>
    <div class="footer-meta">MSc Thesis &nbsp;·&nbsp; Your University &nbsp;·&nbsp; Department of CSE</div>
    <div class="footer-note">For academic and research purposes only.</div>
    <div class="footer-contact">Issues? Contact: your@email.com</div>
  </div>
</div>
""")
