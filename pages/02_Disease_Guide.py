"""MAIZE-XNet — Corn Disease Guide"""
import streamlit as st
import textwrap

st.set_page_config(page_title="Disease Guide | MAIZE-XNet", page_icon="🌽", layout="wide")

def md(html): st.markdown(textwrap.dedent(html), unsafe_allow_html=True)
def frag(html): return textwrap.dedent(html).strip()

with open("static/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

md("""
<div class="app-header">
  <div class="header-left">
    <div class="logo-wordmark">
      <span class="logo-maize">MAIZE</span><span class="logo-x">-X</span><span class="logo-net">Net</span>
    </div>
    <p class="app-subtitle">Corn Leaf Disease Reference Guide</p>
  </div>
</div>
""")

DISEASES = [
    {
        "name": "Northern Leaf Blight",
        "class": "Blight",
        "color": "#ff3b5c",
        "pathogen": "Exserohilum turcicum",
        "severity": "High",
        "symptoms": "Cigar-shaped, tan to gray-green lesions 2.5–15 cm long. Starts on lower leaves, "
                    "progresses upward. Lesions have distinct borders parallel to leaf veins.",
        "conditions": "Cool temperatures (18–27°C), high humidity, extended leaf wetness (>6 hours).",
        "yield_loss": "Up to 50% in severe infections on susceptible hybrids.",
        "treatment": "Apply mancozeb, propiconazole, or azoxystrobin at first sign of disease.",
        "prevention": "Plant Ht1/Ht2 resistant hybrids. Crop rotation. Bury infected residue.",
    },
    {
        "name": "Common Rust",
        "class": "Common_Rust",
        "color": "#ffb627",
        "pathogen": "Puccinia sorghi",
        "severity": "Medium",
        "symptoms": "Small (1–2mm), round to elongated, powdery pustules on both leaf surfaces. "
                    "Golden-brown to brick-red urediniospores. Dense coverage causes yellowing.",
        "conditions": "Warm days (16–23°C), cool nights, high humidity, airborne spore dispersal.",
        "yield_loss": "10–30% on susceptible varieties. Rarely causes major loss on resistant hybrids.",
        "treatment": "Triazole (tebuconazole) or strobilurin (azoxystrobin) fungicides at first pustules.",
        "prevention": "Use rust-resistant varieties (Rp gene resistance). Early planting. Scout weekly.",
    },
    {
        "name": "Gray Leaf Spot",
        "class": "Gray_Leaf_Spot",
        "color": "#ff3df0",
        "pathogen": "Cercospora zeae-maydis",
        "severity": "High",
        "symptoms": "Rectangular, tan to gray lesions with parallel edges bounded by leaf veins. "
                    "Grayish sporulation visible under humid conditions. Lesions may coalesce.",
        "conditions": "Warm temperatures (25–30°C), prolonged leaf wetness, high relative humidity, "
                      "conservation tillage increases inoculum.",
        "yield_loss": "Up to 60% in severe outbreaks. Most damaging foliar disease under no-till.",
        "treatment": "Strobilurin + triazole mixtures applied at VT/R1 growth stages.",
        "prevention": "Crop rotation (minimum 1 year). Tillage to bury residue. Resistant hybrids.",
    },
    {
        "name": "Healthy Corn Leaf",
        "class": "Healthy",
        "color": "#39ff88",
        "pathogen": "None",
        "severity": "None",
        "symptoms": "Vibrant green coloration with no visible lesions, pustules, or discoloration. "
                    "Uniform leaf surface with normal texture and venation pattern.",
        "conditions": "Optimal growth conditions: 25–33°C, adequate moisture, balanced nutrition.",
        "yield_loss": "No yield loss. Continue monitoring for early disease onset.",
        "treatment": "No treatment needed. Maintain current agronomic practices.",
        "prevention": "Balanced NPK fertilization. Consistent irrigation. Weekly scouting from V6 stage.",
    },
]

for d in DISEASES:
    sev_color = {"None": "#39ff88", "Medium": "#ffb627", "High": "#ff3b5c"}.get(d["severity"], "#888")
    md(frag(f"""
    <div class="section-card" style="border-left:4px solid {d['color']};">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
        <div>
          <h3 class="section-title" style="color:{d['color']};">{d['name']}</h3>
          <span style="font-size:.75rem;color:#3a7a5a;font-style:italic;">
            Class: {d['class']} &nbsp;·&nbsp; Pathogen: <i>{d['pathogen']}</i>
          </span>
        </div>
        <span class="severity-badge"
              style="background:{sev_color}22;color:{sev_color};border:1px solid {sev_color}55;">
          {d['severity']} Risk
        </span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-top:14px;">
        <div>
          <div class="action-label">Symptoms</div>
          <p class="action-text">{d['symptoms']}</p>
        </div>
        <div>
          <div class="action-label">Favorable Conditions</div>
          <p class="action-text">{d['conditions']}</p>
        </div>
        <div>
          <div class="action-label" style="color:#ffb627;">Yield Impact</div>
          <p class="action-text">{d['yield_loss']}</p>
        </div>
        <div>
          <div class="action-label">Treatment</div>
          <p class="action-text">{d['treatment']}</p>
        </div>
        <div>
          <div class="action-label" style="color:#39ff88;">Prevention</div>
          <p class="action-text">{d['prevention']}</p>
        </div>
      </div>
    </div>
    """))
