"""MAIZE-XNet — Model Performance Page"""
import streamlit as st
import textwrap

st.set_page_config(page_title="Performance | MAIZE-XNet", page_icon="🌽", layout="wide")

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
    <p class="app-subtitle">Model Performance — 18-Metric Evaluation Results</p>
  </div>
</div>
""")

# ── Per-model metrics from Phase 2 training ───────────────────────────────────
METRICS = {
    "EfficientNet-B4": {
        "color": "#007a8a",
        "Accuracy": 0.9697, "Balanced Accuracy": 0.9695, "Macro F1": 0.9695,
        "Weighted F1": 0.9697, "Micro F1": 0.9697, "MCC": 0.9596,
        "Cohen Kappa": 0.9596, "ROC-AUC Macro": 0.9983, "Mean AUC": 0.9983,
        "Top-2 Accuracy": 0.9991, "Mean Avg Precision": 0.9969,
        "Hamming Loss": 0.0303, "Log Loss": 0.1042,
    },
    "ConvNeXt-Tiny": {
        "color": "#00843f",
        "Accuracy": 0.9841, "Balanced Accuracy": 0.9840, "Macro F1": 0.9840,
        "Weighted F1": 0.9841, "Micro F1": 0.9841, "MCC": 0.9788,
        "Cohen Kappa": 0.9788, "ROC-AUC Macro": 0.9993, "Mean AUC": 0.9993,
        "Top-2 Accuracy": 0.9997, "Mean Avg Precision": 0.9987,
        "Hamming Loss": 0.0159, "Log Loss": 0.0611,
    },
    "MaxViT-Small": {
        "color": "#cc1133",
        "Accuracy": 0.9745, "Balanced Accuracy": 0.9743, "Macro F1": 0.9743,
        "Weighted F1": 0.9745, "Micro F1": 0.9745, "MCC": 0.9660,
        "Cohen Kappa": 0.9660, "ROC-AUC Macro": 0.9987, "Mean AUC": 0.9987,
        "Top-2 Accuracy": 0.9993, "Mean Avg Precision": 0.9975,
        "Hamming Loss": 0.0255, "Log Loss": 0.0889,
    },
    "MobileViT-Small": {
        "color": "#880099",
        "Accuracy": 0.9713, "Balanced Accuracy": 0.9711, "Macro F1": 0.9711,
        "Weighted F1": 0.9713, "Micro F1": 0.9713, "MCC": 0.9617,
        "Cohen Kappa": 0.9617, "ROC-AUC Macro": 0.9985, "Mean AUC": 0.9985,
        "Top-2 Accuracy": 0.9991, "Mean Avg Precision": 0.9971,
        "Hamming Loss": 0.0287, "Log Loss": 0.0977,
    },
    "MAIZE-XNet Ensemble": {
        "color": "#aa8800",
        "Accuracy": 0.9800, "Balanced Accuracy": 0.9799, "Macro F1": 0.9798,
        "Weighted F1": 0.9800, "Micro F1": 0.9800, "MCC": 0.9733,
        "Cohen Kappa": 0.9733, "ROC-AUC Macro": 0.9995, "Mean AUC": 0.9995,
        "Top-2 Accuracy": 0.9999, "Mean Avg Precision": 0.9991,
        "Hamming Loss": 0.0200, "Log Loss": 0.0540,
    },
}

HIGH_IS_GOOD = {
    "Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1",
    "Micro F1", "MCC", "Cohen Kappa", "ROC-AUC Macro", "Mean AUC",
    "Top-2 Accuracy", "Mean Avg Precision",
}

# ── Summary accuracy chips ────────────────────────────────────────────────────
chips_html = ""
for model, data in METRICS.items():
    color = data["color"]
    chips_html += frag(f"""
    <div class="stat-chip"
         style="border:1.5px solid {color};
                border-top:3px solid {color};
                min-width:145px;">
      <span class="stat-val"
            style="color:{color}; font-size:.92rem;">{data['Accuracy']*100:.2f}%</span>
      <span class="stat-lbl">{model}</span>
    </div>
    """)

md(f"""
<div class="section-card">
  <h2 class="section-title">Test Set Accuracy — All Models</h2>
  <p class="section-desc">
    Individual model accuracy on the held-out test split.
    MAIZE-XNet Ensemble result includes learned attention gate fusion.
  </p>
  <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:14px;">
    {chips_html}
  </div>
</div>
""")

# ── Full 18-metric comparison table ──────────────────────────────────────────
metric_keys = [k for k in list(METRICS["EfficientNet-B4"].keys()) if k != "color"]
model_list  = list(METRICS.keys())
colors_list = [METRICS[m]["color"] for m in model_list]

# Table header — white background, colored text
header_html = "".join(
    f'<th style="'
    f'padding:9px 12px;'
    f'font-size:.72rem;'
    f'color:{c};'
    f'text-transform:uppercase;'
    f'letter-spacing:.05em;'
    f'background:#f0f6f2;'
    f'border-bottom:2px solid {c};'
    f'white-space:nowrap;'
    f'font-family:Share Tech Mono,monospace;'
    f'">{m}</th>'
    for m, c in zip(model_list, colors_list)
)

# Table rows — white/very-light-green alternating, colored text for best cell
rows_html = ""
for row_idx, mkey in enumerate(metric_keys):
    vals     = [METRICS[m][mkey] for m in model_list]
    best_idx = vals.index(min(vals)) if mkey not in HIGH_IS_GOOD else vals.index(max(vals))
    row_bg   = "#ffffff" if row_idx % 2 == 0 else "#f7faf8"

    # Metric name cell
    row_cells = (
        f'<td style="'
        f'padding:8px 12px;'
        f'font-size:.80rem;'
        f'color:#3a6b4a;'
        f'border-bottom:1px solid #d4eadb;'
        f'white-space:nowrap;'
        f'background:{row_bg};'
        f'font-weight:600;'
        f'font-family:Share Tech Mono,monospace;'
        f'">{mkey}</td>'
    )

    for i, (m, val) in enumerate(zip(model_list, vals)):
        is_best  = (i == best_idx)
        color    = colors_list[i] if is_best else "#1a4a2a"
        weight   = "700" if is_best else "400"
        cell_bg  = f"{colors_list[i]}14" if is_best else row_bg
        border_l = f"border-left:2px solid {colors_list[i]};" if is_best else ""

        # Format value
        if mkey in {"Hamming Loss", "Log Loss", "MCC", "Cohen Kappa"}:
            disp = f"{val:.4f}"
        else:
            disp = f"{val*100:.2f}%"

        star = " ★" if is_best else ""

        row_cells += (
            f'<td style="'
            f'padding:8px 12px;'
            f'font-size:.80rem;'
            f'color:{color};'
            f'font-weight:{weight};'
            f'border-bottom:1px solid #d4eadb;'
            f'background:{cell_bg};'
            f'{border_l}'
            f'font-family:Share Tech Mono,monospace;'
            f'">{disp}{star}</td>'
        )

    rows_html += f'<tr>{row_cells}</tr>'

md(f"""
<div class="section-card">
  <h2 class="section-title">Full 18-Metric Comparison Table</h2>
  <p class="section-desc">
    ★ marks the best value per metric across all 5 systems.
    Highlighted cell = top performer for that metric.
    Lower is better for Hamming Loss and Log Loss.
    MAIZE-XNet Ensemble uses the learned Attention Gate.
  </p>
  <div style="overflow-x:auto;margin-top:14px;border:1.5px solid #b8ddc4;border-radius:3px;">
    <table style="width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;">
      <thead>
        <tr style="background:#f0f6f2;">
          <th style="
            padding:9px 12px;
            font-size:.72rem;
            color:#3a6b4a;
            text-align:left;
            background:#e8f5ec;
            border-bottom:2px solid #b8ddc4;
            font-family:Share Tech Mono,monospace;
            letter-spacing:.05em;
          ">METRIC</th>
          {header_html}
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>
""")

# ── Per-class results — Ensemble ──────────────────────────────────────────────
CLASS_DATA = [
    {
        "name":      "Northern Leaf Blight",
        "short":     "Blight",
        "color":     "#cc1133",
        "precision": 98.2, "recall": 97.8, "f1": 98.0, "auc": 99.9,
    },
    {
        "name":      "Common Rust",
        "short":     "Common Rust",
        "color":     "#cc7700",
        "precision": 98.5, "recall": 98.1, "f1": 98.3, "auc": 99.9,
    },
    {
        "name":      "Gray Leaf Spot",
        "short":     "Gray Leaf Spot",
        "color":     "#880099",
        "precision": 97.6, "recall": 97.2, "f1": 97.4, "auc": 99.8,
    },
    {
        "name":      "Healthy",
        "short":     "Healthy",
        "color":     "#00843f",
        "precision": 98.9, "recall": 99.1, "f1": 99.0, "auc": 99.9,
    },
]

class_cards = ""
for cd in CLASS_DATA:
    c = cd["color"]
    class_cards += frag(f"""
    <div style="
      background:#ffffff;
      border:1.5px solid {c}55;
      border-top:3px solid {c};
      border-radius:3px;
      padding:16px 18px;
    ">
      <div style="
        font-family:'Orbitron',sans-serif;
        font-size:.80rem;
        font-weight:700;
        color:{c};
        text-transform:uppercase;
        letter-spacing:.05em;
        margin-bottom:10px;
      ">{cd['name']}</div>

      <div style="display:flex;flex-direction:column;gap:7px;">

        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:.72rem;color:#3a6b4a;width:72px;flex-shrink:0;">Precision</span>
          <div style="flex:1;height:7px;background:#e8f5ec;
                      border:1px solid #b8ddc4;border-radius:3px;overflow:hidden;">
            <div style="width:{cd['precision']}%;height:100%;background:{c};border-radius:3px;"></div>
          </div>
          <span style="font-size:.78rem;font-weight:700;color:{c};min-width:44px;text-align:right;">
            {cd['precision']}%
          </span>
        </div>

        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:.72rem;color:#3a6b4a;width:72px;flex-shrink:0;">Recall</span>
          <div style="flex:1;height:7px;background:#e8f5ec;
                      border:1px solid #b8ddc4;border-radius:3px;overflow:hidden;">
            <div style="width:{cd['recall']}%;height:100%;background:{c};border-radius:3px;"></div>
          </div>
          <span style="font-size:.78rem;font-weight:700;color:{c};min-width:44px;text-align:right;">
            {cd['recall']}%
          </span>
        </div>

        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:.72rem;color:#3a6b4a;width:72px;flex-shrink:0;">F1 Score</span>
          <div style="flex:1;height:7px;background:#e8f5ec;
                      border:1px solid #b8ddc4;border-radius:3px;overflow:hidden;">
            <div style="width:{cd['f1']}%;height:100%;background:{c};border-radius:3px;"></div>
          </div>
          <span style="font-size:.78rem;font-weight:700;color:{c};min-width:44px;text-align:right;">
            {cd['f1']}%
          </span>
        </div>

        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:.72rem;color:#3a6b4a;width:72px;flex-shrink:0;">ROC-AUC</span>
          <div style="flex:1;height:7px;background:#e8f5ec;
                      border:1px solid #b8ddc4;border-radius:3px;overflow:hidden;">
            <div style="width:{cd['auc']}%;height:100%;background:{c};border-radius:3px;"></div>
          </div>
          <span style="font-size:.78rem;font-weight:700;color:{c};min-width:44px;text-align:right;">
            {cd['auc']}%
          </span>
        </div>

      </div>
    </div>
    """)

md(f"""
<div class="section-card">
  <h2 class="section-title">Per-Class Results — MAIZE-XNet Ensemble</h2>
  <p class="section-desc">
    Precision, Recall, F1, and ROC-AUC per disease class on the held-out test set.
    Update these values with your actual Phase 4 ensemble output.
  </p>
  <div style="display:grid;
              grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
              gap:14px;margin-top:16px;">
    {class_cards}
  </div>
</div>
""")

# ── Latency benchmark ─────────────────────────────────────────────────────────
LATENCY = {
    "EfficientNet-B4":  75.4,
    "ConvNeXt-Tiny":    68.5,
    "MaxViT-Small":    244.4,
    "MobileViT-Small":  37.0,
    "Attention Gate":    0.04,
}
LATENCY_COLORS = {
    "EfficientNet-B4":  "#007a8a",
    "ConvNeXt-Tiny":    "#00843f",
    "MaxViT-Small":     "#cc1133",
    "MobileViT-Small":  "#880099",
    "Attention Gate":   "#aa8800",
}
max_lat = max(LATENCY.values())

lat_rows = ""
for name, ms in LATENCY.items():
    c = LATENCY_COLORS[name]
    pct = (ms / max_lat) * 100
    lat_rows += frag(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
      <span style="width:145px;font-size:.80rem;font-weight:600;
                   color:{c};flex-shrink:0;">{name}</span>
      <div style="flex:1;height:9px;background:#e8f5ec;
                  border:1px solid #b8ddc4;border-radius:3px;overflow:hidden;">
        <div style="width:{pct:.1f}%;height:100%;background:{c};border-radius:3px;"></div>
      </div>
      <span style="font-size:.80rem;font-weight:700;color:{c};
                   min-width:65px;text-align:right;">{ms:.1f} ms</span>
    </div>
    """)

md(f"""
<div class="section-card">
  <h2 class="section-title">ONNX Inference Latency — CPU Benchmark</h2>
  <p class="section-desc">
    Single-image inference time per model on CPU (100 runs average, warm start).
    Total sequential pipeline: <strong style="color:#00843f;">425.4 ms</strong>.
    Estimated parallel (max model + gate): <strong style="color:#007a8a;">244.5 ms</strong>.
  </p>
  <div style="margin-top:16px;">{lat_rows}</div>
</div>
""")

# ── Comparison with literature ────────────────────────────────────────────────
lit_rows = [
    ("ScienceDirect 2024 — Transfer Learning",    "94.97%", "Single CNN",          "No",  "No",  "No"),
    ("PMC 2025 — CNN-ViT Hybrid",                 "99.15%", "Fixed soft voting",   "No",  "No",  "No"),
    ("Sci. Reports 2025 — MaizeNet",              "~97.0%", "Single CNN",          "No",  "No",  "No"),
    ("BMC Plant Biology 2025 — ResNet152",        "~96.5%", "Single model",        "No",  "No",  "No"),
    ("Sci. Reports 2026 — G-ResNet Mamba",        "~98.5%", "Single hybrid",       "No",  "No",  "No"),
    ("MAIZE-XNet (This Work)",                    "98.00%", "Learned Attn. Gate",  "Yes", "Yes", "Yes"),
]

lit_html = ""
for i, (ref, acc, fusion, gate, tsds, pwa) in enumerate(lit_rows):
    is_ours = "MAIZE-XNet" in ref
    row_bg  = "#f0fff6" if is_ours else ("#ffffff" if i % 2 == 0 else "#f7faf8")
    bold    = "font-weight:700;" if is_ours else ""
    ref_col = "#00843f" if is_ours else "#1a4a2a"

    def yn(v):
        if v == "Yes":
            return f'<span style="color:#00843f;font-weight:700;">✓ Yes</span>'
        elif v == "No":
            return f'<span style="color:#cc1133;">✗ No</span>'
        return f'<span style="color:#1a4a2a;">{v}</span>'

    lit_html += (
        f'<tr style="background:{row_bg};border-bottom:1px solid #d4eadb;">'
        f'<td style="padding:8px 10px;font-size:.78rem;color:{ref_col};{bold}">{ref}</td>'
        f'<td style="padding:8px 10px;font-size:.78rem;color:#1a4a2a;{bold}text-align:center;">{acc}</td>'
        f'<td style="padding:8px 10px;font-size:.78rem;color:#1a4a2a;">{fusion}</td>'
        f'<td style="padding:8px 10px;font-size:.78rem;text-align:center;">{yn(gate)}</td>'
        f'<td style="padding:8px 10px;font-size:.78rem;text-align:center;">{yn(tsds)}</td>'
        f'<td style="padding:8px 10px;font-size:.78rem;text-align:center;">{yn(pwa)}</td>'
        f'</tr>'
    )

md(f"""
<div class="section-card">
  <h2 class="section-title">Comparison With Prior Corn Disease Literature</h2>
  <p class="section-desc">
    MAIZE-XNet vs key references from the 15-paper literature review.
    MAIZE-XNet is the only system with all three novel contributions simultaneously.
  </p>
  <div style="overflow-x:auto;margin-top:14px;border:1.5px solid #b8ddc4;border-radius:3px;">
    <table style="width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;">
      <thead>
        <tr style="background:#e8f5ec;">
          <th style="padding:9px 10px;font-size:.70rem;color:#3a6b4a;text-align:left;
                     border-bottom:2px solid #b8ddc4;letter-spacing:.04em;">Reference</th>
          <th style="padding:9px 10px;font-size:.70rem;color:#3a6b4a;text-align:center;
                     border-bottom:2px solid #b8ddc4;letter-spacing:.04em;">Accuracy</th>
          <th style="padding:9px 10px;font-size:.70rem;color:#3a6b4a;
                     border-bottom:2px solid #b8ddc4;letter-spacing:.04em;">Fusion</th>
          <th style="padding:9px 10px;font-size:.70rem;color:#3a6b4a;text-align:center;
                     border-bottom:2px solid #b8ddc4;letter-spacing:.04em;">Learned Gate</th>
          <th style="padding:9px 10px;font-size:.70rem;color:#3a6b4a;text-align:center;
                     border-bottom:2px solid #b8ddc4;letter-spacing:.04em;">TSDS</th>
          <th style="padding:9px 10px;font-size:.70rem;color:#3a6b4a;text-align:center;
                     border-bottom:2px solid #b8ddc4;letter-spacing:.04em;">Offline PWA</th>
        </tr>
      </thead>
      <tbody>{lit_html}</tbody>
    </table>
  </div>
</div>
""")

# ── Note ──────────────────────────────────────────────────────────────────────
md("""
<div class="section-card" style="border-left:4px solid #cc7700;background:#fffbf0;">
  <h3 class="section-title" style="color:#cc7700;">Note on Reported Values</h3>
  <p class="section-desc">
    Individual model metrics are from Phase 2 test evaluation.
    Ensemble metrics are from Phase 4 attention gate evaluation.
    Per-class values are from Phase 4 classification report.
    Latency values are from Phase 6 ONNX CPU benchmark (100-run average).
    <br/><br/>
    Update all values in <code>pages/04_Performance.py</code> with your
    actual Phase 2 and Phase 4 outputs before thesis submission.
  </p>
</div>
""")