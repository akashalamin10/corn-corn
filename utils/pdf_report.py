"""
MAIZE-XNet PDF Diagnostic Report Generator
Builds a professional multi-section PDF using reportlab.
Includes: diagnosis summary, Grad-CAM heatmaps, TSDS, gate weights,
class probabilities, treatment recommendations, and system info.

THEME: Clean white-background report. All text uses dark green/charcoal
tones for contrast against white/light-gray panels — no leftover dark
panels from the original dark-mode design remain.
"""

import io
import datetime
import numpy as np
from PIL import Image


def build_pdf(image, pred_class, final_probs, individual_probs,
              gate_weights, tsds_score, class_names, class_info,
              model_names, gradcam_maps=None):
    """
    Build and return PDF bytes for the MAIZE-XNet diagnostic report.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor, white
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, Image as RLImage, KeepTogether, PageBreak
        )
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.platypus.flowables import Flowable
    except ImportError:
        raise ImportError("reportlab is required: pip install reportlab")

    # ── Colors — WHITE BACKGROUND PDF, dark text throughout ─────────────────
    GREEN_DARK   = HexColor("#005c2a")   # used only for header banner bg
    GREEN_MID    = HexColor("#00843f")   # accent lines, bars
    GREEN_LIGHT  = HexColor("#00a84f")   # H2 headings (readable on white)
    GREEN_BG     = HexColor("#ffffff")   # main panel background = white
    ROW_ALT_BG   = HexColor("#f2f8f4")   # light green-gray for alt rows (NOT dark)
    HEADER_BG    = GREEN_DARK            # table header row background
    CORN_YELLOW  = HexColor("#aa8800")
    CYAN         = HexColor("#007a8a")
    RED          = HexColor("#cc1133")
    AMBER        = HexColor("#cc7700")
    MAG          = HexColor("#880099")

    TEXT_DARK    = HexColor("#10231a")   # primary body text — near-black green, dark enough for white bg
    TEXT_MID     = HexColor("#3a6b4a")   # secondary/dim text — still readable on white
    BORDER       = HexColor("#b8ddc4")
    WHITE        = white
    WHITE_HEX    = HexColor("#ffffff")

    SEV_COLORS = {
        "None":   HexColor("#00843f"),
        "Medium": HexColor("#b35c00"),
        "High":   HexColor("#cc1133"),
    }
    MODEL_COLORS_HEX = ["#007a8a", "#00843f", "#cc1133", "#880099"]
    MODEL_COLORS     = [HexColor(h) for h in MODEL_COLORS_HEX]

    info       = class_info[pred_class]
    pred_idx   = int(np.argmax(final_probs))
    confidence = float(final_probs[pred_idx])
    now        = datetime.datetime.now()

    # TSDS label — colors kept vivid but readable; backgrounds for these
    # boxes are light tints, not dark panels (set further down).
    if tsds_score >= 0.50:
        tsds_text  = "High Stability"
        tsds_color = HexColor("#00843f")
        tsds_box_bg = HexColor("#eafaf0")
        tsds_desc  = "High saliency stability — consistent attention region across all augmented passes."
    elif tsds_score >= 0.30:
        tsds_text  = "Moderate Stability"
        tsds_color = HexColor("#b35c00")
        tsds_box_bg = HexColor("#fff6ea")
        tsds_desc  = "Partial spatial consistency — moderate prediction trust."
    else:
        tsds_text  = "Low Stability"
        tsds_color = HexColor("#cc1133")
        tsds_box_bg = HexColor("#fdeeee")
        tsds_desc  = "Significant saliency drift detected — expert verification recommended."

    # ── Document ──────────────────────────────────────────────────────────────
    buf    = io.BytesIO()
    W, H   = A4
    margin = 16 * mm

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=10 * mm, bottomMargin=14 * mm,
        title="MAIZE-XNet Diagnostic Report",
        author="MAIZE-XNet System",
        subject="Corn Leaf Disease Diagnosis"
    )

    usable_w = W - 2 * margin

    # ── Styles ────────────────────────────────────────────────────────────────
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    # Header banner text stays WHITE because its background (GREEN_DARK) is dark.
    H1   = S("H1",   fontName="Helvetica-Bold",   fontSize=20, textColor=WHITE_HEX,  leading=24, spaceAfter=2)
    H1s  = S("H1s",  fontName="Helvetica-Bold",   fontSize=13, textColor=WHITE_HEX,  leading=16, spaceAfter=2)

    # Everything below sits on white/light backgrounds → dark text.
    H2   = S("H2",   fontName="Helvetica-Bold",   fontSize=13, textColor=GREEN_LIGHT, leading=17, spaceAfter=4)
    H3   = S("H3",   fontName="Helvetica-Bold",   fontSize=10, textColor=CYAN,        leading=13, spaceAfter=3)
    BODY = S("BODY", fontName="Helvetica",         fontSize=9,  textColor=TEXT_DARK,  leading=14, spaceAfter=2)
    BDIM = S("BDIM", fontName="Helvetica",         fontSize=9,  textColor=TEXT_MID,   leading=13)
    SMAL = S("SMAL", fontName="Helvetica",         fontSize=8,  textColor=TEXT_MID,   leading=11)
    LBL  = S("LBL",  fontName="Helvetica-Bold",    fontSize=8,  textColor=TEXT_MID,   leading=10)
    CTR  = S("CTR",  fontName="Helvetica",         fontSize=9,  textColor=TEXT_DARK,  leading=12, alignment=TA_CENTER)
    CORN = S("CORN", fontName="Helvetica-Bold",    fontSize=18, textColor=CORN_YELLOW, leading=22)

    # Table header rows have a dark green background → their text must stay white.
    LBL_ON_DARK = S("LBL_ON_DARK", fontName="Helvetica-Bold", fontSize=8, textColor=WHITE_HEX, leading=10)

    # ── Custom Flowables ──────────────────────────────────────────────────────
    class ColorBar(Flowable):
        def __init__(self, value, max_val=1.0, color=GREEN_MID, width=None, height=8):
            super().__init__()
            self.value     = value
            self.max_val   = max_val
            self.color     = color
            self.bar_width = width or (usable_w - 20)
            self.bar_h     = height

        def wrap(self, *a):
            return self.bar_width, self.bar_h + 4

        def draw(self):
            c   = self.canv
            pct = min(self.value / self.max_val, 1.0)
            c.setFillColor(BORDER)
            c.roundRect(0, 0, self.bar_width, self.bar_h, 3, fill=1, stroke=0)
            if pct > 0:
                c.setFillColor(self.color)
                c.roundRect(0, 0, self.bar_width * pct, self.bar_h, 3, fill=1, stroke=0)

    def pil_to_rl(pil_img, max_w=None, max_h=None):
        b2 = io.BytesIO()
        pil_img.save(b2, format="PNG")
        b2.seek(0)
        ri = RLImage(b2)
        if max_w and ri.drawWidth > max_w:
            r = max_w / ri.drawWidth
            ri.drawWidth  = max_w
            ri.drawHeight = ri.drawHeight * r
        if max_h and ri.drawHeight > max_h:
            r = max_h / ri.drawHeight
            ri.drawHeight = max_h
            ri.drawWidth  = ri.drawWidth * r
        return ri

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    thumb = image.copy()
    thumb.thumbnail((200, 200), Image.LANCZOS)

    # ── Story ─────────────────────────────────────────────────────────────────
    story = []

    # ── PAGE 1: HEADER BANNER (dark green bg, white text — intentional) ──────
    header_data = [[
        Paragraph("🌽  MAIZE-XNet", H1),
        Paragraph("Corn Disease Diagnostic Report", H1s),
    ]]
    hdr_t = Table(header_data, colWidths=[usable_w * 0.50, usable_w * 0.50])
    hdr_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), GREEN_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (0, 0),  20),
        ("RIGHTPADDING",  (-1, 0), (-1, 0), 16),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (1, 0), (1, 0),   "RIGHT"),
        ("LINEBELOW",     (0, 0), (-1, 0),  2, GREEN_LIGHT),
    ]))
    story.append(hdr_t)
    story.append(Spacer(1, 3 * mm))

    # Metadata row — white background, dark text
    meta_data = [[
        Paragraph(f"<b>Date:</b> {now.strftime('%B %d, %Y')}", BODY),
        Paragraph(f"<b>Time:</b> {now.strftime('%H:%M:%S')}", BODY),
        Paragraph(f"<b>System:</b> MAIZE-XNet v1.0", BODY),
        Paragraph(f"<b>Dataset:</b> 4-Class Corn", BODY),
    ]]
    meta_t = Table(meta_data, colWidths=[usable_w / 4] * 4)
    meta_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), GREEN_BG),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 5 * mm))

    # ── SECTION 1: DIAGNOSIS SUMMARY ─────────────────────────────────────────
    story.append(Paragraph("1. Diagnosis Summary", H2))
    story.append(HRFlowable(width=usable_w, thickness=1.5, color=GREEN_MID, spaceAfter=4))

    sev_color = SEV_COLORS.get(info["severity"], TEXT_MID)

    diag_left = [
        [Paragraph("DETECTED DISEASE", LBL)],
        [Paragraph(info["display"],
                   ParagraphStyle("dn", fontName="Helvetica-Bold", fontSize=17,
                                  textColor=sev_color, leading=21))],
        [Spacer(1, 2 * mm)],
        [Paragraph(f"<b>Pathogen:</b> <i>{info['pathogen']}</i>", BODY)],
        [Paragraph(f"<b>Severity:</b> {info['severity']}", BODY)],
        [Spacer(1, 2 * mm)],
        [Paragraph(f"<b>Description:</b> {info['description']}", BDIM)],
        [Spacer(1, 3 * mm)],
        [Paragraph(f"<b>Ensemble Confidence:</b> {confidence*100:.1f}%", BODY)],
        [ColorBar(confidence, color=sev_color, width=usable_w * 0.50, height=10)],
        [Spacer(1, 3 * mm)],
        [Paragraph(f"<b>TSDS Trust Score:</b> {tsds_score:.4f} — {tsds_text}", BODY)],
        [ColorBar(tsds_score, color=tsds_color, width=usable_w * 0.50, height=10)],
        [Spacer(1, 1 * mm)],
        [Paragraph(tsds_desc, BDIM)],
    ]
    diag_left_t = Table(diag_left, colWidths=[usable_w * 0.56])
    diag_left_t.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
    ]))

    img_rl   = pil_to_rl(thumb, max_w=usable_w * 0.36, max_h=58 * mm)
    diag_row = Table([[diag_left_t, img_rl]], colWidths=[usable_w * 0.60, usable_w * 0.40])
    diag_row.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND",    (0, 0), (-1, -1), GREEN_BG),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (0, 0),  14),
        ("LEFTPADDING",   (1, 0), (1, 0),  8),
        ("RIGHTPADDING",  (-1, 0), (-1, 0), 14),
    ]))
    story.append(diag_row)
    story.append(Spacer(1, 5 * mm))

    # ── SECTION 2: TREATMENT & PREVENTION ────────────────────────────────────
    story.append(Paragraph("2. Clinical Recommendations", H2))
    story.append(HRFlowable(width=usable_w, thickness=1.5, color=GREEN_MID, spaceAfter=4))

    # FIX: both panels now use light backgrounds (not black) with dark text.
    rec_data = [
        [Paragraph("⚕  Recommended Treatment", H3),
         Paragraph("🛡  Prevention Measures", H3)],
        [Paragraph(info["treatment"], BODY),
         Paragraph(info["prevention"], BODY)],
    ]
    rec_t = Table(rec_data, colWidths=[usable_w / 2 - 4, usable_w / 2 - 4])
    rec_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), HexColor("#fdf2f2")),   # very light red tint
        ("BACKGROUND",    (1, 0), (1, -1), HexColor("#eefaf2")),   # very light green tint
        ("BOX",           (0, 0), (0, -1), 0.5, RED),
        ("BOX",           (1, 0), (1, -1), 0.5, GREEN_MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(rec_t)
    story.append(Spacer(1, 5 * mm))

    # ── SECTION 3: INDIVIDUAL MODEL PREDICTIONS ───────────────────────────────
    story.append(Paragraph("3. Individual Model Predictions", H2))
    story.append(HRFlowable(width=usable_w, thickness=1.5, color=GREEN_MID, spaceAfter=4))

    model_rows = [[
        Paragraph("Model", LBL_ON_DARK), Paragraph("Prediction", LBL_ON_DARK),
        Paragraph("Confidence", LBL_ON_DARK), Paragraph("Gate Weight", LBL_ON_DARK),
        Paragraph("Agreement", LBL_ON_DARK),
    ]]
    for i, (mname, mhex, mcolor) in enumerate(
            zip(model_names, MODEL_COLORS_HEX, MODEL_COLORS)):
        m_pred_idx  = int(np.argmax(individual_probs[i]))
        m_pred_name = class_info[class_names[m_pred_idx]]["display"]
        m_conf      = float(individual_probs[i].max())
        m_weight    = float(gate_weights[i])
        agrees      = m_pred_idx == pred_idx
        model_rows.append([
            Paragraph(f'<font color="{mhex}"><b>{mname}</b></font>', BODY),
            Paragraph(m_pred_name, BODY),
            Paragraph(f"{m_conf*100:.1f}%", BODY),
            Paragraph(f"{m_weight*100:.1f}%", BODY),
            Paragraph(
                f'<font color="{"#00843f" if agrees else "#cc1133"}"><b>'
                f'{"✓ Agrees" if agrees else "✗ Differs"}</b></font>', BODY),
        ])

    col_w = [usable_w * f for f in [0.26, 0.26, 0.16, 0.16, 0.16]]
    mod_t = Table(model_rows, colWidths=col_w)
    mod_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), GREEN_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GREEN_BG, ROW_ALT_BG]),   # FIX: light alt rows, not dark
        ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
    ]))
    story.append(mod_t)
    story.append(Spacer(1, 5 * mm))

    # ── SECTION 4: CLASS PROBABILITY TABLE ───────────────────────────────────
    story.append(Paragraph("4. Ensemble Final Class Probabilities", H2))
    story.append(HRFlowable(width=usable_w, thickness=1.5, color=GREEN_MID, spaceAfter=4))

    sorted_idx = np.argsort(final_probs)[::-1]
    prob_rows  = [[
        Paragraph("Rank", LBL_ON_DARK), Paragraph("Disease Class", LBL_ON_DARK),
        Paragraph("Severity", LBL_ON_DARK), Paragraph("Probability", LBL_ON_DARK),
        Paragraph("Confidence Bar", LBL_ON_DARK),
    ]]
    for rank, idx in enumerate(sorted_idx, 1):
        cn     = class_names[idx]
        ci     = class_info[cn]
        prob   = float(final_probs[idx])
        sc     = SEV_COLORS.get(ci["severity"], TEXT_MID)
        is_top = rank == 1
        prob_rows.append([
            Paragraph(f"<b>#{rank}</b>" if is_top else f"#{rank}", BODY),
            Paragraph(f'<b><font color="{ci["color"]}">{ci["display"]}</font></b>'
                      if is_top else ci["display"], BODY),
            Paragraph(ci["severity"], BODY),
            Paragraph(f"<b>{prob*100:.1f}%</b>" if is_top else f"{prob*100:.1f}%", BODY),
            ColorBar(prob, color=sc if is_top else TEXT_MID,
                     width=usable_w * 0.22, height=8),
        ])

    prob_col_w = [usable_w * f for f in [0.08, 0.30, 0.14, 0.12, 0.36]]
    prob_t = Table(prob_rows, colWidths=prob_col_w)
    prob_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), GREEN_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GREEN_BG, ROW_ALT_BG]),   # FIX: light alt rows
        ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(prob_t)
    story.append(Spacer(1, 5 * mm))

    # ── SECTION 5: ATTENTION GATE WEIGHTS ────────────────────────────────────
    story.append(Paragraph("5. Attention Gate Fusion Weights", H2))
    story.append(HRFlowable(width=usable_w, thickness=1.5, color=GREEN_MID, spaceAfter=4))
    story.append(Paragraph(
        "The learned multi-scale attention gate dynamically assigns a per-sample trust weight "
        "to each backbone model conditioned on prediction confidence and detected lesion spatial "
        "scale. These weights are NOT fixed — they are computed per input image, which is the "
        "primary architectural novelty of MAIZE-XNet over prior soft-voting ensemble approaches.",
        BODY))
    story.append(Spacer(1, 4 * mm))

    for i, (mname, mhex, mcolor) in enumerate(
            zip(model_names, MODEL_COLORS_HEX, MODEL_COLORS)):
        w = float(gate_weights[i])
        gr = Table([[
            Paragraph(f'<font color="{mhex}"><b>{mname}</b></font>', BODY),
            ColorBar(w, color=mcolor, width=usable_w * 0.60, height=10),
            Paragraph(f"<b>{w*100:.1f}%</b>", BODY),
        ]], colWidths=[usable_w * 0.22, usable_w * 0.62, usable_w * 0.16])
        gr.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (0, 0),  0),
        ]))
        story.append(gr)

    story.append(Spacer(1, 5 * mm))

    # ── SECTION 6: TSDS XAI ANALYSIS ─────────────────────────────────────────
    story.append(Paragraph("6. TSDS — Temporal Saliency Drift Score (XAI Trust Certificate)", H2))
    story.append(HRFlowable(width=usable_w, thickness=1.5, color=GREEN_MID, spaceAfter=4))

    # FIX: tsds_box_bg is now a light tint (chosen above based on score), not black/dark green.
    tsds_box = Table([[Paragraph(
        f"TSDS = <b>{tsds_score:.4f}</b>  &nbsp;·&nbsp;  {tsds_text}<br/>"
        f"<font size='8'>{tsds_desc}<br/><br/>"
        f"The TSDS is computed as the mean pairwise Intersection-over-Union (IoU) of "
        f"Grad-CAM attention masks across all 4 backbone models, binarized at the 80th "
        f"activation percentile. A high TSDS indicates that all models consistently attend "
        f"to the SAME leaf region regardless of input augmentation — providing a biologically "
        f"grounded per-prediction trust certificate absent from all 15 prior corn disease papers. "
        f"Score range: 0.0 (complete spatial disagreement) → 1.0 (perfect consensus).</font>",
        BODY)]])
    tsds_box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), tsds_box_bg),
        ("BOX",           (0, 0), (-1, -1), 1.5, tsds_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
    ]))
    story.append(tsds_box)
    story.append(Spacer(1, 3 * mm))
    story.append(ColorBar(tsds_score, color=tsds_color, width=usable_w, height=14))
    story.append(Spacer(1, 5 * mm))

    # ── SECTION 7: GRAD-CAM HEATMAPS (if available) ───────────────────────────
    if gradcam_maps is not None:
        story.append(Paragraph("7. Grad-CAM Saliency Maps", H2))
        story.append(HRFlowable(width=usable_w, thickness=1.5, color=GREEN_MID, spaceAfter=4))
        story.append(Paragraph(
            "Each heatmap shows which regions of the corn leaf each model attended to when "
            "making its prediction. Warm tones (red/orange) indicate high attention regions. "
            "Consistent focus across models contributes to a higher TSDS.", BODY))
        story.append(Spacer(1, 4 * mm))

        try:
            import matplotlib.cm as mplcm
            cam_imgs = []
            for cam in gradcam_maps:
                colored = mplcm.jet(cam)[:, :, :3]
                ci_pil  = Image.fromarray((colored * 255).astype(np.uint8))
                orig_r  = image.convert("RGB").resize((224, 224), Image.BILINEAR)
                blended = Image.blend(orig_r, ci_pil, alpha=0.5)
                cam_imgs.append(blended)

            cam_w = (usable_w - 3 * 4 * mm) / 4
            cam_row_data = [[pil_to_rl(ci, max_w=cam_w, max_h=cam_w) for ci in cam_imgs]]
            cam_t = Table(cam_row_data, colWidths=[cam_w + 4 * mm] * 4)
            cam_t.setStyle(TableStyle([
                ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
                ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(cam_t)

            label_row = [[Paragraph(f'<font color="{mc}">{mn}</font>',
                                     S("cl", fontName="Helvetica-Bold", fontSize=8,
                                       textColor=HexColor(mc), leading=10,
                                       alignment=TA_CENTER))
                          for mn, mc in zip(model_names, MODEL_COLORS_HEX)]]
            lbl_t = Table(label_row, colWidths=[cam_w + 4 * mm] * 4)
            lbl_t.setStyle(TableStyle([
                ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING",    (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(lbl_t)
        except Exception:
            story.append(Paragraph("(Grad-CAM heatmap rendering unavailable in PDF)", BDIM))

        story.append(Spacer(1, 5 * mm))
        sec8 = "8"
    else:
        sec8 = "7"

    # ── SYSTEM INFO ───────────────────────────────────────────────────────────
    story.append(Paragraph(f"{sec8}. System Information", H2))
    story.append(HRFlowable(width=usable_w, thickness=1.5, color=GREEN_MID, spaceAfter=4))

    sys_data = [
        ["Parameter",          "Value"],
        ["System",             "MAIZE-XNet — Multi-Scale Attention-Gated Cross-Architecture Ensemble"],
        ["Architectures",      "EfficientNet-B4 (380px) + ConvNeXt-Tiny (224px) + MaxViT-Small (224px) + MobileViT-Small (256px)"],
        ["Ensemble Fusion",    "Learned Multi-Scale Attention Gate (per-sample adaptive weights)"],
        ["XAI Method",         "Grad-CAM + TSDS (Temporal Saliency Drift Score) — novel trust certificate"],
        ["Disease Classes",    "Northern Leaf Blight | Common Rust | Gray Leaf Spot | Healthy"],
        ["Accuracy (Ensemble)","98.00% test accuracy (4-class corn disease dataset)"],
        ["Novel Contribution", "TSDS: first augmentation-driven saliency stability metric in corn disease literature"],
        ["Deployment",         "Streamlit PWA + ONNX Runtime + Hugging Face Spaces"],
        ["Report Generated",   now.strftime("%Y-%m-%d %H:%M:%S")],
        ["Thesis",             "MSc in CSE — Your University, Department of Computer Science"],
    ]
    sys_col_w = [usable_w * 0.28, usable_w * 0.72]
    sys_t = Table(
        [[Paragraph(r[0], LBL_ON_DARK if i == 0 else BDIM),
          Paragraph(r[1], LBL_ON_DARK if i == 0 else BODY)]
         for i, r in enumerate(sys_data)],
        colWidths=sys_col_w
    )
    sys_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), GREEN_DARK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GREEN_BG, ROW_ALT_BG]),   # FIX: light alt rows
        ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 9),
    ]))
    story.append(sys_t)
    story.append(Spacer(1, 6 * mm))

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    # FIX: light amber tint background instead of near-black, dark text instead of dim gray.
    disc = Table([[Paragraph(
        "⚠️  RESEARCH DISCLAIMER: This report is generated by an AI research system "
        "developed as part of an MSc thesis (MAIZE-XNet). It is intended for academic "
        "and research purposes only. Always consult a qualified agronomist or certified "
        "plant pathologist for definitive field diagnosis and treatment decisions. "
        "The authors accept no liability for decisions made solely based on this report.",
        S("disc", fontName="Helvetica", fontSize=8, textColor=TEXT_DARK, leading=11))]])
    disc.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#fff8e8")),
        ("BOX",           (0, 0), (-1, -1), 0.5, AMBER),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    story.append(disc)

    # ── Page numbers ──────────────────────────────────────────────────────────
    def add_page_number(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.setFillColor(TEXT_MID)
        pg = canvas_obj.getPageNumber()
        canvas_obj.drawCentredString(
            W / 2, 7 * mm,
            f"MAIZE-XNet Diagnostic Report  ·  Page {pg}  ·  "
            f"Generated {now.strftime('%Y-%m-%d %H:%M')}  ·  MSc Thesis"
        )
        if pg > 1:
            canvas_obj.setFillColor(GREEN_MID)
            canvas_obj.rect(margin, H - 10 * mm, W - 2 * margin, 1.5, fill=1, stroke=0)
        canvas_obj.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    buf.seek(0)
    return buf.read()
