"""
generate_L3_pdf.py
------------------
Generates a magazine-style lecture notes PDF for Lecture 3:
Regularization Strategies & Cross-Validation
(Statistisches Lernen 2 — FH Kufstein Tirol)

Output language: PT-BR  |  Technical terms: English
Dependencies: reportlab, matplotlib, numpy, scipy, sklearn
"""

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm, laplace
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut,
    cross_val_score, train_test_split,
)
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import mean_squared_error

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

# ---------------------------------------------------------------------------
# PALETTE  (L3 uses forest-green / gold tones, distinct from L1 navy and L2 purple)
# ---------------------------------------------------------------------------
FOREST      = colors.HexColor("#1B4332")
EMERALD     = colors.HexColor("#2D6A4F")
LIME        = colors.HexColor("#52B788")
GOLD        = colors.HexColor("#D4940A")
RUST        = colors.HexColor("#C0392B")
TEAL_LIGHT  = colors.HexColor("#74C69D")
LIGHT_BG    = colors.HexColor("#FFFFFF")   # pure white — body area background
SIDEBAR_BG  = colors.HexColor("#EBF5EC")   # very subtle green for sidebar only
ROW_ALT     = colors.HexColor("#F2FAF4")   # barely-green for table alternating rows
WHITE       = colors.white
BLACK       = colors.black
NEAR_BLACK  = colors.HexColor("#111111")

PAGE_W, PAGE_H = A4
MARGIN        = 1.8 * cm
SIDEBAR_W     = 1.2 * cm
CONTENT_W     = PAGE_W - 2 * MARGIN - SIDEBAR_W
HEADER_H      = 1.6 * cm
FOOTER_H      = 1.2 * cm

np.random.seed(42)


# ---------------------------------------------------------------------------
# CUSTOM CANVAS
# ---------------------------------------------------------------------------
class LectureCanvas:
    def __init__(self, total_pages_ref):
        self._total = total_pages_ref

    def draw_cover(self, canvas, doc):
        canvas.saveState()
        w, h = PAGE_W, PAGE_H
        steps = 60
        for i in range(steps):
            t = i / steps
            r = int(27  + t * (16 - 27))
            g = int(67  + t * (40 - 67))
            b = int(50  + t * (30 - 50))
            canvas.setFillColorRGB(r/255, g/255, b/255)
            canvas.rect(0, h * i / steps, w, h / steps + 1, fill=1, stroke=0)

        # Decorative circles
        for cx, cy, cr, alpha in [
            (w * 0.80, h * 0.75, 3.6*cm, 0.12),
            (w * 0.10, h * 0.20, 5.0*cm, 0.09),
            (w * 0.60, h * 0.35, 2.4*cm, 0.16),
        ]:
            canvas.setFillColor(WHITE)
            canvas.setFillAlpha(alpha)
            canvas.circle(cx, cy, cr, fill=1, stroke=0)
        canvas.setFillAlpha(1.0)

        # Gold accent bar
        canvas.setFillColor(GOLD)
        canvas.rect(0, h * 0.55, SIDEBAR_W * 1.5, h * 0.45, fill=1, stroke=0)

        # Course tag
        canvas.setFillColor(EMERALD)
        canvas.roundRect(MARGIN, h * 0.78, 7*cm, 0.9*cm, 4, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(MARGIN + 0.3*cm, h * 0.78 + 0.25*cm,
                          "STATISTISCHES LERNEN 2")

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 28)
        canvas.drawString(MARGIN, h * 0.62, "Aula 3")
        canvas.setFont("Helvetica", 18)
        canvas.drawString(MARGIN, h * 0.56, "Regularization Strategies")
        canvas.drawString(MARGIN, h * 0.51, "& Cross-Validation")

        canvas.setFillColor(GOLD)
        canvas.rect(MARGIN, h * 0.49, 8*cm, 0.06*cm, fill=1, stroke=0)

        canvas.setFillColor(colors.HexColor("#A8D5A2"))
        canvas.setFont("Helvetica", 10)
        canvas.drawString(MARGIN, h * 0.45, "Prof. Johannes Schwab, PhD")
        canvas.drawString(MARGIN, h * 0.42, "FH Kufstein Tirol")

        canvas.setFillColor(GOLD)
        canvas.rect(0, 0, w, 0.4*cm, fill=1, stroke=0)
        canvas.setFillColor(EMERALD)
        canvas.rect(0, 0.4*cm, w, 0.15*cm, fill=1, stroke=0)
        canvas.restoreState()

    def draw_page(self, canvas, doc):
        canvas.saveState()
        w, h = PAGE_W, PAGE_H
        pn = doc.page

        canvas.setFillColor(SIDEBAR_BG)
        canvas.rect(0, FOOTER_H, SIDEBAR_W, h - HEADER_H - FOOTER_H,
                    fill=1, stroke=0)
        canvas.setFillColor(EMERALD)
        canvas.rect(SIDEBAR_W - 0.15*cm, FOOTER_H,
                    0.15*cm, h - HEADER_H - FOOTER_H, fill=1, stroke=0)

        canvas.setFillColor(FOREST)
        canvas.rect(0, h - HEADER_H, w, HEADER_H, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(0, h - HEADER_H - 0.12*cm, w, 0.12*cm, fill=1, stroke=0)

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(SIDEBAR_W + 0.4*cm, h - HEADER_H + 0.55*cm,
                          "Statistisches Lernen 2  |  Aula 3 — Regularization & Cross-Validation")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 0.4*cm, h - HEADER_H + 0.55*cm, "FH Kufstein Tirol")

        canvas.setFillColor(colors.HexColor("#F8F8F8"))
        canvas.rect(0, 0, w, FOOTER_H, fill=1, stroke=0)
        canvas.setFillColor(FOREST)
        canvas.rect(0, FOOTER_H - 0.1*cm, w, 0.1*cm, fill=1, stroke=0)

        canvas.setFillColor(colors.HexColor("#333333"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(SIDEBAR_W + 0.4*cm, 0.38*cm,
                          "Prof. Johannes Schwab, PhD  —  FH Kufstein Tirol")
        total = self._total[0] if self._total[0] else "?"
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(FOREST)
        canvas.drawRightString(w - 0.4*cm, 0.38*cm, f"Pagina {pn} de {total}")

        canvas.saveState()
        canvas.translate(SIDEBAR_W * 0.5, h * 0.5)
        canvas.rotate(90)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(FOREST)
        canvas.drawCentredString(0, 0, "REGULARIZATION  |  CROSS-VALIDATION")
        canvas.restoreState()
        canvas.restoreState()


# ---------------------------------------------------------------------------
# PARAGRAPH STYLES
# ---------------------------------------------------------------------------
def build_styles():
    def ps(name, **kwargs):
        return ParagraphStyle(name, **kwargs)

    return {
        "title": ps("title", fontName="Helvetica-Bold", fontSize=22,
                    textColor=FOREST, spaceAfter=6, spaceBefore=4),
        "section": ps("section", fontName="Helvetica-Bold", fontSize=14,
                      textColor=WHITE, spaceBefore=14, spaceAfter=4),
        "subsection": ps("subsection", fontName="Helvetica-Bold", fontSize=11,
                         textColor=FOREST, spaceBefore=8, spaceAfter=3),
        "body": ps("body", fontName="Helvetica", fontSize=10,
                   textColor=NEAR_BLACK, leading=16,
                   spaceBefore=4, spaceAfter=4, alignment=TA_LEFT),
        "bullet": ps("bullet", fontName="Helvetica", fontSize=9.5,
                     textColor=NEAR_BLACK, leading=15,
                     spaceBefore=2, spaceAfter=2, leftIndent=16, bulletIndent=4),
        "caption": ps("caption", fontName="Helvetica-Oblique", fontSize=8,
                      textColor=colors.HexColor("#444444"),
                      spaceBefore=2, spaceAfter=6, alignment=TA_CENTER),
        "highlight": ps("highlight", fontName="Helvetica-Bold", fontSize=10,
                        textColor=FOREST, spaceBefore=4, spaceAfter=4,
                        leftIndent=10),
        "code": ps("code", fontName="Courier", fontSize=8.5,
                   textColor=colors.HexColor("#1A1A1A"),
                   backColor=colors.HexColor("#F4F4F4"),
                   spaceBefore=4, spaceAfter=4,
                   leftIndent=8, rightIndent=8, leading=13),
        "quote": ps("quote", fontName="Helvetica-Oblique", fontSize=10,
                    textColor=colors.HexColor("#333333"), leading=16,
                    leftIndent=20, rightIndent=20, spaceBefore=6, spaceAfter=6,
                    alignment=TA_LEFT),
    }


# ---------------------------------------------------------------------------
# HELPER BUILDERS
# ---------------------------------------------------------------------------
def section_header(title, styles):
    data = [[Paragraph(title, styles["section"])]]
    tbl = Table(data, colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), FOREST),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LINEBELOW",     (0,0), (-1,-1), 3, GOLD),
    ]))
    return tbl


def why_box(why_text, apply_text, styles, color=EMERALD):
    header_s = ParagraphStyle("wh", fontName="Helvetica-Bold", fontSize=9.5,
                               textColor=WHITE, leading=13)
    body_s   = ParagraphStyle("wb", fontName="Helvetica", fontSize=9.5,
                               textColor=NEAR_BLACK,
                               leading=14, alignment=TA_LEFT)
    data = [
        [Paragraph("Por que isso e importante para Data Science?", header_s)],
        [Paragraph(why_text, body_s)],
        [Paragraph("Aplicacao Pratica em Machine Learning", header_s)],
        [Paragraph(apply_text, body_s)],
    ]
    tbl = Table(data, colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), color),
        ("BACKGROUND",    (0,1), (-1,1), WHITE),
        ("BACKGROUND",    (0,2), (-1,2), FOREST),
        ("BACKGROUND",    (0,3), (-1,3), WHITE),
        ("LINEABOVE",     (0,0), (-1,0),  1.5, GOLD),
        ("LINEBELOW",     (0,3), (-1,-1), 1.5, GOLD),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    return tbl


def formula_block(latex_str, label="", bg_hex="#1B4332", formula_size=15):
    w_in = CONTENT_W / 72
    has_label = bool(label.strip())
    h_in = 1.05 if not has_label else 1.55

    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.patch.set_facecolor(bg_hex)
    ax.set_facecolor(bg_hex)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([0,1], [0.985, 0.985], color="#E9C46A", linewidth=3.5,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0,1], [0.015, 0.015], color="#E9C46A", linewidth=3.5,
            transform=ax.transAxes, clip_on=False)

    y_f = 0.63 if has_label else 0.50
    ax.text(0.5, y_f, f"${latex_str}$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=formula_size, color="white",
            math_fontfamily="dejavusans")

    if has_label:
        ax.text(0.5, 0.20, label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#A8D5A2", fontstyle="italic")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=bg_hex, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    img_h = (h_in / w_in) * CONTENT_W
    return Image(buf, width=CONTENT_W, height=img_h)


def comparison_table(headers, rows, col_widths):
    h_style = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=9,
                              textColor=WHITE, alignment=TA_CENTER)
    c_style = ParagraphStyle("td", fontName="Helvetica", fontSize=9,
                              textColor=NEAR_BLACK, alignment=TA_LEFT)
    data = [[Paragraph(h, h_style) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), c_style) for c in row])
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0), FOREST),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, ROW_ALT]),
        ("GRID",           (0,0), (-1,-1), 0.5, colors.HexColor("#AACFB5")),
        ("LINEBELOW",      (0,0), (-1,0),  1.5, GOLD),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
        ("RIGHTPADDING",   (0,0), (-1,-1), 8),
        ("TOPPADDING",     (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 6),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
    ]))
    return tbl


def mat_image(fig, width_cm=14):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_cm*cm, height=width_cm*cm*0.5)


def mat_image_tall(fig, width_cm=14, aspect=0.65):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_cm*cm, height=width_cm*cm*aspect)


def divider(color=EMERALD, thickness=0.8):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=4, spaceBefore=4)


def spacer(h_cm=0.3):
    return Spacer(1, h_cm*cm)


# ---------------------------------------------------------------------------
# CHART GENERATORS
# ---------------------------------------------------------------------------

def chart_three_reg_types():
    """3-panel overview: Hard, Soft, Output regularization."""
    T1, T2 = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
    Loss = (T1 - 2)**2 + 0.5*(T2 - 1.5)**2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")

    # --- Hard regularization (L2 ball) ---
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    ax.contour(T1, T2, Loss, levels=15, cmap="RdYlGn_r", alpha=0.7)
    theta_c = np.linspace(0, 2*np.pi, 300)
    R = 1.5
    ax.fill(R*np.cos(theta_c), R*np.sin(theta_c), alpha=0.15, color="#1B4332")
    ax.plot(R*np.cos(theta_c), R*np.sin(theta_c), color="#1B4332", lw=2.5)
    ax.scatter([2], [1.5], s=120, color="red", zorder=6, label="opt. sem reg.")
    th_r = np.array([2.0, 1.5]) * R / np.sqrt(2.0**2 + 1.5**2)
    ax.scatter(th_r[0], th_r[1], s=130, color="#1B4332", marker="*",
               zorder=6, label="opt. com restr.")
    ax.set_title("Hard Regularization\nth in C (conjunto viavel)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("theta_1"); ax.set_ylabel("theta_2")
    ax.legend(fontsize=8); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_aspect("equal")

    # --- Soft regularization (penalty) ---
    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    Penalty = T1**2 + T2**2
    Total = Loss + 0.8 * Penalty
    ax.contour(T1, T2, Loss,  levels=12, cmap="RdYlGn_r", alpha=0.5, linestyles="--")
    ax.contour(T1, T2, Total, levels=12, cmap="Greens",   alpha=0.8)
    ax.scatter([2], [1.5], s=120, color="red",     zorder=6, label="sem lambda")
    ax.scatter([0.8], [0.65], s=130, color="#40916C", marker="*",
               zorder=6, label="com lambda")
    ax.set_title("Soft / Variational Regularization\nL(th) + lambda * R(th)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("theta_1"); ax.set_ylabel("theta_2")
    ax.legend(fontsize=8); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)

    # --- Output regularization (label smoothing example) ---
    ax = axes[2]
    ax.set_facecolor("#FAFAFA")
    prob = np.linspace(0.001, 0.999, 400)
    ce_hard   = -np.log(prob)
    ce_smooth = -0.9*np.log(prob) - 0.1*np.log(1 - prob)
    ax.plot(prob, ce_hard,   color="#E76F51", lw=2.5, label="Hard label (target=1.0)")
    ax.plot(prob, ce_smooth, color="#40916C", lw=2.5, label="Label Smoothing (eps=0.1)")
    ax.set_xlabel("Probabilidade prevista p")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Output Regularization\nExemplo: Label Smoothing", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 5); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("Os Tres Tipos de Regularization — Visao Geral",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_ridge_path():
    """Ridge regularization path: coefficients vs lambda (log scale)."""
    X_r, y_r = make_regression(n_samples=100, n_features=20, noise=10.0,
                                n_informative=5, random_state=42)
    lambdas = np.logspace(-3, 4, 60)
    coefs   = np.array([Ridge(alpha=lam).fit(X_r, y_r).coef_ for lam in lambdas])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    for j in range(20):
        ax.plot(lambdas, coefs[:, j], lw=1.4, alpha=0.75)
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("lambda (forca da regularizacao)")
    ax.set_ylabel("valor do coeficiente")
    ax.set_title("Ridge Regularization Path\nTodos os coefs. encolhem suavemente — nunca zerados",
                 fontsize=9.5, fontweight="bold")
    ax.grid(alpha=0.3)

    # Right: MSE train vs test vs lambda for p > N
    np.random.seed(7)
    N, p = 50, 100
    X_hd = np.random.randn(N, p)
    theta_real = np.zeros(p)
    theta_real[:5] = [3.0, -2.0, 1.5, -1.0, 0.8]
    y_hd = X_hd @ theta_real + np.random.normal(0, 0.5, N)
    X_tr, X_te, y_tr, y_te = train_test_split(X_hd, y_hd, test_size=0.3, random_state=42)

    lams_test = np.logspace(-4, 3, 40)
    mse_tr_l2, mse_te_l2 = [], []
    for lam in lams_test:
        r = Ridge(alpha=lam).fit(X_tr, y_tr)
        mse_tr_l2.append(mean_squared_error(y_tr, r.predict(X_tr)))
        mse_te_l2.append(mean_squared_error(y_te, r.predict(X_te)))

    best_idx = np.argmin(mse_te_l2)
    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    ax.semilogx(lams_test, mse_tr_l2, color="#40916C", lw=2.5, marker="o", ms=4, label="MSE Treino")
    ax.semilogx(lams_test, mse_te_l2, color="#E76F51", lw=2.5, marker="s", ms=4, label="MSE Teste")
    ax.axvline(lams_test[best_idx], color="#1B4332", ls="--", lw=2,
               label=f"lambda* = {lams_test[best_idx]:.3f}")
    ax.set_xlabel("lambda (Log-Scale)")
    ax.set_ylabel("MSE")
    ax.set_title(f"Ridge com p={p} > N={N}\nUnder vs. Over-Regularization",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("L2 Regularization (Ridge / Weight Decay)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_geometry_l0_l1_l2():
    """Geometric comparison: L0 (axes), L1 (diamond), L2 (circle)."""
    T1, T2 = np.meshgrid(np.linspace(-2.5, 2.5, 300), np.linspace(-2.5, 2.5, 300))
    Loss = (T1 - 2.0)**2 + 0.5*(T2 - 1.5)**2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    R = 1.5

    for ax in axes:
        ax.set_facecolor("#FAFAFA")
        ax.contour(T1, T2, Loss, levels=15, cmap="RdYlGn_r", alpha=0.7)
        ax.scatter([2.0], [1.5], s=120, color="red", zorder=7, label="theta* otimo")
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("theta_1"); ax.set_ylabel("theta_2")
        ax.set_aspect("equal")

    # L0: eixos coordenados
    ax = axes[0]
    ax.axhline(0, color="#888888", lw=5, alpha=0.6, label="B_R^0: eixos (s=1)")
    ax.axvline(0, color="#888888", lw=5, alpha=0.6)
    ax.scatter([R], [0], s=160, color="#888888", marker="*", zorder=7,
               label="theta_L0 (no eixo)")
    ax.set_title("L0 — Sparsidade Ideal (NP-Hard)\nNao convexa, nao diferenciavel",
                 fontsize=9.5, fontweight="bold")
    ax.text(-2.2, 1.6, "NP-Hard:\nexponencial\nem p", fontsize=9,
            color="#C0392B", fontweight="bold")
    ax.legend(fontsize=8)

    # L1: diamante
    ax = axes[1]
    dx = [R, 0, -R, 0, R]
    dy = [0, R,  0, -R, 0]
    ax.fill(dx, dy, alpha=0.18, color="#E9C46A")
    ax.plot(dx, dy, color="#E9C46A", lw=2.5)
    ax.scatter([0], [R], s=170, color="#E9C46A", marker="*", zorder=7,
               label="theta_L1: toca vertice!\n(theta_1=0 — esparso)")
    ax.set_title("L1 — Lasso (Diamante)\nVertices nos eixos => esparsidade!",
                 fontsize=9.5, fontweight="bold")
    ax.annotate("Vertice => theta_1=0\n(feature removida!)",
                xy=(0.05, R-0.05), xytext=(0.6, 2.0),
                arrowprops=dict(arrowstyle="->", color="#E9C46A"),
                color="#B8860B", fontsize=9)
    ax.legend(fontsize=8)

    # L2: circulo
    ax = axes[2]
    theta_c = np.linspace(0, 2*np.pi, 300)
    ax.fill(R*np.cos(theta_c), R*np.sin(theta_c), alpha=0.12, color="#40916C")
    ax.plot(R*np.cos(theta_c), R*np.sin(theta_c), color="#40916C", lw=2.5)
    th_l2 = np.array([2.0, 1.5]) * R / np.sqrt(2.0**2 + 1.5**2)
    ax.scatter(th_l2[0], th_l2[1], s=170, color="#40916C", marker="*", zorder=7,
               label="theta_L2: borda circular\n(sem esparsidade exata)")
    ax.set_title("L2 — Ridge (Circulo)\nEncolhimento suave — nunca zera exatamente",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=8)

    plt.suptitle("Geometria das Restricoes: L0 vs. L1 vs. L2",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_reg_paths_comparison():
    """Side-by-side regularization paths: Ridge, Lasso, Elastic Net."""
    X_sp, y_sp = make_regression(n_samples=150, n_features=20, noise=8.0,
                                  n_informative=4, random_state=42)
    lambdas_path = np.logspace(-4, 2, 80)

    coefs_ridge = np.array([Ridge(alpha=lam).fit(X_sp, y_sp).coef_
                             for lam in lambdas_path])
    coefs_lasso = np.array([Lasso(alpha=lam, max_iter=10000).fit(X_sp, y_sp).coef_
                             for lam in lambdas_path])
    coefs_enet  = np.array([ElasticNet(alpha=lam, l1_ratio=0.5, max_iter=10000).fit(X_sp, y_sp).coef_
                             for lam in lambdas_path])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")
    configs = [
        (axes[0], coefs_ridge, "Ridge (L2)\nEncolhimento suave — nunca zera exatamente", "#40916C"),
        (axes[1], coefs_lasso, "Lasso (L1)\nEsparsidade: coefs vao a zero!", "#E9C46A"),
        (axes[2], coefs_enet,  "Elastic Net (L1 + L2)\nEsparsidade + grupos de features", "#E76F51"),
    ]
    for ax, coefs, titulo, cor in configs:
        ax.set_facecolor("#FAFAFA")
        for j in range(20):
            alpha_j = 0.9 if np.max(np.abs(coefs[:, j])) > 5 else 0.4
            ax.plot(lambdas_path, coefs[:, j], lw=1.5, alpha=alpha_j)
        ax.axhline(0, color="black", lw=1.5, ls="--")
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("lambda (Log-Scale)"); ax.set_ylabel("theta_j")
        ax.set_title(titulo, fontsize=9.5, fontweight="bold")
        ax.grid(alpha=0.3)

    n_zeros = np.sum(np.abs(coefs_lasso[-1]) < 1e-6)
    plt.suptitle(f"Regularization Paths: Ridge vs. Lasso vs. Elastic Net\n"
                 f"(Lasso: {n_zeros}/20 coefs zerados com lambda maximo)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_early_stopping():
    """Early stopping simulation with polynomial regression + gradient descent."""
    f_true = lambda x: np.sin(2 * np.pi * x)
    np.random.seed(13)
    x_tr  = np.sort(np.random.uniform(0, 1, 30))
    y_tr  = f_true(x_tr) + np.random.normal(0, 0.2, 30)
    x_val = np.sort(np.random.uniform(0, 1, 20))
    y_val = f_true(x_val) + np.random.normal(0, 0.2, 20)

    grau = 15
    def poly_feat(x, g):
        return np.column_stack([x**k for k in range(g+1)])

    B_tr  = poly_feat(x_tr, grau)
    B_val = poly_feat(x_val, grau)

    theta = np.zeros(grau + 1)
    eta = 1e-3
    n_epocas, patience = 5000, 200
    hist_tr, hist_val = [], []
    melhor_val = np.inf; melhor_theta = theta.copy(); melhor_ep = 0
    sem_melhora = 0; ep_stop = n_epocas

    for ep in range(n_epocas):
        grad = -2/len(x_tr) * B_tr.T @ (y_tr - B_tr @ theta)
        theta = theta - eta * grad
        mse_t = mean_squared_error(y_tr,  B_tr  @ theta)
        mse_v = mean_squared_error(y_val, B_val @ theta)
        hist_tr.append(mse_t); hist_val.append(mse_v)
        if mse_v < melhor_val:
            melhor_val = mse_v; melhor_theta = theta.copy()
            melhor_ep = ep; sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= patience and ep_stop == n_epocas:
                ep_stop = ep

    x_plot = np.linspace(0, 1, 300)
    B_plot = poly_feat(x_plot, grau)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    ax.semilogy(hist_tr,  color="#40916C", lw=2, label="Loss Treino")
    ax.semilogy(hist_val, color="#E76F51", lw=2, label="Loss Validacao")
    ax.axvline(melhor_ep, color="#1B4332", ls="--", lw=2,
               label=f"Early Stop (epoca {melhor_ep})")
    ax.axvline(ep_stop, color="#E9C46A", ls=":", lw=2,
               label=f"Patience esgotada ({ep_stop})")
    ax.set_xlabel("Epoca"); ax.set_ylabel("MSE (log)")
    ax.set_title("Early Stopping:\nMonitorando Loss de Validacao",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    ax.scatter(x_tr, y_tr, s=40, color="#888", alpha=0.7, zorder=5, label="Treino")
    ax.plot(x_plot, f_true(x_plot), "k--", alpha=0.4, lw=1.5, label="f verdadeira")
    ax.plot(x_plot, B_plot @ theta, color="#E76F51", lw=2,
            label=f"Sem Early Stop (epoca {n_epocas})")
    ax.plot(x_plot, B_plot @ melhor_theta, color="#40916C", lw=2.5,
            label=f"Com Early Stop (epoca {melhor_ep})")
    ax.set_ylim(-2.5, 2.5)
    ax.set_title("Curva ajustada: com vs. sem Early Stopping",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("Early Stopping — Regularizacao via Interrupcao da Otimizacao",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_dropout():
    """Dropout neuron masking visualization."""
    np.random.seed(42)
    h = np.random.randn(20)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")

    for ax, p_drop, titulo in [
        (axes[0], 0.0, "Sem Dropout (p=0)\nTreino normal"),
        (axes[1], 0.5, "Dropout p=0.5\n~50% dos neuronios zerados"),
        (axes[2], 0.8, "Dropout p=0.8\n~80% dos neuronios zerados"),
    ]:
        ax.set_facecolor("#FAFAFA")
        mask = np.random.binomial(1, 1 - p_drop, size=len(h))
        h_dropped = h * mask
        cores_bar = ["#40916C" if m == 1 else "#E76F51" for m in mask]
        ax.bar(range(len(h)), h_dropped, color=cores_bar, alpha=0.85)
        ax.axhline(0, color="black", lw=1)
        n_ativos = int(np.sum(mask))
        ax.set_title(f"{titulo}\n({n_ativos}/{len(h)} neuronios ativos)",
                     fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Indice do neuronio"); ax.set_ylabel("Ativacao")
        ax.grid(alpha=0.3)

    azul_p = mpatches.Patch(color="#40916C", label="Neuronio ativo")
    verm_p = mpatches.Patch(color="#E76F51", label="Neuronio zerado (dropped)")
    axes[2].legend(handles=[azul_p, verm_p], fontsize=9, loc="upper right")

    plt.suptitle("Dropout: Regularizacao por Sabotagem Aleatoria durante o Treino",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_bayesian_priors():
    """Gaussian vs Laplace priors and their penalty functions."""
    theta_vis = np.linspace(-4, 4, 400)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    for tau, lw_v, al in [(0.5, 3, 1.0), (1.0, 2, 0.7), (2.0, 1.5, 0.5)]:
        ax.plot(theta_vis, norm.pdf(theta_vis, 0, tau), lw=lw_v, alpha=al,
                label=f"N(0, {tau}^2)")
    ax.set_title("Gaussian Prior\np(theta) = N(0, tau^2 I)\n=> L2 Regularization",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("theta"); ax.set_ylabel("p(theta)")
    ax.legend(fontsize=9); ax.axvline(0, color="gray", ls=":", lw=1); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    for b, lw_v, al in [(0.5, 3, 1.0), (1.0, 2, 0.7), (2.0, 1.5, 0.5)]:
        ax.plot(theta_vis, laplace.pdf(theta_vis, 0, b), lw=lw_v, alpha=al,
                label=f"Laplace(0, {b})")
    ax.set_title("Laplace Prior\np(theta) ~ exp(-|theta|/b)\n=> L1 Regularization (Lasso)",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("theta"); ax.set_ylabel("p(theta)")
    ax.legend(fontsize=9); ax.axvline(0, color="gray", ls=":", lw=1); ax.grid(alpha=0.3)
    ax.annotate("Pico pontiagudo em zero:\nacredita que theta = 0",
                xy=(0, laplace.pdf(0, 0, 0.5)), xytext=(1.2, 0.6),
                arrowprops=dict(arrowstyle="->", color="#E9C46A"),
                color="#B8860B", fontsize=9)

    ax = axes[2]
    ax.set_facecolor("#FAFAFA")
    ax.plot(theta_vis, theta_vis**2, color="#40916C", lw=2.5,
            label="||theta||^2 (L2 / Gaussian)")
    ax.plot(theta_vis, np.abs(theta_vis), color="#E9C46A", lw=2.5,
            label="||theta||_1 (L1 / Laplace)")
    ax.plot(theta_vis, (np.abs(theta_vis) > 0).astype(float), color="#888",
            lw=2, ls="--", label="||theta||_0 (L0 — NP-hard)")
    ax.set_xlim(-3, 3); ax.set_ylim(-0.1, 4)
    ax.set_title("Penalidades = -log p(theta)\nGeometria das priors",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("theta"); ax.set_ylabel("-log p(theta)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("The Bayesian Connection: Prior => Regularization\n"
                 "Gaussian Prior = L2  |  Laplace Prior = L1",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_cv_patterns():
    """Visual patterns for Hold-Out, K-Fold, Stratified, LOO."""
    N_vis = 30
    K = 5
    COR_TREINO = "#40916C"
    COR_VAL    = "#E76F51"

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), facecolor="white")

    # Hold-Out
    ax = axes[0]
    split = int(0.8 * N_vis)
    cores_ho = [COR_TREINO]*split + [COR_VAL]*(N_vis - split)
    ax.barh([0]*N_vis, [1]*N_vis, left=range(N_vis),
            color=cores_ho, edgecolor="white", height=0.6)
    ax.set_yticks([0]); ax.set_yticklabels(["Hold-Out"])
    ax.set_xlim(0, N_vis); ax.set_title("Hold-Out (80/20 split) — rapido, instavel")
    ax.set_xlabel("Indice das amostras")
    tr_p = mpatches.Patch(color=COR_TREINO, label="Treino")
    va_p = mpatches.Patch(color=COR_VAL,   label="Validacao")
    ax.legend(handles=[tr_p, va_p], loc="lower right", fontsize=9)

    # K-Fold
    ax = axes[1]
    kf = KFold(n_splits=K, shuffle=False)
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(N_vis))):
        cores_fold = np.array([COR_TREINO]*N_vis)
        cores_fold[val_idx] = COR_VAL
        ax.barh([fold_idx]*N_vis, [1]*N_vis, left=range(N_vis),
                color=cores_fold, edgecolor="white", height=0.6)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"Fold {k+1}" for k in range(K)])
    ax.set_title(f"{K}-Fold Cross-Validation — estimativa confiavel e eficiente")
    ax.set_xlabel("Indice das amostras"); ax.set_xlim(0, N_vis)

    # Stratified K-Fold
    ax = axes[2]
    y_strat = np.array([0]*24 + [1]*6)
    np.random.shuffle(y_strat)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(N_vis), y_strat)):
        cores_strat = np.array([COR_TREINO]*N_vis)
        cores_strat[val_idx] = COR_VAL
        ax.barh([fold_idx]*N_vis, [1]*N_vis, left=range(N_vis),
                color=cores_strat, edgecolor="white", height=0.6)
        n_pos = np.sum(y_strat[val_idx] == 1)
        ax.text(N_vis + 0.4, fold_idx, f"{n_pos}/{len(val_idx)} pos.",
                va="center", fontsize=8, color="#6B2D00")
    ax.set_yticks(range(K)); ax.set_yticklabels([f"Fold {k+1}" for k in range(K)])
    ax.set_title(f"Stratified {K}-Fold — preserva proporcao de classes (~20% positivos)")
    ax.set_xlabel("Indice das amostras"); ax.set_xlim(0, N_vis + 5)

    # LOO
    ax = axes[3]
    N_loo = 15
    for sample_idx in range(min(N_loo, 10)):
        cores_loo = np.array([COR_TREINO]*N_loo)
        cores_loo[sample_idx] = COR_VAL
        ax.barh([sample_idx]*N_loo, [1]*N_loo, left=range(N_loo),
                color=cores_loo, edgecolor="white", height=0.6)
    ax.set_yticks(range(10)); ax.set_yticklabels([f"LOO-{k}" for k in range(10)])
    ax.set_title(f"Leave-One-Out (K=N={N_loo}) — minimo vies, custo O(N)")
    ax.set_xlabel("Indice das amostras"); ax.set_xlim(0, N_loo)

    plt.suptitle("Variantes de Cross-Validation — Padroes de Divisao",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_cv_variability():
    """Boxplots comparing CV method stability across seeds."""
    X_cv, y_cv = make_regression(n_samples=80, n_features=10, noise=15.0,
                                  n_informative=5, random_state=42)
    modelo_cv = Ridge(alpha=1.0)
    n_rep = 50
    resultados = {"Hold-Out": [], "K-Fold (K=5)": [], "K-Fold (K=10)": [], "LOO": []}

    for seed in range(n_rep):
        np.random.seed(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X_cv, y_cv, test_size=0.2, random_state=seed)
        modelo_cv.fit(X_tr, y_tr)
        resultados["Hold-Out"].append(mean_squared_error(y_te, modelo_cv.predict(X_te)))

        s5 = cross_val_score(modelo_cv, X_cv, y_cv,
                             cv=KFold(5, shuffle=True, random_state=seed),
                             scoring="neg_mean_squared_error")
        resultados["K-Fold (K=5)"].append(-s5.mean())

        s10 = cross_val_score(modelo_cv, X_cv, y_cv,
                              cv=KFold(10, shuffle=True, random_state=seed),
                              scoring="neg_mean_squared_error")
        resultados["K-Fold (K=10)"].append(-s10.mean())

    s_loo = cross_val_score(modelo_cv, X_cv, y_cv, cv=LeaveOneOut(),
                            scoring="neg_mean_squared_error")
    mse_loo = -s_loo.mean()
    resultados["LOO"] = [mse_loo] * n_rep

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    metodos  = list(resultados.keys())
    dados    = [resultados[m] for m in metodos]
    bp = ax.boxplot(dados, labels=metodos, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    cores_box = ["#E76F51", "#40916C", "#1B4332", "#E9C46A"]
    for patch, cor in zip(bp["boxes"], cores_box):
        patch.set_facecolor(cor); patch.set_alpha(0.55)

    ax.set_ylabel("MSE estimado")
    ax.set_title(f"Comparacao de CV ({n_rep} repeticoes com seeds diferentes)\n"
                 "Variabilidade menor = estimativa mais confiavel", fontsize=10)
    for i, (m, dados_m) in enumerate(resultados.items()):
        ax.text(i+1, ax.get_ylim()[1]*0.97, f"std={np.std(dados_m):.1f}",
                ha="center", fontsize=9, color=cores_box[i], fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def chart_coarse_to_fine():
    """Coarse-to-fine log-grid lambda search for Ridge."""
    np.random.seed(42)
    X_tune, y_tune = make_regression(n_samples=200, n_features=30, noise=12.0,
                                      n_informative=8, random_state=42)
    X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X_tune, y_tune,
                                                        test_size=0.25, random_state=42)
    kf_tune = KFold(n_splits=5, shuffle=True, random_state=42)

    lams_c = np.logspace(-5, 5, 20)
    sc_c   = [-cross_val_score(Ridge(alpha=lam), X_tr_t, y_tr_t,
                               cv=kf_tune, scoring="neg_mean_squared_error").mean()
              for lam in lams_c]
    best_c = lams_c[np.argmin(sc_c)]

    lams_f = np.logspace(np.log10(best_c)-1, np.log10(best_c)+1, 30)
    sc_f   = [-cross_val_score(Ridge(alpha=lam), X_tr_t, y_tr_t,
                               cv=kf_tune, scoring="neg_mean_squared_error").mean()
              for lam in lams_f]
    best_f = lams_f[np.argmin(sc_f)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    ax.semilogx(lams_c, sc_c, color="#40916C", lw=2.5, marker="o", ms=7, label="CV MSE (Coarse)")
    ax.axvline(best_c, color="#E76F51", ls="--", lw=2, label=f"Melhor coarse: {best_c:.4f}")
    ax.set_xlabel("lambda (Log-Scale)"); ax.set_ylabel("MSE (CV)")
    ax.set_title("Passo 1: Coarse Sweep\n(Log-Grid amplo: 10^-5 a 10^5)",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    ax.semilogx(lams_f, sc_f, color="#E9C46A", lw=2.5, marker="s", ms=7, label="CV MSE (Fine)")
    ax.axvline(best_f, color="#E76F51", ls="--", lw=2, label=f"Melhor fine: {best_f:.4f}")
    ax.set_xlabel("lambda (Log-Scale)"); ax.set_ylabel("MSE (CV)")
    ax.set_title(f"Passo 2: Fine Sweep\n(ao redor de lambda={best_c:.4f})",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("Estrategia Coarse-to-Fine com Log-Grid\n"
                 "Por que log-scale? Cada salto = ordem de magnitude",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_diagnosis_curves():
    """Under vs. Over-Regularization learning curves."""
    np.random.seed(5)
    X_diag = np.sort(np.random.uniform(0, 1, 60)).reshape(-1, 1)
    y_diag = np.sin(2*np.pi*X_diag.ravel()) + np.random.normal(0, 0.2, 60)

    tamanhos = np.arange(10, 61, 5)
    configs_diag = [
        (1e-6, "lambda=1e-6\n(Under-Regularization — High Variance)"),
        (1e-1, "lambda=0.1\n(Boa Regularizacao)"),
        (10.0, "lambda=10\n(Over-Regularization — High Bias)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")

    for ax, (lam, label) in zip(axes, configs_diag):
        ax.set_facecolor("#FAFAFA")
        erros_tr, erros_val = [], []
        for N in tamanhos:
            X_n, y_n = X_diag[:N], y_diag[:N]
            m = make_pipeline(PolynomialFeatures(8), Ridge(alpha=lam))
            cv_s = cross_val_score(m, X_n, y_n, cv=min(5, N),
                                   scoring="neg_mean_squared_error")
            m.fit(X_n, y_n)
            erros_tr.append(mean_squared_error(y_n, m.predict(X_n)))
            erros_val.append(-cv_s.mean())
        ax.plot(tamanhos, erros_tr,  color="#40916C", lw=2, label="Erro Treino")
        ax.plot(tamanhos, erros_val, color="#E76F51", lw=2, label="Erro Validacao")
        ax.fill_between(tamanhos, erros_tr, erros_val, alpha=0.15, color="#E76F51")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("N (amostras de treino)"); ax.set_ylabel("MSE")
        ax.set_ylim(0, 1.0); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("Diagnostico: Under vs. Over-Regularization via Learning Curves",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CONTENT BUILDER
# ---------------------------------------------------------------------------
def build_content(styles):
    S = styles
    story = []

    def P(text, style="body"):
        return Paragraph(text, S[style])

    # ---- Cover page story element (title area) ----------------------------
    story.append(spacer(0.5))
    story.append(P("Regularization Strategies & Cross-Validation", "title"))
    story.append(divider(GOLD, 2))
    story.append(spacer(0.15))
    story.append(P(
        "Esta aula une o rigor estatistico as tecnicas mais avancadas do Deep Learning "
        "moderno. Voce entendera que <b>regularizar um modelo</b> significa aplicar um "
        "filtro de sobriedade matematica aos seus parametros, seja restringindo-os "
        "geometricamente atraves de <b>circulos</b> (L2) ou <b>diamantes</b> (L1), ou "
        "sabotando o modelo intencionalmente para torna-lo robusto (<b>Dropout</b>). "
        "Veremos por que a restricao ideal de esparsidade (L0) e matematicamente "
        "intratavel e como a L1 e sua <b>convexificacao elegante</b>. Alem disso, voce "
        "aprendera a validar essas escolhas usando o ecossistema de <b>Cross-Validation</b>.",
        "body"))
    story.append(spacer(0.3))

    # Roteiro
    tbl_roteiro = comparison_table(
        ["#", "Topico", "Conceito Central"],
        [
            ["1", "Conceito de Regularizacao", "Hard, Soft/Variacional e Output Regularization"],
            ["2", "L2 Regularization (Ridge / Weight Decay)", "Geometria do circulo, solucao analitica"],
            ["3", "Sparsity & L1 (Lasso) + Elastic Net", "L0 NP-hard; L1 como convexificacao"],
            ["4", "Modern Regularization Techniques", "Early Stopping, Dropout, Label Smoothing"],
            ["5", "Probabilistic / Bayesian Connection", "Gaussian Prior => L2; Laplace Prior => L1"],
            ["6", "Cross-Validation Rigorosa", "Hold-Out, K-Fold, Stratified, LOO"],
            ["7", "Hyperparameter Tuning & Diagnostico", "Log-Grid, Coarse-to-Fine, Learning Curves"],
        ],
        [0.8*cm, CONTENT_W * 0.44, CONTENT_W * 0.51])
    story.append(tbl_roteiro)
    story.append(spacer(0.3))

    # =========================================================
    # SECTION 1 — Conceito de Regularizacao
    # =========================================================
    story.append(section_header("1. The Concept of Regularization", S))
    story.append(spacer(0.15))

    story.append(P(
        "<b>Regularization</b> (Regularizacao) e o conjunto de metodos para "
        "<b>controlar a complexidade do modelo</b> com o objetivo de melhorar a "
        "<b>generalizacao</b> — ou seja, o desempenho em dados novos nao vistos durante o treino. "
        "<b>Analogia do estudante superinteligente:</b> imagine um aluno que decora todas as "
        "respostas da lista de exercicios, incluindo erros de digitacao. Na prova, com questoes "
        "ligeiramente diferentes, ele falha. Regularizacao e como o professor que diz: "
        "'voce so pode usar formulas simples' — forcando o aluno a entender, nao decorar.",
        "body"))

    story.append(P("O framework unificado da Loss Function:", "subsection"))
    story.append(formula_block(
        r"\mathcal{L}(\theta;\, X,Y) = \frac{1}{N}\sum_{i=1}^{N}"
        r"\left\|f_\theta(x_i) - y_i\right\|^2",
        label="D(f_theta(x_i), y_i) = ||f_theta(x_i) - y_i||^2   (distancia MSE)",
        formula_size=14))
    story.append(spacer(0.2))

    story.append(P("Sem regularizacao, o otimizador encontra o theta* que minimiza L "
                   "nos dados de treino — potencialmente com <b>Overfitting severo</b>. "
                   "Os tres tipos de regularizacao abordam esse problema de formas distintas:",
                   "body"))
    story.append(spacer(0.15))

    for item in [
        "<b>1a. Hard Regularization:</b> o espaco de parametros e <b>restrito a um conjunto</b> "
        "C. O otimizador so pode explorar solucoes dentro desse conjunto — elegante "
        "matematicamente, mas dificil de implementar com gradiente descendente padrao.",
        "<b>1b. Soft / Variational Regularization:</b> em vez de proibir regioes, "
        "<b>adicionamos uma penalidade</b> lambda * R(theta) a Loss Function. O otimizador "
        "ainda pode ir a qualquer lugar, mas paga um custo crescente por parametros grandes.",
        "<b>1c. Output Regularization:</b> em vez de penalizar os parametros theta "
        "diretamente, penalizamos propriedades da <b>saida</b> f_theta(x_i). "
        "Exemplo: Label Smoothing evita confiar excessivamente em um unico label.",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.15))
    story.append(P("Formulacao de cada tipo:", "subsection"))
    story.append(formula_block(
        r"\text{Hard:}\;\min_\theta \mathcal{L}(\theta)\;\text{s.t.}\;\theta \in C"
        r"\quad|\quad"
        r"\text{Soft:}\;\mathcal{L}(\theta) + \lambda\,\mathcal{R}(\theta)"
        r"\quad|\quad"
        r"\text{Output:}\;\mathcal{L}(\theta) + \mathcal{R}(y_i)",
        label="lambda=0: sem reg.  |  lambda->inf: theta->0  |  C: conjunto viavel (ex: bola L2)",
        formula_size=10))
    story.append(spacer(0.2))

    fig1 = chart_three_reg_types()
    story.append(mat_image(fig1, 16))
    story.append(P(
        "Figura 1 — Hard Regularization (esquerda): otimizador projetado na bola de raio R. "
        "Soft Regularization (centro): as curvas de nivel da Loss total incluem o termo de penalidade. "
        "Output Regularization (direita): Label Smoothing suaviza a cross-entropy — penaliza "
        "confiar demais em um unico label.",
        "caption"))

    story.append(why_box(
        "Regularizacao e a resposta fundamental ao problema de Overfitting. Sem ela, qualquer "
        "modelo suficientemente complexo decora o ruido dos dados de treino e falha em producao. "
        "O framework unificado permite entender Ridge, Lasso, Dropout e Label Smoothing como "
        "manifestacoes do mesmo principio matematico.",
        "Em sklearn: Ridge(alpha=lambda), Lasso(alpha=lambda), ElasticNet(alpha=lambda, l1_ratio). "
        "Em PyTorch/TensorFlow: weight_decay nos otimizadores (L2), Dropout(p=0.5), "
        "LabelSmoothingCrossEntropy(eps=0.1).",
        S))
    story.append(PageBreak())

    # =========================================================
    # SECTION 2 — L2 Regularization
    # =========================================================
    story.append(section_header("2. L2 Regularization (Ridge / Weight Decay)", S))
    story.append(spacer(0.15))

    story.append(P(
        "A <b>L2 Regularization</b> restringe a <b>norma euclidiana quadratica</b> dos "
        "parametros. Conhecida como <b>Ridge Regression</b> em estatistica, "
        "<b>Weight Decay</b> em Deep Learning e <b>Tikhonov Regularization</b> em matematica "
        "aplicada. <b>Analogia da mola:</b> cada coeficiente theta_j e como um peso preso a "
        "uma mola ancorada na origem — os dados puxam o peso para o valor otimo, mas a mola "
        "resiste, nao deixando o coeficiente crescer alem do necessario.",
        "body"))

    story.append(P("Formulacao Soft (variacional — mais usada na pratica):", "subsection"))
    story.append(formula_block(
        r"\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\left\|f_\theta(x_i) - y_i\right\|^2"
        r" + \lambda\|\theta\|^2",
        label="MSE (Data Fit)  +  lambda * ||theta||^2 (penalidade L2)",
        formula_size=13))
    story.append(spacer(0.15))

    story.append(P("Solucao analitica fechada para regressao linear:", "subsection"))
    story.append(formula_block(
        r"\hat{\theta}_{\mathrm{Ridge}} = \left(X^TX + \lambda I\right)^{-1}X^Ty",
        label="Comparar com OLS: (X^T X)^{-1} X^T y — lambda I garante invertibilidade mesmo com X singular",
        formula_size=14))
    story.append(spacer(0.2))

    for item in [
        "<b>lambda = 0:</b> solucao OLS (sem regularizacao)",
        "<b>lambda -> inf:</b> theta -> 0 (underfitting total — modelo constante)",
        "<b>Efeito geometrico:</b> a regiao viavel Hard L2 e uma <b>bola euclidiana</b> (circulo em 2D). "
        "L2 encolhe todos os coeficientes proporcionalmente — <b>raramente zerando</b> algum exatamente.",
        "<b>Efeito de Smoothness:</b> L2 favorece solucoes com coeficientes pequenos e distribuicao "
        "uniforme — ideal para problemas de regressao com todas as features relevantes.",
        "<b>Escolha de lambda:</b> sempre via <b>Cross-Validation em Log-Grid</b> (ver Secao 7).",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.2))
    fig2 = chart_ridge_path()
    story.append(mat_image(fig2, 16))
    story.append(P(
        "Figura 2 — Esquerda: Ridge Regularization Path — todos os coeficientes encolhem "
        "suavemente com lambda crescente, mas nenhum chega a zero exatamente. "
        "Direita: com p=100 features e N=50 amostras (p>N), lambda pequeno causa "
        "Overfitting severo; lambda grande causa Underfitting. O ponto otimo esta no meio.",
        "caption"))

    story.append(why_box(
        "Ridge e a primeira linha de defesa contra Overfitting em modelos lineares. "
        "Quando p > N (mais features do que amostras), a matriz X<super>T</super>X e singular "
        "e OLS nao tem solucao unica — Ridge resolve isso adicionando lambda*I ao diagonal, "
        "tornando o sistema sempre invertivel. E o modelo padrao para regressao regularizada.",
        "Ridge em sklearn: Ridge(alpha=lambda). Em Deep Learning: o parametro 'weight_decay' "
        "no otimizador Adam ou SGD implementa exatamente L2 nos pesos. "
        "Regra pratica: comece sempre com Ridge antes de tentar modelos mais complexos.",
        S))
    story.append(PageBreak())

    # =========================================================
    # SECTION 3 — L1 / Lasso / Elastic Net
    # =========================================================
    story.append(section_header("3. The Sparsity Problem & L1 Regularization (Lasso)", S))
    story.append(spacer(0.15))

    story.append(P(
        "Muitas vezes, de centenas de features disponiveis, apenas algumas dezenas realmente "
        "importam. Um modelo <b>esparso</b> (muitos coeficientes exatamente zero) e mais "
        "interpretavel, mais rapido e mais robusto. "
        "<b>Analogia da mochila:</b> voce vai acampar e quer levar apenas o essencial. "
        "L0 pergunta: 'quais itens posso eliminar completamente?' (esparsidade real). "
        "L2 pergunta: 'posso usar versoes menores de todos os itens?' (encolhimento sem eliminacao).",
        "body"))

    story.append(P("L0 — A restricao ideal (NP-Hard):", "subsection"))
    story.append(formula_block(
        r"\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\left\|f_\theta(x_i) - y_i\right\|^2"
        r" + \lambda\|\theta\|_0 \quad \text{onde}\quad \|\theta\|_0 = \#\{j:\theta_j\neq 0\}",
        label="||theta||_0 nao e convexa, nao e diferenciavel, e NP-hard (exponencial em p)",
        formula_size=11))
    story.append(spacer(0.15))

    story.append(P("L1 (Lasso) — A convexificacao elegante:", "subsection"))
    story.append(formula_block(
        r"\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\left\|f_\theta(x_i) - y_i\right\|^2"
        r" + \lambda\|\theta\|_1 = \cdots + \lambda\sum_{j=1}^{d}|\theta_j|",
        label="Regiao viavel: diamante (octaedro em d dimensoes) — vertices nos eixos => esparsidade!",
        formula_size=11))
    story.append(spacer(0.15))

    story.append(P("Elastic Net — O melhor dos dois mundos:", "subsection"))
    story.append(formula_block(
        r"\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\left\|f_\theta(x_i) - y_i\right\|^2"
        r" + \lambda_1\|\theta\|_1 + \lambda_2\|\theta\|^2",
        label="l1_ratio = lambda_1/(lambda_1+lambda_2). Seleciona grupos de features correlacionadas.",
        formula_size=11))
    story.append(spacer(0.2))

    for item in [
        "<b>Por que L0 e inutilizavel?</b> Nao diferenciavel (gradiente nao existe), nao convexa "
        "(regiao viavel = uniao de eixos coordenados), e NP-hard para otimizar exatamente.",
        "<b>Por que L1 induz esparsidade?</b> Os <b>vertices do diamante</b> estao exatamente nos "
        "eixos coordenados (pontos onde theta_j=0 para j diferente de k). As curvas de nivel da "
        "Loss tendem a tocar o diamante nesses vertices — forcando coeficientes a zero!",
        "<b>Elastic Net:</b> quando features sao correlacionadas, Lasso seleciona arbitrariamente "
        "uma delas. Elastic Net seleciona <b>grupos</b> de features correlacionadas juntas.",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.2))
    fig3 = chart_geometry_l0_l1_l2()
    story.append(mat_image(fig3, 16))
    story.append(P(
        "Figura 3 — L0 (esquerda): a regiao viavel sao os proprios eixos coordenados — nao convexa, "
        "exige enumerar todos os 2<super>p</super> subconjuntos. L1 (centro): "
        "o diamante toca o eixo (theta_1=0) antes de atingir o otimo — esparsidade automatica! "
        "L2 (direita): o circulo toca a borda em um ponto generico — sem esparsidade exata.",
        "caption"))

    story.append(spacer(0.2))
    fig4 = chart_reg_paths_comparison()
    story.append(mat_image(fig4, 16))
    story.append(P(
        "Figura 4 — Regularization Paths (lendo da direita para esquerda = aumentando lambda). "
        "Ridge: todos os coeficientes encolhem continuamente. "
        "Lasso: coeficientes vao a zero um a um (esparsidade progressiva). "
        "Elastic Net: comportamento hibrido — esparsidade com selecao de grupos.",
        "caption"))

    story.append(why_box(
        "A escolha entre Ridge, Lasso e Elastic Net depende da hipotese sobre os dados. "
        "Se acreditamos que TODAS as features contribuem (mesmo que pouco) => Ridge. "
        "Se acreditamos que apenas POUCAS features importam => Lasso. "
        "Se as features sao correlacionadas e queremos esparsidade => Elastic Net.",
        "sklearn: Lasso(alpha=lambda), ElasticNet(alpha=lambda, l1_ratio=0.5). "
        "Para scikit-learn Lasso, max_iter=10000 evita warnings de convergencia. "
        "A escolha de l1_ratio no Elastic Net e outro hiperparametro a tunar via CV.",
        S, color=EMERALD))
    story.append(PageBreak())

    # =========================================================
    # SECTION 4 — Modern Regularization Techniques
    # =========================================================
    story.append(section_header("4. Modern Regularization Techniques", S))
    story.append(spacer(0.15))

    # 4a Early Stopping
    story.append(P("<b>4a. Early Stopping</b>", "subsection"))
    story.append(P(
        "<b>Early Stopping</b> reduz a Variance interrompendo o processo de otimizacao "
        "<b>antes</b> que o otimizador encontre o minimizador theta* da Loss de treino. "
        "<b>Analogia:</b> e como estudar para uma prova — parar na hora certa (quando "
        "entende o conceito) e otimo; continuar ate decorar cada detalhe "
        "(incluindo erros do livro) causa memorizar o irrelevante.",
        "body"))
    story.append(spacer(0.1))

    story.append(P("Para modelos lineares otimizados com Gradient Descent:", "highlight"))
    story.append(formula_block(
        r"\hat{\theta}_{\mathrm{EarlyStop}}(t) \approx "
        r"\hat{\theta}_{\mathrm{Ridge}}\!\left(\lambda = \frac{1}{\eta\, t}\right)",
        label="Parar na epoca t com lr=eta e matematicamente equivalente a L2 com lambda=1/(eta*t)",
        formula_size=13))
    story.append(spacer(0.2))

    story.append(P("<b>4b. Dropout</b>", "subsection"))
    story.append(P(
        "<b>Dropout</b> e uma tecnica para redes neurais: durante cada iteracao de treino, "
        "zeramos aleatoriamente uma fracao p dos neuronios. "
        "<b>Analogia do time de futebol:</b> treinar cada jogador para jogar bem mesmo sem "
        "alguns companheiros — quando o time completo jogar, cada um sera mais resiliente. "
        "Com d neuronios, existem 2<super>d</super> sub-redes possiveis — Dropout treina "
        "implicitamente um <b>Ensemble exponencial de modelos</b>.",
        "body"))
    story.append(spacer(0.1))

    story.append(formula_block(
        r"\tilde{h}_j = m_j \cdot h_j,\quad m_j \sim \mathrm{Bernoulli}(1-p)"
        r"\quad\Rightarrow\quad h_j^{\mathrm{test}} = (1-p)\cdot h_j",
        label="Treino: mascara binaria aleatoria  |  Teste: escalar por (1-p) compensa a media",
        formula_size=12))
    story.append(spacer(0.15))

    story.append(P("<b>4c. Label Smoothing</b>", "subsection"))
    story.append(P(
        "<b>Label Smoothing</b> substitui targets rigidos (One-Hot) por distribuicoes suaves, "
        "evitando que o modelo fique <b>superconfiante</b> (probabilidades muito proximas de 1.0). "
        "Usada principalmente em Transformers e classificacao de imagens.",
        "body"))
    story.append(formula_block(
        r"y_k^{\mathrm{smooth}} = y_k\,(1 - \epsilon) + \frac{\epsilon}{K}",
        label="epsilon=0.1, K=5 classes: hard label [1,0,0,0,0] => smooth [0.92, 0.02, 0.02, 0.02, 0.02]",
        formula_size=14))
    story.append(spacer(0.2))

    fig5 = chart_early_stopping()
    story.append(mat_image(fig5, 16))
    story.append(P(
        "Figura 5 — Early Stopping: a Loss de validacao atinge o minimo na epoca ~800, "
        "depois comeca a subir (Overfitting). O mecanismo de patience para o treino antes "
        "do maximo Overfitting. A curva azul (com Early Stopping) esta muito mais proxima "
        "da funcao verdadeira do que a curva vermelha (sem Early Stopping).",
        "caption"))

    story.append(spacer(0.2))
    fig6 = chart_dropout()
    story.append(mat_image(fig6, 16))
    story.append(P(
        "Figura 6 — Dropout com taxas crescentes. p=0 (esquerda): todos os neuronios ativos. "
        "p=0.5 (centro): ~50% zerados aleatoriamente a cada iteracao de treino. "
        "p=0.8 (direita): ~80% zerados — regularizacao mais agressiva. "
        "Em inferencia, NENHUM neuronio e zerado; os pesos sao escalados por (1-p).",
        "caption"))

    story.append(why_box(
        "Early Stopping e de longe a tecnica mais usada em Deep Learning porque nao requer "
        "adicionar termos extras a Loss Function — apenas monitorar a Loss de validacao. "
        "Dropout e essencial em redes profundas densas (MLP, Transformers). Label Smoothing "
        "melhora calibracao de probabilidade em classificadores.",
        "PyTorch: nn.Dropout(p=0.5) nas camadas ocultas; callbacks de EarlyStopping no "
        "PyTorch Lightning e Keras. sklearn nao tem Dropout (exclusivo de redes neurais). "
        "Label Smoothing: CrossEntropyLoss(label_smoothing=0.1) no PyTorch >= 1.10.",
        S, color=LIME))
    story.append(PageBreak())

    # =========================================================
    # SECTION 5 — Bayesian Connection
    # =========================================================
    story.append(section_header("5. The Probabilistic Interpretation — Bayesian Connection", S))
    story.append(spacer(0.15))

    story.append(P(
        "Aqui esta um dos resultados mais elegantes do Machine Learning: "
        "<b>toda regularizacao e equivalente a um prior Bayesiano sobre os parametros</b>. "
        "Quando voce escolhe L2, esta secretamente dizendo: 'acredito que os pesos verdadeiros "
        "sao pequenos e gaussianos'. "
        "<b>Analogia do detetive bayesiano:</b> antes de ver qualquer evidencia (dados), o "
        "detetive tem uma suspeita inicial (prior). Depois de ver as evidencias (likelihood), "
        "atualiza sua crenca (posterior). A regularizacao e a suspeita inicial codificada "
        "matematicamente.",
        "body"))

    story.append(P("Estimador MAP (Maximum A Posteriori):", "subsection"))
    story.append(formula_block(
        r"\hat{\theta}_{\mathrm{MAP}} = "
        r"\arg\min_\theta \left[-\log p(\mathcal{D}\mid\theta) - \log p(\theta)\right]",
        label="Teorema de Bayes: p(theta|D) proporcional a p(D|theta) * p(theta)",
        formula_size=13))
    story.append(spacer(0.2))

    story.append(P("Prova 1: Gaussian Prior => L2 Regularization:", "subsection"))
    story.append(formula_block(
        r"p(\theta) = \mathcal{N}(0,\,\tau^2 I)"
        r"\;\Rightarrow\; -\log p(\theta) = \frac{1}{2\tau^2}\|\theta\|^2 + C"
        r"\;\Rightarrow\; \lambda = \frac{\sigma^2}{\tau^2}",
        label="Gaussian Prior acredita que os pesos sao pequenos — induz encolhimento uniforme (L2)",
        formula_size=11))
    story.append(spacer(0.15))

    story.append(P("Prova 2: Laplace Prior => L1 Regularization (Lasso):", "subsection"))
    story.append(formula_block(
        r"p(\theta) \propto \exp\!\left(-\frac{|\theta|}{b}\right)"
        r"\;\Rightarrow\; -\log p(\theta) = \frac{1}{b}\|\theta\|_1 + C"
        r"\;\Rightarrow\; \lambda = \frac{\sigma^2}{b}",
        label="Laplace Prior tem pico pontiagudo em zero — acredita que a maioria dos pesos e ZERO (L1)",
        formula_size=11))
    story.append(spacer(0.2))

    for item in [
        "<b>-log p(D|theta)</b>: e o MSE (Negative Log-Likelihood com ruido gaussiano) — o "
        "termo de ajuste aos dados.",
        "<b>-log p(theta)</b>: e o termo de regularizacao — encoda a crenca a priori sobre "
        "os parametros antes de ver qualquer dado.",
        "<b>Gaussian (tau<super>2</super>):</b> quanto menor tau, mais forte a crenca "
        "em pesos pequenos => maior lambda L2.",
        "<b>Laplace (b):</b> o pico pontiagudo em zero e a geometria do diamante L1 "
        "sao dois lados da mesma moeda — esparsidade = crenca a priori de que a maioria "
        "dos pesos e zero.",
        "<b>Insight operacional:</b> lambda = sigma<super>2</super>/tau<super>2</super> "
        "(L2) ou lambda = sigma<super>2</super>/b (L1) — "
        "o hyperparametro lambda encoda quanto voce confia nos dados (sigma<super>2</super>) "
        "vs. no prior (tau<super>2</super> ou b).",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.2))
    fig7 = chart_bayesian_priors()
    story.append(mat_image(fig7, 16))
    story.append(P(
        "Figura 7 — Gaussian Prior (esquerda): curva em sino simetrica — acredita que pesos "
        "pequenos sao mais provaveis. Laplace Prior (centro): pico pontiagudo em zero — "
        "acredita <i>fortemente</i> que a maioria dos pesos deveria ser exatamente zero. "
        "Direita: -log p(theta) como funcao de theta — L2 (parabola), L1 (V), L0 (degrau).",
        "caption"))

    story.append(why_box(
        "A conexao Bayesiana e mais do que um exercicio academico — ela oferece um "
        "framework interpretativo poderoso: toda escolha de regularizacao e uma hipotese "
        "sobre a estrutura do mundo. Isso permite incorporar conhecimento de dominio "
        "diretamente na funcao objetivo via escolha do prior.",
        "Inferencia Bayesiana completa (MCMC, Variational Inference) e usada quando se "
        "precisa nao apenas do ponto MAP mas de toda a distribuicao posterior. Em problemas "
        "pequenos: BayesianRidge e ARDRegression do sklearn implementam inferencia "
        "Bayesiana eficiente.",
        S, color=FOREST))
    story.append(PageBreak())

    # =========================================================
    # SECTION 6 — Cross-Validation
    # =========================================================
    story.append(section_header("6. Rigorous Model Evaluation via Cross-Validation", S))
    story.append(spacer(0.15))

    story.append(P(
        "<b>Cross-Validation</b> e o conjunto de tecnicas para <b>estimar o erro de "
        "generalizacao</b> de forma confiavel, sem contaminar o conjunto de teste. "
        "Principio fundamental: <b>nunca use o conjunto de teste para tomar decisoes</b> "
        "— nem para escolher lambda, nem para comparar modelos. "
        "<b>Analogia escolar:</b> dados de treino = lista de exercicios; "
        "validacao = simulado (ajusta estrategia); teste = prova final (avalia uma unica vez).",
        "body"))

    story.append(P("K-Fold Cross-Validation (metodo mais usado na pratica):", "subsection"))
    story.append(formula_block(
        r"\text{CV}_K = \frac{1}{K}\sum_{k=1}^{K}\text{MSE}_k",
        label="K=5 ou K=10 sao os valores mais usados. K pequeno: rapido. K grande: mais confiavel.",
        formula_size=15))
    story.append(spacer(0.2))

    tbl_cv = comparison_table(
        ["Metodo", "Como funciona / Vantagem", "Desvantagem", "Quando usar"],
        [
            ["Hold-Out",
             "Split unico (80/20). Rapido — O(1) treinos.",
             "Estimativa instavel, depende do split aleatorio.",
             "N muito grande (>100k)"],
            ["K-Fold (K=5-10)",
             "K splits rotacionados, media dos K erros. Balanceado custo/precisao.",
             "K vezes mais lento que Hold-Out.",
             "Padrao para a maioria dos casos"],
            ["Stratified K-Fold",
             "Como K-Fold mas preserva % de classes em cada fold.",
             "Mesmo custo do K-Fold.",
             "Classificacao com classes raras (<10%)"],
            ["Leave-One-Out",
             "K=N: cada amostra e validacao uma vez. Minimo vies possivel.",
             "O(N) treinos — impraticavel para N grande.",
             "N muito pequeno (<50-100)"],
        ],
        [2.8*cm, CONTENT_W*0.42, CONTENT_W*0.28, CONTENT_W*0.22])
    story.append(tbl_cv)
    story.append(spacer(0.2))

    fig8 = chart_cv_patterns()
    story.append(mat_image_tall(fig8, 16, aspect=0.8))
    story.append(P(
        "Figura 8 — Padroes de divisao para cada variante de Cross-Validation. "
        "Verde: treino. Vermelho: validacao. Hold-Out: um unico split. "
        "K-Fold: cada fold e validacao uma vez. Stratified: proporcao de classes preservada "
        "em cada fold (indicado pelo texto 'X/Y pos.' a direita). LOO: cada amostra e "
        "validacao exatamente uma vez.",
        "caption"))

    story.append(spacer(0.2))
    fig9 = chart_cv_variability()
    story.append(mat_image(fig9, 15))
    story.append(P(
        "Figura 9 — Comparacao da variabilidade de cada metodo em 50 repeticoes com seeds "
        "diferentes. Hold-Out tem a maior variabilidade (desvio padrao alto). "
        "K-Fold (K=10) e LOO tem estimativas mais estaveis. LOO e deterministico "
        "(mesma estimativa a cada execucao).",
        "caption"))

    story.append(why_box(
        "A razao pela qual nao podemos usar o conjunto de teste para tunar lambda e que "
        "estariam otimizando para aquele conjunto especifico de amostras — vazando informacao "
        "do teste para o treino. O erro final reportado seria otimisticamente enviesado. "
        "Isso e um dos erros mais comuns em competicoes de ML.",
        "sklearn: cross_val_score(modelo, X, y, cv=KFold(5), scoring='neg_mean_squared_error'). "
        "Para classificacao desbalanceada: StratifiedKFold(n_splits=5, shuffle=True). "
        "Para series temporais (dados com dependencia temporal): TimeSeriesSplit em vez de KFold.",
        S, color=EMERALD))
    story.append(PageBreak())

    # =========================================================
    # SECTION 7 — Hyperparameter Tuning & Diagnosis
    # =========================================================
    story.append(section_header("7. Hyperparameter Tuning & Diagnosis Workflow", S))
    story.append(spacer(0.15))

    story.append(P(
        "O impacto da regularizacao opera em <b>ordens de magnitude</b>, nao em escala "
        "linear. A diferenca entre lambda=0.001 e lambda=0.01 e enorme (fator 10x), "
        "enquanto a diferenca entre lambda=0.1 e lambda=0.2 e minima. "
        "<b>Log-Grid</b> cobre uniformemente o espaco logaritmico onde o impacto e real:",
        "body"))

    story.append(formula_block(
        r"\lambda \in \{10^{-5},\,10^{-4},\,10^{-3},\,10^{-2},\,10^{-1},"
        r"\,10^0,\,10^1,\,10^2,\,10^3,\,10^4,\,10^5\}",
        label="np.logspace(-5, 5, 20)  — nunca use escala linear para lambda!",
        formula_size=11))
    story.append(spacer(0.2))

    story.append(P("<b>Estrategia Coarse-to-Fine:</b>", "subsection"))
    for item in [
        "<b>Passo 1 — Coarse Sweep:</b> Log-Grid amplo (10<super>-5</super> a "
        "10<super>5</super>) para encontrar a <b>ordem de magnitude certa</b>.",
        "<b>Passo 2 — Fine Sweep:</b> Log-Grid estreito ao redor do melhor valor "
        "encontrado — refina a precisao sem custo computacional proibitivo.",
        "<b>Regra de ouro:</b> usar sempre <b>K-Fold CV</b> para avaliar cada valor "
        "de lambda durante o tuning — NUNCA o conjunto de teste.",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.15))
    fig10 = chart_coarse_to_fine()
    story.append(mat_image(fig10, 16))
    story.append(P(
        "Figura 10 — Estrategia Coarse-to-Fine. Passo 1 (esquerda): sweep amplo identifica "
        "que o melhor lambda esta na regiao de 10<super>-1</super> a 10<super>1</super>. "
        "Passo 2 (direita): sweep fino ao redor do melhor valor encontrado refina a estimativa "
        "sem repetir todo o processo. Escala log-linear e essencial para visibilidade.",
        "caption"))

    story.append(spacer(0.2))
    story.append(P("<b>Diagnostico via Learning Curves:</b>", "subsection"))
    story.append(spacer(0.1))

    tbl_diag = comparison_table(
        ["Padrao observado", "Diagnostico", "Acao recomendada"],
        [
            ["Erro treino baixo, erro val. alto, GAP GRANDE",
             "Under-Regularization (High Variance / Overfitting)",
             "Aumentar lambda; mais dados; reduzir complexidade"],
            ["Erro treino alto, erro val. alto, GAP PEQUENO",
             "Over-Regularization (High Bias / Underfitting)",
             "Diminuir lambda; mais features; modelo mais complexo"],
            ["Erro treino aprox. erro val. aprox. Irred. Error",
             "Otimo — regularizacao bem calibrada",
             "Manter lambda; focar em features e dados"],
        ],
        [CONTENT_W * 0.32, CONTENT_W * 0.35, CONTENT_W * 0.33])
    story.append(tbl_diag)
    story.append(spacer(0.2))

    fig11 = chart_diagnosis_curves()
    story.append(mat_image(fig11, 16))
    story.append(P(
        "Figura 11 — Learning Curves para tres valores de lambda. Esquerda: lambda muito "
        "pequeno causa alto gap entre treino e validacao (Overfitting). Centro: lambda otimo "
        "— os dois erros convergem para um valor aceitavel. Direita: lambda muito grande "
        "causa ambos os erros elevados (Underfitting — modelo nao aprende nada util).",
        "caption"))

    story.append(why_box(
        "O Log-Grid e a estrategia Coarse-to-Fine sao o protocolo padrao da industria para "
        "tuning de hiperparametros de regularizacao. Usar escala linear desperdicaria a "
        "maioria dos pontos avaliados em regioes sem impacto real. O diagnostico via "
        "Learning Curves permite identificar QUAL componente (Bias ou Variance) domina o "
        "erro — sem isso, qualquer intervencao e cega.",
        "sklearn: GridSearchCV(Ridge(), param_grid={'alpha': np.logspace(-5,5,20)}, cv=5). "
        "Para Lasso: LassoCV(alphas=np.logspace(-5,5,20), cv=5).fit(X,y) — implementacao "
        "eficiente do CV via Coordinate Descent. RidgeCV e LassoCV sao alternativas "
        "altamente otimizadas ao GridSearchCV para esses modelos especificos.",
        S, color=GOLD))
    story.append(PageBreak())

    # =========================================================
    # SECTION 8 — Resumo
    # =========================================================
    story.append(section_header("8. Resumo Final & Checklist", S))
    story.append(spacer(0.15))

    tbl_resumo = comparison_table(
        ["Tecnica", "Penalidade / Mecanismo", "Prior Bayesiano", "Quando usar"],
        [
            ["L2 / Ridge", "lambda * ||theta||^2", "Gaussian N(0, tau^2)", "Padrao; p > N; colinearidade"],
            ["L1 / Lasso", "lambda * ||theta||_1", "Laplace(0, b)", "Feature Selection; esparsidade"],
            ["Elastic Net", "lambda1*||theta||_1 + lambda2*||theta||^2",
             "Laplace + Gaussian", "Features correlacionadas"],
            ["Early Stopping", "Parar na epoca t (equiv. L2 com lambda=1/eta*t)",
             "Implicitamente Gaussian", "Deep Learning; qualquer iterativo"],
            ["Dropout", "Mascarar p% dos neuronios (Ensemble implicito)",
             "—", "Redes neurais profundas (MLP, Transformer)"],
            ["Label Smoothing", "y_smooth = y*(1-eps) + eps/K",
             "—", "Classificacao; Transformers; evitar overconfidence"],
        ],
        [CONTENT_W*0.15, CONTENT_W*0.33, CONTENT_W*0.20, CONTENT_W*0.32])
    story.append(tbl_resumo)
    story.append(spacer(0.3))

    story.append(P("<b>Checklist de Regularizacao em novos projetos:</b>", "subsection"))
    for item in [
        "Sempre dividir dados em treino / validacao / teste antes de qualquer exploracao.",
        "Comecar com Ridge (L2) como baseline — simples, robusto, sempre funciona.",
        "Se interpretabilidade e esparsidade sao prioritarias => tentar Lasso ou Elastic Net.",
        "Sempre usar Log-Grid para buscar lambda: np.logspace(-5, 5, 20) como ponto de partida.",
        "Usar K-Fold CV (K=5 ou K=10) para avaliar cada lambda — NUNCA o conjunto de teste.",
        "Diagnosticar via Learning Curves: gap grande = Variance alta = aumentar lambda; "
        "ambos erros altos = Bias alto = diminuir lambda.",
        "Para Deep Learning: adicionar Dropout(p=0.3-0.5) e monitorar val loss para Early Stopping.",
        "NUNCA reportar metricas do conjunto de validacao como resultado final — "
        "reservar o teste para uma unica avaliacao no final.",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.3))
    story.append(divider(GOLD))
    story.append(spacer(0.15))

    story.append(P(
        "Na proxima aula, aprofundaremos os <b>metodos baseados em arvores de decisao</b> "
        "(Random Forests, Gradient Boosting) e veremos como os principios de regularizacao "
        "desta aula se manifestam em modelos nao-lineares — por exemplo, a profundidade maxima "
        "da arvore como Hard Regularization, e o shrinkage do Boosting como Soft Regularization. "
        "Os fundamentos de Cross-Validation desta aula sao identicos para avaliar esses modelos.",
        "body"))

    return story


# ---------------------------------------------------------------------------
# DOCUMENT ASSEMBLY
# ---------------------------------------------------------------------------
def build_pdf(output_path):
    total_pages = [None]
    canvas_cb   = LectureCanvas(total_pages)

    content_frame = Frame(
        SIDEBAR_W + MARGIN, FOOTER_H + 0.3*cm,
        CONTENT_W, PAGE_H - HEADER_H - FOOTER_H - 0.6*cm,
        id="content",
        leftPadding=0, rightPadding=0, topPadding=4, bottomPadding=4)

    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, id="cover",
                        leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0)

    cover_tpl   = PageTemplate(id="Cover",   frames=[cover_frame],
                                onPage=canvas_cb.draw_cover)
    content_tpl = PageTemplate(id="Content", frames=[content_frame],
                                onPage=canvas_cb.draw_page)

    doc = BaseDocTemplate(
        output_path, pagesize=A4,
        pageTemplates=[cover_tpl, content_tpl],
        leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0,
        title="Aula 3 — Regularization Strategies & Cross-Validation",
        author="Prof. Johannes Schwab, PhD — FH Kufstein Tirol",
        subject="Statistisches Lernen 2")

    styles = build_styles()
    story  = [NextPageTemplate("Content")] + build_content(styles)
    doc.build(story)

    # Two-pass for correct "Page X of Y" footer
    try:
        import PyPDF2
        with open(output_path, "rb") as f:
            total_pages[0] = len(PyPDF2.PdfReader(f).pages)
        doc2 = BaseDocTemplate(
            output_path, pagesize=A4,
            pageTemplates=[cover_tpl, content_tpl],
            leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0,
            title="Aula 3 — Regularization Strategies & Cross-Validation",
            author="Prof. Johannes Schwab, PhD — FH Kufstein Tirol",
            subject="Statistisches Lernen 2")
        story2 = [NextPageTemplate("Content")] + build_content(styles)
        doc2.build(story2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    out  = os.path.join(base, "L3_Regularization_CrossValidation.pdf")
    print(f"Gerando PDF: {out}")
    build_pdf(out)
    print("PDF gerado com sucesso!")
