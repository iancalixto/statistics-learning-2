"""
generate_L4_pdf.py
------------------
Generates a magazine-style lecture notes PDF for Lecture 4:
Probabilistic View, Bayesian Regularization, Cross-Validation & Bootstrapping
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
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut,
    cross_val_score, learning_curve, train_test_split,
)
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

# ---------------------------------------------------------------------------
# PALETTE  (L4 — deep burgundy / crimson / gold, distinct from L1-L3)
# ---------------------------------------------------------------------------
BURGUNDY   = colors.HexColor("#5C1A1A")   # very dark red — headers
CRIMSON    = colors.HexColor("#A93226")   # bright crimson — accents
ROSE       = colors.HexColor("#E74C3C")   # lighter red — highlights
GOLD       = colors.HexColor("#D4AC0D")   # warm gold — decorative
SLATE      = colors.HexColor("#2C3E50")   # dark blue-gray — contrast
LIGHT_BG   = colors.HexColor("#FFFFFF")   # pure white — body background
SIDEBAR_BG = colors.HexColor("#F4F4F4")   # neutral light grey — sidebar
ROW_ALT    = colors.HexColor("#F5F5F5")   # neutral light grey — table alternating rows
FORMULA_BG = colors.HexColor("#F0F4F8")   # very light blue-grey — formula boxes
WHITE      = colors.white
NEAR_BLACK = colors.HexColor("#111111")

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
        # Deep dark gradient: near-black at top, very dark burgundy at bottom
        steps = 60
        for i in range(steps):
            t = i / steps
            r = int(38 + t * (72 - 38))
            g = int(5  + t * (18 - 5))
            b = int(5  + t * (18 - 5))
            canvas.setFillColorRGB(r/255, g/255, b/255)
            canvas.rect(0, h * i / steps, w, h / steps + 1, fill=1, stroke=0)

        for cx, cy, cr, alpha in [
            (w * 0.82, h * 0.73, 4.0*cm, 0.11),
            (w * 0.08, h * 0.22, 5.0*cm, 0.08),
            (w * 0.62, h * 0.38, 2.5*cm, 0.15),
        ]:
            canvas.setFillColor(WHITE)
            canvas.setFillAlpha(alpha)
            canvas.circle(cx, cy, cr, fill=1, stroke=0)
        canvas.setFillAlpha(1.0)

        canvas.setFillColor(GOLD)
        canvas.rect(0, h * 0.55, SIDEBAR_W * 1.5, h * 0.45, fill=1, stroke=0)

        canvas.setFillColor(CRIMSON)
        canvas.roundRect(MARGIN, h * 0.78, 7.5*cm, 0.9*cm, 4, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(MARGIN + 0.3*cm, h * 0.78 + 0.25*cm,
                          "STATISTISCHES LERNEN 2")

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 28)
        canvas.drawString(MARGIN, h * 0.62, "Aula 4")
        canvas.setFont("Helvetica", 18)
        canvas.drawString(MARGIN, h * 0.56, "Probabilistic View &")
        canvas.drawString(MARGIN, h * 0.51, "Bayesian Regularization")

        canvas.setFillColor(GOLD)
        canvas.rect(MARGIN, h * 0.49, 8*cm, 0.07*cm, fill=1, stroke=0)

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(MARGIN, h * 0.45, "Prof. Johannes Schwab, PhD")
        canvas.drawString(MARGIN, h * 0.42, "FH Kufstein Tirol")

        canvas.setFillColor(GOLD)
        canvas.rect(0, 0, w, 0.4*cm, fill=1, stroke=0)
        canvas.setFillColor(CRIMSON)
        canvas.rect(0, 0.4*cm, w, 0.15*cm, fill=1, stroke=0)
        canvas.restoreState()

    def draw_page(self, canvas, doc):
        canvas.saveState()
        w, h = PAGE_W, PAGE_H
        pn = doc.page

        canvas.setFillColor(SIDEBAR_BG)
        canvas.rect(0, FOOTER_H, SIDEBAR_W, h - HEADER_H - FOOTER_H,
                    fill=1, stroke=0)
        canvas.setFillColor(BURGUNDY)
        canvas.rect(SIDEBAR_W - 0.15*cm, FOOTER_H,
                    0.15*cm, h - HEADER_H - FOOTER_H, fill=1, stroke=0)

        canvas.setFillColor(BURGUNDY)
        canvas.rect(0, h - HEADER_H, w, HEADER_H, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(0, h - HEADER_H - 0.12*cm, w, 0.12*cm, fill=1, stroke=0)

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(SIDEBAR_W + 0.4*cm, h - HEADER_H + 0.55*cm,
                          "Statistisches Lernen 2  |  Aula 4 — Probabilistic View & Bayesian Regularization")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 0.4*cm, h - HEADER_H + 0.55*cm, "FH Kufstein Tirol")

        canvas.setFillColor(colors.HexColor("#F8F8F8"))
        canvas.rect(0, 0, w, FOOTER_H, fill=1, stroke=0)
        canvas.setFillColor(BURGUNDY)
        canvas.rect(0, FOOTER_H - 0.1*cm, w, 0.1*cm, fill=1, stroke=0)

        canvas.setFillColor(colors.HexColor("#333333"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(SIDEBAR_W + 0.4*cm, 0.38*cm,
                          "Prof. Johannes Schwab, PhD  —  FH Kufstein Tirol")
        total = self._total[0] if self._total[0] else "?"
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(BURGUNDY)
        canvas.drawRightString(w - 0.4*cm, 0.38*cm, f"Pagina {pn} de {total}")

        canvas.saveState()
        canvas.translate(SIDEBAR_W * 0.5, h * 0.5)
        canvas.rotate(90)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(BURGUNDY)
        canvas.drawCentredString(0, 0, "PROBABILISTIC VIEW  |  BAYESIAN REG.  |  BOOTSTRAPPING")
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
                    textColor=BURGUNDY, spaceAfter=6, spaceBefore=4),
        "section": ps("section", fontName="Helvetica-Bold", fontSize=14,
                      textColor=WHITE, spaceBefore=14, spaceAfter=4),
        "subsection": ps("subsection", fontName="Helvetica-Bold", fontSize=11,
                         textColor=BURGUNDY, spaceBefore=8, spaceAfter=3),
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
                        textColor=BURGUNDY, spaceBefore=4, spaceAfter=4,
                        leftIndent=10),
        "code": ps("code", fontName="Courier", fontSize=8.5,
                   textColor=NEAR_BLACK,
                   backColor=colors.HexColor("#F8F0F0"),
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
        ("BACKGROUND",    (0,0), (-1,-1), BURGUNDY),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LINEBELOW",     (0,0), (-1,-1), 3, GOLD),
    ]))
    return tbl


def why_box(why_text, apply_text, styles, color=CRIMSON):
    WHY_HDR_BG  = colors.HexColor("#FDECEA")   # very light rose tint for header rows
    APP_HDR_BG  = colors.HexColor("#EAF0FD")   # very light blue tint for apply rows
    header_s = ParagraphStyle("wh", fontName="Helvetica-Bold", fontSize=9.5,
                               textColor=BURGUNDY, leading=13)
    header_a = ParagraphStyle("wha", fontName="Helvetica-Bold", fontSize=9.5,
                               textColor=SLATE, leading=13)
    body_s   = ParagraphStyle("wb", fontName="Helvetica", fontSize=9.5,
                               textColor=NEAR_BLACK, leading=14, alignment=TA_LEFT)
    data = [
        [Paragraph("Por que isso e importante para Data Science?", header_s)],
        [Paragraph(why_text, body_s)],
        [Paragraph("Aplicacao Pratica em Machine Learning", header_a)],
        [Paragraph(apply_text, body_s)],
    ]
    tbl = Table(data, colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), WHY_HDR_BG),
        ("BACKGROUND",    (0,1), (-1,1), WHITE),
        ("BACKGROUND",    (0,2), (-1,2), APP_HDR_BG),
        ("BACKGROUND",    (0,3), (-1,3), WHITE),
        ("LINEBEFORE",    (0,0), (-1,1),  4, CRIMSON),
        ("LINEBEFORE",    (0,2), (-1,3),  4, SLATE),
        ("LINEABOVE",     (0,0), (-1,0),  1.0, colors.HexColor("#CCCCCC")),
        ("LINEBELOW",     (0,3), (-1,-1), 1.0, colors.HexColor("#CCCCCC")),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#E0E0E0")),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    return tbl


def formula_block(latex_str, label="", bg_hex="#F0F4F8", formula_size=15):
    w_in = CONTENT_W / 72
    has_label = bool(label.strip())
    h_in = 1.05 if not has_label else 1.55

    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.patch.set_facecolor(bg_hex)
    ax.set_facecolor(bg_hex)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([0,1], [0.985, 0.985], color="#5C1A1A", linewidth=3.5,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0,1], [0.015, 0.015], color="#5C1A1A", linewidth=3.5,
            transform=ax.transAxes, clip_on=False)

    y_f = 0.63 if has_label else 0.50
    ax.text(0.5, y_f, f"${latex_str}$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=formula_size, color="#1A1A2E",
            math_fontfamily="dejavusans")

    if has_label:
        ax.text(0.5, 0.20, label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#444444", fontstyle="italic")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=bg_hex, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    img_h = (h_in / w_in) * CONTENT_W
    return Image(buf, width=CONTENT_W, height=img_h)


def data_table(headers, rows, col_widths):
    h_style = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=9,
                              textColor=WHITE, alignment=TA_CENTER)
    c_style = ParagraphStyle("td", fontName="Helvetica", fontSize=9,
                              textColor=NEAR_BLACK, alignment=TA_LEFT)
    data = [[Paragraph(h, h_style) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), c_style) for c in row])
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0), BURGUNDY),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, ROW_ALT]),
        ("GRID",           (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ("LINEBELOW",      (0,0), (-1,0),  1.5, GOLD),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
        ("RIGHTPADDING",   (0,0), (-1,-1), 8),
        ("TOPPADDING",     (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 6),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
    ]))
    return tbl


def mat_image(fig, width_cm=15):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_cm*cm, height=width_cm*cm*0.5)


def mat_image_tall(fig, width_cm=15, aspect=0.65):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_cm*cm, height=width_cm*cm*aspect)


def divider(color=CRIMSON, thickness=0.8):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=4, spaceBefore=4)


def spacer(h_cm=0.3):
    return Spacer(1, h_cm*cm)


# ---------------------------------------------------------------------------
# CHART GENERATORS
# ---------------------------------------------------------------------------

def chart_mle_vs_map():
    """MLE vs MAP L2 vs MAP L1 on a small dataset with an outlier."""
    theta_real = np.array([0.5, 1.2])
    x = np.array([-1.5, -0.8, -0.2, 0.3, 0.9, 1.5])
    ruido = np.array([0.05, -0.20, 0.15, -0.05, 0.10, 1.10])
    y = theta_real[0] + theta_real[1] * x + ruido
    X_mat = np.column_stack([np.ones_like(x), x])

    theta_mle = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ y)

    def gradient_map(theta, X, y_data, lam, tipo="l2", sigma=0.8):
        res = y_data - X @ theta
        grad = -(X.T @ res) / sigma**2
        if tipo == "l2":
            grad += lam * theta
        else:
            grad += lam * np.sign(theta)
        return grad

    def ajustar_map(X, y_data, lam, tipo="l2", n=8000, lr=0.01):
        theta = np.zeros(X.shape[1])
        for _ in range(n):
            theta -= lr * gradient_map(theta, X, y_data, lam, tipo)
        return theta

    lam = 2.0
    theta_l2 = ajustar_map(X_mat, y, lam, "l2")
    theta_l1 = ajustar_map(X_mat, y, lam, "l1")

    x_plot = np.linspace(-1.8, 1.8, 200)
    X_plot = np.column_stack([np.ones(200), x_plot])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    ax.scatter(x, y, s=80, color="#2C3E50", zorder=4, label="dados (treino)")
    ax.scatter(x[-1], y[-1], s=160, color="#E74C3C", zorder=5, marker="*", label="outlier")
    ax.plot(x_plot, X_plot @ theta_real, "k--", alpha=0.4, lw=1.5, label="real (simulado)")
    ax.plot(x_plot, X_plot @ theta_mle,  color="#E74C3C", lw=2, label="MLE (sem prior)")
    ax.plot(x_plot, X_plot @ theta_l2,   color="#27AE60", lw=2, label=f"MAP L2 (lambda={lam})")
    ax.plot(x_plot, X_plot @ theta_l1,   color="#8E44AD", lw=2, label=f"MAP L1 (lambda={lam})")
    ax.set_title("Efeito do prior: MLE vs MAP\nO prior protege contra o outlier",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    nomes = ["MLE\n(sem prior)", f"MAP L2\n(lambda={lam})", f"MAP L1\n(lambda={lam})", "Real"]
    incls = [theta_mle[1], theta_l2[1], theta_l1[1], theta_real[1]]
    cores = ["#E74C3C", "#27AE60", "#8E44AD", "#2C3E50"]
    bars = ax.bar(nomes, incls, color=cores, alpha=0.8, edgecolor="white", width=0.5)
    ax.axhline(theta_real[1], color="#2C3E50", ls="--", lw=1.5, alpha=0.6)
    ax.set_title("Inclinacao estimada theta_1\nPrior puxa para zero, reduzindo efeito do outlier",
                 fontsize=9.5, fontweight="bold")
    ax.set_ylabel("theta_1"); ax.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars, incls):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Probabilistic View: MLE vs MAP — o prior regulariza automaticamente",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_priors_and_penalties():
    """Gaussian vs Laplace prior + -log penalty side by side."""
    theta_v = np.linspace(-4, 4, 400)
    pr_gauss  = stats.norm(0, 1).pdf(theta_v)
    pr_lapl   = stats.laplace(0, 1).pdf(theta_v)
    pen_gauss = theta_v**2
    pen_lapl  = np.abs(theta_v)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    ax.plot(theta_v, pr_gauss, color="#27AE60", lw=2.5, label="Gaussiano => L2 Ridge")
    ax.plot(theta_v, pr_lapl,  color="#8E44AD", lw=2.5, ls="--", label="Laplace => L1 Lasso")
    ax.fill_between(theta_v, pr_gauss, alpha=0.10, color="#27AE60")
    ax.fill_between(theta_v, pr_lapl,  alpha=0.10, color="#8E44AD")
    ax.annotate("Laplace: cuspide em zero\n=> cre que theta = 0",
                xy=(0, pr_lapl[200]), xytext=(1.0, 0.40),
                arrowprops=dict(arrowstyle="->", color="#8E44AD"),
                color="#8E44AD", fontsize=9)
    ax.set_title("Prior p(theta)\nGaussiano (suave) vs Laplace (pontiagudo em zero)",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("theta"); ax.set_ylabel("p(theta)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    ax.plot(theta_v, pen_gauss, color="#27AE60", lw=2.5, label="||theta||^2 (L2/Gaussiano)")
    ax.plot(theta_v, pen_lapl,  color="#8E44AD", lw=2.5, ls="--", label="|theta| (L1/Laplace)")
    ax.plot(theta_v, (np.abs(theta_v)>0).astype(float),
            color="#888888", lw=2, ls=":", label="||theta||_0 (L0 — NP-hard)")
    ax.set_xlim(-3, 3); ax.set_ylim(-0.1, 4)
    ax.set_title("-log p(theta) = Penalidade de Regularizacao\nL2 parabola | L1 em V | L0 degrau",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("theta"); ax.set_ylabel("-log p(theta)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("Bayesian Connection: Prior => Regularizacao",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_sparsity():
    """Ridge vs Lasso coefficients — sparsity demonstration."""
    X_sp, y_sp = make_regression(n_samples=80, n_features=20, n_informative=5,
                                  noise=10, random_state=0)
    ridge = Ridge(alpha=1.0).fit(X_sp, y_sp)
    lasso = Lasso(alpha=1.0, max_iter=5000).fit(X_sp, y_sp)

    n_ridge_zero = np.sum(np.abs(ridge.coef_) < 0.01)
    n_lasso_zero = np.sum(lasso.coef_ == 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    idx = np.arange(20)

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    cols_r = ["#27AE60" if abs(c) > 0.01 else "#CCCCCC" for c in ridge.coef_]
    ax.bar(idx, ridge.coef_, color=cols_r, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Ridge (L2) — todos os coeficientes nao nulos\n"
                 f"aprox. zeros: {n_ridge_zero}/20 (encolhimento suave)",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Indice da feature"); ax.set_ylabel("Coeficiente")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    cols_l = ["#8E44AD" if c != 0 else "#CCCCCC" for c in lasso.coef_]
    ax.bar(idx, lasso.coef_, color=cols_l, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Lasso (L1) — esparsidade automatica!\n"
                 f"zeros exatos: {n_lasso_zero}/20 (feature selection)",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Indice da feature"); ax.set_ylabel("Coeficiente")
    ax.grid(alpha=0.3, axis="y")
    roxo_p = mpatches.Patch(color="#8E44AD", label="Feature ativa")
    cinza_p = mpatches.Patch(color="#CCCCCC", label="Feature zerada (eliminada)")
    ax.legend(handles=[roxo_p, cinza_p], fontsize=9)

    plt.suptitle("Sparsity: Lasso forca coeficientes a zero — Feature Selection automatica",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_coarse_to_fine():
    """Coarse-to-fine lambda search."""
    np.random.seed(0)
    X_hp, y_hp = make_regression(n_samples=200, n_features=15, n_informative=6,
                                  noise=20, random_state=0)
    X_hp = StandardScaler().fit_transform(X_hp)
    X_tr, X_val, y_tr, y_val = train_test_split(X_hp, y_hp, test_size=0.25, random_state=0)

    lams_c = np.logspace(-6, 2, 30)
    mse_c  = [mean_squared_error(y_val, Ridge(alpha=l).fit(X_tr, y_tr).predict(X_val))
              for l in lams_c]
    best_c = lams_c[np.argmin(mse_c)]

    lams_f = np.logspace(np.log10(best_c)-1, np.log10(best_c)+1, 40)
    mse_f  = [mean_squared_error(y_val, Ridge(alpha=l).fit(X_tr, y_tr).predict(X_val))
              for l in lams_f]
    best_f = lams_f[np.argmin(mse_f)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    ax.semilogx(lams_c, mse_c, "o-", color="#2C3E50", lw=2, ms=5)
    ax.axvline(best_c, color="#E74C3C", ls="--", lw=2, label=f"lambda* coarse = {best_c:.1e}")
    ax.axvspan(best_c/10, best_c*10, alpha=0.12, color="#D4AC0D", label="regiao para refinamento")
    ax.set_title("Fase 1: Coarse Sweep\nLog-grid amplo: 10^-6 a 10^2",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("lambda"); ax.set_ylabel("Val-MSE")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    ax.semilogx(lams_f, mse_f, "s-", color="#A93226", lw=2, ms=5)
    ax.axvline(best_f, color="#E74C3C", ls="--", lw=2, label=f"lambda* fine = {best_f:.1e}")
    ax.set_title(f"Fase 2: Fine Sweep\nAo redor de lambda={best_c:.1e}",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("lambda"); ax.set_ylabel("Val-MSE")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("Coarse-to-Fine Log-Grid Hyperparameter Search — Ridge (L2)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_multi_seed():
    """Multi-seed averaging for small N hyperparameter search."""
    np.random.seed(5)
    X_t, y_t = make_regression(n_samples=40, n_features=10, n_informative=4,
                                noise=15, random_state=5)
    X_t = StandardScaler().fit_transform(X_t)

    lams = np.logspace(-5, 2, 40)
    n_seeds = 5
    all_mse = np.zeros((n_seeds, len(lams)))
    for s in range(n_seeds):
        Xtr, Xv, ytr, yv = train_test_split(X_t, y_t, test_size=0.25, random_state=s)
        for j, lam in enumerate(lams):
            all_mse[s, j] = mean_squared_error(yv, Ridge(alpha=lam).fit(Xtr, ytr).predict(Xv))

    media = all_mse.mean(axis=0)
    std   = all_mse.std(axis=0)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    for s in range(n_seeds):
        ax.semilogx(lams, all_mse[s], alpha=0.20, color="gray", lw=1)
    ax.semilogx(lams, media, color="#5C1A1A", lw=2.5, label=f"Media ({n_seeds} seeds)")
    ax.fill_between(lams, media - std, media + std, alpha=0.20, color="#A93226",
                    label="+/- 1 std")
    best_idx = np.argmin(media)
    ax.axvline(lams[best_idx], color="#D4AC0D", ls="--", lw=2,
               label=f"lambda* medio = {lams[best_idx]:.1e}")
    ax.set_title("Multi-seed averaging (N=40) — reduz variabilidade na escolha de lambda\n"
                 "Linhas cinzas: val-MSE de cada seed individual",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("lambda"); ax.set_ylabel("Val-MSE")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def chart_learning_curves():
    """Under / Correct / Over-regularization learning curves."""
    np.random.seed(1)
    X_d, y_d = make_regression(n_samples=300, n_features=20, n_informative=8,
                                noise=25, random_state=1)
    X_d = StandardScaler().fit_transform(X_d)

    cenarios = [
        ("Under-Regularizacao\n(lambda=1e-6, High Variance)", Ridge(alpha=1e-6), "#E74C3C"),
        ("Regularizacao Correta\n(lambda=1.0)", Ridge(alpha=1.0), "#27AE60"),
        ("Over-Regularizacao\n(lambda=1e4, High Bias)", Ridge(alpha=1e4), "#2980B9"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")
    tamanhos = np.linspace(0.1, 1.0, 10)

    for ax, (titulo, modelo, cor) in zip(axes, cenarios):
        ax.set_facecolor("#FAFAFA")
        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        sizes, tr_s, val_s = learning_curve(
            modelo, X_d, y_d, train_sizes=tamanhos,
            cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
        tr_mse  = -tr_s.mean(axis=1)
        val_mse = -val_s.mean(axis=1)
        ax.plot(sizes, tr_mse,  "o-", color=cor,   lw=2, label="Train MSE")
        ax.plot(sizes, val_mse, "s--", color="#888", lw=2, label="Val MSE")
        ax.fill_between(sizes, tr_mse, val_mse, alpha=0.12, color=cor)
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.set_xlabel("N amostras de treino"); ax.set_ylabel("MSE")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        gap = val_mse[-1] - tr_mse[-1]
        ax.annotate(f"gap={gap:.0f}", xy=(sizes[-1], (val_mse[-1]+tr_mse[-1])/2),
                    ha="right", fontsize=9, color=cor)

    plt.suptitle("Diagnosis via Learning Curves — Under / Correct / Over-Regularization",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_cv_splits():
    """Visual CV split patterns — Hold-Out, K-Fold, Stratified, LOO."""
    N_vis = 20
    COR_TR  = "#27AE60"
    COR_VAL = "#E74C3C"

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), facecolor="white")

    # Hold-Out
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    split = int(0.75 * N_vis)
    cores_ho = [COR_TR]*split + [COR_VAL]*(N_vis - split)
    ax.barh([0]*N_vis, [1]*N_vis, left=range(N_vis),
            color=cores_ho, edgecolor="white", height=0.6)
    ax.set_yticks([0]); ax.set_yticklabels(["Hold-Out"])
    ax.set_xlim(0, N_vis); ax.set_title("Hold-Out (75/25 split) — 1 treino, rapido")
    tr_p = mpatches.Patch(color=COR_TR,  label="Treino")
    va_p = mpatches.Patch(color=COR_VAL, label="Validacao")
    ax.legend(handles=[tr_p, va_p], loc="lower right", fontsize=9)

    # K-Fold
    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    kf = KFold(n_splits=5, shuffle=False)
    for fold_idx, (tr_i, val_i) in enumerate(kf.split(np.arange(N_vis))):
        cf = np.array([COR_TR]*N_vis); cf[val_i] = COR_VAL
        ax.barh([fold_idx]*N_vis, [1]*N_vis, left=range(N_vis),
                color=cf, edgecolor="white", height=0.6)
    ax.set_yticks(range(5)); ax.set_yticklabels([f"Fold {k+1}" for k in range(5)])
    ax.set_title("5-Fold Cross-Validation — padrao recomendado")
    ax.set_xlim(0, N_vis)

    # Stratified K-Fold
    ax = axes[2]
    ax.set_facecolor("#FAFAFA")
    y_cls = (np.arange(N_vis) % 3).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for fold_idx, (tr_i, val_i) in enumerate(skf.split(np.zeros((N_vis,1)), y_cls)):
        cf = np.array([COR_TR]*N_vis); cf[val_i] = COR_VAL
        ax.barh([fold_idx]*N_vis, [1]*N_vis, left=range(N_vis),
                color=cf, edgecolor="white", height=0.6)
        ax.text(N_vis+0.3, fold_idx, f"cls.={np.unique(y_cls[val_i])}",
                va="center", fontsize=7.5, color="#555")
    ax.set_yticks(range(5)); ax.set_yticklabels([f"Fold {k+1}" for k in range(5)])
    ax.set_title("Stratified K-Fold — preserva proporcao de classes por fold")
    ax.set_xlim(0, N_vis+5)

    # LOO
    ax = axes[3]
    ax.set_facecolor("#FAFAFA")
    for sample_idx in range(min(N_vis, 10)):
        cf = np.array([COR_TR]*N_vis); cf[sample_idx] = COR_VAL
        ax.barh([sample_idx]*N_vis, [1]*N_vis, left=range(N_vis),
                color=cf, edgecolor="white", height=0.6)
    ax.set_yticks(range(10)); ax.set_yticklabels([f"LOO-{k}" for k in range(10)])
    ax.set_title(f"Leave-One-Out — 1 amostra e validacao por vez (N={N_vis} treinos total)")
    ax.set_xlim(0, N_vis)

    plt.suptitle("Padroes de Divisao: Hold-Out | K-Fold | Stratified | LOO",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_cv_comparison():
    """CV method comparison — MSE and variability."""
    np.random.seed(3)
    X_cv, y_cv = make_regression(n_samples=120, n_features=10, n_informative=5,
                                  noise=20, random_state=3)
    X_cv = StandardScaler().fit_transform(X_cv)
    modelo = Ridge(alpha=1.0)

    holdout = []
    for seed in range(30):
        Xtr, Xv, ytr, yv = train_test_split(X_cv, y_cv, test_size=0.25, random_state=seed)
        modelo.fit(Xtr, ytr)
        holdout.append(mean_squared_error(yv, modelo.predict(Xv)))

    s5  = -cross_val_score(modelo, X_cv, y_cv, cv=KFold(5,  shuffle=True, random_state=0),
                           scoring="neg_mean_squared_error")
    s10 = -cross_val_score(modelo, X_cv, y_cv, cv=KFold(10, shuffle=True, random_state=0),
                           scoring="neg_mean_squared_error")
    s_loo = -cross_val_score(modelo, X_cv, y_cv, cv=LeaveOneOut(),
                             scoring="neg_mean_squared_error")

    resultados = {"Hold-Out\n(30 rep)": np.array(holdout),
                  "5-Fold CV": s5, "10-Fold CV": s10,
                  f"LOO\n(N={len(X_cv)})": s_loo}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    nomes = list(resultados.keys())
    medias = [v.mean() for v in resultados.values()]
    stds   = [v.std()  for v in resultados.values()]
    cores_cv = ["#E74C3C", "#27AE60", "#2980B9", "#8E44AD"]

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    bars = ax.bar(nomes, medias, color=cores_cv, alpha=0.8, yerr=stds,
                  capsize=6, edgecolor="white")
    ax.set_title("MSE medio por metodo de CV\n(barras de erro = std dos scores)",
                 fontsize=9.5, fontweight="bold")
    ax.set_ylabel("MSE (menor = melhor)"); ax.grid(alpha=0.3, axis="y")
    for bar, v, s in zip(bars, medias, stds):
        ax.text(bar.get_x()+bar.get_width()/2, v+s+0.5,
                f"{v:.0f}", ha="center", fontsize=8)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    bp = ax.boxplot([v for v in resultados.values()], labels=nomes,
                    patch_artist=True, medianprops=dict(color="black", lw=2))
    for patch, cor in zip(bp["boxes"], cores_cv):
        patch.set_facecolor(cor); patch.set_alpha(0.5)
    ax.set_title("Variabilidade dos scores por fold\n(boxplot — variancia menor = CV mais estavel)",
                 fontsize=9.5, fontweight="bold")
    ax.set_ylabel("|MSE| por fold"); ax.grid(alpha=0.3, axis="y")
    for i, (m, s) in enumerate(zip(medias, stds)):
        ax.text(i+1, ax.get_ylim()[1]*0.95, f"std={s:.0f}",
                ha="center", fontsize=8, color=cores_cv[i], fontweight="bold")

    plt.suptitle("Comparacao de variantes de Cross-Validation — MSE e estabilidade",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_bootstrap_coefs():
    """Bootstrap coefficient uncertainty — histograms with CI."""
    np.random.seed(42)
    N_boot, p_boot = 80, 5
    X_b, y_b = make_regression(n_samples=N_boot, n_features=p_boot,
                                n_informative=3, noise=20, random_state=42)
    X_b = StandardScaler().fit_transform(X_b)

    B = 500
    coefs_boot = np.zeros((B, p_boot))
    modelo_b = Ridge(alpha=1.0)
    modelo_b.fit(X_b, y_b)
    coefs_orig = modelo_b.coef_.copy()

    for b in range(B):
        idx = np.random.choice(N_boot, size=N_boot, replace=True)
        Ridge(alpha=1.0).fit(X_b[idx], y_b[idx])
        m = Ridge(alpha=1.0); m.fit(X_b[idx], y_b[idx])
        coefs_boot[b] = m.coef_

    fig, axes = plt.subplots(1, p_boot, figsize=(15, 4.5),
                             sharey=False, facecolor="white")
    for j, ax in enumerate(axes):
        ax.set_facecolor("#FAFAFA")
        ci_lo = np.percentile(coefs_boot[:, j], 2.5)
        ci_hi = np.percentile(coefs_boot[:, j], 97.5)
        sig = ci_lo > 0 or ci_hi < 0
        ax.hist(coefs_boot[:, j], bins=28, color="#A93226" if sig else "#888888",
                alpha=0.65, density=True, edgecolor="white")
        ax.axvline(coefs_orig[j], color="#5C1A1A", lw=2.5,
                   label=f"Orig={coefs_orig[j]:.2f}")
        ax.axvline(ci_lo, color="#D4AC0D", ls="--", lw=1.5, label="IC 95%")
        ax.axvline(ci_hi, color="#D4AC0D", ls="--", lw=1.5)
        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.set_title(f"Feature {j+1}" + (" (*)" if sig else ""),
                     fontsize=10, fontweight="bold",
                     color="#5C1A1A" if sig else "#555")
        ax.set_xlabel(f"theta_{j+1}"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle("Bootstrap Coefficient Uncertainty (B=500) — (*) IC nao inclui zero => feature significativa",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_bagging():
    """Bagging: bootstrap + averaging reduces variance."""
    np.random.seed(10)
    N_bag = 60
    x_bag = np.sort(np.random.uniform(-3, 3, N_bag))
    y_bag = np.sin(x_bag) + np.random.normal(0, 0.4, N_bag)
    x_pred = np.linspace(-3, 3, 300)

    def poly_pred(x_tr, y_tr, x_p, grau=7):
        return np.polyval(np.polyfit(x_tr, y_tr, grau), x_p)

    B_bag = 200
    preds = np.zeros((B_bag, len(x_pred)))
    for b in range(B_bag):
        idx = np.random.choice(N_bag, N_bag, replace=True)
        try:
            preds[b] = poly_pred(x_bag[idx], y_bag[idx], x_pred)
        except Exception:
            preds[b] = np.zeros(len(x_pred))

    pred_bag    = preds.mean(axis=0)
    pred_single = poly_pred(x_bag, y_bag, x_pred)
    mse_single  = np.mean((np.sin(x_pred) - pred_single)**2)
    mse_bag     = np.mean((np.sin(x_pred) - pred_bag)**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    for b in range(0, B_bag, 20):
        ax.plot(x_pred, preds[b], alpha=0.15, color="#A93226", lw=1)
    ax.plot(x_pred, pred_single, color="#E74C3C", lw=2.5, label=f"Modelo unico (MSE={mse_single:.3f})")
    ax.scatter(x_bag, y_bag, s=20, color="#555", zorder=3, alpha=0.5)
    ax.plot(x_pred, np.sin(x_pred), "k--", lw=2, label="sin(x) verdadeiro")
    ax.set_ylim(-3, 3); ax.set_title("Modelos individuais bootstrap — alta variancia",
                                      fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    ax.plot(x_pred, pred_bag, color="#5C1A1A", lw=2.5,
            label=f"Bagging B={B_bag} (MSE={mse_bag:.3f})")
    ax.fill_between(x_pred, preds.mean(0)-preds.std(0), preds.mean(0)+preds.std(0),
                    alpha=0.18, color="#A93226", label="+/- 1 std bootstrap")
    ax.plot(x_pred, np.sin(x_pred), "k--", lw=2, label="sin(x) verdadeiro")
    ax.scatter(x_bag, y_bag, s=20, color="#555", zorder=3, alpha=0.5)
    ax.set_ylim(-3, 3)
    reducao = (mse_single - mse_bag) / mse_single * 100
    ax.set_title(f"Bagging: media das {B_bag} previsoes bootstrap\nReducao MSE: {reducao:.1f}%",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("Bagging = Bootstrap + Averaging => reduz variancia",
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

    # ---- Title & Roteiro --------------------------------------------------
    story.append(spacer(0.5))
    story.append(P("Visao Probabilistica, Bayesian Regularization & Bootstrapping", "title"))
    story.append(divider(GOLD, 2))
    story.append(spacer(0.15))
    story.append(P(
        "Esta aula revela o <b>fundamento probabilistico</b> de tudo que aprendemos ate agora. "
        "O MSE nao e arbitrario — emerge da hipotese de ruido Gaussiano. "
        "A regularizacao nao e um truque — e a incorporacao de conhecimento previo via <b>prior Bayesiano</b>. "
        "Depois da teoria, exploramos o <b>protocolo pratico</b> de escolha de hiperparametros, "
        "o diagnostico de Under vs. Over-regularizacao, e encerramos com "
        "<b>Bootstrapping</b> — a tecnica de resampling para quantificar incerteza "
        "e a base do Bagging (Random Forests).",
        "body"))
    story.append(spacer(0.25))

    tbl_roteiro = data_table(
        ["#", "Topico", "Conceito Central"],
        [
            ["1", "Probabilistic View", "Likelihood, Posterior, MAP — por que MSE e regularizacao emergem naturalmente"],
            ["2", "Prior Choices", "Gaussian Prior => L2 Ridge | Laplace Prior => L1 Lasso"],
            ["3", "Decision Process", "Quando usar qual regularizacao — 4 cenarios do slide"],
            ["4", "Hyperparameter Choice", "Log-grid, Coarse-to-Fine, Multi-seed, Protocolo completo"],
            ["5", "Diagnosis", "Under vs. Over-regularizacao via Learning Curves"],
            ["6", "Cross-Validation", "Hold-Out, K-Fold, Stratified, LOO — comparacao quantitativa"],
            ["7", "Bootstrapping", "Resampling com reposicao, Confidence Intervals, Bagging"],
        ],
        [0.8*cm, CONTENT_W * 0.35, CONTENT_W * 0.56])
    story.append(tbl_roteiro)
    story.append(spacer(0.3))

    # =========================================================
    # SECTION 1 — Probabilistic View
    # =========================================================
    story.append(section_header("1. Probabilistic View — Por que MSE e Regularizacao emergem do Bayes", S))
    story.append(spacer(0.15))

    story.append(P(
        "Ate agora, treinar um modelo significava <b>minimizar um erro</b> (ex: MSE). "
        "A <b>Probabilistic View</b> pergunta algo diferente: dado que observamos os dados "
        "D = {(x_i, y_i)}, qual conjunto de parametros theta e mais <b>provavel</b>? "
        "Essa mudanca conecta dois conceitos:", "body"))

    for item in [
        "<b>Likelihood p(D|theta):</b> 'Se o modelo tivesse estes parametros, quao provaveis seriam os dados observados?'",
        "<b>Prior p(theta):</b> 'Antes de ver qualquer dado, qual configuracao de theta acredito ser mais provavel?'",
        "<b>Posterior p(theta|D):</b> 'Dado que vi estes dados, quao provavel e cada configuracao de theta?'",
        "<b>MAP (Maximum A Posteriori):</b> escolher theta que maximiza o Posterior — e exatamente o que fazemos no treino com loss + regularizador.",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.2))
    story.append(P("Passo 1 — Likelihood com ruido Gaussiano:", "subsection"))
    story.append(formula_block(
        r"p(D|\theta) = \prod_{i=1}^{N}\exp\left(-\|f_\theta(x_i)-y_i\|^2\right)"
        r"\quad\Rightarrow\quad"
        r"-\log p(D|\theta) = \sum_{i=1}^{N}\|f_\theta(x_i)-y_i\|^2 \equiv \text{MSE}",
        label="-log Likelihood com ruido Gaussiano = MSE / SSE  — nao e arbitrario!",
        formula_size=11))
    story.append(spacer(0.15))

    story.append(P("Passo 2 — Teorema de Bayes:", "subsection"))
    story.append(formula_block(
        r"p(\theta|D) \propto p(D|\theta)\cdot p(\theta)",
        label="Posterior  proporcional a  Likelihood  x  Prior",
        formula_size=16))
    story.append(spacer(0.15))

    story.append(P("Passo 3 — Objetivo MAP (minimizar -log Posterior):", "subsection"))
    story.append(formula_block(
        r"\hat{\theta}_{\text{MAP}} = \arg\min_\theta"
        r"\left[-\log p(D|\theta) - \log p(\theta)\right]"
        r"= \arg\min_\theta\left[\text{MSE} + \lambda\,\mathcal{R}(\theta)\right]",
        label="-log p(D|theta) = MSE  |  -log p(theta) = Regularizacao  |  lambda conecta os dois",
        formula_size=11))
    story.append(spacer(0.2))

    story.append(P(
        "Este resultado e poderoso: o treinamento com <b>Loss + Regularizador</b> e "
        "<b>identico</b> a encontrar o estimador MAP. A regularizacao nao e um truque "
        "ad-hoc — e a incorporacao rigorosa de conhecimento a priori via Prior Bayesiano.",
        "body"))

    story.append(spacer(0.2))
    fig1 = chart_mle_vs_map()
    story.append(mat_image(fig1, 16))
    story.append(P(
        "Figura 1 — Dataset com um outlier. MLE (vermelho) e muito influenciado pelo outlier "
        "e estima uma inclinacao muito maior que a real. MAP com L2 (verde) e MAP com L1 "
        "(roxo) sao 'puxados' pelo prior para valores menores, se aproximando do verdadeiro "
        "theta_1 = 1.2. O prior regulariza automaticamente.",
        "caption"))

    story.append(why_box(
        "A Probabilistic View explica POR QUE usamos MSE (ruido Gaussiano) e POR QUE "
        "regularizamos (prior sobre parametros). Isso permite escolher a loss function "
        "corretamente para cada problema: ruido Gaussiano => MSE, labels binarios => "
        "cross-entropy (ruido Bernoulli), dados de contagem => Poisson deviance.",
        "Em sklearn: Ridge e Lasso implementam MAP com Gaussian e Laplace prior "
        "respectivamente. Em PyTorch: o parametro weight_decay no Adam implementa "
        "exatamente -log p(theta) para um prior Gaussiano. Para definir priors customizados, "
        "use Pyro ou PyMC (frameworks de probabilistic programming).",
        S))
    story.append(PageBreak())

    # =========================================================
    # SECTION 2 — Prior Choices
    # =========================================================
    story.append(section_header("2. Prior Choices — Gaussian => L2 e Laplace => L1", S))
    story.append(spacer(0.15))

    story.append(P(
        "A escolha do <b>Prior p(theta)</b> determina qual tipo de regularizacao surge na "
        "otimizacao. Dois priors sao especialmente importantes por terem formas fechadas e "
        "interpretacao geometrica clara.",
        "body"))

    story.append(P("Gaussian Prior => L2 Regularization (Ridge / Weight Decay):", "subsection"))
    story.append(formula_block(
        r"p(\theta) = \exp\left(-\|\theta\|^2\right)"
        r"\;\Rightarrow\; -\log p(\theta) = \|\theta\|^2"
        r"\;\Rightarrow\; \min_\theta \text{MSE} + \lambda\|\theta\|^2",
        label="Gaussian: caudas leves, penaliza pesos grandes suavemente — encolhimento uniforme",
        formula_size=12))
    story.append(spacer(0.15))

    story.append(P("Laplace Prior => L1 Regularization (Lasso / Basis Pursuit):", "subsection"))
    story.append(formula_block(
        r"p(\theta) = \exp\left(-\|\theta\|_1\right)"
        r"\;\Rightarrow\; -\log p(\theta) = \|\theta\|_1"
        r"\;\Rightarrow\; \min_\theta \text{MSE} + \lambda\|\theta\|_1",
        label="Laplace: cuspide em zero — acredita fortemente que theta = 0 => esparsidade!",
        formula_size=12))
    story.append(spacer(0.2))

    for item in [
        "<b>Use L2 (Gaussian Prior)</b> quando: acredita que todos os parametros sao relevantes mas devem "
        "ser pequenos. Numericamente estavel, tem solucao analitica (X<super>T</super>X + lambda*I)<super>-1</super>X<super>T</super>y.",
        "<b>Use L1 (Laplace Prior)</b> quando: acredita que muitos parametros sao irrelevantes "
        "(deseja <b>feature selection</b> automatica). Produz solucoes esparsas — muitos theta_j = 0 exatamente.",
        "<b>Diferenca geometrica:</b> Gaussiana tem caudas leves (penaliza pesos grandes suavemente). "
        "Laplace tem caudas pesadas e uma cuspide em zero — a probabilidade maxima esta em zero!",
        "<b>Ambos os priors centrados em zero:</b> expressam a crenca de que parametros menores "
        "sao mais provaveis a priori — equivale a dizer que o modelo nao deve ser desnecessariamente complexo.",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.2))
    fig2 = chart_priors_and_penalties()
    story.append(mat_image(fig2, 16))
    story.append(P(
        "Figura 2 — Esquerda: Prior p(theta). A Gaussiana e suave e simetrica; a Laplace tem um "
        "pico pontiagudo em zero, expressando crenca forte de que theta deveria ser zero. "
        "Direita: -log p(theta) = penalidade de regularizacao. L2 (parabola), L1 (em V com cuspide em 0), "
        "L0 (degrau — NP-hard). A cuspide do L1 em zero e a razao geometrica da esparsidade.",
        "caption"))

    story.append(spacer(0.2))
    fig3 = chart_sparsity()
    story.append(mat_image(fig3, 16))
    story.append(P(
        "Figura 3 — Ridge (L2) vs Lasso (L1) nos mesmos dados com 20 features e apenas 5 informativas. "
        "Ridge encolhe todos os coeficientes mas nenhum chega a zero exatamente (barras verdes). "
        "Lasso zera automaticamente as features irrelevantes (barras cinzas) — fazendo Feature Selection.",
        "caption"))

    story.append(why_box(
        "A dualidade Gaussian/L2 vs Laplace/L1 e o framework unificado de regularizacao. "
        "Conhecendo a conexao Bayesiana, voce pode definir seus proprios priors para "
        "problemas especificos: prior de simetria, prior de suavidade, prior de positividade. "
        "Cada prior se traduz em um regularizador diferente com propriedades especificas.",
        "sklearn: Ridge(alpha=lambda) para L2, Lasso(alpha=lambda, max_iter=10000) para L1. "
        "Lembrete: sempre normalize as features com StandardScaler antes de aplicar Ridge/Lasso "
        "— a regularizacao e sensivel a escala! Sem normalizacao, features com escala maior "
        "sao penalizadas de forma desproporcional.",
        S, color=CRIMSON))
    story.append(PageBreak())

    # =========================================================
    # SECTION 3 — Decision Process
    # =========================================================
    story.append(section_header("3. Decision Process — Quando Usar Qual Regularizacao?", S))
    story.append(spacer(0.15))

    story.append(P(
        "Nao existe um metodo universal. O slide apresenta um <b>processo de decisao</b> "
        "estruturado em 4 cenarios baseados nas caracteristicas do problema. "
        "O ponto de partida e sempre a estrutura dos dados e do problema.",
        "body"))
    story.append(spacer(0.15))

    tbl_decision = data_table(
        ["Cenario", "Recomendacao", "Dicas Praticas"],
        [
            ["Small data + many features\n(N pequeno, p grande)",
             "L2 Ridge ou L1 Lasso\nEarly Stopping\nArquitetura simples",
             "Standardizacao + CV para lambda\nComece com Ridge, tente Lasso para feature selection"],
            ["Structured Outputs\n(imagens, series temporais, grafos)",
             "TV / Laplace\n(Output Regularization)\nVariantes estruturais",
             "TV (Total Variation) preserva bordas\nSmoothing priors para series temporais"],
            ["Strong Prior Knowledge\n(dominio especifico)",
             "Regularization by Design\nConstraints fisicas",
             "Limitar capacidade pela estrutura\nUsars inductive bias do dominio"],
            ["Optimization Unstable\n(gradientes explodem/nan)",
             "Gradient Clipping\nMax-Norm por camada\nNao-negatividade",
             "Gradient clip por norma (max_norm=1.0)\nReduzir Learning Rate"],
        ],
        [CONTENT_W*0.22, CONTENT_W*0.32, CONTENT_W*0.46])
    story.append(tbl_decision)
    story.append(spacer(0.2))

    story.append(P("Hierarquia de decisao recomendada:", "subsection"))
    for item in [
        "<b>Passo 1 — Volume de dados:</b> N < 1000 => sempre regularizar (L2 ou L1). "
        "N > 100k => regularizacao menos critica, foco em arquitetura.",
        "<b>Passo 2 — Numero de features:</b> p >> N => L2 obrigatorio (OLS singular). "
        "p << N com esparsidade esperada => L1 ou Elastic Net.",
        "<b>Passo 3 — Estrutura do output:</b> saidas com estrutura espacial/temporal "
        "=> Output Regularization (TV, Laplacian smoothing).",
        "<b>Passo 4 — Estabilidade do treino:</b> gradientes explodem => Gradient Clipping "
        "e/ou Max-Norm. Pesos crescem sem limite => Weight Decay (L2 implicito).",
        "<b>Passo 5 — Interpretabilidade:</b> precisa saber quais features importam "
        "=> Lasso ou Elastic Net para sparsidade automatica.",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.2))
    story.append(why_box(
        "O Decision Process evita o erro de aplicar a mesma regularizacao em todos os "
        "problemas. Ridge e excelente para regressao com todas as features relevantes. "
        "Lasso e melhor quando se suspeita que a maioria das features e irrelevante. "
        "Output Regularization e frequentemente ignorada mas essencial para problemas "
        "com saidas estruturadas (imagens, sinais, grafos).",
        "Cenario mais comum na pratica (Small data + many features): "
        "StandardScaler() + Ridge(alpha=lambda) com lambda via RidgeCV ou GridSearchCV. "
        "Para Deep Learning com gradientes instáveis: torch.nn.utils.clip_grad_norm_(params, max_norm=1.0). "
        "Para Max-Norm: implementar manualmente com torch.renorm() apos cada passo de otimizacao.",
        S, color=ROSE))
    story.append(PageBreak())

    # =========================================================
    # SECTION 4 — Hyperparameter Choice
    # =========================================================
    story.append(section_header("4. Hyperparameter Choice — Guidelines e Protocolo", S))
    story.append(spacer(0.15))

    story.append(P(
        "Apos escolher o tipo de regularizacao, e preciso encontrar o valor otimo de lambda "
        "<b>sem contaminar o conjunto de teste</b>. O slide apresenta um protocolo estruturado "
        "em 3 eixos: normalizacao, busca e estabilidade.",
        "body"))

    story.append(P("Ranges tipicos para cada hiperparametro:", "subsection"))
    tbl_hp = data_table(
        ["Hiperparametro", "Faixa Tipica", "Observacao"],
        [
            ["L2 (Ridge / Weight Decay)", "1e-6 ... 1e-2",
             "Sempre Log-Grid; valores menores para N grande"],
            ["L1 (Lasso)", "1e-6 ... 1e-3",
             "Lasso mais sensivel — faixa estreita"],
            ["Max-Norm (por camada)", "1 ... 3",
             "Limita a norma maxima dos pesos por camada de rede neural"],
            ["Dropout rate", "0.1 ... 0.5",
             "0.5 para MLP denso, 0.1-0.2 para conv. e transformer"],
            ["Patience (Early Stopping)", "5 ... 10 epocas",
             "Monitorar val loss; restaurar melhor checkpoint"],
        ],
        [CONTENT_W*0.28, CONTENT_W*0.22, CONTENT_W*0.50])
    story.append(tbl_hp)
    story.append(spacer(0.2))

    story.append(P("Por que Log-Grid e nao Linear-Grid?", "subsection"))
    story.append(formula_block(
        r"\lambda \in \{10^{-6},\;10^{-5},\;\ldots,\;10^{-1},\;10^0,\;10^1,\;\ldots,\;10^2\}",
        label="np.logspace(-6, 2, 30)  — lambda age multiplicativamente, nao aditivamente",
        formula_size=12))
    story.append(spacer(0.15))

    story.append(P("Protocolo completo de escolha de hiperparametros (6 passos):", "subsection"))
    for item in [
        "<b>Passo 1 — Split limpo:</b> Train / Val / Test sem contaminacao. "
        "Estratificacao para classificacao. Nunca olhar o Test antes do fim.",
        "<b>Passo 2 — Coarse Sweep:</b> cada regularizador com lambda em Log-Grid amplo "
        "(np.logspace(-6, 2, 30)). Poucas epocas para eficiencia.",
        "<b>Passo 3 — Refinamento:</b> apenas nas top 2-3 regioes promissoras. "
        "Log-Grid estreito ao redor do melhor lambda coarse.",
        "<b>Passo 4 — Multi-seed:</b> media sobre 3-5 seeds se N for pequeno (N < 500). "
        "Reduz variabilidade da estimativa de performance.",
        "<b>Passo 5 — Modelo final:</b> retreinar em Train+Val com o melhor lambda. "
        "Mais dados de treino = melhor estimativa dos parametros.",
        "<b>Passo 6 — Teste UMA VEZ:</b> reportar no conjunto de teste apenas uma vez. "
        "Nunca usar o Test para tomar decisoes — isso vazaria informacao.",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.2))
    fig4 = chart_coarse_to_fine()
    story.append(mat_image(fig4, 16))
    story.append(P(
        "Figura 4 — Coarse-to-Fine Log-Grid Search. Fase 1 (esquerda): sweep amplo de 10^-6 a 10^2 "
        "identifica a regiao otima (zona amarela). Fase 2 (direita): sweep fino ao redor do "
        "melhor lambda coarse refina a estimativa com custo computacional minimo. "
        "A escala logaritmica e essencial para ver o detalhe onde importa.",
        "caption"))

    story.append(spacer(0.2))
    fig5 = chart_multi_seed()
    story.append(mat_image(fig5, 16))
    story.append(P(
        "Figura 5 — Multi-seed averaging com N=40 (dataset pequeno). "
        "Linhas cinzas: val-MSE de cada seed individual — alta variabilidade. "
        "Linha escura: media dos 5 seeds com banda de +/- 1 std. "
        "A media e muito mais estavel e confiavel para escolher lambda em N pequeno.",
        "caption"))

    story.append(why_box(
        "O protocolo de 6 passos previne os dois erros mais comuns em ML: "
        "(1) data leakage — contaminar o Test durante a escolha de hiperparametros; "
        "(2) lambda tuning em escala linear — desperdicando a maioria dos pontos avaliados. "
        "Seguir o protocolo e a diferenca entre uma avaliacao credivel e uma ilusao estatistica.",
        "sklearn: RidgeCV(alphas=np.logspace(-6,2,30), cv=5).fit(X_tr,y_tr) — implementacao "
        "eficiente que faz CV internamente. LassoCV idem para L1. "
        "Para Deep Learning: ModelCheckpoint + EarlyStopping no PyTorch Lightning. "
        "Para multi-seed: usar cross_val_score com random_state diferente em cada call.",
        S, color=SLATE))
    story.append(PageBreak())

    # =========================================================
    # SECTION 5 — Diagnosis
    # =========================================================
    story.append(section_header("5. Diagnosis — Under vs. Over-Regularization", S))
    story.append(spacer(0.15))

    story.append(P(
        "Apos escolher lambda, e essencial <b>diagnosticar</b> se a regularizacao esta "
        "corretamente calibrada. O instrumento padrao e a <b>Learning Curve</b>: "
        "plotar o erro de treino e validacao em funcao do tamanho do dataset N.",
        "body"))

    story.append(formula_block(
        r"\text{gap}(N) = \text{Val-MSE}(N) - \text{Train-MSE}(N)",
        label="gap grande => High Variance (Under-Reg.) | gap pequeno mas alto => High Bias (Over-Reg.)",
        formula_size=14))
    story.append(spacer(0.15))

    tbl_diag = data_table(
        ["Padrao da Learning Curve", "Diagnostico", "Acao Recomendada"],
        [
            ["Erro treino BAIXO, erro val. ALTO, GAP GRANDE",
             "Under-Regularizacao — High Variance / Overfitting",
             "Aumentar lambda; Early Stopping mais agressivo; mais dados"],
            ["Erro treino ALTO, erro val. ALTO, GAP PEQUENO",
             "Over-Regularizacao — High Bias / Underfitting",
             "Diminuir lambda; modelo mais complexo; mais features"],
            ["Erro treino aprox. erro val. aprox. nivel aceitavel",
             "Regularizacao otima — bom equilibrio",
             "Manter lambda; investir em mais dados ou novas features"],
        ],
        [CONTENT_W*0.30, CONTENT_W*0.35, CONTENT_W*0.35])
    story.append(tbl_diag)
    story.append(spacer(0.2))

    fig6 = chart_learning_curves()
    story.append(mat_image(fig6, 16))
    story.append(P(
        "Figura 6 — Learning Curves para tres valores de lambda. "
        "Esquerda (Under-Regularizacao, lambda=1e-6): gap enorme entre treino e validacao. "
        "Centro (lambda=1.0): gap fecha conforme N cresce — bom calibracao. "
        "Direita (Over-Regularizacao, lambda=1e4): ambos os erros altos e proximos — modelo nao aprende nada util.",
        "caption"))

    story.append(why_box(
        "O diagnostico via Learning Curves e o mapa de orientacao para qualquer problema "
        "de regularizacao. Sem ele, qualquer ajuste de lambda e cego. Gap grande indica "
        "High Variance — a solucao e mais regularizacao OU mais dados. "
        "Ambos erros altos indica High Bias — a solucao e menos regularizacao OU modelo mais rico.",
        "sklearn: learning_curve(modelo, X, y, train_sizes=np.linspace(0.1,1,10), cv=5, "
        "scoring='neg_mean_squared_error'). Plotar train vs val MSE vs N. "
        "Para Deep Learning: monitorar train loss e val loss por epoca no tensorboard ou wandb. "
        "O padrao e identico — apenas o eixo x muda de N para epocas.",
        S, color=CRIMSON))
    story.append(PageBreak())

    # =========================================================
    # SECTION 6 — Cross-Validation
    # =========================================================
    story.append(section_header("6. Cross-Validation — Estimativa Confiavel do Erro Real", S))
    story.append(spacer(0.15))

    story.append(P(
        "<b>Cross-Validation</b> e uma familia de tecnicas para estimar como um modelo "
        "performa em dados nao vistos. Pode ser usada para avaliar modelos, selecionar "
        "entre candidatos, ajustar hiperparametros e evitar overfitting ao conjunto de validacao. "
        "<b>Principio:</b> nunca use o conjunto de teste para tomar decisoes.",
        "body"))

    story.append(P("Estimativa K-Fold (formula central):", "subsection"))
    story.append(formula_block(
        r"\hat{\text{Err}}_{K\text{-Fold}} = \frac{1}{K}\sum_{k=1}^{K}\mathcal{L}(f_{-k}, D_k)",
        label="f_{-k}: modelo treinado em todos os folds exceto o k-esimo  |  D_k: fold de validacao k",
        formula_size=13))
    story.append(spacer(0.2))

    tbl_cv = data_table(
        ["Variante", "Mecanismo", "Vantagem", "Quando usar"],
        [
            ["Hold-Out",
             "Split unico (ex: 75/25). Um treino, uma avaliacao.",
             "Muito rapido — custo O(1).",
             "N muito grande (>50k)"],
            ["K-Fold (K=5 ou 10)",
             "K splits rotacionados; media dos K erros.",
             "Balanceia custo e confiabilidade.",
             "Padrao para a maioria dos casos"],
            ["Stratified K-Fold",
             "K-Fold preservando % de classes em cada fold.",
             "Robusto a desbalanceamento de classes.",
             "Classificacao com classes raras"],
            ["Leave-One-Out (LOO)",
             "K=N: cada amostra e validacao uma vez.",
             "Minimo vies — usa N-1 amostras no treino.",
             "N muito pequeno (<50-100)"],
        ],
        [CONTENT_W*0.17, CONTENT_W*0.33, CONTENT_W*0.26, CONTENT_W*0.24])
    story.append(tbl_cv)
    story.append(spacer(0.2))

    fig7 = chart_cv_splits()
    story.append(mat_image_tall(fig7, 16, aspect=0.78))
    story.append(P(
        "Figura 7 — Padroes de divisao: verde = treino, vermelho = validacao. "
        "Hold-Out: um unico split. K-Fold: cada fold e validacao uma vez de forma rotacionada. "
        "Stratified K-Fold: mesma logica, mas preserva a proporcao de classes em cada fold "
        "(essencial para classificacao desbalanceada). LOO: cada amostra e validacao uma vez.",
        "caption"))

    story.append(spacer(0.15))
    fig8 = chart_cv_comparison()
    story.append(mat_image(fig8, 16))
    story.append(P(
        "Figura 8 — Comparacao quantitativa: MSE medio e variabilidade por metodo. "
        "Hold-Out tem a maior variabilidade entre seeds. "
        "K-Fold (K=5 e K=10) oferecem estimativas mais estaveis. "
        "LOO e deterministico mas muito lento para N grande. "
        "O std indicado em cada barra e a variabilidade da estimativa.",
        "caption"))

    story.append(why_box(
        "A variabilidade do Hold-Out (depende fortemente do split aleatorio) e o motivo "
        "pelo qual K-Fold e o padrao. Com K=5 e shuffle=True, a estimativa e robusta e "
        "computacionalmente razoavel. Stratified K-Fold e obrigatorio em qualquer problema "
        "de classificacao com desbalanceamento — Hold-Out padrao pode ter 0% de positivos "
        "em algum fold, tornando a estimativa completamente errada.",
        "sklearn: KFold(n_splits=5, shuffle=True, random_state=42) + "
        "cross_val_score(modelo, X, y, cv=kf, scoring='neg_mean_squared_error'). "
        "Para classificacao: StratifiedKFold. Para series temporais: TimeSeriesSplit "
        "(nunca use KFold com dados temporais — vazaria o futuro para o treino!)",
        S, color=CRIMSON))
    story.append(PageBreak())

    # =========================================================
    # SECTION 7 — Bootstrapping
    # =========================================================
    story.append(section_header("7. Bootstrapping — Resampling, Incerteza e Bagging", S))
    story.append(spacer(0.15))

    story.append(P(
        "<b>Bootstrapping</b> e uma tecnica de <b>resampling</b> para quantificar a "
        "incerteza de um estimador sem precisar de formulas analiticas. "
        "<b>Ideia central:</b> o dataset observado D e tratado como uma <b>populacao empirica</b>. "
        "Reamostrar D <b>com reposicao</b> simula o processo de coletar novos datasets "
        "da populacao real. A variabilidade entre reamostras aproxima a incerteza do estimador.",
        "body"))

    story.append(P("Algoritmo Nonparametric Bootstrap:", "subsection"))
    for item in [
        "<b>Input:</b> dados D = {z_1, ..., z_N}, estatistica S(D), replicatas B (tipicamente 500-2000).",
        "<b>Para b = 1, ..., B:</b> amostrar D<super>b</super> com reposicao (N amostras); calcular s<super>b</super> = S(D<super>b</super>).",
        "<b>Output:</b> distribuicao {s<super>1</super>, ..., s<super>B</super>} — aproxima a distribuicao amostral de S.",
        "<b>Erro padrao estimado:</b> std({s<super>b</super>}). "
        "<b>IC 95%:</b> percentis 2.5% e 97.5% da distribuicao bootstrap.",
        "<b>Importante:</b> o mesmo pipeline completo deve rodar dentro de S — preprocessing, "
        "feature selection, tuning — para evitar vazamento de informacao.",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.15))
    story.append(P("Tres tipos de Bootstrap Confidence Intervals (95%):", "subsection"))
    tbl_ci = data_table(
        ["Tipo", "Formula / Logica", "Quando usar"],
        [
            ["Normal approximation",
             "s_hat +/- 1.96 * std({s^b})",
             "Distribuicao bootstrap aproximadamente Normal"],
            ["Percentile interval",
             "[q_2.5%({s^b}),  q_97.5%({s^b})]",
             "Distribuicao assimetrica — mais robusto"],
            ["BCa (Bias-Corrected Accelerated)",
             "Corrige vies e aceleracao — mais complexo",
             "Distribuicao enviesada — mais preciso"],
        ],
        [CONTENT_W*0.25, CONTENT_W*0.42, CONTENT_W*0.33])
    story.append(tbl_ci)
    story.append(spacer(0.2))

    fig9 = chart_bootstrap_coefs()
    story.append(mat_image(fig9, 16))
    story.append(P(
        "Figura 9 — Distribuicao bootstrap dos coeficientes (B=500 replicatas). "
        "Histograma: distribuicao de cada coeficiente sobre as replicatas. "
        "Linha escura: estimativa original no dataset completo. "
        "Linhas amarelas: IC 95% percentile. "
        "Features marcadas com (*) tem IC que nao inclui zero — provavelmente significativas.",
        "caption"))

    story.append(spacer(0.2))
    story.append(P("Bagging — Bootstrap Aggregating:", "subsection"))
    story.append(P(
        "<b>Bagging</b> e o uso de Bootstrap para reduzir variancia: treinar o mesmo modelo "
        "em B amostras bootstrap diferentes e calcular a media das previsoes. "
        "E a base do <b>Random Forest</b> (Bagging de arvores de decisao). "
        "A media de B estimadores instáveis produz um estimador muito mais estavel.",
        "body"))

    story.append(formula_block(
        r"\hat{f}_{\text{bag}}(x) = \frac{1}{B}\sum_{b=1}^{B}\hat{f}^{(b)}(x)",
        label="Cada f^(b) e treinado em uma amostra bootstrap D^b — alta variancia individual, baixa na media",
        formula_size=14))
    story.append(spacer(0.15))

    fig10 = chart_bagging()
    story.append(mat_image(fig10, 16))
    story.append(P(
        "Figura 10 — Bagging com polinomio de grau 7. "
        "Esquerda: modelos individuais (B=200 linhas finas vermelhas) — alta variancia, "
        "muita oscilacao. Direita: media das B previsoes bootstrap (linha grossa escura) "
        "e muito mais proxima de sin(x) verdadeiro, com banda de incerteza coerente. "
        "A reducao de MSE em relacao ao modelo unico demonstra o poder do Bagging.",
        "caption"))

    story.append(why_box(
        "Bootstrapping e a base teorica do Random Forest (Bagging de arvores). "
        "Alem de reduzir variancia, o Bootstrap tambem fornece as amostras Out-of-Bag (OOB) "
        "— as amostras nao incluidas em cada replicata — que podem ser usadas como um "
        "conjunto de validacao gratuito, sem precisar de CV separado.",
        "sklearn: BaggingRegressor(Ridge(), n_estimators=200, bootstrap=True). "
        "Random Forest ja implementa Bagging + seleção aleatoria de features. "
        "Para IC de coeficientes: implementar Bootstrap manualmente com loop "
        "'for b in range(B): idx = np.random.choice(N, N, replace=True)'. "
        "Percentile IC: np.percentile(coefs_boot, [2.5, 97.5], axis=0).",
        S, color=CRIMSON))
    story.append(PageBreak())

    # =========================================================
    # SECTION 8 — Resumo
    # =========================================================
    story.append(section_header("8. Resumo Final — Mapa Conceitual da Aula 4", S))
    story.append(spacer(0.15))

    tbl_resumo = data_table(
        ["Conceito", "Ideia Central", "Formula / Ferramenta Chave"],
        [
            ["Probabilistic View",
             "MSE emerge do ruido Gaussiano; regularizacao e Prior Bayesiano",
             "-log p(D|theta) = MSE | -log p(theta) = Regularizacao"],
            ["MAP Estimation",
             "Minimizar Loss + Regularizador = Maximizar Posterior",
             "theta_MAP = argmin [MSE + lambda * R(theta)]"],
            ["Gaussian Prior => L2",
             "Prior Gaussiano => penalidade ||theta||^2 => Ridge",
             "Ridge(alpha=lambda).fit(X, y)"],
            ["Laplace Prior => L1",
             "Prior Laplace => penalidade ||theta||_1 => Lasso (esparsidade)",
             "Lasso(alpha=lambda, max_iter=10000).fit(X, y)"],
            ["Decision Process",
             "4 cenarios: Small data, Structured Output, Prior Known, Unstable Opt.",
             "Hierarquia: volume -> features -> output -> estabilidade"],
            ["Log-Grid Search",
             "Lambda age multiplicativamente => busca em Log-Grid, nao linear",
             "np.logspace(-6, 2, 30) + Coarse-to-Fine"],
            ["Diagnosis",
             "Learning Curves diagnosticam High Variance vs High Bias",
             "gap grande => Under-Reg. | ambos altos => Over-Reg."],
            ["K-Fold CV",
             "Estimativa confiavel do erro real sem contaminar o Test",
             "KFold(5, shuffle=True) + cross_val_score(...)"],
            ["Bootstrap",
             "Resampling com reposicao => distribuicao de qualquer estatistica",
             "IC Percentile: np.percentile(s_boot, [2.5, 97.5])"],
            ["Bagging",
             "Media de B modelos bootstrap => reduz variancia drasticamente",
             "BaggingRegressor(base, n_estimators=200, bootstrap=True)"],
        ],
        [CONTENT_W*0.22, CONTENT_W*0.40, CONTENT_W*0.38])
    story.append(tbl_resumo)
    story.append(spacer(0.3))

    story.append(P("<b>Checklist para qualquer novo projeto:</b>", "subsection"))
    for item in [
        "Escolher loss function baseada na distribuicao do ruido (Gaussiano => MSE, Bernoulli => cross-entropy).",
        "Definir prior implicito: L2 para todos relevantes, L1 para feature selection, sem prior para MLE.",
        "Usar Log-Grid para busca de lambda — NUNCA escala linear.",
        "Aplicar Coarse-to-Fine para eficiencia computacional.",
        "Diagnosticar via Learning Curves antes de aceitar qualquer lambda.",
        "Validar com K-Fold (K=5 ou 10), Stratified para classificacao desbalanceada.",
        "Usar Bootstrap para quantificar incerteza de coeficientes ou metricas em N pequeno.",
        "Retreinar em Train+Val com o melhor lambda. Reportar no Test UMA UNICA VEZ.",
    ]:
        story.append(P(f"<bullet>-</bullet> {item}", "bullet"))

    story.append(spacer(0.3))
    story.append(divider(GOLD))
    story.append(spacer(0.15))
    story.append(P(
        "Na proxima aula, exploraremos <b>metodos baseados em arvores de decisao</b> — "
        "Random Forests (Bagging de arvores visto nesta aula) e Gradient Boosting "
        "(Boosting como forma de reducao de Bias). Os conceitos de Probabilistic View, "
        "Cross-Validation e Bootstrap desta aula sao identicos para avaliar e tunar esses modelos.",
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
        title="Aula 4 — Probabilistic View & Bayesian Regularization",
        author="Prof. Johannes Schwab, PhD — FH Kufstein Tirol",
        subject="Statistisches Lernen 2")

    styles = build_styles()
    # PageBreak after NextPageTemplate keeps the cover page empty (only canvas art);
    # all story content starts on page 2 with the Content template.
    story  = [NextPageTemplate("Content"), PageBreak()] + build_content(styles)
    doc.build(story)

    try:
        import PyPDF2
        with open(output_path, "rb") as f:
            total_pages[0] = len(PyPDF2.PdfReader(f).pages)
        doc2 = BaseDocTemplate(
            output_path, pagesize=A4,
            pageTemplates=[cover_tpl, content_tpl],
            leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0,
            title="Aula 4 — Probabilistic View & Bayesian Regularization",
            author="Prof. Johannes Schwab, PhD — FH Kufstein Tirol",
            subject="Statistisches Lernen 2")
        story2 = [NextPageTemplate("Content"), PageBreak()] + build_content(styles)
        doc2.build(story2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    out  = os.path.join(base, "L4_Probabilistic_View_Bayesian_Regularization.pdf")
    print(f"Gerando PDF: {out}")
    build_pdf(out)
    print("PDF gerado com sucesso!")
