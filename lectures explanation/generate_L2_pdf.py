"""
generate_L2_pdf.py
------------------
Generates a magazine-style lecture notes PDF for Lecture 2:
Bias-Variance Trade-Off & Model Selection
(Statistisches Lernen 2 — FH Kufstein Tirol)

Output language: PT-BR  |  Technical terms: English
Dependencies: reportlab, matplotlib, numpy, scipy, sklearn
"""

import io
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.datasets import make_regression
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
# PALETTE  (L2 uses warmer purple/amber tones to distinguish from L1)
# ---------------------------------------------------------------------------
DEEP_PURPLE = colors.HexColor("#2D1B6E")
VIOLET      = colors.HexColor("#6C3FC5")
AMBER       = colors.HexColor("#F5A623")
CORAL       = colors.HexColor("#E8534A")
TEAL        = colors.HexColor("#0D7377")
MINT        = colors.HexColor("#00C9A7")
LIGHT_BG    = colors.HexColor("#F5F3FF")
SIDEBAR_BG  = colors.HexColor("#EAE6F8")
WHITE       = colors.white
BLACK       = colors.black

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
            r = int(45 + t * (20 - 45))
            g = int(27 + t * (10 - 27))
            b = int(110 + t * (80 - 110))
            canvas.setFillColorRGB(r/255, g/255, b/255)
            canvas.rect(0, h * i / steps, w, h / steps + 1, fill=1, stroke=0)

        # Decorative circles
        for cx, cy, cr, alpha in [
            (w * 0.82, h * 0.72, 3.8*cm, 0.12),
            (w * 0.08, h * 0.22, 5.5*cm, 0.09),
            (w * 0.65, h * 0.38, 2.2*cm, 0.18),
        ]:
            canvas.setFillColor(colors.HexColor("#FFFFFF"))
            canvas.setFillAlpha(alpha)
            canvas.circle(cx, cy, cr, fill=1, stroke=0)
        canvas.setFillAlpha(1.0)

        # Amber accent bar
        canvas.setFillColor(AMBER)
        canvas.rect(0, h * 0.55, SIDEBAR_W * 1.5, h * 0.45, fill=1, stroke=0)

        # Course tag
        canvas.setFillColor(TEAL)
        canvas.roundRect(MARGIN, h * 0.78, 7*cm, 0.9*cm, 4, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(MARGIN + 0.3*cm, h * 0.78 + 0.25*cm,
                          "STATISTISCHES LERNEN 2")

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 28)
        canvas.drawString(MARGIN, h * 0.62, "Aula 2")
        canvas.setFont("Helvetica", 18)
        canvas.drawString(MARGIN, h * 0.56, "Bias-Variance Trade-Off")
        canvas.drawString(MARGIN, h * 0.51, "& Model Selection")

        canvas.setFillColor(AMBER)
        canvas.rect(MARGIN, h * 0.49, 8*cm, 0.06*cm, fill=1, stroke=0)

        canvas.setFillColor(colors.HexColor("#CCBBEE"))
        canvas.setFont("Helvetica", 10)
        canvas.drawString(MARGIN, h * 0.45, "Prof. Johannes Schwab, PhD")
        canvas.drawString(MARGIN, h * 0.42, "FH Kufstein Tirol")

        canvas.setFillColor(AMBER)
        canvas.rect(0, 0, w, 0.4*cm, fill=1, stroke=0)
        canvas.setFillColor(VIOLET)
        canvas.rect(0, 0.4*cm, w, 0.15*cm, fill=1, stroke=0)
        canvas.restoreState()

    def draw_page(self, canvas, doc):
        canvas.saveState()
        w, h = PAGE_W, PAGE_H
        pn = doc.page

        canvas.setFillColor(SIDEBAR_BG)
        canvas.rect(0, FOOTER_H, SIDEBAR_W, h - HEADER_H - FOOTER_H,
                    fill=1, stroke=0)
        canvas.setFillColor(VIOLET)
        canvas.rect(SIDEBAR_W - 0.15*cm, FOOTER_H,
                    0.15*cm, h - HEADER_H - FOOTER_H, fill=1, stroke=0)

        canvas.setFillColor(DEEP_PURPLE)
        canvas.rect(0, h - HEADER_H, w, HEADER_H, fill=1, stroke=0)
        canvas.setFillColor(AMBER)
        canvas.rect(0, h - HEADER_H - 0.12*cm, w, 0.12*cm, fill=1, stroke=0)

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(SIDEBAR_W + 0.4*cm, h - HEADER_H + 0.55*cm,
                          "Statistisches Lernen 2  |  Aula 2 — Bias-Variance Trade-Off & Model Selection")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 0.4*cm, h - HEADER_H + 0.55*cm, "FH Kufstein Tirol")

        canvas.setFillColor(LIGHT_BG)
        canvas.rect(0, 0, w, FOOTER_H, fill=1, stroke=0)
        canvas.setFillColor(DEEP_PURPLE)
        canvas.rect(0, FOOTER_H - 0.08*cm, w, 0.08*cm, fill=1, stroke=0)

        canvas.setFillColor(DEEP_PURPLE)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(SIDEBAR_W + 0.4*cm, 0.38*cm,
                          "Prof. Johannes Schwab, PhD  —  FH Kufstein Tirol")
        total = self._total[0] if self._total[0] else "?"
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawRightString(w - 0.4*cm, 0.38*cm, f"Pagina {pn} de {total}")

        canvas.setFillColor(VIOLET)
        canvas.saveState()
        canvas.translate(SIDEBAR_W * 0.5, h * 0.5)
        canvas.rotate(90)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(DEEP_PURPLE)
        canvas.drawCentredString(0, 0, "BIAS-VARIANCE TRADE-OFF  |  MODEL SELECTION")
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
                    textColor=DEEP_PURPLE, spaceAfter=6, spaceBefore=4),
        "section": ps("section", fontName="Helvetica-Bold", fontSize=15,
                      textColor=VIOLET, spaceBefore=14, spaceAfter=4),
        "subsection": ps("subsection", fontName="Helvetica-Bold", fontSize=11,
                         textColor=DEEP_PURPLE, spaceBefore=8, spaceAfter=3),
        "body": ps("body", fontName="Helvetica", fontSize=10,
                   textColor=colors.HexColor("#2B2B2B"), leading=15,
                   spaceBefore=3, spaceAfter=3, alignment=TA_JUSTIFY),
        "bullet": ps("bullet", fontName="Helvetica", fontSize=9.5,
                     textColor=colors.HexColor("#2B2B2B"), leading=14,
                     spaceBefore=1, spaceAfter=1, leftIndent=14, bulletIndent=4),
        "caption": ps("caption", fontName="Helvetica-Oblique", fontSize=8,
                      textColor=colors.HexColor("#555555"),
                      spaceBefore=2, spaceAfter=6, alignment=TA_CENTER),
        "highlight": ps("highlight", fontName="Helvetica-Bold", fontSize=9.5,
                        textColor=DEEP_PURPLE, spaceBefore=4, spaceAfter=4,
                        leftIndent=10),
        "code": ps("code", fontName="Courier", fontSize=8.5,
                   textColor=colors.HexColor("#1E1E1E"),
                   backColor=colors.HexColor("#F0EEF8"),
                   spaceBefore=4, spaceAfter=4,
                   leftIndent=8, rightIndent=8, leading=13),
        "quote": ps("quote", fontName="Helvetica-Oblique", fontSize=10,
                    textColor=colors.HexColor("#444444"), leading=15,
                    leftIndent=20, rightIndent=20, spaceBefore=6, spaceAfter=6,
                    alignment=TA_JUSTIFY),
    }


# ---------------------------------------------------------------------------
# HELPER BUILDERS
# ---------------------------------------------------------------------------
def section_header(title, styles):
    data = [[Paragraph(title, styles["section"])]]
    tbl = Table(data, colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), LIGHT_BG),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEBELOW",     (0,0), (-1,-1), 2, VIOLET),
        ("LINEABOVE",     (0,0), (-1,0),  0.5, VIOLET),
    ]))
    return tbl


def why_box(why_text, apply_text, styles, color=VIOLET):
    header_s = ParagraphStyle("wh", fontName="Helvetica-Bold", fontSize=9.5,
                               textColor=WHITE, leading=13)
    body_s   = ParagraphStyle("wb", fontName="Helvetica", fontSize=9,
                               textColor=colors.HexColor("#111111"),
                               leading=13, alignment=TA_JUSTIFY)
    data = [
        [Paragraph("Por que isso é importante para Data Science?", header_s)],
        [Paragraph(why_text, body_s)],
        [Paragraph("Aplicacao Pratica em Machine Learning", header_s)],
        [Paragraph(apply_text, body_s)],
    ]
    tbl = Table(data, colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), color),
        ("BACKGROUND",    (0,1), (-1,1), colors.HexColor("#EEF0FF")),
        ("BACKGROUND",    (0,2), (-1,2), DEEP_PURPLE),
        ("BACKGROUND",    (0,3), (-1,3), colors.HexColor("#EAE6F8")),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return tbl


def formula_block(latex_str, label="", bg_hex="#2D1B6E", formula_size=15):
    w_in = CONTENT_W / 72
    has_label = bool(label.strip())
    h_in = 1.05 if not has_label else 1.55

    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.patch.set_facecolor(bg_hex)
    ax.set_facecolor(bg_hex)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([0,1], [0.985, 0.985], color="#F5A623", linewidth=3.5,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0,1], [0.015, 0.015], color="#F5A623", linewidth=3.5,
            transform=ax.transAxes, clip_on=False)

    y_f = 0.63 if has_label else 0.50
    ax.text(0.5, y_f, f"${latex_str}$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=formula_size, color="white",
            math_fontfamily="dejavusans")

    if has_label:
        ax.text(0.5, 0.20, label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#CCBBFF", fontstyle="italic")

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
                              textColor=DEEP_PURPLE, alignment=TA_LEFT)
    data = [[Paragraph(h, h_style) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), c_style) for c in row])
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0), DEEP_PURPLE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
            [LIGHT_BG, colors.HexColor("#DDD8F0")]),
        ("GRID",           (0,0), (-1,-1), 0.4, colors.HexColor("#AAAACC")),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
        ("RIGHTPADDING",   (0,0), (-1,-1), 8),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
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


def divider(color=VIOLET, thickness=0.8):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=4, spaceBefore=4)


def spacer(h_cm=0.3):
    return Spacer(1, h_cm*cm)


# ---------------------------------------------------------------------------
# CHART GENERATORS
# ---------------------------------------------------------------------------
f_true_global = lambda x: np.sin(1.5 * np.pi * x)


def chart_core_dilemma():
    """Underfitting / good fit / overfitting — 3 panels."""
    N = 22
    x_tr = np.sort(np.random.uniform(0, 2, N))
    y_tr = f_true_global(x_tr) + np.random.normal(0, 0.3, N)
    x_p  = np.linspace(0, 2, 400)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")
    configs = [
        (1,  "#E8534A", "Underfitting  (High Bias)\nGrau 1"),
        (5,  "#00C9A7", "Bom Equilibrio\nGrau 5"),
        (18, "#6C3FC5", "Overfitting  (High Variance)\nGrau 18"),
    ]
    for ax, (grau, cor, titulo) in zip(axes, configs):
        ax.set_facecolor("#FAFAFA")
        modelo = make_pipeline(PolynomialFeatures(grau), LinearRegression())
        modelo.fit(x_tr.reshape(-1,1), y_tr)
        y_plot = modelo.predict(x_p.reshape(-1,1))
        mse_tr = mean_squared_error(y_tr, modelo.predict(x_tr.reshape(-1,1)))
        ax.scatter(x_tr, y_tr, s=45, color="#888", zorder=5, alpha=0.85)
        ax.plot(x_p, f_true_global(x_p), "k--", alpha=0.4, lw=1.5,
                label="f verdadeira")
        ax.plot(x_p, y_plot, color=cor, lw=2.5,
                label=f"Grau {grau}")
        ax.set_title(f"{titulo}\nMSE treino = {mse_tr:.3f}", fontsize=9.5,
                     fontweight="bold")
        ax.set_ylim(-2.5, 2.5); ax.grid(alpha=0.25)
        ax.set_xlabel("x", fontsize=9)
    axes[0].legend(fontsize=8)
    plt.suptitle("The Core Dilemma: Data Fit vs. Generalization",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_bv_decomp_variability():
    """Monte-Carlo visualization of prediction variability."""
    def simular(grau, n_sim=150, N=25, sigma=0.4):
        x_t = np.linspace(0, 2, 50)
        preds = np.zeros((n_sim, len(x_t)))
        for i in range(n_sim):
            x_r = np.sort(np.random.uniform(0, 2, N))
            y_r = f_true_global(x_r) + np.random.normal(0, sigma, N)
            m = make_pipeline(PolynomialFeatures(grau), Ridge(alpha=1e-6))
            m.fit(x_r.reshape(-1,1), y_r)
            preds[i] = m.predict(x_t.reshape(-1,1))
        f_t = f_true_global(x_t)
        media = preds.mean(axis=0)
        b2 = np.mean((f_t - media)**2)
        var = np.mean(preds.var(axis=0))
        return b2, var, sigma**2, preds, x_t, f_t, media

    graus = [1, 4, 12]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")
    colors_g = ["#E8534A", "#00C9A7", "#6C3FC5"]
    for ax, grau, col in zip(axes, graus, colors_g):
        b2, var, irr, preds, x_t, f_t, media = simular(grau)
        ax.set_facecolor("#FAFAFA")
        for i in range(0, 150, 8):
            ax.plot(x_t, preds[i], alpha=0.09, color=col, lw=1)
        ax.plot(x_t, media, color=col, lw=2.5, label="Media prevista")
        ax.plot(x_t, f_t, "k--", lw=2, alpha=0.6, label="f verdadeira")
        ax.set_title(f"Grau {grau}\nBias²={b2:.3f} | Var={var:.3f} | Total={b2+var+irr:.3f}",
                     fontsize=9.5, fontweight="bold")
        ax.set_ylim(-3, 3); ax.grid(alpha=0.25); ax.set_xlabel("x", fontsize=9)
        if grau == 1:
            ax.legend(fontsize=8)
    plt.suptitle("Decomposicao Empirica: variabilidade das previsoes entre 150 datasets",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_bv_curve():
    """Classic Bias-Variance trade-off U-shaped curve."""
    def simular_fast(grau, n_sim=120, N=25, sigma=0.4):
        x_t = np.linspace(0, 2, 40)
        f_t = f_true_global(x_t)
        preds = np.zeros((n_sim, len(x_t)))
        for i in range(n_sim):
            x_r = np.sort(np.random.uniform(0, 2, N))
            y_r = f_true_global(x_r) + np.random.normal(0, sigma, N)
            m = make_pipeline(PolynomialFeatures(grau), Ridge(alpha=1e-6))
            m.fit(x_r.reshape(-1,1), y_r)
            preds[i] = m.predict(x_t.reshape(-1,1))
        media = preds.mean(axis=0)
        b2  = np.mean((f_t - media)**2)
        var = np.mean(preds.var(axis=0))
        return b2, var, sigma**2

    graus = list(range(1, 15))
    b2s, vars_, tots = [], [], []
    for g in graus:
        b2, v, irr = simular_fast(g)
        b2s.append(b2); vars_.append(v); tots.append(b2 + v + irr)

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    ax.plot(graus, b2s,  color="#E8534A", lw=2.5, marker="o",  ms=6, label="Bias²")
    ax.plot(graus, vars_, color="#6C3FC5", lw=2.5, marker="s",  ms=6, label="Variance")
    ax.plot(graus, tots,  color="#00C9A7", lw=3,   marker="^",  ms=7,
            label="Total Error")
    ax.axhline(0.4**2, color="#888888", ls="--", lw=1.8,
               label="Irreducible Error (sigma² = 0.16)")

    idx_min = int(np.argmin(tots))
    ax.axvline(graus[idx_min], color="#00C9A7", ls=":", lw=2, alpha=0.8)
    ax.annotate(f"Complexidade\notima: grau {graus[idx_min]}",
                xy=(graus[idx_min], tots[idx_min]),
                xytext=(graus[idx_min]+1.5, tots[idx_min]+0.04),
                arrowprops=dict(arrowstyle="->", color="#333"),
                fontsize=9.5)
    ax.text(1.5,  0.36, "High Bias\n(Underfitting)", color="#E8534A",
            fontsize=10.5, fontweight="bold")
    ax.text(10.5, 0.36, "High Variance\n(Overfitting)", color="#6C3FC5",
            fontsize=10.5, fontweight="bold")

    ax.set_xlabel("Complexidade do Modelo (Grau Polinomial)", fontsize=11)
    ax.set_ylabel("Erro", fontsize=11)
    ax.set_title("Bias-Variance Trade-Off — A Curva Classica", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9.5); ax.set_ylim(0, 0.65)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def chart_learning_curves():
    """Learning curves showing high-bias vs high-variance regimes."""
    X_full = np.sort(np.random.uniform(0, 2, 200)).reshape(-1,1)
    y_full = f_true_global(X_full.ravel()) + np.random.normal(0, 0.4, 200)
    tamanhos = np.arange(12, 201, 12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
    for ax, (grau, titulo, diagn, col) in zip(axes, [
        (1, "Grau 1 — High Bias (Underfitting)",
         "Treino alto e Val alto — ambos convergem para erro alto", "#E8534A"),
        (10, "Grau 10 — High Variance (Overfitting)",
         "Gap grande entre Treino e Validacao", "#6C3FC5"),
    ]):
        ax.set_facecolor("#FAFAFA")
        errs_tr, errs_val = [], []
        for N in tamanhos:
            idx = np.random.choice(200, N, replace=False)
            Xn, yn = X_full[idx], y_full[idx]
            m = make_pipeline(PolynomialFeatures(grau), Ridge(alpha=1e-4))
            sc = cross_val_score(m, Xn, yn, cv=KFold(5, shuffle=True, random_state=7),
                                 scoring="neg_mean_squared_error")
            m.fit(Xn, yn)
            errs_tr.append(mean_squared_error(yn, m.predict(Xn)))
            errs_val.append(-sc.mean())

        ax.plot(tamanhos, errs_tr, "#00C9A7", lw=2, label="Erro Treino")
        ax.plot(tamanhos, errs_val, col, lw=2, label="Erro Validacao (CV)")
        ax.fill_between(tamanhos, errs_tr, errs_val, alpha=0.12, color=col)
        ax.set_title(f"{titulo}\n{diagn}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("N (tamanho do treino)"); ax.set_ylabel("MSE")
        ax.legend(fontsize=9); ax.set_ylim(0, 0.65); ax.grid(alpha=0.25)

    plt.suptitle("Learning Curves — Diagnostico de Bias vs. Variance",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_cv_selection():
    """Cross-validation selects optimal polynomial degree."""
    X_cv = np.sort(np.random.uniform(0, 2, 80)).reshape(-1,1)
    y_cv = f_true_global(X_cv.ravel()) + np.random.normal(0, 0.4, 80)
    graus = range(1, 14)
    medios, stds = [], []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for g in graus:
        sc = cross_val_score(
            make_pipeline(PolynomialFeatures(g), Ridge(alpha=1e-4)),
            X_cv, y_cv, cv=kf, scoring="neg_mean_squared_error")
        medios.append(-sc.mean()); stds.append(sc.std())

    medios = np.array(medios); stds = np.array(stds)
    melhor = list(graus)[int(np.argmin(medios))]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    ax.plot(graus, medios, "#6C3FC5", lw=2.5, marker="o", ms=7,
            label="MSE medio (10-fold CV)")
    ax.fill_between(graus, medios - stds, medios + stds,
                    alpha=0.18, color="#6C3FC5", label="+/- 1 desvio padrao")
    ax.axvline(melhor, color="#00C9A7", ls="--", lw=2,
               label=f"Melhor grau: {melhor}")
    ax.set_xlabel("Grau Polinomial"); ax.set_ylabel("MSE (Cross-Validation)")
    ax.set_title(f"10-Fold Cross-Validation para Selecao de Complexidade\n"
                 f"Melhor grau = {melhor}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def chart_regularization():
    """Ridge lambda effect on model fit."""
    X_cv = np.sort(np.random.uniform(0, 2, 80)).reshape(-1,1)
    y_cv = f_true_global(X_cv.ravel()) + np.random.normal(0, 0.4, 80)
    x_p  = np.linspace(0, 2, 400).reshape(-1,1)
    grau = 10
    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="white")
    axes = axes.ravel()
    for ax, lam in zip(axes, lambdas):
        ax.set_facecolor("#FAFAFA")
        m = make_pipeline(PolynomialFeatures(grau), Ridge(alpha=lam))
        m.fit(X_cv, y_cv)
        sc = cross_val_score(m, X_cv, y_cv, cv=5,
                             scoring="neg_mean_squared_error")
        mse_tr = mean_squared_error(y_cv, m.predict(X_cv))
        ax.scatter(X_cv.ravel(), y_cv, s=22, color="#888", alpha=0.7, zorder=5)
        ax.plot(np.linspace(0,2,400), f_true_global(np.linspace(0,2,400)),
                "k--", alpha=0.35, lw=1.5)
        ax.plot(x_p.ravel(), m.predict(x_p), "#6C3FC5", lw=2.5)
        ax.set_title(f"lambda = {lam}\nMSE treino={mse_tr:.3f} | CV={-sc.mean():.3f}",
                     fontsize=9)
        ax.set_ylim(-3, 3); ax.grid(alpha=0.22); ax.set_xlabel("x", fontsize=8)
    plt.suptitle(f"Ridge Regularization (L2) — Grau {grau}\n"
                 "lambda=0: overfitting | lambda grande: underfitting",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_N_vs_p():
    """Log-scale map of N vs p with model recommendations."""
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="white")
    ax.set_facecolor("#FAFAFA")

    ax.axvspan(50, 1000,     alpha=0.10, color="#E8534A")
    ax.axvspan(1000, 100000, alpha=0.10, color="#F5A623")
    ax.axvspan(100000, 1.2e6, alpha=0.10, color="#00C9A7")

    p_line = np.logspace(0, 4, 300)
    ax.plot(p_line, p_line, "k--", lw=2, label="N = p  (fronteira critica)")

    ax.text(200,     0.5,  "Small Data\n(N < 1k)", color="#E8534A",
            fontsize=9.5, fontweight="bold", ha="center")
    ax.text(8000,    0.5,  "Medium Data\n(1k – 100k)", color="#CC7700",
            fontsize=9.5, fontweight="bold", ha="center")
    ax.text(5e5,     0.5,  "Large Data\n(N > 100k)", color="#007755",
            fontsize=9.5, fontweight="bold", ha="center")

    cenarios = [
        (300,   50,   "Ridge / Lasso\nSVM\nDecision Tree rasa", "#E8534A"),
        (300,   800,  "Lasso + Feature\nSelection\n(N < p!)",   "#CC0000"),
        (10000, 25,   "Gradient Boosted\nTrees / Random Forest","#F5A623"),
        (8000,  3000, "GBT + Feature\nSelection / L1",          "#CC7700"),
        (4e5,   60,   "Deep Learning\nTransfer Learning\nSGD",  "#00C9A7"),
    ]
    for N, p, txt, cor in cenarios:
        ax.scatter([N], [p], s=180, color=cor, zorder=6, edgecolors="white", lw=1)
        ax.annotate(txt, xy=(N, p), xytext=(N*1.3, p*1.5),
                    fontsize=8, color=cor, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=cor, lw=1))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (numero de amostras)", fontsize=11)
    ax.set_ylabel("p (numero de features)", fontsize=11)
    ax.set_title("Model Selection Rules of Thumb — Mapa N vs. p",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(50, 1.2e6); ax.set_ylim(0.4, 12000)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def chart_model_vs_N():
    """Ridge vs. RF vs. GBT performance across data sizes."""
    X_pool, y_pool = make_regression(n_samples=2000, n_features=10, noise=15.0,
                                      n_informative=5, random_state=42)
    tamanhos = [50, 100, 250, 500, 1000, 2000]
    modelos = {
        "Ridge (L2)":        Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(n_estimators=50, random_state=7),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=7),
    }
    res = {n: [] for n in modelos}
    for N in tamanhos:
        for nome, m in modelos.items():
            reps = []
            for _ in range(12):
                idx = np.random.choice(len(X_pool), N, replace=False)
                sc  = cross_val_score(m, X_pool[idx], y_pool[idx], cv=5,
                                      scoring="neg_mean_squared_error")
                reps.append(-sc.mean())
            res[nome].append(np.mean(reps))

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    cores = {"Ridge (L2)": "#E8534A",
             "Random Forest": "#6C3FC5",
             "Gradient Boosting": "#00C9A7"}
    for nome, vals in res.items():
        ax.plot(tamanhos, vals, marker="o", lw=2.5, ms=7,
                label=nome, color=cores[nome])
    ax.axvline(1000, color="#AAAAAA", ls="--", alpha=0.7,
               label="Fronteira ~1k amostras")
    ax.set_xlabel("N (numero de amostras de treino)", fontsize=11)
    ax.set_ylabel("MSE (Cross-Validation)", fontsize=11)
    ax.set_title("Model Selection by Data Size:\nRidge vs. Random Forest vs. Gradient Boosting",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def chart_occams_razor():
    """MSE vs training time — Occam's Razor in practice."""
    X_m, y_m = make_regression(n_samples=5000, n_features=20, noise=10.0,
                                n_informative=8, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X_m, y_m, test_size=0.2, random_state=42)

    experimentos = [
        ("Dummy (media)",           DummyRegressor(strategy="mean")),
        ("Ridge (simples)",         Ridge(alpha=1.0)),
        ("Random Forest (50)",      RandomForestRegressor(n_estimators=50, random_state=42)),
        ("GBT (100 arv, depth 4)",  GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)),
        ("GBT (500 arv, depth 6)",  GradientBoostingRegressor(n_estimators=500, max_depth=6, random_state=42)),
    ]
    nomes, mses, tempos = [], [], []
    for nome, m in experimentos:
        t0 = time.time()
        m.fit(Xtr, ytr)
        tempos.append(time.time() - t0)
        mses.append(mean_squared_error(yte, m.predict(Xte)))
        nomes.append(nome)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    cores_e = ["#AAAAAA", "#F5A623", "#6C3FC5", "#00C9A7", "#2D1B6E"]
    for ax, (vals, xlabel, title) in zip(axes, [
        (mses,   "MSE (Teste)",            "Desempenho (MSE no Teste)"),
        (tempos, "Tempo de Treino (s)",    "Custo Computacional"),
    ]):
        ax.set_facecolor("#FAFAFA")
        bars = ax.barh(nomes, vals, color=cores_e, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + max(vals)*0.01,
                    bar.get_y() + bar.get_height()/2,
                    f"{v:.2f}" if xlabel.startswith("MSE") else f"{v:.2f}s",
                    va="center", fontsize=9.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        if xlabel.startswith("MSE"):
            ax.invert_xaxis()
        ax.grid(alpha=0.2, axis="x")

    plt.suptitle("Navalha de Occam no ML: Ganho Marginal vs. Custo Marginal",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_4phase():
    """4-Phase Workflow: candidate comparison + phase summary."""
    X_wf, y_wf = make_regression(n_samples=2500, n_features=15, n_informative=8,
                                   noise=20.0, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_wf, y_wf, test_size=0.2, random_state=42)
    N, p = X_dev.shape
    cv_wf = KFold(n_splits=5, shuffle=True, random_state=42)

    dummy  = DummyRegressor(strategy="mean")
    ridge_ = Ridge(alpha=1.0)
    mse_du = -cross_val_score(dummy,  X_dev, y_dev, cv=cv_wf,
                              scoring="neg_mean_squared_error").mean()
    mse_ri = -cross_val_score(ridge_, X_dev, y_dev, cv=cv_wf,
                              scoring="neg_mean_squared_error").mean()

    candidatos = {
        "Ridge (ML Baseline)":  Ridge(alpha=1.0),
        "Lasso (L1)":           Lasso(alpha=0.1, max_iter=5000),
        "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    res_cand = {}
    for nome, m in candidatos.items():
        sc = cross_val_score(m, X_dev, y_dev, cv=cv_wf,
                             scoring="neg_mean_squared_error")
        res_cand[nome] = (-sc.mean(), sc.std())

    melhor_nome = min(res_cand, key=lambda x: res_cand[x][0])
    mse_final = res_cand[melhor_nome][0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")

    # Candidate bar chart
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    nomes_c = list(res_cand.keys())
    mses_c  = [res_cand[n][0] for n in nomes_c]
    stds_c  = [res_cand[n][1] for n in nomes_c]
    cols_c  = ["#AAAACC", "#AAAACC", "#6C3FC5", "#00C9A7"]
    ax.barh(nomes_c, mses_c, xerr=stds_c, color=cols_c, alpha=0.85, capsize=5)
    ax.axvline(mse_du, color="#E8534A", ls="--", lw=2,
               label=f"Dummy ({mse_du:.0f})")
    ax.axvline(mse_ri, color="#F5A623", ls="--", lw=2,
               label=f"Ridge baseline ({mse_ri:.0f})")
    ax.set_xlabel("MSE (5-fold Cross-Validation)", fontsize=10)
    ax.set_title("Phase 4: Candidate Shortlist\n(verde = finalista selecionado)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.2, axis="x")

    # Phase summary boxes
    ax = axes[1]
    ax.axis("off")
    fases = [
        ("Phase 1", "Define Decision Criteria",
         "Metrica: MSE | Sem restricao de latencia\nInterpretabilidade moderada",
         "#2196F3"),
        ("Phase 2", "Characterize Data & Problem",
         f"N={N}, p={p}, N/p={N/p:.0f} → Medium Data\nTipo: Tabular → GBT recomendado",
         "#4CAF50"),
        ("Phase 3", "Establish Baselines",
         f"Dummy MSE={mse_du:.0f} | Ridge ML MSE={mse_ri:.0f}\nMelhoria baseline: {(1-mse_ri/mse_du)*100:.0f}%",
         "#FF9800"),
        ("Phase 4", "Candidate Shortlist (2-4)",
         f"Finalistas: RF + GBT\nMelhor (CV): {melhor_nome}\nMSE={mse_final:.0f}",
         "#9C27B0"),
    ]
    for i, (fase, titulo, det, cor) in enumerate(fases):
        y0 = 0.86 - i*0.23
        rect = mpatches.FancyBboxPatch(
            (0.02, y0-0.09), 0.96, 0.18,
            boxstyle="round,pad=0.02",
            facecolor=cor, alpha=0.13, edgecolor=cor, lw=2,
            transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.05, y0+0.055, f"{fase}: {titulo}",
                transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color=cor, va="top")
        ax.text(0.05, y0-0.015, det,
                transform=ax.transAxes, fontsize=8.5, va="top",
                color="#333333")
    ax.set_title("4-Phase Workflow — Resumo", fontsize=10, fontweight="bold")

    plt.suptitle("Structured 4-Phase Model Selection Workflow",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CONTENT BUILDER
# ---------------------------------------------------------------------------
def build_content(styles):
    S = styles
    story = []
    story.append(PageBreak())

    # ---- Intro page -------------------------------------------------------
    story.append(spacer(0.4))
    story.append(Paragraph(
        "Bem-vindo a Aula 2 — Bias-Variance Trade-Off & Model Selection", S["title"]))
    story.append(divider(AMBER, 1.5))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Esta aula aborda o <b>coracao estrategico da modelagem preditiva</b>: o "
        "equilibrio perfeito entre um modelo simplista demais (<b>High Bias</b>) e "
        "um modelo excessivamente complexo que decora o ruido (<b>High Variance</b>). "
        "Voce aprendera a decompor matematicamente o erro em tres partes irredutíveis "
        "e dominara um <b>framework de 4 fases</b> para selecionar modelos de forma "
        "sistematica e justificada — como um cientista de dados senior.",
        S["body"]))
    story.append(spacer(0.25))

    story.append(Paragraph("Roteiro desta aula:", S["subsection"]))
    for item in [
        "1. The Core Dilemma — Data Fit vs. Generalization",
        "2. Mathematical Error Decomposition — Bias² + Variance + Irreducible Error",
        "3. Visualizando o Trade-Off — a curva classica de complexidade vs. erro",
        "4. Metodos para Balancear — Cross-Validation, Regularization, Ensembles",
        "5. Model Selection Rules of Thumb — heuristicas por N e p",
        "6. Data Characteristics & Constraints — tipo, ruido, interpretabilidade",
        "7. Structured 4-Phase Model Selection Workflow",
        "8. Metacognicao: A Mentalidade do Cientista Preguicoso",
    ]:
        story.append(Paragraph(f"<bullet>-</bullet> {item}", S["bullet"]))
    story.append(PageBreak())

    # ================================================================ SECTION 1
    story.append(section_header("1. The Core Dilemma: Data Fit vs. Generalization", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Todo modelo preditivo vive sob uma tensao fundamental entre dois objetivos "
        "conflitantes: <b>Data Fit</b> (aprender bem os padroes dos dados de treino) e "
        "<b>Generalization</b> (prever corretamente dados nunca vistos). E geralmente "
        "impossivel maximizar ambos ao mesmo tempo. Esta tensao irresolvivel tem um nome: "
        "<b>Bias-Variance Trade-Off</b>.",
        S["body"]))
    story.append(spacer(0.2))

    story.append(Paragraph("Analogia do Estudante:", S["subsection"]))
    for item in [
        "<b>High Variance (Overfitting):</b> o aluno decora as respostas da lista. "
        "Vai bem no exercicio, mas mal na prova com questoes novas.",
        "<b>High Bias (Underfitting):</b> o aluno usa apenas uma regra geral muito "
        "simples. Erra tanto na lista quanto na prova.",
        "<b>Equilibrio ideal:</b> o aluno entende os conceitos e generaliza — "
        "Low Bias e Low Variance.",
    ]:
        story.append(Paragraph(f"<bullet>-</bullet> {item}", S["bullet"]))
    story.append(spacer(0.2))

    tbl1 = comparison_table(
        ["Cenario", "Erro Treino", "Erro Teste", "Diagnostico"],
        [
            ["Modelo muito simples", "Alto", "Alto",
             "Underfitting / High Bias"],
            ["Modelo ideal", "Medio", "Medio",
             "Boa generalizacao"],
            ["Modelo muito complexo", "Baixo", "Alto",
             "Overfitting / High Variance"],
        ],
        [4.5*cm, 2.5*cm, 2.5*cm, CONTENT_W - 9.5*cm])
    story.append(tbl1)
    story.append(spacer(0.2))

    fig1 = chart_core_dilemma()
    story.append(mat_image(fig1, 15))
    story.append(Paragraph(
        "Figura 1 — Underfitting (vermelho, grau 1): erro estrutural alto em treino e teste. "
        "Bom equilibrio (verde, grau 5): captura a estrutura sem decorar o ruido. "
        "Overfitting (roxo, grau 18): MSE de treino baixissimo, mas oscilacoes violentas.",
        S["caption"]))

    story.append(why_box(
        "Entender que Data Fit e Generalization sao conflitantes e o primeiro passo "
        "para deixar de otimizar a metrica errada. Muitos projetos fracassam porque "
        "o modelo tem MSE de treino excelente mas desempenho catastrofico em producao — "
        "classico sintoma de High Variance nao detectado.",
        "Antes de treinar qualquer modelo, pergunte: 'Estou avaliando no treino ou em "
        "dados reservados?' Se voce nao tem um holdout ou Cross-Validation, qualquer "
        "metrica de desempenho e enganosa. Em sklearn: sempre use cross_val_score() "
        "com cv=KFold(shuffle=True) ao inves de score() no treino.",
        S, color=VIOLET))
    story.append(PageBreak())

    # ================================================================ SECTION 2
    story.append(section_header("2. Mathematical Error Decomposition", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Por que o erro de teste e sempre maior que o erro de treino? Porque o "
        "erro total de um modelo e composto por <b>tres partes distintas</b>, com "
        "causas e remedios completamente diferentes. Compreender cada uma e "
        "fundamental para saber onde agir.",
        S["body"]))
    story.append(spacer(0.15))

    story.append(Paragraph("Analogia do Arqueiro:", S["subsection"]))
    for item in [
        "<b>Bias²:</b> o arqueiro mira sistematicamente a esquerda do alvo. "
        "Nao importa quantas flechas — todas erram no mesmo sentido. Erro estrutural do modelo.",
        "<b>Variance:</b> o arqueiro mira bem mas sua mao treme. As flechas caem "
        "aleatoriamente ao redor do alvo. Sensibilidade excessiva ao dataset de treino.",
        "<b>Irreducible Error:</b> o vento sopra de forma imprevivel. Nada pode ser "
        "feito. E o ruido intrinse co dos dados (sigma²).",
    ]:
        story.append(Paragraph(f"<bullet>-</bullet> {item}", S["bullet"]))
    story.append(spacer(0.2))

    story.append(Paragraph("Resultado da decomposicao matematica completa:", S["subsection"]))
    story.append(formula_block(
        r"\text{MSE}(x) = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2_{\epsilon}",
        label="Decomposicao fundamental: cada modelo paga Bias^2 + Variance + ruido irredutivel",
        formula_size=13))
    story.append(spacer(0.1))
    story.append(formula_block(
        r"\text{Bias}^2 = \left(f(x) - \mathbb{E}[\hat{f}(x)]\right)^2 ,\quad"
        r"\text{Var} = \mathbb{E}\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]",
        label="f(x): verdade  |  E[f_hat]: previsao media  |  sigma^2: ruido dos dados",
        formula_size=11))
    story.append(spacer(0.2))

    story.append(Paragraph("Derivacao resumida em 4 passos:", S["subsection"]))
    for step in [
        "Passo 1: MSE(x) = E[(y - f_hat(x))^<super>2</super>]  com  y = f(x) + epsilon",
        "Passo 2: Expandindo o quadrado — aparecem tres termos",
        "Passo 3: O termo cruzado desaparece porque E[epsilon] = 0 e epsilon independente de x",
        "Passo 4: Adicionar/subtrair E[f_hat] no primeiro termo revela Bias² + Variance",
    ]:
        story.append(Paragraph(f"<bullet>-&gt;</bullet> {step}", S["bullet"]))
    story.append(spacer(0.2))

    tbl2 = comparison_table(
        ["Componente", "Formula", "Causa", "Remedio"],
        [
            ["Bias²",
             "(f - E[f_hat])^2",
             "Modelo muito simples (suposicoes erradas)",
             "Aumentar complexidade, melhores features"],
            ["Variance",
             "E[(f_hat - E[f_hat])^2]",
             "Modelo muito complexo (decora o ruido)",
             "Regularizacao, mais dados, Ensembles"],
            ["Irreducible Error",
             "sigma²",
             "Ruido intrinse co dos dados",
             "NADA pode ser feito"],
        ],
        [2.2*cm, 3.8*cm, 4.5*cm, CONTENT_W - 10.5*cm])
    story.append(tbl2)
    story.append(spacer(0.2))

    fig2 = chart_bv_decomp_variability()
    story.append(mat_image(fig2, 16))
    story.append(Paragraph(
        "Figura 2 — Cada linha azul fina e a previsao de um modelo treinado em um "
        "dataset diferente. Grau 1: todas as linhas se agrupam (Low Variance, High Bias). "
        "Grau 12: as linhas divergem drasticamente (High Variance, Low Bias).",
        S["caption"]))

    story.append(why_box(
        "Essa decomposicao e o framework mais importante de toda a teoria de Machine "
        "Learning. Ela prove diagnostico preciso: se erro treino alto -> High Bias; "
        "se erro treino baixo e erro teste alto -> High Variance. Sem saber qual "
        "componente domina, qualquer intervencao e cega.",
        "Em Python, voce pode estimar empiricamente Bias^2 e Variance treinando o mesmo "
        "modelo em muitos sub-datasets (Monte Carlo). predicoes = np.zeros((n_sim, n_teste)) "
        "-> bias2 = np.mean((f_verdadeiro - predicoes.mean(axis=0))**2) "
        "e variancia = np.mean(predicoes.var(axis=0)).",
        S, color=CORAL))
    story.append(PageBreak())

    # ================================================================ SECTION 3
    story.append(section_header("3. Visualizando o Bias-Variance Trade-Off", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "A curva classica do trade-off mostra como Bias² e Variance se movem em "
        "direcoes opostas conforme aumentamos a <b>complexidade do modelo</b>. "
        "A zona ideal fica onde a soma Bias² + Variance e minima — "
        "nem muito simples, nem muito complexo.",
        S["body"]))
    story.append(spacer(0.2))

    fig3 = chart_bv_curve()
    story.append(mat_image(fig3, 15))
    story.append(Paragraph(
        "Figura 3 — A curva classica: Bias² (vermelho) decresce com a complexidade; "
        "Variance (roxo) cresce; Total Error (verde) tem um minimo na complexidade otima. "
        "O Irreducible Error (tracejado cinza) e o piso absoluto.",
        S["caption"]))
    story.append(spacer(0.2))

    story.append(Paragraph("Diagnostico pratico por Learning Curves:", S["subsection"]))
    story.append(Paragraph(
        "Plote o erro de treino e o erro de validacao em funcao do tamanho do dataset. "
        "O padrao revela o problema dominante:",
        S["body"]))
    for d in [
        "<b>High Bias:</b> erro treino alto + erro validacao alto + curvas convergem "
        "para o mesmo valor alto. Solucao: mais complexidade ou melhores features.",
        "<b>High Variance:</b> erro treino baixo + erro validacao alto + gap grande "
        "entre as curvas. Solucao: mais dados, regularizacao, ou modelo mais simples.",
    ]:
        story.append(Paragraph(f"<bullet>-</bullet> {d}", S["bullet"]))
    story.append(spacer(0.2))

    fig4 = chart_learning_curves()
    story.append(mat_image(fig4, 15))
    story.append(Paragraph(
        "Figura 4 — Learning Curves: grau 1 (High Bias) — ambos os erros convergem "
        "alto. Grau 10 (High Variance) — gap persistente entre treino e validacao.",
        S["caption"]))

    story.append(why_box(
        "Learning Curves sao a ferramenta de diagnostico mais pratica da modelagem. "
        "Elas respondem a pergunta 'mais dados ajudam?' de forma empirica: se o gap "
        "de High Variance diminui com mais N, mais dados sao a solucao. Se o gap "
        "nao diminui (High Bias), mais dados nao ajudam — precisamos de um modelo melhor.",
        "Em sklearn: plote learning_curve(modelo, X, y, cv=5) ou implemente manualmente "
        "com cross_val_score em subsets crescentes de X. Sempre plote ambas as curvas "
        "(treino e validacao) — ver so o erro de treino e um erro classico de iniciante.",
        S, color=AMBER))
    story.append(PageBreak())

    # ================================================================ SECTION 4
    story.append(section_header("4. Metodos para Balancear Bias e Variance", S))
    story.append(spacer(0.2))

    tbl4 = comparison_table(
        ["Metodo", "Atua sobre", "Como ajuda"],
        [
            ["Cross-Validation", "Avaliacao",
             "Estima erro de generalizacao sem vazar o teste"],
            ["Regularization (Ridge/Lasso)", "Variance",
             "Penaliza coeficientes grandes, reduz overfitting"],
            ["Feature Selection", "Bias + Variance",
             "Remove features irrelevantes e ruidosas"],
            ["Ensemble Methods (RF, GBT)", "Variance (principalmente)",
             "Combina modelos para reduzir instabilidade"],
        ],
        [4.0*cm, 3.5*cm, CONTENT_W - 7.5*cm])
    story.append(tbl4)
    story.append(spacer(0.25))

    story.append(Paragraph("4a. Cross-Validation", S["subsection"]))
    story.append(Paragraph(
        "Em vez de avaliar o modelo em um unico split treino/teste (que pode ser "
        "sortudo ou azarado), dividimos os dados em K folds e rotacionamos quem "
        "serve de teste. A estimativa final e a media dos K erros.",
        S["body"]))
    story.append(formula_block(
        r"\text{CV}_{(K)} = \frac{1}{K} \sum_{k=1}^{K} \text{MSE}_k",
        label="K: numero de folds  |  MSE_k: erro no fold k  |  "
              "K=5 ou K=10 sao as escolhas mais comuns"))
    story.append(spacer(0.2))

    fig5 = chart_cv_selection()
    story.append(mat_image(fig5, 14))
    story.append(Paragraph(
        "Figura 5 — 10-Fold CV seleciona automaticamente o grau polinomial "
        "otimo: graus baixos (High Bias) e altos (High Variance) tem MSE elevado; "
        "o minimo indica o ponto de equilibrio.",
        S["caption"]))
    story.append(spacer(0.2))

    story.append(Paragraph("4b. Regularization — Ridge (L2) e Lasso (L1)", S["subsection"]))
    story.append(Paragraph(
        "Adicionamos uma penalidade aos coeficientes grandes na funcao de custo. "
        "Isso forca o modelo a ser mais simples, reduzindo a Variance ao custo de "
        "um leve aumento no Bias. O parametro lambda controla o trade-off.",
        S["body"]))
    story.append(formula_block(
        r"\hat{a}_{\text{Ridge}} = \arg\min_{a}\|y - Ba\|^2 + \lambda\|a\|^2"
        r"\;\Rightarrow\; (B^TB + \lambda I)^{-1}B^Ty",
        label="Ridge (L2): solucao analitica  |  lambda grande -> mais restricao -> menos Variance",
        formula_size=11, bg_hex="#1A0E3E"))
    story.append(spacer(0.15))
    story.append(formula_block(
        r"\hat{a}_{\text{Lasso}} = \arg\min_{a}\|y - Ba\|^2 + \lambda\|a\|_1",
        label="Lasso (L1): sem solucao fechada  |  produz coeficientes EXATAMENTE zero (Feature Selection implicita)",
        formula_size=12, bg_hex="#1A0E3E"))
    story.append(spacer(0.2))

    fig6 = chart_regularization()
    story.append(mat_image(fig6, 16))
    story.append(Paragraph(
        "Figura 6 — Ridge com grau 10: lambda=0 (overfitting extremo) ate lambda=10 "
        "(underfitting suave). O MSE de CV identifica o lambda otimo em torno de 0.01-0.1.",
        S["caption"]))

    story.append(why_box(
        "Regularizacao e a ferramenta mais poderosa contra High Variance. Ridge "
        "distribui coeficientes uniformemente (bom quando todas as features importam); "
        "Lasso zera coeficientes irrelevantes (Feature Selection automatica — ideal "
        "quando voce suspeita que muitas features sao ruido).",
        "Em sklearn: Ridge(alpha=lambda) e Lasso(alpha=lambda). Escolha lambda via "
        "RidgeCV ou LassoCV (busca automatica em grade logaritmica). Regra de ouro: "
        "sempre tente lambda na faixa [1e-4, 1e4] em escala logaritmica, usando "
        "np.logspace(-4, 4, 20) como grade de busca.",
        S, color=TEAL))
    story.append(PageBreak())

    # ================================================================ SECTION 5
    story.append(section_header("5. Model Selection Rules of Thumb", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Antes de qualquer experimento, podemos usar o tamanho dos dados "
        "(<b>N</b> = numero de amostras, <b>p</b> = numero de features) como "
        "bussola inicial. O mapa abaixo nao e uma lei universal — e um ponto de "
        "partida racional para evitar gastar dias treinando uma rede neural quando "
        "uma regressao linear teria bastado.",
        S["body"]))
    story.append(spacer(0.15))

    story.append(formula_block(
        r"N < p \;\Rightarrow\; \text{sistema subdeterminado: infinitas solucoes, Variance explosiva}",
        label="Regra pratica: precisamos de pelo menos N > 10*p para modelos lineares sem regularizacao",
        formula_size=10, bg_hex="#1A3A1A"))
    story.append(spacer(0.2))

    tbl5 = comparison_table(
        ["Faixa de N", "Foco", "Modelos recomendados", "Heuristica-chave"],
        [
            ["Small Data\n(N < 1.000 ou N < p)",
             "Variance baixa",
             "Ridge, Lasso, SVM,\nDecision Trees rasas",
             "Modelos simples e regularizados.\nCV agressiva (10 folds)."],
            ["Medium Data\n(1k < N < 100k)",
             "Efeitos nao-lineares\n+ interpretabilidade",
             "Gradient Boosted Trees,\nRandom Forest, Ridge",
             "Para tabular: GBT primeiro.\nRidge como baseline obrigatorio."],
            ["Large Data\n(N > 100k)",
             "Flexibilidade\nvale o custo",
             "Deep Learning,\nTransfer Learning, SGD",
             "Para texto/imagem: Transfer Learning.\nTabular: GBT ainda compete."],
        ],
        [3.0*cm, 2.5*cm, 4.0*cm, CONTENT_W - 9.5*cm])
    story.append(tbl5)
    story.append(spacer(0.2))

    fig7 = chart_N_vs_p()
    story.append(mat_image(fig7, 15))
    story.append(Paragraph(
        "Figura 7 — Mapa N vs. p em escala log-log. A linha N=p e a fronteira critica: "
        "acima dela, o sistema e subdeterminado. Cada ponto representa um cenario tipico "
        "com o modelo recomendado.",
        S["caption"]))
    story.append(spacer(0.2))

    fig8 = chart_model_vs_N()
    story.append(mat_image(fig8, 14))
    story.append(Paragraph(
        "Figura 8 — Comparacao Ridge vs. Random Forest vs. GBT em diferentes N. "
        "Com N pequeno, Ridge compete fortemente com metodos complexos. "
        "Com N > 500, GBT e RF passam a superar consistentemente o Ridge.",
        S["caption"]))

    story.append(why_box(
        "A razao N/p e um dos indicadores mais praticos que um Data Scientist calcula "
        "antes de qualquer modelagem. Quando N/p < 10, qualquer modelo flexivel sem "
        "regularizacao vai overfitar garantidamente. Isso e matematica, nao opiniao.",
        "Calcule sempre no inicio do projeto: N, p = X_treino.shape; print(N/p). "
        "Se N/p < 5, use Lasso ou Ridge com CrossValidation forte. "
        "Se N/p > 100 e dados forem tabulares, GBT (XGBoost, LightGBM) e seu "
        "ponto de partida natural. Documente essa decisao antes de escrever codigo.",
        S, color=VIOLET))
    story.append(PageBreak())

    # ================================================================ SECTION 6
    story.append(section_header("6. Data Characteristics & Constraints", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Alem do tamanho N, tres caracteristicas dos dados moldam profundamente a "
        "escolha do algoritmo: o <b>tipo de dado</b> (tabular, texto, imagem), o "
        "<b>nivel de ruido</b> (qualidade dos labels) e as "
        "<b>restricoes de interpretabilidade</b>.",
        S["body"]))
    story.append(spacer(0.15))

    story.append(Paragraph("6a. Data Type", S["subsection"]))
    tbl6a = comparison_table(
        ["Data Type", "Ponto de Partida", "Notas importantes"],
        [
            ["Tabular", "Gradient Boosted Trees\n(XGBoost, LightGBM)",
             "Quase sempre o melhor para dados tabulares"],
            ["Texto", "Transfer Learning (BERT, GPT)",
             "Fine-tuning se N pequeno; Transformer proprio se N grande"],
            ["Imagem / Audio", "Transfer Learning (ResNet, EfficientNet)",
             "Do zero so se os dados forem muito especificos"],
            ["Series Temporais", "ARIMA, regressao com lags",
             "Nao pule a baseline ingenua (prever ultimo valor observado)"],
        ],
        [3.0*cm, 4.5*cm, CONTENT_W - 7.5*cm])
    story.append(tbl6a)
    story.append(spacer(0.2))

    story.append(Paragraph("6b. Noise Level / Label Quality", S["subsection"]))
    story.append(Paragraph(
        "Se os proprios labels y estao errados ou ruidosos (anotadores humanos "
        "discordantes, sensores defeituosos), aumentar a complexidade do modelo "
        "<b>nao ajuda</b> — voce apenas memoriza os erros com mais precisao. "
        "Neste caso, melhorar a qualidade dos labels tem impacto muito maior "
        "do que qualquer truque de modelagem.",
        S["body"]))
    story.append(spacer(0.15))

    story.append(Paragraph("6c. Interpretability (Interpretabilidade)", S["subsection"]))
    tbl6c = comparison_table(
        ["Necessidade de interpretabilidade", "Modelos adequados"],
        [
            ["Alta + baixa latencia",
             "Regressao Linear/Logistica, Decision Trees rasas"],
            ["Alta acuracia + interpretabilidade parcial",
             "GBT com SHAP values (explicacoes por instancia)"],
            ["Maxima acuracia, caixa-preta aceitavel",
             "Deep Learning, Random Forest, Ensembles complexos"],
        ],
        [5.5*cm, CONTENT_W - 5.5*cm])
    story.append(tbl6c)
    story.append(spacer(0.2))

    story.append(Paragraph(
        "<b>Mensagem central:</b> nao existe algoritmo universalmente superior. "
        "A escolha otima emerge da combinacao de N, p, tipo de dado, nivel de ruido "
        "e restricoes de negocio. Um cientista de dados senior raciocina sobre esses "
        "eixos <i>antes</i> de abrir o codigo.",
        S["highlight"]))

    story.append(why_box(
        "Em setores regulados (credito bancario, diagnostico medico, decisoes juridicas), "
        "a interpretabilidade nao e um luxo — e um requisito legal. Um modelo de Deep "
        "Learning com 95% de acuracia pode ser rejeitado pelo departamento juridico se "
        "nao conseguir explicar cada decisao individual.",
        "Para alta interpretabilidade: use statsmodels.OLS (fornece coeficientes, "
        "p-values, intervalos de confianca). Para 'alta acuracia + alguma explicabilidade': "
        "use sklearn GBT + SHAP (pip install shap). shap.TreeExplainer(modelo).shap_values(X) "
        "gera explicacoes por feature para cada previsao individual.",
        S, color=MINT))
    story.append(PageBreak())

    # ================================================================ SECTION 7
    story.append(section_header("7. Structured 4-Phase Model Selection Workflow", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "O professor apresenta um <b>framework corporativo de 4 fases</b> para "
        "selecao de modelos. Este processo separa o Data Scientist junior "
        "(que testa algoritmos aleatoriamente) do senior "
        "(que usa um metodo sistematico e justificado).",
        S["body"]))
    story.append(spacer(0.2))

    # Four phases as colored rows
    fases_data = [
        ("Phase 1", "Define Decision Criteria", CORAL,
         "Antes de tocar nos dados, traduza os objetivos de negocio em metricas mensuraveis. "
         "Responda: qual metrica de sucesso? (MSE, AUC, F1?) Existe restricao de latencia? "
         "O setor exige interpretabilidade? Qual o custo de cada tipo de erro?"),
        ("Phase 2", "Characterize Data & Problem", TEAL,
         "Analise exploratoria focada em informacoes que guiam a selecao. "
         "Calcule N, p, N/p. Verifique o tipo de dado. "
         "Avalie proporcao de valores ausentes, outliers e qualidade dos labels. "
         "Identifique se as relacoes sao lineares ou nao-lineares."),
        ("Phase 3", "Establish Baselines", VIOLET,
         "Crie dois modelos de referencia ANTES de qualquer fine-tuning. "
         "(1) Dummy Baseline: prever sempre a media — define o piso minimo. "
         "(2) ML Baseline: Ridge ou Logistic Regression com parametros padrao. "
         "Qualquer candidato deve superar o ML Baseline para justificar maior complexidade."),
        ("Phase 4", "Candidate Shortlist", AMBER,
         "Usando as heuristicas da Secao 5, reduza o espaco de busca para 2 a 4 "
         "familias de modelos antes de qualquer fine-tuning de hiperparametros. "
         "Avalie rapidamente com CV e hiperparametros padrao. "
         "Refine apenas os 2-4 finalistas — evita overfitting de hiperparametros."),
    ]

    h_style = ParagraphStyle("ph", fontName="Helvetica-Bold", fontSize=10,
                              textColor=WHITE, leading=13)
    b_style = ParagraphStyle("pb", fontName="Helvetica", fontSize=9,
                              textColor=colors.HexColor("#111111"),
                              leading=13, alignment=TA_JUSTIFY)

    for fase, titulo, cor, texto in fases_data:
        row_data = [
            [Paragraph(f"{fase}: {titulo}", h_style)],
            [Paragraph(texto, b_style)],
        ]
        tbl = Table(row_data, colWidths=[CONTENT_W])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), cor),
            ("BACKGROUND",    (0,1), (-1,1), colors.HexColor("#F5F3FF")),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LINEBELOW",     (0,-1), (-1,-1), 0.5,
             colors.HexColor("#CCBBEE")),
        ]))
        story.append(tbl)
        story.append(spacer(0.08))

    story.append(spacer(0.2))

    fig9 = chart_4phase()
    story.append(mat_image(fig9, 16))
    story.append(Paragraph(
        "Figura 9 — Esquerda: comparacao dos candidatos em 5-fold CV (verde = finalista). "
        "Direita: resumo visual das 4 fases com metricas reais calculadas no dataset simulado.",
        S["caption"]))

    story.append(why_box(
        "O erro mais comum em projetos de ML e pular direto para hiperparameter tuning "
        "sem baseline. Um Dummy Baseline que prevê sempre a media pode ter MSE=1000. "
        "Um Ridge bem ajustado chega a MSE=400 em 2 segundos. So entao vale questionar "
        "se GBT ou Deep Learning justificam mais esforco.",
        "Implemente sempre: dummy = DummyRegressor(strategy='mean'); "
        "ridge_base = Ridge(alpha=1.0); "
        "mse_dummy = -cross_val_score(dummy, X_dev, y_dev, cv=5, scoring='neg_mean_squared_error').mean(); "
        "mse_ridge = -cross_val_score(ridge_base, X_dev, y_dev, cv=5, scoring='neg_mean_squared_error').mean(). "
        "Se MSE_ridge < MSE_dummy * 0.5, o problema tem estrutura aprendivel.",
        S, color=DEEP_PURPLE))
    story.append(PageBreak())

    # ================================================================ SECTION 8
    story.append(section_header("8. Metacognicao: A Mentalidade do Cientista Preguicoso", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Se um modelo linear simples (ML Baseline) atinge 85% de precisao em 2 "
        "segundos de computacao, e uma rede neural complexa leva 3 horas para "
        "atingir 86%, o ganho marginal de 1% justifica a complexidade de manutencao "
        "e os custos de infraestrutura?",
        S["quote"]))
    story.append(spacer(0.15))
    story.append(Paragraph(
        "Esta e a pergunta que todo Data Scientist deve se fazer <b>antes</b> de "
        "avancar para modelos mais complexos. O Principio da Navalha de Occam aplicado "
        "ao ML: prefira o modelo mais simples que satisfaca os criterios de decisao "
        "definidos na Phase 1.",
        S["body"]))
    story.append(spacer(0.2))

    tbl8 = comparison_table(
        ["Custo da complexidade", "Impacto real"],
        [
            ["Tempo de treinamento",
             "Horas a dias para grandes redes neurais"],
            ["Custo computacional",
             "GPU/TPU sao caros — modelos simples rodam em CPU"],
            ["Manutencao",
             "Modelos complexos quebram de formas mais sutis (drift, etc.)"],
            ["Interpretabilidade",
             "Dificil auditar e debugar em producao"],
            ["Estabilidade",
             "Alta Variance: pequenas mudancas nos dados -> grandes mudancas no modelo"],
        ],
        [5.0*cm, CONTENT_W - 5.0*cm])
    story.append(tbl8)
    story.append(spacer(0.2))

    fig10 = chart_occams_razor()
    story.append(mat_image(fig10, 15))
    story.append(Paragraph(
        "Figura 10 — Navalha de Occam na pratica: cada passo de complexidade adicional "
        "traz ganho marginal decrescente de MSE a um custo de treino crescente. "
        "A pergunta e sempre: o ganho marginal justifica o custo marginal?",
        S["caption"]))

    story.append(why_box(
        "A mentalidade do Cientista Preguicoso nao e preguica — e eficiencia. "
        "Em vez de gastar 2 semanas ajustando uma rede neural, um profissional senior "
        "gasta 30 minutos estabelecendo uma baseline solida, entende o piso de "
        "desempenho, e so entao decide se a complexidade adicional e justificavel.",
        "Adote o mantra: Dummy -> ML Baseline -> GBT -> Deep Learning (so se necessario). "
        "Registre o MSE de cada etapa e o tempo investido. Se GBT atinge 92% do "
        "desempenho otimo em 5% do tempo de uma rede neural, o GBT e a escolha correta "
        "para 90% dos projetos corporativos.",
        S, color=CORAL))
    story.append(PageBreak())

    # ================================================================ SECTION 9 — Resumo
    story.append(section_header("9. Resumo & Checklist do Cientista de Dados", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Use esta tabela como referencia rapida antes de iniciar qualquer projeto "
        "de modelagem preditiva.",
        S["body"]))
    story.append(spacer(0.15))

    tbl9 = comparison_table(
        ["Conceito", "Ideia Central", "Ferramenta/Formula"],
        [
            ["Core Dilemma", "Data Fit vs. Generalizacao",
             "Learning Curves (treino vs. validacao)"],
            ["Error Decomposition", "3 componentes distintos",
             "MSE = Bias^2 + Variance + sigma^2"],
            ["Bias^2", "Erro estrutural do modelo",
             "(f - E[f_hat])^2"],
            ["Variance", "Sensibilidade ao dataset",
             "E[(f_hat - E[f_hat])^2]"],
            ["Irreducible Error", "Ruido intrinse co",
             "sigma^2 — nada a fazer"],
            ["Cross-Validation", "Estimador confiavel do erro",
             "CV_K = (1/K) * sum MSE_k"],
            ["Regularization", "Reduz Variance",
             "Ridge: lambda*||a||^2  |  Lasso: lambda*||a||_1"],
            ["Rules of Thumb", "N/p guia a escolha",
             "N<1k: Ridge/SVM; N<100k: GBT; N>100k: DL"],
            ["4-Phase Workflow", "Framework sistematico",
             "Define -> Characterize -> Baseline -> Shortlist"],
            ["Navalha de Occam", "Ganho marginal vs. custo marginal",
             "Dummy -> Ridge -> GBT -> Deep Learning"],
        ],
        [4.0*cm, 5.0*cm, CONTENT_W - 9.0*cm])
    story.append(tbl9)
    story.append(spacer(0.25))
    story.append(divider(AMBER))
    story.append(spacer(0.15))
    story.append(Paragraph(
        "Na proxima aula, aprofundaremos a <b>Regularizacao</b> (Ridge, Lasso, "
        "Elastic Net, Dropout) e a <b>Cross-Validation</b> sob uma perspectiva mais "
        "tecnica, incluindo Regularization Paths, Early Stopping e Stratified K-Fold. "
        "Os conceitos de Bias-Variance desta aula sao o fundamento de toda essa teoria.",
        S["body"]))
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
        title="Aula 2 — Bias-Variance Trade-Off & Model Selection",
        author="Prof. Johannes Schwab, PhD — FH Kufstein Tirol",
        subject="Statistisches Lernen 2")

    styles = build_styles()
    story  = [NextPageTemplate("Content")] + build_content(styles)
    doc.build(story)

    # Two-pass for correct page count in footer
    try:
        import PyPDF2
        with open(output_path, "rb") as f:
            total_pages[0] = len(PyPDF2.PdfReader(f).pages)
        doc2 = BaseDocTemplate(
            output_path, pagesize=A4,
            pageTemplates=[cover_tpl, content_tpl],
            leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0,
            title="Aula 2 — Bias-Variance Trade-Off & Model Selection",
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
    out  = os.path.join(base, "L2_Bias_Variance_Model_Selection.pdf")
    print(f"Gerando PDF: {out}")
    build_pdf(out)
    print("PDF gerado com sucesso!")
