"""
generate_L1_pdf.py
------------------
Generates a magazine-style lecture notes PDF for Lecture 1:
General Linear Models & Basis Functions
(Statistisches Lernen 2 — FH Kufstein Tirol)

Output language: PT-BR  |  Technical terms: English
Dependencies: reportlab, matplotlib, numpy, scipy
"""

import io
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

# ---------------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------------
NAVY       = colors.HexColor("#1A2B4A")
TEAL       = colors.HexColor("#0D7377")
GOLD       = colors.HexColor("#F5A623")
CORAL      = colors.HexColor("#E8534A")
LAVENDER   = colors.HexColor("#6C63FF")
MINT       = colors.HexColor("#00C9A7")
LIGHT_BG   = colors.HexColor("#F4F6FA")
SIDEBAR_BG = colors.HexColor("#E8EDF5")
WHITE      = colors.white
BLACK      = colors.black

PAGE_W, PAGE_H = A4          # 595.27 x 841.89 pt
MARGIN        = 1.8 * cm
SIDEBAR_W     = 1.2 * cm
CONTENT_W     = PAGE_W - 2 * MARGIN - SIDEBAR_W
HEADER_H      = 1.6 * cm
FOOTER_H      = 1.2 * cm

# ---------------------------------------------------------------------------
# CUSTOM CANVAS — geometric backgrounds, sidebar, header, footer
# ---------------------------------------------------------------------------
class LectureCanvas:
    """Mixin-style callable used as onPage / onFirstPage callback."""

    def __init__(self, total_pages_ref: list):
        # total_pages_ref is a 1-element list so we can mutate it after build
        self._total = total_pages_ref

    def draw_cover(self, canvas, doc):
        """Draws the decorative cover page."""
        canvas.saveState()
        w, h = PAGE_W, PAGE_H

        # Background gradient approximation via stacked rectangles
        steps = 60
        for i in range(steps):
            t = i / steps
            r = int(26 + t * (13 - 26))
            g = int(43 + t * (115 - 43))
            b = int(74 + t * (119 - 74))
            canvas.setFillColorRGB(r/255, g/255, b/255)
            canvas.rect(0, h * i / steps, w, h / steps + 1, fill=1, stroke=0)

        # Decorative circles
        for cx, cy, cr, alpha in [
            (w * 0.85, h * 0.75, 3.5 * cm, 0.15),
            (w * 0.1,  h * 0.25, 5.0 * cm, 0.10),
            (w * 0.7,  h * 0.35, 2.0 * cm, 0.20),
        ]:
            canvas.setFillColor(colors.HexColor("#FFFFFF"))
            canvas.setFillAlpha(alpha)
            canvas.circle(cx, cy, cr, fill=1, stroke=0)

        canvas.setFillAlpha(1.0)

        # Gold accent bar
        canvas.setFillColor(GOLD)
        canvas.rect(0, h * 0.55, SIDEBAR_W * 1.5, h * 0.45, fill=1, stroke=0)

        # Course tag
        canvas.setFillColor(TEAL)
        canvas.roundRect(MARGIN, h * 0.78, 7 * cm, 0.9 * cm, 4, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(MARGIN + 0.3 * cm, h * 0.78 + 0.25 * cm,
                          "STATISTISCHES LERNEN 2")

        # Main title
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 28)
        canvas.drawString(MARGIN, h * 0.62, "Aula 1")
        canvas.setFont("Helvetica", 18)
        canvas.drawString(MARGIN, h * 0.56, "General Linear Models")
        canvas.drawString(MARGIN, h * 0.51, "& Basis Functions")

        # Subtitle line
        canvas.setFillColor(GOLD)
        canvas.rect(MARGIN, h * 0.49, 8 * cm, 0.06 * cm, fill=1, stroke=0)

        # Author & institution
        canvas.setFillColor(colors.HexColor("#CCDDEE"))
        canvas.setFont("Helvetica", 10)
        canvas.drawString(MARGIN, h * 0.45, "Prof. Johannes Schwab, PhD")
        canvas.drawString(MARGIN, h * 0.42, "FH Kufstein Tirol")

        # Bottom decoration
        canvas.setFillColor(GOLD)
        canvas.rect(0, 0, w, 0.4 * cm, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.rect(0, 0.4 * cm, w, 0.15 * cm, fill=1, stroke=0)

        canvas.restoreState()

    def draw_page(self, canvas, doc):
        """Draws header, sidebar, footer for every content page."""
        canvas.saveState()
        w, h = PAGE_W, PAGE_H
        pn = doc.page

        # Left sidebar
        canvas.setFillColor(SIDEBAR_BG)
        canvas.rect(0, FOOTER_H, SIDEBAR_W, h - HEADER_H - FOOTER_H,
                    fill=1, stroke=0)

        # Sidebar accent strip
        canvas.setFillColor(TEAL)
        canvas.rect(SIDEBAR_W - 0.15 * cm, FOOTER_H,
                    0.15 * cm, h - HEADER_H - FOOTER_H, fill=1, stroke=0)

        # Header bar
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - HEADER_H, w, HEADER_H, fill=1, stroke=0)

        # Header gold accent
        canvas.setFillColor(GOLD)
        canvas.rect(0, h - HEADER_H - 0.12 * cm, w, 0.12 * cm, fill=1, stroke=0)

        # Header text — left
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(SIDEBAR_W + 0.4 * cm,
                          h - HEADER_H + 0.55 * cm,
                          "Statistisches Lernen 2  |  Aula 1 — General Linear Models & Basis Functions")

        # Header text — right
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 0.4 * cm,
                               h - HEADER_H + 0.55 * cm,
                               "FH Kufstein Tirol")

        # Footer bar
        canvas.setFillColor(LIGHT_BG)
        canvas.rect(0, 0, w, FOOTER_H, fill=1, stroke=0)
        canvas.setFillColor(NAVY)
        canvas.rect(0, FOOTER_H - 0.08 * cm, w, 0.08 * cm, fill=1, stroke=0)

        # Footer text — left
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(SIDEBAR_W + 0.4 * cm, 0.38 * cm,
                          "Prof. Johannes Schwab, PhD  —  FH Kufstein Tirol")

        # Page counter (total resolved after build via 2-pass trick)
        total = self._total[0] if self._total[0] else "?"
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawRightString(w - 0.4 * cm, 0.38 * cm,
                               f"Página {pn} de {total}")

        # Sidebar rotated chapter label
        canvas.setFillColor(TEAL)
        canvas.saveState()
        canvas.translate(SIDEBAR_W * 0.5, h * 0.5)
        canvas.rotate(90)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(NAVY)
        canvas.drawCentredString(0, 0, "GENERAL LINEAR MODELS  |  BASIS FUNCTIONS")
        canvas.restoreState()

        canvas.restoreState()


# ---------------------------------------------------------------------------
# PARAGRAPH STYLES
# ---------------------------------------------------------------------------
def build_styles():
    base = getSampleStyleSheet()

    def ps(name, **kwargs):
        return ParagraphStyle(name, **kwargs)

    styles = {
        "title": ps("title",
            fontName="Helvetica-Bold", fontSize=22,
            textColor=NAVY, spaceAfter=6, spaceBefore=4,
            alignment=TA_LEFT),

        "section": ps("section",
            fontName="Helvetica-Bold", fontSize=15,
            textColor=TEAL, spaceBefore=14, spaceAfter=4,
            borderPadding=(0, 0, 2, 0)),

        "subsection": ps("subsection",
            fontName="Helvetica-Bold", fontSize=11,
            textColor=NAVY, spaceBefore=8, spaceAfter=3),

        "why_header": ps("why_header",
            fontName="Helvetica-Bold", fontSize=10,
            textColor=WHITE, spaceBefore=10, spaceAfter=2),

        "why_body": ps("why_body",
            fontName="Helvetica", fontSize=9.5,
            textColor=colors.HexColor("#222222"),
            spaceBefore=2, spaceAfter=2,
            leftIndent=6, alignment=TA_JUSTIFY),

        "body": ps("body",
            fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor("#2B2B2B"),
            leading=15, spaceBefore=3, spaceAfter=3,
            alignment=TA_JUSTIFY),

        "bullet": ps("bullet",
            fontName="Helvetica", fontSize=9.5,
            textColor=colors.HexColor("#2B2B2B"),
            leading=14, spaceBefore=1, spaceAfter=1,
            leftIndent=14, bulletIndent=4),

        "formula_label": ps("formula_label",
            fontName="Helvetica-Bold", fontSize=9,
            textColor=NAVY, spaceBefore=2, spaceAfter=0,
            alignment=TA_CENTER),

        "caption": ps("caption",
            fontName="Helvetica-Oblique", fontSize=8,
            textColor=colors.HexColor("#555555"),
            spaceBefore=2, spaceAfter=6, alignment=TA_CENTER),

        "highlight": ps("highlight",
            fontName="Helvetica-Bold", fontSize=9.5,
            textColor=NAVY, spaceBefore=4, spaceAfter=4,
            leftIndent=10),

        "code": ps("code",
            fontName="Courier", fontSize=8.5,
            textColor=colors.HexColor("#1E1E1E"),
            backColor=colors.HexColor("#F0F0F0"),
            spaceBefore=4, spaceAfter=4,
            leftIndent=8, rightIndent=8, leading=13),
    }
    return styles


# ---------------------------------------------------------------------------
# HELPER BUILDERS
# ---------------------------------------------------------------------------
def section_header(title: str, styles: dict):
    """Returns a colored header block for a major section."""
    data = [[Paragraph(title, styles["section"])]]
    tbl = Table(data, colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LINEBELOW",    (0, 0), (-1, -1), 2, TEAL),
        ("LINEABOVE",    (0, 0), (-1, 0),  0.5, TEAL),
    ]))
    return tbl


def why_box(why_text: str, apply_text: str, styles: dict,
            color: colors.Color = TEAL):
    """Two-row colored box for 'Por que isso é importante' + 'Aplicação Prática'."""
    header_style = ParagraphStyle(
        "wh", fontName="Helvetica-Bold", fontSize=9.5,
        textColor=WHITE, leading=13)
    body_style = ParagraphStyle(
        "wb", fontName="Helvetica", fontSize=9,
        textColor=colors.HexColor("#111111"), leading=13,
        alignment=TA_JUSTIFY)

    data = [
        [Paragraph("Por que isso é importante para Data Science?", header_style)],
        [Paragraph(why_text, body_style)],
        [Paragraph("Aplicação Prática em Machine Learning", header_style)],
        [Paragraph(apply_text, body_style)],
    ]
    tbl = Table(data, colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), color),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EAF7F5")),
        ("BACKGROUND", (0, 2), (-1, 2), NAVY),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#E8EDF5")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [color, colors.HexColor("#EAF7F5"),
                                               NAVY, colors.HexColor("#E8EDF5")]),
    ]))
    return tbl


def formula_block(latex_str: str, label: str = "",
                  bg_hex: str = "#1A2B4A",
                  formula_size: int = 15) -> Image:
    """
    Renders a LaTeX formula via matplotlib mathtext and returns a ReportLab Image.
    This avoids font-missing artefacts that occur when rendering Unicode math
    symbols inside Helvetica-based Paragraph cells.
    """
    w_in = CONTENT_W / 72          # content width in inches (72 pt per inch)
    has_label = bool(label.strip())
    h_in = 1.05 if not has_label else 1.55

    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.patch.set_facecolor(bg_hex)
    ax.set_facecolor(bg_hex)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Gold border strips (top and bottom) drawn in data coordinates
    ax.plot([0, 1], [0.985, 0.985], color="#F5A623", linewidth=3.5,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0, 1], [0.015, 0.015], color="#F5A623", linewidth=3.5,
            transform=ax.transAxes, clip_on=False)

    y_f = 0.63 if has_label else 0.50
    ax.text(0.5, y_f, f"${latex_str}$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=formula_size, color="white",
            math_fontfamily="dejavusans")

    if has_label:
        ax.text(0.5, 0.20, label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#AACCEE", fontstyle="italic")

    buf = io.BytesIO()
    # Use exact figsize (no tight bbox) so aspect ratio is predictable
    fig.savefig(buf, format="png", dpi=150,
                facecolor=bg_hex, edgecolor="none")
    buf.seek(0)
    plt.close(fig)

    # Scale to CONTENT_W and keep aspect ratio
    img_h = (h_in / w_in) * CONTENT_W
    return Image(buf, width=CONTENT_W, height=img_h)


def comparison_table(headers: list, rows: list, col_widths: list):
    """Renders a styled comparison table."""
    header_style = ParagraphStyle(
        "th", fontName="Helvetica-Bold", fontSize=9,
        textColor=WHITE, alignment=TA_CENTER)
    cell_style = ParagraphStyle(
        "td", fontName="Helvetica", fontSize=9,
        textColor=NAVY, alignment=TA_LEFT)

    data = [[Paragraph(h, header_style) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), cell_style) for c in row])

    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("BACKGROUND",    (0, 1), (-1, -1), LIGHT_BG),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
            [LIGHT_BG, colors.HexColor("#DDEEFF")]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#BBCCDD")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def mat_image(fig, width_cm: float = 14) -> Image:
    """Converts a matplotlib figure to a ReportLab Image flowable."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_cm * cm,
                 height=width_cm * cm * 0.5)   # 2:1 aspect default


def divider(color: colors.Color = TEAL, thickness: float = 0.8):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=4, spaceBefore=4)


def spacer(h_cm: float = 0.3):
    return Spacer(1, h_cm * cm)


# ---------------------------------------------------------------------------
# CHART GENERATORS
# ---------------------------------------------------------------------------
np.random.seed(42)

def chart_statistical_learning():
    """Y = f(x) + epsilon — componentes do modelo."""
    f = lambda x: np.sin(2 * np.pi * x)
    x_all = np.linspace(0, 1, 300)
    n = 22
    x_tr = np.sort(np.random.rand(n))
    eps = np.random.normal(0, 0.2, n)
    y_tr = f(x_tr) + eps

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    ax.plot(x_all, f(x_all), color="#1A2B4A", lw=2.2,
            label=r"$f(x)$ — relação sistemática")
    ax.scatter(x_tr, y_tr, color="#0D7377", s=55, zorder=5,
               label=r"$Y = f(x) + \epsilon$ (observado)")
    for xi, yi in zip(x_tr, y_tr):
        ax.plot([xi, xi], [f(xi), yi], color="#E8534A",
                alpha=0.45, lw=1)
    ax.plot([], [], color="#E8534A", alpha=0.7,
            label=r"$\epsilon$ (ruído irredutível)")
    ax.set_title(r"Modelo: $Y = f(x) + \epsilon$", fontsize=11, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("Y")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    ax.plot(x_all, f(x_all), color="#1A2B4A", lw=2.2,
            label=r"$f(x)$ estimada")
    x_new = 0.72
    ax.scatter([x_new], [f(x_new)], color="#E8534A", s=130,
               zorder=6, label=f"Prediction: x={x_new:.2f}")
    ax.annotate("Inference:\ncomo muda Y\nquando x varia?",
                xy=(0.28, f(0.28)), xytext=(0.03, -1.2),
                arrowprops=dict(arrowstyle="->", color="#00C9A7", lw=1.5),
                color="#00C9A7", fontsize=8.5)
    ax.set_title("Prediction vs. Inference", fontsize=11, fontweight="bold")
    ax.set_xlabel("x")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def chart_linear_regression():
    """Linear regression design matrix and normal equations fit."""
    n = 30
    x = np.sort(np.random.uniform(-2, 2, n))
    y = 1.5 + 2.3 * x + np.random.normal(0, 0.5, n)
    X = np.column_stack([np.ones(n), x])
    a = np.linalg.solve(X.T @ X, X.T @ y)
    x_p = np.linspace(-2.2, 2.2, 200)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    ax.scatter(x, y, color="#0D7377", s=45, zorder=5, label="Dados")
    ax.plot(x_p, a[0] + a[1] * x_p, color="#E8534A", lw=2.5,
            label=f"$\\hat{{y}} = {a[0]:.2f} + {a[1]:.2f}x$")
    for xi, yi in zip(x, y):
        ax.plot([xi, xi], [a[0] + a[1]*xi, yi],
                color="#6C63FF", alpha=0.3, lw=1)
    ax.set_title("Regressão Linear — Normal Equations", fontsize=11,
                 fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    rows = [[r"$\mathbf{1}$", r"$x_1$"],
            [r"$\mathbf{1}$", r"$x_2$"],
            [r"$\vdots$",     r"$\vdots$"],
            [r"$\mathbf{1}$", r"$x_K$"]]
    cell_w = 1.4
    colors_grid = [["#D4E6F1", "#FDEBD0"],
                   ["#D4E6F1", "#FDEBD0"],
                   ["white",   "white"],
                   ["#D4E6F1", "#FDEBD0"]]
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            left = 0.1 + c_idx * cell_w
            bottom = 3.0 - r_idx * 0.8
            rect = plt.Rectangle((left, bottom), cell_w - 0.05, 0.7,
                                  facecolor=colors_grid[r_idx][c_idx],
                                  edgecolor="#AABBCC", linewidth=0.8)
            ax.add_patch(rect)
            ax.text(left + cell_w * 0.5, bottom + 0.35, val,
                    ha="center", va="center", fontsize=11)

    ax.text(0.1 + cell_w, 3.9, "Design Matrix X", fontsize=10,
            fontweight="bold", color="#1A2B4A", ha="center")
    ax.set_xlim(0, 3.2); ax.set_ylim(-0.2, 4.3)
    ax.axis("off")
    ax.set_title("Estrutura da Design Matrix", fontsize=11, fontweight="bold")

    plt.tight_layout()
    return fig


def chart_basis_overview():
    """Side-by-side comparison of 4 basis function types."""
    x_tr = np.sort(np.random.uniform(-1, 1, 28))
    y_tr = np.sin(3 * x_tr) + np.random.normal(0, 0.15, 28)
    x_p  = np.linspace(-1, 1, 300)
    f_t  = np.sin(3 * x_p)

    def poly_B(x, d): return np.column_stack([x**m for m in range(d+1)])
    def rbf_B(x, c, s): return np.column_stack(
        [np.exp(-(x-ci)**2/(2*s**2)) for ci in c])
    def four_B(x, k, T=2.0):
        cols = [np.ones_like(x)]
        for ki in range(1, k+1):
            cols += [np.sin(2*np.pi*ki*x/T), np.cos(2*np.pi*ki*x/T)]
        return np.column_stack(cols)

    centers = np.linspace(-1, 1, 8)
    # Build a full clamped knot vector: (k+1) boundary repeats + interior knots
    _k = 3
    _xlo, _xhi = x_tr[0], x_tr[-1]
    _interior = np.linspace(_xlo, _xhi, 8)[1:-1]
    knots_i = np.r_[(_xlo,)*(_k+1), _interior, (_xhi,)*(_k+1)]

    def fit(B_tr, B_pr): return B_pr @ np.linalg.lstsq(B_tr, y_tr, rcond=None)[0]

    configs = [
        ("Polynomial (grau 5)", "#6C63FF",
         fit(poly_B(x_tr, 5), poly_B(x_p, 5))),
        ("Gaussian RBF (8 centros)", "#F5A623",
         fit(rbf_B(x_tr, centers, 0.3), rbf_B(x_p, centers, 0.3))),
        ("Fourier (4 frequências)", "#00C9A7",
         fit(four_B(x_tr, 4), four_B(x_p, 4))),
    ]
    spl = make_lsq_spline(x_tr, y_tr, knots_i, k=_k)
    configs.append(("B-Spline (cúbico, 6 knots)", "#E8534A", spl(x_p)))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor="white")
    for ax, (title, color, y_pred) in zip(axes, configs):
        ax.set_facecolor("#FAFAFA")
        ax.scatter(x_tr, y_tr, s=28, color="#888888", zorder=5, alpha=0.8)
        ax.plot(x_p, f_t, "k--", alpha=0.35, lw=1.5)
        ax.plot(x_p, y_pred, color=color, lw=2.5, label=title)
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.set_ylim(-2, 2)
        ax.grid(alpha=0.25)
        ax.set_xlabel("x", fontsize=8)
    fig.suptitle("Comparação de Basis Functions — mesmo dado, diferentes representações",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_polynomial():
    """Polynomial basis — underfitting to overfitting."""
    x_tr = np.sort(np.random.uniform(-1, 1, 25))
    y_tr = np.sin(3 * x_tr) + np.random.normal(0, 0.15, 25)
    x_p  = np.linspace(-1.05, 1.05, 300)

    def poly_B(x, d): return np.column_stack([x**m for m in range(d+1)])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="white")
    for ax, deg, col in zip(axes, [2, 5, 12],
                             ["#6C63FF", "#0D7377", "#E8534A"]):
        B_tr = poly_B(x_tr, deg)
        a = np.linalg.lstsq(B_tr, y_tr, rcond=None)[0]
        y_p = poly_B(x_p, deg) @ a
        ax.set_facecolor("#FAFAFA")
        ax.scatter(x_tr, y_tr, s=38, color="#888888", zorder=5, alpha=0.8)
        ax.plot(x_p, np.sin(3*x_p), "k--", alpha=0.4, lw=1.5)
        ax.plot(x_p, y_p, color=col, lw=2.5, label=f"Grau {deg}")
        ax.set_title(f"Polynomial Basis — Grau {deg}", fontsize=10,
                     fontweight="bold")
        ax.set_ylim(-2.5, 2.5); ax.grid(alpha=0.25)
        ax.set_xlabel("x", fontsize=9)
    plt.suptitle("Grau baixo → underfitting  |  Grau alto → overfitting (Runge's phenomenon)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_rbf():
    """RBF basis functions and sigma effect."""
    centers = np.linspace(-1, 1, 7)
    x_vis   = np.linspace(-1.3, 1.3, 300)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    cmap = plt.cm.tab10
    for i, c in enumerate(centers):
        rbf = np.exp(-(x_vis - c)**2 / (2 * 0.28**2))
        ax.plot(x_vis, rbf, color=cmap(i/7), lw=2,
                label=f"c={c:.2f}")
        ax.axvline(c, color=cmap(i/7), lw=0.6, ls="--", alpha=0.5)
    ax.set_title("Gaussian RBF Basis Functions (σ=0.28)", fontsize=10,
                 fontweight="bold")
    ax.set_xlabel("x"); ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)

    x_tr = np.sort(np.random.uniform(-1, 1, 28))
    y_tr = np.sin(3 * x_tr) + np.random.normal(0, 0.15, 28)
    x_p  = np.linspace(-1.1, 1.1, 300)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    for sig, col in [(0.1, "#E8534A"), (0.3, "#0D7377"), (1.0, "#6C63FF")]:
        def rbfB(x):
            return np.column_stack([np.exp(-(x-c)**2/(2*sig**2)) for c in centers])
        a = np.linalg.lstsq(rbfB(x_tr), y_tr, rcond=None)[0]
        ax.plot(x_p, rbfB(x_p) @ a, color=col, lw=2.2,
                label=f"σ={sig}")
    ax.scatter(x_tr, y_tr, s=28, color="#888", zorder=5, alpha=0.7)
    ax.plot(x_p, np.sin(3*x_p), "k--", alpha=0.4, lw=1.5, label="f verdadeira")
    ax.set_title("Efeito do Width (σ) na RBF", fontsize=10, fontweight="bold")
    ax.set_xlabel("x"); ax.legend(fontsize=9)
    ax.set_ylim(-2.3, 2.3); ax.grid(alpha=0.25)

    plt.tight_layout()
    return fig


def chart_fourier():
    """Fourier basis functions and fit quality."""
    T  = 1.0
    x_per = np.sort(np.random.uniform(0, T, 40))
    f_per = (lambda x: 0.5*np.sin(2*np.pi*x)
                     + 0.3*np.cos(4*np.pi*x)
                     + 0.2*np.sin(6*np.pi*x))
    y_per = f_per(x_per) + np.random.normal(0, 0.08, 40)
    x_p   = np.linspace(0, T, 400)

    def four_B(x, k):
        cols = [np.ones_like(x)]
        for ki in range(1, k+1):
            cols += [np.sin(2*np.pi*ki*x/T), np.cos(2*np.pi*ki*x/T)]
        return np.column_stack(cols)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="white")

    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    styles_ls = ["-", "--", ":", "-.", "-"]
    for k in range(1, 4):
        ax.plot(x_p, np.sin(2*np.pi*k*x_p),
                ls=styles_ls[k-1], color="#0D7377", lw=1.6,
                label=f"sin({2*k}πx)")
        ax.plot(x_p, np.cos(2*np.pi*k*x_p),
                ls=styles_ls[k-1], color="#E8534A", lw=1.6,
                label=f"cos({2*k}πx)")
    ax.set_title("Fourier Basis (3 frequências)", fontsize=10, fontweight="bold")
    ax.set_xlabel("x"); ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    for n_f, col in [(1, "#6C63FF"), (3, "#F5A623"), (6, "#0D7377")]:
        B_tr = four_B(x_per, n_f)
        a    = np.linalg.lstsq(B_tr, y_per, rcond=None)[0]
        ax.plot(x_p, four_B(x_p, n_f) @ a, color=col, lw=2.2,
                label=f"{n_f} freq.")
    ax.scatter(x_per, y_per, s=28, color="#888", zorder=5, alpha=0.7)
    ax.plot(x_p, f_per(x_p), "k--", alpha=0.5, lw=1.5, label="f verdadeira")
    ax.set_title("Ajuste Fourier — mais frequências = mais detalhes",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("x"); ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    return fig


def chart_bspline():
    """B-Spline: basis functions, partition of unity, compact support."""
    from scipy.interpolate import make_lsq_spline, BSpline as BS

    def cox_de_boor(x, knots, m, p):
        t = knots
        if p == 0:
            return np.where((t[m] <= x) & (x < t[m+1]), 1.0, 0.0)
        L = np.zeros_like(x, dtype=float)
        R = np.zeros_like(x, dtype=float)
        dl = t[m+p] - t[m]
        if dl > 0:
            L = (x - t[m]) / dl * cox_de_boor(x, knots, m, p-1)
        dr = t[m+p+1] - t[m+1]
        if dr > 0:
            R = (t[m+p+1] - x) / dr * cox_de_boor(x, knots, m+1, p-1)
        return L + R

    grau = 3
    ki   = np.linspace(0, 1, 6)
    knots = np.concatenate([[0]*grau, ki, [1]*grau])
    n_b   = len(knots) - grau - 1
    x_vis = np.linspace(0, 1 - 1e-9, 400)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor="white")

    # Basis functions
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    cmap = plt.cm.tab10
    soma = np.zeros_like(x_vis)
    for m in range(n_b):
        bm = cox_de_boor(x_vis, knots, m, grau)
        ax.plot(x_vis, bm, lw=2, color=cmap(m/n_b), label=f"$B_{{{m},3}}$")
        soma += bm
    for k in ki[1:-1]:
        ax.axvline(k, color="#AAAAAA", lw=0.7, ls="--")
    ax.set_title(f"B-Spline Basis (grau {grau}, {n_b} bases)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("x"); ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.2)

    # Partition of Unity
    ax = axes[1]
    ax.set_facecolor("#FAFAFA")
    ax.plot(x_vis, soma, color="#1A2B4A", lw=3,
            label="$\\sum_m B_{m,3}(x)$")
    ax.axhline(1, color="#E8534A", ls="--", lw=1.5,
               label="Esperado = 1")
    ax.set_ylim(0, 1.35)
    ax.set_title("Partition of Unity", fontsize=10, fontweight="bold")
    ax.set_xlabel("x"); ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Compact support
    ax = axes[2]
    ax.set_facecolor("#FAFAFA")
    x_cs = np.sort(np.random.uniform(0, 1, 50))
    y_cs = np.sin(2*np.pi*x_cs) + np.random.normal(0, 0.1, 50)
    _k3 = 3
    _t_cs = np.r_[(x_cs[0],)*(_k3+1),
                  np.linspace(x_cs[0], x_cs[-1], 10)[1:-1],
                  (x_cs[-1],)*(_k3+1)]
    spl  = make_lsq_spline(x_cs, y_cs, _t_cs, k=_k3)
    x_sp = np.linspace(0, 1, 400)
    c_mod = spl.c.copy()
    idx   = len(c_mod) // 2
    c_mod[idx] += 1.0
    spl2  = BS(spl.t, c_mod, spl.k)
    ax.scatter(x_cs, y_cs, s=22, color="#888", alpha=0.7, zorder=5)
    ax.plot(x_sp, spl(x_sp), color="#0D7377", lw=2, ls="--",
            alpha=0.7, label="Original")
    ax.plot(x_sp, spl2(x_sp), color="#E8534A", lw=2.5,
            label=f"Coef. {idx} +1.0")
    ax.axvspan(spl.t[idx], spl.t[idx+spl.k+1],
               alpha=0.12, color="#E8534A", label="Região afetada")
    ax.set_title("Compact Support", fontsize=10, fontweight="bold")
    ax.set_xlabel("x"); ax.legend(fontsize=8.5)
    ax.grid(alpha=0.2)

    plt.suptitle("B-Splines: basis functions, Partition of Unity, e Compact Support",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# COVER FLOWABLE
# ---------------------------------------------------------------------------
class CoverFlowable:
    """Placeholder that triggers cover canvas drawing — zero-height flowable."""
    def __init__(self): pass
    def wrap(self, aw, ah): return (aw, 0)
    def draw(self): pass


# ---------------------------------------------------------------------------
# CONTENT BUILDER
# ---------------------------------------------------------------------------
def build_content(styles: dict) -> list:
    S = styles
    story = []

    # ------------------------------------------------------------------ cover
    story.append(PageBreak())  # blank; canvas handles the cover drawing

    # ------------------------------------------------------------------ page 1 — intro
    story.append(spacer(0.4))
    story.append(Paragraph(
        "Bem-vindo à Aula 1 — General Linear Models & Basis Functions",
        S["title"]))
    story.append(divider(GOLD, 1.5))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Esta aula é o ponto de partida de todo o curso. Você aprenderá a formalizar "
        "o problema de Aprendizado Estatístico, construir modelos que vão além da "
        "reta simples usando <b>Basis Functions</b>, e resolver tudo com álgebra linear "
        "pura. Ao final, você dominará uma linguagem matemática unificada que serve de "
        "base para regularização, bias-variance tradeoff, e muito mais.",
        S["body"]))
    story.append(spacer(0.3))

    # Roteiro
    story.append(Paragraph("Roteiro desta aula:", S["subsection"]))
    for item in [
        "1. O Problema do Aprendizado Estatístico — Y = f(x) + epsilon",
        "2. Prediction vs. Inference — dois objetivos completamente diferentes",
        "3. Regressão Linear — forma matricial e a Design Matrix",
        "4. Normal Equations — solução analítica exata",
        "5. Basis Functions — o truque não-linear que mantém a álgebra simples",
        "6. Polynomial Basis — potências de x, underfitting e overfitting",
        "7. Gaussian RBF Basis — sinos locais com center e width",
        "8. Fourier Basis — para dados periódicos",
        "9. B-Splines — Compact Support, Partition of Unity, Cox-de Boor",
    ]:
        story.append(Paragraph(
            f"<bullet>•</bullet> {item}", S["bullet"]))
    story.append(PageBreak())

    # ================================================================ SECTION 1
    story.append(section_header("1. O Problema do Aprendizado Estatístico", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Imagine que você quer prever o preço de uma casa a partir de sua área. "
        "Você tem dados — pares (área, preço) — mas o mundo real é bagunçado: "
        "duas casas com a mesma área podem ter preços diferentes por fatores "
        "que você não mediu. O <b>Statistical Learning Problem</b> formaliza isso "
        "de forma elegante: existe uma relação <b>sistemática</b> entre entrada e "
        "saída, mais um <b>ruído irredutível</b>.",
        S["body"]))
    story.append(spacer(0.25))

    story.append(Paragraph("A Equação Fundamental:", S["subsection"]))
    story.append(formula_block(
        r"Y = f(\mathbf{x}) + \epsilon",
        label="Y: variável alvo  |  x: vetor de features  |"
              "  f: função desconhecida  |  ε: ruído com média zero"))
    story.append(spacer(0.2))

    story.append(Paragraph(
        "O símbolo <b>f</b> representa toda a <i>informação sistemática</i> que "
        "<b>x</b> carrega sobre <b>Y</b>. Nosso trabalho é <b>estimar f</b> a partir "
        "de dados observados. O ruído <b>epsilon</b> é irredutível — não importa quão bom "
        "seja o modelo, ele sempre estará presente.",
        S["body"]))

    tbl = comparison_table(
        ["Símbolo", "Nome", "Papel no modelo"],
        [
            ["Y", "Variável alvo (target)", "O que queremos prever ou entender"],
            ["x = (x1, …, xp)", "Vetor de features", "Informação disponível (entrada)"],
            ["f", "Função sistemática", "Relação real entre x e Y (desconhecida)"],
            ["epsilon", "Ruído irredutível", "Variação que não depende de x"],
        ],
        [3.5*cm, 4.5*cm, CONTENT_W - 8*cm])
    story.append(tbl)
    story.append(spacer(0.2))

    fig1 = chart_statistical_learning()
    img1 = mat_image(fig1, width_cm=15)
    story.append(img1)
    story.append(Paragraph(
        "Figura 1 — Esquerda: componentes do modelo Y = f(x) + epsilon. "
        "Direita: distinção entre Prediction (ponto vermelho) e Inference (seta verde).",
        S["caption"]))
    story.append(spacer(0.2))

    story.append(why_box(
        "Todo problema de Machine Learning começa aqui. Entender que os dados têm "
        "estrutura (f) + aleatoriedade (epsilon) é o que separa uma análise ingênua de "
        "uma abordagem científica. Sem essa formalização, você não consegue "
        "quantificar incerteza, comparar modelos, ou entender o que é possível prever.",
        "Na prática, essa distinção guia a escolha do modelo: para Prediction, "
        "modelos complexos (Random Forests, Neural Networks) são aceitáveis mesmo "
        "sem interpretabilidade. Para Inference, modelos lineares simples ou GAMs "
        "(Generalized Additive Models) são preferidos porque permitem ler os "
        "coeficientes como efeitos causais.",
        S))
    story.append(PageBreak())

    # ================================================================ SECTION 2
    story.append(section_header("2. Prediction vs. Inference", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Há dois motivos completamente diferentes para estimar f — e confundi-los "
        "leva a escolhas de modelo erradas. <b>Prediction</b> quer apenas que "
        "f̂(x) ≈ Y; a estrutura interna não importa. <b>Inference</b> quer "
        "entender <i>como</i> Y muda quando xᵢ aumenta em uma unidade — e para isso "
        "o modelo precisa ser interpretável.",
        S["body"]))
    story.append(spacer(0.25))

    tbl2 = comparison_table(
        ["Objetivo", "Pergunta central", "Exemplo real", "Modelo típico"],
        [
            ["Prediction",
             "Dado x novo, qual será Y?",
             "Prever se um e-mail é spam",
             "Random Forest, Neural Network"],
            ["Inference",
             "Como Y muda quando xᵢ aumenta?",
             "Efeito de 1 ano de educação no salário",
             "Regressão Linear, GAM"],
        ],
        [2.5*cm, 4.0*cm, 4.5*cm, CONTENT_W - 11*cm])
    story.append(tbl2)
    story.append(spacer(0.25))

    story.append(Paragraph(
        "<b>Insight crítico:</b> para Prediction, a caixa-preta pode ser opaca. "
        "Para Inference, precisamos saber exatamente o que cada coeficiente significa. "
        "General Linear Models são ótimos para Inference porque os coeficientes têm "
        "interpretação direta.",
        S["highlight"]))
    story.append(spacer(0.25))

    story.append(why_box(
        "Essa distinção é fundamental no dia-a-dia de um Data Scientist. "
        "Quando um médico quer saber se um medicamento causa melhora (Inference), "
        "ele precisa de um modelo interpretável — não apenas de uma previsão acurada. "
        "Quando uma Netflix quer recomendar um filme (Prediction), interpretabilidade "
        "é secundária; acurácia é o que importa.",
        "Antes de escolher um algoritmo, sempre se pergunte: o cliente quer "
        "uma <i>previsão</i> ou quer <i>entender</i> o fenômeno? Isso determina "
        "se você usa sklearn's RandomForestRegressor (Prediction) ou "
        "statsmodels.OLS com coeficientes e p-values (Inference).",
        S, color=LAVENDER))
    story.append(PageBreak())

    # ================================================================ SECTION 3
    story.append(section_header("3. Regressão Linear — Forma Matricial", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "A <b>Regressão Linear</b> assume que f é linear nos parâmetros: a saída y "
        "é uma combinação ponderada das features. Pense como uma receita — o preço "
        "de uma casa é R$ X por m² mais R$ Y por quarto, mais um valor base.",
        S["body"]))
    story.append(spacer(0.2))

    story.append(Paragraph("Forma escalar (para uma observação j):", S["subsection"]))
    story.append(formula_block(
        r"y_j = a_0 + a_1 x_j + a_2 x_j^2 + \cdots + a_n x_j^n + \epsilon_j",
        label="Cada observação j tem n features; a₀ é o intercepto (bias)",
        bg_hex="#0D3B6E"))
    story.append(spacer(0.2))

    story.append(Paragraph("A Design Matrix — forma compacta:", S["subsection"]))
    story.append(Paragraph(
        "Adicionamos uma coluna artificial de 1s para absorver o intercepto a₀. "
        "Com K observações e n+1 coeficientes, empilhamos tudo em uma única "
        "equação matricial:",
        S["body"]))
    story.append(formula_block(
        r"\mathbf{y} = X\,\mathbf{a} + \epsilon",
        label="y ∈ R^K (targets)  |  X ∈ R^(K × n+1) (Design Matrix)  |  a ∈ R^(n+1) (coeficientes)"))
    story.append(spacer(0.2))

    story.append(Paragraph(
        "A Design Matrix X tem uma <b>linha por observação</b> e uma <b>coluna "
        "por coeficiente</b>. A primeira coluna é sempre 1 (para o intercepto):",
        S["body"]))
    story.append(spacer(0.15))

    fig2 = chart_linear_regression()
    img2 = mat_image(fig2, width_cm=15)
    story.append(img2)
    story.append(Paragraph(
        "Figura 2 — Esquerda: ajuste linear com Normal Equations (resíduos em roxo). "
        "Direita: estrutura da Design Matrix (coluna de 1s + coluna de features).",
        S["caption"]))

    story.append(why_box(
        "A forma matricial generaliza para qualquer número de features sem mudar "
        "a equação. Com 1 feature ou com 1000 features, o modelo ainda é "
        "y = X·a + epsilon. Isso torna o código extremamente genérico — você escreve "
        "uma vez e funciona para qualquer dimensão.",
        "Em Python, você construiu a Design Matrix com np.column_stack "
        "([np.ones(n), x]). Em sklearn, o PolynomialFeatures faz isso "
        "automaticamente para você. Entender a estrutura interna permite "
        "depurar bugs, implementar variantes customizadas, e entender o que "
        "o modelo realmente está otimizando.",
        S, color=CORAL))
    story.append(PageBreak())

    # ================================================================ SECTION 4
    story.append(section_header("4. Normal Equations — Solução Analítica Exata", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Queremos encontrar os coeficientes â que minimizem a <b>soma dos erros "
        "ao quadrado</b> (Least Squares): ||y - X·a||<super>2</super>. A derivação é "
        "surpreendentemente elegante: derivamos em relação a a, igualamos a zero, "
        "e obtemos a solução analítica exata.",
        S["body"]))
    story.append(spacer(0.2))

    story.append(Paragraph("Derivação das Normal Equations:", S["subsection"]))
    for step in [
        "Minimizar: L(a) = ||y - Xa||<super>2</super> = (y - Xa)<super>T</super>(y - Xa)",
        "Derivar: dL/da = -2X<super>T</super>(y - Xa) = 0",
        "Rearranjar: X<super>T</super>Xa = X<super>T</super>y  →  (sistema linear em a)",
        "Solução: â = (X<super>T</super>X)<super>-1</super> X<super>T</super>y",
    ]:
        story.append(Paragraph(f"<bullet>→</bullet> {step}", S["bullet"]))
    story.append(spacer(0.15))

    story.append(formula_block(
        r"\hat{a} = (X^T X)^{-1} X^T \mathbf{y}",
        label="Esta é a solução analítica exata — não uma aproximação iterativa!",
        bg_hex="#0A1628", formula_size=17))
    story.append(spacer(0.2))

    story.append(Paragraph("np.linalg.solve vs. np.linalg.inv:", S["subsection"]))
    tbl3 = comparison_table(
        ["Abordagem", "Código Python", "Estabilidade"],
        [
            ["Inversão explícita (ruim)",
             "a = np.linalg.inv(X.T@X) @ X.T @ y",
             "Baixa — amplifica erros numéricos"],
            ["solve (recomendado)",
             "a = np.linalg.solve(X.T@X, X.T@y)",
             "Alta — usa fatoração LU/Cholesky"],
            ["lstsq (mais robusto)",
             "a = np.linalg.lstsq(X, y)[0]",
             "Máxima — usa SVD, lida com rank-deficient"],
        ],
        [3.5*cm, 5.5*cm, CONTENT_W - 9*cm])
    story.append(tbl3)
    story.append(spacer(0.2))

    story.append(why_box(
        "A solução analítica exata é uma vantagem enorme: não há hiperparâmetros "
        "de otimização, não há convergência, não há learning rate. Para datasets "
        "de tamanho moderado, é sempre preferível ao Gradient Descent porque "
        "garante o mínimo global em uma única operação.",
        "Sempre use np.linalg.solve(X.T@X, X.T@y) em vez de "
        "np.linalg.inv(X.T@X) @ X.T@y. A inversão explícita acumula erros "
        "de ponto flutuante que podem tornar os coeficientes instáveis quando "
        "X^T X está próxima de ser singular (o que acontece com grau polinomial "
        "alto ou features colineares).",
        S, color=MINT))
    story.append(PageBreak())

    # ================================================================ SECTION 5
    story.append(section_header("5. Basis Functions — O Truque Não-Linear", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "E se os dados não forem lineares? A <b>Basis Functions trick</b> resolve "
        "isso de forma genial: em vez de aplicar os coeficientes diretamente em x, "
        "primeiro <b>transformamos</b> x com funções B_m(x). O modelo continua "
        "<b>linear nos coeficientes</b> — só as features mudam!",
        S["body"]))
    story.append(Paragraph(
        "<b>Analogia:</b> pense em origami. Você dobra uma folha plana (transforma "
        "os dados), e dobras simples criam formas complexas. As dobras são as Basis "
        "Functions. O processo de encontrar os pesos é linear; a transformação "
        "é que traz a não-linearidade.",
        S["body"]))
    story.append(spacer(0.2))

    story.append(Paragraph("O modelo com M Basis Functions:", S["subsection"]))
    story.append(formula_block(
        r"y_j = \sum_{m} a_m B_m(x_j) + \epsilon_j \;\Rightarrow\; \mathbf{y} = B\,\mathbf{a} + \epsilon",
        label="B ∈ R^(K×M) (Basis Matrix)  |  B_jm = B_m(x_j)  |"
              "  â = (BᵀB)⁻¹Bᵀy  (mesma fórmula das Normal Equations!)",
        formula_size=13))
    story.append(spacer(0.2))

    story.append(Paragraph(
        "<b>Por que funciona?</b> Os coeficientes a_m multiplicam as funções B_m, "
        "e a_m entram de forma <i>linear</i>. A não-linearidade está em B_m(x), "
        "mas o sistema de equações para encontrar a permanece linear — logo, "
        "a solução analítica exata via Normal Equations ainda funciona.",
        S["highlight"]))
    story.append(spacer(0.2))

    fig_overview = chart_basis_overview()
    img_overview = mat_image(fig_overview, width_cm=16)
    story.append(img_overview)
    story.append(Paragraph(
        "Figura 3 — Quatro tipos de Basis Functions aplicadas ao mesmo dataset: "
        "Polynomial, RBF, Fourier e B-Spline. Mesma fórmula â = (B^T B)^(-1) B^T y para todas.",
        S["caption"]))

    story.append(why_box(
        "General Linear Models (GLMs) são 'lineares' em um sentido muito específico: "
        "linear nos parâmetros a, não nas features x. Isso é o que permite usar "
        "toda a teoria de regressão linear (Normal Equations, intervalos de confiança, "
        "testes de hipótese) mesmo com modelos não-lineares nas features.",
        "No pipeline sklearn, PolynomialFeatures().fit_transform(X) constrói "
        "exatamente a Basis Matrix B. Depois, LinearRegression().fit(B, y) resolve "
        "as Normal Equations. Todo pipeline não-linear do sklearn funciona nesse "
        "paradigma: transforme primeiro, regresse depois.",
        S, color=TEAL))
    story.append(PageBreak())

    # ================================================================ SECTION 6
    story.append(section_header("6. Polynomial Basis", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "A <b>Polynomial Basis</b> é a mais intuitiva: usamos potências de x como "
        "funções de base. Grau 1 é a reta clássica; grau 2 adiciona curvatura; "
        "graus altos capturam oscilações cada vez mais finas.",
        S["body"]))
    story.append(spacer(0.15))

    story.append(formula_block(
        r"B_m(x) = x^m \qquad m = 0,\;1,\;2,\;\ldots,\;M",
        label="Cada coluna de B_poly = [1, x, x², …, xᴹ] — uma potência por coluna",
        bg_hex="#2D1B6E"))
    story.append(spacer(0.2))

    story.append(Paragraph("Construção em Python:", S["subsection"]))
    story.append(Paragraph(
        "def polynomial_basis_matrix(x, degree):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;return np.column_stack([x**m for m in range(degree + 1)])<br/>"
        "<br/>"
        "# Uma list comprehension cria cada coluna x^m e column_stack as empilha",
        S["code"]))
    story.append(spacer(0.2))

    fig_poly = chart_polynomial()
    img_poly = mat_image(fig_poly, width_cm=15)
    story.append(img_poly)
    story.append(Paragraph(
        "Figura 4 — Polynomial Basis: grau 2 (underfitting), grau 5 (bom equilíbrio), "
        "grau 12 (overfitting — Runge's phenomenon nas bordas).",
        S["caption"]))
    story.append(spacer(0.2))

    tbl4 = comparison_table(
        ["Propriedade", "Polynomial Basis"],
        [
            ["Vantagem", "Simples de construir; solução analítica disponível"],
            ["Desvantagem", "Graus altos → overfitting e oscilações nas bordas"],
            ["Localidade", "NENHUMA — mudar um ponto afeta o polinômio inteiro"],
            ["sklearn equivalente", "PolynomialFeatures(degree=M)"],
        ],
        [4*cm, CONTENT_W - 4*cm])
    story.append(tbl4)

    story.append(why_box(
        "A Polynomial Basis é o primeiro passo conceitual, mas raramente a escolha "
        "final em produção. O fenômeno de Runge (oscilações nas bordas com grau "
        "alto) é um problema clássico que motiva o uso de Splines — que são "
        "polinômios locais e evitam o problema.",
        "No sklearn: from sklearn.preprocessing import PolynomialFeatures; "
        "poly = PolynomialFeatures(degree=5); X_poly = poly.fit_transform(X). "
        "Combine com Ridge para regularizar os coeficientes de alta potência, "
        "que tendem a ser muito grandes e instáveis.",
        S, color=LAVENDER))
    story.append(PageBreak())

    # ================================================================ SECTION 7
    story.append(section_header("7. Gaussian RBF Basis (Radial Basis Functions)", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Imagine colocar <b>sinos gaussianos</b> espalhados pelo eixo x. O modelo "
        "é a soma ponderada desses sinos. Cada sino 'ativa' quando x está próximo "
        "de seu centro. Isso resolve o problema de <b>localidade</b> dos polinômios.",
        S["body"]))
    story.append(Paragraph(
        "<b>Analogia:</b> pense em um time de especialistas. Cada especialista (RBF) "
        "domina uma região do espaço de entrada e contribui com sua expertise quando "
        "o ponto de consulta está na sua vizinhança. Os coeficientes a_m dizem "
        "o quanto cada especialista contribui.",
        S["body"]))
    story.append(spacer(0.15))

    story.append(formula_block(
        r"B_m(x) = \exp\!\left(-\,\frac{(x - c_m)^2}{2\sigma^2}\right)",
        label="c_m: center (onde o sino está centrado)  |  σ: width (largura do sino)",
        bg_hex="#4A1A2A", formula_size=16))
    story.append(spacer(0.2))

    fig_rbf = chart_rbf()
    img_rbf = mat_image(fig_rbf, width_cm=15)
    story.append(img_rbf)
    story.append(Paragraph(
        "Figura 5 — Esquerda: 7 Gaussian RBFs centradas em posições equidistantes. "
        "Direita: efeito do width sigma — pequeno overfita, grande underfita.",
        S["caption"]))
    story.append(spacer(0.2))

    tbl5 = comparison_table(
        ["Parâmetro", "Valor pequeno", "Valor grande"],
        [
            ["sigma (width)", "Sinos estreitos → muito localizado → overfitting",
             "Sinos largos → suave → underfitting"],
            ["M (# centros)", "Poucas bases → underfitting",
             "Muitas bases → overfitting (sem regularização)"],
        ],
        [3*cm, (CONTENT_W-3*cm)/2, (CONTENT_W-3*cm)/2])
    story.append(tbl5)
    story.append(spacer(0.2))

    story.append(why_box(
        "RBFs são a base dos Support Vector Machines com kernel RBF "
        "(sklearn's SVC/SVR com kernel='rbf'). Também são fundamentais em "
        "redes neurais RBF e em Gaussian Processes. Entender o papel de center "
        "e width é entender o hiperparâmetro gamma = 1/(2*sigma^2) do SVM.",
        "No sklearn: sklearn.svm.SVR(kernel='rbf', gamma=0.5). "
        "O gamma controla exatamente sigma. Ao fazer Grid Search, você está otimizando "
        "a largura dos sinos RBF implícitos no kernel. sigma pequeno = "
        "modelo complexo = mais risco de overfitting.",
        S, color=GOLD))
    story.append(PageBreak())

    # ================================================================ SECTION 8
    story.append(section_header("8. Fourier Basis", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Senos e cossenos são as basis functions naturais para <b>dados periódicos</b>. "
        "Som, ondas de rádio, temperaturas sazonais — todos têm periodicidade. "
        "A <b>Fourier Basis</b> decompõe qualquer sinal periódico em suas frequências "
        "constituintes.",
        S["body"]))
    story.append(Paragraph(
        "<b>Analogia:</b> qualquer música pode ser decomposta em notas puras. "
        "Cada nota é uma frequência, e a música é a soma ponderada dessas notas. "
        "A Fourier Basis faz exatamente isso — os coeficientes a_m dizem "
        "quão forte é cada frequência no sinal.",
        S["body"]))
    story.append(spacer(0.15))

    story.append(formula_block(
        r"B_{2k-1}(x)=\sin\!\left(\frac{2\pi k x}{T}\right),\quad"
        r"B_{2k}(x)=\cos\!\left(\frac{2\pi k x}{T}\right)",
        label="k = 1, 2, …, K frequências  |  T = período do sinal",
        bg_hex="#1A3A2A", formula_size=13))
    story.append(spacer(0.2))

    fig_four = chart_fourier()
    img_four = mat_image(fig_four, width_cm=15)
    story.append(img_four)
    story.append(Paragraph(
        "Figura 6 — Esquerda: basis functions de Fourier (senos azuis, cossenos vermelhos). "
        "Direita: mais frequências capturam padrões mais finos no sinal periódico.",
        S["caption"]))
    story.append(spacer(0.2))

    story.append(Paragraph("Construção em Python:", S["subsection"]))
    story.append(Paragraph(
        "def fourier_basis_matrix(x, n_freqs, periodo=1.0):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;cols = [np.ones_like(x)]  # termo constante a0<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;for k in range(1, n_freqs + 1):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cols.append(np.sin(2*np.pi*k*x/periodo))<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cols.append(np.cos(2*np.pi*k*x/periodo))<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;return np.column_stack(cols)",
        S["code"]))

    story.append(why_box(
        "Fourier Analysis é a espinha dorsal do processamento de sinais, compressão "
        "de imagens (JPEG usa DCT — uma variante de Fourier), análise de séries "
        "temporais sazonais, e até de redes neurais modernas (Fourier Neural "
        "Operators para PDEs). Para dados com periodicidade clara, é a "
        "representação mais eficiente que existe.",
        "Para séries temporais sazonais (vendas mensais, temperatura diária), "
        "adicione features de Fourier ao seu pipeline antes de treinar qualquer "
        "modelo. No Prophet (biblioteca de forecasting do Facebook), as "
        "sazonalidades são modeladas exatamente com Fourier Basis. "
        "Você pode reproduzir esse comportamento manualmente com fourier_basis_matrix.",
        S, color=MINT))
    story.append(PageBreak())

    # ================================================================ SECTION 9
    story.append(section_header("9. B-Splines — Compact Support e Cox-de Boor", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "<b>B-Splines</b> (Basis Splines) resolvem os problemas dos polinômios "
        "globais de forma elegante. A ideia: dividir o eixo x em segmentos com "
        "pontos de divisão chamados <b>Knots</b>, e ajustar polinômios separados "
        "em cada segmento, garantindo suavidade nas junções.",
        S["body"]))
    story.append(Paragraph(
        "<b>Analogia perfeita:</b> é como montar um caminho em curva com peças de "
        "trilho de trem. Cada peça é levemente curvada, mas as junções são suaves "
        "— você não sente o impacto ao passar de uma para outra. Os Knots são "
        "os pontos de junção.",
        S["body"]))
    story.append(spacer(0.2))

    story.append(Paragraph("As duas propriedades fundamentais:", S["subsection"]))
    for prop in [
        "<b>Compact Support:</b> cada B_{m,p}(x) é exatamente zero fora de um "
        "intervalo pequeno. Mudar a_m afeta <i>apenas a região local</i> — ao "
        "contrário de polinômios globais que afetam o gráfico inteiro.",
        "<b>Partition of Unity:</b> em qualquer ponto x, as B-Splines ativas "
        "somam exatamente 1: sum_m B_{m,p}(x) = 1. Isso garante estabilidade "
        "numérica e que os coeficientes têm interpretação de pesos.",
    ]:
        story.append(Paragraph(f"<bullet>•</bullet> {prop}", S["bullet"]))
    story.append(spacer(0.2))

    story.append(Paragraph("Fórmula de Cox-de Boor (recursiva):", S["subsection"]))
    story.append(formula_block(
        r"B_{m,0}(x) = 1 \;\mathrm{se}\; t_m \leq x < t_{m+1},\quad 0 \;\mathrm{caso}\;\mathrm{contrário}",
        label="Caso base: funções degrau constantes (grau 0)",
        bg_hex="#2A1A0A", formula_size=12))
    story.append(spacer(0.15))
    story.append(formula_block(
        r"B_{m,p}(x)=\frac{x-t_m}{t_{m+p}-t_m}\,B_{m,p-1}(x)"
        r"+\frac{t_{m+p+1}-x}{t_{m+p+1}-t_{m+1}}\,B_{m+1,p-1}(x)",
        label="Recursão de grau p: combinação convexa de duas bases de grau p-1",
        bg_hex="#2A1A0A", formula_size=11))
    story.append(spacer(0.2))

    fig_bs = chart_bspline()
    img_bs = mat_image(fig_bs, width_cm=16)
    story.append(img_bs)
    story.append(Paragraph(
        "Figura 7 — Esquerda: B-Spline basis functions (grau 3). Centro: Partition of Unity "
        "(soma = 1 em todo ponto). Direita: Compact Support — perturbar um coeficiente "
        "afeta apenas a região local (área vermelha).",
        S["caption"]))
    story.append(spacer(0.2))

    tbl6 = comparison_table(
        ["Propriedade", "Polynomial", "RBF", "B-Spline"],
        [
            ["Compact Support", "Não", "Aproximado", "Sim (exato)"],
            ["Partition of Unity", "Não", "Não", "Sim"],
            ["Suavidade controlável", "Não", "Indiretamente", "Sim (pelo grau)"],
            ["Estabilidade numérica", "Baixa", "Média", "Alta"],
            ["sklearn/scipy", "PolynomialFeatures", "RBFSampler", "make_lsq_spline"],
        ],
        [4.5*cm, 3.0*cm, 3.0*cm, CONTENT_W - 10.5*cm])
    story.append(tbl6)
    story.append(spacer(0.2))

    story.append(Paragraph("Uso com scipy:", S["subsection"]))
    story.append(Paragraph(
        "from scipy.interpolate import make_lsq_spline, BSpline<br/>"
        "<br/>"
        "knots_internos = np.linspace(0.1, 0.9, 8)<br/>"
        "spl = make_lsq_spline(x_treino, y_treino, knots_internos, k=3)<br/>"
        "y_pred = spl(x_novo)  # avaliação vetorizada",
        S["code"]))

    story.append(why_box(
        "B-Splines são o padrão-ouro em modelagem de curvas suaves. Toda vez que "
        "você vê uma curva bonita em gráficos científicos (spline interpolation, "
        "smoothing splines), provavelmente são B-Splines por baixo. Também são "
        "a base dos Generalized Additive Models (GAMs), onde cada feature tem "
        "uma spline independente.",
        "Em Python, scipy.interpolate.make_lsq_spline é a implementação "
        "estável e eficiente. Para GAMs completos (com regularização automática "
        "de suavidade), use a biblioteca pygam: from pygam import LinearGAM. "
        "GAMs são cada vez mais populares por combinar flexibilidade de B-Splines "
        "com interpretabilidade (você pode plotar o efeito de cada feature separadamente).",
        S, color=NAVY))
    story.append(PageBreak())

    # ================================================================ SECTION 10 — Resumo
    story.append(section_header("10. Resumo & Pipeline Unificado", S))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Independentemente do tipo de Basis Function escolhida, o pipeline de "
        "ajuste é sempre o mesmo. A escolha da basis é uma decisão de design; "
        "a matemática de solução é universal.",
        S["body"]))
    story.append(spacer(0.15))

    story.append(formula_block(
        r"\hat{a} = (B^T B)^{-1} B^T \mathbf{y}",
        label="Passo 1: escolha as Basis Functions  "
              "→  Passo 2: construa B  "
              "→  Passo 3: resolva as Normal Equations",
        formula_size=17))
    story.append(spacer(0.2))

    tbl7 = comparison_table(
        ["Conceito", "Ideia Central", "Equação Chave"],
        [
            ["Statistical Learning", "Estrutura + ruído", "Y = f(x) + epsilon"],
            ["Linear Regression", "Linear nos coeficientes", "y = Xa + epsilon"],
            ["Normal Equations", "Solução analítica exata",
             "a_hat = (X^T X)^(-1) X^T y"],
            ["Basis Functions", "Truque não-linear",
             "y_j = sum_m a_m B_m(x_j)"],
            ["Polynomial Basis", "Potências de x", "B_m(x) = x^m"],
            ["Gaussian RBF", "Sinos locais",
             "B_m(x) = exp(-(x-c_m)^2 / 2*sigma^2)"],
            ["Fourier Basis", "Frequências periódicas",
             "sin(2*pi*k*x/T),  cos(2*pi*k*x/T)"],
            ["B-Splines", "Suporte compacto", "Cox-de Boor recursivo"],
        ],
        [4*cm, 5*cm, CONTENT_W - 9*cm])
    story.append(tbl7)
    story.append(spacer(0.3))
    story.append(divider(GOLD))
    story.append(spacer(0.2))
    story.append(Paragraph(
        "Na próxima aula, veremos como escolher a complexidade certa do modelo "
        "(número de basis functions, grau do polinômio, sigma da RBF) usando "
        "<b>Bias-Variance Tradeoff</b>, <b>Cross-Validation</b> e "
        "<b>Regularização</b>. Os conceitos desta aula são o ponto de partida "
        "para tudo que vem a seguir.",
        S["body"]))

    return story


# ---------------------------------------------------------------------------
# DOCUMENT ASSEMBLY
# ---------------------------------------------------------------------------
def build_pdf(output_path: str):
    total_pages = [None]   # mutable container for 2-pass page count
    canvas_cb   = LectureCanvas(total_pages)

    # Frame for content pages (leaves room for sidebar, header, footer)
    content_frame = Frame(
        SIDEBAR_W + MARGIN,
        FOOTER_H + 0.3 * cm,
        CONTENT_W,
        PAGE_H - HEADER_H - FOOTER_H - 0.6 * cm,
        id="content",
        leftPadding=0,
        rightPadding=0,
        topPadding=4,
        bottomPadding=4,
    )

    # Cover frame — full page, no margins
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, id="cover",
                        leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0)

    cover_template = PageTemplate(
        id="Cover",
        frames=[cover_frame],
        onPage=canvas_cb.draw_cover)

    content_template = PageTemplate(
        id="Content",
        frames=[content_frame],
        onPage=canvas_cb.draw_page)

    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        pageTemplates=[cover_template, content_template],
        leftMargin=0, rightMargin=0,
        topMargin=0, bottomMargin=0,
        title="Aula 1 — General Linear Models & Basis Functions",
        author="Prof. Johannes Schwab, PhD — FH Kufstein Tirol",
        subject="Statistisches Lernen 2",
    )

    styles = build_styles()
    story  = [NextPageTemplate("Content")] + build_content(styles)

    # First pass — build to get page count
    doc.build(story)

    # Extract total pages and rebuild so footer shows correct count
    import PyPDF2
    try:
        with open(output_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total_pages[0] = len(reader.pages)
    except Exception:
        # PyPDF2 not available — skip 2-pass; footer shows "?"
        total_pages[0] = "?"
        return

    doc2 = BaseDocTemplate(
        output_path,
        pagesize=A4,
        pageTemplates=[cover_template, content_template],
        leftMargin=0, rightMargin=0,
        topMargin=0, bottomMargin=0,
        title="Aula 1 — General Linear Models & Basis Functions",
        author="Prof. Johannes Schwab, PhD — FH Kufstein Tirol",
        subject="Statistisches Lernen 2",
    )

    story2  = [NextPageTemplate("Content")] + build_content(styles)
    doc2.build(story2)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    out  = os.path.join(base, "L1_General_Linear_Models.pdf")
    print(f"Gerando PDF: {out}")
    build_pdf(out)
    print("PDF gerado com sucesso!")
