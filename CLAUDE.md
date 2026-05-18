# Statistics Learning 2 — Claude Context

## Who I am
Ian Calixto, master's student at FH Kufstein (Austria), 2nd semester.
This course is taught in English. I prefer answers and notebooks in **PT-BR** unless asked otherwise.

## Course overview
**Statistisches Lernen 2** — supervised statistical/machine learning, linear models through Bayesian regularization.
4 lectures, each with a PDF explanation, a deep-dive notebook, and a walkthrough notebook.

## Folder structure

```
statistics-learning-2-semester-2/
├── CLAUDE.md                          ← this file
├── lectures/                          ← original professor slides (PDFs)
├── lectures explanation/              ← my custom explanation PDFs + generator scripts
│   ├── generate_L1_pdf.py             ← ReportLab script → L1_General_Linear_Models.pdf
│   ├── generate_L2_pdf.py             ← ReportLab script → L2_Bias_Variance_Model_Selection.pdf
│   ├── generate_L3_pdf.py             ← ReportLab script → L3_Regularization_CrossValidation.pdf
│   └── generate_L4_pdf.py             ← ReportLab script → L4_Probabilistic_View_Bayesian_Regularization.pdf
├── lectures understanding/            ← deep-dive Jupyter notebooks (full derivations + plots)
│   ├── L1_General_Linear_Models_Basis_Functions.ipynb
│   ├── L2_Bias_Variance_Model_Selection.ipynb
│   ├── L3_Regularization_CrossValidation.ipynb
│   └── L4_Probabilistic_View_Bayesian_Regularization.ipynb
├── understanding notebooks/           ← step-by-step walkthrough notebooks
│   ├── L1_walkthrough.ipynb
│   ├── L2_walkthrough.ipynb
│   ├── L3_walkthrough.ipynb
│   ├── linear_regression_walkthrough.ipynb
│   ├── mnist_regularization_walkthrough.ipynb
│   ├── prior_MAP_walkthrough.ipynb
│   └── regularization_pytorch_simple_walkthrough.ipynb
├── Probeklausur/                      ← practice exam + solutions
│   ├── Probeklausur_Solutions.ipynb   ← full solutions with visualizations (PT-BR)
│   └── Possiveis_Questoes_Exame.ipynb ← possible exam questions in Probeklausur style (PT-BR)
├── linear_regression.ipynb            ← root-level practice notebooks
├── mnist_regularization_pytorch.ipynb
├── prior_MAP.ipynb
├── regularization_pytorch_simple.ipynb
└── .venv/                             ← Python virtualenv (all dependencies installed here)
```

## Python environment
Always use the project venv:
```bash
source .venv/bin/activate
# or run with:
.venv/bin/python script.py
.venv/bin/jupyter lab
```
Key packages installed: `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `torch`, `reportlab`, `PyPDF2`, `jupyter`.

If a generate script fails with `ModuleNotFoundError`, install the missing package with `.venv/bin/pip install <package>`.

To run a generate script:
```bash
cd "lectures explanation"
../.venv/bin/python generate_L4_pdf.py
```

To execute a notebook headlessly:
```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Notebook.ipynb
```

## Lecture topics

| # | Title | Key concepts |
|---|-------|-------------|
| L1 | General Linear Models & Basis Functions | Design matrix B, Normal Equation `θ = (BᵀB)⁻¹Bᵀy`, polynomial/RBF/Fourier basis |
| L2 | Bias-Variance & Model Selection | MSE = Bias² + Var + σ², learning curves, Hold-Out / K-Fold / LOO / Stratified CV |
| L3 | Regularization & Cross-Validation | Ridge (L2) / Lasso (L1), Tikhonov `(BᵀB + λI)⁻¹Bᵀy`, early stopping ≈ Ridge, CV for λ selection |
| L4 | Probabilistic View & Bayesian Regularization | MLE, MAP, Gaussian prior→Ridge, Laplace prior→Lasso, bias-variance-noise decomposition |

## generate_Lx_pdf.py — ReportLab patterns

All 4 scripts share the same architecture. Key things to know:

**Cover page isolation** — always use `[NextPageTemplate("Content"), PageBreak()]` at the start of story. Without `PageBreak()`, body content renders inside the cover frame on top of the gradient art.

**formula_block()** — renders LaTeX via matplotlib as a PNG image. Background `bg_hex="#F0F4F8"` (light blue-grey), text color `"#1A1A2E"` (dark navy). Never use dark backgrounds here.

**why_box()** — 4-row ReportLab Table (Why header / Why body / How to apply header / How to apply body). Headers use light tinted backgrounds (`#FDECEA` rose, `#EAF0FD` blue) with left accent borders. Never use solid dark CRIMSON/BURGUNDY as header background.

**Color palette constants:**
```python
CRIMSON  = HexColor("#8B0000")
BURGUNDY = HexColor("#5C1A1A")
SLATE    = HexColor("#2C3E6B")
SIDEBAR_BG = HexColor("#F4F4F4")   # neutral, not rosy
ROW_ALT    = HexColor("#F5F5F5")
FORMULA_BG = HexColor("#F0F4F8")
```

## Notebook patterns

**Polynomial numerical stability** — for x ∈ [0,3] with degree > 8, always normalize: `x_n = x / 3.0`. High-degree raw basis causes ill-conditioned matrices and gradient divergence.

**Ridge closed form:**
```python
theta = np.linalg.solve(B.T @ B + lam * np.eye(p+1), B.T @ y)
```
Add `lam=1e-6` as a minimum to avoid singular matrices even when "unregularized".

**Early stopping simulation** — use decreasing-lambda Ridge sequence instead of raw gradient descent (avoids divergence, correctly represents the equivalence theorem):
```python
lambdas = np.logspace(2, -4, 150)   # strong reg → no reg = simulates more training steps
```

**Bias-Variance decomposition plots** — always clip extreme predictions before computing bias/variance: `np.clip(pred, -5, 8)`.

**Matplotlib backend for scripts** — always set at the top of any non-interactive script:
```python
import matplotlib
matplotlib.use('Agg')
```

## Probeklausur style
When creating exam-style questions, follow this format:
- `**Aufgabe N** (X Punkte)` as exercise header
- Sub-questions labeled `a)`, `b)`, `c)` with point values in parentheses
- Typical total: 20 points + 5 bonus
- Answers in PT-BR, LaTeX math in markdown cells, visualizations in code cells

## Key formulas quick reference

| Formula | Expression |
|---------|-----------|
| Normal Equation | `θ* = (BᵀB)⁻¹Bᵀy` |
| Ridge (Tikhonov) | `θ_λ = (BᵀB + λI)⁻¹Bᵀy` |
| MSE decomposition | `MSE = Bias² + Variance + σ²_ε` |
| MLE (Gaussian) | `argmin ‖Bθ - y‖²` |
| MAP (Gaussian prior) | `argmin ‖Bθ - y‖² + λ‖θ‖²` (→ Ridge) |
| MAP (Laplace prior) | `argmin ‖Bθ - y‖² + λ‖θ‖₁` (→ Lasso) |
| Bootstrap CI (percentile) | `[θ*_(α/2), θ*_(1-α/2)]` over B bootstrap samples |
