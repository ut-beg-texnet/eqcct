# Overleaf package (EQCCTPro draft)

## Files to upload

Upload these into a **new blank Overleaf project** (same folder level):

| File | Role |
|------|------|
| `main.tex` | Entry point: title, abstract, `\input{body}`, appendix `\input{supplement}` |
| `body.tex` | Main text (Introduction through References) |
| `supplement.tex` | Figures 1--8 and Tables 1--6 |

## Figures

1. Create a folder named **`figures`** in the project root (next to `main.tex`).
2. Copy the image files from `eqcctpro/docs/figures/` into `figures/`, keeping these names:

- `fig1.JPG`
- `fig2.JPG`
- `fig3.PNG`
- `fig4_runtime_3d.png`
- `fig5.png`
- `fig6.png`
- `fig7_serial_vs_ripper.png`
- `fig8_serial_vs_modelactor.png`

**Linux/macOS** (from `eqcctpro/docs/overleaf`):

```bash
mkdir -p figures
cp ../figures/fig1.JPG ../figures/fig2.JPG ../figures/fig3.PNG \
   ../figures/fig4_runtime_3d.png ../figures/fig5.png ../figures/fig6.png \
   ../figures/fig7_serial_vs_ripper.png ../figures/fig8_serial_vs_modelactor.png \
   figures/
```

## Compiler

Use **pdfLaTeX** (Overleaf Menu → Compiler → pdfLaTeX). The preamble is written for pdfLaTeX + T1 fonts.

## After upload

1. Set **Main document** to `main.tex` (Overleaf: Menu → Main file).
2. Edit `\author{...}` in `main.tex` with real names and affiliations.
3. Recompile.

## Regenerating `body.tex` from Markdown (optional)

If you change `docs/EQCCTPro_Draft.md` and want to refresh the LaTeX body, you can adapt `md_to_tex_body.py`; the checked-in `body.tex` has been hand-corrected for section titles, `\texttt{}` underscores, and `\S` references, so a straight regeneration may need the same fixes.
