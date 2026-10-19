# Sources for "Choose the Noise, Keep the Flow"

Blog post: `_posts/2026-10-18-choose-the-noise.md` · PDF: `files/choose-the-noise.pdf`.
Jekyll ignores this folder (its name starts with an underscore).

- `experiment/` — the 2D experiment of Section 4 (PyTorch, runs in a few minutes on a laptop GPU).
  `python noise_routes.py` trains the frozen flow and the one-step student (cached in
  `results/decoders.pt`), then all five routes for three seeds → `results/results.json`.
  `python gaussian_floor.py` computes the best per-state Gaussian → `results/gaussian_floor.json`.
- `figures/` — one script per figure. Each writes SVG and PNG to
  `assets/img/posts/choose-the-noise/` and PDF to `paper/figures/`.
  Set `FONT_DIR` to a folder with `EBGaramond.ttf` and `EBGaramond-Italic.ttf` (SIL OFL) to match the site.
  `fig4_experiment.py` (Figures 4 and 5) reads `experiment/results/` and uses `STATE_IDX=12` (goal angle 67.5°).
- `paper/` — the LaTeX version. `python3 md2tex.py` converts the post into `body.tex` and `refs.tex`
  (with the more formal wording listed in `FORMAL`); then `pdflatex main.tex` twice, and copy
  `main.pdf` to `files/choose-the-noise.pdf`.
