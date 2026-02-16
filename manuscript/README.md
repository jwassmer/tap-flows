# Manuscript

This folder contains the LaTeX manuscript source and revision artifacts.

- `main.tex`: current manuscript source.
- `references.bib`: bibliography database.
- `main_FEB_2026.tex`: snapshot baseline.
- `main-before-breakpoint-predictions.tex`: pre-breakpoint-analysis snapshot.
- `revisions/`: diff files and revision-specific outputs.

Build locally from this directory:

```bash
latexmk -pdf main.tex
```
