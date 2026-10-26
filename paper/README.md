# Paper

LaTeX source of the paper. Build from this folder (needs `latexmk`, `pdflatex`, `pandas`):

```bash
./build.sh
```

`make_tables.py` writes `tables/*.tex` and `tables/numbers.txt` from `../results/tables/*.csv`, so every table
number comes straight from the result files. Figures are copies of files from `../results/figures/`.
Set the revision name in `paper.tex` (`\revision`) to the git tag of the frozen version.
