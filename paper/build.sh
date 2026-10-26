#!/bin/sh
# run from the paper/ folder: tables are generated from ../results/tables
set -e
python3 make_tables.py ../results/tables tables
latexmk -pdf -interaction=nonstopmode paper.tex
