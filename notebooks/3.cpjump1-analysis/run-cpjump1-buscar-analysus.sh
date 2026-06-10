#!/bin/bash
set -e

# Activate your environment (adjust as needed)
conda activate buscar

# convert all
jupyter nbconvert --to script --output-dir=./nbconverted ./*.ipynb

# execute all scripts
python ./nbconverted/1.generate-on-off-signatures.py
python ./nbconverted/2.run-compound-replicate-consistency-analysis.py
python ./nbconverted/3.run-crispr-replicate-consistency-analysis.py
python ./nbconverted/4.plot-replicate-consistency-analysis.py
python ./nbconverted/5.cpjump1-titration-analysis.py
