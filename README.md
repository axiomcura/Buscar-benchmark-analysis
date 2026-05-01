# Buscar Benchmark Analysis

This repository contains the benchmark analysis for the **Buscar** pipeline across several biological datasets. Buscar is designed to quantify phenotypic shifts in morphological profiles, specifically measuring how treatments can "rescue" diseased states toward a healthy target.

## Datasets Covered

- **CFReT (Cardiac Fibrosis Pilot)**: Cell Painting dataset of cardiac cells. The analysis identifies compounds that shift failing cardiomyocytes towards a healthy phenotype.
- **MitoCheck**: Single-cell profiles used to validate the pipeline's ability to distinguish and score diverse morphological phenotypes.
- **CPJUMP1**: Assessment of Buscar's consistency across technical replicates and different plates.

## Analysis Methodology

The benchmark analysis focuses on validating the **Buscar** pipeline's ability to identify phenotypic rescue using the following methodologies:

### 1. Phenotypic Rescue Scoring (CFReT)
Quantifies how treatments shift diseased morphological profiles toward a healthy state.
- **Reference (Diseased):** Failing cardiomyocytes treated with DMSO.
- **Target (Healthy):** Healthy cardiomyocytes treated with DMSO.
- **Goal:** Identifying treatments (e.g., TGFRi) that minimize the distance between treated-failing cells and the healthy target.

### 2. Leave-One-Gene-Out (LOGO) Validation (MitoCheck)
Assesses whether phenotypic activity scores are inflated by data leakage.
- **Process:** For a target phenotype (e.g., *Prometaphase*), cells associated with one specific gene are excluded when building signatures.
- **Evaluation:** The held-out gene's cells are scored against signatures built from *other* genes.
- **Inference:** Low scores for held-out genes confirm that the morphological signal is biological and not an artifact of specific gene inclusion.

### 3. Replicate Consistency (CPJUMP1)
Evaluates the robustness of Buscar scoring across technical variations.
- **Cross-Plate Analysis:** Assessing if Buscar reliably scores perturbations when replicates are spread across different plates vs. co-located on a single plate.

## Getting Started

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

The analysis is organized into [Jupyter notebooks](notebooks/). You can explore each dataset's folder to see the specific pipeline implementation and results.
