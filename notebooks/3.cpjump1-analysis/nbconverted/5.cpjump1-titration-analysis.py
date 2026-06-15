#!/usr/bin/env python

# # Titration Analysis
#
# This notebook investigates how the number of single cells impacts the stability and performance of Buscar scores. We specifically analyze how score variation increases as cell count decreases, helping to determine the minimum cell count required to reliably use Buscar for high-content screening.
#
# Analysis methods:
# - **Cell Titration:** Gradually decreasing the number of pooled perturbed cells while keeping the reference set (negative controls + reference-plate perturbations) unchanged.
#
# The primary goal is to establish practical recommendations for using Buscar in high-content screening by understanding how score stability degrades at low cell counts.
#
# We use the **CPJUMP1 compound dataset**, focusing on treatments that demonstrated high replicate consistency. These stable signals provide an ideal baseline for stress-testing Buscar's sensitivity to decreasing cell numbers.

# In[1]:


import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm.auto import tqdm

# loading repo level utils module
from utils.io_utils import load_sc_profiles

# adding buscar src to path for importing buscar functions
sys.path.insert(0, "../../buscar/src/")
from buscar.metrics import calculate_buscar_scores
from buscar.signatures import identify_signatures

# ## Helper functions
#
#

# ## Setting Paths and Loading Data
#
# Here we load the replicate consistency scores for U2OS and A549 cells from the CPJUMP1 compound dataset, along with the single-cell morphological profiles.

# In[2]:


# setting module results dir
module_results_dir = pathlib.Path("./results/replicate_analysis")

# setting output directory
titration_results_dir = pathlib.Path("./results/titration_analysis")
titration_results_dir.mkdir(parents=True, exist_ok=True)


# Load the replicate consistency scores (used to identify candidate treatments) and the CPJUMP1 single-cell morphological profiles.

# In[3]:


# loading in the replicate consistency data
u2os_rep_trt_df = pl.read_ndjson(
    module_results_dir / "U2OS_original_compound-replicate-tracking.jsonl"
)
a549_rep_trt_df = pl.read_ndjson(
    module_results_dir / "A549_original_compound-replicate-tracking.jsonl"
)

# path to cpjump1 data
cpjump1_meta_feats, cpjump1_feats, cpjump1_compound_df = load_sc_profiles(
    data_name="cpjump1", datatype="compound"
)


# Preprocess the data for titration analysis: remove randomly paired perturbations (used as the null baseline in replicate scoring), then subset the CPJUMP1 profiles to retain only negative controls and compound-treated cells, dropping any other control types.

# In[4]:


# Remove randomly paired perturbations from the replicate score dataframes
u2os_rep_trt_df = u2os_rep_trt_df.filter(~pl.col("random_perturbations"))
a549_rep_trt_df = a549_rep_trt_df.filter(~pl.col("random_perturbations"))

# Preprocessing cpjump1 profiles for titration analysis
# Split the cpjump1 data into cell lines and control types
cpjump1_u2os_df = cpjump1_compound_df.filter(pl.col("Metadata_cell_type") == "U2OS")
cpjump1_a549_df = cpjump1_compound_df.filter(pl.col("Metadata_cell_type") == "A549")

# update the cpjump1 dataframes to include both negative controls and perturbed cells
# removes other existing controls
cpjump1_u2os_df = pl.concat(
    [
        cpjump1_u2os_df.filter(pl.col("Metadata_control_type") == "negcon"),
        cpjump1_u2os_df.filter(pl.col("Metadata_control_type").is_null()),
    ]
)
cpjump1_a549_df = pl.concat(
    [
        cpjump1_a549_df.filter(pl.col("Metadata_control_type") == "negcon"),
        cpjump1_a549_df.filter(pl.col("Metadata_control_type").is_null()),
    ]
)

# check if profiles are empty
if cpjump1_u2os_df.is_empty():
    print("Warning: U2OS profile is empty")
if cpjump1_a549_df.is_empty():
    print("Warning: A549 profile is empty")


# ## Finding Candidate Treatments
#
# We select treatments that showed consistently high on-Buscar scores in the replicate analysis — indicating a strong, reproducible morphological signal. These treatments serve as ideal stress-test candidates because a clear signal makes it easier to detect when score stability begins to break down.
#
# Treatments are ranked by a consistency score that combines their mean absolute distance from the ideal on-Buscar score of 1.0 with variance normalized by that error. This keeps treatments close to 1.0 while penalizing unstable scores.

# In[5]:


def find_top_treatments(
    replicate_buscar_scores_df: pl.DataFrame,
    top_treatments: int = 3,
) -> pl.DataFrame:
    """Rank treatments by how consistently their on-Buscar score stays close to 1.0.

    Parameters:
    -----------
    replicate_buscar_scores_df: pl.DataFrame
        DataFrame containing treatment names, Buscar scores, and a
        `random_perturbations` boolean column.
    top_treatments: int
        Number of top-ranked treatments to return (default: 3).

    Returns:
    --------
    pl.DataFrame
        Treatments ranked by a consistency score that combines mean absolute
        distance from the ideal score (1.0) and variance normalized by that
        distance, with per-treatment summary statistics.
    """
    required_cols = {"perturbation", "on_score"}
    missing_cols = required_cols - set(replicate_buscar_scores_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # Rank treatments by a combined consistency score. Normalizing variance by
    # score error prevents a treatment with very low mean error but unstable
    # replicate scores from being ranked above a slightly less accurate but much
    # more consistent treatment.
    ranked_df = (
        replicate_buscar_scores_df.group_by("perturbation")
        .agg(
            pl.col("on_score").mean().alias("mean_on_score"),
            pl.col("on_score").median().alias("median_on_score"),
            pl.col("on_score").std().fill_null(0).alias("std_on_score"),
            pl.col("on_score").var().fill_null(0).alias("var_on_score"),
            pl.col("on_score").min().alias("min_on_score"),
            pl.col("on_score").max().alias("max_on_score"),
            (pl.col("on_score") - 1.0)
            .abs()
            .mean()
            .alias("mean_abs_score_error_from_1"),
        )
        .with_columns(
            (
                pl.col("var_on_score") / (pl.col("mean_abs_score_error_from_1") + 1e-12)
            ).alias("variance_normalized_by_error")
        )
        .with_columns(
            (
                pl.col("mean_abs_score_error_from_1")
                + pl.col("variance_normalized_by_error")
            ).alias("consistency_rank_score")
        )
        .sort("consistency_rank_score")
    )

    return ranked_df.head(top_treatments)


# In[6]:


u2os_ranked_treatments_df = find_top_treatments(
    u2os_rep_trt_df,
    top_treatments=u2os_rep_trt_df["perturbation"].n_unique(),
).with_columns(pl.lit("U2OS").alias("cell_type"))
a549_ranked_treatments_df = find_top_treatments(
    a549_rep_trt_df,
    top_treatments=a549_rep_trt_df["perturbation"].n_unique(),
).with_columns(pl.lit("A549").alias("cell_type"))

ranked_treatments_df = pl.concat(
    [u2os_ranked_treatments_df, a549_ranked_treatments_df]
).select(
    [
        "cell_type",
        "perturbation",
        "mean_on_score",
        "median_on_score",
        "std_on_score",
        "var_on_score",
        "mean_abs_score_error_from_1",
        "variance_normalized_by_error",
        "consistency_rank_score",
        "min_on_score",
        "max_on_score",
    ]
)
u2os_top_treatments_df = u2os_ranked_treatments_df.head(3)
a549_top_treatments_df = a549_ranked_treatments_df.head(3)
ranked_treatments_df


# ## Running Titration Analysis
#
# For each selected treatment, we iteratively reduce the number of pooled perturbed cells in steps of 10% (from 100% retained down to 1%). At each titration level, Buscar scoring is repeated 20 times with different random subsamples to measure how score stability changes with cell count.

# In[7]:


# parameters
rng_seed = 0
negcon_subsample_fraction = 0.02
cell_removal_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
n_buscar_iterations = 5

u2os_top_treatments = u2os_top_treatments_df["perturbation"].to_list()
a549_top_treatments = a549_top_treatments_df["perturbation"].to_list()


# In[8]:


# get unique plates for both cell types
u2os_unique_plates = cpjump1_u2os_df["Metadata_Plate"].unique().to_list()
a549_unique_plates = cpjump1_a549_df["Metadata_Plate"].unique().to_list()


# In[9]:


# create a standalone random state for reproducible local sampling
rng = np.random.RandomState(rng_seed)

# mapping of cell types to their profiles and top treatments for titration
# analysis
profiles_by_cell_type = {
    "U2OS": cpjump1_u2os_df,
    "A549": cpjump1_a549_df,
}
top_treatments_by_cell_type = {
    "U2OS": u2os_top_treatments,
    "A549": a549_top_treatments,
}

# defining the metadata columns to include in the Buscar scoring dataframe, which
# will be used to label the reference vs titrated groups for scoring
meta_cols_with_titration_label = cpjump1_meta_feats + ["_titration_label"]
titration_scores = []
checkpoint_path = (
    titration_results_dir / "_cpjump1_compound_titration_scores_checkpoint.jsonl"
)

# iterate through each cell type and its corresponding top treatments to perform
# the titration analysis
for cell_type, selected_treatments in top_treatments_by_cell_type.items():
    profiles = profiles_by_cell_type[cell_type]

    plate_ids = profiles["Metadata_Plate"].unique().to_list()
    if len(plate_ids) != 4:
        raise ValueError(
            f"Expected 4 unique plates for {cell_type}, but found {len(plate_ids)}"
        )

    # randomly select one plate to serve as the reference plate
    selected_plate_id = rng.choice(plate_ids)

    for treatment in tqdm(
        selected_treatments, desc=f"{cell_type} treatments", unit="treatment"
    ):
        # all profiles from the reference plate (negative controls + perturbed cells)
        ref_plate_profiles = profiles.filter(
            pl.col("Metadata_Plate") == selected_plate_id
        )

        # pool perturbed cells from all other plates to use as the titration target
        pooled_perturbed_cells = profiles.filter(
            (pl.col("Metadata_Plate") != selected_plate_id)
            & (pl.col("Metadata_pert_iname") == treatment)
        )

        for cell_removal_percentage in tqdm(
            cell_removal_percentages,
            desc=f"  [{cell_type} | {treatment}] titration",
            unit="level",
            leave=False,
        ):
            # fraction of pooled perturbed cells to retain at this titration level
            perturbed_keep_fraction = (100 - cell_removal_percentage) / 100

            for iteration in tqdm(
                range(n_buscar_iterations),
                desc=f"  remove={cell_removal_percentage}%",
                unit="iter",
                leave=False,
            ):
                # deterministic seed per iteration for reproducibility
                iter_id = iteration + 1
                iter_seed = iter_id + (cell_removal_percentage * 1_000)

                # subsample a small fraction of negative controls from the reference plate
                ref_negcon = ref_plate_profiles.filter(
                    pl.col("Metadata_control_type") == "negcon"
                ).sample(
                    fraction=negcon_subsample_fraction,
                    seed=iter_seed,
                    with_replacement=True,
                )

                # all perturbed cells from the reference plate (used to define signatures)
                ref_perturbation_cells = ref_plate_profiles.filter(
                    pl.col("Metadata_pert_iname") == treatment
                )

                # subsample the pooled perturbed cells at this titration level
                titrated_perturbed_cells = pooled_perturbed_cells.sample(
                    fraction=perturbed_keep_fraction,
                    seed=iter_seed,
                    with_replacement=False,
                )

                try:
                    # derive on/off morphological signatures from the reference plate
                    on_sig, off_sig, _ = identify_signatures(
                        ref_profiles=ref_negcon.select(cpjump1_feats),
                        target_profiles=ref_perturbation_cells.select(cpjump1_feats),
                        morph_feats=cpjump1_feats,
                        seed=iter_seed,
                    )

                    # combine all three groups and assign a role label for Buscar
                    combined_profiles = pl.concat(
                        [ref_negcon, ref_perturbation_cells, titrated_perturbed_cells],
                        how="vertical",
                    ).with_columns(
                        pl.when(pl.col("Metadata_control_type") == "negcon")
                        .then(pl.lit("negcon"))
                        .when(pl.col("Metadata_Plate") == selected_plate_id)
                        .then(pl.lit("ref_treated"))
                        .otherwise(pl.lit("pooled_titrated"))
                        .alias("_titration_label")
                    )

                    # score the titrated pooled cells against the reference
                    scores_df = calculate_buscar_scores(
                        profiles=combined_profiles,
                        meta_cols=meta_cols_with_titration_label,
                        on_morphology_signature=on_sig,
                        off_morphology_signature=off_sig,
                        target="ref_treated",
                        ref_state="negcon",
                        perturbation_col="_titration_label",
                        state_col="_titration_label",
                        seed=iter_seed,
                        n_threads=1,
                    )

                    # extract Buscar scores for the titrated (pooled) cells
                    titrated_score = scores_df.filter(
                        pl.col("perturbation") == "pooled_titrated"
                    ).row(0, named=True)

                    record = {
                        "cell_type": cell_type,
                        "perturbation": treatment,
                        "ref_plate": selected_plate_id,
                        "cell_removal_percentage": cell_removal_percentage,
                        "perturbed_keep_fraction": perturbed_keep_fraction,
                        "on_score": titrated_score["on_buscar_scores"],
                        "off_score": titrated_score["off_buscar_scores"],
                        "n_ref_cells": ref_negcon.height,
                        "n_ref_perturbation_cells": ref_perturbation_cells.height,
                        "n_pooled_perturbed_cells": pooled_perturbed_cells.height,
                        "n_titrated_perturbed_cells": titrated_perturbed_cells.height,
                        "n_on_signature_features": len(on_sig),
                        "n_off_signature_features": len(off_sig),
                        "iteration": iter_id,
                        "error": None,
                    }
                    titration_scores.append(record)
                    with open(checkpoint_path, "a") as f:
                        f.write(json.dumps(record) + "\n")

                except Exception as err:
                    record = {
                        "cell_type": cell_type,
                        "perturbation": treatment,
                        "ref_plate": selected_plate_id,
                        "cell_removal_percentage": cell_removal_percentage,
                        "perturbed_keep_fraction": perturbed_keep_fraction,
                        "on_score": None,
                        "off_score": None,
                        "n_ref_cells": ref_negcon.height,
                        "n_ref_perturbation_cells": ref_perturbation_cells.height,
                        "n_pooled_perturbed_cells": pooled_perturbed_cells.height,
                        "n_titrated_perturbed_cells": titrated_perturbed_cells.height,
                        "n_on_signature_features": None,
                        "n_off_signature_features": None,
                        "iteration": iter_id,
                        "error": str(err),
                    }
                    titration_scores.append(record)
                    with open(checkpoint_path, "a") as f:
                        f.write(json.dumps(record) + "\n")


# save the titration scores to a parquet file
titration_scores_df = pl.DataFrame(titration_scores)
titration_scores_df.write_parquet(
    titration_results_dir / "cpjump1_compound_titration_scores.parquet"
)


# ## Titration Analysis: On-Buscar Score vs. Cell Count
#
# Line-dot plots showing how the mean on-Buscar score changes as the number of titrated (pooled) perturbed cells decreases. The shaded band represents ± 1 standard deviation across iterations.

# In[15]:


# per-treatment stats at each removal level for both score types
titration_plot_df = (
    titration_scores_df.group_by(
        [
            "cell_type",
            "perturbation",
            "ref_plate",
            "cell_removal_percentage",
            "perturbed_keep_fraction",
        ]
    )
    .agg(
        pl.col("n_titrated_perturbed_cells")
        .mean()
        .round(0)
        .cast(pl.Int64)
        .alias("mean_n_cells"),
        pl.col("on_score").mean().alias("mean_on_score"),
        pl.col("on_score").std().fill_nan(0).alias("std_on_score"),
        pl.col("off_score").mean().alias("mean_off_score"),
        pl.col("off_score").std().fill_nan(0).alias("std_off_score"),
    )
    .with_columns((100 - pl.col("cell_removal_percentage")).alias("keep_pct"))
    .sort(["cell_type", "perturbation", "cell_removal_percentage"])
)

# Scale each cell-type column to the treatment with the most titrated cells.
cell_type_max_cells_df = (
    titration_plot_df.sort(
        ["cell_type", "mean_n_cells"],
        descending=[False, True],
    )
    .group_by("cell_type", maintain_order=True)
    .agg(
        pl.col("perturbation").first().alias("max_cell_treatment"),
        pl.col("mean_n_cells").first().alias("max_n_cells"),
    )
)
titration_plot_df = titration_plot_df.join(
    cell_type_max_cells_df, on="cell_type", how="left"
).with_columns(
    (pl.col("mean_n_cells") / pl.col("max_n_cells") * 100).alias(
        "cell_count_pct_of_max"
    )
)

# Representative tick labels come only from the treatment with the highest
# cell count in that cell type, so the 100% tick is the true maximum.
tick_ref_df = (
    titration_plot_df.filter(pl.col("perturbation") == pl.col("max_cell_treatment"))
    .group_by(["cell_type", "cell_removal_percentage"])
    .agg(
        pl.col("cell_count_pct_of_max").max().alias("tick_x"),
        pl.col("mean_n_cells").max().cast(pl.Int64).alias("tick_n_cells"),
    )
    .sort(["cell_type", "cell_removal_percentage"])
)

# plotting parameters for later use in plotting functions
score_types = [
    ("mean_on_score", "std_on_score", "On-Buscar Score"),
    ("mean_off_score", "std_off_score", "Off-Buscar Score"),
]
cell_types = titration_plot_df["cell_type"].unique().sort().to_list()


# In[ ]:


# plotting Buscar scores across titration levels for each treatment and cell type
fig, axes = plt.subplots(
    len(score_types),
    len(cell_types),
    figsize=(7 * len(cell_types), 5 * len(score_types)),
    sharey="row",
    sharex="col",
)

for row_idx, (score_col, std_col, score_label) in enumerate(score_types):
    for col_idx, cell_type in enumerate(cell_types):
        ax = axes[row_idx][col_idx]
        ct_df = titration_plot_df.filter(pl.col("cell_type") == cell_type)
        treatments = ct_df["perturbation"].unique().sort().to_list()

        # plot each treatment's mean score across titration levels with ±1 SD band
        for treatment in treatments:
            trt_df = ct_df.filter(pl.col("perturbation") == treatment).sort(
                "cell_count_pct_of_max"
            )
            x = trt_df["cell_count_pct_of_max"].to_numpy()
            y = trt_df[score_col].to_numpy()
            std = trt_df[std_col].to_numpy()

            (line,) = ax.plot(
                x, y, marker="o", linewidth=1.8, markersize=5, label=treatment
            )
            ax.fill_between(x, y - std, y + std, alpha=0.2, color=line.get_color())

        # x-tick labels show cell count and % retained; only drawn on the bottom row
        if row_idx == len(score_types) - 1:
            ticks = tick_ref_df.filter(pl.col("cell_type") == cell_type).sort(
                "cell_removal_percentage"
            )
            tick_x = ticks["tick_x"].to_list()
            tick_labels = [
                f"{pct:.0f}%\n({n:,})"
                for pct, n in zip(ticks["tick_x"], ticks["tick_n_cells"])
            ]
            ax.set_xticks(tick_x)
            ax.set_xticklabels(tick_labels, fontsize=7.5)
            ax.set_xlabel("Titrated cells (% of max; mean n)", fontsize=11)

        ax.set_xlim(0, 105)

        # dashed reference line at 1.0 (the ideal on-Buscar score)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(title="Treatment", fontsize=9, title_fontsize=9)

        if row_idx == 0:
            ax.set_title(cell_type, fontsize=13, fontweight="bold")

        if col_idx == 0:
            ax.set_ylabel(score_label, fontsize=11)

fig.suptitle(
    "Buscar Score Stability Across Cell Titration Levels",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
fig.tight_layout()

plot_path = titration_results_dir / "cpjump1_compound_titration_scores.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {plot_path}")


# In[ ]:
