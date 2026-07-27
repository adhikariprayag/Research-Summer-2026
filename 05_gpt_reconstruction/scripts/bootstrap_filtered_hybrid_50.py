from pathlib import Path

import numpy as np
import pandas as pd


REVIEW_PATH = Path(
    "02_dataset/large_scale_dataset/full_v4_private/"
    "validation_review_50/hybrid_prediction_review_template_50.csv"
)

MISSED_PATH = Path(
    "02_dataset/large_scale_dataset/full_v4_private/"
    "validation_review_50/missed_skill_review_template_50.csv"
)

OUTPUT_DIR = Path(
    "07_results/hybrid_manual_evaluation_50"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
N_BOOTSTRAPS = 10000
RANDOM_SEED = 42

SYSTEMS = [
    "GPT",
    "Raw Hybrid",
    "CV Filtered Hybrid",
]


def normalize(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            numerator,
            dtype=float,
        ),
        where=denominator != 0,
    )


def metrics_from_counts(tp, fp, fn):
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)

    f1 = divide(
        2 * precision * recall,
        precision + recall,
    )

    return precision, recall, f1


review = pd.read_csv(REVIEW_PATH)
missed = pd.read_csv(MISSED_PATH)

review["resume_id"] = (
    review["resume_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

review["source_dataset"] = (
    review["source_dataset"]
    .fillna("")
    .astype(str)
    .str.strip()
)

review["source_clean"] = normalize(
    review["prediction_source"]
)

review["decision_clean"] = normalize(
    review["review_decision"]
)

review["entity_norm"] = normalize(
    review["entity_text"]
)

missed["resume_id"] = (
    missed["resume_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

missed["missed_norm"] = normalize(
    missed["missed_entity_text"]
)

missed = missed[
    missed["missed_norm"].ne("")
].drop_duplicates(
    subset=["resume_id", "missed_norm"]
)


# Recreate the same stratified five-fold assignment.
metadata_columns = [
    "resume_id",
    "source_dataset",
]

if "prediction_band" in review.columns:
    metadata_columns.append(
        "prediction_band"
    )

resume_metadata = (
    review[metadata_columns]
    .drop_duplicates()
    .copy()
)

strata_columns = ["source_dataset"]

if "prediction_band" in resume_metadata.columns:
    resume_metadata["prediction_band"] = (
        resume_metadata["prediction_band"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    strata_columns.append("prediction_band")


fold_parts = []

for stratum_number, (_, group) in enumerate(
    resume_metadata.groupby(
        strata_columns,
        sort=True,
        dropna=False,
    )
):
    shuffled = group.sample(
        frac=1,
        random_state=(
            RANDOM_SEED + stratum_number
        ),
    ).reset_index(drop=True)

    shuffled["fold"] = (
        shuffled.index % N_FOLDS
    ) + 1

    fold_parts.append(
        shuffled[["resume_id", "fold"]]
    )


fold_map = pd.concat(
    fold_parts,
    ignore_index=True,
)

review = review.merge(
    fold_map,
    on="resume_id",
    how="left",
    validate="many_to_one",
)


# Store out-of-fold prediction inclusion.
review["keep_gpt"] = False
review["keep_raw_hybrid"] = True
review["keep_filtered_hybrid"] = False


for fold in range(1, N_FOLDS + 1):
    training_v4 = review[
        review["fold"].ne(fold)
        & review["source_clean"].eq("v4")
    ].copy()

    testing_mask = review["fold"].eq(fold)

    term_quality = (
        training_v4.groupby(
            "entity_norm",
            as_index=False,
        )
        .agg(
            total=("decision_clean", "size"),
            valid=(
                "decision_clean",
                lambda values: values.isin(
                    {"correct", "partial"}
                ).sum(),
            ),
            incorrect=(
                "decision_clean",
                lambda values: values.eq(
                    "incorrect"
                ).sum(),
            ),
        )
    )

    blacklist = set(
        term_quality[
            term_quality["total"].ge(2)
            & term_quality["incorrect"].gt(
                term_quality["valid"]
            )
        ]["entity_norm"]
    )

    testing = review.loc[testing_mask]

    gpt_keep = testing[
        "source_clean"
    ].isin({"gpt", "both"})

    filtered_keep = (
        testing["source_clean"].isin(
            {"gpt", "both"}
        )
        |
        (
            testing["source_clean"].eq("v4")
            & ~testing["entity_norm"].isin(
                blacklist
            )
        )
    )

    review.loc[
        testing.index,
        "keep_gpt",
    ] = gpt_keep.to_numpy()

    review.loc[
        testing.index,
        "keep_filtered_hybrid",
    ] = filtered_keep.to_numpy()


missed_counts = (
    missed.groupby("resume_id")
    .size()
    .to_dict()
)

keep_columns = {
    "GPT": "keep_gpt",
    "Raw Hybrid": "keep_raw_hybrid",
    "CV Filtered Hybrid": (
        "keep_filtered_hybrid"
    ),
}


# Calculate strict TP, FP, and FN for each resume.
records = []

for resume_id, group in review.groupby(
    "resume_id",
    sort=False,
):
    dataset = group[
        "source_dataset"
    ].iloc[0]

    gold_skills = int(
        group["decision_clean"]
        .isin({"correct", "partial"})
        .sum()
    ) + int(
        missed_counts.get(resume_id, 0)
    )

    for system, keep_column in (
        keep_columns.items()
    ):
        kept = group[group[keep_column]]

        tp = int(
            kept["decision_clean"]
            .eq("correct")
            .sum()
        )

        fp = int(
            kept["decision_clean"]
            .isin({"incorrect", "partial"})
            .sum()
        )

        fn = gold_skills - tp

        records.append({
            "resume_id": resume_id,
            "dataset": dataset,
            "system": system,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })


per_resume = pd.DataFrame(records)

dataset_names = list(
    per_resume["dataset"].drop_duplicates()
)

analysis_groups = dataset_names + ["Combined"]

rng = np.random.default_rng(
    RANDOM_SEED
)

metric_records = []
difference_records = []


for analysis_group in analysis_groups:
    if analysis_group == "Combined":
        group_data = per_resume.copy()
    else:
        group_data = per_resume[
            per_resume["dataset"].eq(
                analysis_group
            )
        ].copy()

    resume_ids = (
        group_data["resume_id"]
        .drop_duplicates()
        .tolist()
    )

    resume_to_index = {
        resume_id: index
        for index, resume_id
        in enumerate(resume_ids)
    }

    system_arrays = {}

    for system in SYSTEMS:
        system_rows = group_data[
            group_data["system"].eq(system)
        ].copy()

        system_rows["resume_index"] = (
            system_rows["resume_id"]
            .map(resume_to_index)
        )

        system_rows = system_rows.sort_values(
            "resume_index"
        )

        system_arrays[system] = {
            "tp": system_rows[
                "tp"
            ].to_numpy(),
            "fp": system_rows[
                "fp"
            ].to_numpy(),
            "fn": system_rows[
                "fn"
            ].to_numpy(),
        }

    if analysis_group == "Combined":
        sampled_indices_parts = []

        for dataset in dataset_names:
            dataset_ids = (
                group_data[
                    group_data["dataset"].eq(
                        dataset
                    )
                ]["resume_id"]
                .drop_duplicates()
                .tolist()
            )

            dataset_indices = np.array(
                [
                    resume_to_index[resume_id]
                    for resume_id in dataset_ids
                ]
            )

            sampled_indices_parts.append(
                rng.choice(
                    dataset_indices,
                    size=(
                        N_BOOTSTRAPS,
                        len(dataset_indices),
                    ),
                    replace=True,
                )
            )

        sampled_indices = np.concatenate(
            sampled_indices_parts,
            axis=1,
        )

    else:
        all_indices = np.arange(
            len(resume_ids)
        )

        sampled_indices = rng.choice(
            all_indices,
            size=(
                N_BOOTSTRAPS,
                len(all_indices),
            ),
            replace=True,
        )

    bootstrap_f1 = {}
    point_f1 = {}

    for system in SYSTEMS:
        arrays = system_arrays[system]

        point_tp = arrays["tp"].sum()
        point_fp = arrays["fp"].sum()
        point_fn = arrays["fn"].sum()

        (
            point_precision,
            point_recall,
            original_f1,
        ) = metrics_from_counts(
            np.array([point_tp]),
            np.array([point_fp]),
            np.array([point_fn]),
        )

        point_f1[system] = float(
            original_f1[0]
        )

        boot_tp = arrays["tp"][
            sampled_indices
        ].sum(axis=1)

        boot_fp = arrays["fp"][
            sampled_indices
        ].sum(axis=1)

        boot_fn = arrays["fn"][
            sampled_indices
        ].sum(axis=1)

        (
            boot_precision,
            boot_recall,
            boot_f1,
        ) = metrics_from_counts(
            boot_tp,
            boot_fp,
            boot_fn,
        )

        bootstrap_f1[system] = boot_f1

        metric_records.append({
            "dataset": analysis_group,
            "system": system,
            "precision": float(
                point_precision[0]
            ),
            "precision_ci_lower": (
                np.percentile(
                    boot_precision,
                    2.5,
                )
            ),
            "precision_ci_upper": (
                np.percentile(
                    boot_precision,
                    97.5,
                )
            ),
            "recall": float(
                point_recall[0]
            ),
            "recall_ci_lower": (
                np.percentile(
                    boot_recall,
                    2.5,
                )
            ),
            "recall_ci_upper": (
                np.percentile(
                    boot_recall,
                    97.5,
                )
            ),
            "f1": point_f1[system],
            "f1_ci_lower": np.percentile(
                boot_f1,
                2.5,
            ),
            "f1_ci_upper": np.percentile(
                boot_f1,
                97.5,
            ),
        })

    f1_difference = (
        bootstrap_f1[
            "CV Filtered Hybrid"
        ]
        - bootstrap_f1["GPT"]
    )

    difference_records.append({
        "dataset": analysis_group,
        "filtered_minus_gpt_f1": (
            point_f1[
                "CV Filtered Hybrid"
            ]
            - point_f1["GPT"]
        ),
        "difference_ci_lower": (
            np.percentile(
                f1_difference,
                2.5,
            )
        ),
        "difference_ci_upper": (
            np.percentile(
                f1_difference,
                97.5,
            )
        ),
        "probability_filtered_better": (
            (f1_difference > 0).mean()
        ),
    })


metrics = pd.DataFrame(
    metric_records
)

differences = pd.DataFrame(
    difference_records
)

metrics_path = (
    OUTPUT_DIR
    / "bootstrap_system_confidence_intervals_50.csv"
)

differences_path = (
    OUTPUT_DIR
    / "bootstrap_filtered_vs_gpt_difference_50.csv"
)

per_resume_path = (
    OUTPUT_DIR
    / "bootstrap_per_resume_strict_counts_50.csv"
)

metrics.to_csv(
    metrics_path,
    index=False,
)

differences.to_csv(
    differences_path,
    index=False,
)

per_resume.to_csv(
    per_resume_path,
    index=False,
)


metrics_display = metrics.copy()

metric_columns = [
    column
    for column in metrics_display.columns
    if column not in {
        "dataset",
        "system",
    }
]

metrics_display[metric_columns] = (
    metrics_display[
        metric_columns
    ].round(3)
)

differences_display = differences.copy()

difference_columns = [
    column
    for column in differences_display.columns
    if column != "dataset"
]

differences_display[
    difference_columns
] = differences_display[
    difference_columns
].round(3)


print(
    "\nBOOTSTRAP 95% CONFIDENCE INTERVALS\n"
)

print(
    metrics_display.to_string(
        index=False
    )
)

print(
    "\nFILTERED HYBRID MINUS GPT F1\n"
)

print(
    differences_display.to_string(
        index=False
    )
)

print("\nSaved:")
print(metrics_path)
print(differences_path)
print(per_resume_path)
