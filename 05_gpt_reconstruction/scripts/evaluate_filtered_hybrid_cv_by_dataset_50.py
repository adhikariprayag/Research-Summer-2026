from pathlib import Path
import pandas as pd


REVIEW_PATH = Path(
    "02_dataset/large_scale_dataset/full_v4_private/"
    "validation_review_50/hybrid_prediction_review_template_50.csv"
)

MISSED_PATH = Path(
    "02_dataset/large_scale_dataset/full_v4_private/"
    "validation_review_50/missed_skill_review_template_50.csv"
)

OUTPUT_PATH = Path(
    "07_results/hybrid_manual_evaluation_50/"
    "filtered_hybrid_cv_by_dataset_50.csv"
)

N_FOLDS = 5
RANDOM_SEED = 42


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def calculate_f1(precision, recall):
    return safe_divide(
        2 * precision * recall,
        precision + recall,
    )


def normalize(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


review = pd.read_csv(REVIEW_PATH)
missed = pd.read_csv(MISSED_PATH)

review["resume_id"] = (
    review["resume_id"].astype(str).str.strip()
)

review["source_dataset"] = (
    review["source_dataset"].astype(str).str.strip()
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
    missed["resume_id"].astype(str).str.strip()
)

missed["source_dataset"] = (
    missed["source_dataset"].astype(str).str.strip()
)

missed["missed_norm"] = normalize(
    missed["missed_entity_text"]
)

missed = missed[
    missed["missed_norm"].ne("")
].drop_duplicates(
    subset=["resume_id", "missed_norm"]
)


metadata_columns = [
    "resume_id",
    "source_dataset",
]

if "prediction_band" in review.columns:
    metadata_columns.append("prediction_band")

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


fold_assignments = []

for stratum_number, (_, group) in enumerate(
    resume_metadata.groupby(
        strata_columns,
        sort=True,
        dropna=False,
    )
):
    shuffled = group.sample(
        frac=1,
        random_state=RANDOM_SEED + stratum_number,
    ).reset_index(drop=True)

    shuffled["fold"] = (
        shuffled.index % N_FOLDS
    ) + 1

    fold_assignments.append(
        shuffled[["resume_id", "fold"]]
    )


fold_map = pd.concat(
    fold_assignments,
    ignore_index=True,
)

review = review.merge(
    fold_map,
    on="resume_id",
    how="left",
    validate="many_to_one",
)

missed = missed.merge(
    fold_map,
    on="resume_id",
    how="left",
    validate="many_to_one",
)


records = []

systems = [
    "GPT",
    "Raw Hybrid",
    "CV Filtered Hybrid",
]


def evaluate_subset(
    fold,
    dataset,
    system,
    rows,
    missed_rows,
    keep_mask,
):
    kept = rows[keep_mask]

    gold_skills = (
        rows["decision_clean"]
        .isin({"correct", "partial"})
        .sum()
        + len(missed_rows)
    )

    correct = int(
        kept["decision_clean"].eq("correct").sum()
    )

    incorrect = int(
        kept["decision_clean"].eq("incorrect").sum()
    )

    partial = int(
        kept["decision_clean"].eq("partial").sum()
    )

    strict_tp = correct
    strict_fp = incorrect + partial
    strict_fn = gold_skills - strict_tp

    relaxed_tp = correct + partial
    relaxed_fp = incorrect
    relaxed_fn = gold_skills - relaxed_tp

    records.append({
        "fold": fold,
        "dataset": dataset,
        "system": system,
        "predictions": len(kept),
        "gold_skills": gold_skills,
        "correct": correct,
        "incorrect": incorrect,
        "partial": partial,
        "strict_tp": strict_tp,
        "strict_fp": strict_fp,
        "strict_fn": strict_fn,
        "relaxed_tp": relaxed_tp,
        "relaxed_fp": relaxed_fp,
        "relaxed_fn": relaxed_fn,
    })


for fold in range(1, N_FOLDS + 1):
    train = review[
        review["fold"].ne(fold)
        & review["source_clean"].eq("v4")
    ].copy()

    test = review[
        review["fold"].eq(fold)
    ].copy()

    test_missed = missed[
        missed["fold"].eq(fold)
    ].copy()

    quality = (
        train.groupby(
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
        quality[
            quality["total"].ge(2)
            & quality["incorrect"].gt(
                quality["valid"]
            )
        ]["entity_norm"]
    )

    datasets = list(
        test["source_dataset"].drop_duplicates()
    )

    datasets.append("Combined")

    for dataset in datasets:
        if dataset == "Combined":
            rows = test.copy()
            missed_rows = test_missed.copy()
        else:
            rows = test[
                test["source_dataset"].eq(dataset)
            ].copy()

            missed_rows = test_missed[
                test_missed["source_dataset"].eq(
                    dataset
                )
            ].copy()

        gpt_mask = rows["source_clean"].isin(
            {"gpt", "both"}
        )

        raw_mask = pd.Series(
            True,
            index=rows.index,
        )

        filtered_mask = (
            rows["source_clean"].isin(
                {"gpt", "both"}
            )
            |
            (
                rows["source_clean"].eq("v4")
                & ~rows["entity_norm"].isin(
                    blacklist
                )
            )
        )

        evaluate_subset(
            fold,
            dataset,
            "GPT",
            rows,
            missed_rows,
            gpt_mask,
        )

        evaluate_subset(
            fold,
            dataset,
            "Raw Hybrid",
            rows,
            missed_rows,
            raw_mask,
        )

        evaluate_subset(
            fold,
            dataset,
            "CV Filtered Hybrid",
            rows,
            missed_rows,
            filtered_mask,
        )


fold_results = pd.DataFrame(records)

aggregate = (
    fold_results.groupby(
        ["dataset", "system"],
        as_index=False,
    )
    .agg(
        predictions=("predictions", "sum"),
        gold_skills=("gold_skills", "sum"),
        correct=("correct", "sum"),
        incorrect=("incorrect", "sum"),
        partial=("partial", "sum"),
        strict_tp=("strict_tp", "sum"),
        strict_fp=("strict_fp", "sum"),
        strict_fn=("strict_fn", "sum"),
        relaxed_tp=("relaxed_tp", "sum"),
        relaxed_fp=("relaxed_fp", "sum"),
        relaxed_fn=("relaxed_fn", "sum"),
    )
)

aggregate["strict_precision"] = (
    aggregate["strict_tp"]
    / (
        aggregate["strict_tp"]
        + aggregate["strict_fp"]
    )
)

aggregate["strict_recall"] = (
    aggregate["strict_tp"]
    / (
        aggregate["strict_tp"]
        + aggregate["strict_fn"]
    )
)

aggregate["strict_f1"] = (
    2
    * aggregate["strict_precision"]
    * aggregate["strict_recall"]
    / (
        aggregate["strict_precision"]
        + aggregate["strict_recall"]
    )
)

aggregate["relaxed_precision"] = (
    aggregate["relaxed_tp"]
    / (
        aggregate["relaxed_tp"]
        + aggregate["relaxed_fp"]
    )
)

aggregate["relaxed_recall"] = (
    aggregate["relaxed_tp"]
    / (
        aggregate["relaxed_tp"]
        + aggregate["relaxed_fn"]
    )
)

aggregate["relaxed_f1"] = (
    2
    * aggregate["relaxed_precision"]
    * aggregate["relaxed_recall"]
    / (
        aggregate["relaxed_precision"]
        + aggregate["relaxed_recall"]
    )
)

system_order = {
    "GPT": 1,
    "Raw Hybrid": 2,
    "CV Filtered Hybrid": 3,
}

aggregate["system_order"] = (
    aggregate["system"].map(system_order)
)

aggregate = aggregate.sort_values(
    ["dataset", "system_order"]
).drop(columns=["system_order"])

aggregate.to_csv(
    OUTPUT_PATH,
    index=False,
)

display = aggregate[
    [
        "dataset",
        "system",
        "predictions",
        "gold_skills",
        "strict_precision",
        "strict_recall",
        "strict_f1",
        "relaxed_precision",
        "relaxed_recall",
        "relaxed_f1",
    ]
].copy()

metric_columns = [
    "strict_precision",
    "strict_recall",
    "strict_f1",
    "relaxed_precision",
    "relaxed_recall",
    "relaxed_f1",
]

display[metric_columns] = (
    display[metric_columns].round(3)
)

print(
    "\nCROSS-VALIDATED RESULTS BY DATASET\n"
)

print(display.to_string(index=False))

print("\nSaved:")
print(OUTPUT_PATH)
