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

OUTPUT_DIR = Path(
    "07_results/hybrid_manual_evaluation_50"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(
    dataset: str,
    system: str,
    predictions: int,
    correct: int,
    incorrect: int,
    partial: int,
    missed_by_both: int,
    recovered_by_other_system: int,
) -> dict:
    """
    recovered_by_other_system means valid skills found only by the
    alternative system.

    For V4:
        Valid GPT-only predictions are skills V4 missed.

    For GPT:
        Valid V4-only predictions are skills GPT missed.

    For Hybrid:
        This value is zero because the hybrid includes both systems.
    """

    # Strict scoring:
    # Partial predictions count as one FP and one FN.
    strict_tp = correct
    strict_fp = incorrect + partial
    strict_fn = (
        missed_by_both
        + recovered_by_other_system
        + partial
    )

    strict_precision = safe_divide(
        strict_tp,
        strict_tp + strict_fp,
    )

    strict_recall = safe_divide(
        strict_tp,
        strict_tp + strict_fn,
    )

    strict_f1 = safe_divide(
        2 * strict_precision * strict_recall,
        strict_precision + strict_recall,
    )

    # Relaxed scoring:
    # Partial predictions are accepted as correct.
    relaxed_tp = correct + partial
    relaxed_fp = incorrect
    relaxed_fn = (
        missed_by_both
        + recovered_by_other_system
    )

    relaxed_precision = safe_divide(
        relaxed_tp,
        relaxed_tp + relaxed_fp,
    )

    relaxed_recall = safe_divide(
        relaxed_tp,
        relaxed_tp + relaxed_fn,
    )

    relaxed_f1 = safe_divide(
        2 * relaxed_precision * relaxed_recall,
        relaxed_precision + relaxed_recall,
    )

    gold_skills = strict_tp + strict_fn

    return {
        "dataset": dataset,
        "system": system,
        "predictions": predictions,
        "gold_skills": gold_skills,
        "correct": correct,
        "incorrect": incorrect,
        "partial": partial,
        "missed_by_both": missed_by_both,
        "recovered_by_other_system": recovered_by_other_system,
        "strict_tp": strict_tp,
        "strict_fp": strict_fp,
        "strict_fn": strict_fn,
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": strict_f1,
        "relaxed_tp": relaxed_tp,
        "relaxed_fp": relaxed_fp,
        "relaxed_fn": relaxed_fn,
        "relaxed_precision": relaxed_precision,
        "relaxed_recall": relaxed_recall,
        "relaxed_f1": relaxed_f1,
    }


review = pd.read_csv(REVIEW_PATH)

required_review_columns = {
    "resume_id",
    "source_dataset",
    "prediction_source",
    "review_decision",
}

missing_review_columns = (
    required_review_columns - set(review.columns)
)

if missing_review_columns:
    raise ValueError(
        "Review file is missing columns: "
        f"{sorted(missing_review_columns)}"
    )


review["source_dataset"] = (
    review["source_dataset"]
    .fillna("")
    .astype(str)
    .str.strip()
)

review["prediction_source_clean"] = (
    review["prediction_source"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

review["review_decision_clean"] = (
    review["review_decision"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)


valid_sources = {"v4", "gpt", "both"}

invalid_sources = sorted(
    set(review["prediction_source_clean"]) - valid_sources
)

if invalid_sources:
    raise ValueError(
        f"Invalid prediction sources found: {invalid_sources}"
    )


valid_decisions = {"correct", "incorrect", "partial"}

invalid_decisions = sorted(
    set(review["review_decision_clean"]) - valid_decisions
)

if invalid_decisions:
    raise ValueError(
        f"Invalid review decisions found: {invalid_decisions}"
    )


missed = pd.read_csv(MISSED_PATH)

required_missed_columns = {
    "resume_id",
    "source_dataset",
    "missed_entity_text",
}

missing_missed_columns = (
    required_missed_columns - set(missed.columns)
)

if missing_missed_columns:
    raise ValueError(
        "Missed-skill file is missing columns: "
        f"{sorted(missing_missed_columns)}"
    )


missed["source_dataset"] = (
    missed["source_dataset"]
    .fillna("")
    .astype(str)
    .str.strip()
)

missed["missed_entity_text_clean"] = (
    missed["missed_entity_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

missed = missed[
    missed["missed_entity_text_clean"].ne("")
].copy()

# Avoid duplicate missed skills within one resume.
missed["missed_norm"] = (
    missed["missed_entity_text_clean"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)

missed = missed.drop_duplicates(
    subset=["resume_id", "missed_norm"]
)


dataset_order = list(
    review["source_dataset"].drop_duplicates()
)

results = []


def evaluate_dataset(
    dataset_name: str,
    dataset_review: pd.DataFrame,
    dataset_missed: pd.DataFrame,
) -> None:
    missed_by_both = len(dataset_missed)

    system_definitions = {
        "V4": {
            "included_sources": {"v4", "both"},
            "other_only_source": "gpt",
        },
        "GPT": {
            "included_sources": {"gpt", "both"},
            "other_only_source": "v4",
        },
        "Hybrid V4 + GPT": {
            "included_sources": {"v4", "gpt", "both"},
            "other_only_source": None,
        },
    }

    for system_name, settings in system_definitions.items():
        system_rows = dataset_review[
            dataset_review["prediction_source_clean"].isin(
                settings["included_sources"]
            )
        ].copy()

        decision_counts = (
            system_rows["review_decision_clean"]
            .value_counts()
            .to_dict()
        )

        correct = int(decision_counts.get("correct", 0))
        incorrect = int(decision_counts.get("incorrect", 0))
        partial = int(decision_counts.get("partial", 0))

        other_only_source = settings["other_only_source"]

        if other_only_source is None:
            recovered_by_other_system = 0
        else:
            other_only_valid = dataset_review[
                dataset_review["prediction_source_clean"].eq(
                    other_only_source
                )
                & dataset_review["review_decision_clean"].isin(
                    {"correct", "partial"}
                )
            ]

            recovered_by_other_system = len(
                other_only_valid
            )

        results.append(
            calculate_metrics(
                dataset=dataset_name,
                system=system_name,
                predictions=len(system_rows),
                correct=correct,
                incorrect=incorrect,
                partial=partial,
                missed_by_both=missed_by_both,
                recovered_by_other_system=(
                    recovered_by_other_system
                ),
            )
        )


for dataset in dataset_order:
    dataset_review = review[
        review["source_dataset"].eq(dataset)
    ].copy()

    dataset_missed = missed[
        missed["source_dataset"].eq(dataset)
    ].copy()

    evaluate_dataset(
        dataset_name=dataset,
        dataset_review=dataset_review,
        dataset_missed=dataset_missed,
    )


evaluate_dataset(
    dataset_name="Combined",
    dataset_review=review,
    dataset_missed=missed,
)


comparison = pd.DataFrame(results)


# Confirm each system is evaluated against the same gold total
# within each dataset.
for dataset, group in comparison.groupby("dataset"):
    gold_counts = group["gold_skills"].unique()

    if len(gold_counts) != 1:
        raise ValueError(
            f"Gold-skill totals do not match for {dataset}: "
            f"{gold_counts.tolist()}"
        )


by_dataset_path = (
    OUTPUT_DIR
    / "v4_gpt_hybrid_comparison_by_dataset_50.csv"
)

overall_path = (
    OUTPUT_DIR
    / "v4_gpt_hybrid_comparison_overall_50.csv"
)

comparison.to_csv(
    by_dataset_path,
    index=False,
)

comparison[
    comparison["dataset"].eq("Combined")
].to_csv(
    overall_path,
    index=False,
)


display_columns = [
    "dataset",
    "system",
    "predictions",
    "gold_skills",
    "correct",
    "incorrect",
    "partial",
    "missed_by_both",
    "recovered_by_other_system",
    "strict_precision",
    "strict_recall",
    "strict_f1",
    "relaxed_precision",
    "relaxed_recall",
    "relaxed_f1",
]

display = comparison[display_columns].copy()

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
    "\nV4 VS GPT VS HYBRID — "
    "50-RESUME MANUAL EVALUATION\n"
)

print(display.to_string(index=False))

print("\nOVERALL STRICT COMPARISON\n")

overall_display = display[
    display["dataset"].eq("Combined")
][
    [
        "system",
        "predictions",
        "gold_skills",
        "strict_precision",
        "strict_recall",
        "strict_f1",
    ]
]

print(overall_display.to_string(index=False))

print("\nSaved:")
print(by_dataset_path)
print(overall_path)
