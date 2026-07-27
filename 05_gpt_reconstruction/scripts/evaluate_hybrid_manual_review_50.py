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


def source_from_resume_id(resume_id: str) -> str:
    """Derive the dataset source from the anonymized resume ID."""
    resume_id = str(resume_id).strip().upper()

    if resume_id.startswith("JIT_"):
        return "Jithin"
    if resume_id.startswith("SAU_"):
        return "Saugata"
    if resume_id.startswith("SNE_"):
        return "Snehaan"

    return "Unknown"


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(
    dataset: str,
    correct: int,
    incorrect: int,
    partial: int,
    missed: int,
) -> dict:
    predictions = correct + incorrect + partial
    gold_skills = correct + partial + missed

    # Strict exact-match scoring:
    # A partial match is one false positive and one false negative.
    strict_tp = correct
    strict_fp = incorrect + partial
    strict_fn = missed + partial

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
    # A partial prediction is accepted as correct.
    relaxed_tp = correct + partial
    relaxed_fp = incorrect
    relaxed_fn = missed

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

    return {
        "dataset": dataset,
        "predictions": predictions,
        "gold_skills": gold_skills,
        "correct": correct,
        "incorrect": incorrect,
        "partial": partial,
        "missed": missed,
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
    "review_decision",
}

missing_columns = required_review_columns - set(review.columns)

if missing_columns:
    raise ValueError(
        f"Review file is missing columns: {sorted(missing_columns)}"
    )

review["review_decision_clean"] = (
    review["review_decision"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

unreviewed_count = review["review_decision_clean"].eq("").sum()

if unreviewed_count:
    raise ValueError(
        f"{unreviewed_count} prediction rows are still unreviewed."
    )

valid_decisions = {"correct", "incorrect", "partial"}

invalid_decisions = sorted(
    set(review["review_decision_clean"]) - valid_decisions
)

if invalid_decisions:
    raise ValueError(
        f"Invalid review decisions found: {invalid_decisions}"
    )

if "source_dataset" not in review.columns:
    review["source_dataset"] = (
        review["resume_id"].apply(source_from_resume_id)
    )
else:
    review["source_dataset"] = (
        review["source_dataset"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    blank_source = review["source_dataset"].eq("")

    review.loc[blank_source, "source_dataset"] = (
        review.loc[blank_source, "resume_id"]
        .apply(source_from_resume_id)
    )


missed = pd.read_csv(MISSED_PATH)

required_missed_columns = {
    "resume_id",
    "missed_entity_text",
}

missing_columns = required_missed_columns - set(missed.columns)

if missing_columns:
    raise ValueError(
        f"Missed-skill file is missing columns: {sorted(missing_columns)}"
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

if "source_dataset" not in missed.columns:
    missed["source_dataset"] = (
        missed["resume_id"].apply(source_from_resume_id)
    )
else:
    missed["source_dataset"] = (
        missed["source_dataset"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    blank_source = missed["source_dataset"].eq("")

    missed.loc[blank_source, "source_dataset"] = (
        missed.loc[blank_source, "resume_id"]
        .apply(source_from_resume_id)
    )

# Avoid accidentally counting the same missed skill twice
# within the same resume.
missed["missed_norm"] = (
    missed["missed_entity_text_clean"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)

missed = missed.drop_duplicates(
    subset=["resume_id", "missed_norm"]
)

datasets = sorted(review["source_dataset"].unique())

results = []

for dataset in datasets:
    dataset_review = review[
        review["source_dataset"].eq(dataset)
    ]

    decision_counts = (
        dataset_review["review_decision_clean"]
        .value_counts()
        .to_dict()
    )

    dataset_missed = missed[
        missed["source_dataset"].eq(dataset)
    ]

    results.append(
        calculate_metrics(
            dataset=dataset,
            correct=int(decision_counts.get("correct", 0)),
            incorrect=int(decision_counts.get("incorrect", 0)),
            partial=int(decision_counts.get("partial", 0)),
            missed=len(dataset_missed),
        )
    )

overall_counts = (
    review["review_decision_clean"]
    .value_counts()
    .to_dict()
)

overall_result = calculate_metrics(
    dataset="Combined",
    correct=int(overall_counts.get("correct", 0)),
    incorrect=int(overall_counts.get("incorrect", 0)),
    partial=int(overall_counts.get("partial", 0)),
    missed=len(missed),
)

results.append(overall_result)

summary = pd.DataFrame(results)

by_dataset_path = (
    OUTPUT_DIR / "hybrid_manual_evaluation_by_dataset_50.csv"
)

overall_path = (
    OUTPUT_DIR / "hybrid_manual_evaluation_overall_50.csv"
)

summary.to_csv(by_dataset_path, index=False)

summary[
    summary["dataset"].eq("Combined")
].to_csv(overall_path, index=False)

display_columns = [
    "dataset",
    "predictions",
    "gold_skills",
    "correct",
    "incorrect",
    "partial",
    "missed",
    "strict_precision",
    "strict_recall",
    "strict_f1",
    "relaxed_precision",
    "relaxed_recall",
    "relaxed_f1",
]

display = summary[display_columns].copy()

metric_columns = [
    "strict_precision",
    "strict_recall",
    "strict_f1",
    "relaxed_precision",
    "relaxed_recall",
    "relaxed_f1",
]

display[metric_columns] = display[metric_columns].round(3)

print("\nHYBRID V4 + GPT MANUAL EVALUATION — 50 RESUMES\n")
print(display.to_string(index=False))

print("\nSaved:")
print(by_dataset_path)
print(overall_path)
