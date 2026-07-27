from pathlib import Path
from collections import Counter

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
RANDOM_SEED = 42


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_f1(precision: float, recall: float) -> float:
    return safe_divide(
        2 * precision * recall,
        precision + recall,
    )


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def evaluate_system(
    fold: int,
    system: str,
    test_rows: pd.DataFrame,
    test_missed: pd.DataFrame,
    keep_mask: pd.Series,
    blacklist_size: int,
) -> dict:
    kept = test_rows[keep_mask].copy()
    removed = test_rows[~keep_mask].copy()

    all_valid_predictions = test_rows[
        test_rows["decision_clean"].isin(
            {"correct", "partial"}
        )
    ]

    gold_skills = len(all_valid_predictions) + len(test_missed)

    correct = int(
        kept["decision_clean"].eq("correct").sum()
    )

    incorrect = int(
        kept["decision_clean"].eq("incorrect").sum()
    )

    partial = int(
        kept["decision_clean"].eq("partial").sum()
    )

    removed_correct = int(
        removed["decision_clean"].eq("correct").sum()
    )

    removed_incorrect = int(
        removed["decision_clean"].eq("incorrect").sum()
    )

    removed_partial = int(
        removed["decision_clean"].eq("partial").sum()
    )

    # Strict exact-match scoring:
    # partial = one false positive and one false negative.
    strict_tp = correct
    strict_fp = incorrect + partial
    strict_fn = gold_skills - strict_tp

    strict_precision = safe_divide(
        strict_tp,
        strict_tp + strict_fp,
    )

    strict_recall = safe_divide(
        strict_tp,
        strict_tp + strict_fn,
    )

    strict_f1 = calculate_f1(
        strict_precision,
        strict_recall,
    )

    # Relaxed scoring:
    # partial predictions are accepted as correct.
    relaxed_tp = correct + partial
    relaxed_fp = incorrect
    relaxed_fn = gold_skills - relaxed_tp

    relaxed_precision = safe_divide(
        relaxed_tp,
        relaxed_tp + relaxed_fp,
    )

    relaxed_recall = safe_divide(
        relaxed_tp,
        relaxed_tp + relaxed_fn,
    )

    relaxed_f1 = calculate_f1(
        relaxed_precision,
        relaxed_recall,
    )

    return {
        "fold": fold,
        "system": system,
        "test_resumes": test_rows["resume_id"].nunique(),
        "predictions": len(kept),
        "gold_skills": gold_skills,
        "correct": correct,
        "incorrect": incorrect,
        "partial": partial,
        "missed_by_all_systems": len(test_missed),
        "removed_correct": removed_correct,
        "removed_incorrect": removed_incorrect,
        "removed_partial": removed_partial,
        "blacklist_size": blacklist_size,
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
missed = pd.read_csv(MISSED_PATH)

required_review_columns = {
    "resume_id",
    "source_dataset",
    "prediction_source",
    "review_decision",
    "entity_text",
}

missing_review_columns = (
    required_review_columns - set(review.columns)
)

if missing_review_columns:
    raise ValueError(
        "Review file is missing columns: "
        f"{sorted(missing_review_columns)}"
    )

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

review["source_clean"] = normalize_text(
    review["prediction_source"]
)

review["decision_clean"] = normalize_text(
    review["review_decision"]
)

review["entity_norm"] = normalize_text(
    review["entity_text"]
)


valid_sources = {"v4", "gpt", "both"}
invalid_sources = sorted(
    set(review["source_clean"]) - valid_sources
)

if invalid_sources:
    raise ValueError(
        f"Invalid prediction sources: {invalid_sources}"
    )


valid_decisions = {"correct", "incorrect", "partial"}
invalid_decisions = sorted(
    set(review["decision_clean"]) - valid_decisions
)

if invalid_decisions:
    raise ValueError(
        f"Invalid review decisions: {invalid_decisions}"
    )


missed["resume_id"] = (
    missed["resume_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

missed["source_dataset"] = (
    missed["source_dataset"]
    .fillna("")
    .astype(str)
    .str.strip()
)

missed["missed_norm"] = normalize_text(
    missed["missed_entity_text"]
)

missed = missed[
    missed["missed_norm"].ne("")
].drop_duplicates(
    subset=["resume_id", "missed_norm"]
).copy()


# Build one row per resume.
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

if resume_metadata["resume_id"].duplicated().any():
    raise ValueError(
        "A resume has inconsistent source or band metadata."
    )


# Use source and prediction band as strata when available.
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

if fold_map["resume_id"].duplicated().any():
    raise ValueError(
        "A resume was assigned to more than one fold."
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

if review["fold"].isna().any():
    raise ValueError(
        "Some review rows were not assigned a fold."
    )


results = []
blacklist_records = []


for fold in range(1, N_FOLDS + 1):
    train_rows = review[
        review["fold"].ne(fold)
    ].copy()

    test_rows = review[
        review["fold"].eq(fold)
    ].copy()

    test_missed = missed[
        missed["fold"].eq(fold)
    ].copy()

    train_v4_only = train_rows[
        train_rows["source_clean"].eq("v4")
    ].copy()

    term_quality = (
        train_v4_only.groupby(
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

    # Conservative learned blacklist:
    # - term must appear at least twice in training
    # - incorrect examples must outnumber valid examples
    blacklist_rows = term_quality[
        term_quality["total"].ge(2)
        & term_quality["incorrect"].gt(
            term_quality["valid"]
        )
    ].copy()

    blacklist = set(
        blacklist_rows["entity_norm"]
    )

    for term in sorted(blacklist):
        term_row = blacklist_rows[
            blacklist_rows["entity_norm"].eq(term)
        ].iloc[0]

        blacklist_records.append({
            "fold": fold,
            "entity_norm": term,
            "training_total": int(
                term_row["total"]
            ),
            "training_valid": int(
                term_row["valid"]
            ),
            "training_incorrect": int(
                term_row["incorrect"]
            ),
        })

    raw_hybrid_mask = pd.Series(
        True,
        index=test_rows.index,
    )

    gpt_mask = test_rows["source_clean"].isin(
        {"gpt", "both"}
    )

    filtered_hybrid_mask = (
        test_rows["source_clean"].isin(
            {"gpt", "both"}
        )
        |
        (
            test_rows["source_clean"].eq("v4")
            & ~test_rows["entity_norm"].isin(
                blacklist
            )
        )
    )

    results.append(
        evaluate_system(
            fold=fold,
            system="GPT",
            test_rows=test_rows,
            test_missed=test_missed,
            keep_mask=gpt_mask,
            blacklist_size=len(blacklist),
        )
    )

    results.append(
        evaluate_system(
            fold=fold,
            system="Raw Hybrid",
            test_rows=test_rows,
            test_missed=test_missed,
            keep_mask=raw_hybrid_mask,
            blacklist_size=len(blacklist),
        )
    )

    results.append(
        evaluate_system(
            fold=fold,
            system="CV Filtered Hybrid",
            test_rows=test_rows,
            test_missed=test_missed,
            keep_mask=filtered_hybrid_mask,
            blacklist_size=len(blacklist),
        )
    )


fold_results = pd.DataFrame(results)


# Aggregate counts across all five held-out folds,
# then calculate metrics from the combined counts.
overall_rows = []

for system, group in fold_results.groupby(
    "system",
    sort=False,
):
    predictions = int(group["predictions"].sum())
    gold_skills = int(group["gold_skills"].sum())
    correct = int(group["correct"].sum())
    incorrect = int(group["incorrect"].sum())
    partial = int(group["partial"].sum())

    strict_tp = correct
    strict_fp = incorrect + partial
    strict_fn = gold_skills - strict_tp

    strict_precision = safe_divide(
        strict_tp,
        strict_tp + strict_fp,
    )

    strict_recall = safe_divide(
        strict_tp,
        strict_tp + strict_fn,
    )

    strict_f1 = calculate_f1(
        strict_precision,
        strict_recall,
    )

    relaxed_tp = correct + partial
    relaxed_fp = incorrect
    relaxed_fn = gold_skills - relaxed_tp

    relaxed_precision = safe_divide(
        relaxed_tp,
        relaxed_tp + relaxed_fp,
    )

    relaxed_recall = safe_divide(
        relaxed_tp,
        relaxed_tp + relaxed_fn,
    )

    relaxed_f1 = calculate_f1(
        relaxed_precision,
        relaxed_recall,
    )

    overall_rows.append({
        "system": system,
        "predictions": predictions,
        "gold_skills": gold_skills,
        "correct": correct,
        "incorrect": incorrect,
        "partial": partial,
        "removed_correct": int(
            group["removed_correct"].sum()
        ),
        "removed_incorrect": int(
            group["removed_incorrect"].sum()
        ),
        "removed_partial": int(
            group["removed_partial"].sum()
        ),
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": strict_f1,
        "relaxed_precision": relaxed_precision,
        "relaxed_recall": relaxed_recall,
        "relaxed_f1": relaxed_f1,
    })


overall = pd.DataFrame(overall_rows)


blacklist_details = pd.DataFrame(
    blacklist_records
)

if not blacklist_details.empty:
    blacklist_stability = (
        blacklist_details.groupby(
            "entity_norm",
            as_index=False,
        )
        .agg(
            blacklisted_folds=("fold", "nunique"),
            mean_training_total=(
                "training_total",
                "mean",
            ),
            mean_training_valid=(
                "training_valid",
                "mean",
            ),
            mean_training_incorrect=(
                "training_incorrect",
                "mean",
            ),
        )
        .sort_values(
            [
                "blacklisted_folds",
                "mean_training_incorrect",
            ],
            ascending=False,
        )
    )
else:
    blacklist_stability = pd.DataFrame(
        columns=[
            "entity_norm",
            "blacklisted_folds",
            "mean_training_total",
            "mean_training_valid",
            "mean_training_incorrect",
        ]
    )


fold_path = (
    OUTPUT_DIR
    / "filtered_hybrid_cv_fold_metrics_50.csv"
)

overall_path = (
    OUTPUT_DIR
    / "filtered_hybrid_cv_overall_50.csv"
)

blacklist_path = (
    OUTPUT_DIR
    / "filtered_hybrid_cv_blacklist_stability_50.csv"
)

fold_results.to_csv(
    fold_path,
    index=False,
)

overall.to_csv(
    overall_path,
    index=False,
)

blacklist_stability.to_csv(
    blacklist_path,
    index=False,
)


display = overall.copy()

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
    "\n5-FOLD RESUME-LEVEL CROSS-VALIDATION\n"
)

print(
    display[
        [
            "system",
            "predictions",
            "gold_skills",
            "correct",
            "incorrect",
            "partial",
            "removed_correct",
            "removed_incorrect",
            "strict_precision",
            "strict_recall",
            "strict_f1",
            "relaxed_precision",
            "relaxed_recall",
            "relaxed_f1",
        ]
    ].to_string(index=False)
)

print(
    "\nMOST STABLE LEARNED BLACKLIST TERMS\n"
)

if blacklist_stability.empty:
    print("No terms met the blacklist rule.")
else:
    blacklist_display = (
        blacklist_stability.head(25).copy()
    )

    numeric_columns = [
        "mean_training_total",
        "mean_training_valid",
        "mean_training_incorrect",
    ]

    blacklist_display[numeric_columns] = (
        blacklist_display[numeric_columns]
        .round(2)
    )

    print(
        blacklist_display.to_string(
            index=False
        )
    )

print("\nSaved:")
print(fold_path)
print(overall_path)
print(blacklist_path)
