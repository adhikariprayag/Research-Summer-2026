from pathlib import Path

import pandas as pd


# Locate the main project folder automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETAIL_FILES = [
    PROJECT_ROOT
    / "07_results"
    / "ocr_evaluation_outputs"
    / "ocr_gold_vs_combined_details.csv",

    PROJECT_ROOT
    / "07_results"
    / "week4_evaluation_outputs"
    / "week4_gold_vs_combined_details.csv",

    PROJECT_ROOT
    / "07_results"
    / "batch2_evaluation_outputs"
    / "batch2_gold_vs_combined_details.csv",
]

RESUME_RESULTS_FILE = (
    PROJECT_ROOT
    / "07_results"
    / "final_20_evaluation"
    / "final_20_per_resume_results.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "07_results"
    / "week5_6_entity_analysis"
    / "per_entity_results"
)

ENTITY_ORDER = ["PERSON", "ORG", "DATE", "GPE", "SKILL"]


def calculate_metrics(group: pd.DataFrame) -> pd.Series:
    """Calculate strict exact-match evaluation metrics."""

    correct = int((group["status"] == "Correct").sum())
    missed = int((group["status"] == "Missed").sum())
    incorrect = int((group["status"] == "Incorrect").sum())

    gold_count = correct + missed
    system_count = correct + incorrect

    precision = correct / system_count if system_count > 0 else 0.0
    recall = correct / gold_count if gold_count > 0 else 0.0

    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    return pd.Series(
        {
            "gold_count": gold_count,
            "system_count": system_count,
            "correct": correct,
            "missed": missed,
            "incorrect": incorrect,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
        }
    )


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load the detailed evaluation files.
    detail_frames = []

    for file_path in DETAIL_FILES:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing detail file: {file_path}")

        df = pd.read_csv(file_path)
        df["source_file"] = file_path.name
        detail_frames.append(df)

    details = pd.concat(detail_frames, ignore_index=True)

    # Standardize text values.
    details["resume_id"] = details["resume_id"].astype(str).str.strip()
    details["entity_text"] = details["entity_text"].astype(str).str.strip()
    details["entity_type"] = (
        details["entity_type"].astype(str).str.strip().str.upper()
    )
    details["status"] = details["status"].astype(str).str.strip().str.title()

    # Load the final 20 resume IDs and their categories.
    resume_results = pd.read_csv(RESUME_RESULTS_FILE)

    resume_results["resume_id"] = (
        resume_results["resume_id"].astype(str).str.strip()
    )
    resume_results["resume_type"] = (
        resume_results["resume_type"].astype(str).str.strip().str.title()
    )

    resume_categories = resume_results[
        ["resume_id", "resume_type"]
    ].drop_duplicates()

    final_resume_ids = set(resume_categories["resume_id"])

    # Keep only resumes that belong to the final 20-resume experiment.
    details = details[details["resume_id"].isin(final_resume_ids)].copy()

    # Attach Standard or Creative labels.
    details = details.merge(
        resume_categories,
        on="resume_id",
        how="left",
        validate="many_to_one",
    )

    # Validate entity and status values.
    unexpected_entities = sorted(
        set(details["entity_type"]) - set(ENTITY_ORDER)
    )

    unexpected_statuses = sorted(
        set(details["status"]) - {"Correct", "Missed", "Incorrect"}
    )

    if unexpected_entities:
        raise ValueError(
            f"Unexpected entity types found: {unexpected_entities}"
        )

    if unexpected_statuses:
        raise ValueError(
            f"Unexpected status values found: {unexpected_statuses}"
        )

    missing_resume_ids = final_resume_ids - set(details["resume_id"])

    if missing_resume_ids:
        raise ValueError(
            f"Final resumes missing from detail files: "
            f"{sorted(missing_resume_ids)}"
        )

    # Overall results for each entity type.
    overall_results = (
        details.groupby("entity_type", sort=False)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )

    # Standard versus Creative results for each entity type.
    group_results = (
        details.groupby(["resume_type", "entity_type"], sort=False)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )

    # Per-resume results for each entity type.
    per_resume_results = (
        details.groupby(
            ["resume_id", "resume_type", "entity_type"],
            sort=False,
        )
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )

    entity_rank = {
        entity_type: index
        for index, entity_type in enumerate(ENTITY_ORDER)
    }

    type_rank = {"Standard": 0, "Creative": 1}

    overall_results["_entity_rank"] = overall_results["entity_type"].map(
        entity_rank
    )
    overall_results = (
        overall_results.sort_values("_entity_rank")
        .drop(columns="_entity_rank")
        .reset_index(drop=True)
    )

    group_results["_type_rank"] = group_results["resume_type"].map(type_rank)
    group_results["_entity_rank"] = group_results["entity_type"].map(
        entity_rank
    )
    group_results = (
        group_results.sort_values(["_type_rank", "_entity_rank"])
        .drop(columns=["_type_rank", "_entity_rank"])
        .reset_index(drop=True)
    )

    per_resume_results["_entity_rank"] = per_resume_results[
        "entity_type"
    ].map(entity_rank)

    per_resume_results = (
        per_resume_results.sort_values(
            ["resume_type", "resume_id", "_entity_rank"]
        )
        .drop(columns="_entity_rank")
        .reset_index(drop=True)
    )

    # Save all results.
    overall_output = OUTPUT_FOLDER / "per_entity_overall.csv"
    group_output = OUTPUT_FOLDER / "per_entity_by_resume_type.csv"
    resume_output = OUTPUT_FOLDER / "per_entity_per_resume.csv"
    combined_details_output = OUTPUT_FOLDER / "final_20_entity_details.csv"

    overall_results.to_csv(overall_output, index=False)
    group_results.to_csv(group_output, index=False)
    per_resume_results.to_csv(resume_output, index=False)
    details.to_csv(combined_details_output, index=False)

    print("Entity-level evaluation completed.")
    print(f"Final resumes analyzed: {details['resume_id'].nunique()}")
    print(f"Detailed evaluation rows: {len(details)}")

    print("\nOverall per-entity results:")
    print(overall_results.to_string(index=False))

    print("\nFiles created:")
    print(overall_output)
    print(group_output)
    print(resume_output)
    print(combined_details_output)


if __name__ == "__main__":
    main()
