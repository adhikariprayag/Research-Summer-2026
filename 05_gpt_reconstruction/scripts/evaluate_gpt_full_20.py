from pathlib import Path
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GPT_OUTPUT = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "full_outputs"
    / "gpt_full_20_combined_entities.csv"
)

FULL_LIST = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "full_inputs"
    / "gpt_full_20_resume_list.csv"
)

GOLD_STANDARD = (
    PROJECT_ROOT
    / "02_dataset"
    / "gold_standard_annotations"
    / "gold_standard_final_20.csv"
)

BASELINE_RESULTS = (
    PROJECT_ROOT
    / "07_results"
    / "final_20_evaluation"
    / "final_20_per_resume_results.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "05_gpt_reconstruction" / "full_evaluation_outputs"

ENTITY_ORDER = ["PERSON", "ORG", "DATE", "GPE", "SKILL"]


def normalize_text(text: str) -> str:
    """Normalize entity text for strict exact-match comparison."""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def calculate_metrics(correct: int, missed: int, incorrect: int) -> dict:
    gold_count = correct + missed
    system_count = correct + incorrect

    precision = correct / system_count if system_count > 0 else 0.0
    recall = correct / gold_count if gold_count > 0 else 0.0

    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    return {
        "gold_count": gold_count,
        "system_count": system_count,
        "correct": correct,
        "missed": missed,
        "incorrect": incorrect,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
    }


def evaluate_group(gold_df: pd.DataFrame, system_df: pd.DataFrame) -> dict:
    gold_set = set(
        zip(
            gold_df["resume_id"],
            gold_df["normalized_text"],
            gold_df["entity_type"],
        )
    )

    system_set = set(
        zip(
            system_df["resume_id"],
            system_df["normalized_text"],
            system_df["entity_type"],
        )
    )

    correct_items = gold_set & system_set
    missed_items = gold_set - system_set
    incorrect_items = system_set - gold_set

    return calculate_metrics(
        correct=len(correct_items),
        missed=len(missed_items),
        incorrect=len(incorrect_items),
    )


def create_detail_rows(gold_df: pd.DataFrame, system_df: pd.DataFrame) -> pd.DataFrame:
    gold_lookup = {
        (row.resume_id, row.normalized_text, row.entity_type): row.entity_text
        for row in gold_df.itertuples(index=False)
    }

    system_lookup = {
        (row.resume_id, row.normalized_text, row.entity_type): row.entity_text
        for row in system_df.itertuples(index=False)
    }

    gold_set = set(gold_lookup)
    system_set = set(system_lookup)

    rows = []

    for key in sorted(gold_set & system_set):
        resume_id, _, entity_type = key
        rows.append(
            {
                "resume_id": resume_id,
                "entity_text": gold_lookup[key],
                "entity_type": entity_type,
                "status": "Correct",
            }
        )

    for key in sorted(gold_set - system_set):
        resume_id, _, entity_type = key
        rows.append(
            {
                "resume_id": resume_id,
                "entity_text": gold_lookup[key],
                "entity_type": entity_type,
                "status": "Missed",
            }
        )

    for key in sorted(system_set - gold_set):
        resume_id, _, entity_type = key
        rows.append(
            {
                "resume_id": resume_id,
                "entity_text": system_lookup[key],
                "entity_type": entity_type,
                "status": "Incorrect",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    full_list = pd.read_csv(FULL_LIST)
    full_resume_ids = full_list["resume_id"].astype(str).str.strip().tolist()

    gold = pd.read_csv(GOLD_STANDARD)
    gpt = pd.read_csv(GPT_OUTPUT)
    baseline = pd.read_csv(BASELINE_RESULTS)

    gold["resume_id"] = gold["resume_id"].astype(str).str.strip()
    gpt["resume_id"] = gpt["resume_id"].astype(str).str.strip()
    baseline["resume_id"] = baseline["resume_id"].astype(str).str.strip()

    gold["entity_text"] = gold["entity_text"].astype(str).str.strip()
    gpt["entity_text"] = gpt["entity_text"].astype(str).str.strip()

    gold["entity_type"] = gold["entity_type"].astype(str).str.strip().str.upper()
    gpt["entity_type"] = gpt["entity_type"].astype(str).str.strip().str.upper()

    gold = gold[gold["resume_id"].isin(full_resume_ids)].copy()
    gpt = gpt[gpt["resume_id"].isin(full_resume_ids)].copy()

    gold["normalized_text"] = gold["entity_text"].apply(normalize_text)
    gpt["normalized_text"] = gpt["entity_text"].apply(normalize_text)

    # Remove duplicate GPT predictions before evaluation.
    gpt = gpt.drop_duplicates(
        subset=["resume_id", "normalized_text", "entity_type"]
    ).copy()

    # Overall GPT evaluation.
    overall_metrics = evaluate_group(gold, gpt)
    overall_df = pd.DataFrame([{"condition": "GPT Reconstruction", **overall_metrics}])

    # Per-resume GPT evaluation.
    per_resume_rows = []

    for resume_id in full_resume_ids:
        gold_subset = gold[gold["resume_id"] == resume_id]
        gpt_subset = gpt[gpt["resume_id"] == resume_id]

        metrics = evaluate_group(gold_subset, gpt_subset)
        per_resume_rows.append({"resume_id": resume_id, **metrics})

    per_resume_df = pd.DataFrame(per_resume_rows)

    # Per-entity GPT evaluation.
    per_entity_rows = []

    for entity_type in ENTITY_ORDER:
        gold_subset = gold[gold["entity_type"] == entity_type]
        gpt_subset = gpt[gpt["entity_type"] == entity_type]

        metrics = evaluate_group(gold_subset, gpt_subset)
        per_entity_rows.append({"entity_type": entity_type, **metrics})

    per_entity_df = pd.DataFrame(per_entity_rows)

    # Baseline comparison on the same 20 resumes.
    baseline_full = baseline[baseline["resume_id"].isin(full_resume_ids)].copy()

    baseline_correct = int(baseline_full["correct"].sum())
    baseline_missed = int(baseline_full["missed"].sum())
    baseline_incorrect = int(baseline_full["incorrect"].sum())

    baseline_metrics = calculate_metrics(
        correct=baseline_correct,
        missed=baseline_missed,
        incorrect=baseline_incorrect,
    )

    comparison_df = pd.DataFrame(
        [
            {"condition": "spaCy + Skill Baseline", **baseline_metrics},
            {"condition": "GPT Reconstruction", **overall_metrics},
        ]
    )

    details_df = create_detail_rows(gold, gpt)

    # Save outputs.
    overall_path = OUTPUT_FOLDER / "gpt_full_20_overall_summary.csv"
    per_resume_path = OUTPUT_FOLDER / "gpt_full_20_per_resume_summary.csv"
    per_entity_path = OUTPUT_FOLDER / "gpt_full_20_per_entity_summary.csv"
    comparison_path = OUTPUT_FOLDER / "gpt_full_20_vs_baseline_summary.csv"
    details_path = OUTPUT_FOLDER / "gpt_full_20_vs_gold_details.csv"

    overall_df.to_csv(overall_path, index=False)
    per_resume_df.to_csv(per_resume_path, index=False)
    per_entity_df.to_csv(per_entity_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    details_df.to_csv(details_path, index=False)

    print("GPT full 20 evaluation complete.")
    print("=" * 60)

    print("\nGPT overall summary:")
    print(overall_df.to_string(index=False))

    print("\nGPT vs baseline on all 20 resumes:")
    print(comparison_df.to_string(index=False))

    print("\nGPT per-entity summary:")
    print(per_entity_df.to_string(index=False))

    print("\nFiles created:")
    print(overall_path)
    print(per_resume_path)
    print(per_entity_path)
    print(comparison_path)
    print(details_path)


if __name__ == "__main__":
    main()
