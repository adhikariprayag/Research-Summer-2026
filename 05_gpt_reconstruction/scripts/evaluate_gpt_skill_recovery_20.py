from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "annotation_subset_private"
    / "gold_standard_100_template.csv"
)

GPT_FILE = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "current_20_outputs"
    / "gpt_skill_recovery_current_20_combined.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "current_20_evaluation"
)

OVERALL_OUTPUT = OUTPUT_DIR / "gpt_skill_recovery_overall_20.csv"
PER_RESUME_OUTPUT = OUTPUT_DIR / "gpt_skill_recovery_per_resume_20.csv"
DETAIL_OUTPUT = OUTPUT_DIR / "gpt_skill_recovery_details_20.csv"


def normalize(text: str) -> str:
    """Normalize entity text for strict case-insensitive exact matching."""
    text = str(text).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_metrics(gold_count: int, system_count: int, correct: int) -> dict:
    missed = gold_count - correct
    incorrect = system_count - correct

    precision = safe_divide(correct, system_count)
    recall = safe_divide(correct, gold_count)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "gold": gold_count,
        "system": system_count,
        "correct": correct,
        "missed": missed,
        "incorrect": incorrect,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def main() -> None:
    if not GOLD_FILE.exists():
        raise FileNotFoundError(f"Gold file not found: {GOLD_FILE}")

    if not GPT_FILE.exists():
        raise FileNotFoundError(f"GPT file not found: {GPT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gold = pd.read_csv(GOLD_FILE)
    gpt = pd.read_csv(GPT_FILE)

    required_columns = {"resume_id", "entity_text", "entity_type"}

    for name, df in [("Gold", gold), ("GPT", gpt)]:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} file is missing required columns: {sorted(missing)}"
            )

    gold = gold[
        gold["entity_type"].astype(str).str.strip().str.upper() == "SKILL"
    ].copy()

    gpt = gpt[
        gpt["entity_type"].astype(str).str.strip().str.upper() == "SKILL"
    ].copy()

    gold["resume_id"] = gold["resume_id"].astype(str).str.strip()
    gpt["resume_id"] = gpt["resume_id"].astype(str).str.strip()

    gold["norm"] = gold["entity_text"].map(normalize)
    gpt["norm"] = gpt["entity_text"].map(normalize)

    gold = gold[gold["norm"] != ""].copy()
    gpt = gpt[gpt["norm"] != ""].copy()

    gold = gold.drop_duplicates(
        subset=["resume_id", "norm", "entity_type"]
    )

    gpt = gpt.drop_duplicates(
        subset=["resume_id", "norm", "entity_type"]
    )

    gold_keys = set(
        zip(gold["resume_id"], gold["norm"])
    )

    gpt_keys = set(
        zip(gpt["resume_id"], gpt["norm"])
    )

    correct_keys = gold_keys & gpt_keys
    missed_keys = gold_keys - gpt_keys
    incorrect_keys = gpt_keys - gold_keys

    overall_metrics = calculate_metrics(
        gold_count=len(gold_keys),
        system_count=len(gpt_keys),
        correct=len(correct_keys),
    )

    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(OVERALL_OUTPUT, index=False)

    resume_ids = sorted(
        set(gold["resume_id"]) | set(gpt["resume_id"])
    )

    per_resume_rows = []

    for resume_id in resume_ids:
        gold_resume = {
            key for key in gold_keys if key[0] == resume_id
        }

        gpt_resume = {
            key for key in gpt_keys if key[0] == resume_id
        }

        correct_resume = gold_resume & gpt_resume

        metrics = calculate_metrics(
            gold_count=len(gold_resume),
            system_count=len(gpt_resume),
            correct=len(correct_resume),
        )

        metrics["resume_id"] = resume_id
        per_resume_rows.append(metrics)

    per_resume_df = pd.DataFrame(per_resume_rows)
    per_resume_df = per_resume_df[
        [
            "resume_id",
            "gold",
            "system",
            "correct",
            "missed",
            "incorrect",
            "precision",
            "recall",
            "f1",
        ]
    ]

    per_resume_df.to_csv(PER_RESUME_OUTPUT, index=False)

    gold_lookup = {
        (row.resume_id, row.norm): row.entity_text
        for row in gold.itertuples()
    }

    gpt_lookup = {
        (row.resume_id, row.norm): row.entity_text
        for row in gpt.itertuples()
    }

    detail_rows = []

    for key in sorted(correct_keys):
        detail_rows.append(
            {
                "resume_id": key[0],
                "entity_text": gold_lookup[key],
                "entity_type": "SKILL",
                "result": "correct",
            }
        )

    for key in sorted(missed_keys):
        detail_rows.append(
            {
                "resume_id": key[0],
                "entity_text": gold_lookup[key],
                "entity_type": "SKILL",
                "result": "missed",
            }
        )

    for key in sorted(incorrect_keys):
        detail_rows.append(
            {
                "resume_id": key[0],
                "entity_text": gpt_lookup[key],
                "entity_type": "SKILL",
                "result": "incorrect",
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(DETAIL_OUTPUT, index=False)

    print("GPT skill-recovery evaluation complete.")
    print()
    print("Overall results:")
    for key, value in overall_metrics.items():
        print(f"{key}: {value}")

    print()
    print(f"Saved overall summary: {OVERALL_OUTPUT}")
    print(f"Saved per-resume summary: {PER_RESUME_OUTPUT}")
    print(f"Saved private detail file: {DETAIL_OUTPUT}")


if __name__ == "__main__":
    main()
