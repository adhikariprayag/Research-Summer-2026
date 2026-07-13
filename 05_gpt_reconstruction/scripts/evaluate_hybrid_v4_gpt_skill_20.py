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

V4_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "annotation_subset_private"
    / "baseline_test_20"
    / "baseline_system_entities_20_v4_skill_relabel.csv"
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
    / "hybrid_v4_gpt_evaluation"
)

OVERALL_OUTPUT = OUTPUT_DIR / "hybrid_v4_gpt_skill_overall_20.csv"
PER_RESUME_OUTPUT = OUTPUT_DIR / "hybrid_v4_gpt_skill_per_resume_20.csv"
DETAIL_OUTPUT = OUTPUT_DIR / "hybrid_v4_gpt_skill_details_20.csv"
COMBINED_OUTPUT = OUTPUT_DIR / "hybrid_v4_gpt_skill_predictions_20.csv"


def normalize(text: str) -> str:
    text = str(text).strip().casefold()
    return re.sub(r"\s+", " ", text)


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
    for path in [GOLD_FILE, V4_FILE, GPT_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gold = pd.read_csv(GOLD_FILE)
    v4 = pd.read_csv(V4_FILE)
    gpt = pd.read_csv(GPT_FILE)

    required_columns = {"resume_id", "entity_text", "entity_type"}

    for name, df in [("Gold", gold), ("V4", v4), ("GPT", gpt)]:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} file is missing required columns: {sorted(missing)}"
            )

    gold = gold[
        gold["entity_type"].astype(str).str.strip().str.upper() == "SKILL"
    ].copy()

    v4 = v4[
        v4["entity_type"].astype(str).str.strip().str.upper() == "SKILL"
    ].copy()

    gpt = gpt[
        gpt["entity_type"].astype(str).str.strip().str.upper() == "SKILL"
    ].copy()

    for df in [gold, v4, gpt]:
        df["resume_id"] = df["resume_id"].astype(str).str.strip()
        df["entity_text"] = df["entity_text"].astype(str).str.strip()
        df["norm"] = df["entity_text"].map(normalize)

    gold = gold[gold["norm"] != ""].copy()
    v4 = v4[v4["norm"] != ""].copy()
    gpt = gpt[gpt["norm"] != ""].copy()

    gold = gold.drop_duplicates(subset=["resume_id", "norm"])
    v4 = v4.drop_duplicates(subset=["resume_id", "norm"])
    gpt = gpt.drop_duplicates(subset=["resume_id", "norm"])

    v4["source"] = "V4"
    gpt["source"] = "GPT"

    combined = pd.concat(
        [
            v4[["resume_id", "entity_text", "entity_type", "norm", "source"]],
            gpt[["resume_id", "entity_text", "entity_type", "norm", "source"]],
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        by=["resume_id", "norm", "source"]
    )

    combined = combined.drop_duplicates(
        subset=["resume_id", "norm"],
        keep="first",
    )

    combined.to_csv(COMBINED_OUTPUT, index=False)

    gold_keys = set(zip(gold["resume_id"], gold["norm"]))
    hybrid_keys = set(zip(combined["resume_id"], combined["norm"]))

    correct_keys = gold_keys & hybrid_keys
    missed_keys = gold_keys - hybrid_keys
    incorrect_keys = hybrid_keys - gold_keys

    overall_metrics = calculate_metrics(
        gold_count=len(gold_keys),
        system_count=len(hybrid_keys),
        correct=len(correct_keys),
    )

    pd.DataFrame([overall_metrics]).to_csv(
        OVERALL_OUTPUT,
        index=False,
    )

    resume_ids = sorted(
        set(gold["resume_id"]) | set(combined["resume_id"])
    )

    per_resume_rows = []

    for resume_id in resume_ids:
        gold_resume = {
            key for key in gold_keys if key[0] == resume_id
        }

        hybrid_resume = {
            key for key in hybrid_keys if key[0] == resume_id
        }

        correct_resume = gold_resume & hybrid_resume

        metrics = calculate_metrics(
            gold_count=len(gold_resume),
            system_count=len(hybrid_resume),
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

    per_resume_df.to_csv(
        PER_RESUME_OUTPUT,
        index=False,
    )

    gold_lookup = {
        (row.resume_id, row.norm): row.entity_text
        for row in gold.itertuples()
    }

    hybrid_lookup = {
        (row.resume_id, row.norm): row.entity_text
        for row in combined.itertuples()
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
                "entity_text": hybrid_lookup[key],
                "entity_type": "SKILL",
                "result": "incorrect",
            }
        )

    pd.DataFrame(detail_rows).to_csv(
        DETAIL_OUTPUT,
        index=False,
    )

    print("Hybrid V4 + GPT SKILL evaluation complete.")
    print()
    print("Overall results:")

    for key, value in overall_metrics.items():
        print(f"{key}: {value}")

    print()
    print(f"Saved overall summary: {OVERALL_OUTPUT}")
    print(f"Saved per-resume summary: {PER_RESUME_OUTPUT}")
    print(f"Saved private hybrid predictions: {COMBINED_OUTPUT}")
    print(f"Saved private detail file: {DETAIL_OUTPUT}")


if __name__ == "__main__":
    main()
