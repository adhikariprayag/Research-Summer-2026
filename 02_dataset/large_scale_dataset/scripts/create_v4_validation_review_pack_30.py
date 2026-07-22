from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


ROOT = Path(
    "/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research"
)

SAMPLE_PATH = (
    ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "generalized_v4_validation_sample_30.csv"
)

ENTITY_PATH = (
    ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "generalized_v4_system_entities_1020.csv"
)

JSON_ROOT = (
    ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "json_records_private"
)

OUTPUT_DIR = (
    ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "validation_review_30"
)

TEXT_OUTPUT_DIR = OUTPUT_DIR / "resume_texts"

REVIEW_CSV = (
    OUTPUT_DIR
    / "v4_prediction_review_template_30.csv"
)

RESUME_SUMMARY_CSV = (
    OUTPUT_DIR
    / "validation_resume_summary_30.csv"
)


def load_json_map() -> dict[str, Path]:
    json_map = {}

    for path in JSON_ROOT.rglob("*.json"):
        json_map[path.stem] = path

    return json_map


def main() -> None:
    sample = pd.read_csv(SAMPLE_PATH)
    entities = pd.read_csv(ENTITY_PATH)

    sample["resume_id"] = (
        sample["resume_id"]
        .astype(str)
        .str.strip()
    )

    entities["resume_id"] = (
        entities["resume_id"]
        .astype(str)
        .str.strip()
    )

    sample_ids = set(sample["resume_id"])

    selected_entities = entities[
        entities["resume_id"].isin(sample_ids)
    ].copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEXT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_map = load_json_map()

    missing_json = []
    text_rows = []

    for _, row in sample.iterrows():
        resume_id = row["resume_id"]

        json_path = json_map.get(resume_id)

        if json_path is None:
            missing_json.append(resume_id)
            continue

        data = json.loads(
            json_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )

        resume_text = str(
            data.get("resume_text", "")
        ).strip()

        text_path = (
            TEXT_OUTPUT_DIR
            / f"{resume_id}.txt"
        )

        header = (
            f"RESUME ID: {resume_id}\n"
            f"SOURCE: {row['source_dataset']}\n"
            f"CATEGORY: {row['original_category']}\n"
            f"PREDICTION BAND: {row['prediction_band']}\n"
            f"TOTAL V4 PREDICTIONS: "
            f"{row['total_predictions']}\n"
            + "=" * 80
            + "\n\n"
        )

        text_path.write_text(
            header + resume_text,
            encoding="utf-8",
        )

        text_rows.append(
            {
                "resume_id": resume_id,
                "source_dataset": row[
                    "source_dataset"
                ],
                "original_category": row[
                    "original_category"
                ],
                "prediction_band": row[
                    "prediction_band"
                ],
                "total_predictions": row[
                    "total_predictions"
                ],
                "text_file": str(text_path),
            }
        )

    review = selected_entities.merge(
        sample[
            [
                "resume_id",
                "source_dataset",
                "original_category",
                "prediction_band",
                "total_predictions",
            ]
        ],
        on="resume_id",
        how="left",
        validate="many_to_one",
    )

    review = review[
        [
            "resume_id",
            "source_dataset",
            "original_category",
            "prediction_band",
            "entity_type",
            "entity_text",
            "source",
            "total_predictions",
        ]
    ].copy()

    review["review_decision"] = ""
    review["corrected_entity_text"] = ""
    review["corrected_entity_type"] = ""
    review["review_notes"] = ""

    review = review.sort_values(
        [
            "source_dataset",
            "prediction_band",
            "resume_id",
            "entity_type",
            "entity_text",
        ]
    )

    review.to_csv(
        REVIEW_CSV,
        index=False,
    )

    pd.DataFrame(text_rows).to_csv(
        RESUME_SUMMARY_CSV,
        index=False,
    )

    print("Validation review package created.")
    print(f"Sample resumes: {len(sample)}")
    print(
        "Predictions to review: "
        f"{len(review)}"
    )
    print(
        "Resume text files created: "
        f"{len(text_rows)}"
    )

    if missing_json:
        print(
            "Missing JSON files: "
            f"{len(missing_json)}"
        )
        print(missing_json)

    print("\nReview CSV:")
    print(REVIEW_CSV)

    print("\nResume text folder:")
    print(TEXT_OUTPUT_DIR)

    print("\nResume summary:")
    print(RESUME_SUMMARY_CSV)

    print("\nReview-decision values to use:")
    print("correct")
    print("incorrect")
    print("partial")
    print("unsure")


if __name__ == "__main__":
    main()
