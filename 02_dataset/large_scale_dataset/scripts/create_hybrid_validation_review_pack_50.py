from pathlib import Path
import json

import pandas as pd

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "generalized_v4_validation_sample_50.csv"
)

HYBRID_FILE = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "validation_50_hybrid"
    / "hybrid_v4_gpt_skill_predictions_50.csv"
)

JSON_ROOT = PROJECT_ROOT / "02_dataset" / "large_scale_dataset" / "json_records_private"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "validation_review_50"
)

RESUME_TEXT_DIR = OUTPUT_DIR / "resume_texts"

PREDICTION_REVIEW_OUTPUT = OUTPUT_DIR / "hybrid_prediction_review_template_50.csv"

MISSED_SKILL_OUTPUT = OUTPUT_DIR / "missed_skill_review_template_50.csv"

RESUME_SUMMARY_OUTPUT = OUTPUT_DIR / "validation_resume_summary_50.csv"

INSTRUCTIONS_OUTPUT = OUTPUT_DIR / "manual_review_instructions_50.txt"

MISSED_SKILL_SLOTS_PER_RESUME = 30


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------


def build_json_file_map() -> dict[str, Path]:
    """
    Create a lookup from resume ID to the corresponding
    private JSON file.
    """

    json_file_map = {}
    duplicate_ids = []

    for json_path in JSON_ROOT.rglob("*.json"):
        resume_id = json_path.stem.strip()

        if resume_id in json_file_map:
            duplicate_ids.append(resume_id)
            continue

        json_file_map[resume_id] = json_path

    if duplicate_ids:
        raise ValueError(
            "Duplicate JSON resume IDs were found:\n"
            + "\n".join(sorted(set(duplicate_ids)))
        )

    return json_file_map


def load_json_record(json_path: Path) -> dict:
    """Load one private resume JSON record."""

    try:
        raw_text = json_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        data = json.loads(raw_text)

    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {json_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"JSON record must be an object: {json_path}")

    return data


def clean_text(value: object) -> str:
    """Convert a value into clean text."""

    if value is None:
        return ""

    return str(value).strip()


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------


def main() -> None:
    """
    Create the manual-review package for the
    50-resume Hybrid V4 + GPT validation set.
    """

    required_paths = [
        SAMPLE_FILE,
        HYBRID_FILE,
        JSON_ROOT,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path was not found:\n{path}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESUME_TEXT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample = pd.read_csv(SAMPLE_FILE)
    hybrid = pd.read_csv(HYBRID_FILE)

    print(f"Loaded validation sample: " f"{len(sample)} resumes")

    print(f"Loaded hybrid predictions: " f"{len(hybrid)}")

    # -----------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------

    sample_required_columns = {
        "resume_id",
        "dataset_group",
        "prediction_band",
        "selection_stage",
    }

    hybrid_required_columns = {
        "resume_id",
        "entity_text",
        "entity_type",
        "prediction_source",
        "v4_entity_text",
        "gpt_entity_text",
    }

    missing_sample_columns = sample_required_columns - set(sample.columns)

    missing_hybrid_columns = hybrid_required_columns - set(hybrid.columns)

    if missing_sample_columns:
        raise ValueError(
            "Sample file is missing required columns: "
            f"{sorted(missing_sample_columns)}"
        )

    if missing_hybrid_columns:
        raise ValueError(
            "Hybrid file is missing required columns: "
            f"{sorted(missing_hybrid_columns)}"
        )

    if len(sample) != 50:
        raise ValueError(f"Expected 50 resumes, but found " f"{len(sample)}.")

    # -----------------------------------------------------
    # Clean resume IDs
    # -----------------------------------------------------

    sample["resume_id"] = sample["resume_id"].astype(str).str.strip()

    hybrid["resume_id"] = hybrid["resume_id"].astype(str).str.strip()

    if sample["resume_id"].duplicated().any():
        duplicate_ids = (
            sample.loc[
                sample["resume_id"].duplicated(keep=False),
                "resume_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Duplicate resume IDs were found "
            "in the validation sample:\n" + "\n".join(sorted(duplicate_ids))
        )

    selected_resume_ids = set(sample["resume_id"])

    hybrid_resume_ids = set(hybrid["resume_id"])

    unexpected_hybrid_ids = hybrid_resume_ids - selected_resume_ids

    if unexpected_hybrid_ids:
        raise ValueError(
            "Hybrid predictions contain resume IDs "
            "outside the selected 50:\n" + "\n".join(sorted(unexpected_hybrid_ids))
        )

    # -----------------------------------------------------
    # Locate private resume JSON files
    # -----------------------------------------------------

    json_file_map = build_json_file_map()

    missing_json_ids = sorted(selected_resume_ids - set(json_file_map))

    if missing_json_ids:
        raise FileNotFoundError(
            "Missing JSON files for these resumes:\n" + "\n".join(missing_json_ids)
        )

    # -----------------------------------------------------
    # Export resume text and collect metadata
    # -----------------------------------------------------

    metadata_rows = []

    for position, sample_row in sample.iterrows():
        resume_id = sample_row["resume_id"]
        json_path = json_file_map[resume_id]

        record = load_json_record(json_path)

        resume_text = clean_text(
            record.get(
                "resume_text",
                "",
            )
        )

        if not resume_text:
            raise ValueError(f"No resume_text was found for " f"{resume_id}")

        source_dataset = clean_text(
            record.get(
                "source_dataset",
                sample_row.get(
                    "dataset_group",
                    "",
                ),
            )
        )

        original_category = clean_text(
            record.get(
                "original_category",
                "",
            )
        )

        validation_order = sample_row.get(
            "validation_order",
            position + 1,
        )

        text_output_path = RESUME_TEXT_DIR / f"{resume_id}.txt"

        text_output_path.write_text(
            resume_text,
            encoding="utf-8",
        )

        metadata_rows.append(
            {
                "validation_order": validation_order,
                "resume_id": resume_id,
                "dataset_group": sample_row["dataset_group"],
                "source_dataset": source_dataset,
                "original_category": original_category,
                "prediction_band": sample_row["prediction_band"],
                "selection_stage": sample_row["selection_stage"],
                "text_character_count": len(resume_text),
                "text_file": str(text_output_path),
            }
        )

    metadata = pd.DataFrame(metadata_rows)

    # -----------------------------------------------------
    # Count hybrid predictions per resume
    # -----------------------------------------------------

    hybrid_counts = (
        hybrid.groupby("resume_id")
        .size()
        .rename("hybrid_total_predictions")
        .reset_index()
    )

    source_counts = (
        hybrid.groupby(
            [
                "resume_id",
                "prediction_source",
            ]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for source_name in [
        "V4",
        "GPT",
        "both",
    ]:
        if source_name not in source_counts.columns:
            source_counts[source_name] = 0

    source_counts = source_counts.rename(
        columns={
            "V4": "v4_only_predictions",
            "GPT": "gpt_only_predictions",
            "both": "both_predictions",
        }
    )

    metadata = metadata.merge(
        hybrid_counts,
        on="resume_id",
        how="left",
        validate="one_to_one",
    )

    metadata = metadata.merge(
        source_counts[
            [
                "resume_id",
                "v4_only_predictions",
                "gpt_only_predictions",
                "both_predictions",
            ]
        ],
        on="resume_id",
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        "hybrid_total_predictions",
        "v4_only_predictions",
        "gpt_only_predictions",
        "both_predictions",
    ]

    metadata[count_columns] = metadata[count_columns].fillna(0).astype(int)

    metadata = metadata.sort_values("validation_order").reset_index(drop=True)

    # -----------------------------------------------------
    # Create prediction-review CSV
    # -----------------------------------------------------

    # The hybrid prediction file already contains some
    # metadata columns. Remove them before merging so pandas
    # does not create _x and _y column names.
    hybrid_for_review = hybrid.drop(
        columns=[
            "validation_order",
            "dataset_group",
            "prediction_band",
            "selection_stage",
        ],
        errors="ignore",
    ).copy()

    review_metadata_columns = [
        "validation_order",
        "resume_id",
        "dataset_group",
        "source_dataset",
        "original_category",
        "prediction_band",
        "selection_stage",
        "hybrid_total_predictions",
        "text_file",
    ]

    review = hybrid_for_review.merge(
        metadata[review_metadata_columns],
        on="resume_id",
        how="left",
        validate="many_to_one",
    )

    if review["validation_order"].isna().any():
        missing_metadata_ids = (
            review.loc[
                review["validation_order"].isna(),
                "resume_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Metadata was missing for these "
            "hybrid predictions:\n" + "\n".join(sorted(missing_metadata_ids))
        )

    review["review_decision"] = ""
    review["corrected_entity_text"] = ""
    review["review_notes"] = ""

    review_columns = [
        "validation_order",
        "resume_id",
        "dataset_group",
        "source_dataset",
        "original_category",
        "prediction_band",
        "selection_stage",
        "entity_type",
        "entity_text",
        "prediction_source",
        "v4_entity_text",
        "gpt_entity_text",
        "hybrid_total_predictions",
        "review_decision",
        "corrected_entity_text",
        "review_notes",
        "text_file",
    ]

    review = review[review_columns]

    review["_entity_sort"] = review["entity_text"].astype(str).str.casefold()

    review = review.sort_values(
        by=[
            "validation_order",
            "_entity_sort",
        ]
    ).reset_index(drop=True)

    review = review.drop(
        columns=[
            "_entity_sort",
        ]
    )

    review.to_csv(
        PREDICTION_REVIEW_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Create missed-skill review CSV
    # -----------------------------------------------------

    missed_rows = []

    for _, metadata_row in metadata.iterrows():
        for slot_number in range(
            1,
            MISSED_SKILL_SLOTS_PER_RESUME + 1,
        ):
            missed_rows.append(
                {
                    "validation_order": metadata_row["validation_order"],
                    "resume_id": metadata_row["resume_id"],
                    "dataset_group": metadata_row["dataset_group"],
                    "source_dataset": metadata_row["source_dataset"],
                    "original_category": metadata_row["original_category"],
                    "prediction_band": metadata_row["prediction_band"],
                    "missed_slot": slot_number,
                    "missed_entity_text": "",
                    "entity_type": "SKILL",
                    "review_notes": "",
                    "text_file": metadata_row["text_file"],
                }
            )

    missed_review = pd.DataFrame(missed_rows)

    missed_review.to_csv(
        MISSED_SKILL_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Save resume-level summary
    # -----------------------------------------------------

    metadata.to_csv(
        RESUME_SUMMARY_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Create manual-review instructions
    # -----------------------------------------------------

    instructions = """
HYBRID V4 + GPT — 50-RESUME MANUAL REVIEW

FILES

1. hybrid_prediction_review_template_50.csv

Review every Hybrid V4 + GPT skill prediction.

2. missed_skill_review_template_50.csv

Record valid technical skills that appear in the resume
but were missed by both V4 and GPT.

3. validation_resume_summary_50.csv

Contains resume-level metadata and prediction counts.

4. resume_texts/

Contains the resume text used by V4 and GPT.


PREDICTION REVIEW DECISIONS

Enter one of these values in the review_decision column:

correct

The full predicted phrase is a valid technical skill and
appears in the resume.

incorrect

The prediction is not a technical skill, does not appear
in the resume, or is too generic to count.

partial

The prediction refers to a real technical skill, but the
phrase is incomplete or includes unnecessary words.

unsure

The reviewer cannot confidently decide.


CORRECTED ENTITY TEXT

For a partial prediction, enter the complete correct phrase
in corrected_entity_text.

Example:

Predicted entity:
REST

Corrected entity:
RESTful APIs


MISSED SKILL REVIEW

Read each resume and record technical skills that do not
appear in the Hybrid prediction review file.

Use one row for each missed skill.

Leave unused rows blank.

If a resume needs more than 30 missed-skill rows, copy
additional rows for the same resume ID.


COUNT AS TECHNICAL SKILLS

Examples include:

- programming languages
- frameworks
- libraries
- databases
- cloud platforms
- cloud services
- developer tools
- operating systems
- networking technologies
- cybersecurity tools
- cybersecurity concepts
- machine-learning methods
- data-engineering technologies
- specific multi-word technical phrases


DO NOT COUNT GENERIC WORDS

Do not count broad terms such as:

- software
- technology
- computer
- development
- programming

unless they are part of a specific technical phrase.


METRICS

The prediction review is used to calculate precision.

The missed-skill review is required to calculate recall.

Precision:

Correct predictions divided by all system predictions.

Recall:

Correct predictions divided by all actual skills.

F1-score:

The balance between precision and recall.
""".strip()

    INSTRUCTIONS_OUTPUT.write_text(
        instructions,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Final validation checks
    # -----------------------------------------------------

    exported_text_files = list(RESUME_TEXT_DIR.glob("*.txt"))

    if len(exported_text_files) != 50:
        raise ValueError(
            f"Expected 50 resume text files, "
            f"but found "
            f"{len(exported_text_files)}."
        )

    if len(review) != len(hybrid):
        raise ValueError(
            "Prediction-review row count does not "
            "match the hybrid prediction count.\n"
            f"Hybrid predictions: {len(hybrid)}\n"
            f"Review rows: {len(review)}"
        )

    expected_missed_rows = 50 * MISSED_SKILL_SLOTS_PER_RESUME

    if len(missed_review) != expected_missed_rows:
        raise ValueError(
            f"Expected {expected_missed_rows} "
            f"missed-skill rows, but found "
            f"{len(missed_review)}."
        )

    if len(metadata) != 50:
        raise ValueError(
            f"Expected 50 resume-summary rows, " f"but found {len(metadata)}."
        )

    if review["resume_id"].nunique() != 50:
        raise ValueError(
            "The prediction-review file does not " "contain all 50 resumes."
        )

    # -----------------------------------------------------
    # Print final summary
    # -----------------------------------------------------

    print("\n50-resume manual-review package created.")

    print(f"Resume text files: " f"{len(exported_text_files)}")

    print(f"Hybrid prediction review rows: " f"{len(review)}")

    print(f"Missed-skill review rows: " f"{len(missed_review)}")

    print(f"Resume summary rows: " f"{len(metadata)}")

    print(f"\nOutput directory:\n" f"{OUTPUT_DIR}")

    print(f"\nPrediction review file:\n" f"{PREDICTION_REVIEW_OUTPUT}")

    print(f"\nMissed-skill file:\n" f"{MISSED_SKILL_OUTPUT}")

    print(f"\nResume summary file:\n" f"{RESUME_SUMMARY_OUTPUT}")

    print(f"\nInstructions file:\n" f"{INSTRUCTIONS_OUTPUT}")


if __name__ == "__main__":
    main()
