from pathlib import Path
import json
import os
import re
import time

import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------
# Project paths and settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "generalized_v4_validation_sample_50.csv"
)

JSON_ROOT = PROJECT_ROOT / "02_dataset" / "large_scale_dataset" / "json_records_private"

PROMPT_FILE = (
    PROJECT_ROOT / "05_gpt_reconstruction" / "prompts" / "gpt_skill_recovery_prompt.md"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "validation_50_outputs"
)

COMBINED_OUTPUT = OUTPUT_FOLDER / "gpt_skill_recovery_validation_50_combined.csv"

STATUS_OUTPUT = OUTPUT_FOLDER / "gpt_skill_recovery_validation_50_status.csv"

MODEL_NAME = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 8000
MAX_RETRIES = 3
REQUEST_DELAY_SECONDS = 1

ALLOWED_ENTITY_TYPES = {"SKILL"}


# ---------------------------------------------------------
# Text and JSON helper functions
# ---------------------------------------------------------


def clean_json_text(text: str) -> str:
    """Remove accidental Markdown code fences from GPT output."""

    text = str(text).strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    elif text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def normalize_entity_text(text: str) -> str:
    """Remove unnecessary whitespace from an extracted skill."""

    text = str(text).strip()
    return re.sub(r"\s+", " ", text)


def is_valid_json_output(path: Path) -> bool:
    """Check whether an existing GPT output contains a valid skills list."""

    if not path.exists():
        return False

    try:
        raw_text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        data = json.loads(clean_json_text(raw_text))

        return isinstance(data, dict) and isinstance(data.get("skills"), list)

    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
    ):
        return False


def load_resume_text(json_path: Path) -> str:
    """Load the resume_text field from a private JSON record."""

    try:
        data = json.loads(
            json_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )

    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid resume JSON file: {json_path}") from exc

    resume_text = data.get("resume_text", "")

    if not isinstance(resume_text, str):
        resume_text = str(resume_text)

    resume_text = resume_text.strip()

    if not resume_text:
        raise ValueError(f"No resume_text was found in: {json_path}")

    return resume_text


def load_prompt(
    resume_id: str,
    resume_text: str,
) -> str:
    """Insert the resume ID and text into the saved prompt template."""

    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    prompt = prompt_template.replace(
        "RESUME_ID_HERE",
        resume_id,
    )

    prompt = prompt.replace(
        "RESUME_TEXT_HERE",
        resume_text,
    )

    return prompt


# ---------------------------------------------------------
# OpenAI API function
# ---------------------------------------------------------


def call_gpt(
    client: OpenAI,
    prompt: str,
) -> str:
    """Send the skill-recovery prompt to the OpenAI API."""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    return response.output_text


# ---------------------------------------------------------
# GPT output parser
# ---------------------------------------------------------


def parse_skills(
    resume_id: str,
    raw_output: str,
    metadata: dict,
) -> list[dict]:
    """Validate and deduplicate GPT skill predictions."""

    cleaned_output = clean_json_text(raw_output)
    data = json.loads(cleaned_output)

    skills = data.get("skills", [])

    if not isinstance(skills, list):
        raise ValueError("GPT output field 'skills' must be a list.")

    parsed_skills = []
    seen = set()

    for item in skills:
        if not isinstance(item, dict):
            continue

        entity_text = normalize_entity_text(item.get("entity_text", ""))

        entity_type = str(item.get("entity_type", "SKILL")).strip().upper()

        if not entity_text:
            continue

        if entity_type not in ALLOWED_ENTITY_TYPES:
            continue

        deduplication_key = (
            resume_id,
            entity_text.casefold(),
            entity_type,
        )

        if deduplication_key in seen:
            continue

        seen.add(deduplication_key)

        parsed_skills.append(
            {
                "validation_order": metadata.get("validation_order"),
                "resume_id": resume_id,
                "dataset_group": metadata.get(
                    "dataset_group",
                    "",
                ),
                "prediction_band": metadata.get(
                    "prediction_band",
                    "",
                ),
                "selection_stage": metadata.get(
                    "selection_stage",
                    "",
                ),
                "entity_text": entity_text,
                "entity_type": "SKILL",
                "prediction_source": "GPT",
            }
        )

    return parsed_skills


# ---------------------------------------------------------
# Locate resume JSON files
# ---------------------------------------------------------


def build_json_file_map() -> dict[str, Path]:
    """Create a resume ID to JSON-file lookup dictionary."""

    json_file_map = {}
    duplicate_ids = []

    for json_path in JSON_ROOT.rglob("*.json"):
        resume_id = json_path.stem

        if resume_id in json_file_map:
            duplicate_ids.append(resume_id)
            continue

        json_file_map[resume_id] = json_path

    if duplicate_ids:
        raise ValueError(
            "Duplicate resume JSON IDs were found:\n"
            + "\n".join(sorted(set(duplicate_ids)))
        )

    return json_file_map


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------


def main() -> None:
    """Run GPT skill recovery on the 50-resume validation sample."""

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. " "Set it before running this script."
        )

    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Validation sample was not found:\n" f"{SAMPLE_FILE}")

    if not JSON_ROOT.exists():
        raise FileNotFoundError(
            f"Private JSON directory was not found:\n" f"{JSON_ROOT}"
        )

    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"Skill-recovery prompt was not found:\n" f"{PROMPT_FILE}"
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_df = pd.read_csv(SAMPLE_FILE)

    required_columns = {
        "resume_id",
        "dataset_group",
        "prediction_band",
        "selection_stage",
    }

    missing_columns = required_columns - set(sample_df.columns)

    if missing_columns:
        raise ValueError(
            "The 50-resume sample is missing columns: " f"{sorted(missing_columns)}"
        )

    if len(sample_df) != 50:
        raise ValueError(f"Expected 50 resumes, but found " f"{len(sample_df)}.")

    sample_df["resume_id"] = sample_df["resume_id"].astype(str).str.strip()

    if sample_df["resume_id"].duplicated().any():
        duplicate_ids = (
            sample_df.loc[
                sample_df["resume_id"].duplicated(keep=False),
                "resume_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Duplicate IDs were found in the sample:\n"
            + "\n".join(sorted(duplicate_ids))
        )

    if "validation_order" in sample_df.columns:
        sample_df = sample_df.sort_values("validation_order").reset_index(drop=True)

    json_file_map = build_json_file_map()

    missing_json_ids = sorted(set(sample_df["resume_id"]) - set(json_file_map))

    if missing_json_ids:
        raise FileNotFoundError(
            "The following selected resumes have no JSON file:\n"
            + "\n".join(missing_json_ids)
        )

    print(f"Loaded validation sample: " f"{len(sample_df)} resumes")

    print(f"Located private JSON records: " f"{len(json_file_map)}")

    client = OpenAI()

    all_skills = []
    status_rows = []
    failed_resumes = []

    for position, row in sample_df.iterrows():
        resume_id = row["resume_id"]

        metadata = {
            "validation_order": row.get(
                "validation_order",
                position + 1,
            ),
            "dataset_group": row.get(
                "dataset_group",
                "",
            ),
            "prediction_band": row.get(
                "prediction_band",
                "",
            ),
            "selection_stage": row.get(
                "selection_stage",
                "",
            ),
        }

        resume_json_path = json_file_map[resume_id]

        json_output_path = OUTPUT_FOLDER / f"{resume_id}_gpt_skills.json"

        print(f"\n[{position + 1}/50] {resume_id}")

        # Reuse a valid existing output.
        if is_valid_json_output(json_output_path):
            try:
                raw_output = json_output_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                parsed_skills = parse_skills(
                    resume_id,
                    raw_output,
                    metadata,
                )

                all_skills.extend(parsed_skills)

                status_rows.append(
                    {
                        "validation_order": metadata["validation_order"],
                        "resume_id": resume_id,
                        "dataset_group": metadata["dataset_group"],
                        "prediction_band": metadata["prediction_band"],
                        "selection_stage": metadata["selection_stage"],
                        "status": "skipped_existing",
                        "skills_extracted": len(parsed_skills),
                        "error_type": "",
                        "error_message": "",
                    }
                )

                print("Skipped: valid GPT output " "already exists.")

                print(f"Skills loaded: " f"{len(parsed_skills)}")

                continue

            except Exception as exc:
                print(
                    "Existing output could not be reused. "
                    "The API request will be rerun."
                )

                print(f"Existing output error: {exc}")

        try:
            resume_text = load_resume_text(resume_json_path)

        except Exception as exc:
            print(f"FAILED: {resume_id}")
            print(f"Error: {exc}")

            failed_resumes.append(resume_id)

            status_rows.append(
                {
                    "validation_order": metadata["validation_order"],
                    "resume_id": resume_id,
                    "dataset_group": metadata["dataset_group"],
                    "prediction_band": metadata["prediction_band"],
                    "selection_stage": metadata["selection_stage"],
                    "status": "failed",
                    "skills_extracted": 0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

            continue

        print(f"Resume text length: " f"{len(resume_text)} characters")

        prompt = load_prompt(
            resume_id,
            resume_text,
        )

        successful = False
        final_error = None

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):
            try:
                print(f"API attempt " f"{attempt}/{MAX_RETRIES}...")

                raw_output = call_gpt(
                    client,
                    prompt,
                )

                parsed_skills = parse_skills(
                    resume_id,
                    raw_output,
                    metadata,
                )

                json_output_path.write_text(
                    raw_output,
                    encoding="utf-8",
                )

                all_skills.extend(parsed_skills)

                status_rows.append(
                    {
                        "validation_order": metadata["validation_order"],
                        "resume_id": resume_id,
                        "dataset_group": metadata["dataset_group"],
                        "prediction_band": metadata["prediction_band"],
                        "selection_stage": metadata["selection_stage"],
                        "status": "processed",
                        "skills_extracted": len(parsed_skills),
                        "error_type": "",
                        "error_message": "",
                    }
                )

                print(f"Skills extracted: " f"{len(parsed_skills)}")

                print(f"Saved: {json_output_path}")

                successful = True
                break

            except Exception as exc:
                final_error = exc

                print(f"Attempt {attempt} failed.")

                print(f"Error type: " f"{type(exc).__name__}")

                print(f"Error message: {exc}")

                if attempt < MAX_RETRIES:
                    wait_time = attempt * 3

                    print(f"Retrying after " f"{wait_time} seconds...")

                    time.sleep(wait_time)

        if not successful:
            failed_resumes.append(resume_id)

            status_rows.append(
                {
                    "validation_order": metadata["validation_order"],
                    "resume_id": resume_id,
                    "dataset_group": metadata["dataset_group"],
                    "prediction_band": metadata["prediction_band"],
                    "selection_stage": metadata["selection_stage"],
                    "status": "failed",
                    "skills_extracted": 0,
                    "error_type": (
                        type(final_error).__name__ if final_error else "UnknownError"
                    ),
                    "error_message": (
                        str(final_error) if final_error else "Unknown API error"
                    ),
                }
            )

        time.sleep(REQUEST_DELAY_SECONDS)

    # -----------------------------------------------------
    # Save combined GPT predictions
    # -----------------------------------------------------

    combined_columns = [
        "validation_order",
        "resume_id",
        "dataset_group",
        "prediction_band",
        "selection_stage",
        "entity_text",
        "entity_type",
        "prediction_source",
    ]

    combined_df = pd.DataFrame(
        all_skills,
        columns=combined_columns,
    )

    if not combined_df.empty:
        combined_df["_normalized_entity"] = (
            combined_df["entity_text"].astype(str).str.strip().str.casefold()
        )

        combined_df = combined_df.drop_duplicates(
            subset=[
                "resume_id",
                "_normalized_entity",
                "entity_type",
            ]
        )

        combined_df = combined_df.sort_values(
            by=[
                "validation_order",
                "_normalized_entity",
            ]
        )

        combined_df = combined_df.drop(columns=["_normalized_entity"])

    combined_df.to_csv(
        COMBINED_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Save processing status
    # -----------------------------------------------------

    status_df = pd.DataFrame(status_rows)

    if not status_df.empty:
        status_df = status_df.sort_values("validation_order")

    status_df.to_csv(
        STATUS_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Print final summary
    # -----------------------------------------------------

    completed_count = len(
        status_df[
            status_df["status"].isin(
                [
                    "processed",
                    "skipped_existing",
                ]
            )
        ]
    )

    print("\nGPT skill recovery run complete.")

    print(f"Resumes completed: " f"{completed_count}/50")

    print(f"Combined valid GPT skills: " f"{len(combined_df)}")

    print(f"Combined CSV:\n" f"{COMBINED_OUTPUT}")

    print(f"Status CSV:\n" f"{STATUS_OUTPUT}")

    if failed_resumes:
        print(f"\nFailed resumes: " f"{len(failed_resumes)}")

        for resume_id in failed_resumes:
            print(f"- {resume_id}")

    else:
        print("\nAll 50 resumes completed successfully.")


if __name__ == "__main__":
    main()
