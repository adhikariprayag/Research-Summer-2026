from pathlib import Path
import json
import os
import re
import time

import pandas as pd
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FULL_LIST = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "gpt_skill_recovery_current_20_input.csv"
)

PROMPT_FILE = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "prompts"
    / "gpt_skill_recovery_prompt.md"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "current_20_outputs"
)

COMBINED_OUTPUT = (
    OUTPUT_FOLDER
    / "gpt_skill_recovery_current_20_combined.csv"
)

MODEL_NAME = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 8000

ALLOWED_ENTITY_TYPES = {"SKILL"}


def clean_json_text(text: str) -> str:
    """Remove accidental Markdown code fences from GPT output."""
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()
    elif text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def normalize_entity_text(text: str) -> str:
    """Remove extra spaces from an extracted skill."""
    text = str(text).strip()
    return re.sub(r"\s+", " ", text)


def is_valid_json_output(path: Path) -> bool:
    """Return True when a saved output contains a valid skills list."""
    if not path.exists():
        return False

    try:
        raw_text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        data = json.loads(clean_json_text(raw_text))

        return (
            isinstance(data, dict)
            and isinstance(data.get("skills"), list)
        )

    except (json.JSONDecodeError, OSError, TypeError):
        return False


def load_prompt(resume_id: str, resume_text: str) -> str:
    """Insert the resume ID and resume text into the prompt template."""
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


def call_gpt(client: OpenAI, prompt: str) -> str:
    """Send the skill-recovery prompt to the OpenAI API."""
    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    return response.output_text


def parse_skills(
    resume_id: str,
    raw_output: str,
) -> list[dict]:
    """Parse, validate, and deduplicate GPT skill predictions."""
    cleaned_output = clean_json_text(raw_output)
    data = json.loads(cleaned_output)

    skills = data.get("skills", [])

    if not isinstance(skills, list):
        raise ValueError(
            "GPT output field 'skills' must be a list."
        )

    parsed_skills = []
    seen = set()

    for item in skills:
        if not isinstance(item, dict):
            continue

        entity_text = normalize_entity_text(
            item.get("entity_text", "")
        )

        entity_type = str(
            item.get("entity_type", "SKILL")
        ).strip().upper()

        if not entity_text:
            continue

        if entity_type not in ALLOWED_ENTITY_TYPES:
            continue

        key = (
            resume_id,
            entity_text.casefold(),
            entity_type,
        )

        if key in seen:
            continue

        seen.add(key)

        parsed_skills.append(
            {
                "resume_id": resume_id,
                "entity_text": entity_text,
                "entity_type": "SKILL",
            }
        )

    return parsed_skills


def main() -> None:
    """Run GPT skill recovery on the existing 20-resume list."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Set it before running this script."
        )

    if not FULL_LIST.exists():
        raise FileNotFoundError(
            f"Resume list not found: {FULL_LIST}"
        )

    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"Skill recovery prompt not found: {PROMPT_FILE}"
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = OpenAI()
    full_df = pd.read_csv(FULL_LIST)

    required_columns = {"resume_id", "text_file"}

    missing_columns = required_columns - set(full_df.columns)

    if missing_columns:
        raise ValueError(
            "Resume list is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    all_skills = []
    failed_resumes = []

    for _, row in full_df.iterrows():
        resume_id = str(row["resume_id"]).strip()
        text_file = (
            PROJECT_ROOT
            / str(row["text_file"]).strip()
        )

        json_output_path = (
            OUTPUT_FOLDER
            / f"{resume_id}_gpt_skills.json"
        )

        if not text_file.exists():
            print(
                f"FAILED: Missing text file for "
                f"{resume_id}: {text_file}"
            )

            failed_resumes.append(resume_id)
            continue

        if is_valid_json_output(json_output_path):
            print(
                f"Skipping {resume_id}: "
                "valid skill JSON already exists."
            )

            raw_output = json_output_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            try:
                parsed_skills = parse_skills(
                    resume_id,
                    raw_output,
                )

                all_skills.extend(parsed_skills)

            except Exception as exc:
                print(
                    f"Existing output could not be parsed "
                    f"for {resume_id}: {exc}"
                )

                failed_resumes.append(resume_id)

            continue

        resume_text = text_file.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        print(f"Processing {resume_id}...")
        print(
            f"Text length: "
            f"{len(resume_text)} characters"
        )

        try:
            prompt = load_prompt(
                resume_id,
                resume_text,
            )

            raw_output = call_gpt(
                client,
                prompt,
            )

            json_output_path.write_text(
                raw_output,
                encoding="utf-8",
            )

            parsed_skills = parse_skills(
                resume_id,
                raw_output,
            )

            all_skills.extend(parsed_skills)

            print(
                f"Skills extracted: "
                f"{len(parsed_skills)}"
            )

            print(f"Saved: {json_output_path}")

        except Exception as exc:
            print(f"FAILED: {resume_id}")
            print(
                f"Error type: "
                f"{type(exc).__name__}"
            )
            print(f"Error message: {exc}")

            failed_resumes.append(resume_id)

        time.sleep(1)

    combined_df = pd.DataFrame(
        all_skills,
        columns=[
            "resume_id",
            "entity_text",
            "entity_type",
        ],
    )

    if not combined_df.empty:
        combined_df = combined_df.drop_duplicates(
            subset=[
                "resume_id",
                "entity_text",
                "entity_type",
            ]
        )

        combined_df = combined_df.sort_values(
            by=[
                "resume_id",
                "entity_text",
            ],
            key=lambda column: column.astype(str).str.casefold(),
        )

    combined_df.to_csv(
        COMBINED_OUTPUT,
        index=False,
    )

    print("\nGPT skill recovery run complete.")
    print(
        f"Combined valid skills: "
        f"{len(combined_df)}"
    )
    print(
        f"Saved combined CSV: "
        f"{COMBINED_OUTPUT}"
    )

    if failed_resumes:
        print("\nFailed resumes:")

        for resume_id in failed_resumes:
            print(f"- {resume_id}")

    else:
        print(
            "\nAll 20 resumes completed successfully."
        )


if __name__ == "__main__":
    main()