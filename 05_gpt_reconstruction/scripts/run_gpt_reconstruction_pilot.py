from pathlib import Path
import json
import os
import re
import time

import pandas as pd
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PILOT_LIST = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "pilot_inputs"
    / "gpt_pilot_resume_list.csv"
)

PROMPT_FILE = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "prompts"
    / "gpt_entity_reconstruction_prompt.md"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "pilot_outputs"
)

COMBINED_OUTPUT = (
    OUTPUT_FOLDER
    / "gpt_pilot_combined_entities.csv"
)

MODEL_NAME = "gpt-4o-mini"

ALLOWED_ENTITY_TYPES = {"PERSON", "ORG", "DATE", "GPE", "SKILL"}


def clean_json_text(text: str) -> str:
    """Remove accidental markdown fences if the model returns them."""
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def normalize_entity_text(text: str) -> str:
    """Normalize spacing in entity text."""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_prompt(resume_id: str, resume_text: str) -> str:
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    prompt = prompt_template.replace("RESUME_ID_HERE", resume_id)
    prompt = prompt.replace("RESUME_TEXT_HERE", resume_text)

    return prompt


def call_gpt(client: OpenAI, prompt: str) -> str:
    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0,
    )

    return response.output_text


def parse_entities(resume_id: str, raw_output: str) -> list[dict]:
    cleaned = clean_json_text(raw_output)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse JSON for {resume_id}. Raw output:\n{raw_output}"
        ) from exc

    entities = data.get("entities", [])

    parsed_entities = []
    seen = set()

    for item in entities:
        entity_text = normalize_entity_text(item.get("entity_text", ""))
        entity_type = str(item.get("entity_type", "")).strip().upper()

        if not entity_text:
            continue

        if entity_type not in ALLOWED_ENTITY_TYPES:
            continue

        key = (resume_id, entity_text.lower(), entity_type)

        if key in seen:
            continue

        seen.add(key)

        parsed_entities.append(
            {
                "resume_id": resume_id,
                "entity_text": entity_text,
                "entity_type": entity_type,
            }
        )

    return parsed_entities


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Set it before running this script."
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    pilot_df = pd.read_csv(PILOT_LIST)

    all_entities = []

    for _, row in pilot_df.iterrows():
        resume_id = str(row["resume_id"]).strip()
        text_file = PROJECT_ROOT / str(row["text_file"]).strip()

        if not text_file.exists():
            raise FileNotFoundError(f"Missing text file: {text_file}")

        resume_text = text_file.read_text(encoding="utf-8", errors="ignore")

        print(f"Processing {resume_id}...")
        print(f"Text length: {len(resume_text)} characters")

        prompt = load_prompt(resume_id, resume_text)
        raw_output = call_gpt(client, prompt)

        json_output_path = OUTPUT_FOLDER / f"{resume_id}_gpt_entities.json"
        json_output_path.write_text(raw_output, encoding="utf-8")

        parsed_entities = parse_entities(resume_id, raw_output)
        all_entities.extend(parsed_entities)

        print(f"Entities extracted: {len(parsed_entities)}")
        print(f"Saved: {json_output_path}")

        # Small delay to avoid rapid repeated requests.
        time.sleep(1)

    combined_df = pd.DataFrame(
        all_entities,
        columns=["resume_id", "entity_text", "entity_type"],
    )

    combined_df.to_csv(COMBINED_OUTPUT, index=False)

    print("\nGPT reconstruction pilot complete.")
    print(f"Combined entities: {len(combined_df)}")
    print(f"Saved combined CSV: {COMBINED_OUTPUT}")


if __name__ == "__main__":
    main()
