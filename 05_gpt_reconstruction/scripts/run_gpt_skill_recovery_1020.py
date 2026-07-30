from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import unicodedata

import pandas as pd

# ---------------------------------------------------------------------------
# Project paths and experiment settings
# ---------------------------------------------------------------------------

# This script is intended to live in:
# ATS_NER_Resume_Research/05_gpt_reconstruction/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

JSON_ROOT = PROJECT_ROOT / "02_dataset" / "large_scale_dataset" / "json_records_private"

PROMPT_FILE = (
    PROJECT_ROOT / "05_gpt_reconstruction" / "prompts" / "gpt_skill_recovery_prompt.md"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "full_1020_outputs"
)

COMBINED_PROGRESS_OUTPUT = (
    OUTPUT_FOLDER / "gpt_skill_recovery_1020_combined_progress.csv"
)
COMBINED_FINAL_OUTPUT = OUTPUT_FOLDER / "gpt_skill_recovery_1020_combined.csv"
STATUS_OUTPUT = OUTPUT_FOLDER / "gpt_skill_recovery_1020_status.csv"
ERROR_OUTPUT = OUTPUT_FOLDER / "gpt_skill_recovery_1020_errors.csv"
FAILED_RAW_FOLDER = OUTPUT_FOLDER / "failed_raw_outputs"

MODEL_NAME = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 8000
MAX_RETRIES = 3
DEFAULT_REQUEST_DELAY_SECONDS = 1.0
EXPECTED_CORPUS_SIZE = 1020

ALLOWED_ENTITY_TYPES = {"SKILL"}

DATASET_PREFIXES = {
    "JIT_CS_": "Jithin",
    "SAU_CS_": "Saugata",
    "SNE_IT_": "Snehaan",
}

EXPECTED_DATASET_COUNTS = {
    "Jithin": 400,
    "Saugata": 500,
    "Snehaan": 120,
}

COMBINED_COLUMNS = [
    "corpus_order",
    "resume_id",
    "dataset_group",
    "entity_text",
    "entity_type",
    "prediction_source",
    "norm",
]

STATUS_COLUMNS = [
    "corpus_order",
    "resume_id",
    "dataset_group",
    "status",
    "skills_extracted",
    "attempts",
    "error_type",
    "error_message",
    "updated_at_utc",
]


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely apply the validated GPT SKILL-recovery method to the "
            "full 1,020-resume corpus."
        )
    )
    parser.add_argument(
        "--run-api",
        action="store_true",
        help=(
            "Make API calls. Without this flag, the script performs a "
            "read-only preflight/dry run."
        ),
    )
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help=(
            "Required with --run-api when the selected target is the complete "
            "corpus. This prevents an accidental paid 1,020-resume run."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Inspect or process only the first N selected resumes.",
    )
    parser.add_argument(
        "--start-after",
        default=None,
        help="Start after this resume ID in sorted corpus order.",
    )
    parser.add_argument(
        "--resume-id",
        action="append",
        default=[],
        help=(
            "Select one resume ID. Repeat the option to select several IDs. "
            "Useful for controlled tests or reruns."
        ),
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_CORPUS_SIZE,
        help=(
            "Expected number of JSON records. Keep the default for the real "
            "experiment; a smaller value is only for isolated fixture tests."
        ),
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help="Seconds to wait after each newly processed resume.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer.")

    if args.expected_count <= 0:
        parser.error("--expected-count must be a positive integer.")

    if args.request_delay < 0:
        parser.error("--request-delay cannot be negative.")

    if args.confirm_full_run and not args.run_api:
        parser.error("--confirm-full-run is only valid with --run-api.")

    return args


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    """Normalize display text without changing punctuation or wording."""

    text = unicodedata.normalize("NFKC", str(text)).strip()
    return re.sub(r"\s+", " ", text)


def normalize_entity_key(text: str) -> str:
    """Create the comparison key used for deduplication."""

    return normalize_entity_text(text).casefold()


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file completely before replacing its destination."""

    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def atomic_write_csv(
    dataframe: pd.DataFrame,
    path: Path,
    columns: list[str] | None = None,
) -> None:
    """Checkpoint a CSV without leaving a half-written destination file."""

    temporary_path = path.with_name(f".{path.name}.tmp")

    if columns is not None:
        dataframe = dataframe.reindex(columns=columns)

    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def infer_dataset_group(resume_id: str) -> str:
    for prefix, dataset_group in DATASET_PREFIXES.items():
        if resume_id.startswith(prefix):
            return dataset_group

    raise ValueError(f"Resume ID does not match a known dataset prefix: {resume_id}")


# ---------------------------------------------------------------------------
# Resume and prompt loading
# ---------------------------------------------------------------------------


def load_resume_record(json_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(
            json_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid resume JSON file: {json_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Resume JSON must contain an object: {json_path}")

    resume_text = data.get("resume_text", "")

    if not isinstance(resume_text, str):
        resume_text = str(resume_text)

    resume_text = resume_text.strip()

    if not resume_text:
        raise ValueError(f"No resume_text was found in: {json_path}")

    data["resume_text"] = resume_text
    return data


def load_prompt_template() -> str:
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    required_placeholders = {
        "RESUME_ID_HERE",
        "RESUME_TEXT_HERE",
    }
    missing_placeholders = {
        placeholder
        for placeholder in required_placeholders
        if placeholder not in prompt_template
    }

    if missing_placeholders:
        raise ValueError(
            "The prompt template is missing placeholders: "
            f"{sorted(missing_placeholders)}"
        )

    return prompt_template


def create_prompt(
    prompt_template: str,
    resume_id: str,
    resume_text: str,
) -> str:
    prompt = prompt_template.replace("RESUME_ID_HERE", resume_id)
    return prompt.replace("RESUME_TEXT_HERE", resume_text)


# ---------------------------------------------------------------------------
# Corpus discovery and validation
# ---------------------------------------------------------------------------


def build_json_file_map() -> dict[str, Path]:
    json_file_map: dict[str, Path] = {}
    duplicate_ids: list[str] = []

    for json_path in sorted(JSON_ROOT.rglob("*.json")):
        resume_id = json_path.stem.strip()

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


def validate_corpus(
    json_file_map: dict[str, Path],
    expected_count: int,
) -> list[dict[str, Any]]:
    if len(json_file_map) != expected_count:
        raise ValueError(
            f"Expected {expected_count} JSON resumes, but found "
            f"{len(json_file_map)}."
        )

    entries: list[dict[str, Any]] = []

    for corpus_order, resume_id in enumerate(sorted(json_file_map), start=1):
        entries.append(
            {
                "corpus_order": corpus_order,
                "resume_id": resume_id,
                "dataset_group": infer_dataset_group(resume_id),
                "json_path": json_file_map[resume_id],
            }
        )

    if expected_count == EXPECTED_CORPUS_SIZE:
        actual_counts = pd.Series(
            [entry["dataset_group"] for entry in entries]
        ).value_counts()

        count_errors = []

        for dataset_group, expected_dataset_count in EXPECTED_DATASET_COUNTS.items():
            actual_dataset_count = int(actual_counts.get(dataset_group, 0))

            if actual_dataset_count != expected_dataset_count:
                count_errors.append(
                    f"{dataset_group}: expected {expected_dataset_count}, "
                    f"found {actual_dataset_count}"
                )

        if count_errors:
            raise ValueError(
                "Dataset-source count validation failed:\n" + "\n".join(count_errors)
            )

    return entries


def select_entries(
    entries: list[dict[str, Any]],
    requested_resume_ids: list[str],
    start_after: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = entries
    all_ids = {entry["resume_id"] for entry in entries}

    if requested_resume_ids:
        cleaned_requested_ids = []

        for resume_id in requested_resume_ids:
            resume_id = str(resume_id).strip()

            if resume_id and resume_id not in cleaned_requested_ids:
                cleaned_requested_ids.append(resume_id)

        missing_ids = sorted(set(cleaned_requested_ids) - all_ids)

        if missing_ids:
            raise ValueError(
                "Requested resume IDs were not found:\n" + "\n".join(missing_ids)
            )

        requested_set = set(cleaned_requested_ids)
        selected = [entry for entry in entries if entry["resume_id"] in requested_set]

    if start_after:
        matching_positions = [
            index
            for index, entry in enumerate(selected)
            if entry["resume_id"] == start_after
        ]

        if not matching_positions:
            raise ValueError(
                f"--start-after ID was not found in the selected set: " f"{start_after}"
            )

        selected = selected[matching_positions[0] + 1 :]

    if limit is not None:
        selected = selected[:limit]

    if not selected:
        raise ValueError("No resumes remain after applying the selection options.")

    return selected


# ---------------------------------------------------------------------------
# GPT output validation and parsing
# ---------------------------------------------------------------------------


def parse_skills(
    resume_id: str,
    raw_output: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    cleaned_output = clean_json_text(raw_output)
    data = json.loads(cleaned_output)

    if not isinstance(data, dict):
        raise ValueError("GPT output must be a JSON object.")

    skills = data.get("skills", [])

    if not isinstance(skills, list):
        raise ValueError("GPT output field 'skills' must be a list.")

    parsed_skills: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for item in skills:
        if not isinstance(item, dict):
            continue

        entity_text = normalize_entity_text(item.get("entity_text", ""))
        entity_type = str(item.get("entity_type", "SKILL")).strip().upper()

        if not entity_text or entity_type not in ALLOWED_ENTITY_TYPES:
            continue

        norm = normalize_entity_key(entity_text)
        deduplication_key = (resume_id, norm, entity_type)

        if deduplication_key in seen:
            continue

        seen.add(deduplication_key)
        parsed_skills.append(
            {
                "corpus_order": metadata["corpus_order"],
                "resume_id": resume_id,
                "dataset_group": metadata["dataset_group"],
                "entity_text": entity_text,
                "entity_type": "SKILL",
                "prediction_source": "GPT",
                "norm": norm,
            }
        )

    return parsed_skills


def read_valid_existing_output(
    output_path: Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if not output_path.exists():
        return None

    try:
        raw_output = output_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        return parse_skills(
            metadata["resume_id"],
            raw_output,
            metadata,
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# API, retry, and private failure logging
# ---------------------------------------------------------------------------


def call_gpt(client: Any, prompt: str) -> str:
    """Use the same model and decoding settings as the 50-resume validation."""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return response.output_text


def retry_wait_seconds(attempt: int) -> int:
    return min(60, 3 * (2 ** (attempt - 1)))


def save_failed_raw_output(
    resume_id: str,
    attempt: int,
    raw_output: str,
) -> None:
    FAILED_RAW_FOLDER.mkdir(parents=True, exist_ok=True)
    failed_path = FAILED_RAW_FOLDER / f"{resume_id}_attempt_{attempt}.txt"
    atomic_write_text(failed_path, raw_output)


# ---------------------------------------------------------------------------
# Status and combined-output checkpoints
# ---------------------------------------------------------------------------


def load_status_map() -> dict[str, dict[str, Any]]:
    if not STATUS_OUTPUT.exists():
        return {}

    try:
        status_df = pd.read_csv(STATUS_OUTPUT, keep_default_na=False)
    except (OSError, pd.errors.ParserError):
        return {}

    if "resume_id" not in status_df.columns:
        return {}

    return {
        str(row["resume_id"]).strip(): row.to_dict() for _, row in status_df.iterrows()
    }


def checkpoint_status(
    status_map: dict[str, dict[str, Any]],
) -> None:
    status_df = pd.DataFrame(status_map.values())

    if not status_df.empty and "corpus_order" in status_df.columns:
        status_df = status_df.sort_values("corpus_order")

    atomic_write_csv(status_df, STATUS_OUTPUT, STATUS_COLUMNS)

    if status_df.empty:
        error_df = pd.DataFrame(columns=STATUS_COLUMNS)
    else:
        error_df = status_df.loc[status_df["status"] == "failed"].copy()

    atomic_write_csv(error_df, ERROR_OUTPUT, STATUS_COLUMNS)


def make_status_row(
    metadata: dict[str, Any],
    status: str,
    skills_extracted: int,
    attempts: int,
    error: Exception | None = None,
) -> dict[str, Any]:
    return {
        "corpus_order": metadata["corpus_order"],
        "resume_id": metadata["resume_id"],
        "dataset_group": metadata["dataset_group"],
        "status": status,
        "skills_extracted": skills_extracted,
        "attempts": attempts,
        "error_type": type(error).__name__ if error else "",
        "error_message": str(error) if error else "",
        "updated_at_utc": utc_now_text(),
    }


def build_combined_progress(
    entries: list[dict[str, Any]],
) -> tuple[pd.DataFrame, int]:
    all_skills: list[dict[str, Any]] = []
    valid_output_count = 0

    for metadata in entries:
        output_path = OUTPUT_FOLDER / (f"{metadata['resume_id']}_gpt_skills.json")
        parsed_skills = read_valid_existing_output(output_path, metadata)

        if parsed_skills is None:
            continue

        valid_output_count += 1
        all_skills.extend(parsed_skills)

    combined_df = pd.DataFrame(all_skills, columns=COMBINED_COLUMNS)

    if not combined_df.empty:
        combined_df = combined_df.drop_duplicates(
            subset=["resume_id", "norm", "entity_type"]
        )
        combined_df = combined_df.sort_values(["corpus_order", "norm"]).reset_index(
            drop=True
        )

    atomic_write_csv(
        combined_df,
        COMBINED_PROGRESS_OUTPUT,
        COMBINED_COLUMNS,
    )

    return combined_df, valid_output_count


# ---------------------------------------------------------------------------
# Read-only preflight
# ---------------------------------------------------------------------------


def run_preflight(
    entries: list[dict[str, Any]],
    selected_entries: list[dict[str, Any]],
    prompt_template: str,
) -> None:
    existing_valid = 0
    existing_invalid = 0
    total_characters = 0

    for metadata in selected_entries:
        resume_record = load_resume_record(metadata["json_path"])
        resume_text = resume_record["resume_text"]
        total_characters += len(resume_text)

        # Construct the prompt to verify that both placeholders resolve, but
        # never print resume text or the full prompt.
        prompt = create_prompt(
            prompt_template,
            metadata["resume_id"],
            resume_text,
        )

        if "RESUME_ID_HERE" in prompt or "RESUME_TEXT_HERE" in prompt:
            raise ValueError(
                f"Prompt placeholders did not resolve for " f"{metadata['resume_id']}."
            )

        output_path = OUTPUT_FOLDER / (f"{metadata['resume_id']}_gpt_skills.json")

        if output_path.exists():
            if read_valid_existing_output(output_path, metadata) is not None:
                existing_valid += 1
            else:
                existing_invalid += 1

    required_api_calls = len(selected_entries) - existing_valid

    print("\nREAD-ONLY PREFLIGHT COMPLETE")
    print(f"Corpus records validated: {len(entries)}")
    print(f"Selected records checked: {len(selected_entries)}")
    print(f"Selected resume characters: {total_characters:,}")
    print(f"Valid existing outputs reusable: {existing_valid}")
    print(f"Invalid existing outputs to rerun: {existing_invalid}")
    print(f"New API calls required for selection: {required_api_calls}")
    print(f"Model preserved from validation: {MODEL_NAME}")
    print("No API calls were made and no output files were changed.")


# ---------------------------------------------------------------------------
# Live resumable processing
# ---------------------------------------------------------------------------


def run_api_pipeline(
    entries: list[dict[str, Any]],
    selected_entries: list[dict[str, Any]],
    prompt_template: str,
    request_delay: float,
) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Set it before using --run-api."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required only for --run-api. "
            "Install it in the project virtual environment."
        ) from exc

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    status_map = load_status_map()
    client = OpenAI()

    processed_now = 0
    skipped_existing = 0
    failed_now = 0

    print(f"\nSelected resumes: {len(selected_entries)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Private output folder: {OUTPUT_FOLDER}")

    for position, metadata in enumerate(selected_entries, start=1):
        resume_id = metadata["resume_id"]
        output_path = OUTPUT_FOLDER / f"{resume_id}_gpt_skills.json"

        print(f"\n[{position}/{len(selected_entries)}] {resume_id}")

        parsed_existing = read_valid_existing_output(
            output_path,
            metadata,
        )

        if parsed_existing is not None:
            skipped_existing += 1
            status_map[resume_id] = make_status_row(
                metadata=metadata,
                status="skipped_existing",
                skills_extracted=len(parsed_existing),
                attempts=0,
            )
            checkpoint_status(status_map)
            print(f"Skipped valid existing output " f"({len(parsed_existing)} skills).")
            continue

        try:
            resume_record = load_resume_record(metadata["json_path"])
            resume_text = resume_record["resume_text"]
            prompt = create_prompt(
                prompt_template,
                resume_id,
                resume_text,
            )
        except Exception as exc:
            failed_now += 1
            status_map[resume_id] = make_status_row(
                metadata=metadata,
                status="failed",
                skills_extracted=0,
                attempts=0,
                error=exc,
            )
            checkpoint_status(status_map)
            print(f"FAILED before API call: {type(exc).__name__}: {exc}")
            continue

        successful = False
        final_error: Exception | None = None
        attempts_used = 0

        for attempt in range(1, MAX_RETRIES + 1):
            attempts_used = attempt
            raw_output = ""

            try:
                print(f"API attempt {attempt}/{MAX_RETRIES}...")
                raw_output = call_gpt(client, prompt)
                parsed_skills = parse_skills(
                    resume_id,
                    raw_output,
                    metadata,
                )

                # The canonical per-resume output is changed only after the
                # response passes JSON/schema parsing.
                atomic_write_text(output_path, raw_output)

                status_map[resume_id] = make_status_row(
                    metadata=metadata,
                    status="processed",
                    skills_extracted=len(parsed_skills),
                    attempts=attempt,
                )
                checkpoint_status(status_map)

                processed_now += 1
                successful = True
                print(f"Saved valid output ({len(parsed_skills)} skills).")
                break

            except Exception as exc:
                final_error = exc
                print(f"Attempt failed: {type(exc).__name__}: {exc}")

                if raw_output:
                    save_failed_raw_output(
                        resume_id=resume_id,
                        attempt=attempt,
                        raw_output=raw_output,
                    )

                if attempt < MAX_RETRIES:
                    wait_seconds = retry_wait_seconds(attempt)
                    print(f"Retrying after {wait_seconds} seconds...")
                    time.sleep(wait_seconds)

        if not successful:
            failed_now += 1
            status_map[resume_id] = make_status_row(
                metadata=metadata,
                status="failed",
                skills_extracted=0,
                attempts=attempts_used,
                error=final_error,
            )
            checkpoint_status(status_map)

        if successful and request_delay:
            time.sleep(request_delay)

    combined_df, valid_output_count = build_combined_progress(entries)

    # Never present a partial result as the final full-corpus result.
    if valid_output_count == len(entries):
        atomic_write_csv(
            combined_df,
            COMBINED_FINAL_OUTPUT,
            COMBINED_COLUMNS,
        )

    print("\nGPT SKILL-RECOVERY RUN FINISHED")
    print(f"Processed in this run: {processed_now}")
    print(f"Skipped valid existing: {skipped_existing}")
    print(f"Failed in this run: {failed_now}")
    print(
        f"Valid per-resume outputs now available: "
        f"{valid_output_count}/{len(entries)}"
    )
    print(f"Combined progress skills: {len(combined_df)}")
    print(f"Status checkpoint: {STATUS_OUTPUT}")
    print(f"Error log: {ERROR_OUTPUT}")
    print(f"Combined progress CSV: {COMBINED_PROGRESS_OUTPUT}")

    if valid_output_count == len(entries):
        print(f"Final complete CSV: {COMBINED_FINAL_OUTPUT}")
    else:
        print(
            "Final complete CSV was not written because the corpus is "
            "not yet 100% complete."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if not JSON_ROOT.exists():
        raise FileNotFoundError(f"Private JSON directory was not found:\n{JSON_ROOT}")

    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Skill-recovery prompt was not found:\n{PROMPT_FILE}")

    prompt_template = load_prompt_template()
    json_file_map = build_json_file_map()
    entries = validate_corpus(
        json_file_map=json_file_map,
        expected_count=args.expected_count,
    )
    selected_entries = select_entries(
        entries=entries,
        requested_resume_ids=args.resume_id,
        start_after=args.start_after,
        limit=args.limit,
    )

    full_corpus_selected = (
        len(selected_entries) == len(entries)
        and not args.resume_id
        and args.start_after is None
        and args.limit is None
    )

    if args.run_api and full_corpus_selected and not args.confirm_full_run:
        raise PermissionError(
            "The complete paid API run requires both --run-api and "
            "--confirm-full-run. Run the read-only preflight first."
        )

    if not args.run_api:
        run_preflight(
            entries=entries,
            selected_entries=selected_entries,
            prompt_template=prompt_template,
        )
        return

    run_api_pipeline(
        entries=entries,
        selected_entries=selected_entries,
        prompt_template=prompt_template,
        request_delay=args.request_delay,
    )


if __name__ == "__main__":
    main()
