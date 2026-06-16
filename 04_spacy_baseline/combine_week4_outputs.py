from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPACY_FOLDER = PROJECT_ROOT / "04_spacy_baseline" / "week4_spacy_outputs"

SKILL_FOLDER = PROJECT_ROOT / "04_spacy_baseline" / "week4_skill_outputs"

OUTPUT_FOLDER = PROJECT_ROOT / "04_spacy_baseline" / "week4_combined_outputs"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize possible column names."""
    rename_map = {
        "resume_file": "resume_id",
        "text": "entity_text",
        "skill": "entity_text",
        "entity_label": "entity_type",
    }

    df = df.rename(columns=rename_map)

    required_columns = {
        "resume_id",
        "entity_text",
        "entity_type",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df[
        [
            "resume_id",
            "entity_text",
            "entity_type",
        ]
    ]


def clean_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text and remove duplicate entity rows."""
    df = df.copy()

    df["resume_id"] = df["resume_id"].astype(str).str.strip()

    df["entity_text"] = df["entity_text"].astype(str).str.strip()

    df["entity_type"] = df["entity_type"].astype(str).str.strip().str.upper()

    df = df[df["entity_text"] != ""]

    df = df.drop_duplicates(
        subset=[
            "resume_id",
            "entity_text",
            "entity_type",
        ]
    )

    return df


def main() -> None:
    spacy_files = sorted(SPACY_FOLDER.glob("*_spacy_entities.csv"))

    if not spacy_files:
        print(f"No spaCy files found in: {SPACY_FOLDER}")
        return

    print(f"Found {len(spacy_files)} spaCy files.\n")

    for spacy_file in spacy_files:
        resume_id = spacy_file.name.replace(
            "_spacy_entities.csv",
            "",
        )

        skill_file = SKILL_FOLDER / f"{resume_id}_skill_entities.csv"

        if not skill_file.exists():
            print(f"Missing SKILL file for: {resume_id}")
            print(f"Expected: {skill_file.name}\n")
            continue

        try:
            spacy_df = pd.read_csv(spacy_file)
            skill_df = pd.read_csv(skill_file)

            spacy_df = normalize_columns(spacy_df)
            skill_df = normalize_columns(skill_df)

            combined_df = pd.concat(
                [spacy_df, skill_df],
                ignore_index=True,
            )

            combined_df = clean_entities(combined_df)

            combined_df = combined_df.sort_values(
                by=[
                    "entity_type",
                    "entity_text",
                ],
                key=lambda column: column.astype(str).str.lower(),
            )

            output_name = f"{resume_id}_combined_entities.csv"

            output_path = OUTPUT_FOLDER / output_name

            combined_df.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
            )

            print(f"Processed: {resume_id}")
            print(f"spaCy rows: {len(spacy_df)}")
            print(f"SKILL rows: {len(skill_df)}")
            print("Combined after duplicate removal: " f"{len(combined_df)}")

            counts = combined_df["entity_type"].value_counts().to_dict()

            print(f"Entity counts: {counts}")
            print(f"Saved: {output_name}\n")

        except Exception as error:
            print(f"Failed: {resume_id}")
            print(f"Reason: {error}\n")

    print("Week 4 output combination completed.")
    print(f"Outputs saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
