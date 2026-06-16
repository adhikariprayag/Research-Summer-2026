from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPACY_FOLDER = (
    PROJECT_ROOT
    / "04_spacy_baseline"
    / "batch2_spacy_outputs"
)

SKILL_FOLDER = (
    PROJECT_ROOT
    / "04_spacy_baseline"
    / "batch2_skill_outputs"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "04_spacy_baseline"
    / "batch2_combined_outputs"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def load_csv_safely(file_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(
            columns=[
                "resume_id",
                "entity_text",
                "entity_type",
            ]
        )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "resume_file": "resume_id",
        "text": "entity_text",
        "skill": "entity_text",
        "entity_label": "entity_type",
    }

    df = df.rename(columns=rename_map)

    required_columns = [
        "resume_id",
        "entity_text",
        "entity_type",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = pd.Series(dtype="object")

    return df[required_columns]


def clean_entities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df.empty:
        return df

    df["resume_id"] = (
        df["resume_id"]
        .astype(str)
        .str.strip()
    )

    df["entity_text"] = (
        df["entity_text"]
        .astype(str)
        .str.strip()
    )

    df["entity_type"] = (
        df["entity_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[
        (df["entity_text"] != "")
        & (df["entity_text"].str.lower() != "nan")
    ]

    df = df.drop_duplicates(
        subset=[
            "resume_id",
            "entity_text",
            "entity_type",
        ]
    )

    return df


def main():
    spacy_files = sorted(
        SPACY_FOLDER.glob("*_spacy_entities.csv")
    )

    if not spacy_files:
        print(f"No Batch 2 spaCy files found in: {SPACY_FOLDER}")
        return

    print(f"Found {len(spacy_files)} Batch 2 spaCy files.\n")

    for spacy_file in spacy_files:
        resume_id = spacy_file.name.replace(
            "_spacy_entities.csv",
            "",
        )

        skill_file = (
            SKILL_FOLDER
            / f"{resume_id}_skill_entities.csv"
        )

        if not skill_file.exists():
            print(f"Missing SKILL file for: {resume_id}\n")
            continue

        try:
            spacy_df = load_csv_safely(spacy_file)
            skill_df = load_csv_safely(skill_file)

            spacy_df = normalize_columns(spacy_df)
            skill_df = normalize_columns(skill_df)

            combined_df = pd.concat(
                [spacy_df, skill_df],
                ignore_index=True,
            )

            combined_df = clean_entities(combined_df)

            output_file = (
                OUTPUT_FOLDER
                / f"{resume_id}_combined_entities.csv"
            )

            combined_df.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig",
            )

            print(f"Processed: {resume_id}")
            print(f"spaCy rows: {len(spacy_df)}")
            print(f"SKILL rows: {len(skill_df)}")
            print(f"Combined rows: {len(combined_df)}")
            print(f"Saved: {output_file.name}\n")

        except Exception as error:
            print(f"Failed: {resume_id}")
            print(f"Reason: {error}\n")

    print("Batch 2 output combination completed.")
    print(f"Outputs saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()