from pathlib import Path
import pandas as pd

# Folders
spacy_folder = Path("sample_ner_outputs")
skills_folder = Path("skill_matcher_outputs")
combined_folder = Path("combined_entity_outputs")

combined_folder.mkdir(exist_ok=True)


def clean_column_names(df):
    """
    Removes hidden spaces and normalizes column names.
    Example: ' entity_text ' becomes 'entity_text'
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def standardize_spacy_columns(df, file_name):
    """
    Converts spaCy CSV columns into:
    resume_id, text, entity_type
    """

    df = clean_column_names(df)

    print("spaCy columns found:", list(df.columns))

    rename_map = {
        "resume_file": "resume_id",
        "entity_text": "text",
        "entity_label": "entity_type",
    }

    df = df.rename(columns=rename_map)

    required_columns = ["resume_id", "text", "entity_type"]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"ERROR in {file_name}: missing columns {missing}")
        print("Available columns:", list(df.columns))
        raise KeyError(f"Could not standardize spaCy columns for {file_name}")

    return df[required_columns]


def standardize_skill_columns(df, file_name):
    """
    Converts skill matcher CSV columns into:
    resume_id, text, entity_type
    """

    df = clean_column_names(df)

    print("Skill columns found:", list(df.columns))

    # Handle possible skill column names
    if "skill" in df.columns:
        df = df.rename(columns={"skill": "text"})

    if "resume_file" in df.columns:
        df = df.rename(columns={"resume_file": "resume_id"})

    if "resume_id" not in df.columns:
        df["resume_id"] = file_name.replace("_skills.csv", "")

    if "text" not in df.columns:
        print(f"ERROR in {file_name}: no skill/text column found")
        print("Available columns:", list(df.columns))
        raise KeyError(f"Could not standardize skill columns for {file_name}")

    df["entity_type"] = "SKILL"

    return df[["resume_id", "text", "entity_type"]]


for spacy_file in spacy_folder.glob("*_spacy_entities.csv"):

    base_name = spacy_file.stem.replace("_spacy_entities", "")
    skill_file = skills_folder / f"{base_name}_skills.csv"

    print(f"\nProcessing: {base_name}")

    # Read spaCy CSV
    spacy_df = pd.read_csv(spacy_file)
    spacy_df = standardize_spacy_columns(spacy_df, spacy_file.name)

    print("spaCy entity types:")
    print(spacy_df["entity_type"].value_counts())

    # Read skill CSV
    if skill_file.exists():
        skills_df = pd.read_csv(skill_file)
        skills_df = standardize_skill_columns(skills_df, skill_file.name)

        print("Skill entity types:")
        print(skills_df["entity_type"].value_counts())
    else:
        print(f"Warning: No skill file found for {base_name}")
        skills_df = pd.DataFrame(columns=["resume_id", "text", "entity_type"])

    # Combine
    combined_df = pd.concat([spacy_df, skills_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates()

    # Save
    output_file = combined_folder / f"{base_name}_combined_entities.csv"
    combined_df.to_csv(output_file, index=False)

    print("Combined entity types:")
    print(combined_df["entity_type"].value_counts())

    print(f"Saved: {output_file}")

print("\nAll combined entity files created successfully.")
