import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

input_path = BASE_DIR / "02_dataset/large_scale_dataset/annotation_subset_private/draft_saugata_annotations_003_050.csv"
output_path = BASE_DIR / "02_dataset/large_scale_dataset/annotation_subset_private/draft_saugata_annotations_003_050_cleaned.csv"

df = pd.read_csv(input_path)

# Remove exact duplicate resume/entity/type combinations
df["entity_text_clean"] = df["entity_text"].astype(str).str.strip()
df["entity_lower"] = df["entity_text_clean"].str.lower()

bad_entities = {
    "company name",
    "city, state",
    "city state",
    "n/a",
    "none",
    "nan",
    "management",
    "team leadership",
    "verbal communication",
    "customer support",
    "inventory management",
}

df = df[~df["entity_lower"].isin(bad_entities)].copy()

df = df.drop_duplicates(
    subset=["resume_id", "entity_lower", "entity_type"],
    keep="first"
)

df = df.drop(columns=["entity_text_clean", "entity_lower"])

df.to_csv(output_path, index=False)

print("Cleaned draft created.")
print(f"Original rows: {len(pd.read_csv(input_path))}")
print(f"Cleaned rows: {len(df)}")
print(f"Saved to: {output_path}")

print("\nEntity type counts:")
print(df["entity_type"].value_counts())

print("\nRows for SAU_CS_0003 after cleaning:")
print(df[df["resume_id"] == "SAU_CS_0003"].to_string(index=False))
