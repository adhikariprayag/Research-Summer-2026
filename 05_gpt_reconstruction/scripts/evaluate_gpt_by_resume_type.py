import pandas as pd
import re
from pathlib import Path
from collections import Counter

# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

gold_path = PROJECT_ROOT / "02_dataset/gold_standard_annotations/gold_standard_final_20.csv"
gpt_path = PROJECT_ROOT / "05_gpt_reconstruction/full_outputs/gpt_full_20_combined_entities.csv"

output_dir = PROJECT_ROOT / "05_gpt_reconstruction/full_evaluation_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "gpt_by_resume_type_summary.csv"

# -----------------------------
# Resume type mapping
# -----------------------------

creative_resumes = {
    "R002_creative",
    "R003_complex",
    "R004_creative_text",
    "R005_complex_text",
    "R014_kaggle",
    "R017_kaggle",
    "R026_kaggle_creative",
    "R027_creative",
    "R028_creative",
    "R029_creative",
}

standard_resumes = {
    "R006_kaggle",
    "R007_kaggle",
    "R008_kaggle",
    "R010_kaggle",
    "R011_kaggle",
    "R012_kaggle",
    "R015_kaggle",
    "R018_kaggle",
    "R019_kaggle",
    "R034_kaggle",
}

resume_type_map = {}

for rid in creative_resumes:
    resume_type_map[rid] = "Creative"

for rid in standard_resumes:
    resume_type_map[rid] = "Standard"

# -----------------------------
# Helpers
# -----------------------------

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def standardize_columns(df):
    """
    Makes the script more flexible in case column names differ slightly.
    Expected final columns:
    resume_id, entity_text, entity_type
    """

    rename_map = {}

    for col in df.columns:
        lower = col.lower().strip()

        if lower in ["resume_id", "resume", "file_id"]:
            rename_map[col] = "resume_id"

        elif lower in ["entity_text", "text", "entity", "extracted_text"]:
            rename_map[col] = "entity_text"

        elif lower in ["entity_type", "label", "type"]:
            rename_map[col] = "entity_type"

    df = df.rename(columns=rename_map)

    required = {"resume_id", "entity_text", "entity_type"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}. Current columns: {list(df.columns)}")

    return df[["resume_id", "entity_text", "entity_type"]].copy()

def make_counter(df):
    """
    Uses Counter instead of set so duplicate Gold Standard entities are counted correctly.
    """

    keys = []

    for _, row in df.iterrows():
        key = (
            str(row["resume_id"]).strip(),
            normalize_text(row["entity_text"]),
            str(row["entity_type"]).strip().upper()
        )
        keys.append(key)

    return Counter(keys)

def evaluate(gold_df, system_df):
    gold_counter = make_counter(gold_df)
    system_counter = make_counter(system_df)

    correct = 0

    for key in system_counter:
        correct += min(system_counter[key], gold_counter.get(key, 0))

    gold_total = sum(gold_counter.values())
    system_total = sum(system_counter.values())

    missed = gold_total - correct
    incorrect = system_total - correct

    precision = correct / system_total if system_total else 0
    recall = correct / gold_total if gold_total else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {
        "Gold": gold_total,
        "System": system_total,
        "Correct": correct,
        "Missed": missed,
        "Incorrect": incorrect,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3)
    }

# -----------------------------
# Load data
# -----------------------------

gold_df = pd.read_csv(gold_path)
gpt_df = pd.read_csv(gpt_path)

gold_df = standardize_columns(gold_df)
gpt_df = standardize_columns(gpt_df)

gold_df["resume_type"] = gold_df["resume_id"].map(resume_type_map)
gpt_df["resume_type"] = gpt_df["resume_id"].map(resume_type_map)

# Check for missing resume types
missing_gold = gold_df[gold_df["resume_type"].isna()]["resume_id"].unique()
missing_gpt = gpt_df[gpt_df["resume_type"].isna()]["resume_id"].unique()

if len(missing_gold) > 0:
    print("Warning: Missing resume type for Gold resumes:", missing_gold)

if len(missing_gpt) > 0:
    print("Warning: Missing resume type for GPT resumes:", missing_gpt)

# -----------------------------
# Evaluate by resume type
# -----------------------------

rows = []

for resume_type in ["Standard", "Creative"]:
    gold_subset = gold_df[gold_df["resume_type"] == resume_type]
    gpt_subset = gpt_df[gpt_df["resume_type"] == resume_type]

    result = evaluate(gold_subset, gpt_subset)
    result["Condition"] = "GPT Reconstruction"
    result["Resume Type"] = resume_type

    rows.append(result)

# Overall GPT result
overall_result = evaluate(gold_df, gpt_df)
overall_result["Condition"] = "GPT Reconstruction"
overall_result["Resume Type"] = "Overall"
rows.append(overall_result)

summary_df = pd.DataFrame(rows)

summary_df = summary_df[
    [
        "Condition",
        "Resume Type",
        "Gold",
        "System",
        "Correct",
        "Missed",
        "Incorrect",
        "Precision",
        "Recall",
        "F1"
    ]
]

summary_df.to_csv(output_path, index=False)

print("\nGPT Reconstruction by Resume Type")
print(summary_df.to_string(index=False))
print(f"\nSaved summary to: {output_path}")