from pathlib import Path
import pandas as pd

# Input paths
gold_file = Path("../02_dataset/gold_standard_annotations/gold_standard_sample.csv")
combined_folder = Path("../04_spacy_baseline/combined_entity_outputs")

# Output folder
output_folder = Path("sample_evaluation_outputs")
output_folder.mkdir(exist_ok=True)


def clean_text(value):
    return str(value).strip()


def normalize_resume_id(value):
    """
    Makes resume IDs match even if one file has:
    R001_standard
    R001_standard_pymupdf
    R001_standard_pymupdf.txt
    R001_standard_pymupdf.pdf
    """
    value = str(value).strip()

    value = value.replace(".txt", "")
    value = value.replace(".pdf", "")
    value = value.replace("_combined_entities", "")

    return value


def make_match_key(entity_text, entity_type):
    """
    Exact matching for now:
    same text + same label.
    """
    return f"{str(entity_text).strip().lower()}||{str(entity_type).strip().upper()}"


# Load Gold Standard
gold_df = pd.read_csv(gold_file)

# Clean column names
gold_df.columns = gold_df.columns.str.strip().str.lower()

# Clean values
gold_df["resume_id"] = gold_df["resume_id"].apply(normalize_resume_id)
gold_df["entity_text"] = gold_df["entity_text"].apply(clean_text)
gold_df["entity_type"] = gold_df["entity_type"].apply(lambda x: str(x).strip().upper())

print("\nGold Standard resume IDs found:")
print(gold_df["resume_id"].unique())

all_results = []
summary_rows = []

for combined_file in combined_folder.glob("*_combined_entities.csv"):
    print(f"\nProcessing: {combined_file.name}")

    combined_df = pd.read_csv(combined_file)

    # Clean column names
    combined_df.columns = combined_df.columns.str.strip().str.lower()

    # Rename combined output column if needed
    if "text" in combined_df.columns:
        combined_df = combined_df.rename(columns={"text": "entity_text"})

    # Normalize combined file values
    combined_df["resume_id"] = combined_df["resume_id"].apply(normalize_resume_id)
    combined_df["entity_text"] = combined_df["entity_text"].apply(clean_text)
    combined_df["entity_type"] = combined_df["entity_type"].apply(
        lambda x: str(x).strip().upper()
    )

    # Use resume_id from file name, not from first row
    resume_id = combined_file.stem.replace("_combined_entities", "")
    resume_id = normalize_resume_id(resume_id)

    print("Using resume_id:", resume_id)

    gold_resume = gold_df[gold_df["resume_id"] == resume_id].copy()
    system_resume = combined_df[combined_df["resume_id"] == resume_id].copy()

    print("Gold rows found:", len(gold_resume))
    print("System rows found:", len(system_resume))

    if len(gold_resume) == 0:
        print("WARNING: No matching Gold Standard rows found for this resume.")
        print(
            "Check whether the resume_id in gold_standard_sample.csv matches:",
            resume_id,
        )

    # Create match keys
    gold_resume["match_key"] = gold_resume.apply(
        lambda row: make_match_key(row["entity_text"], row["entity_type"]), axis=1
    )

    system_resume["match_key"] = system_resume.apply(
        lambda row: make_match_key(row["entity_text"], row["entity_type"]), axis=1
    )

    gold_keys = set(gold_resume["match_key"])
    system_keys = set(system_resume["match_key"])

    correct_keys = gold_keys.intersection(system_keys)
    missed_keys = gold_keys - system_keys
    incorrect_keys = system_keys - gold_keys

    # Detailed results: Gold Standard side
    for _, row in gold_resume.iterrows():
        status = "Correct" if row["match_key"] in correct_keys else "Missed"
        all_results.append(
            {
                "resume_id": resume_id,
                "entity_text": row["entity_text"],
                "entity_type": row["entity_type"],
                "status": status,
            }
        )

    # Detailed results: System-only incorrect outputs
    for _, row in system_resume.iterrows():
        if row["match_key"] in incorrect_keys:
            all_results.append(
                {
                    "resume_id": resume_id,
                    "entity_text": row["entity_text"],
                    "entity_type": row["entity_type"],
                    "status": "Incorrect",
                }
            )

    correct = len(correct_keys)
    missed = len(missed_keys)
    incorrect = len(incorrect_keys)

    precision = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
    recall = correct / (correct + missed) if (correct + missed) > 0 else 0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    summary_rows.append(
        {
            "resume_id": resume_id,
            "gold_rows": len(gold_resume),
            "system_rows": len(system_resume),
            "correct": correct,
            "missed": missed,
            "incorrect": incorrect,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
    )

    print(f"Correct: {correct}")
    print(f"Missed: {missed}")
    print(f"Incorrect: {incorrect}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

# Save outputs
details_df = pd.DataFrame(all_results)
summary_df = pd.DataFrame(summary_rows)

details_output = output_folder / "sample_gold_vs_combined_details.csv"
summary_output = output_folder / "sample_gold_vs_combined_summary.csv"

details_df.to_csv(details_output, index=False)
summary_df.to_csv(summary_output, index=False)

print("\nEvaluation complete.")
print(f"Saved details to: {details_output}")
print(f"Saved summary to: {summary_output}")
