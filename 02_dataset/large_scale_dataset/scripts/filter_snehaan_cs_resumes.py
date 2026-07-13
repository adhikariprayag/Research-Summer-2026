import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

input_path = BASE_DIR / "02_dataset/large_scale_dataset/raw_private/Resume/Resume.csv"

json_output_dir = BASE_DIR / "02_dataset/large_scale_dataset/json_records_private/snehaan_information_technology"
metadata_dir = BASE_DIR / "02_dataset/large_scale_dataset/metadata"

json_output_dir.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

cs_df = df[df["Category"] == "INFORMATION-TECHNOLOGY"].copy()

metadata_rows = []

for i, row in cs_df.reset_index(drop=True).iterrows():
    resume_id = f"SNE_IT_{i+1:04d}"

    resume_text = str(row["Resume_str"])

    json_record = {
        "resume_id": resume_id,
        "source_dataset": "snehaanbhawal/resume-dataset",
        "original_category": row["Category"],
        "record_type": "resume_text",
        "file_type": "csv_text",
        "resume_text": resume_text
    }

    output_path = json_output_dir / f"{resume_id}.json"

    pd.Series(json_record).to_json(output_path, indent=2)

    metadata_rows.append({
        "resume_id": resume_id,
        "source_dataset": "snehaanbhawal/resume-dataset",
        "original_category": row["Category"],
        "file_type": "csv_text",
        "is_cs_related": "yes",
        "selected_for_large_dataset": "yes",
        "selected_for_annotation": "no",
        "word_count": len(resume_text.split()),
        "notes": "INFORMATION-TECHNOLOGY category"
    })

metadata_df = pd.DataFrame(metadata_rows)

metadata_output_path = metadata_dir / "snehaan_information_technology_metadata.csv"
metadata_df.to_csv(metadata_output_path, index=False)

print("Done.")
print(f"CS resumes selected: {len(cs_df)}")
print(f"JSON records saved to: {json_output_dir}")
print(f"Metadata saved to: {metadata_output_path}")
