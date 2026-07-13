import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

input_path = BASE_DIR / "02_dataset/large_scale_dataset/raw_private/jithin_resume_dataset/gpt_dataset.csv"

json_output_dir = BASE_DIR / "02_dataset/large_scale_dataset/json_records_private/jithin_cs_resumes"
metadata_dir = BASE_DIR / "02_dataset/large_scale_dataset/metadata"

json_output_dir.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

metadata_rows = []

for i, row in df.reset_index(drop=True).iterrows():
    resume_id = f"JIT_CS_{i+1:04d}"

    resume_text = str(row["Resume"])
    category = str(row["Category"])

    json_record = {
        "resume_id": resume_id,
        "source_dataset": "jithinjagadeesh/resume-dataset",
        "original_category": category,
        "record_type": "resume_text",
        "file_type": "csv_text",
        "source_note": "possibly synthetic/generated because source file is named gpt_dataset.csv",
        "resume_text": resume_text
    }

    output_path = json_output_dir / f"{resume_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_record, f, indent=2)

    metadata_rows.append({
        "resume_id": resume_id,
        "source_dataset": "jithinjagadeesh/resume-dataset",
        "original_category": category,
        "file_type": "csv_text",
        "is_cs_related": "yes",
        "selected_for_large_dataset": "yes",
        "selected_for_annotation": "no",
        "word_count": len(resume_text.split()),
        "notes": "CS-specific resume text; possibly synthetic/generated"
    })

metadata_df = pd.DataFrame(metadata_rows)

metadata_output_path = metadata_dir / "jithin_cs_resume_metadata.csv"
metadata_df.to_csv(metadata_output_path, index=False)

print("Done.")
print(f"CS resume records selected: {len(df)}")
print(f"JSON records saved to: {json_output_dir}")
print(f"Metadata saved to: {metadata_output_path}")
