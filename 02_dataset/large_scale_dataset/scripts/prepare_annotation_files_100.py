import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

metadata_path = BASE_DIR / "02_dataset/large_scale_dataset/metadata/annotation_subset_100_metadata.csv"

json_base = BASE_DIR / "02_dataset/large_scale_dataset/json_records_private"

output_text_dir = BASE_DIR / "02_dataset/large_scale_dataset/annotation_subset_private/resume_texts_100"
output_template_dir = BASE_DIR / "02_dataset/large_scale_dataset/annotation_subset_private"

output_text_dir.mkdir(parents=True, exist_ok=True)
output_template_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(metadata_path)

source_to_folder = {
    "snehaanbhawal/resume-dataset": "snehaan_information_technology",
    "saugataroyarghya/resume-dataset": "saugata_cs_resumes",
}

missing = []

for _, row in df.iterrows():
    resume_id = row["resume_id"]
    source = row["source_dataset"]

    folder_name = source_to_folder.get(source)

    if folder_name is None:
        missing.append((resume_id, "Unknown source"))
        continue

    json_path = json_base / folder_name / f"{resume_id}.json"

    if not json_path.exists():
        missing.append((resume_id, str(json_path)))
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    resume_text = record.get("resume_text", "")

    txt_path = output_text_dir / f"{resume_id}.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Resume ID: {resume_id}\n")
        f.write(f"Source: {source}\n")
        f.write(f"Original Category: {row.get('original_category', '')}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        f.write(resume_text)

gold_template = pd.DataFrame(columns=[
    "resume_id",
    "entity_text",
    "entity_type",
    "notes"
])

gold_template_path = output_template_dir / "gold_standard_100_template.csv"
gold_template.to_csv(gold_template_path, index=False)

tracker = df[["resume_id", "source_dataset", "original_category", "word_count"]].copy()
tracker["annotation_status"] = "not_started"
tracker["annotator_notes"] = ""

tracker_path = output_template_dir / "annotation_tracker_100.csv"
tracker.to_csv(tracker_path, index=False)

print("Annotation files prepared.")
print(f"Text files saved to: {output_text_dir}")
print(f"Gold template saved to: {gold_template_path}")
print(f"Tracker saved to: {tracker_path}")
print(f"Total resumes prepared: {len(df) - len(missing)}")

if missing:
    print("\nMissing files:")
    for item in missing:
        print(item)
