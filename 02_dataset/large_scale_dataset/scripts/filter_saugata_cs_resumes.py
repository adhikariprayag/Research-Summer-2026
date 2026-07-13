import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

input_path = BASE_DIR / "02_dataset/large_scale_dataset/raw_private/saugata_resume_dataset/resume_data.csv"

json_output_dir = BASE_DIR / "02_dataset/large_scale_dataset/json_records_private/saugata_cs_resumes"
metadata_dir = BASE_DIR / "02_dataset/large_scale_dataset/metadata"

json_output_dir.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

# Fix weird column name if it has hidden BOM character
df.columns = [col.replace("\ufeff", "") for col in df.columns]

cs_keywords = [
    "software", "developer", "data", "machine learning", "ai engineer",
    "artificial intelligence", "database", "dba", "system administrator",
    "network", "devops", "cloud", "ios", "android", "it", "computer science",
    "python", "java", "react", "sql", "aws", "azure", "cybersecurity"
]

def text_or_empty(value):
    if pd.isna(value):
        return ""
    return str(value)

def is_cs_related(row):
    check_text = " ".join([
        text_or_empty(row.get("job_position_name", "")),
        text_or_empty(row.get("positions", "")),
        text_or_empty(row.get("skills", "")),
        text_or_empty(row.get("skills_required", "")),
        text_or_empty(row.get("major_field_of_studies", "")),
        text_or_empty(row.get("educationaL_requirements", "")),
    ]).lower()

    return any(keyword in check_text for keyword in cs_keywords)

cs_df = df[df.apply(is_cs_related, axis=1)].copy()

# Use only first 500 for now
cs_df = cs_df.head(500)

metadata_rows = []

for i, row in cs_df.reset_index(drop=True).iterrows():
    resume_id = f"SAU_CS_{i+1:04d}"

    # Build resume text only from resume-side fields, not job posting requirements
    resume_parts = [
        text_or_empty(row.get("career_objective", "")),
        text_or_empty(row.get("skills", "")),
        text_or_empty(row.get("educational_institution_name", "")),
        text_or_empty(row.get("degree_names", "")),
        text_or_empty(row.get("passing_years", "")),
        text_or_empty(row.get("major_field_of_studies", "")),
        text_or_empty(row.get("professional_company_names", "")),
        text_or_empty(row.get("start_dates", "")),
        text_or_empty(row.get("end_dates", "")),
        text_or_empty(row.get("related_skils_in_job", "")),
        text_or_empty(row.get("positions", "")),
        text_or_empty(row.get("locations", "")),
        text_or_empty(row.get("responsibilities", "")),
        text_or_empty(row.get("certification_providers", "")),
        text_or_empty(row.get("certification_skills", "")),
        text_or_empty(row.get("languages", "")),
    ]

    resume_text = "\n".join([part for part in resume_parts if part.strip()])

    job_position = text_or_empty(row.get("job_position_name", ""))

    json_record = {
        "resume_id": resume_id,
        "source_dataset": "saugataroyarghya/resume-dataset",
        "original_category": job_position,
        "record_type": "structured_resume_text",
        "file_type": "csv_structured",
        "resume_text": resume_text
    }

    output_path = json_output_dir / f"{resume_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_record, f, indent=2)

    metadata_rows.append({
        "resume_id": resume_id,
        "source_dataset": "saugataroyarghya/resume-dataset",
        "original_category": job_position,
        "file_type": "csv_structured",
        "is_cs_related": "yes",
        "selected_for_large_dataset": "yes",
        "selected_for_annotation": "no",
        "word_count": len(resume_text.split()),
        "notes": "Filtered as CS-related using job/skill/position keywords"
    })

metadata_df = pd.DataFrame(metadata_rows)

metadata_output_path = metadata_dir / "saugata_cs_resume_metadata.csv"
metadata_df.to_csv(metadata_output_path, index=False)

print("Done.")
print(f"CS-like rows found before limit: {len(df[df.apply(is_cs_related, axis=1)])}")
print(f"CS resume records saved: {len(cs_df)}")
print(f"JSON records saved to: {json_output_dir}")
print(f"Metadata saved to: {metadata_output_path}")
