import pandas as pd
import ast
from pathlib import Path

BASE_DIR = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

input_path = BASE_DIR / "02_dataset/large_scale_dataset/raw_private/saugata_resume_dataset/resume_data.csv"

output_dir = BASE_DIR / "02_dataset/large_scale_dataset/annotation_subset_private"
output_path = output_dir / "draft_saugata_annotations_003_050.csv"

df = pd.read_csv(input_path)
df.columns = [col.replace("\ufeff", "") for col in df.columns]

include_keywords = [
    "software engineer", "software developer", "senior software engineer",
    "machine learning", "ml engineer", "ai engineer", "data engineer",
    "data science", "data scientist", "database administrator", "dba",
    "system administrator", "network support engineer", "network engineer",
    "devops", "cloud engineer", "ios engineer", "android developer",
    "mobile app developer", "full stack", "frontend", "backend",
    "python developer", "java developer", "cybersecurity", "it support",
    "executive -it", "executive - it", "information technology"
]

exclude_keywords = [
    "marketing", "trade marketing", "mechanical", "civil", "hr",
    "human resource", "business development", "vat", "audit", "finance",
    "accounting", "sales", "apparel", "administrative", "internal control",
    "compliance", "banking", "icc", "sevp", "dmd"
]

def text_or_empty(value):
    if pd.isna(value):
        return ""
    return str(value)

def is_cs_related(row):
    check_text = " ".join([
        text_or_empty(row.get("job_position_name", "")),
        text_or_empty(row.get("positions", "")),
        text_or_empty(row.get("educationaL_requirements", "")),
        text_or_empty(row.get("major_field_of_studies", "")),
    ]).lower()

    has_include = any(keyword in check_text for keyword in include_keywords)
    has_exclude = any(keyword in check_text for keyword in exclude_keywords)

    return has_include and not has_exclude

def parse_list_cell(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if text in ["", "nan", "None", "[None]", "['N/A']", "N/A"]:
        return []

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return [text]

    def flatten(x):
        result = []
        if isinstance(x, list):
            for item in x:
                result.extend(flatten(item))
        elif x is not None:
            result.append(str(x))
        return result

    items = flatten(parsed)

    clean_items = []
    for item in items:
        item = item.strip()
        if item and item not in ["N/A", "None", "nan"]:
            clean_items.append(item)

    return clean_items

def add_entities(rows, resume_id, items, entity_type, note="draft; review"):
    seen = set()

    for item in items:
        item = item.strip()

        if not item:
            continue

        key = (resume_id, item.lower(), entity_type)

        if key in seen:
            continue

        seen.add(key)

        rows.append({
            "resume_id": resume_id,
            "entity_text": item,
            "entity_type": entity_type,
            "notes": note
        })

cs_df = df[df.apply(is_cs_related, axis=1)].copy().head(50)

rows = []

# Skip first two because you already annotated SAU_CS_0001 and SAU_CS_0002 manually
for i, (_, row) in enumerate(cs_df.iloc[2:].iterrows(), start=3):
    resume_id = f"SAU_CS_{i:04d}"

    add_entities(rows, resume_id, parse_list_cell(row.get("educational_institution_name", "")), "ORG")
    add_entities(rows, resume_id, parse_list_cell(row.get("professional_company_names", "")), "ORG")

    add_entities(rows, resume_id, parse_list_cell(row.get("passing_years", "")), "DATE")
    add_entities(rows, resume_id, parse_list_cell(row.get("start_dates", "")), "DATE")
    add_entities(rows, resume_id, parse_list_cell(row.get("end_dates", "")), "DATE")
    add_entities(rows, resume_id, parse_list_cell(row.get("issue_dates", "")), "DATE")
    add_entities(rows, resume_id, parse_list_cell(row.get("expiry_dates", "")), "DATE")

    add_entities(rows, resume_id, parse_list_cell(row.get("locations", "")), "GPE")

    add_entities(rows, resume_id, parse_list_cell(row.get("skills", "")), "SKILL")
    add_entities(rows, resume_id, parse_list_cell(row.get("related_skils_in_job", "")), "SKILL")
    add_entities(rows, resume_id, parse_list_cell(row.get("certification_skills", "")), "SKILL")

draft_df = pd.DataFrame(rows)
draft_df.to_csv(output_path, index=False)

print("Draft annotation file created.")
print(f"Rows created: {len(draft_df)}")
print(f"Saved to: {output_path}")

print("\nEntity type counts:")
print(draft_df["entity_type"].value_counts())

print("\nResume counts preview:")
print(draft_df["resume_id"].value_counts().head())
