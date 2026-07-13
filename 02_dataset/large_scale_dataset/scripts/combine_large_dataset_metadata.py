import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")
metadata_dir = BASE_DIR / "02_dataset/large_scale_dataset/metadata"

files = [
    metadata_dir / "snehaan_information_technology_metadata.csv",
    metadata_dir / "jithin_cs_resume_metadata.csv",
    metadata_dir / "saugata_cs_resume_metadata.csv",
]

dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

output_path = metadata_dir / "combined_large_cs_resume_metadata.csv"
combined.to_csv(output_path, index=False)

print("Combined metadata created.")
print(f"Total records: {len(combined)}")
print("\nRecords by source:")
print(combined["source_dataset"].value_counts())

print("\nFile types:")
print(combined["file_type"].value_counts())

print("\nSaved to:")
print(output_path)
