import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")
metadata_dir = BASE_DIR / "02_dataset/large_scale_dataset/metadata"

snehaan = pd.read_csv(metadata_dir / "snehaan_information_technology_metadata.csv")
saugata = pd.read_csv(metadata_dir / "saugata_cs_resume_metadata.csv")

snehaan_subset = snehaan.head(50).copy()
saugata_subset = saugata.head(50).copy()

subset = pd.concat([snehaan_subset, saugata_subset], ignore_index=True)

subset["selected_for_annotation"] = "yes"
subset["annotation_status"] = "not_started"

output_path = metadata_dir / "annotation_subset_100_metadata.csv"
subset.to_csv(output_path, index=False)

print("Annotation subset created.")
print(f"Total selected: {len(subset)}")
print("\nBy source:")
print(subset["source_dataset"].value_counts())
print(f"\nSaved to: {output_path}")
