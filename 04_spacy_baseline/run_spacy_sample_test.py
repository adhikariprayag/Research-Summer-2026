import spacy
from pathlib import Path
import csv

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

project_root = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

# Folder containing extracted text files
input_folder = project_root / "03_text_extraction" / "text_based_outputs"

# Output folder for NER results
output_folder = project_root / "04_spacy_baseline" / "sample_ner_outputs"
output_folder.mkdir(parents=True, exist_ok=True)

# We only want PyMuPDF outputs for now
text_files = list(input_folder.glob("*_pymupdf.txt"))

if not text_files:
    print("No PyMuPDF text files found.")
else:
    for text_file in text_files:
        print("\n==============================")
        print("Processing:", text_file.name)

        text = text_file.read_text(encoding="utf-8")

        doc = nlp(text)

        output_csv = output_folder / f"{text_file.stem}_spacy_entities.csv"

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Resume_File", "Entity_Text", "Entity_Label"])

            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "DATE", "GPE"]:
                    writer.writerow([text_file.name, ent.text, ent.label_])

        print("Saved:", output_csv)