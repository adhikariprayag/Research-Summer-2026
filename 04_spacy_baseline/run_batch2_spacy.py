from pathlib import Path
import csv
import spacy

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FOLDER = PROJECT_ROOT / "03_text_extraction" / "batch2_outputs"

OUTPUT_FOLDER = PROJECT_ROOT / "04_spacy_baseline" / "batch2_spacy_outputs"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TARGET_LABELS = {"PERSON", "ORG", "DATE", "GPE"}


def main():
    text_files = sorted(INPUT_FOLDER.glob("*_pymupdf.txt"))

    if not text_files:
        print(f"No Batch 2 text files found in: {INPUT_FOLDER}")
        return

    nlp = spacy.load("en_core_web_lg")

    print(f"Found {len(text_files)} Batch 2 text files.\n")

    for text_file in text_files:
        text = text_file.read_text(encoding="utf-8")
        doc = nlp(text)

        resume_id = text_file.stem.replace(
            "_pymupdf",
            "",
        )

        entities = []

        for entity in doc.ents:
            if entity.label_ in TARGET_LABELS:
                cleaned_text = entity.text.strip()

                if cleaned_text:
                    entities.append(
                        {
                            "resume_id": resume_id,
                            "entity_text": cleaned_text,
                            "entity_type": entity.label_,
                        }
                    )

        output_file = OUTPUT_FOLDER / f"{resume_id}_spacy_entities.csv"

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "resume_id",
                    "entity_text",
                    "entity_type",
                ],
            )

            writer.writeheader()
            writer.writerows(entities)

        print(f"Processed: {text_file.name}")
        print(f"Entities found: {len(entities)}")
        print(f"Saved: {output_file.name}\n")

    print("Batch 2 spaCy processing completed.")
    print(f"Outputs saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
