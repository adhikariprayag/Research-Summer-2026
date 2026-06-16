from pathlib import Path
import csv
import spacy

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FOLDER = PROJECT_ROOT / "03_text_extraction" / "week4_batch_outputs"

OUTPUT_FOLDER = PROJECT_ROOT / "04_spacy_baseline" / "week4_spacy_outputs"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


TARGET_LABELS = {"PERSON", "ORG", "DATE", "GPE"}


def load_spacy_model():
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        print("spaCy model en_core_web_lg is not installed.")
        print("Run:")
        print("python -m spacy download en_core_web_lg")
        raise


def process_resume(nlp, text_file: Path):
    text = text_file.read_text(encoding="utf-8")

    doc = nlp(text)

    entities = []

    for entity in doc.ents:
        if entity.label_ in TARGET_LABELS:
            entities.append(
                {
                    "resume_id": text_file.stem.replace("_pymupdf", ""),
                    "entity_text": entity.text.strip(),
                    "entity_type": entity.label_,
                }
            )

    return entities


def save_entities(output_path: Path, entities):
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
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


def main():
    text_files = sorted(INPUT_FOLDER.glob("*.txt"))

    if not text_files:
        print(f"No text files found in: {INPUT_FOLDER}")
        return

    print(f"Found {len(text_files)} text files.\n")

    nlp = load_spacy_model()

    for text_file in text_files:
        try:
            entities = process_resume(nlp, text_file)

            output_name = (
                f"{text_file.stem.replace('_pymupdf', '')}" "_spacy_entities.csv"
            )

            output_path = OUTPUT_FOLDER / output_name

            save_entities(output_path, entities)

            print(f"Processed: {text_file.name}")
            print(f"Entities found: {len(entities)}")
            print(f"Saved: {output_name}\n")

        except Exception as error:
            print(f"Failed: {text_file.name}")
            print(f"Reason: {error}\n")

    print("Week 4 spaCy processing completed.")
    print(f"Outputs saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
