from pathlib import Path
import fitz


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "03_text_extraction"
    / "batch2_outputs"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TARGET_FILES = [
    "R007_kaggle.pdf",
    "R011_kaggle.pdf",
    "R015_kaggle.pdf",
    "R019_kaggle.pdf",
    "R027_creative.pdf",
    "R028_creative.pdf",
    "R029_creative.pdf",
    "R034_kaggle.pdf",
]


def find_pdf(filename: str):
    matches = list(PROJECT_ROOT.rglob(filename))

    if not matches:
        return None

    return matches[0]


def extract_text(pdf_path: Path) -> str:
    document = fitz.open(pdf_path)
    pages = []

    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text("text")

        pages.append(
            f"--- Page {page_number} ---\n"
            f"{page_text.strip()}"
        )

    document.close()

    return "\n\n".join(pages)


def main():
    print(f"Processing {len(TARGET_FILES)} Batch 2 resumes.\n")

    for filename in TARGET_FILES:
        pdf_path = find_pdf(filename)

        if pdf_path is None:
            print(f"Not found: {filename}\n")
            continue

        try:
            text = extract_text(pdf_path)

            resume_id = pdf_path.stem

            output_path = (
                OUTPUT_FOLDER
                / f"{resume_id}_pymupdf.txt"
            )

            output_path.write_text(
                text,
                encoding="utf-8",
            )

            word_count = len(text.split())

            print(f"Processed: {filename}")
            print(f"Characters: {len(text)}")
            print(f"Words: {word_count}")
            print(f"Saved: {output_path.name}\n")

        except Exception as error:
            print(f"Failed: {filename}")
            print(f"Reason: {error}\n")

    print("Batch 2 PyMuPDF extraction completed.")
    print(f"Outputs saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
    