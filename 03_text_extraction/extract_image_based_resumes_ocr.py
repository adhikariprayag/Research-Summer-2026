from pathlib import Path
from pdf2image import convert_from_path
import pytesseract


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FOLDER = (
    PROJECT_ROOT
    / "02_dataset"
    / "gold_standard_batch_2"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "03_text_extraction"
    / "ocr_outputs"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


IMAGE_BASED_FILES = [
    "R002_creative.pdf",
    "R003_complex.pdf",
]


def extract_pdf_with_ocr(pdf_path: Path) -> str:
    pages = convert_from_path(
        pdf_path,
        dpi=300,
    )

    extracted_pages = []

    for page_number, image in enumerate(pages, start=1):
        text = pytesseract.image_to_string(
            image,
            lang="eng",
        )

        extracted_pages.append(
            f"--- Page {page_number} ---\n{text.strip()}"
        )

    return "\n\n".join(extracted_pages)


def main() -> None:
    for file_name in IMAGE_BASED_FILES:
        pdf_path = INPUT_FOLDER / file_name

        if not pdf_path.exists():
            print(f"File not found: {pdf_path}")
            continue

        try:
            print(f"Processing: {file_name}")

            text = extract_pdf_with_ocr(pdf_path)

            output_name = (
                f"{pdf_path.stem}_ocr.txt"
            )

            output_path = OUTPUT_FOLDER / output_name

            output_path.write_text(
                text,
                encoding="utf-8",
            )

            print(f"Characters extracted: {len(text)}")
            print(f"Words extracted: {len(text.split())}")
            print(f"Saved: {output_name}\n")

        except Exception as error:
            print(f"Failed: {file_name}")
            print(f"Reason: {error}\n")

    print("OCR extraction completed.")
    print(f"Outputs saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()