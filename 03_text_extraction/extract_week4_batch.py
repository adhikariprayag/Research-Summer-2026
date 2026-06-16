from pathlib import Path
import fitz  # PyMuPDF

# Locate the main project folder automatically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FOLDER = PROJECT_ROOT / "02_dataset" / "gold_standard_batch"
OUTPUT_FOLDER = PROJECT_ROOT / "03_text_extraction" / "week4_batch_outputs"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF using PyMuPDF."""
    extracted_pages = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text")
            extracted_pages.append(f"--- Page {page_number} ---\n{page_text.strip()}")

    return "\n\n".join(extracted_pages)


def main() -> None:
    pdf_files = sorted(INPUT_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {INPUT_FOLDER}")
        return

    print(f"Found {len(pdf_files)} PDF files.\n")

    for pdf_path in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_path)

            output_name = f"{pdf_path.stem}_pymupdf.txt"
            output_path = OUTPUT_FOLDER / output_name

            output_path.write_text(text, encoding="utf-8")

            word_count = len(text.split())

            print(f"Processed: {pdf_path.name}")
            print(f"Output:    {output_name}")
            print(f"Words:     {word_count}\n")

        except Exception as error:
            print(f"Failed: {pdf_path.name}")
            print(f"Reason: {error}\n")

    print("Week 4 text extraction completed.")
    print(f"Outputs saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
