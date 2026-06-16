import fitz
from pdfminer.high_level import extract_text
from pathlib import Path

project_root = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

input_folder = project_root / "02_dataset" / "text_based_sample_resumes"
output_folder = project_root / "03_text_extraction" / "text_based_outputs"

output_folder.mkdir(parents=True, exist_ok=True)

print("Extracting PDFs from:", input_folder)

pdf_files = list(input_folder.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found.")
else:
    for pdf_path in pdf_files:
        resume_id = pdf_path.stem
        print("\n==============================")
        print("Processing:", pdf_path.name)

        # Method 1: PyMuPDF
        pymupdf_text = ""

        try:
            doc = fitz.open(pdf_path)

            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                pymupdf_text += f"\n--- Page {page_number} ---\n"
                pymupdf_text += page_text

            pymupdf_output = output_folder / f"{resume_id}_pymupdf.txt"

            with open(pymupdf_output, "w", encoding="utf-8") as f:
                f.write(pymupdf_text)

            print("PyMuPDF characters:", len(pymupdf_text.strip()))
            print("Saved:", pymupdf_output)

        except Exception as e:
            print("PyMuPDF failed:", e)

        # Method 2: pdfminer.six
        try:
            pdfminer_text = extract_text(str(pdf_path))

            pdfminer_output = output_folder / f"{resume_id}_pdfminer.txt"

            with open(pdfminer_output, "w", encoding="utf-8") as f:
                f.write(pdfminer_text)

            print("pdfminer characters:", len(pdfminer_text.strip()))
            print("Saved:", pdfminer_output)

        except Exception as e:
            print("pdfminer failed:", e)