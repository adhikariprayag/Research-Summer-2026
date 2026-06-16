import fitz
from pdfminer.high_level import extract_text
from pathlib import Path

# Absolute project path
project_root = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

input_folder = project_root / "02_dataset" / "sample_resumes"
output_folder = project_root / "03_text_extraction" / "sample_outputs"
output_folder.mkdir(parents=True, exist_ok=True)

print("Looking for PDFs in:", input_folder)

pdf_files = list(input_folder.glob("*.pdf"))

print("Number of PDFs found:", len(pdf_files))

if not pdf_files:
    print("No PDF files found. Check your folder path.")
else:
    for pdf_path in pdf_files:
        print("\n==============================")
        print("Processing:", pdf_path.name)
        print("Full path:", pdf_path)

        resume_id = pdf_path.stem

        # PyMuPDF extraction
        try:
            doc = fitz.open(pdf_path)
            pymupdf_text = ""

            print("Number of pages:", len(doc))

            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                print(f"PyMuPDF page {page_number} character count:", len(page_text))
                pymupdf_text += page_text + "\n"

            pymupdf_output = output_folder / f"{resume_id}_pymupdf.txt"

            with open(pymupdf_output, "w", encoding="utf-8") as f:
                f.write(pymupdf_text)

            print("PyMuPDF output saved to:", pymupdf_output)
            print("Total PyMuPDF characters:", len(pymupdf_text.strip()))

        except Exception as e:
            print("PyMuPDF error:", e)

        # pdfminer extraction
        try:
            pdfminer_text = extract_text(str(pdf_path))

            pdfminer_output = output_folder / f"{resume_id}_pdfminer.txt"

            with open(pdfminer_output, "w", encoding="utf-8") as f:
                f.write(pdfminer_text)

            print("pdfminer output saved to:", pdfminer_output)
            print("Total pdfminer characters:", len(pdfminer_text.strip()))

        except Exception as e:
            print("pdfminer error:", e)