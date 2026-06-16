import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from pathlib import Path

input_folder = Path("../02_dataset/sample_resumes")
output_folder = Path("sample_outputs")
output_folder.mkdir(exist_ok=True)

pdf_files = list(input_folder.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found in sample_resumes folder.")
else:
    for pdf_path in pdf_files:
        resume_id = pdf_path.stem
        print(f"\nProcessing: {pdf_path.name}")

        # -------------------------
        # Method 1: PyMuPDF
        # -------------------------
        try:
            doc = fitz.open(pdf_path)
            pymupdf_text = ""

            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                pymupdf_text += page_text + "\n"

                if page_text.strip():
                    print(f"PyMuPDF found text on page {page_num}.")
                else:
                    print(f"PyMuPDF found NO text on page {page_num}.")

            pymupdf_output = output_folder / f"{resume_id}_pymupdf.txt"

            with open(pymupdf_output, "w", encoding="utf-8") as f:
                f.write(pymupdf_text)

            if pymupdf_text.strip():
                print("PyMuPDF extraction saved successfully.")
            else:
                print("WARNING: PyMuPDF output is blank.")

        except Exception as e:
            print(f"PyMuPDF failed for {pdf_path.name}: {e}")

        # -------------------------
        # Method 2: pdfminer.six
        # -------------------------
        try:
            pdfminer_text = extract_text(str(pdf_path))

            pdfminer_output = output_folder / f"{resume_id}_pdfminer.txt"

            with open(pdfminer_output, "w", encoding="utf-8") as f:
                f.write(pdfminer_text)

            if pdfminer_text.strip():
                print("pdfminer extraction saved successfully.")
            else:
                print("WARNING: pdfminer output is blank.")

        except Exception as e:
            print(f"pdfminer failed for {pdf_path.name}: {e}")
