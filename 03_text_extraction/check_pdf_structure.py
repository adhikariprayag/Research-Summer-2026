import fitz
from pathlib import Path

project_root = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

# New text-based resume folder
input_folder = project_root / "02_dataset" / "text_based_sample_resumes"

print("Checking PDFs in:", input_folder)

for pdf_path in input_folder.glob("*.pdf"):
    print("\n==============================")
    print("File:", pdf_path.name)

    doc = fitz.open(pdf_path)

    for page_number, page in enumerate(doc, start=1):
        images = page.get_images(full=True)
        drawings = page.get_drawings()
        text = page.get_text("text")
        words = page.get_text("words")

        print(f"Page {page_number}")
        print("Text characters:", len(text.strip()))
        print("Word count:", len(words))
        print("Image count:", len(images))
        print("Drawing/vector objects:", len(drawings))