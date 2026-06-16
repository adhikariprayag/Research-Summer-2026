import fitz
from pathlib import Path

project_root = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")
input_folder = project_root / "02_dataset" / "sample_resumes"

pdf_files = list(input_folder.glob("*.pdf"))

for pdf_path in pdf_files:
    print("\n==============================")
    print("Testing:", pdf_path.name)

    doc = fitz.open(pdf_path)

    for page_number, page in enumerate(doc, start=1):
        print(f"\nPage {page_number}")

        text_default = page.get_text()
        text_text = page.get_text("text")
        text_blocks = page.get_text("blocks")
        text_words = page.get_text("words")
        text_dict = page.get_text("dict")

        print("Default text characters:", len(text_default.strip()))
        print("Text mode characters:", len(text_text.strip()))
        print("Number of blocks:", len(text_blocks))
        print("Number of words:", len(text_words))
        print("Dict blocks:", len(text_dict.get("blocks", [])))

        if text_words:
            print("First 20 words:")
            for word in text_words[:20]:
                print(word[4])
