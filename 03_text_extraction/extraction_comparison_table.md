# Extraction Comparison Table

| Resume ID | Resume Type | PDF Source Type | Machine-Readable? | Better Tool | Extraction Quality | Main Problems | Research Use |
|---|---|---|---|---|---|---|---|
| R001_standard.pdf | Standard | Text-based PDF | Yes | PyMuPDF | Clean | Minor or no major extraction issues | Baseline standard resume |
| R004_creative_text.pdf | Creative two-column | Text-based PDF | Yes | PyMuPDF | Mostly clean but slightly imperfect | Icon artifacts, hyphenated line breaks, possible reading-order differences | Creative layout test case |
| R005_complex_text.pdf | Complex visual design | Text-based PDF | Yes | PyMuPDF | Usable but noisy | Icon artifacts, special symbols, flattened layout, hyphenated words | Complex layout test case |
| R002_creative.pdf | Creative PNG-converted | Image-based PDF | No | Neither | Failed extraction | 0 words, detected as one image | Pilot failure case only |
| R003_complex.pdf | Complex PNG-converted | Image-based PDF | No | Neither | Failed extraction | 0 words, detected as one image | Pilot failure case only |

## Main Week 2 Finding

The updated extraction test shows that creative resumes should be separated into two categories: image-based creative PDFs and text-based creative PDFs. Image-based PDFs can completely block text extraction, while text-based creative PDFs can still produce usable but imperfect text. For the main experiment, text-based PDFs will be used so that the study measures layout-related extraction and NER degradation rather than only image-based extraction failure.

## Extraction Tool Decision

PyMuPDF will be used as the default extraction tool for the main experiment because it produced cleaner output than pdfminer.six for both the creative two-column resume and the complex visual resume. pdfminer.six will remain useful as a comparison tool during early testing.