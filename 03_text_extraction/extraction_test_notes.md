# Week 1–2 PDF Extraction Test Notes

## Test Result

I tested three resume PDFs using PyMuPDF.

### R001_standard.pdf

- Text characters: 2622 on page 1 and 2011 on page 2
- Word count: 345 on page 1 and 277 on page 2
- Image count: 0
- Result: Machine-readable text was successfully detected.

### R002_creative.pdf

- Text characters: 0
- Word count: 0
- Image count: 1
- Result: The resume appears to be stored as one image instead of machine-readable text.

### R003_complex.pdf

- Text characters: 0
- Word count: 0
- Image count: 1
- Result: The resume appears to be stored as one image instead of machine-readable text.

## Research Observation

The standard resume was readable by the extraction tool, but the creative and complex resumes were not. Both creative resumes appeared as image-based PDFs with no detectable text or words. This means the extraction failure happens before Named Entity Recognition begins.

This supports the research problem because ATS-style systems depend on machine-readable text before they can extract entities such as PERSON, ORG, DATE, GPE, and SKILL. If a resume is exported as an image-based PDF, important information may be invisible to the parser even if the resume looks readable to a human.
