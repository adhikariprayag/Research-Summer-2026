# PDF Extraction Evidence Table

| Resume ID         | Resume Type |            Text Characters |               Word Count | Image Count | Machine-Readable? | Interpretation                                                         |
| ----------------- | ----------- | -------------------------: | -----------------------: | ----------: | ----------------- | ---------------------------------------------------------------------- |
| R001_standard.pdf | Standard    | Page 1: 2622, Page 2: 2011 | Page 1: 345, Page 2: 277 |           0 | Yes               | Standard resume contains parser-readable text.                         |
| R002_creative.pdf | Creative    |                          0 |                        0 |           1 | No                | Creative resume appears as one image, so normal text extraction fails. |
| R003_complex.pdf  | Complex     |                          0 |                        0 |           1 | No                | Complex resume appears as one image, so normal text extraction fails.  |

## Key Finding

The standard resume was successfully read by the PDF extraction tool, while the creative and complex resumes were not. The creative and complex resumes contained one image and no detectable words. This shows that visually readable resumes may still fail machine-based extraction if they are exported as image-based PDFs.

## Why This Matters

Named Entity Recognition cannot work properly if the resume text is not extracted first. Therefore, extraction failure is a major ATS risk before the NER stage even begins.
