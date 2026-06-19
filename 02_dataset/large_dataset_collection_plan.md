# Large Resume Dataset Collection Plan

## Purpose

The large resume dataset will be used as a scale-up dataset for the ATS-NER resume research project. Unlike the 20-resume Gold Standard benchmark, the large dataset will not be fully manually annotated. Its purpose is to test extraction scalability, layout diversity, OCR need, and pipeline behavior across a larger number of resumes.

## Difference Between Datasets

| Dataset | Size | Purpose | Manual Annotation |
|---|---:|---|---|
| Gold Standard Benchmark | 20 resumes | Precision, Recall, F1 evaluation | Yes |
| Large Resume Corpus | 1000+ resumes | Extraction success, layout analysis, scalability | No, except possible small subset |

## Planned Uses

The large corpus will be used to measure:

- number of resumes successfully processed
- number of resumes with failed text extraction
- number of resumes requiring OCR
- average word count after extraction
- number of standard versus creative layouts
- number of image-based or low-text PDFs
- pipeline runtime and scalability
- common extraction failure patterns

## Safe Data Sources

Preferred sources:

- Kaggle resume datasets with clear public availability
- public GitHub resume repositories only if license allows use
- public sample resume/template datasets
- synthetic resume datasets
- university career-center sample resumes, if publicly available and allowed

Avoid:

- LinkedIn scraping
- Indeed/private resume databases
- anything behind a login
- personal resumes from websites without clear permission
- resumes containing sensitive personal data unless already public and licensed for research use

## Metadata File

The main metadata file will be:

`02_dataset/large_resume_corpus/metadata/large_resume_metadata.csv`

Planned columns:

| Column | Description |
|---|---|
| resume_id | New large dataset ID, such as L0001 |
| original_file_name | Original file name |
| source_name | Dataset or website name |
| source_url | Source link or dataset identifier |
| license | License if available |
| file_type | PDF, DOCX, TXT, etc. |
| layout_category | Standard, Creative, Complex, Unknown |
| text_based_or_image_based | Text-based, image-based, unknown |
| extraction_method | PyMuPDF, OCR, failed |
| character_count | Number of extracted characters |
| word_count | Number of extracted words |
| extraction_status | success, partial, failed |
| notes | Any important observations |

## GitHub Policy

Raw resumes and extracted full text should not be pushed to GitHub.

Do not commit:

- raw resume files
- extracted full resume text
- files containing personal resume content

Commit only:

- collection scripts
- metadata summaries
- aggregate statistics
- documentation
- analysis code

## Planned Folder Structure

```text
02_dataset/
└── large_resume_corpus/
    ├── raw_resumes/
    ├── extracted_text/
    ├── metadata/
    └── extraction_logs/
