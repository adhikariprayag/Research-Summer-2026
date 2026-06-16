# Week 3–4 Research Summary

## Project Title

**Measuring and Mitigating NER Accuracy Degradation in Creative Resumes**

## Main Goal for Week 3–4

The main objective of Week 3–4 was to collect and categorize resumes, create a manually annotated Gold Standard, process the resumes through the baseline ATS-style pipeline, and evaluate the accuracy of named entity recognition across standard and creative resume formats.

---

## 1. Resume Collection and Categorization

A total of 20 resumes were selected for the primary evaluation dataset.

### Standard Resumes

1. R006_kaggle
2. R007_kaggle
3. R008_kaggle
4. R010_kaggle
5. R011_kaggle
6. R012_kaggle
7. R015_kaggle
8. R018_kaggle
9. R019_kaggle
10. R034_kaggle

### Creative or Complex Resumes

1. R002_creative
2. R003_complex
3. R004_creative_text
4. R005_complex_text
5. R014_kaggle
6. R017_kaggle
7. R026_kaggle_creative
8. R027_creative
9. R028_creative
10. R029_creative

The final dataset therefore contained:

* 10 standard resumes
* 10 creative or complex resumes

The resumes were selected from the original pilot files and Kaggle datasets. The dataset included text-based, multi-column, visually creative, and image-based resumes.

---

## 2. Gold Standard Annotation

A manually annotated Gold Standard was created for all 20 resumes.

The annotation schema included the following entity types:

* PERSON
* ORG
* DATE
* GPE
* SKILL

Two annotation batches were created and later merged.

### Batch 1

The first batch contained 10 resumes and 532 annotated entities.

### Batch 2

The second batch contained 10 resumes and 576 annotated entities.

### Final Gold Standard

The two files were merged into:

```text
gold_standard_final_20.csv
```

The final Gold Standard contained:

* 20 resumes
* 1,108 annotated entities

This completed the proposal requirement of manually annotating 10 standard and 10 creative resumes.

---

## 3. Text Extraction

PyMuPDF was used to extract text from text-based resumes.

The first and second resume batches were processed successfully, and text files were saved for each resume.

The following eight Batch 2 resumes were successfully extracted using PyMuPDF:

* R007_kaggle
* R011_kaggle
* R015_kaggle
* R019_kaggle
* R027_creative
* R028_creative
* R029_creative
* R034_kaggle

All eight produced readable text.

However, R002_creative and R003_complex were image-based resumes. PyMuPDF could not extract meaningful text from them because the visible resume content was stored as images rather than as a readable PDF text layer.

---

## 4. OCR Processing

Tesseract OCR was added to the pipeline to process the two image-based resumes.

### R002_creative

OCR recovered:

* 1,945 characters
* 274 words

The output was usable, but several OCR errors occurred. Examples included:

* “Sep 2078” instead of “Sep 2018”
* “Nov 2027” instead of “Nov 2021”
* fragmented headings and names
* incorrect characters and decorative-symbol noise

### R003_complex

OCR recovered:

* 2,763 characters
* 410 words

The R003 OCR output was generally more readable than R002. It recovered the candidate name, organizations, employment dates, location information, and resume content, although some decorative and layout-related noise remained.

The OCR text was not manually corrected because the OCR errors represent realistic ATS extraction errors and must remain part of the experimental condition.

---

## 5. spaCy Baseline Processing

The extracted text was processed using spaCy with the `en_core_web_lg` model.

spaCy was used to detect:

* PERSON
* ORG
* DATE
* GPE

Separate CSV files were generated for each resume.

All text-based resumes and the two OCR-processed resumes were successfully processed.

The results showed that spaCy frequently detected valid entities, but it also created many false positives. Common problems included:

* technical terms incorrectly labeled as ORG
* resume section headings labeled as ORG
* page markers labeled as PERSON
* locations split into separate entities
* multiword entities partially extracted
* text from separate columns incorrectly joined

---

## 6. Custom Skill Matcher

A custom keyword-based matcher was used to identify SKILL entities.

The matcher detected technical skills such as:

* Python
* Java
* HTML
* CSS
* React
* C#
* Git
* JavaScript

However, the matcher was incomplete.

It missed many resume-relevant skills, particularly skills from nontechnical resumes, including:

* Copywriting
* Digital Marketing
* Editing
* Creative Writing
* Content Writing
* Blog Writing
* UI/UX Design
* Wireframing
* Photoshop
* Illustrator
* Figma
* Quality Assurance
* Software Testing

R019 produced zero detected skills because none of its Gold Standard skills were recognized by the current dictionary.

This showed that the current skill matcher is biased toward technical and computer-science resumes.

---

## 7. Combined Entity Outputs

For every resume, the spaCy entities and skill-matcher entities were combined into one output file.

Duplicate entities were removed using:

* resume ID
* entity text
* entity type

The combined outputs represented the full baseline ATS-style system used for evaluation.

---

## 8. Evaluation Method

The combined system outputs were compared with the manually annotated Gold Standard.

The evaluation used strict exact matching.

An entity was counted as correct only when both of the following matched:

* the normalized entity text
* the entity type

The evaluation calculated:

* Correct entities
* Missed entities
* Incorrect entities
* Precision
* Recall
* F1 score

Strict exact matching caused partial entities to be counted as errors. For example, if the Gold Standard contained “Columbus, Ohio,” but the system extracted “Columbus” and “Ohio” separately, neither system entity counted as a correct match.

---

## 9. OCR Resume Evaluation

### R002_creative

* Gold entities: 38
* System entities: 25
* Correct: 6
* Missed: 32
* Incorrect: 19
* Precision: 0.240
* Recall: 0.158
* F1: 0.190

The six correct entities were technical skills such as Python, Java, HTML, CSS, React, and C#.

The low score was caused by OCR errors, spaCy labeling errors, and missing skills in the custom dictionary.

### R003_complex

* Gold entities: 36
* System entities: 19
* Correct: 8
* Missed: 28
* Incorrect: 11
* Precision: 0.421
* Recall: 0.222
* F1: 0.291

Correct entities included Melissa Ware, Burton Agency, Facebook, Instagram, The Ohio State University, and employment durations.

Many writing and marketing skills were missed because they were absent from the skill dictionary.

### Overall OCR Results

* Correct: 14
* Missed: 60
* Incorrect: 30
* Precision: 0.318
* Recall: 0.189
* F1: 0.237

OCR allowed two previously unreadable resumes to be processed, but many Gold Standard entities remained undetected.

---

## 10. Batch 2 Evaluation

The eight text-based Batch 2 resumes produced the following overall results:

* Correct: 109
* Missed: 393
* Incorrect: 468
* Precision: 0.189
* Recall: 0.217
* F1: 0.202

R019 had the lowest performance:

* Precision: 0.050
* Recall: 0.040
* F1: 0.044

R028 and R029 performed better than most resumes in Batch 2:

### R028_creative

* Precision: 0.347
* Recall: 0.321
* F1: 0.333

### R029_creative

* Precision: 0.340
* Recall: 0.296
* F1: 0.317

---

## 11. Final 20-Resume Results

The final evaluation combined:

* Week 4 Batch 1 results
* Batch 2 text-based results
* OCR results for R002 and R003

A total of 20 resumes were included.

### Standard Resume Results

* Resumes: 10
* Gold entities: 622
* System entities: 698
* Correct: 147
* Missed: 475
* Incorrect: 551
* Precision: 0.211
* Recall: 0.236
* F1: 0.223

### Creative Resume Results

* Resumes: 10
* Gold entities: 486
* System entities: 579
* Correct: 175
* Missed: 311
* Incorrect: 404
* Precision: 0.302
* Recall: 0.360
* F1: 0.329

### Overall Results

* Resumes: 20
* Gold entities: 1,108
* System entities: 1,277
* Correct: 322
* Missed: 786
* Incorrect: 955
* Precision: 0.252
* Recall: 0.291
* F1: 0.270

---

## 12. Main Findings

The baseline system performed poorly overall, with an F1 score of 0.270.

The system missed 786 Gold Standard entities and generated 955 incorrect entities.

The results revealed three major sources of error:

1. Text extraction and OCR errors
2. spaCy entity recognition and classification errors
3. An incomplete and technically biased skill dictionary

The creative group unexpectedly achieved a higher F1 score than the standard group:

* Standard F1: 0.223
* Creative F1: 0.329

This result does not prove that creative resumes are easier to parse.

The difference may have been influenced by:

* resume length
* entity density
* annotation differences
* resume content and profession
* technical versus nontechnical skill vocabulary
* spaCy ORG false positives
* strict exact matching
* OCR errors
* differences in layout complexity within each category

The results therefore require deeper per-entity and per-resume analysis before making a final conclusion regarding accuracy degradation caused by creative layouts.

---

## 13. Files Created During Week 3–4

Important files created include:

```text
02_dataset/gold_standard_annotations/gold_standard_final_20.csv
```

```text
03_text_extraction/batch2_outputs/
```

```text
03_text_extraction/ocr_outputs/
```

```text
04_spacy_baseline/week4_spacy_outputs/
```

```text
04_spacy_baseline/week4_skill_outputs/
```

```text
04_spacy_baseline/week4_combined_outputs/
```

```text
04_spacy_baseline/batch2_spacy_outputs/
```

```text
04_spacy_baseline/batch2_skill_outputs/
```

```text
04_spacy_baseline/batch2_combined_outputs/
```

```text
04_spacy_baseline/ocr_spacy_outputs/
```

```text
04_spacy_baseline/ocr_skill_outputs/
```

```text
04_spacy_baseline/ocr_combined_outputs/
```

```text
07_results/final_20_evaluation/final_20_per_resume_results.csv
```

```text
07_results/final_20_evaluation/final_20_group_summary.csv
```

---

## 14. Week 3–4 Completion Status

The main Week 3–4 proposal requirements are complete:

* Resume collection for the 20-resume experiment
* Standard versus creative categorization
* Gold Standard annotation
* PyMuPDF text extraction
* OCR processing for image-based resumes
* spaCy baseline processing
* Custom skill matching
* Combined entity generation
* Exact-match evaluation
* Final standard-versus-creative comparison
* Documentation of findings

The larger advisor-requested collection of 1,000 or more resumes has not yet been completed. That is a separate dataset-scaling task and should not be reported as complete.

---

## 15. Next Phase

The next research phase will focus on deeper error analysis and baseline improvement.

The immediate next task is to calculate precision, recall, and F1 separately for:

* PERSON
* ORG
* DATE
* GPE
* SKILL

This will identify which entity types contribute most strongly to the baseline system’s errors before changes are made to the model or skill dictionary.
