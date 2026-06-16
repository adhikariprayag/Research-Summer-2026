During the PDF extraction test, the standard resume produced readable text output. However, the creative and complex resumes produced blank output in both PyMuPDF and pdfminer.six. This suggests that some visually designed resumes may be image-based or exported without a usable text layer. This is an important early finding because ATS-style systems depend on machine-readable text before NER can identify entities such as PERSON, ORG, DATE, GPE, and SKILL.

This result shows that formatting problems may happen even before Named Entity Recognition begins. In other words, if the text cannot be extracted, then the system cannot reliably identify the applicant's name, education, experience, dates, or technical skills.

## Advisor Feedback Update

After advisor feedback, I collected new creative and complex resume PDFs that were not converted from PNG. PyMuPDF detected more than 0 words in both new files, confirming that they are machine-readable and suitable for the main layout-based extraction test. The earlier PNG-converted resumes will be kept only as a pilot failure case, not as the main creative resume condition.
