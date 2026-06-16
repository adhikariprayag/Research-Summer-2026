# Week 1–2 Mini Conclusion

During Week 1–2, I focused on setting up the research environment and testing whether different resume formats could be converted into machine-readable text. I tested three sample resumes: one standard resume, one creative resume, and one complex resume. The standard resume was a normal text-based resume with section dividers, while the creative resume used a two-column layout. The complex resume included a person’s image, icons, two columns, different colors, and a more visually designed structure.

The standard resume was successfully read by the PDF extraction tools. PyMuPDF detected thousands of text characters and hundreds of words from the standard resume, which shows that this file had a normal machine-readable text layer. However, the creative and complex resumes produced zero text characters and zero words, even though I could visually read the resumes and copy text from them when opening the PDF in Chrome. This was surprising because the resumes appeared readable to me as a human, but the extraction tools interpreted them as image-based PDFs.

A key reason for this result may be that the creative and complex PDFs were converted from PNG files. Even if Chrome allows text selection or copying, PyMuPDF still detected each of those resumes as one image with no readable text blocks or words. This shows an important difference between human readability, browser-assisted text selection, and parser-readable text extraction.

This is an important early finding for my research because Named Entity Recognition cannot begin if the resume text cannot be extracted first. Before spaCy or any ATS-style system can identify entities such as PERSON, ORG, DATE, GPE, or SKILL, the resume must first provide machine-readable text. If a resume is stored or exported in a way that extraction tools see only an image, then the applicant’s qualifications may become invisible to automated systems.

This Week 1–2 test supports the main research problem of the project. Creative formatting does not only risk confusing NER models after extraction; in some cases, it may prevent text extraction entirely. For the next stage, I will need to distinguish between creative resumes that have machine-readable text and creative resumes that are image-based. This separation will make the later benchmarking more accurate because messy extraction and blank extraction are two different types of ATS failure.

## Advisor Feedback Update

After meeting with my advisor, I updated the Week 1 setup by collecting new creative and complex resume PDFs that were not converted from PNG files. I tested these new files with PyMuPDF, and both produced more than 0 words, confirming that they contain machine-readable text.

This changes how I will treat the earlier PNG-converted creative and complex resumes. Those files will remain as a pilot failure case showing that image-based PDFs can completely block text extraction. However, they will not be used as the main creative resume condition because the main experiment should measure layout-based extraction and NER degradation, not just image-based PDF failure.

Moving forward, the main experiment will use text-based PDFs so I can better test whether creative layouts affect reading order, text fragmentation, and downstream NER performance.