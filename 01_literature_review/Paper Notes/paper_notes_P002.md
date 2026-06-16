# Paper Notes: P002

## Paper Title

A Comparative Study of PDF Parsing Tools Across Diverse Document Categories

## Why I Selected This Paper

I selected this paper because it evaluates different PDF parsing tools across diverse document types and layouts. This directly connects to my research because my project depends on converting resume PDFs into machine-readable text before running Named Entity Recognition.

This paper is especially useful because my Week 1 test already showed that PDF extraction quality can vary depending on how the resume is formatted or exported. The paper helps support the idea that PDF parsing should be treated as a separate technical stage before NER.

## Main Problem the Paper Addresses

The paper addresses the lack of clear comparative research on how well modern PDF parsing tools perform across different document categories. PDF files can contain text, images, tables, columns, and layout structures that are difficult for parsers to interpret correctly.

A key problem is that PDF documents do not always store text in the same order that humans visually read it. This can create issues with word order, paragraph structure, and section continuity. These problems are especially important for documents with complex layouts, such as multi-column documents.

The paper also highlights that choosing the wrong PDF parser can affect downstream NLP tasks. If the parser produces incomplete, fragmented, or reordered text, then later tools may not receive reliable input.

## Method Used in the Paper

The authors compare multiple open-source PDF parsing tools across different document categories. The paper includes tools such as PyMuPDF and pdfminer.six, which are the same tools I tested in my Week 1 setup.

The study uses a benchmark dataset with annotated document layouts to compare parser outputs against ground truth text. The authors evaluate the quality of extracted text using metrics that examine character similarity, sequence matching, and local alignment.

This method is useful because it does not assume that all PDF extraction tools behave the same way. Instead, it shows that parser performance can vary depending on document structure and layout complexity.

## Key Findings

- PDF parsing quality depends heavily on document structure. Complex layouts can make text extraction less reliable, especially when documents contain multiple columns, tables, or visually separated sections.

- Word order preservation is a major challenge for PDF parsers. This is especially important for my project because creative resumes may visually appear organized but may convert into text in the wrong reading order.

- Paragraph and section fragmentation can occur when parsers misread layout boundaries. This can affect later NLP tasks because tools like spaCy depend on clean text input before they can identify entities accurately.

- The paper shows that PDF extraction is not a simple automatic step. It must be tested and documented before using the extracted text for NLP or information extraction tasks.

## How This Connects to My Research

This paper connects directly to my Week 1 extraction test because I used PDF parsing tools such as PyMuPDF and pdfminer.six to test whether resumes could be converted into machine-readable text. My early test showed that the standard resume produced readable text, while the creative and complex resumes failed because they appeared as image-based PDFs.

This paper helps explain why PDF extraction should be treated as a separate technical stage before NER. My research is not only about whether spaCy can identify entities correctly. It is also about whether the resume text is extracted cleanly enough for spaCy to work.

The paper also supports the broader design of my project because creative resumes may fail in more than one way. Some may produce no usable text at all, while others may produce text with broken reading order or fragmented sections. These are different types of extraction problems that should be documented separately.

## Useful Quote or Idea

A useful idea from this paper is that PDF parsing tools must preserve not only the text itself but also the correct reading order. This matters because a resume can look organized to a human but still produce text that is incomplete, reordered, or fragmented when extracted by a parser.

For my project, this means I should check extraction quality before running NER. If the input text is already broken, then poor NER results may be caused by extraction failure rather than the NER model alone.

## How I Will Use This Paper

Use for:

- **Background:** I will use this paper to explain that PDF parsing is a technical challenge and that document structure can affect whether text is extracted cleanly.

- **Methods:** I will use this paper to justify comparing PyMuPDF and pdfminer.six before choosing an extraction method for the main resume dataset.

- **Evaluation:** I may use this paper to support the idea of checking extraction quality before measuring NER accuracy.

- **Significance:** I will use this paper to show that parsing errors can affect downstream NLP systems, which is important for ATS-style resume screening.

## Important Research Note

spaCy is not a PDF parser. PyMuPDF and pdfminer.six are used to extract text from the PDF first. Then spaCy processes that extracted text for NER. Therefore, if the PDF parser produces incomplete, fragmented, or reordered text, downstream NLP tools like spaCy may fail to extract entities accurately.

## Status

Read and summarized. Useful as a core source for PDF parsing, extraction quality, reading order problems, and the need to evaluate text extraction before NER.
