# Paper Notes: P001

## Paper Title

Layout-Aware Parsing Meets Efficient LLMs: A Unified, Scalable Framework for Resume Information Extraction and Evaluation

## Why I Selected This Paper

I selected this paper because it focuses on resume information extraction, layout-aware parsing, and LLM-based extraction. This connects closely to my project because I am studying whether creative and non-traditional resume layouts affect automated extraction and Named Entity Recognition accuracy.

This paper is especially useful because it treats resume parsing as both a document-layout problem and an NLP problem. That matches my research direction because my project first tests whether resume text can be extracted correctly and then evaluates how extraction quality affects downstream NER performance.

## Main Problem the Paper Addresses

The paper addresses the difficulty of extracting structured information from resumes at scale. Resumes can vary widely in layout, formatting, section order, and visual design, which makes it difficult for automated systems to convert them into clean structured data.

A major issue discussed in the paper is that multi-column and non-linear layouts can disrupt normal reading order. This matters because many information extraction systems expect text to appear in a logical sequence. When the resume layout breaks this sequence, important information may become fragmented or misread.

The paper also addresses the challenge of using large language models efficiently for resume extraction. While LLMs can understand context well, they may be expensive or slow for large-scale production systems. The paper therefore explores more efficient layout-aware and LLM-based extraction methods.

## Method Used in the Paper

The authors propose a multi-stage resume extraction framework. The framework first uses layout-aware parsing to understand the visual structure of the resume instead of treating the document as plain linear text only.

The system then applies an LLM-based extraction approach to identify resume fields more accurately. Instead of relying only on raw text order, the approach attempts to preserve the connection between layout structure and resume content.

The paper also includes an evaluation framework for comparing extracted fields against ground truth data. This is important because resume extraction is difficult to evaluate when entities appear in different orders, formats, or locations.

## Key Findings

- The paper reports that a meaningful portion of real-world resumes contain multi-column or non-linear layouts that can disrupt normal reading order and make standard text-only extraction less reliable.

- The authors show that layout-aware parsing can improve resume information extraction because it considers document structure instead of relying only on plain text.

- The paper suggests that smaller task-specific LLMs can perform competitively when paired with a structured extraction pipeline.

- The paper highlights that evaluation is difficult in resume extraction because entities may appear in different orders or formats. Alignment methods can help compare extracted fields more reliably.

## How This Connects to My Research

This paper connects closely to my research because it supports the idea that resume information extraction is not only an NLP problem but also a layout problem. My project focuses on whether creative or multi-column resume layouts reduce text extraction quality and NER accuracy compared with standard chronological resumes.

The paper supports my decision to separate layout-related parsing problems from later NER performance problems. Before spaCy or another NER model can identify entities, the resume must first be converted into usable machine-readable text.

The paper also supports the later GPT-4o part of my project because it uses LLM-based extraction methods to improve resume information extraction. While my project is smaller and student-focused, this paper helps show that layout-aware resume parsing is an active research problem in real hiring-related systems.

## Useful Quote or Idea

A useful idea from this paper is the index-based pointer mechanism. Instead of asking the LLM to rewrite long resume fields, the model can point to the original source line range where the information appears. This reduces the chance of hallucination and helps preserve the original text.

This idea may be useful for my project because I want GPT-4o to help reconstruct fragmented resume information without inventing content. A pointer-style or evidence-based extraction method could help make the LLM output more trustworthy.

## How I Will Use This Paper

Use for:

- **Background:** I will use this paper to explain why resume layout matters in automated information extraction and why multi-column or non-linear layouts can create parsing challenges.

- **Methods:** I will use this paper to support my decision to compare standard resumes with creative resumes and to document layout-related extraction failures before running NER.

- **Evaluation:** I may use the paper’s discussion of entity alignment to improve how I compare extracted entities against my Gold Standard annotations.

- **Significance:** I will use this paper to show that resume parsing is not only a student issue but also a real problem in larger automated hiring pipelines.

## Status

Read and summarized. Useful as a core source for layout-aware resume parsing, LLM extraction, and evaluation design.
