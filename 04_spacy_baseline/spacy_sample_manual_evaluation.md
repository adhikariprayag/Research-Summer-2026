
# spaCy Sample Manual Evaluation

## Purpose

This file manually evaluates the first spaCy NER outputs for the three sample resumes. The goal is to compare how spaCy performs on a standard resume, a creative two-column resume, and a complex visual resume.

| Resume ID | Layout Type | Correct Entities | Missed Entities | Incorrect / Questionable Entities | Main Issue |
|---|---|---:|---:|---:|---|
| R001_standard | Standard | Many | Some | Many | Clean layout, but skills mislabeled |
| R004_creative_text | Creative two-column | Some | Some | Many | Icons, artifacts, partial date extraction |
| R005_complex_text | Complex visual | Few | Many | Many | Noisy text and symbol artifacts |

## Initial Interpretation

The initial manual evaluation shows that spaCy can extract entities from all three text-based resumes, but extraction quality decreases as layout complexity increases. The standard resume produced the cleanest results, although many technical skills were mislabeled because spaCy does not have a resume-specific SKILL category. The creative two-column resume was still usable, but contact icons, layout structure, and date formatting caused additional errors. The complex visual resume produced the noisiest output, with many placeholder words, symbols, and broken text fragments being incorrectly labeled as entities. This supports the project’s early finding that text-based resumes may extract successfully but still create NER reliability problems when the layout becomes more creative or complex.