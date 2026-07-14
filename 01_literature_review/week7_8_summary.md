# Week 7–8 Summary: GPT-Assisted Skill Recovery and Hybrid Evaluation

## Project Stage

Weeks 7–8 focused on improving technical SKILL extraction after the advisor redirected the project toward a larger CS/IT resume dataset and better precision on standard resume text.

The work used the current manually reviewed set of 20 resumes from the 100-resume annotation subset.

The 20-resume Gold Standard contained:

- 1,739 annotation rows
- 1,705 unique normalized entities
- 1,374 unique SKILL entities

## Large Dataset Context

A total of 1,020 CS/IT resumes were collected from three public datasets:

- 500 resumes from the Saugata dataset
- 400 resumes from the Jithin dataset
- 120 resumes from the Snehaan dataset

A 100-resume annotation subset was created:

- 50 Snehaan IT resumes
- 50 Saugata CS resumes

Twenty resumes were manually reviewed and used for the current strict evaluation.

## V1 Baseline Results

The original baseline used:

- spaCy for PERSON, ORG, DATE, and GPE
- a custom keyword matcher for SKILL
- strict case-insensitive exact matching

Overall results:

| Metric | Value |
|---|---:|
| Gold | 1,705 |
| System | 1,532 |
| Correct | 348 |
| Missed | 1,357 |
| Incorrect | 1,184 |
| Precision | 0.227 |
| Recall | 0.204 |
| F1 | 0.215 |

The baseline produced many false positives, especially from spaCy ORG predictions.

## V4 Precision Improvement

The V4 method added:

- filtering of noisy spaCy predictions
- relabeling of clear technical terms incorrectly classified as ORG, PERSON, or GPE
- stricter precision-focused rules

Overall V4 results:

| Metric | Value |
|---|---:|
| Gold | 1,705 |
| System | 616 |
| Correct | 373 |
| Missed | 1,332 |
| Incorrect | 243 |
| Precision | 0.606 |
| Recall | 0.219 |
| F1 | 0.321 |

Compared with V1:

- Precision improved from 0.227 to 0.606
- Incorrect predictions decreased from 1,184 to 243
- F1 improved from 0.215 to 0.321

The main remaining weakness was low SKILL recall.

## GPT Skill-Recovery Experiment

A new GPT prompt was created to extract only technical SKILL entities.

The prompt focused on:

- programming languages
- frameworks and libraries
- databases
- cloud platforms
- operating systems
- developer tools
- networking and cybersecurity technologies
- machine learning and data engineering tools
- multi-word technical phrases

GPT skill-recovery results:

| Metric | Value |
|---|---:|
| Gold SKILL entities | 1,374 |
| GPT predictions | 686 |
| Correct | 499 |
| Missed | 875 |
| Incorrect | 187 |
| Precision | 0.727 |
| Recall | 0.363 |
| F1 | 0.484 |

GPT recovered many more correct skills than the V4 SKILL method.

## V4 SKILL versus GPT SKILL

| Method | Correct | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| V4 SKILL | 222 | 0.583 | 0.162 | 0.253 |
| GPT Skill Recovery | 499 | 0.727 | 0.363 | 0.484 |

GPT improved:

- Correct skills by 277
- Precision by 0.144
- Recall by 0.201
- F1 by 0.231

## Hybrid V4 + GPT Evaluation

The hybrid method combined the SKILL predictions from V4 and GPT and removed duplicates.

Hybrid results:

| Metric | Value |
|---|---:|
| Gold | 1,374 |
| System | 915 |
| Correct | 590 |
| Missed | 784 |
| Incorrect | 325 |
| Precision | 0.645 |
| Recall | 0.429 |
| F1 | 0.516 |

Comparison:

| Method | Correct | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| V4 SKILL | 222 | 0.583 | 0.162 | 0.253 |
| GPT Skill Recovery | 499 | 0.727 | 0.363 | 0.484 |
| Hybrid V4 + GPT | 590 | 0.645 | 0.429 | 0.516 |

GPT achieved the highest precision, while the hybrid method achieved the highest recall and F1-score.

## Method Complementarity

The comparison showed:

| Category | Count |
|---|---:|
| Correct in both V4 and GPT | 131 |
| Correct only in V4 | 91 |
| Correct only in GPT | 368 |
| Missed by both | 784 |

This showed that the two methods recovered different useful skills.

GPT added most of the new correct predictions, while V4 still recovered 91 correct skills that GPT missed.

## Remaining Error Analysis

The 784 SKILL entities missed by both methods were grouped by phrase length:

| Phrase Length | Count | Percentage |
|---|---:|---:|
| 1 word | 163 | 20.8% |
| 2 words | 386 | 49.2% |
| 3 words | 143 | 18.2% |
| 4 or more words | 92 | 11.7% |

A total of 79.2% of the remaining missed skills contained two or more words.

This suggests that the main remaining challenge is identifying complete multi-word technical phrases and preserving their correct boundaries.

## Main Findings

The main findings from Weeks 7–8 were:

1. Rule-based V4 filtering greatly improved precision.
2. GPT recovered many technical skills missed by V4.
3. GPT-only extraction achieved the highest precision.
4. The hybrid V4 + GPT method achieved the highest recall and F1-score.
5. V4 and GPT were complementary because each method recovered skills missed by the other.
6. Most remaining missed skills were multi-word technical phrases.

## Dataset Limitation

The large 1,020-resume dataset is stored mainly as CSV or JSON text and does not preserve the original visual resume layout.

Therefore:

- the original PDF experiment remains the main evidence for creative versus standard layout comparison
- the 1,020-resume dataset should be used for large-scale extraction, precision improvement, and skill-recovery testing
- layout-related conclusions should not be drawn from the JSON or CSV dataset

## Week 7–8 Conclusion

The core GPT and skill-recovery experiments were completed.

The strongest SKILL extraction method was the hybrid V4 + GPT method, with:

- Precision: 0.645
- Recall: 0.429
- F1: 0.516

The project is now ready to move into the next phase:

- advisor confirmation of the full 1,020-resume experiment
- final comparison tables and charts
- report writing
- student-facing resume guidance
- symposium poster preparation
