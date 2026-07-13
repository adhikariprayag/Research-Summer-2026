# Hybrid V4 + GPT Skill Recovery on 20 Reviewed Resumes

## Purpose

This experiment evaluated whether combining V4 rule-based SKILL predictions with GPT-assisted SKILL recovery improves technical-skill extraction.

Predictions from both methods were merged and deduplicated by resume ID and normalized entity text. Evaluation used strict, case-insensitive exact matching against the same 20-resume Gold Standard.

## Hybrid Results

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

## Comparison

| Method | Correct | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| V4 SKILL | 222 | 0.583 | 0.162 | 0.253 |
| GPT Skill Recovery | 499 | 0.727 | 0.363 | 0.484 |
| Hybrid V4 + GPT | 590 | 0.645 | 0.429 | 0.516 |

## Interpretation

GPT-only extraction achieved the highest precision, while the hybrid achieved the highest recall and F1-score.

The hybrid recovered 91 more correct skills than GPT alone and 368 more than V4. Precision decreased compared with GPT-only extraction because merging the two systems also introduced additional incorrect predictions.

The overall F1 improvement shows that the two methods recover partly different sets of skills and are therefore complementary.

## Main Takeaway

The strongest overall SKILL extraction performance came from combining the V4 rule-based method with GPT-assisted recovery. This hybrid approach achieved an F1-score of 0.516, compared with 0.484 for GPT alone and 0.253 for V4 alone.
