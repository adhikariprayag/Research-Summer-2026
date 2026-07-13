# Remaining Hybrid SKILL Errors on 20 Reviewed Resumes

## Purpose

This analysis examined the Gold Standard SKILL entities that were missed by both the V4 rule-based method and GPT-assisted skill recovery.

## Complementarity Results

| Category | Count |
|---|---:|
| Correct in both V4 and GPT | 131 |
| Correct only in V4 | 91 |
| Correct only in GPT | 368 |
| Missed by both | 784 |

These results show that the two systems recover partly different SKILL entities. GPT contributed most of the additional correct predictions, while V4 still recovered 91 correct skills that GPT missed.

## Missed Skills by Phrase Length

| Phrase Length | Count | Percentage |
|---|---:|---:|
| 1 word | 163 | 20.8% |
| 2 words | 386 | 49.2% |
| 3 words | 143 | 18.2% |
| 4+ words | 92 | 11.7% |

## Interpretation

Most of the remaining failures involved multi-word technical phrases.

A total of 621 out of 784 missed skills, or 79.2%, contained two or more words. Two-word phrases were the largest category, representing almost half of all skills missed by both systems.

This suggests that the main remaining problem is not only incomplete coverage of individual technologies. It is also difficulty identifying complete technical phrases and preserving their correct boundaries.

## Main Takeaway

The hybrid V4 + GPT method significantly improved SKILL recovery, but the remaining errors are dominated by multi-word skills. Future improvement should focus on controlled phrase recovery and boundary matching rather than simply adding more single-word keywords.
