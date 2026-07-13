# GPT Reconstruction by Resume Type Observations

## Purpose

This analysis compares GPT reconstruction performance on Standard resumes versus Creative/Complex resumes using the full 20-resume Gold Standard dataset.

## Results

| Resume Type | Gold | System | Correct | Missed | Incorrect | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 622 | 284 | 170 | 452 | 114 | 0.599 | 0.273 | 0.375 |
| Creative | 486 | 291 | 206 | 280 | 85 | 0.708 | 0.424 | 0.530 |
| Overall | 1108 | 575 | 376 | 732 | 199 | 0.654 | 0.339 | 0.447 |

## Main Finding

GPT reconstruction performed better on Creative/Complex resumes than on Standard resumes.

The Creative resume group achieved an F1-score of 0.530, while the Standard resume group achieved an F1-score of 0.375. This is a difference of +0.155 F1 in favor of the Creative group.

## Interpretation

This result suggests that GPT reconstruction may be especially useful as a mitigation method for resumes with fragmented or layout-disrupted text. Creative resumes often create noisy extracted text because of sidebars, columns, icons, or unusual reading order. GPT appears to reduce some of these issues by using contextual reasoning to identify likely resume entities.

However, this result should not be interpreted as evidence that creative resumes are generally better for ATS systems. The higher Creative score may be influenced by dataset composition, entity density, anonymization in standard Kaggle resumes, and the strict exact-match evaluation method.

## Standard Resume Observation

Standard resumes had more Gold Standard entities than Creative resumes.

- Standard Gold entities: 622
- Creative Gold entities: 486

However, GPT correctly recovered fewer entities from Standard resumes.

- Standard Correct: 170
- Creative Correct: 206

This caused Standard recall to remain low at 0.273. One likely explanation is that Standard resumes contained many specialized SKILL entities that GPT did not fully extract, especially if they appeared in dense technical lists.

## Creative Resume Observation

Creative resumes achieved stronger GPT performance.

- Precision: 0.708
- Recall: 0.424
- F1: 0.530

This indicates that GPT produced fewer incorrect entities and recovered more correct entities from Creative resumes than from Standard resumes in this benchmark.

## Caution

The evaluation uses strict exact matching. This means GPT may be counted wrong even when it extracts a semantically similar entity with slightly different boundaries or formatting. For example, a date range, organization name, or location may be partially correct but still counted as incorrect if it does not exactly match the Gold Standard.

## Safe Conclusion

GPT reconstruction improved overall extraction performance and showed especially strong performance on Creative/Complex resumes in this 20-resume benchmark. The result supports using GPT as a mitigation strategy for layout-related resume parsing failures, but GPT should not be treated as a complete replacement for structured parsing because it still missed many entities, especially SKILL and DATE entities.