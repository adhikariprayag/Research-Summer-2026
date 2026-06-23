# Full 20-Resume GPT Reconstruction Observations

## Purpose

After completing the six-resume GPT reconstruction pilot, the experiment was expanded to the full 20-resume Gold Standard dataset. The purpose of this step was to test whether the GPT reconstruction improvement observed in the pilot would still hold across the complete benchmark.

The evaluation used the same entity schema as the baseline system:

* PERSON
* ORG
* DATE
* GPE
* SKILL

The GPT reconstruction output was compared against the same Gold Standard annotations used for the spaCy + skill matcher baseline.

## Full Dataset

The full GPT reconstruction experiment used all 20 resumes from the Gold Standard benchmark:

* 10 standard resumes
* 10 creative or complex resumes

The full run produced 20 JSON output files and one combined entity CSV file.

## GPT Output Summary

The full GPT reconstruction run produced 575 total entities.

| Entity Type | GPT Count |
| ----------- | --------: |
| DATE        |       120 |
| GPE         |        38 |
| ORG         |        67 |
| PERSON      |         8 |
| SKILL       |       342 |

This shows that GPT predicted fewer total entities than the baseline system. The baseline produced 1,277 system entities, while GPT produced 575 system entities.

## Full GPT vs Baseline Result

The full 20-resume GPT reconstruction result was compared against the original spaCy + skill matcher baseline on the same 20 resumes.

| Condition              | Gold | System | Correct | Missed | Incorrect | Precision | Recall |    F1 |
| ---------------------- | ---: | -----: | ------: | -----: | --------: | --------: | -----: | ----: |
| spaCy + Skill Baseline | 1108 |   1277 |     322 |    786 |       955 |     0.252 |  0.291 | 0.270 |
| GPT Reconstruction     | 1108 |    575 |     376 |    732 |       199 |     0.654 |  0.339 | 0.447 |

GPT reconstruction improved F1 from 0.270 to 0.447 on the full 20-resume dataset. The largest improvement was in precision, which increased from 0.252 to 0.654.

The baseline produced 955 incorrect entities, while GPT produced 199 incorrect entities. This shows that GPT reconstruction substantially reduced false positives.

## Per-Entity GPT Results

| Entity Type | Gold | System | Correct | Missed | Incorrect | Precision | Recall |    F1 |
| ----------- | ---: | -----: | ------: | -----: | --------: | --------: | -----: | ----: |
| PERSON      |   18 |      8 |       7 |     11 |         1 |     0.875 |  0.389 | 0.538 |
| ORG         |  111 |     67 |      56 |     55 |        11 |     0.836 |  0.505 | 0.629 |
| DATE        |  146 |    120 |      58 |     88 |        62 |     0.483 |  0.397 | 0.436 |
| GPE         |   45 |     38 |      23 |     22 |        15 |     0.605 |  0.511 | 0.554 |
| SKILL       |  788 |    342 |     232 |    556 |       110 |     0.678 |  0.294 | 0.411 |

## Main Findings

The full-dataset GPT reconstruction experiment confirms that GPT improved over the baseline on the complete 20-resume Gold Standard dataset. The improvement was not limited to the six-resume pilot.

The most important improvement was precision. GPT produced fewer total predictions than the baseline, but a much higher percentage of its predictions were correct. This suggests that GPT was more conservative and better at avoiding noisy false positives caused by fragmented extraction, OCR noise, and layout disruption.

ORG extraction improved strongly. In the baseline system, ORG was the weakest entity type because spaCy frequently mislabeled technical tools, concepts, and resume fragments as organizations. In the full GPT experiment, ORG precision was 0.836 and ORG F1 was 0.629.

GPE and PERSON also showed stronger precision under GPT reconstruction. However, PERSON should still be interpreted cautiously because some standard resumes appeared anonymized or did not contain applicant names.

SKILL remained the most difficult entity type because the Gold Standard contained 788 SKILL entities. GPT correctly found 232 skills but missed 556. This means GPT improved skill extraction compared with the baseline, but still did not recover many specialized technical, infrastructure, design, and tool-related skills.

DATE extraction also remained challenging. GPT achieved DATE F1 of 0.436, but missed 88 Gold Standard dates and produced 62 incorrect DATE predictions. This suggests that OCR corruption, date-format variation, and strict exact-match boundaries still affect GPT performance.

## Interpretation

The full 20-resume experiment supports the research hypothesis that GPT-based reconstruction can improve entity extraction from creative and complex resumes. The improvement is mainly caused by a reduction in false positives rather than a complete recovery of all missed entities.

The safest full-dataset conclusion is:

GPT reconstruction improved strict exact-match F1 from 0.270 to 0.447 on the full 20-resume Gold Standard dataset. The improvement was mainly due to higher precision and fewer incorrect predictions. However, GPT still missed many entities, especially SKILL and DATE entities, so GPT reconstruction should be treated as a mitigation strategy rather than a complete solution.

## Next Steps

The next steps are:

1. Update the Overleaf progress report with the full 20-resume GPT result.
2. Compare GPT performance by resume type, if needed.
3. Analyze GPT missed and incorrect entities.
4. Consider relaxed matching for partial boundary matches and OCR-related variations.
5. Decide whether to improve the SKILL dictionary or continue with GPT-based mitigation analysis.


