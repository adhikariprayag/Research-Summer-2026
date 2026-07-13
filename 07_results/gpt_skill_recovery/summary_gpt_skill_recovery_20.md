# GPT-Assisted Skill Recovery on 20 Reviewed Resumes

## Purpose

This experiment evaluated whether a GPT-based skill-only extraction prompt could improve SKILL entity recovery on the current 20 manually reviewed CS/IT resumes.

The experiment used the same strict, case-insensitive exact-match evaluation method as the V4 baseline.

## Dataset

- Reviewed resumes: 20
- Unique Gold Standard SKILL entities: 1,374
- GPT SKILL predictions: 686

## GPT Skill-Recovery Results

| Metric | Value |
|---|---:|
| Gold | 1,374 |
| System | 686 |
| Correct | 499 |
| Missed | 875 |
| Incorrect | 187 |
| Precision | 0.727 |
| Recall | 0.363 |
| F1 | 0.484 |

## Comparison with V4 SKILL Results

| Metric | V4 | GPT Skill Recovery | Change |
|---|---:|---:|---:|
| Correct | 222 | 499 | +277 |
| Precision | 0.583 | 0.727 | +0.144 |
| Recall | 0.162 | 0.363 | +0.201 |
| F1 | 0.253 | 0.484 | +0.231 |

## Per-Resume Stability

The average per-resume results were:

| Metric | Average |
|---|---:|
| Precision | 0.746 |
| Recall | 0.407 |
| F1 | 0.502 |

The improvement was distributed across multiple resumes rather than being caused by only one unusually strong result.

## Interpretation

The skill-focused GPT method substantially improved both precision and recall compared with the V4 rule-based SKILL method.

GPT recovered 499 correct skills, compared with 222 correct skills from V4. Recall more than doubled, while precision also increased.

The remaining weakness is incomplete recovery of long and highly detailed technical skill inventories. Resumes with very large Gold Standard skill sets still showed lower recall.

## Main Takeaway

A focused GPT skill-recovery prompt provided a meaningful improvement over the rule-based V4 SKILL method. This supports using GPT as a controlled recovery layer for technical skills that are missed by a traditional keyword matcher.
