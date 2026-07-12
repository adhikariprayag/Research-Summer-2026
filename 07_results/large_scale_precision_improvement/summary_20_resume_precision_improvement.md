# 20-Resume Precision Improvement Summary

## Purpose

This experiment evaluated whether rule-based post-processing can improve named entity recognition precision on a manually reviewed standard-resume subset.

The evaluation used 20 manually annotated resumes from the 100-resume annotation subset.

## Entity Types

The evaluation used five entity types:

- PERSON
- ORG
- DATE
- GPE
- SKILL

## Gold Standard

The Gold Standard CSV contained 1,739 annotation rows. During evaluation, duplicate repeated entities within the same resume and entity type were normalized and counted once, producing 1,705 unique gold entities.

## V1 Baseline Results

| Metric | Value |
|---|---:|
| Gold entities | 1,705 |
| System entities | 1,532 |
| Correct | 348 |
| Missed | 1,357 |
| Incorrect | 1,184 |
| Precision | 0.227 |
| Recall | 0.204 |
| F1 | 0.215 |

## V4 Improved Method Results

| Metric | Value |
|---|---:|
| Gold entities | 1,705 |
| System entities | 616 |
| Correct | 373 |
| Missed | 1,332 |
| Incorrect | 243 |
| Precision | 0.606 |
| Recall | 0.219 |
| F1 | 0.321 |

## Improvement from V1 to V4

| Metric | V1 | V4 | Change |
|---|---:|---:|---:|
| Precision | 0.227 | 0.606 | +0.379 |
| Recall | 0.204 | 0.219 | +0.015 |
| F1 | 0.215 | 0.321 | +0.106 |
| System entities | 1,532 | 616 | -916 |
| Incorrect predictions | 1,184 | 243 | -941 |
| Correct predictions | 348 | 373 | +25 |

## Interpretation

The V4 method substantially improved precision by reducing noisy false positive predictions from the baseline system. This directly supports the current research goal of improving standard-resume NER precision.

The major improvement came from filtering noisy named-entity predictions and relabeling clear technical terms that the baseline system incorrectly classified as ORG, PERSON, or GPE.

Recall remained low because many true SKILL entities were long, resume-specific technical phrases that were not captured by the starter skill matcher. This suggests that the next stage should focus on skill recovery rather than additional false-positive filtering.

## Main Takeaway

On the 20-resume manually reviewed subset, the improved post-processing method increased precision from 0.227 to 0.606 and reduced incorrect predictions from 1,184 to 243. This demonstrates that targeted rule-based filtering and technical-skill relabeling can significantly improve precision in ATS-style resume NER.
