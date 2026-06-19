# Week 5–6 Entity-Level Evaluation Observations

## Purpose of Week 5–6 Work

The purpose of the Week 5–6 stage was to move beyond only reporting one overall Precision, Recall, and F1 score for the full 20-resume dataset. The main goal was to analyze the baseline system at the entity level for PERSON, ORG, DATE, GPE, and SKILL. This helps identify which entity types caused the most errors and whether the unexpected result, where creative resumes scored higher than standard resumes, was caused by layout or by other dataset factors.

The baseline system being evaluated is:

PDF resume → text extraction/OCR → spaCy NER → custom skill matcher → combined entity output → strict exact-match evaluation against Gold Standard annotations.

## Files and Scripts Created

The main Week 5–6 script created was:

`04_spacy_baseline/evaluation_scripts/entity_level_evaluation.py`

This script used the detailed evaluation outputs from the OCR, Week 4, and Batch 2 evaluation stages and calculated metrics separately for each entity type.

The script generated the following output files:

* `07_results/week5_6_entity_analysis/per_entity_results/per_entity_overall.csv`
* `07_results/week5_6_entity_analysis/per_entity_results/per_entity_by_resume_type.csv`
* `07_results/week5_6_entity_analysis/per_entity_results/per_entity_per_resume.csv`
* `07_results/week5_6_entity_analysis/per_entity_results/final_20_entity_details.csv`

Temporary Terminal checks were also performed to inspect PERSON annotations, ORG false positives, and missed SKILL entities. These checks were exploratory and are summarized in this document.

## Overall Entity-Level Results

| Entity Type | Gold Count | System Count | Correct | Missed | Incorrect | Precision | Recall |    F1 |
| ----------- | ---------: | -----------: | ------: | -----: | --------: | --------: | -----: | ----: |
| PERSON      |         18 |           68 |      14 |      4 |        54 |     0.206 |  0.778 | 0.326 |
| ORG         |        111 |          788 |      56 |     55 |       732 |     0.071 |  0.505 | 0.125 |
| DATE        |        146 |          142 |      47 |     99 |        95 |     0.331 |  0.322 | 0.326 |
| GPE         |         45 |           72 |      20 |     25 |        52 |     0.278 |  0.444 | 0.342 |
| SKILL       |        788 |          207 |     185 |    603 |        22 |     0.894 |  0.235 | 0.372 |

## Main Entity-Level Findings

The weakest entity type was ORG. The system produced 788 ORG predictions, but only 56 were correct. This resulted in very low ORG precision of 0.071. This shows that spaCy frequently overpredicted organizations in resume text.

The SKILL entity had the opposite problem. SKILL precision was very high at 0.894, meaning that when the custom skill matcher found a skill, it was usually correct. However, SKILL recall was low at 0.235, meaning that many real skills in the Gold Standard were missed. This suggests that the skill dictionary was accurate but incomplete.

DATE and GPE had moderate performance. They were not as weak as ORG, but strict exact matching likely penalized partial matches or formatting differences. For example, a date written as “May 2023” in the Gold Standard may not match if the system extracted only “2023” or extracted a longer date range.

PERSON performance must be interpreted carefully because the standard Kaggle resumes appeared to be anonymized and did not contain applicant names. This makes PERSON unfair for direct comparison between standard and creative resume groups.

## Standard vs. Creative Results by Entity Type

| Entity Type | Creative F1 | Standard F1 | Creative - Standard |
| ----------- | ----------: | ----------: | ------------------: |
| DATE        |       0.294 |       0.373 |              -0.079 |
| GPE         |       0.286 |       0.407 |              -0.121 |
| ORG         |       0.188 |       0.082 |              +0.106 |
| PERSON      |       0.444 |       0.000 |              +0.444 |
| SKILL       |       0.456 |       0.313 |              +0.143 |

Creative resumes performed better mainly for ORG and SKILL. Standard resumes performed better for DATE and GPE. PERSON should not be used as a fair comparison because the standard resumes had zero Gold Standard PERSON entities.

After excluding PERSON, creative resumes still performed better overall:

| Resume Type | Gold Count | System Count | Correct | Missed | Incorrect | Precision | Recall |    F1 |
| ----------- | ---------: | -----------: | ------: | -----: | --------: | --------: | -----: | ----: |
| Creative    |        468 |          534 |     161 |    307 |       373 |     0.301 |  0.344 | 0.321 |
| Standard    |        622 |          675 |     147 |    475 |       528 |     0.218 |  0.236 | 0.227 |

This means the unexpected result was not caused only by the PERSON issue. Creative resumes still scored higher after removing PERSON from the comparison.

## PERSON Annotation Observation

An audit showed that the standard resumes had zero Gold Standard PERSON entities. After inspecting the extracted text, the standard Kaggle resumes appeared to begin with job titles such as “INFORMATION TECHNOLOGY TECHNICIAN I” or “INFORMATION TECHNOLOGY MANAGER” instead of applicant names. This suggests that many standard resumes were anonymized or title-based.

Because of this, the PERSON comparison is not balanced between resume types. The system made 23 PERSON predictions for standard resumes, but since the Gold Standard contained no PERSON entities for standard resumes, those predictions were counted as incorrect. This is a dataset limitation rather than a pure model failure.

## ORG Error Observation

ORG was the weakest entity type. Many incorrect ORG predictions were actually technical skills, tools, systems, or concepts.

Examples of incorrect ORG predictions from standard resumes included:

* active directory federation services
* adfs
* android
* azure
* bash
* css
* dhcp
* dns
* distributed file system
* domain name service

These are mostly technologies, technical concepts, or skills, not organizations.

Examples of incorrect ORG predictions from creative resumes included:

* computer science
* jquery
* ui
* ux
* unreal
* api
* aws
* bigquery

This shows that spaCy often misclassified technical resume vocabulary as organizations. This ORG overprediction strongly lowered precision, especially for the standard resume group.

## SKILL Error Observation

The custom skill matcher had high precision but low recall. This means that the system was usually correct when it identified a skill, but it missed many skills that were present in the Gold Standard.

Examples of missed skills from standard resumes included:

* active directory federation services
* adfs
* apple ios
* cisco asa
* data domain
* distributed file system
* exchange online
* ldap
* nmap
* solarwinds
* storage management

These are real IT infrastructure, networking, system administration, or security-related skills. Many were missed because they were not included in the current skill dictionary.

Examples of missed skills from creative resumes included:

* angular
* artificial intelligence
* computer architecture
* figma
* illustrator
* javascript
* jquery
* photoshop
* ui/ux design
* unity 3d
* unreal engine 4
* wireframes
* copywriting
* content writing

These include design, UI/UX, writing, and software-related skills. This shows that the skill dictionary needs broader coverage across IT, software engineering, UI/UX, design, and writing-related resume domains.

## Explanation of the Unexpected Creative Resume Result

The original research expectation was that creative resumes would perform worse because complex layouts can damage text extraction and reading order. However, in the current 20-resume baseline, creative resumes scored higher than standard resumes.

This should not be interpreted as proof that creative resumes are easier for ATS systems. The result appears to be influenced by dataset composition and entity distribution.

Possible explanations include:

1. The standard resumes contained many more Gold Standard skills than creative resumes, creating more opportunities for missed SKILL entities.
2. The standard IT resumes contained specialized technical vocabulary such as ADFS, LDAP, iSCSI, NTFS, and Cisco ASA, which the skill matcher did not fully cover.
3. spaCy produced many incorrect ORG predictions in standard resumes by labeling technical concepts as organizations.
4. Standard resumes appeared to be anonymized and contained no PERSON entities, making PERSON comparison unfair.
5. Strict exact matching penalized partial matches, spelling differences, OCR artifacts, and boundary errors.
6. The creative group may have contained more common or easily recognizable skills and organizations.

Therefore, the safer conclusion is that creative resumes performed better in this specific sample, but the difference is likely affected by dataset composition, vocabulary coverage, and annotation balance rather than layout alone.

## Current Week 5–6 Conclusion

The Week 5–6 entity-level evaluation showed that the baseline system’s main weaknesses are ORG overprediction and low SKILL recall. ORG precision was extremely low because spaCy often mislabeled technical terms as organizations. SKILL precision was high, but recall was low because the custom skill dictionary did not include many specialized IT, design, UI/UX, and writing skills.

Creative resumes scored higher than standard resumes in the current 20-resume sample, even after excluding PERSON. However, this result should be interpreted cautiously. The comparison is influenced by dataset composition, anonymization, skill vocabulary coverage, and differences in entity distribution. The result does not prove that creative layouts are better for ATS parsing.

The next step is to document these findings in the progress report and then decide whether to improve the baseline skill dictionary, perform more detailed error categorization, or begin GPT-based reconstruction experiments.
