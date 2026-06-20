# GPT Reconstruction Pilot Observations

## Purpose

The purpose of the GPT reconstruction pilot was to test whether GPT-4o-mini could improve entity extraction from creative, complex, and OCR-processed resume text. This pilot used the same entity schema as the baseline system:

* PERSON
* ORG
* DATE
* GPE
* SKILL

The GPT output was compared against the same Gold Standard annotations used for the spaCy + skill matcher baseline.

## Pilot Resume Set

Six resumes were selected for the GPT reconstruction pilot:

| Resume ID          | Reason for Selection                             |
| ------------------ | ------------------------------------------------ |
| R002_creative      | OCR-based creative resume                        |
| R003_complex       | OCR-based complex resume                         |
| R004_creative_text | Text-based creative resume                       |
| R005_complex_text  | Text-based complex resume                        |
| R027_creative      | Creative resume with long/complex extracted text |
| R028_creative      | Creative resume with many entities               |

This pilot set was chosen because these resumes represent difficult cases involving creative formatting, OCR noise, complex layout, or large numbers of entities.

## GPT Reconstruction Method

The extracted resume text was passed to GPT-4o-mini using a structured prompt. The model was instructed to extract resume entities using only the information present in the text and return JSON in the following format:

```json
{
  "resume_id": "RESUME_ID",
  "entities": [
    {
      "entity_text": "Example Entity",
      "entity_type": "ORG"
    }
  ]
}
```

The GPT output was converted into CSV format and evaluated against the Gold Standard using strict exact matching.

## Pilot Output Summary

The final GPT pilot produced 167 entities across six resumes.

| Resume ID          | GPT Entities Extracted |
| ------------------ | ---------------------: |
| R002_creative      |                     30 |
| R003_complex       |                     21 |
| R004_creative_text |                     28 |
| R005_complex_text  |                     21 |
| R027_creative      |                     41 |
| R028_creative      |                     26 |

Entity distribution:

| Entity Type | GPT Count |
| ----------- | --------: |
| DATE        |        32 |
| GPE         |        11 |
| ORG         |        22 |
| PERSON      |         6 |
| SKILL       |        96 |

## GPT vs Baseline Result

The GPT reconstruction pilot was compared against the spaCy + skill matcher baseline on the same six resumes.

| Condition              | Gold | System | Correct | Missed | Incorrect | Precision | Recall |    F1 |
| ---------------------- | ---: | -----: | ------: | -----: | --------: | --------: | -----: | ----: |
| spaCy + Skill Baseline |  286 |    334 |     101 |    185 |       233 |     0.302 |  0.353 | 0.326 |
| GPT Reconstruction     |  286 |    167 |     129 |    157 |        38 |     0.772 |  0.451 | 0.570 |

GPT reconstruction improved F1 from 0.326 to 0.570 on the six-resume pilot set. The largest improvement was in precision. The baseline produced 233 incorrect entities, while GPT produced only 38 incorrect entities. This suggests that GPT reconstruction is much better at avoiding noisy false positives.

## GPT Per-Entity Results

| Entity Type | Gold | System | Correct | Missed | Incorrect | Precision | Recall |    F1 |
| ----------- | ---: | -----: | ------: | -----: | --------: | --------: | -----: | ----: |
| PERSON      |   12 |      6 |       6 |      6 |         0 |     1.000 |  0.500 | 0.667 |
| ORG         |   41 |     22 |      21 |     20 |         1 |     0.955 |  0.512 | 0.667 |
| DATE        |   55 |     32 |      21 |     34 |        11 |     0.656 |  0.382 | 0.483 |
| GPE         |   15 |     11 |       8 |      7 |         3 |     0.727 |  0.533 | 0.615 |
| SKILL       |  163 |     96 |      73 |     90 |        23 |     0.760 |  0.448 | 0.564 |

## Main Findings

The GPT reconstruction pilot showed a clear improvement over the spaCy + skill matcher baseline on the same six resumes. GPT produced fewer total entities than the baseline, but a much higher percentage of its predictions were correct. This means GPT was more conservative and cleaner in its extraction.

The strongest improvement was in reducing incorrect predictions. The baseline produced many false positives, especially from spaCy ORG overprediction. GPT avoided many of these noisy predictions and produced much higher precision.

ORG performance improved substantially. In the baseline, ORG was the weakest entity type because spaCy often mislabeled technical tools and concepts as organizations. In the GPT pilot, ORG precision was 0.955 and ORG F1 was 0.667.

SKILL performance also improved compared with the baseline issue of low skill recall, but GPT still missed many skills. GPT found 73 out of 163 Gold Standard skills, giving SKILL recall of 0.448. This is better than the original baseline problem, but still shows that skill recovery remains difficult.

DATE remained challenging. GPT achieved DATE F1 of 0.483, but some date errors were caused by OCR corruption or formatting mismatch. For example, GPT preserved OCR-corrupted dates such as “Sep 2078 - May 2023” instead of automatically correcting them. This is important because GPT should not invent corrections unless the evidence is clear.

## R027 Output Issue and Fix

During the first GPT pilot run, R027_creative caused a JSON output failure. The model repeatedly output the date “2016” many times, which caused the JSON response to become too long and incomplete. This was fixed by strengthening the prompt to discourage duplicate entities and by updating the script to:

* skip already valid JSON outputs,
* limit output length using `max_output_tokens`,
* continue processing if one resume fails,
* rerun only failed or missing resumes.

After this fix, R027_creative processed successfully and R028_creative was also completed.

## Interpretation

The GPT pilot supports the idea that LLM-based reconstruction can improve resume entity extraction from creative and complex layouts. However, this is still a pilot result based on six resumes. The result should not yet be treated as the final project conclusion.

The safest interpretation is:

GPT reconstruction improved strict exact-match F1 from 0.326 to 0.570 on a six-resume creative/complex pilot set, mainly by reducing false positives and improving precision. However, GPT still missed many entities, especially SKILL and DATE entities, so further testing is needed before making final claims.

## Next Steps

The next step is to decide whether to expand GPT reconstruction from the six-resume pilot to the full 20-resume Gold Standard dataset or to first refine the prompt and evaluation method. Possible next actions include:

1. Run GPT reconstruction on all 20 resumes.
2. Compare GPT performance against the full baseline.
3. Analyze GPT missed entities and incorrect predictions.
4. Consider a relaxed matching evaluation for boundary and OCR-related errors.
5. Document GPT results in the Overleaf progress report.

