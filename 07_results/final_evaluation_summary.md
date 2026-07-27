# Final Evaluation Summary

## 1. Main Research-System Results

The primary experiment compared GPT-only, the unfiltered
V4 + GPT hybrid, and the cross-validated filtered hybrid.

| dataset                         | system             |   predictions |   gold_skills |   correct |   incorrect |   partial |   strict_tp |   strict_fp |   strict_fn |   relaxed_tp |   relaxed_fp |   relaxed_fn |   strict_precision |   strict_recall |   strict_f1 |   relaxed_precision |   relaxed_recall |   relaxed_f1 |
|:--------------------------------|:-------------------|--------------:|--------------:|----------:|------------:|----------:|------------:|------------:|------------:|-------------:|-------------:|-------------:|-------------------:|----------------:|------------:|--------------------:|-----------------:|-------------:|
| Combined                        | GPT                |          1061 |          1592 |      1025 |          24 |        12 |        1025 |          36 |         567 |         1037 |           24 |          555 |           0.96607  |        0.643844 |    0.77271  |            0.97738  |         0.651382 |     0.781757 |
| Combined                        | Raw Hybrid         |          1338 |          1592 |      1104 |         211 |        23 |        1104 |         234 |         488 |         1127 |          211 |          465 |           0.825112 |        0.693467 |    0.753584 |            0.842302 |         0.707915 |     0.769283 |
| Combined                        | CV Filtered Hybrid |          1174 |          1592 |      1081 |          79 |        14 |        1081 |          93 |         511 |         1095 |           79 |          497 |           0.920784 |        0.67902  |    0.781634 |            0.932709 |         0.687814 |     0.791757 |
| jithinjagadeesh/resume-dataset  | GPT                |           205 |           274 |       198 |           1 |         6 |         198 |           7 |          76 |          204 |            1 |           70 |           0.965854 |        0.722628 |    0.826722 |            0.995122 |         0.744526 |     0.851775 |
| jithinjagadeesh/resume-dataset  | Raw Hybrid         |           239 |           274 |       204 |          28 |         7 |         204 |          35 |          70 |          211 |           28 |           63 |           0.853556 |        0.744526 |    0.795322 |            0.882845 |         0.770073 |     0.822612 |
| jithinjagadeesh/resume-dataset  | CV Filtered Hybrid |           220 |           274 |       203 |          11 |         6 |         203 |          17 |          71 |          209 |           11 |           65 |           0.922727 |        0.740876 |    0.821862 |            0.95     |         0.762774 |     0.846154 |
| saugataroyarghya/resume-dataset | GPT                |           439 |           577 |       427 |          12 |         0 |         427 |          12 |         150 |          427 |           12 |          150 |           0.972665 |        0.740035 |    0.840551 |            0.972665 |         0.740035 |     0.840551 |
| saugataroyarghya/resume-dataset | Raw Hybrid         |           532 |           577 |       444 |          85 |         3 |         444 |          88 |         133 |          447 |           85 |          130 |           0.834586 |        0.769497 |    0.800721 |            0.840226 |         0.774697 |     0.806132 |
| saugataroyarghya/resume-dataset | CV Filtered Hybrid |           475 |           577 |       438 |          36 |         1 |         438 |          37 |         139 |          439 |           36 |          138 |           0.922105 |        0.759099 |    0.8327   |            0.924211 |         0.760832 |     0.834601 |
| snehaanbhawal/resume-dataset    | GPT                |           417 |           741 |       400 |          11 |         6 |         400 |          17 |         341 |          406 |           11 |          335 |           0.959233 |        0.539811 |    0.690846 |            0.973621 |         0.547908 |     0.701209 |
| snehaanbhawal/resume-dataset    | Raw Hybrid         |           567 |           741 |       456 |          98 |        13 |         456 |         111 |         285 |          469 |           98 |          272 |           0.804233 |        0.615385 |    0.697248 |            0.82716  |         0.632928 |     0.717125 |
| snehaanbhawal/resume-dataset    | CV Filtered Hybrid |           479 |           741 |       440 |          32 |         7 |         440 |          39 |         301 |          447 |           32 |          294 |           0.91858  |        0.593792 |    0.721311 |            0.933194 |         0.603239 |     0.732787 |

### Combined strict results

- GPT-only: Precision 0.966, Recall 0.644, F1 0.773
- Raw V4 + GPT hybrid: Precision 0.825, Recall 0.693, F1 0.754
- Cross-validated filtered hybrid: Precision 0.921,
  Recall 0.679, F1 0.782

The filtered hybrid achieved the highest combined strict F1.
It retained most of the hybrid system's additional recall while
removing many unreliable V4-only predictions.

## 2. Bootstrap Comparison

| dataset                         |   filtered_minus_gpt_f1 |   difference_ci_lower |   difference_ci_upper |   probability_filtered_better |
|:--------------------------------|------------------------:|----------------------:|----------------------:|------------------------------:|
| jithinjagadeesh/resume-dataset  |             -0.00485999 |          -0.0185583   |            0.00750579 |                        0.2362 |
| saugataroyarghya/resume-dataset |             -0.00785156 |          -0.024879    |            0.00775091 |                        0.1682 |
| snehaanbhawal/resume-dataset    |              0.0304652  |           0.015798    |            0.0457083  |                        1      |
| Combined                        |              0.00892399 |          -0.000210336 |            0.0178725  |                        0.9725 |

The filtered hybrid improved combined strict F1 over GPT-only
by approximately 0.009. The overall 95% confidence interval
touched zero, meaning the combined improvement was modest and
not conclusively significant at the 95% level.

On the Snehaan dataset, the filtered hybrid improved strict F1
from 0.691 to 0.721. The bootstrap comparison showed a positive
confidence interval, indicating a clear and stable improvement
on this more difficult dataset.

## 3. Jobscan External Comparison Pilot

| group    |   valid_scans |   mean_match_score |   median_match_score |   minimum_match_score |   maximum_match_score |   visible_matched_skills |   visible_missing_skills |
|:---------|--------------:|-------------------:|---------------------:|----------------------:|----------------------:|-------------------------:|-------------------------:|
| Jithin   |             1 |             30     |                 30   |                    30 |                    30 |                        5 |                        6 |
| Snehaan  |             3 |             32.667 |                 27   |                    21 |                    50 |                        6 |                       27 |
| Combined |             4 |             32     |                 28.5 |                    21 |                    50 |                       11 |                       33 |

Four valid Jobscan comparisons were completed.

- Combined mean Match Score: 32.0
- Combined median Match Score: 28.5
- Score range: 21 to 50
- Visible matched skills: 11
- Visible missing skills: 33

Jobscan Match Score is not directly comparable with precision,
recall, or F1-score. Jobscan also evaluates contact information,
section headings, education, job-title matching, measurable
results, resume length, and alignment with a particular job
description.

Each resume was compared with a different role-specific job
description, and the free reports did not display every missing
hard or soft skill. Therefore, these results are treated as a
small descriptive external comparison rather than a statistical
benchmark against the research systems.

## 4. Overall Conclusion

GPT-only produced the highest precision, while the raw hybrid
produced the highest recall. The cross-validated filtered hybrid
provided the strongest overall balance, achieving the highest
combined strict F1 of 0.782.

The filtered hybrid was especially helpful on the Snehaan
dataset, which contained fuller and more difficult resumes.
The Jobscan pilot provided practical industry context but also
showed that commercial Match Scores measure several factors
outside skill extraction.

The cross-validated filtered V4 + GPT hybrid should therefore
be presented as the primary research method, with GPT-only as
the strongest precision-focused baseline and Jobscan as a
limited external industry comparison.
