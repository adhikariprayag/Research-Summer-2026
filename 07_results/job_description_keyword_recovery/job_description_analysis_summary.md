# 50 Job-Description Technical-Skill Analysis

## Scope and method

- Fifty active U.S. job postings were checked on 2026-07-31.
- The sample contains 10 postings each in Software Engineering, Data/AI, Cybersecurity, Cloud/DevOps, and Systems/IT.
- The sample contains 45 employers, with no more than two postings from one employer.
- Only explicit technical skills in duties, requirements, and preferred qualifications were retained.
- Soft skills, degrees, years of experience, and generic employment language were excluded.
- Each normalized skill was counted at most once per posting, so counts are document frequencies rather than raw mention totals.
- The extraction produced 863 skill-posting incidences and 496 unique normalized technical terms (median 15.5 skills per posting).
- Three postings with broader or unstated experience scope were retained in the main 50 and excluded in a 47-posting sensitivity analysis.

## Most frequently requested technical skills (at least 7 of 50 postings)

| Rank | Skill | Postings | Percent of 50 | Role families |
|---:|---|---:|---:|---:|
| 1 | Python | 28 | 56.0% | 5 |
| 2 | AWS | 11 | 22.0% | 5 |
| 2 | CI/CD | 11 | 22.0% | 4 |
| 3 | Kubernetes | 9 | 18.0% | 2 |
| 3 | Linux | 9 | 18.0% | 5 |
| 3 | Networking | 9 | 18.0% | 3 |
| 4 | Azure | 8 | 16.0% | 5 |
| 4 | Bash | 8 | 16.0% | 3 |
| 4 | Git | 8 | 16.0% | 3 |
| 4 | Java | 8 | 16.0% | 3 |
| 4 | macOS | 8 | 16.0% | 3 |
| 4 | SQL | 8 | 16.0% | 3 |
| 5 | Cloud Infrastructure | 7 | 14.0% | 4 |
| 5 | Data Pipelines | 7 | 14.0% | 2 |
| 5 | Go | 7 | 14.0% | 3 |
| 5 | Incident Response | 7 | 14.0% | 3 |
| 5 | PowerShell | 7 | 14.0% | 3 |
| 5 | Windows | 7 | 14.0% | 3 |

## Leading skills within each role family

Ten skills are shown per role family; equal-frequency ties are ordered alphabetically.

- **Software Engineering:** Python (9/10), Java (5/10), JavaScript (5/10), AWS (4/10), CI/CD (4/10), Algorithms (3/10), C++ (3/10), Data Structures (3/10), Git (3/10), React (3/10)
- **Data and AI:** Python (6/10), Data Pipelines (5/10), Machine Learning (4/10), SQL (4/10), A/B Testing (3/10), Data Visualization (3/10), Git (3/10), Prompt Engineering (3/10), Statistical Analysis (3/10), AI Tools (2/10)
- **Cybersecurity:** Incident Response (5/10), Vulnerability Management (5/10), Security Automation (4/10), SIEM (4/10), AWS (3/10), Azure (3/10), Cloud Security (3/10), GCP (3/10), Access Control (2/10), AZ-900 (2/10)
- **Cloud and DevOps:** Kubernetes (7/10), Python (7/10), CI/CD (5/10), Networking (5/10), Docker (4/10), Go (4/10), Grafana (4/10), Prometheus (4/10), C++ (3/10), Cloud Infrastructure (3/10)
- **Systems and IT:** macOS (6/10), Windows (5/10), Bash (4/10), Google Workspace (4/10), Linux (4/10), Microsoft Intune (4/10), Okta (4/10), PowerShell (4/10), Python (4/10), CompTIA Network+ (3/10)

## Sensitivity check

- Python remained the most frequent skill after the three experience-scope flags were excluded: 27 of 47 postings (57.4%).
- 16 of the 18 skills appearing in at least seven full-sample postings still appeared in at least seven core-sample postings.
- Stable leading skills: Python, AWS, CI/CD, Kubernetes, Linux, Networking, Azure, Bash, Git, Java, macOS, SQL, Data Pipelines, Go, PowerShell, Windows.
- Use the 50-posting results as the primary descriptive analysis and the 47-posting results as a robustness check.

## Interpretation boundaries

- These frequencies describe this purposive, balanced sample; they do not estimate every U.S. CS vacancy.
- A posting count shows how many postings explicitly named a skill, not how many times the wording appeared.
- This analysis identifies industry-relevant keywords. It does not itself measure resume-extraction accuracy.
- Keyword recovery must be calculated separately against the original 20-resume gold annotations and system outputs, split into standard and creative layouts.
