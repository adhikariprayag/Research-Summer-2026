# ATS/NER Resume Research Toolkit

This repository supports the research project **Measuring and Mitigating NER Accuracy Degradation in Creative Resumes**. It evaluates resume information extraction, GPT-based skill recovery, and a filtered hybrid method across standard and creative resumes.

## Current research status

- The 20-resume layout study is complete: 10 standard and 10 creative resumes with 1,108 gold entities.
- The full 1,020-resume corpus was processed without failures.
- A stratified 50-resume sample was manually evaluated.
- The original prediction that creative resumes would perform worse was **not supported** in the 20-resume sample.
- The filtered hybrid is reported as the **best balance**, not as conclusively superior to GPT-only.

### Confirmed strict F1 results

| Evaluation | V4 baseline | GPT-only | Raw hybrid | Filtered hybrid |
|---|---:|---:|---:|---:|
| 50-resume combined sample | 0.276 | 0.773 | 0.754 | **0.782** |

The filtered-hybrid improvement over GPT-only was modest overall. Results should therefore be described cautiously.

## Main reproducibility scripts

| Script | Purpose |
|---|---|
| `run_gpt_skill_recovery_1020.py` | Recovers SKILL entities from the 1,020 private resume JSON records. It performs a dry run unless `--run-api` is supplied. |
| `create_hybrid_v4_gpt_predictions_1020.py` | Combines V4 and GPT predictions using normalized exact deduplication. |
| `create_filtered_hybrid_predictions_1020.py` | Learns the validated V4-only filtering rule from the 50 reviewed resumes and applies it to the full raw hybrid. |
| `build_resume_self_audit_rubric.py` | Builds the student-facing Resume Self-Audit Rubric. |
| `build_resume_optimization_guide.py` | Builds the student-facing Resume Optimization Guide. |

The three experiment scripts are designed to be placed in:

```text
ATS_NER_Resume_Research/05_gpt_reconstruction/scripts/
```

They calculate the project root relative to that location. If the scripts are stored somewhere else, update their path constants or supply the supported command-line path options.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows PowerShell
python -m pip install -r requirements.txt
```

Set the API key only in the terminal or an untracked `.env` file. Never place the key in Python code, reports, screenshots, or GitHub.

```bash
export OPENAI_API_KEY="your-key-here"      # macOS/Linux
$env:OPENAI_API_KEY="your-key-here"       # Windows PowerShell
```

## Safe execution order

Run commands from the project root.

### 1. Preflight the GPT recovery run

The default command does **not** call the API:

```bash
python 05_gpt_reconstruction/scripts/run_gpt_skill_recovery_1020.py
```

Test a single resume before any full paid run:

```bash
python 05_gpt_reconstruction/scripts/run_gpt_skill_recovery_1020.py \
  --run-api --resume-id JIT_CS_0001
```

The full corpus requires both safety flags:

```bash
python 05_gpt_reconstruction/scripts/run_gpt_skill_recovery_1020.py \
  --run-api --confirm-full-run
```

### 2. Validate and create the raw hybrid

```bash
python 05_gpt_reconstruction/scripts/create_hybrid_v4_gpt_predictions_1020.py \
  --check-only
python 05_gpt_reconstruction/scripts/create_hybrid_v4_gpt_predictions_1020.py
```

### 3. Validate and create the filtered hybrid

```bash
python 05_gpt_reconstruction/scripts/create_filtered_hybrid_predictions_1020.py \
  --check-only
python 05_gpt_reconstruction/scripts/create_filtered_hybrid_predictions_1020.py
```

Use `python <script> --help` to view every supported option, including fixture paths and expected-count overrides.

## Privacy and publication rules

- Do not commit resumes, extracted resume text, private JSON records, contact details, manual row-level reviews, API responses, or detailed per-resume predictions.
- Commit only code, prompts, schemas, documentation, de-identified examples, and privacy-safe aggregate summaries.
- Keep the three source datasets' license and access information in the report rather than redistributing their files.
- Review `git status` before every commit and confirm that no private folders or credentials are staged.

## Student-facing deliverables

The completed deliverables are:

- Resume Self-Audit Rubric
- Resume Optimization Guide
- Research symposium poster draft

The report and poster language will receive a final revision after advisor feedback.
