import re
from pathlib import Path
import pandas as pd

SKILLS = [
    "Python",
    "Java",
    "JavaScript",
    "React",
    "HTML",
    "CSS",
    "SQL",
    "C++",
    "Docker",
    "AWS",
    "GCP",
    "Git",
    "GitHub",
    "Linux",
    "BigQuery",
    "Oracle",
    "Flask",
    "Scala",
    "Tailwind",
    "Bootstrap",
    "Figma",
    "Canva",
    "Netlify",
    "Vercel",
    "WordPress",
    "Adobe Photoshop",
    "Illustrator",
    "InDesign",
    "SASS",
    "Next.js" "GitLab",
    "GitLab CI",
    "C++",
    "R",
    "LaTeX",
    "LATEX",
    "Android",
    "macOS",
    "CentOS",
]

input_folder = Path("../03_text_extraction/text_based_outputs")
output_folder = Path("skill_matcher_outputs")
output_folder.mkdir(exist_ok=True)

for txt_file in input_folder.glob("*_pymupdf.txt"):
    text = txt_file.read_text(encoding="utf-8", errors="ignore")

    found_skills = []

    for skill in SKILLS:
        pattern = r"(?<![A-Za-z0-9+#.])" + re.escape(skill) + r"(?![A-Za-z0-9+#.])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found_skills.append(
                {
                    "resume_id": txt_file.stem.replace("_pymupdf", ""),
                    "skill": skill,
                    "entity_type": "SKILL",
                }
            )

    df = pd.DataFrame(found_skills)
    output_path = output_folder / f"{txt_file.stem}_skills.csv"
    df.to_csv(output_path, index=False)

print("Skill matching complete.")
