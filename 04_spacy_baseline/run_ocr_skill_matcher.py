from pathlib import Path
import csv
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FOLDER = PROJECT_ROOT / "03_text_extraction" / "ocr_outputs"

OUTPUT_FOLDER = PROJECT_ROOT / "04_spacy_baseline" / "ocr_skill_outputs"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


SKILLS = [
    "Python",
    "Java",
    "JavaScript",
    "C",
    "C++",
    "C#",
    "R",
    "SQL",
    "T-SQL",
    "HTML",
    "HTML5",
    "CSS",
    "CSS3",
    "React",
    "Next.js",
    "Angular.js",
    "Flask",
    "Scala",
    "LightGBM",
    "scikit-learn",
    "pandas",
    "Docker",
    "Kubernetes",
    "Git",
    "GitHub",
    "GitLab",
    "GitLab CI",
    "GitLab CI/CD",
    "CI/CD",
    "AWS",
    "GCP",
    "Azure",
    "Linux",
    "Ubuntu",
    "CentOS",
    "macOS",
    "Mac OS",
    "Android",
    "iOS",
    "BigQuery",
    "Oracle",
    "MongoDB",
    "Kafka",
    "Flume",
    "Spark",
    "Hive",
    "Solr",
    "Jupyter Lab",
    "LaTeX",
    "Matlab",
    "LabVIEW",
    "PowerShell",
    "VBScript",
    "VB.NET",
    ".NET Framework",
    "Visual Studio",
    "Active Directory",
    "Microsoft Exchange",
    "Office 365",
    "Microsoft Office",
    "VMware",
    "VMware vSphere",
    "Veeam",
    "SCCM",
    "TCP/IP",
    "DNS",
    "DHCP",
    "VPN",
    "LAN",
    "WAN",
    "IIS",
    "SharePoint",
    "AutoCAD",
    "Adobe InDesign",
    "SketchUp",
    "Rhino",
    "Cadence",
    "SPICE",
    "NI Multisim",
    "SAS",
    "SPSS",
    "Machine Learning",
    "Data Mining",
    "Big Data",
    "DevOps",
    "OOP",
    "UI/UX",
    "Project Management",
    "Disaster Recovery",
    "Network Security",
    "Cyber Security",
    "Information Assurance",
]


def build_pattern(skill: str) -> re.Pattern:
    escaped = re.escape(skill)

    return re.compile(
        rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


SKILL_PATTERNS = {
    skill: build_pattern(skill) for skill in sorted(SKILLS, key=len, reverse=True)
}


def find_skills(text: str):
    matches = []

    for skill, pattern in SKILL_PATTERNS.items():
        for match in pattern.finditer(text):
            matches.append(
                {
                    "skill": skill,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    matches.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))

    accepted = []
    occupied_spans = []

    for match in matches:
        span = (match["start"], match["end"])

        overlaps = any(
            not (span[1] <= old_span[0] or span[0] >= old_span[1])
            for old_span in occupied_spans
        )

        if not overlaps:
            accepted.append(match)
            occupied_spans.append(span)

    unique_skills = []
    seen = set()

    for match in accepted:
        normalized = match["skill"].lower()

        if normalized not in seen:
            unique_skills.append(match["skill"])
            seen.add(normalized)

    return unique_skills


def save_skills(output_path: Path, resume_id: str, skills):
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "resume_id",
                "entity_text",
                "entity_type",
            ],
        )

        writer.writeheader()

        for skill in skills:
            writer.writerow(
                {
                    "resume_id": resume_id,
                    "entity_text": skill,
                    "entity_type": "SKILL",
                }
            )


def main():
    text_files = sorted(INPUT_FOLDER.glob("*.txt"))

    if not text_files:
        print(f"No text files found in: {INPUT_FOLDER}")
        return

    print(f"Found {len(text_files)} text files.\n")

    for text_file in text_files:
        try:
            text = text_file.read_text(encoding="utf-8")

            resume_id = text_file.stem.replace("_pymupdf", "")
            skills = find_skills(text)

            output_name = f"{resume_id}_skill_entities.csv"
            output_path = OUTPUT_FOLDER / output_name

            save_skills(output_path, resume_id, skills)

            print(f"Processed: {text_file.name}")
            print(f"Skills found: {len(skills)}")
            print(f"Saved: {output_name}\n")

        except Exception as error:
            print(f"Failed: {text_file.name}")
            print(f"Reason: {error}\n")

    print("Week 4 SKILL matching completed.")
    print(f"Outputs saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
