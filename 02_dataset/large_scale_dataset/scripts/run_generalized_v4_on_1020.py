from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re
import time

import pandas as pd
import spacy


PROJECT_ROOT = Path(__file__).resolve().parents[3]

JSON_ROOT = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "json_records_private"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "metadata"
    / "combined_large_cs_resume_metadata.csv"
)

PRIVATE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
)

PRIVATE_ENTITY_OUTPUT = (
    PRIVATE_OUTPUT_DIR
    / "generalized_v4_system_entities_1020.csv"
)

PRIVATE_PER_RESUME_OUTPUT = (
    PRIVATE_OUTPUT_DIR
    / "generalized_v4_per_resume_counts_1020.csv"
)

SAFE_RESULTS_DIR = (
    PROJECT_ROOT
    / "07_results"
    / "large_scale_full_v4"
)

SAFE_ENTITY_SUMMARY = (
    SAFE_RESULTS_DIR
    / "generalized_v4_entity_counts_1020.csv"
)

SAFE_SOURCE_SUMMARY = (
    SAFE_RESULTS_DIR
    / "generalized_v4_source_summary_1020.csv"
)

SAFE_DATASET_SUMMARY = (
    SAFE_RESULTS_DIR
    / "generalized_v4_dataset_summary_1020.csv"
)

MODEL_NAME = "en_core_web_lg"

ALLOWED_SPACY_TYPES = {
    "PERSON",
    "ORG",
    "DATE",
    "GPE",
}

SKILL_TERMS = {
    "python",
    "java",
    "c",
    "c++",
    "c#",
    ".net",
    "javascript",
    "typescript",
    "html",
    "css",
    "react",
    "angular",
    "vue",
    "node.js",
    "sql",
    "mysql",
    "postgresql",
    "oracle",
    "mongodb",
    "database",
    "dbms",
    "rdbms",
    "linux",
    "unix",
    "windows",
    "windows server",
    "active directory",
    "network",
    "networking",
    "lan",
    "wan",
    "vpn",
    "ssl vpn",
    "router",
    "routing",
    "switches",
    "firewall",
    "cisco",
    "ccna",
    "vmware",
    "citrix",
    "mainframe",
    "as/400",
    "cloud",
    "aws",
    "amazon web services",
    "microsoft azure",
    "azure",
    "google cloud",
    "docker",
    "kubernetes",
    "jenkins",
    "git",
    "github",
    "gitlab",
    "devops",
    "terraform",
    "ansible",
    "hadoop",
    "spark",
    "hive",
    "hdfs",
    "mapreduce",
    "yarn",
    "big data",
    "data science",
    "data analysis",
    "data analytics",
    "data visualization",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "natural language processing",
    "nlp",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "tableau",
    "power bi",
    "powerbi",
    "sas",
    "r",
    "informatica",
    "talend",
    "snowflake",
    "amazon redshift",
    "sharepoint",
    "peoplesoft",
    "workday hcm",
    "visio",
    "excel",
    "powerpoint",
    "photoshop",
    "illustrator",
    "indesign",
    "figma",
    "ui/ux",
    "wireframes",
    "service now",
    "servicenow",
    "remedy",
    "putty",
    "securecrt",
    "vnc",
    "wireshark",
    "solarwinds",
    "toad",
    "sql developer",
    "technical support",
    "help desk",
    "troubleshooting",
    "cybersecurity",
    "network security",
    "information security",
    "information technology",
    "itil",
    "risk management",
    "quality assurance",
    "requirements gathering",
    "technical documentation",
    "backup",
    "recovery",
    "automation",
    "testing",
    "functional testing",
    "unit testing",
    "integration testing",
    "project management",
    "program management",
    "data management",
    "hardware",
    "software",
    "voip",
    "mpls",
    "wireless network",
    "server infrastructure",
}

TECHNICAL_RELABEL_TERMS = {
    "big data",
    "hadoop",
    "hive",
    "mapreduce",
    "spark",
    "java",
    "core java",
    "machine learning",
    "cloud",
    "hdfs",
    "yarn",
    "data science",
    "data structures",
    "dbms",
    "rdbms",
    "informatica",
    "talend",
    "amazon redshift",
    "microsoft azure",
    "data analysis",
    "data analytics",
    "business analysis",
    "sas",
    "powerbi",
    "tableau",
    "data visualization",
    "business analytics",
    "ml system design",
    "statistical analysis",
    "data pipeline design",
    "model training",
    "algorithm analysis",
    "active directory",
    "risk management framework",
    "cisco",
    "routing",
    "linux",
    "voip",
    "oracle",
    "peoplesoft",
    "sql",
    "vpn",
    "visio",
    "sharepoint",
    "mainframe",
    "citrix",
    "toad",
    "service now",
    "wireshark",
    "solarwinds",
    "vmware",
    "windows",
    "windows server",
    "red hat linux",
    "network design",
    "high performance computing",
    "information assurance",
    "information technology",
}

BAD_DATE_TERMS = {
    "daily",
    "weekly",
    "monthly",
    "quarterly",
    "annual",
    "annually",
    "monthly/quarterly",
    "workday",
}

BAD_GENERIC_TERMS = {
    "state",
    "success",
    "enterprise",
    "studies",
    "maintained",
    "troubleshot",
    "company name",
    "city",
    "city state",
    "city, state",
    "none",
    "nan",
    "n/a",
}

ORG_CUES = {
    "university",
    "college",
    "school",
    "institute",
    "foundation",
    "council",
    "commission",
    "corporation",
    "organization",
    "academy",
    "department",
    "agency",
    "association",
    "company",
    "technologies",
    "technology",
    "systems",
    "solutions",
    "services",
    "group",
    "bank",
    "laboratory",
    "labs",
}


def normalize(text: str) -> str:
    text = str(text).casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,;:|/-'\"")

    if text.startswith("the "):
        text = text[4:]

    text = re.sub(r"'s$", "", text)
    return text.strip()


def clean_entity_text(text: str) -> str:
    text = str(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" .,;:|/-'\"")
    text = re.sub(r"[’']s$", "", text)
    return text.strip()


def clean_resume_text(text: str) -> str:
    text = str(text)

    separator = "=" * 80

    if separator in text:
        text = text.split(separator, 1)[1]

    text = text.replace("\x00", " ")
    return text.strip()


def extract_skills(text: str) -> set[str]:
    found = set()
    lowered = text.casefold()

    for skill in SKILL_TERMS:
        pattern = (
            r"(?<![A-Za-z0-9+#.])"
            + re.escape(skill.casefold())
            + r"(?![A-Za-z0-9+#.])"
        )

        if re.search(pattern, lowered):
            found.add(skill)

    return found


def generalized_postprocess(
    resume_id: str,
    entity_text: str,
    entity_type: str,
    source: str,
) -> list[dict]:
    text = clean_entity_text(entity_text)
    norm = normalize(text)

    if not text or not norm:
        return []

    if norm in BAD_GENERIC_TERMS:
        return []

    if (
        entity_type in {"ORG", "PERSON", "GPE"}
        and norm in TECHNICAL_RELABEL_TERMS
    ):
        return [
            {
                "resume_id": resume_id,
                "entity_text": text,
                "entity_type": "SKILL",
                "source": f"{source}_relabel_to_skill",
            }
        ]

    if entity_type == "DATE":
        if norm in BAD_DATE_TERMS:
            return []

        if len(text) > 60:
            return []

        return [
            {
                "resume_id": resume_id,
                "entity_text": text,
                "entity_type": "DATE",
                "source": f"{source}_generalized_v4",
            }
        ]

    if entity_type == "PERSON":
        word_count = len(text.split())

        if word_count < 2 or word_count > 4:
            return []

        if any(char.isdigit() for char in text):
            return []

        return [
            {
                "resume_id": resume_id,
                "entity_text": text,
                "entity_type": "PERSON",
                "source": f"{source}_generalized_v4",
            }
        ]

    if entity_type == "ORG":
        if len(text) > 100:
            return []

        has_org_cue = any(cue in norm for cue in ORG_CUES)

        if not has_org_cue:
            return []

        return [
            {
                "resume_id": resume_id,
                "entity_text": text,
                "entity_type": "ORG",
                "source": f"{source}_generalized_v4",
            }
        ]

    if entity_type == "GPE":
        if len(text) > 60:
            return []

        if any(char.isdigit() for char in text):
            return []

        return [
            {
                "resume_id": resume_id,
                "entity_text": text,
                "entity_type": "GPE",
                "source": f"{source}_generalized_v4",
            }
        ]

    if entity_type == "SKILL":
        return [
            {
                "resume_id": resume_id,
                "entity_text": text,
                "entity_type": "SKILL",
                "source": source,
            }
        ]

    return []


def load_json_records() -> list[dict]:
    records = []

    for path in sorted(JSON_ROOT.rglob("*.json")):
        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            )
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid JSON: {path.name}: {exc}")
            continue

        if not isinstance(data, dict):
            print(f"Skipping non-dictionary JSON: {path.name}")
            continue

        resume_id = str(
            data.get("resume_id", path.stem)
        ).strip()

        resume_text = clean_resume_text(
            data.get("resume_text", "")
        )

        if not resume_id or not resume_text:
            print(f"Skipping empty record: {path.name}")
            continue

        records.append(
            {
                "resume_id": resume_id,
                "resume_text": resume_text,
                "json_path": str(path),
            }
        )

    return records


def build_safe_summaries(
    metadata: pd.DataFrame,
    entities: pd.DataFrame,
    per_resume: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    SAFE_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    entity_summary = (
        entities.groupby("entity_type")
        .size()
        .reset_index(name="prediction_count")
        .sort_values("entity_type")
    )

    entity_summary.to_csv(
        SAFE_ENTITY_SUMMARY,
        index=False,
    )

    source_counts = (
        per_resume.merge(
            metadata[
                [
                    "resume_id",
                    "source_dataset",
                    "file_type",
                ]
            ],
            on="resume_id",
            how="left",
        )
        .groupby(
            [
                "source_dataset",
                "file_type",
            ],
            dropna=False,
        )
        .agg(
            resume_count=("resume_id", "nunique"),
            total_predictions=("total_predictions", "sum"),
            average_predictions_per_resume=(
                "total_predictions",
                "mean",
            ),
            average_skill_predictions=(
                "skill_predictions",
                "mean",
            ),
        )
        .reset_index()
    )

    source_counts[
        "average_predictions_per_resume"
    ] = source_counts[
        "average_predictions_per_resume"
    ].round(2)

    source_counts[
        "average_skill_predictions"
    ] = source_counts[
        "average_skill_predictions"
    ].round(2)

    source_counts.to_csv(
        SAFE_SOURCE_SUMMARY,
        index=False,
    )

    dataset_summary = pd.DataFrame(
        [
            {
                "resumes_processed": (
                    per_resume["resume_id"].nunique()
                ),
                "total_predictions": len(entities),
                "unique_entity_types": (
                    entities["entity_type"].nunique()
                ),
                "average_predictions_per_resume": round(
                    per_resume[
                        "total_predictions"
                    ].mean(),
                    2,
                ),
                "median_predictions_per_resume": round(
                    per_resume[
                        "total_predictions"
                    ].median(),
                    2,
                ),
                "minimum_predictions_per_resume": int(
                    per_resume[
                        "total_predictions"
                    ].min()
                ),
                "maximum_predictions_per_resume": int(
                    per_resume[
                        "total_predictions"
                    ].max()
                ),
                "elapsed_seconds": round(
                    elapsed_seconds,
                    2,
                ),
            }
        ]
    )

    dataset_summary.to_csv(
        SAFE_DATASET_SUMMARY,
        index=False,
    )


def main() -> None:
    start_time = time.time()

    if not JSON_ROOT.exists():
        raise FileNotFoundError(
            f"JSON directory not found: {JSON_ROOT}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    PRIVATE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading spaCy model: {MODEL_NAME}")
    nlp = spacy.load(MODEL_NAME)

    metadata = pd.read_csv(METADATA_FILE)
    metadata["resume_id"] = (
        metadata["resume_id"]
        .astype(str)
        .str.strip()
    )

    records = load_json_records()

    print(f"JSON records loaded: {len(records)}")

    if len(records) != 1020:
        print(
            "Warning: expected 1,020 records, "
            f"but loaded {len(records)}."
        )

    system_rows = []
    processing_rows = []

    texts = [
        record["resume_text"]
        for record in records
    ]

    resume_ids = [
        record["resume_id"]
        for record in records
    ]

    print("Running spaCy pipeline...")

    docs = nlp.pipe(
        texts,
        batch_size=16,
    )

    for index, (
        resume_id,
        resume_text,
        doc,
    ) in enumerate(
        zip(
            resume_ids,
            texts,
            docs,
        ),
        start=1,
    ):
        raw_rows = []

        for ent in doc.ents:
            if ent.label_ in ALLOWED_SPACY_TYPES:
                raw_rows.append(
                    {
                        "resume_id": resume_id,
                        "entity_text": ent.text,
                        "entity_type": ent.label_,
                        "source": "spacy",
                    }
                )

        for skill in extract_skills(resume_text):
            raw_rows.append(
                {
                    "resume_id": resume_id,
                    "entity_text": skill,
                    "entity_type": "SKILL",
                    "source": "skill_dictionary",
                }
            )

        processed_rows = []

        for row in raw_rows:
            processed_rows.extend(
                generalized_postprocess(
                    resume_id=row["resume_id"],
                    entity_text=row["entity_text"],
                    entity_type=row["entity_type"],
                    source=row["source"],
                )
            )

        seen = set()
        unique_rows = []

        for row in processed_rows:
            norm = normalize(
                row["entity_text"]
            )

            key = (
                row["resume_id"],
                row["entity_type"],
                norm,
            )

            if not norm or key in seen:
                continue

            seen.add(key)

            row["norm"] = norm
            unique_rows.append(row)

        system_rows.extend(unique_rows)

        entity_counter = Counter(
            row["entity_type"]
            for row in unique_rows
        )

        processing_rows.append(
            {
                "resume_id": resume_id,
                "text_character_count": len(
                    resume_text
                ),
                "total_predictions": len(
                    unique_rows
                ),
                "person_predictions": (
                    entity_counter["PERSON"]
                ),
                "org_predictions": (
                    entity_counter["ORG"]
                ),
                "date_predictions": (
                    entity_counter["DATE"]
                ),
                "gpe_predictions": (
                    entity_counter["GPE"]
                ),
                "skill_predictions": (
                    entity_counter["SKILL"]
                ),
            }
        )

        if (
            index == 1
            or index % 50 == 0
            or index == len(records)
        ):
            print(
                f"Processed {index}/{len(records)} resumes"
            )

    entities = pd.DataFrame(
        system_rows,
        columns=[
            "resume_id",
            "entity_text",
            "entity_type",
            "source",
            "norm",
        ],
    )

    per_resume = pd.DataFrame(
        processing_rows
    )

    entities.to_csv(
        PRIVATE_ENTITY_OUTPUT,
        index=False,
    )

    per_resume.to_csv(
        PRIVATE_PER_RESUME_OUTPUT,
        index=False,
    )

    elapsed_seconds = (
        time.time() - start_time
    )

    build_safe_summaries(
        metadata=metadata,
        entities=entities,
        per_resume=per_resume,
        elapsed_seconds=elapsed_seconds,
    )

    print("\nGeneralized V4 full run complete.")
    print(
        f"Resumes processed: "
        f"{per_resume['resume_id'].nunique()}"
    )
    print(
        f"Total predictions: "
        f"{len(entities)}"
    )

    print("\nPredictions by entity type:")
    print(
        entities["entity_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nPrivate entity output:"
    )
    print(PRIVATE_ENTITY_OUTPUT)

    print(
        "\nPrivate per-resume output:"
    )
    print(PRIVATE_PER_RESUME_OUTPUT)

    print(
        "\nSafe aggregate results:"
    )
    print(SAFE_RESULTS_DIR)

    print(
        f"\nElapsed time: "
        f"{elapsed_seconds:.2f} seconds"
    )


if __name__ == "__main__":
    main()
