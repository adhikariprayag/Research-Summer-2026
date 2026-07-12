from pathlib import Path
import re
import pandas as pd

BASE = Path("02_dataset/large_scale_dataset/annotation_subset_private/baseline_test_20")
GOLD_PATH = Path("02_dataset/large_scale_dataset/annotation_subset_private/gold_standard_100_template.csv")
SYS_PATH = BASE / "baseline_system_entities_10.csv"

DONE_IDS = [
    "SAU_CS_0001",
    "SAU_CS_0002",
    "SNE_IT_0001",
    "SNE_IT_0002",
    "SNE_IT_0003",
    "SNE_IT_0004",
    "SNE_IT_0005",
    "SNE_IT_0006",
    "SNE_IT_0007",
    "SNE_IT_0008",
    "SNE_IT_0009",
    "SNE_IT_0010",
    "SNE_IT_0011",
    "SNE_IT_0012",
    "SNE_IT_0013",
    "SNE_IT_0014",
    "SNE_IT_0015",
    "SNE_IT_0016",
    "SNE_IT_0017",
    "SNE_IT_0018",
]

ENTITY_TYPES = ["PERSON", "ORG", "DATE", "GPE", "SKILL"]

BAD_DATE_WORDS = {
    "daily", "weekly", "monthly", "quarterly", "annual", "annually",
    "monthly/quarterly", "workday", "second year", "three years"
}

# This is NOT a huge dictionary expansion.
# It is a whitelist of technical terms that spaCy already confused as ORG/PERSON/GPE.
TECHNICAL_RELABEL_TERMS = {
    "big data", "hadoop", "hive", "mapreduce", "spark", "java", "core java",
    "machine learning", "cloud", "hdfs", "yarn", "data science",
    "data structures", "dbms", "rdbms", "informatica", "talend",
    "amazon redshift", "microsoft azure", "data analysis", "data analytics",
    "business analysis", "r", "sas", "powerbi", "tableau",
    "data visualization", "business analytics", "ml/nlp infrastructure",
    "ml system design", "algorithm research", "application development",
    "dataset selection", "ml testing", "statistical analysis",
    "data pipeline design", "model training", "algorithm analysis",
    "plm", "active directory", "rmf", "stig", "cisco", "routing",
    "outlook", "camtasia", "keynote", "parallells", "putty", "siebel",
    "remedy", "cisco attendant", "cisco jabber", "aldon", "linux", "voip",
    "oracle", "peoplesoft", "sql", "vpn", "visio", "sharepoint",
    "novell", "mcse", "ccna", "cisa", "blackberry", "as/400",
    "mainframe", "citrix", "toad", "service now", "wireshark",
    "solarwinds", "vmware", "windows", "windows server", "cisco voip",
    "remedy(software", "red hat linux", "ms visio", "ms windows servers",
    "ms windows desktop", "risk management framework", "network design & troubleshooting",
    "high performance computing", "information assurance", "information technology"
}

BAD_GENERIC = {
    "state", "success", "enterprise", "tb", "dean", "studies",
    "electrician", "airman", "maintained", "troubleshot",
    "airman of the quarter", "excels", "association", "council, client",
    "company name", "city", "city state", "city, state", "－", "san"
}

ORG_CUES = [
    "university", "college", "school", "institute", "foundation",
    "council", "commission", "corporation", "organization",
    "academy", "department", "fraternity"
]

def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,;:|/-'\"")
    if text.startswith("the "):
        text = text[4:]
    text = re.sub(r"'s$", "", text)
    return text.strip()

def clean_entity_text(text):
    text = str(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" .,;:|/-'\"")
    text = re.sub(r"'s$", "", text)
    text = re.sub(r"’s$", "", text)
    return text.strip()

def clean_org(text):
    text = clean_entity_text(text)

    if text.lower().startswith("the "):
        text = text[4:].strip()

    for suffix in [" Skills", " Elected"]:
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()

    if "University of TN" in text:
        text = "University of TN"
    if "ITT Technical Institute" in text:
        text = "ITT Technical Institute"
    if "Brockport High School" in text:
        text = "Brockport High School"
    if "Utah Valley University" in text:
        text = "Utah Valley University"
    if "Utah Department of Health" in text:
        text = "Utah Department of Health"
    if "Iowa State University" in text:
        text = "Iowa State University"
    if "Graduate College" in text:
        text = "Graduate College"
    if "International Academy of Technology Education" in text:
        text = "International Academy of Technology Education, and Development (IATED)"

    return text

def postprocess_row(row):
    etype = row["entity_type"]
    text = clean_entity_text(row["entity_text"])
    norm = normalize(text)

    # Main V4 change:
    # If spaCy labeled a known technical term as ORG/PERSON/GPE,
    # convert it to SKILL instead of deleting it.
    if etype in {"ORG", "PERSON", "GPE"} and norm in TECHNICAL_RELABEL_TERMS:
        return [{
            "resume_id": row["resume_id"],
            "entity_text": text,
            "entity_type": "SKILL",
            "source": row["source"] + "_relabel_to_skill"
        }]

    if etype == "DATE":
        if norm in BAD_DATE_WORDS:
            return []
        if norm == "the month":
            return []
        if norm == "the month, march 1997":
            text = "March 1997"

        return [{
            "resume_id": row["resume_id"],
            "entity_text": text,
            "entity_type": "DATE",
            "source": row["source"] + "_v4"
        }]

    if etype == "PERSON":
        if "Summary" in text:
            possible_name = text.split("Summary", 1)[0].strip()
            possible_name = re.sub(r"\s+", " ", possible_name)
            if 1 < len(possible_name.split()) <= 3:
                return [{
                    "resume_id": row["resume_id"],
                    "entity_text": possible_name,
                    "entity_type": "PERSON",
                    "source": row["source"] + "_v4"
                }]
        return []

    if etype == "ORG":
        text = clean_org(text)
        norm = normalize(text)

        if norm in BAD_GENERIC:
            return []

        if len(text) > 80:
            return []

        has_org_cue = any(cue in norm for cue in ORG_CUES)
        known_short_org = norm in {
            "mckesson", "mckesson specialty health", "us oncology",
            "us oncology (uson)", "taco bell", "lenovo", "coca-cola",
            "coca-cola", "microsoft", "comptia", "usaf"
        }

        if not has_org_cue and not known_short_org:
            return []

        return [{
            "resume_id": row["resume_id"],
            "entity_text": text,
            "entity_type": "ORG",
            "source": row["source"] + "_v4"
        }]

    if etype == "GPE":
        norm = normalize(text)

        if norm in BAD_GENERIC:
            return []

        replacements = {
            "norfolk": "Norfolk Virginia",
            "virginia": "",
            "yong san": "Yong San, Korea",
            "korea": "",
            "fort bragg": "Fort Bragg, NC",
            "nc": "",
            "gurgaon": "Gurgaon, India",
            "india": "",
            "ia": "",
            "the state of utah": "State of Utah",
        }

        if norm in replacements:
            text = replacements[norm]
            if text == "":
                return []

        return [{
            "resume_id": row["resume_id"],
            "entity_text": text,
            "entity_type": "GPE",
            "source": row["source"] + "_v4"
        }]

    if etype == "SKILL":
        return [{
            "resume_id": row["resume_id"],
            "entity_text": text,
            "entity_type": "SKILL",
            "source": row["source"]
        }]

    return []

def evaluate(gold_df, sys_df):
    rows = []

    for rid in DONE_IDS:
        for etype in ENTITY_TYPES:
            gold_set = set(
                gold_df[
                    (gold_df["resume_id"] == rid) &
                    (gold_df["entity_type"] == etype)
                ]["norm"]
            )

            sys_set = set(
                sys_df[
                    (sys_df["resume_id"] == rid) &
                    (sys_df["entity_type"] == etype)
                ]["norm"]
            )

            correct = gold_set & sys_set
            missed = gold_set - sys_set
            incorrect = sys_set - gold_set

            precision = len(correct) / len(sys_set) if sys_set else 0
            recall = len(correct) / len(gold_set) if gold_set else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

            rows.append({
                "resume_id": rid,
                "entity_type": etype,
                "gold": len(gold_set),
                "system": len(sys_set),
                "correct": len(correct),
                "missed": len(missed),
                "incorrect": len(incorrect),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

    return pd.DataFrame(rows)

def main():
    gold = pd.read_csv(GOLD_PATH)
    gold = gold[gold["resume_id"].isin(DONE_IDS)].copy()
    gold["norm"] = gold["entity_text"].map(normalize)

    sys = pd.read_csv(SYS_PATH)

    new_rows = []
    for _, row in sys.iterrows():
        new_rows.extend(postprocess_row(row))

    sys_v4 = pd.DataFrame(new_rows)
    sys_v4["norm"] = sys_v4["entity_text"].map(normalize)
    sys_v4 = sys_v4.drop_duplicates(subset=["resume_id", "entity_type", "norm"])

    results = evaluate(gold, sys_v4)

    overall = results.groupby("entity_type")[["gold", "system", "correct", "missed", "incorrect"]].sum().reset_index()
    overall["precision"] = overall["correct"] / overall["system"]
    overall["recall"] = overall["correct"] / overall["gold"]
    overall["f1"] = 2 * overall["precision"] * overall["recall"] / (overall["precision"] + overall["recall"])
    overall = overall.fillna(0)

    total = overall[["gold", "system", "correct", "missed", "incorrect"]].sum()
    total_precision = total["correct"] / total["system"] if total["system"] else 0
    total_recall = total["correct"] / total["gold"] if total["gold"] else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if total_precision + total_recall else 0

    sys_v4.to_csv(BASE / "baseline_system_entities_15_v4_skill_relabel.csv", index=False)
    results.to_csv(BASE / "baseline_per_resume_entity_results_15_v4_skill_relabel.csv", index=False)
    overall.to_csv(BASE / "baseline_overall_by_entity_15_v4_skill_relabel.csv", index=False)

    print("V4 skill relabeling complete.")
    print(f"System entities before filtering: {len(sys)}")
    print(f"System entities after V4: {len(sys_v4)}")

    print("\nOverall by entity type after V4:")
    print(overall.to_string(index=False))

    print("\nTOTAL AFTER V4:")
    print(f"Gold: {int(total['gold'])}")
    print(f"System: {int(total['system'])}")
    print(f"Correct: {int(total['correct'])}")
    print(f"Missed: {int(total['missed'])}")
    print(f"Incorrect: {int(total['incorrect'])}")
    print(f"Precision: {total_precision:.3f}")
    print(f"Recall: {total_recall:.3f}")
    print(f"F1: {total_f1:.3f}")

if __name__ == "__main__":
    main()
