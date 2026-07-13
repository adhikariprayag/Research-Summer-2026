from pathlib import Path
import re
import pandas as pd

BASE = Path("02_dataset/large_scale_dataset/annotation_subset_private/baseline_test_10")
GOLD_PATH = Path("02_dataset/large_scale_dataset/annotation_subset_private/gold_standard_100_template.csv")
SYS_PATH = BASE / "baseline_system_entities_10.csv"

DONE_IDS = [
    "SAU_CS_0001", "SAU_CS_0002",
    "SNE_IT_0001", "SNE_IT_0002", "SNE_IT_0003", "SNE_IT_0004",
    "SNE_IT_0005", "SNE_IT_0006", "SNE_IT_0007", "SNE_IT_0008"
]

ENTITY_TYPES = ["PERSON", "ORG", "DATE", "GPE", "SKILL"]

SKILL_LIKE = {
    "big data", "hadoop", "spark", "java", "core java", "data science",
    "data structures", "dbms", "rdbms", "informatica", "amazon redshift",
    "microsoft azure", "data analysis", "data analytics", "sas", "tableau",
    "data visualization", "business analytics", "nlp infrastructure",
    "plm", "project", "hardware", "rmf", "active directory",
    "network design & troubleshooting", "high performance computing",
    "information technology", "stig", "cisco", "routing", "outlook",
    "camtasia", "keynote", "parallells", "putty", "siebel", "remedy",
    "cisco attendant", "cisco jabber", "aldon", "linux", "voip",
    "oracle", "peoplesoft", "sql", "vpn", "visio", "sharepoint",
    "novell", "mcse", "ccna", "cisa", "blackberry", "as/400",
    "mainframe", "citrix", "toad", "service now", "wireshark",
    "solarwinds", "vmware", "windows", "windows server"
}

BAD_GENERIC = {
    "state", "success", "enterprise", "tb", "dean", "studies",
    "electrician", "airman", "maintained", "troubleshot", "jun 2016",
    "04/2015 sr.", "－", "company name", "city", "city state",
    "city, state"
}

ORG_CUES = [
    "university", "college", "school", "institute", "foundation",
    "council", "commission", "corporation", "organization",
    "academy", "department", "fraternity", "association"
]

def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,;:|/-'\"")
    if text.startswith("the "):
        text = text[4:]
    return text

def is_bad_non_skill(row):
    etype = row["entity_type"]
    text = str(row["entity_text"])
    norm = row["norm"]

    if etype in {"ORG", "PERSON", "GPE"}:
        if norm in SKILL_LIKE:
            return True

        if norm in BAD_GENERIC:
            return True

        if "\n" in text:
            return True

        if len(text) > 70:
            return True

    if etype == "PERSON":
        if any(char.isdigit() for char in text):
            return True
        if "summary" in norm:
            return True
        if len(norm.split()) > 4:
            return True

    if etype == "ORG":
        has_org_cue = any(cue in norm for cue in ORG_CUES)
        if not has_org_cue:
            return True

    if etype == "GPE":
        if len(norm) <= 2 and norm not in {"ia", "nc"}:
            return True
        if norm in SKILL_LIKE:
            return True

    return False

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
    sys["norm"] = sys["entity_text"].map(normalize)

    before = len(sys)

    sys_filtered = sys[~sys.apply(is_bad_non_skill, axis=1)].copy()
    sys_filtered = sys_filtered.drop_duplicates(subset=["resume_id", "entity_type", "norm"])

    after = len(sys_filtered)

    results = evaluate(gold, sys_filtered)

    overall = results.groupby("entity_type")[["gold", "system", "correct", "missed", "incorrect"]].sum().reset_index()
    overall["precision"] = overall["correct"] / overall["system"]
    overall["recall"] = overall["correct"] / overall["gold"]
    overall["f1"] = 2 * overall["precision"] * overall["recall"] / (overall["precision"] + overall["recall"])
    overall = overall.fillna(0)

    total = overall[["gold", "system", "correct", "missed", "incorrect"]].sum()
    total_precision = total["correct"] / total["system"] if total["system"] else 0
    total_recall = total["correct"] / total["gold"] if total["gold"] else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if total_precision + total_recall else 0

    sys_filtered.to_csv(BASE / "baseline_system_entities_10_v2_filtered.csv", index=False)
    results.to_csv(BASE / "baseline_per_resume_entity_results_10_v2_filtered.csv", index=False)
    overall.to_csv(BASE / "baseline_overall_by_entity_10_v2_filtered.csv", index=False)

    print("V2 filtering complete.")
    print(f"System entities before filtering: {before}")
    print(f"System entities after filtering: {after}")

    print("\nOverall by entity type after filtering:")
    print(overall.to_string(index=False))

    print("\nTOTAL AFTER FILTERING:")
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
