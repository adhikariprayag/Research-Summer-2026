from pathlib import Path
import re
import pandas as pd
import spacy

TEXT_DIR = Path(
    "02_dataset/large_scale_dataset/annotation_subset_private/resume_texts_100"
)
GOLD_PATH = Path(
    "02_dataset/large_scale_dataset/annotation_subset_private/gold_standard_100_template.csv"
)
OUT_DIR = Path(
    "02_dataset/large_scale_dataset/annotation_subset_private/baseline_test_15"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
]

SKILL_TERMS = [
    "python",
    "java",
    "c++",
    "sql",
    "html",
    "css",
    "javascript",
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
    "routers",
    "routing",
    "switches",
    "firewall",
    "cisco",
    "ccna",
    "vmware",
    "oracle",
    "peoplesoft",
    "workday hcm",
    "sharepoint",
    "visio",
    "ms office",
    "ms project",
    "excel",
    "powerpoint",
    "photoshop",
    "illustrator",
    "indesign",
    "dreamweaver",
    "flash",
    "sql developer",
    "toad",
    "remedy",
    "service now",
    "putty",
    "securecrt",
    "vnc",
    "wireshark",
    "solarwinds",
    "citrix",
    "mainframe",
    "as/400",
    "blackberry",
    "novell",
    "mcse",
    "cisa",
    "project management",
    "program management",
    "data analysis",
    "data management",
    "database",
    "hardware",
    "software",
    "technical support",
    "help desk",
    "troubleshooting",
    "security",
    "network security",
    "cybersecurity",
    "information technology",
    "itil",
    "change management",
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
    "pos",
    "printers",
    "scanners",
    "voip",
    "phone system",
    "mpls",
    "wireless network",
    "server infrastructure",
]


def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:|/-")


def clean_text(raw):
    if (
        "================================================================================"
        in raw
    ):
        raw = raw.split(
            "================================================================================",
            1,
        )[1]
    return raw


def extract_skills(text):
    found = set()
    low = text.lower()

    for skill in SKILL_TERMS:
        pattern = (
            r"(?<![A-Za-z0-9+#.])" + re.escape(skill.lower()) + r"(?![A-Za-z0-9+#.])"
        )
        if re.search(pattern, low):
            found.add(skill)

    return found


def evaluate(gold_df, sys_df):
    rows = []

    for rid in DONE_IDS:
        for etype in ["PERSON", "ORG", "DATE", "GPE", "SKILL"]:
            gold_set = set(
                gold_df[
                    (gold_df["resume_id"] == rid) & (gold_df["entity_type"] == etype)
                ]["norm"]
            )

            sys_set = set(
                sys_df[(sys_df["resume_id"] == rid) & (sys_df["entity_type"] == etype)][
                    "norm"
                ]
            )

            correct = gold_set & sys_set
            missed = gold_set - sys_set
            incorrect = sys_set - gold_set

            precision = len(correct) / len(sys_set) if sys_set else 0
            recall = len(correct) / len(gold_set) if gold_set else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0
            )

            rows.append(
                {
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
                }
            )

    return pd.DataFrame(rows)


def main():
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_lg")

    gold = pd.read_csv(GOLD_PATH)
    gold = gold[gold["resume_id"].isin(DONE_IDS)].copy()
    gold["norm"] = gold["entity_text"].map(normalize)

    system_rows = []

    for rid in DONE_IDS:
        path = TEXT_DIR / f"{rid}.txt"
        text = clean_text(path.read_text(errors="ignore"))

        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "DATE", "GPE"}:
                system_rows.append(
                    {
                        "resume_id": rid,
                        "entity_text": ent.text.strip(),
                        "entity_type": ent.label_,
                        "source": "spacy",
                    }
                )

        for skill in extract_skills(text):
            system_rows.append(
                {
                    "resume_id": rid,
                    "entity_text": skill,
                    "entity_type": "SKILL",
                    "source": "skill_dictionary",
                }
            )

    system = pd.DataFrame(system_rows)
    system["norm"] = system["entity_text"].map(normalize)
    system = system.drop_duplicates(subset=["resume_id", "entity_type", "norm"])

    results = evaluate(gold, system)

    overall = (
        results.groupby("entity_type")[
            ["gold", "system", "correct", "missed", "incorrect"]
        ]
        .sum()
        .reset_index()
    )
    overall["precision"] = overall["correct"] / overall["system"]
    overall["recall"] = overall["correct"] / overall["gold"]
    overall["f1"] = (
        2
        * overall["precision"]
        * overall["recall"]
        / (overall["precision"] + overall["recall"])
    )
    overall = overall.fillna(0)

    total = overall[["gold", "system", "correct", "missed", "incorrect"]].sum()
    total_precision = total["correct"] / total["system"] if total["system"] else 0
    total_recall = total["correct"] / total["gold"] if total["gold"] else 0
    total_f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if total_precision + total_recall
        else 0
    )

    system.to_csv(OUT_DIR / "baseline_system_entities_10.csv", index=False)
    results.to_csv(OUT_DIR / "baseline_per_resume_entity_results_10.csv", index=False)
    overall.to_csv(OUT_DIR / "baseline_overall_by_entity_10.csv", index=False)

    print("\nSaved outputs to:")
    print(OUT_DIR)

    print("\nOverall by entity type:")
    print(overall.to_string(index=False))

    print("\nTOTAL:")
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
