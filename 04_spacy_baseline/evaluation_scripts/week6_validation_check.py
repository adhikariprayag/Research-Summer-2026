from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_RESULTS = (
    PROJECT_ROOT
    / "07_results"
    / "final_20_evaluation"
    / "final_20_per_resume_results.csv"
)

GOLD_STANDARD = (
    PROJECT_ROOT
    / "02_dataset"
    / "gold_standard_annotations"
    / "gold_standard_final_20.csv"
)

PER_ENTITY_OVERALL = (
    PROJECT_ROOT
    / "07_results"
    / "week5_6_entity_analysis"
    / "per_entity_results"
    / "per_entity_overall.csv"
)

PER_ENTITY_BY_TYPE = (
    PROJECT_ROOT
    / "07_results"
    / "week5_6_entity_analysis"
    / "per_entity_results"
    / "per_entity_by_resume_type.csv"
)


def check_equal(label, actual, expected):
    if actual == expected:
        print(f"PASS: {label}: {actual}")
    else:
        print(f"FAIL: {label}: expected {expected}, got {actual}")


def main():
    print("Week 6 Validation Check")
    print("=" * 60)

    final_results = pd.read_csv(FINAL_RESULTS)
    gold = pd.read_csv(GOLD_STANDARD)
    per_entity = pd.read_csv(PER_ENTITY_OVERALL)
    per_entity_by_type = pd.read_csv(PER_ENTITY_BY_TYPE)

    print("\nChecking final 20-resume dataset...")
    check_equal("Final resume count", final_results["resume_id"].nunique(), 20)

    type_counts = final_results["resume_type"].value_counts().to_dict()
    check_equal("Standard resume count", type_counts.get("Standard", 0), 10)
    check_equal("Creative resume count", type_counts.get("Creative", 0), 10)

    print("\nChecking Gold Standard...")
    check_equal("Gold Standard entity count", len(gold), 1108)

    print("\nChecking final overall evaluation totals...")
    check_equal("Total gold count", int(final_results["gold_count"].sum()), 1108)
    check_equal("Total system count", int(final_results["system_count"].sum()), 1277)
    check_equal("Total correct", int(final_results["correct"].sum()), 322)
    check_equal("Total missed", int(final_results["missed"].sum()), 786)
    check_equal("Total incorrect", int(final_results["incorrect"].sum()), 955)

    print("\nChecking per-entity totals...")
    check_equal("Per-entity gold total", int(per_entity["gold_count"].sum()), 1108)
    check_equal("Per-entity system total", int(per_entity["system_count"].sum()), 1277)
    check_equal("Per-entity correct total", int(per_entity["correct"].sum()), 322)
    check_equal("Per-entity missed total", int(per_entity["missed"].sum()), 786)
    check_equal("Per-entity incorrect total", int(per_entity["incorrect"].sum()), 955)

    print("\nChecking entity types...")
    expected_entities = {"PERSON", "ORG", "DATE", "GPE", "SKILL"}
    actual_entities = set(per_entity["entity_type"].astype(str))

    if actual_entities == expected_entities:
        print(f"PASS: Entity types: {sorted(actual_entities)}")
    else:
        print(f"FAIL: Entity types mismatch: {sorted(actual_entities)}")

    print("\nChecking resume-type entity table...")
    check_equal("Rows in per_entity_by_resume_type", len(per_entity_by_type), 10)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
