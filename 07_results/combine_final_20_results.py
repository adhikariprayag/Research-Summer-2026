from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUMMARY_FILES = [
    (
        PROJECT_ROOT
        / "07_results"
        / "week4_evaluation_outputs"
        / "week4_gold_vs_combined_summary.csv"
    ),
    (
        PROJECT_ROOT
        / "07_results"
        / "batch2_evaluation_outputs"
        / "batch2_gold_vs_combined_summary.csv"
    ),
    (
        PROJECT_ROOT
        / "07_results"
        / "ocr_evaluation_outputs"
        / "ocr_gold_vs_combined_summary.csv"
    ),
]

OUTPUT_FOLDER = PROJECT_ROOT / "07_results" / "final_20_evaluation"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


STANDARD_RESUMES = {
    "R006_kaggle",
    "R007_kaggle",
    "R008_kaggle",
    "R010_kaggle",
    "R011_kaggle",
    "R012_kaggle",
    "R015_kaggle",
    "R018_kaggle",
    "R019_kaggle",
    "R034_kaggle",
}

CREATIVE_RESUMES = {
    "R002_creative",
    "R003_complex",
    "R004_creative_text",
    "R005_complex_text",
    "R014_kaggle",
    "R017_kaggle",
    "R026_kaggle_creative",
    "R027_creative",
    "R028_creative",
    "R029_creative",
}


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_group_metrics(
    dataframe: pd.DataFrame,
    group_name: str,
) -> dict:
    correct = int(dataframe["correct"].sum())
    missed = int(dataframe["missed"].sum())
    incorrect = int(dataframe["incorrect"].sum())

    precision = safe_divide(
        correct,
        correct + incorrect,
    )

    recall = safe_divide(
        correct,
        correct + missed,
    )

    f1_score = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "group": group_name,
        "resume_count": len(dataframe),
        "gold_count": int(dataframe["gold_count"].sum()),
        "system_count": int(dataframe["system_count"].sum()),
        "correct": correct,
        "missed": missed,
        "incorrect": incorrect,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
    }


def main() -> None:
    dataframes = []

    for summary_file in SUMMARY_FILES:
        if not summary_file.exists():
            print(f"Missing summary file: {summary_file}")
            return

        dataframe = pd.read_csv(summary_file)
        dataframes.append(dataframe)

    final_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    final_df = final_df.drop_duplicates(
        subset=["resume_id"],
        keep="first",
    )

    final_df["resume_type"] = final_df["resume_id"].apply(
        lambda resume_id: (
            "Standard"
            if resume_id in STANDARD_RESUMES
            else "Creative" if resume_id in CREATIVE_RESUMES else "Unknown"
        )
    )

    unknown_resumes = final_df[final_df["resume_type"] == "Unknown"]

    if not unknown_resumes.empty:
        print("Unknown resume IDs found:")

        for resume_id in unknown_resumes["resume_id"]:
            print(f"- {resume_id}")

        return

    final_df = final_df.sort_values(by=["resume_type", "resume_id"])

    standard_df = final_df[final_df["resume_type"] == "Standard"]

    creative_df = final_df[final_df["resume_type"] == "Creative"]

    group_rows = [
        calculate_group_metrics(
            standard_df,
            "Standard",
        ),
        calculate_group_metrics(
            creative_df,
            "Creative",
        ),
        calculate_group_metrics(
            final_df,
            "Overall",
        ),
    ]

    group_df = pd.DataFrame(group_rows)

    per_resume_path = OUTPUT_FOLDER / "final_20_per_resume_results.csv"

    group_path = OUTPUT_FOLDER / "final_20_group_summary.csv"

    final_df.to_csv(
        per_resume_path,
        index=False,
        encoding="utf-8-sig",
    )

    group_df.to_csv(
        group_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Total resumes combined: {len(final_df)}")
    print(f"Standard resumes: {len(standard_df)}")
    print(f"Creative resumes: {len(creative_df)}\n")

    for row in group_rows:
        print(f"{row['group'].upper()} RESULTS")
        print(f"Resumes: {row['resume_count']}")
        print(f"Gold: {row['gold_count']}")
        print(f"System: {row['system_count']}")
        print(f"Correct: {row['correct']}")
        print(f"Missed: {row['missed']}")
        print(f"Incorrect: {row['incorrect']}")
        print(f"Precision: {row['precision']}")
        print(f"Recall: {row['recall']}")
        print(f"F1: {row['f1_score']}\n")

    print("Final 20-resume evaluation completed.")
    print(f"Saved per-resume results: {per_resume_path}")
    print(f"Saved group summary: {group_path}")


if __name__ == "__main__":
    main()
