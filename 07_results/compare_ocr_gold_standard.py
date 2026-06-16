from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLD_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "gold_standard_annotations"
    / "gold_standard_final_20.csv"
)

SYSTEM_FOLDER = (
    PROJECT_ROOT
    / "04_spacy_baseline"
    / "ocr_combined_outputs"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "07_results"
    / "ocr_evaluation_outputs"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OCR_RESUME_IDS = {
    "R002_creative",
    "R003_complex",
}


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "resume_id",
        "entity_text",
        "entity_type",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = df.copy()

    df["resume_id"] = (
        df["resume_id"]
        .astype(str)
        .str.strip()
        .str.replace("_ocr", "", regex=False)
    )

    df["entity_text"] = (
        df["entity_text"]
        .astype(str)
        .str.strip()
    )

    df["entity_type"] = (
        df["entity_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["normalized_text"] = (
        df["entity_text"]
        .apply(normalize_text)
    )

    df = df.drop_duplicates(
        subset=[
            "resume_id",
            "normalized_text",
            "entity_type",
        ]
    )

    return df


def load_system_outputs() -> pd.DataFrame:
    files = sorted(
        SYSTEM_FOLDER.glob("*_ocr_combined_entities.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No OCR combined files found in: {SYSTEM_FOLDER}"
        )

    dataframes = []

    for file in files:
        dataframe = pd.read_csv(file)
        dataframes.append(dataframe)

    return pd.concat(
        dataframes,
        ignore_index=True,
    )


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_resume(
    resume_id: str,
    gold_df: pd.DataFrame,
    system_df: pd.DataFrame,
):
    gold_resume = gold_df[
        gold_df["resume_id"] == resume_id
    ].copy()

    system_resume = system_df[
        system_df["resume_id"] == resume_id
    ].copy()

    gold_keys = set(
        zip(
            gold_resume["normalized_text"],
            gold_resume["entity_type"],
        )
    )

    system_keys = set(
        zip(
            system_resume["normalized_text"],
            system_resume["entity_type"],
        )
    )

    correct_keys = gold_keys & system_keys
    missed_keys = gold_keys - system_keys
    incorrect_keys = system_keys - gold_keys

    correct = len(correct_keys)
    missed = len(missed_keys)
    incorrect = len(incorrect_keys)

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

    detail_rows = []

    for text, entity_type in sorted(correct_keys):
        detail_rows.append(
            {
                "resume_id": resume_id,
                "entity_text": text,
                "entity_type": entity_type,
                "status": "Correct",
            }
        )

    for text, entity_type in sorted(missed_keys):
        detail_rows.append(
            {
                "resume_id": resume_id,
                "entity_text": text,
                "entity_type": entity_type,
                "status": "Missed",
            }
        )

    for text, entity_type in sorted(incorrect_keys):
        detail_rows.append(
            {
                "resume_id": resume_id,
                "entity_text": text,
                "entity_type": entity_type,
                "status": "Incorrect",
            }
        )

    summary_row = {
        "resume_id": resume_id,
        "gold_count": len(gold_keys),
        "system_count": len(system_keys),
        "correct": correct,
        "missed": missed,
        "incorrect": incorrect,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
    }

    return summary_row, detail_rows


def main() -> None:
    if not GOLD_FILE.exists():
        print(f"Gold Standard file not found: {GOLD_FILE}")
        return

    try:
        gold_df = pd.read_csv(GOLD_FILE)
        system_df = load_system_outputs()

        gold_df = prepare_dataframe(gold_df)
        system_df = prepare_dataframe(system_df)

        gold_df = gold_df[
            gold_df["resume_id"].isin(OCR_RESUME_IDS)
        ]

        system_df = system_df[
            system_df["resume_id"].isin(OCR_RESUME_IDS)
        ]

        summary_rows = []
        detail_rows = []

        for resume_id in sorted(OCR_RESUME_IDS):
            summary, details = evaluate_resume(
                resume_id,
                gold_df,
                system_df,
            )

            summary_rows.append(summary)
            detail_rows.extend(details)

            print(f"Resume: {resume_id}")
            print(f"Gold: {summary['gold_count']}")
            print(f"System: {summary['system_count']}")
            print(f"Correct: {summary['correct']}")
            print(f"Missed: {summary['missed']}")
            print(f"Incorrect: {summary['incorrect']}")
            print(f"Precision: {summary['precision']}")
            print(f"Recall: {summary['recall']}")
            print(f"F1: {summary['f1_score']}\n")

        summary_df = pd.DataFrame(summary_rows)
        details_df = pd.DataFrame(detail_rows)

        summary_path = (
            OUTPUT_FOLDER
            / "ocr_gold_vs_combined_summary.csv"
        )

        details_path = (
            OUTPUT_FOLDER
            / "ocr_gold_vs_combined_details.csv"
        )

        summary_df.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        details_df.to_csv(
            details_path,
            index=False,
            encoding="utf-8-sig",
        )

        total_correct = summary_df["correct"].sum()
        total_missed = summary_df["missed"].sum()
        total_incorrect = summary_df["incorrect"].sum()

        overall_precision = safe_divide(
            total_correct,
            total_correct + total_incorrect,
        )

        overall_recall = safe_divide(
            total_correct,
            total_correct + total_missed,
        )

        overall_f1 = safe_divide(
            2 * overall_precision * overall_recall,
            overall_precision + overall_recall,
        )

        print("OVERALL OCR RESULTS")
        print(f"Correct: {total_correct}")
        print(f"Missed: {total_missed}")
        print(f"Incorrect: {total_incorrect}")
        print(f"Precision: {overall_precision:.3f}")
        print(f"Recall: {overall_recall:.3f}")
        print(f"F1: {overall_f1:.3f}")

        print("\nOCR evaluation completed.")
        print(f"Saved summary: {summary_path}")
        print(f"Saved details: {details_path}")

    except Exception as error:
        print("OCR evaluation failed.")
        print(f"Reason: {error}")


if __name__ == "__main__":
    main()
    