from pathlib import Path
import pandas as pd


ROOT = Path("/Users/prayagadhikari/Desktop/ATS_NER_Resume_Research")

PER_RESUME_PATH = (
    ROOT
    / "02_dataset/large_scale_dataset/full_v4_private"
    / "generalized_v4_per_resume_counts_1020.csv"
)

METADATA_PATH = (
    ROOT
    / "02_dataset/large_scale_dataset/metadata"
    / "combined_large_cs_resume_metadata.csv"
)

OUT_DIR = (
    ROOT
    / "02_dataset/large_scale_dataset/full_v4_private"
)

OUT_PATH = OUT_DIR / "generalized_v4_validation_sample_30.csv"

RANDOM_STATE = 42


def choose_from_band(df, label, n):
    if len(df) <= n:
        chosen = df.copy()
    else:
        chosen = df.sample(
            n=n,
            random_state=RANDOM_STATE,
        )

    chosen = chosen.copy()
    chosen["prediction_band"] = label
    return chosen


def sample_source(source_df):
    source_df = source_df.sort_values(
        "total_predictions"
    ).reset_index(drop=True)

    low_cut = source_df["total_predictions"].quantile(0.33)
    high_cut = source_df["total_predictions"].quantile(0.67)

    low = source_df[
        source_df["total_predictions"] <= low_cut
    ]

    medium = source_df[
        (source_df["total_predictions"] > low_cut)
        & (source_df["total_predictions"] < high_cut)
    ]

    high = source_df[
        source_df["total_predictions"] >= high_cut
    ]

    selected = pd.concat(
        [
            choose_from_band(low, "low", 3),
            choose_from_band(medium, "medium", 4),
            choose_from_band(high, "high", 3),
        ],
        ignore_index=True,
    )

    return selected


def main():
    per_resume = pd.read_csv(PER_RESUME_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    per_resume["resume_id"] = (
        per_resume["resume_id"]
        .astype(str)
        .str.strip()
    )

    metadata["resume_id"] = (
        metadata["resume_id"]
        .astype(str)
        .str.strip()
    )

    merged = per_resume.merge(
        metadata[
            [
                "resume_id",
                "source_dataset",
                "original_category",
                "file_type",
            ]
        ],
        on="resume_id",
        how="left",
        validate="one_to_one",
    )

    samples = []

    for source, source_df in merged.groupby(
        "source_dataset"
    ):
        selected = sample_source(source_df)
        samples.append(selected)

    sample = pd.concat(
        samples,
        ignore_index=True,
    )

    sample = sample.sort_values(
        [
            "source_dataset",
            "prediction_band",
            "total_predictions",
            "resume_id",
        ]
    )

    output_columns = [
        "resume_id",
        "source_dataset",
        "original_category",
        "file_type",
        "prediction_band",
        "total_predictions",
        "person_predictions",
        "org_predictions",
        "date_predictions",
        "gpe_predictions",
        "skill_predictions",
        "text_character_count",
    ]

    sample = sample[output_columns]

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample.to_csv(
        OUT_PATH,
        index=False,
    )

    print("Validation sample created.")
    print(f"Rows: {len(sample)}")
    print(f"Output: {OUT_PATH}")

    print("\nBy source:")
    print(
        sample["source_dataset"]
        .value_counts()
        .to_string()
    )

    print("\nBy source and prediction band:")
    print(
        sample.groupby(
            [
                "source_dataset",
                "prediction_band",
            ]
        )
        .size()
        .to_string()
    )

    print("\nPrediction-count range by source:")
    print(
        sample.groupby(
            "source_dataset"
        )["total_predictions"]
        .agg(
            [
                "min",
                "median",
                "max",
            ]
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
