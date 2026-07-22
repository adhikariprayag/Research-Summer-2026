from pathlib import Path
import re

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "generalized_v4_validation_sample_50.csv"
)

V4_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "generalized_v4_system_entities_1020.csv"
)

GPT_FILE = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "validation_50_outputs"
    / "gpt_skill_recovery_validation_50_combined.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "validation_50_hybrid"
)

HYBRID_OUTPUT = (
    OUTPUT_DIR
    / "hybrid_v4_gpt_skill_predictions_50.csv"
)

PER_RESUME_OUTPUT = (
    OUTPUT_DIR
    / "hybrid_v4_gpt_skill_counts_per_resume_50.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "hybrid_v4_gpt_skill_summary_50.csv"
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def normalize(text: object) -> str:
    """Normalize entity text for matching and deduplication."""

    value = str(text).strip().casefold()
    value = re.sub(r"\s+", " ", value)

    return value


def choose_display_text(group: pd.DataFrame) -> str:
    """
    Choose the displayed entity text.

    Prefer GPT wording when GPT found the skill.
    Otherwise, use the V4 wording.
    """

    gpt_rows = group[
        group["prediction_system"] == "GPT"
    ]

    if not gpt_rows.empty:
        return str(
            gpt_rows.iloc[0]["entity_text"]
        ).strip()

    return str(
        group.iloc[0]["entity_text"]
    ).strip()


def determine_prediction_source(
    group: pd.DataFrame,
) -> str:
    """Return V4, GPT, or both."""

    systems = set(
        group["prediction_system"]
        .astype(str)
        .str.strip()
    )

    if systems == {"V4", "GPT"}:
        return "both"

    if systems == {"GPT"}:
        return "GPT"

    return "V4"


def get_source_text(
    group: pd.DataFrame,
    system_name: str,
) -> str:
    """Return the entity wording from one prediction system."""

    system_rows = group[
        group["prediction_system"] == system_name
    ]

    if system_rows.empty:
        return ""

    return str(
        system_rows.iloc[0]["entity_text"]
    ).strip()


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main() -> None:
    """Create Hybrid V4 + GPT SKILL predictions for 50 resumes."""

    required_files = [
        SAMPLE_FILE,
        V4_FILE,
        GPT_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file was not found:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample = pd.read_csv(SAMPLE_FILE)
    v4 = pd.read_csv(V4_FILE)
    gpt = pd.read_csv(GPT_FILE)

    print(
        f"Loaded validation sample: "
        f"{len(sample)} resumes"
    )

    print(
        f"Loaded all V4 predictions: "
        f"{len(v4)}"
    )

    print(
        f"Loaded GPT skill predictions: "
        f"{len(gpt)}"
    )

    # -----------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------

    sample_required = {
        "resume_id",
        "dataset_group",
        "prediction_band",
        "selection_stage",
    }

    prediction_required = {
        "resume_id",
        "entity_text",
        "entity_type",
    }

    missing_sample_columns = (
        sample_required - set(sample.columns)
    )

    if missing_sample_columns:
        raise ValueError(
            "Sample file is missing columns: "
            f"{sorted(missing_sample_columns)}"
        )

    for name, dataframe in [
        ("V4", v4),
        ("GPT", gpt),
    ]:
        missing_columns = (
            prediction_required - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{name} file is missing columns: "
                f"{sorted(missing_columns)}"
            )

    if len(sample) != 50:
        raise ValueError(
            f"Expected 50 resumes, but found "
            f"{len(sample)}."
        )

    # -----------------------------------------------------
    # Clean resume IDs
    # -----------------------------------------------------

    for dataframe in [
        sample,
        v4,
        gpt,
    ]:
        dataframe["resume_id"] = (
            dataframe["resume_id"]
            .astype(str)
            .str.strip()
        )

    selected_resume_ids = set(
        sample["resume_id"]
    )

    # -----------------------------------------------------
    # Keep SKILL predictions for selected resumes only
    # -----------------------------------------------------

    v4 = v4[
        (
            v4["resume_id"].isin(
                selected_resume_ids
            )
        )
        & (
            v4["entity_type"]
            .astype(str)
            .str.strip()
            .str.upper()
            == "SKILL"
        )
    ].copy()

    gpt = gpt[
        (
            gpt["resume_id"].isin(
                selected_resume_ids
            )
        )
        & (
            gpt["entity_type"]
            .astype(str)
            .str.strip()
            .str.upper()
            == "SKILL"
        )
    ].copy()

    print(
        f"Selected V4 skills for 50 resumes: "
        f"{len(v4)}"
    )

    print(
        f"Selected GPT skills for 50 resumes: "
        f"{len(gpt)}"
    )

    # -----------------------------------------------------
    # Normalize skill text
    # -----------------------------------------------------

    for dataframe in [
        v4,
        gpt,
    ]:
        dataframe["entity_text"] = (
            dataframe["entity_text"]
            .astype(str)
            .str.strip()
        )

        dataframe["norm"] = (
            dataframe["entity_text"]
            .map(normalize)
        )

        dataframe.drop(
            dataframe[
                dataframe["norm"] == ""
            ].index,
            inplace=True,
        )

    v4 = v4.drop_duplicates(
        subset=[
            "resume_id",
            "norm",
        ]
    )

    gpt = gpt.drop_duplicates(
        subset=[
            "resume_id",
            "norm",
        ]
    )

    v4["prediction_system"] = "V4"
    gpt["prediction_system"] = "GPT"

    # -----------------------------------------------------
    # Combine V4 and GPT
    # -----------------------------------------------------

    combined = pd.concat(
        [
            v4[
                [
                    "resume_id",
                    "entity_text",
                    "entity_type",
                    "norm",
                    "prediction_system",
                ]
            ],
            gpt[
                [
                    "resume_id",
                    "entity_text",
                    "entity_type",
                    "norm",
                    "prediction_system",
                ]
            ],
        ],
        ignore_index=True,
    )

    hybrid_rows = []

    grouped_predictions = combined.groupby(
        [
            "resume_id",
            "norm",
        ],
        sort=False,
    )

    for (
        resume_id,
        normalized_skill,
    ), group in grouped_predictions:

        hybrid_rows.append(
            {
                "resume_id": resume_id,
                "entity_text": choose_display_text(
                    group
                ),
                "entity_type": "SKILL",
                "norm": normalized_skill,
                "prediction_source":
                    determine_prediction_source(
                        group
                    ),
                "v4_entity_text":
                    get_source_text(
                        group,
                        "V4",
                    ),
                "gpt_entity_text":
                    get_source_text(
                        group,
                        "GPT",
                    ),
            }
        )

    hybrid = pd.DataFrame(
        hybrid_rows
    )

    # -----------------------------------------------------
    # Add validation metadata
    # -----------------------------------------------------

    metadata_columns = [
        "resume_id",
        "dataset_group",
        "prediction_band",
        "selection_stage",
    ]

    if "validation_order" in sample.columns:
        metadata_columns.insert(
            0,
            "validation_order",
        )

    metadata = sample[
        metadata_columns
    ].copy()

    hybrid = hybrid.merge(
        metadata,
        on="resume_id",
        how="left",
        validate="many_to_one",
    )

    if hybrid["dataset_group"].isna().any():
        missing_metadata_ids = (
            hybrid.loc[
                hybrid["dataset_group"].isna(),
                "resume_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Metadata was missing for these resumes:\n"
            + "\n".join(
                sorted(missing_metadata_ids)
            )
        )

    # -----------------------------------------------------
    # Organize and sort hybrid output
    # -----------------------------------------------------

    output_columns = [
        "validation_order",
        "resume_id",
        "dataset_group",
        "prediction_band",
        "selection_stage",
        "entity_text",
        "entity_type",
        "prediction_source",
        "v4_entity_text",
        "gpt_entity_text",
        "norm",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in hybrid.columns
    ]

    hybrid = hybrid[
        output_columns
    ]

    sort_columns = [
        column
        for column in [
            "validation_order",
            "resume_id",
            "norm",
        ]
        if column in hybrid.columns
    ]

    hybrid = hybrid.sort_values(
        sort_columns
    ).reset_index(drop=True)

    hybrid.to_csv(
        HYBRID_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Create per-resume counts
    # -----------------------------------------------------

    source_counts = (
        hybrid.groupby(
            [
                "resume_id",
                "prediction_source",
            ]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for source_name in [
        "V4",
        "GPT",
        "both",
    ]:
        if source_name not in source_counts.columns:
            source_counts[source_name] = 0

    source_counts = source_counts.rename(
        columns={
            "V4": "v4_only_count",
            "GPT": "gpt_only_count",
            "both": "both_count",
        }
    )

    source_counts["hybrid_total"] = (
        source_counts["v4_only_count"]
        + source_counts["gpt_only_count"]
        + source_counts["both_count"]
    )

    per_resume = metadata.merge(
        source_counts,
        on="resume_id",
        how="left",
    )

    count_columns = [
        "v4_only_count",
        "gpt_only_count",
        "both_count",
        "hybrid_total",
    ]

    per_resume[count_columns] = (
        per_resume[count_columns]
        .fillna(0)
        .astype(int)
    )

    if "validation_order" in per_resume.columns:
        per_resume = per_resume.sort_values(
            "validation_order"
        )

    per_resume.to_csv(
        PER_RESUME_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Create aggregate summary
    # -----------------------------------------------------

    summary_rows = []

    for dataset_group in [
        "Jithin",
        "Saugata",
        "Snehaan",
    ]:
        source_df = hybrid[
            hybrid["dataset_group"]
            == dataset_group
        ]

        summary_rows.append(
            {
                "dataset_group": dataset_group,
                "resumes": (
                    sample[
                        sample["dataset_group"]
                        == dataset_group
                    ]["resume_id"]
                    .nunique()
                ),
                "hybrid_predictions":
                    len(source_df),
                "v4_only":
                    (
                        source_df[
                            "prediction_source"
                        ]
                        == "V4"
                    ).sum(),
                "gpt_only":
                    (
                        source_df[
                            "prediction_source"
                        ]
                        == "GPT"
                    ).sum(),
                "both":
                    (
                        source_df[
                            "prediction_source"
                        ]
                        == "both"
                    ).sum(),
            }
        )

    summary_rows.append(
        {
            "dataset_group": "Combined",
            "resumes": sample[
                "resume_id"
            ].nunique(),
            "hybrid_predictions": len(
                hybrid
            ),
            "v4_only": (
                hybrid["prediction_source"]
                == "V4"
            ).sum(),
            "gpt_only": (
                hybrid["prediction_source"]
                == "GPT"
            ).sum(),
            "both": (
                hybrid["prediction_source"]
                == "both"
            ).sum(),
        }
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Final checks
    # -----------------------------------------------------

    hybrid_resume_ids = set(
        hybrid["resume_id"]
    )

    resumes_with_no_predictions = sorted(
        selected_resume_ids
        - hybrid_resume_ids
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print(
        "\nHybrid V4 + GPT prediction creation complete."
    )

    print(
        f"Hybrid unique skill predictions: "
        f"{len(hybrid)}"
    )

    print(
        "\nPredictions by source:"
    )

    print(
        hybrid["prediction_source"]
        .value_counts()
        .to_string()
    )

    print(
        "\nSummary by dataset:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    if resumes_with_no_predictions:
        print(
            "\nWarning: The following resumes had "
            "no hybrid skill predictions:"
        )

        for resume_id in resumes_with_no_predictions:
            print(f"- {resume_id}")

    else:
        print(
            "\nAll 50 resumes have at least one "
            "hybrid skill prediction."
        )

    print(
        f"\nHybrid prediction file:\n"
        f"{HYBRID_OUTPUT}"
    )

    print(
        f"\nPer-resume count file:\n"
        f"{PER_RESUME_OUTPUT}"
    )

    print(
        f"\nSummary file:\n"
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()
    