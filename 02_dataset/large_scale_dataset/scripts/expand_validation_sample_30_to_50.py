from pathlib import Path

import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_SEED = 2026

PROJECT_ROOT = Path.home() / "Desktop" / "ATS_NER_Resume_Research"

FULL_COUNTS_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "generalized_v4_per_resume_counts_1020.csv"
)

EXISTING_SAMPLE_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "generalized_v4_validation_sample_30.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "generalized_v4_validation_sample_50.csv"
)


# Final target distribution.
FINAL_TARGETS = {
    "Jithin": {
        "low": 5,
        "medium": 5,
        "high": 5,
    },
    "Saugata": {
        "low": 6,
        "medium": 8,
        "high": 6,
    },
    "Snehaan": {
        "low": 5,
        "medium": 5,
        "high": 5,
    },
}


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------


def find_column(
    dataframe: pd.DataFrame,
    possible_names: list[str],
) -> str:
    """Return the first matching column name."""

    for name in possible_names:
        if name in dataframe.columns:
            return name

    raise ValueError(
        "Could not find any of these columns: "
        + ", ".join(possible_names)
        + f"\nAvailable columns: {list(dataframe.columns)}"
    )


def source_from_resume_id(resume_id: object) -> str:
    """Determine the source dataset from the resume ID prefix."""

    value = str(resume_id).strip().upper()

    if value.startswith("JIT_"):
        return "Jithin"

    if value.startswith("SAU_"):
        return "Saugata"

    if value.startswith("SNE_"):
        return "Snehaan"

    raise ValueError(
        f"Could not determine the dataset source from resume ID: " f"{resume_id}"
    )


def normalize_band(band_value: object) -> str:
    """Normalize prediction-band labels."""

    value = str(band_value).strip().lower()

    if value in {"low", "medium", "high"}:
        return value

    raise ValueError(f"Unrecognized prediction band: {band_value}")


# ---------------------------------------------------------
# Check that input files exist
# ---------------------------------------------------------

if not FULL_COUNTS_FILE.exists():
    raise FileNotFoundError(
        f"Full per-resume counts file was not found:\n" f"{FULL_COUNTS_FILE}"
    )

if not EXISTING_SAMPLE_FILE.exists():
    raise FileNotFoundError(
        f"Existing 30-resume sample was not found:\n" f"{EXISTING_SAMPLE_FILE}"
    )


# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------

full_counts = pd.read_csv(FULL_COUNTS_FILE)
existing_sample = pd.read_csv(EXISTING_SAMPLE_FILE)

print(f"Loaded full dataset: {len(full_counts)} resumes")
print(f"Loaded existing sample: {len(existing_sample)} resumes")


# ---------------------------------------------------------
# Identify required columns
# ---------------------------------------------------------

resume_id_column = find_column(
    full_counts,
    [
        "resume_id",
        "Resume ID",
        "resume",
    ],
)

prediction_count_column = find_column(
    full_counts,
    [
        "total_predictions",
        "prediction_count",
        "total_entity_predictions",
        "entity_count",
    ],
)

existing_resume_id_column = find_column(
    existing_sample,
    [
        "resume_id",
        "Resume ID",
        "resume",
    ],
)


# Look for an existing prediction-band column.
existing_band_column = None

for possible_name in [
    "prediction_band",
    "prediction_level",
    "band",
]:
    if possible_name in existing_sample.columns:
        existing_band_column = possible_name
        break


# ---------------------------------------------------------
# Clean resume IDs
# ---------------------------------------------------------

full_counts[resume_id_column] = full_counts[resume_id_column].astype(str).str.strip()

existing_sample[existing_resume_id_column] = (
    existing_sample[existing_resume_id_column].astype(str).str.strip()
)


# ---------------------------------------------------------
# Determine source from resume ID
# ---------------------------------------------------------

full_counts["dataset_group"] = full_counts[resume_id_column].apply(
    source_from_resume_id
)


# ---------------------------------------------------------
# Assign low, medium, and high prediction bands
# ---------------------------------------------------------

# The ranking is performed separately for each source.
full_counts["prediction_rank"] = full_counts.groupby("dataset_group")[
    prediction_count_column
].rank(
    method="first",
    pct=True,
)


def assign_prediction_band(percentile: float) -> str:
    """Assign a prediction band from a percentile."""

    if percentile <= 1 / 3:
        return "low"

    if percentile <= 2 / 3:
        return "medium"

    return "high"


full_counts["prediction_band"] = full_counts["prediction_rank"].apply(
    assign_prediction_band
)


# ---------------------------------------------------------
# Find the existing 30 resumes
# ---------------------------------------------------------

existing_ids = set(existing_sample[existing_resume_id_column].tolist())

selected_existing = full_counts[full_counts[resume_id_column].isin(existing_ids)].copy()

found_existing_ids = set(selected_existing[resume_id_column].tolist())

missing_existing_ids = existing_ids - found_existing_ids

if missing_existing_ids:
    raise ValueError(
        "The following existing sample IDs were not found in "
        "the full counts file:\n" + "\n".join(sorted(missing_existing_ids))
    )

if len(selected_existing) != 30:
    raise ValueError(
        f"Expected 30 existing resumes, but found " f"{len(selected_existing)}."
    )


# ---------------------------------------------------------
# Preserve original bands from the 30-resume sample
# ---------------------------------------------------------

if existing_band_column is not None:
    existing_band_map = (
        existing_sample[
            [
                existing_resume_id_column,
                existing_band_column,
            ]
        ]
        .drop_duplicates(subset=[existing_resume_id_column])
        .set_index(existing_resume_id_column)[existing_band_column]
        .apply(normalize_band)
        .to_dict()
    )

    selected_existing["prediction_band"] = (
        selected_existing[resume_id_column]
        .map(existing_band_map)
        .fillna(selected_existing["prediction_band"])
    )

selected_existing["selection_stage"] = "existing_30"


# ---------------------------------------------------------
# Show current 30-resume distribution
# ---------------------------------------------------------

print("\nExisting 30-resume distribution:")

existing_distribution = (
    selected_existing.groupby(
        [
            "dataset_group",
            "prediction_band",
        ]
    )
    .size()
    .unstack(fill_value=0)
    .reindex(
        index=[
            "Jithin",
            "Saugata",
            "Snehaan",
        ],
        columns=[
            "low",
            "medium",
            "high",
        ],
        fill_value=0,
    )
)

existing_distribution["total"] = existing_distribution.sum(axis=1)

print(existing_distribution)


# ---------------------------------------------------------
# Select the additional 20 resumes
# ---------------------------------------------------------

current_distribution = (
    selected_existing.groupby(
        [
            "dataset_group",
            "prediction_band",
        ]
    )
    .size()
    .to_dict()
)

new_selections = []
seed_counter = 0

for dataset_group, band_targets in FINAL_TARGETS.items():
    for prediction_band, final_target in band_targets.items():

        current_count = current_distribution.get(
            (
                dataset_group,
                prediction_band,
            ),
            0,
        )

        number_to_add = final_target - current_count

        if number_to_add < 0:
            raise ValueError(
                f"The existing sample already contains more "
                f"resumes than the final target for:\n"
                f"Source: {dataset_group}\n"
                f"Band: {prediction_band}\n"
                f"Existing: {current_count}\n"
                f"Target: {final_target}"
            )

        if number_to_add == 0:
            continue

        candidate_pool = full_counts[
            (full_counts["dataset_group"] == dataset_group)
            & (full_counts["prediction_band"] == prediction_band)
            & (~full_counts[resume_id_column].isin(existing_ids))
        ].copy()

        if len(candidate_pool) < number_to_add:
            raise ValueError(
                f"Not enough candidates available for:\n"
                f"Source: {dataset_group}\n"
                f"Band: {prediction_band}\n"
                f"Needed: {number_to_add}\n"
                f"Available: {len(candidate_pool)}"
            )

        selected_additions = candidate_pool.sample(
            n=number_to_add,
            random_state=RANDOM_SEED + seed_counter,
        ).copy()

        selected_additions["selection_stage"] = "added_20"

        new_selections.append(selected_additions)

        print(
            f"Selected {number_to_add} additional "
            f"{dataset_group} {prediction_band} resumes."
        )

        seed_counter += 1


if not new_selections:
    raise ValueError("No additional resumes were selected.")

added_resumes = pd.concat(
    new_selections,
    ignore_index=True,
)


# ---------------------------------------------------------
# Combine the original 30 and additional 20
# ---------------------------------------------------------

final_sample = pd.concat(
    [
        selected_existing,
        added_resumes,
    ],
    ignore_index=True,
)


# ---------------------------------------------------------
# Validate total counts
# ---------------------------------------------------------

if len(added_resumes) != 20:
    raise ValueError(
        f"Expected to add 20 resumes, but selected " f"{len(added_resumes)}."
    )

if len(final_sample) != 50:
    raise ValueError(
        f"Expected a final sample of 50 resumes, but found " f"{len(final_sample)}."
    )


# ---------------------------------------------------------
# Check for duplicate resume IDs
# ---------------------------------------------------------

duplicate_mask = final_sample[resume_id_column].duplicated(keep=False)

if duplicate_mask.any():
    duplicate_ids = (
        final_sample.loc[
            duplicate_mask,
            resume_id_column,
        ]
        .drop_duplicates()
        .tolist()
    )

    raise ValueError(
        "Duplicate resume IDs were found:\n" + "\n".join(sorted(duplicate_ids))
    )


# ---------------------------------------------------------
# Validate source totals
# ---------------------------------------------------------

expected_source_totals = {
    "Jithin": 15,
    "Saugata": 20,
    "Snehaan": 15,
}

actual_source_totals = final_sample["dataset_group"].value_counts().to_dict()

for source_name, expected_total in expected_source_totals.items():
    actual_total = actual_source_totals.get(
        source_name,
        0,
    )

    if actual_total != expected_total:
        raise ValueError(
            f"Incorrect final total for {source_name}.\n"
            f"Expected: {expected_total}\n"
            f"Actual: {actual_total}"
        )


# ---------------------------------------------------------
# Validate prediction-band totals
# ---------------------------------------------------------

for dataset_group, band_targets in FINAL_TARGETS.items():
    for prediction_band, expected_count in band_targets.items():

        actual_count = len(
            final_sample[
                (final_sample["dataset_group"] == dataset_group)
                & (final_sample["prediction_band"] == prediction_band)
            ]
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Incorrect count for:\n"
                f"Source: {dataset_group}\n"
                f"Band: {prediction_band}\n"
                f"Expected: {expected_count}\n"
                f"Actual: {actual_count}"
            )


# ---------------------------------------------------------
# Sort final sample
# ---------------------------------------------------------

source_order = {
    "Jithin": 1,
    "Saugata": 2,
    "Snehaan": 3,
}

band_order = {
    "low": 1,
    "medium": 2,
    "high": 3,
}

selection_stage_order = {
    "existing_30": 1,
    "added_20": 2,
}

final_sample["_source_order"] = final_sample["dataset_group"].map(source_order)

final_sample["_band_order"] = final_sample["prediction_band"].map(band_order)

final_sample["_selection_stage_order"] = final_sample["selection_stage"].map(
    selection_stage_order
)

final_sample = final_sample.sort_values(
    by=[
        "_source_order",
        "_band_order",
        "_selection_stage_order",
        prediction_count_column,
        resume_id_column,
    ]
).reset_index(drop=True)

final_sample["validation_order"] = range(
    1,
    len(final_sample) + 1,
)


# ---------------------------------------------------------
# Select and organize output columns
# ---------------------------------------------------------

preferred_columns = [
    "validation_order",
    resume_id_column,
    "dataset_group",
    "prediction_band",
    "selection_stage",
    "text_character_count",
    "total_predictions",
    "person_predictions",
    "org_predictions",
    "date_predictions",
    "gpe_predictions",
    "skill_predictions",
]

output_columns = [
    column for column in preferred_columns if column in final_sample.columns
]

final_sample = final_sample[output_columns]


# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

final_sample.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# Print final summary
# ---------------------------------------------------------

print("\nValidation sample created successfully.")

print(f"\nOutput file:\n{OUTPUT_FILE}")

print("\nFinal distribution by source and prediction band:")

final_distribution = (
    final_sample.groupby(
        [
            "dataset_group",
            "prediction_band",
        ]
    )
    .size()
    .unstack(fill_value=0)
    .reindex(
        index=[
            "Jithin",
            "Saugata",
            "Snehaan",
        ],
        columns=[
            "low",
            "medium",
            "high",
        ],
        fill_value=0,
    )
)

final_distribution["total"] = final_distribution.sum(axis=1)

print(final_distribution)

print("\nSelection stages:")

print(final_sample["selection_stage"].value_counts())

print("\nAll validation checks passed.")
