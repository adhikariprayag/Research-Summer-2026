from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Project paths and experiment settings
# ---------------------------------------------------------------------------

# This script is intended to live in:
# ATS_NER_Resume_Research/05_gpt_reconstruction/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_JSON_ROOT = (
    PROJECT_ROOT / "02_dataset" / "large_scale_dataset" / "json_records_private"
)

DEFAULT_V4_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "generalized_v4_system_entities_1020.csv"
)

DEFAULT_GPT_FILE = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "full_1020_outputs"
    / "gpt_skill_recovery_1020_combined.csv"
)

DEFAULT_PRIVATE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "full_1020_hybrid"
)

DEFAULT_SUMMARY_OUTPUT_DIR = PROJECT_ROOT / "07_results" / "large_scale_full_hybrid"

EXPECTED_CORPUS_SIZE = 1020
EXPECTED_GPT_SKILL_ROWS = 20756

DATASET_PREFIXES = {
    "JIT_CS_": "Jithin",
    "SAU_CS_": "Saugata",
    "SNE_IT_": "Snehaan",
}

EXPECTED_DATASET_COUNTS = {
    "Jithin": 400,
    "Saugata": 500,
    "Snehaan": 120,
}

PREDICTION_COLUMNS = {
    "resume_id",
    "entity_text",
    "entity_type",
}

HYBRID_COLUMNS = [
    "corpus_order",
    "resume_id",
    "dataset_group",
    "entity_text",
    "entity_type",
    "prediction_source",
    "v4_entity_text",
    "gpt_entity_text",
    "norm",
]

PER_RESUME_COLUMNS = [
    "corpus_order",
    "resume_id",
    "dataset_group",
    "v4_only_count",
    "gpt_only_count",
    "both_count",
    "hybrid_total",
]


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine generalized V4 and GPT SKILL predictions for the full "
            "1,020-resume corpus using the same normalized exact-deduplication "
            "logic as the validated 50-resume raw hybrid."
        )
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate the corpus and both prediction files without creating "
            "or changing output files."
        ),
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=DEFAULT_JSON_ROOT,
        help="Private folder containing the 1,020 resume JSON records.",
    )
    parser.add_argument(
        "--v4-file",
        type=Path,
        default=DEFAULT_V4_FILE,
        help="Full-corpus generalized V4 entity CSV.",
    )
    parser.add_argument(
        "--gpt-file",
        type=Path,
        default=DEFAULT_GPT_FILE,
        help="Completed full-corpus GPT SKILL CSV.",
    )
    parser.add_argument(
        "--private-output-dir",
        type=Path,
        default=DEFAULT_PRIVATE_OUTPUT_DIR,
        help="Private destination for detailed hybrid and per-resume outputs.",
    )
    parser.add_argument(
        "--summary-output-dir",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_DIR,
        help="Destination for privacy-safe aggregate summaries.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_CORPUS_SIZE,
        help=(
            "Expected corpus size. Keep the default for the real experiment; "
            "a smaller value is only for isolated fixture tests."
        ),
    )
    parser.add_argument(
        "--expected-gpt-rows",
        type=int,
        default=EXPECTED_GPT_SKILL_ROWS,
        help=(
            "Expected number of rows in the completed GPT file. Keep 20,756 "
            "for the confirmed full run; override only for fixture testing."
        ),
    )
    parser.add_argument(
        "--top-skills",
        type=int,
        default=25,
        help="Number of aggregate skills to report per dataset and combined.",
    )

    args = parser.parse_args()

    if args.expected_count <= 0:
        parser.error("--expected-count must be positive.")

    if args.expected_gpt_rows < 0:
        parser.error("--expected-gpt-rows cannot be negative.")

    if args.top_skills <= 0:
        parser.error("--top-skills must be positive.")

    return args


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def normalize(text: object) -> str:
    """
    Match the normalized exact-deduplication rule used by the 50-resume script.

    The method intentionally changes only case and repeated whitespace. It does
    not perform fuzzy matching, abbreviation expansion, or punctuation removal.
    """

    value = str(text).strip().casefold()
    return re.sub(r"\s+", " ", value)


def clean_display_text(text: object) -> str:
    value = str(text).strip()
    return re.sub(r"\s+", " ", value)


def infer_dataset_group(resume_id: str) -> str:
    for prefix, dataset_group in DATASET_PREFIXES.items():
        if resume_id.startswith(prefix):
            return dataset_group

    raise ValueError(f"Resume ID does not match a known dataset prefix: {resume_id}")


def atomic_write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found:\n{path}")


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    label: str,
) -> None:
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"{label} is missing columns: {sorted(missing_columns)}")


def make_corpus_metadata(
    json_root: Path,
    expected_count: int,
) -> pd.DataFrame:
    if not json_root.is_dir():
        raise FileNotFoundError(
            f"Private JSON corpus folder was not found:\n{json_root}"
        )

    resume_ids: list[str] = []
    duplicate_ids: list[str] = []
    seen: set[str] = set()

    for json_path in sorted(json_root.rglob("*.json")):
        resume_id = json_path.stem.strip()

        if resume_id in seen:
            duplicate_ids.append(resume_id)
            continue

        seen.add(resume_id)
        resume_ids.append(resume_id)

    if duplicate_ids:
        raise ValueError(
            "Duplicate resume JSON IDs were found:\n"
            + "\n".join(sorted(set(duplicate_ids)))
        )

    if len(resume_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} JSON resumes, but found " f"{len(resume_ids)}."
        )

    metadata = pd.DataFrame(
        {
            "resume_id": sorted(resume_ids),
        }
    )
    metadata.insert(
        0,
        "corpus_order",
        range(1, len(metadata) + 1),
    )
    metadata["dataset_group"] = metadata["resume_id"].map(infer_dataset_group)

    if expected_count == EXPECTED_CORPUS_SIZE:
        actual_counts = metadata["dataset_group"].value_counts().to_dict()
        count_errors = []

        for dataset_group, expected_dataset_count in EXPECTED_DATASET_COUNTS.items():
            actual_dataset_count = int(actual_counts.get(dataset_group, 0))

            if actual_dataset_count != expected_dataset_count:
                count_errors.append(
                    f"{dataset_group}: expected {expected_dataset_count}, "
                    f"found {actual_dataset_count}"
                )

        if count_errors:
            raise ValueError(
                "Dataset-source count validation failed:\n" + "\n".join(count_errors)
            )

    return metadata


def prepare_prediction_file(
    dataframe: pd.DataFrame,
    label: str,
    valid_resume_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    validate_required_columns(
        dataframe,
        PREDICTION_COLUMNS,
        label,
    )

    dataframe = dataframe.copy()
    dataframe["resume_id"] = dataframe["resume_id"].astype(str).str.strip()
    dataframe["entity_type"] = (
        dataframe["entity_type"].astype(str).str.strip().str.upper()
    )

    unknown_ids = sorted(set(dataframe["resume_id"]) - valid_resume_ids)

    if unknown_ids:
        preview = "\n".join(unknown_ids[:20])
        suffix = (
            f"\n... and {len(unknown_ids) - 20} more" if len(unknown_ids) > 20 else ""
        )
        raise ValueError(
            f"{label} contains resume IDs outside the validated corpus:\n"
            f"{preview}{suffix}"
        )

    input_rows = len(dataframe)
    non_skill_rows = int((dataframe["entity_type"] != "SKILL").sum())

    dataframe = dataframe[dataframe["entity_type"] == "SKILL"].copy()
    skill_rows = len(dataframe)

    dataframe["entity_text"] = dataframe["entity_text"].map(clean_display_text)
    dataframe["norm"] = dataframe["entity_text"].map(normalize)

    empty_rows = int((dataframe["norm"] == "").sum())
    dataframe = dataframe[dataframe["norm"] != ""].copy()

    before_deduplication = len(dataframe)
    dataframe = dataframe.drop_duplicates(
        subset=[
            "resume_id",
            "norm",
        ],
        keep="first",
    ).reset_index(drop=True)
    duplicate_rows = before_deduplication - len(dataframe)

    stats = {
        "input_rows": input_rows,
        "non_skill_rows_removed": non_skill_rows,
        "skill_rows": skill_rows,
        "empty_skill_rows_removed": empty_rows,
        "duplicate_skill_rows_removed": duplicate_rows,
        "unique_skill_rows": len(dataframe),
        "resumes_with_predictions": dataframe["resume_id"].nunique(),
    }

    return dataframe, stats


def print_preparation_stats(
    label: str,
    stats: dict[str, int],
) -> None:
    print(f"\n{label}:")
    print(f"  Input rows: {stats['input_rows']:,}")
    print("  Non-SKILL rows removed: " f"{stats['non_skill_rows_removed']:,}")
    print(f"  SKILL rows before cleanup: {stats['skill_rows']:,}")
    print("  Empty SKILL rows removed: " f"{stats['empty_skill_rows_removed']:,}")
    print(
        "  Normalized duplicates removed: " f"{stats['duplicate_skill_rows_removed']:,}"
    )
    print("  Unique SKILL rows retained: " f"{stats['unique_skill_rows']:,}")
    print(
        "  Resumes with at least one prediction: "
        f"{stats['resumes_with_predictions']:,}"
    )


def choose_display_text(group: pd.DataFrame) -> str:
    """
    Preserve the validated 50-resume display rule.

    Prefer GPT wording when GPT found the normalized skill. Otherwise use V4.
    """

    gpt_rows = group[group["prediction_system"] == "GPT"]

    if not gpt_rows.empty:
        return clean_display_text(gpt_rows.iloc[0]["entity_text"])

    return clean_display_text(group.iloc[0]["entity_text"])


def determine_prediction_source(group: pd.DataFrame) -> str:
    systems = set(group["prediction_system"].astype(str).str.strip())

    if systems == {"V4", "GPT"}:
        return "both"

    if systems == {"GPT"}:
        return "GPT"

    if systems == {"V4"}:
        return "V4"

    raise ValueError(f"Unexpected prediction-system combination: {sorted(systems)}")


def get_source_text(
    group: pd.DataFrame,
    system_name: str,
) -> str:
    system_rows = group[group["prediction_system"] == system_name]

    if system_rows.empty:
        return ""

    return clean_display_text(system_rows.iloc[0]["entity_text"])


# ---------------------------------------------------------------------------
# Hybrid creation and summaries
# ---------------------------------------------------------------------------


def create_hybrid(
    metadata: pd.DataFrame,
    v4: pd.DataFrame,
    gpt: pd.DataFrame,
) -> pd.DataFrame:
    v4 = v4.copy()
    gpt = gpt.copy()
    v4["prediction_system"] = "V4"
    gpt["prediction_system"] = "GPT"

    combined = pd.concat(
        [
            v4[
                [
                    "resume_id",
                    "entity_text",
                    "norm",
                    "prediction_system",
                ]
            ],
            gpt[
                [
                    "resume_id",
                    "entity_text",
                    "norm",
                    "prediction_system",
                ]
            ],
        ],
        ignore_index=True,
    )

    hybrid_rows = []

    for (resume_id, normalized_skill), group in combined.groupby(
        [
            "resume_id",
            "norm",
        ],
        sort=False,
    ):
        hybrid_rows.append(
            {
                "resume_id": resume_id,
                "entity_text": choose_display_text(group),
                "entity_type": "SKILL",
                "prediction_source": determine_prediction_source(group),
                "v4_entity_text": get_source_text(group, "V4"),
                "gpt_entity_text": get_source_text(group, "GPT"),
                "norm": normalized_skill,
            }
        )

    hybrid = pd.DataFrame(hybrid_rows)

    if hybrid.empty:
        hybrid = pd.DataFrame(
            columns=[
                "resume_id",
                "entity_text",
                "entity_type",
                "prediction_source",
                "v4_entity_text",
                "gpt_entity_text",
                "norm",
            ]
        )

    hybrid = hybrid.merge(
        metadata,
        on="resume_id",
        how="left",
        validate="many_to_one",
    )

    if hybrid["dataset_group"].isna().any():
        raise ValueError(
            "One or more hybrid rows could not be matched to corpus metadata."
        )

    return (
        hybrid[HYBRID_COLUMNS]
        .sort_values(
            [
                "corpus_order",
                "norm",
            ]
        )
        .reset_index(drop=True)
    )


def create_per_resume_counts(
    metadata: pd.DataFrame,
    hybrid: pd.DataFrame,
) -> pd.DataFrame:
    if hybrid.empty:
        source_counts = pd.DataFrame(
            {
                "resume_id": metadata["resume_id"],
            }
        )
    else:
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

    per_resume = metadata.merge(
        source_counts[
            [
                "resume_id",
                "v4_only_count",
                "gpt_only_count",
                "both_count",
            ]
        ],
        on="resume_id",
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        "v4_only_count",
        "gpt_only_count",
        "both_count",
    ]
    per_resume[count_columns] = per_resume[count_columns].fillna(0).astype(int)
    per_resume["hybrid_total"] = per_resume[count_columns].sum(axis=1)

    return per_resume[PER_RESUME_COLUMNS]


def dataset_sequence() -> Iterable[str]:
    yield from [
        "Jithin",
        "Saugata",
        "Snehaan",
        "Combined",
    ]


def create_source_summary(
    metadata: pd.DataFrame,
    hybrid: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []

    for dataset_group in dataset_sequence():
        if dataset_group == "Combined":
            source_metadata = metadata
            source_hybrid = hybrid
        else:
            source_metadata = metadata[metadata["dataset_group"] == dataset_group]
            source_hybrid = hybrid[hybrid["dataset_group"] == dataset_group]

        summary_rows.append(
            {
                "dataset_group": dataset_group,
                "resumes": source_metadata["resume_id"].nunique(),
                "hybrid_predictions": len(source_hybrid),
                "v4_only": int((source_hybrid["prediction_source"] == "V4").sum()),
                "gpt_only": int((source_hybrid["prediction_source"] == "GPT").sum()),
                "both": int((source_hybrid["prediction_source"] == "both").sum()),
            }
        )

    return pd.DataFrame(summary_rows)


def create_distribution_summary(
    per_resume: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []

    for dataset_group in dataset_sequence():
        if dataset_group == "Combined":
            source_df = per_resume
        else:
            source_df = per_resume[per_resume["dataset_group"] == dataset_group]

        values = source_df["hybrid_total"]

        summary_rows.append(
            {
                "dataset_group": dataset_group,
                "resumes": len(source_df),
                "total_hybrid_predictions": int(values.sum()),
                "mean_per_resume": round(float(values.mean()), 3),
                "median_per_resume": round(float(values.median()), 3),
                "minimum_per_resume": int(values.min()),
                "first_quartile": round(float(values.quantile(0.25)), 3),
                "third_quartile": round(float(values.quantile(0.75)), 3),
                "maximum_per_resume": int(values.max()),
                "resumes_with_zero_predictions": int((values == 0).sum()),
            }
        )

    return pd.DataFrame(summary_rows)


def most_frequent_display_text(group: pd.DataFrame) -> str:
    display_counts = (
        group["entity_text"]
        .value_counts()
        .rename_axis("entity_text")
        .reset_index(name="count")
        .sort_values(
            [
                "count",
                "entity_text",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )
    return str(display_counts.iloc[0]["entity_text"])


def create_top_skills_summary(
    metadata: pd.DataFrame,
    hybrid: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    summary_rows = []

    for dataset_group in dataset_sequence():
        if dataset_group == "Combined":
            source_df = hybrid
            resume_count = metadata["resume_id"].nunique()
        else:
            source_df = hybrid[hybrid["dataset_group"] == dataset_group]
            resume_count = metadata[metadata["dataset_group"] == dataset_group][
                "resume_id"
            ].nunique()

        skill_rows = []

        for normalized_skill, group in source_df.groupby(
            "norm",
            sort=False,
        ):
            skill_rows.append(
                {
                    "skill": most_frequent_display_text(group),
                    "norm": normalized_skill,
                    "resumes_with_skill": group["resume_id"].nunique(),
                }
            )

        ranked = pd.DataFrame(skill_rows)

        if ranked.empty:
            continue

        ranked = (
            ranked.sort_values(
                [
                    "resumes_with_skill",
                    "norm",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        for rank, row in enumerate(
            ranked.itertuples(index=False),
            start=1,
        ):
            summary_rows.append(
                {
                    "dataset_group": dataset_group,
                    "rank": rank,
                    "skill": row.skill,
                    "norm": row.norm,
                    "resumes_with_skill": int(row.resumes_with_skill),
                    "percentage_of_resumes": round(
                        100.0 * int(row.resumes_with_skill) / resume_count,
                        3,
                    ),
                }
            )

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    require_file(args.v4_file, "Full-corpus V4 file")
    require_file(args.gpt_file, "Completed full-corpus GPT file")

    metadata = make_corpus_metadata(
        args.json_root,
        args.expected_count,
    )
    valid_resume_ids = set(metadata["resume_id"])

    v4_raw = pd.read_csv(
        args.v4_file,
        keep_default_na=False,
    )
    gpt_raw = pd.read_csv(
        args.gpt_file,
        keep_default_na=False,
    )

    if len(gpt_raw) != args.expected_gpt_rows:
        raise ValueError(
            "The GPT file row count does not match the confirmed completed "
            f"run: expected {args.expected_gpt_rows:,}, found "
            f"{len(gpt_raw):,}."
        )

    v4, v4_stats = prepare_prediction_file(
        v4_raw,
        "Generalized V4 file",
        valid_resume_ids,
    )
    gpt, gpt_stats = prepare_prediction_file(
        gpt_raw,
        "Full-corpus GPT file",
        valid_resume_ids,
    )

    print("FULL 1,020-RESUME RAW-HYBRID PREFLIGHT")
    print(f"Corpus records validated: {len(metadata):,}")
    print(
        "Dataset distribution: "
        + ", ".join(
            f"{name} " f"{int((metadata['dataset_group'] == name).sum()):,}"
            for name in [
                "Jithin",
                "Saugata",
                "Snehaan",
            ]
        )
    )
    print_preparation_stats("Generalized V4", v4_stats)
    print_preparation_stats("GPT", gpt_stats)

    if args.check_only:
        print("\nREAD-ONLY PREFLIGHT COMPLETE")
        print("No output files or directories were changed.")
        return

    hybrid = create_hybrid(
        metadata,
        v4,
        gpt,
    )
    per_resume = create_per_resume_counts(
        metadata,
        hybrid,
    )
    source_summary = create_source_summary(
        metadata,
        hybrid,
    )
    distribution_summary = create_distribution_summary(per_resume)
    top_skills_summary = create_top_skills_summary(
        metadata,
        hybrid,
        args.top_skills,
    )

    args.private_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.summary_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hybrid_output = args.private_output_dir / "hybrid_v4_gpt_skill_predictions_1020.csv"
    per_resume_output = (
        args.private_output_dir / "hybrid_v4_gpt_skill_counts_per_resume_1020.csv"
    )
    source_summary_output = (
        args.summary_output_dir / "hybrid_v4_gpt_skill_summary_1020.csv"
    )
    distribution_output = (
        args.summary_output_dir / "hybrid_v4_gpt_skill_distribution_1020.csv"
    )
    top_skills_output = (
        args.summary_output_dir / "hybrid_v4_gpt_top_skills_by_source_1020.csv"
    )

    atomic_write_csv(hybrid, hybrid_output)
    atomic_write_csv(per_resume, per_resume_output)
    atomic_write_csv(source_summary, source_summary_output)
    atomic_write_csv(distribution_summary, distribution_output)
    atomic_write_csv(top_skills_summary, top_skills_output)

    source_total = (
        hybrid["prediction_source"]
        .value_counts()
        .reindex(
            [
                "GPT",
                "V4",
                "both",
            ],
            fill_value=0,
        )
    )

    print("\nFULL 1,020-RESUME RAW HYBRID COMPLETE")
    print(f"Hybrid unique skill predictions: {len(hybrid):,}")
    print(
        "Prediction sources: "
        f"GPT-only {int(source_total['GPT']):,}, "
        f"V4-only {int(source_total['V4']):,}, "
        f"both {int(source_total['both']):,}"
    )
    print("\nSummary by dataset:")
    print(source_summary.to_string(index=False))
    print(
        "\nImportant: these are descriptive full-corpus counts, not "
        "precision, recall, or F1 scores."
    )
    print(f"\nPrivate hybrid file:\n{hybrid_output}")
    print(f"\nPrivate per-resume file:\n{per_resume_output}")
    print(f"\nAggregate source summary:\n{source_summary_output}")
    print(f"\nAggregate distribution summary:\n{distribution_output}")
    print(f"\nAggregate top-skills summary:\n{top_skills_output}")


if __name__ == "__main__":
    main()
