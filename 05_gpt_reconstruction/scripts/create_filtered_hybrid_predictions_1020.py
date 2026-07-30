from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Project paths and validated experiment settings
# ---------------------------------------------------------------------------

# This script is intended to live in:
# ATS_NER_Resume_Research/05_gpt_reconstruction/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_JSON_ROOT = (
    PROJECT_ROOT / "02_dataset" / "large_scale_dataset" / "json_records_private"
)

DEFAULT_REVIEW_FILE = (
    PROJECT_ROOT
    / "02_dataset"
    / "large_scale_dataset"
    / "full_v4_private"
    / "validation_review_50"
    / "hybrid_prediction_review_template_50.csv"
)

DEFAULT_RAW_HYBRID_FILE = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "full_1020_hybrid"
    / "hybrid_v4_gpt_skill_predictions_1020.csv"
)

DEFAULT_PRIVATE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "05_gpt_reconstruction"
    / "gpt_skill_recovery_private"
    / "full_1020_filtered_hybrid"
)

DEFAULT_SUMMARY_OUTPUT_DIR = (
    PROJECT_ROOT / "07_results" / "large_scale_full_filtered_hybrid"
)

EXPECTED_CORPUS_SIZE = 1020
EXPECTED_REVIEW_RESUMES = 50
EXPECTED_RAW_HYBRID_ROWS = 25238

EXPECTED_DATASET_COUNTS = {
    "Jithin": 400,
    "Saugata": 500,
    "Snehaan": 120,
}

EXPECTED_RAW_SOURCE_COUNTS = {
    "GPT": 15815,
    "V4": 4482,
    "both": 4941,
}

DATASET_PREFIXES = {
    "JIT_CS_": "Jithin",
    "SAU_CS_": "Saugata",
    "SNE_IT_": "Snehaan",
}

REVIEW_COLUMNS = {
    "resume_id",
    "entity_text",
    "prediction_source",
    "review_decision",
}

RAW_HYBRID_COLUMNS = {
    "corpus_order",
    "resume_id",
    "dataset_group",
    "entity_text",
    "entity_type",
    "prediction_source",
    "v4_entity_text",
    "gpt_entity_text",
    "norm",
}

FILTERED_HYBRID_COLUMNS = [
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
    "raw_hybrid_total",
    "removed_v4_only_count",
    "filtered_v4_only_count",
    "gpt_only_count",
    "both_count",
    "filtered_hybrid_total",
]


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refit the cross-validated V4-only blacklist on all 50 manually "
            "reviewed resumes, then apply that validated rule to the full "
            "1,020-resume raw hybrid for descriptive large-scale analysis."
        )
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate all inputs and preview filtering effects without "
            "creating or changing output files."
        ),
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=DEFAULT_JSON_ROOT,
        help="Private folder containing the 1,020 resume JSON records.",
    )
    parser.add_argument(
        "--review-file",
        type=Path,
        default=DEFAULT_REVIEW_FILE,
        help="Completed 50-resume manual hybrid-review CSV.",
    )
    parser.add_argument(
        "--raw-hybrid-file",
        type=Path,
        default=DEFAULT_RAW_HYBRID_FILE,
        help="Completed full-corpus raw-hybrid prediction CSV.",
    )
    parser.add_argument(
        "--private-output-dir",
        type=Path,
        default=DEFAULT_PRIVATE_OUTPUT_DIR,
        help=(
            "Private destination for detailed filtered predictions, removed "
            "rows, learned terms, and per-resume outputs."
        ),
    )
    parser.add_argument(
        "--summary-output-dir",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_DIR,
        help="Destination for privacy-safe aggregate result summaries.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_CORPUS_SIZE,
        help=(
            "Expected corpus size. Keep 1,020 for the real experiment; "
            "override only for isolated fixture tests."
        ),
    )
    parser.add_argument(
        "--expected-review-resumes",
        type=int,
        default=EXPECTED_REVIEW_RESUMES,
        help=(
            "Expected number of manually reviewed resumes. Keep 50 for the "
            "real experiment; override only for fixture tests."
        ),
    )
    parser.add_argument(
        "--expected-raw-rows",
        type=int,
        default=EXPECTED_RAW_HYBRID_ROWS,
        help=(
            "Expected raw-hybrid row count. Keep 25,238 for the confirmed "
            "full run; override only for fixture tests."
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

    if args.expected_review_resumes <= 0:
        parser.error("--expected-review-resumes must be positive.")

    if args.expected_raw_rows < 0:
        parser.error("--expected-raw-rows cannot be negative.")

    if args.top_skills <= 0:
        parser.error("--top-skills must be positive.")

    return args


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def normalize(text: object) -> str:
    """
    Preserve the validated 50-resume normalization rule exactly.

    Only leading/trailing whitespace, case, and repeated internal whitespace
    are normalized. No punctuation removal, stemming, fuzzy matching, or
    abbreviation expansion is performed.
    """

    value = str(text).strip().lower()
    return re.sub(r"\s+", " ", value)


def normalize_raw_hybrid_key(text: object) -> str:
    """
    Reproduce the raw-hybrid builder's stored deduplication key.

    The 50-resume filtering script used lowercasing, while the confirmed
    1,020-resume raw-hybrid builder stored casefolded ``norm`` keys. Keeping
    these checks separate preserves both methods exactly.
    """

    value = str(text).strip().casefold()
    return re.sub(r"\s+", " ", value)


def clean_display_text(text: object) -> str:
    value = str(text).strip()
    return re.sub(r"\s+", " ", value)


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


def infer_dataset_group(resume_id: str) -> str:
    for prefix, dataset_group in DATASET_PREFIXES.items():
        if resume_id.startswith(prefix):
            return dataset_group

    raise ValueError(f"Resume ID does not match a known dataset prefix: {resume_id}")


def dataset_sequence() -> Iterable[str]:
    yield from [
        "Jithin",
        "Saugata",
        "Snehaan",
        "Combined",
    ]


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

    metadata = pd.DataFrame({"resume_id": sorted(resume_ids)})
    metadata.insert(
        0,
        "corpus_order",
        range(1, len(metadata) + 1),
    )
    metadata["dataset_group"] = metadata["resume_id"].map(infer_dataset_group)

    if expected_count == EXPECTED_CORPUS_SIZE:
        actual_counts = metadata["dataset_group"].value_counts().to_dict()
        errors = []

        for dataset_group, expected_dataset_count in EXPECTED_DATASET_COUNTS.items():
            actual_dataset_count = int(actual_counts.get(dataset_group, 0))

            if actual_dataset_count != expected_dataset_count:
                errors.append(
                    f"{dataset_group}: expected {expected_dataset_count}, "
                    f"found {actual_dataset_count}"
                )

        if errors:
            raise ValueError(
                "Dataset-source count validation failed:\n" + "\n".join(errors)
            )

    return metadata


# ---------------------------------------------------------------------------
# Learn the final deployment blacklist from all 50 reviewed resumes
# ---------------------------------------------------------------------------


def prepare_review_file(
    review: pd.DataFrame,
    expected_review_resumes: int,
) -> pd.DataFrame:
    validate_required_columns(
        review,
        REVIEW_COLUMNS,
        "Manual review file",
    )

    review = review.copy()
    review["resume_id"] = review["resume_id"].fillna("").astype(str).str.strip()
    review["entity_text"] = review["entity_text"].map(clean_display_text)
    review["entity_norm"] = review["entity_text"].map(normalize)
    review["source_clean"] = (
        review["prediction_source"].fillna("").astype(str).map(normalize)
    )
    review["decision_clean"] = (
        review["review_decision"].fillna("").astype(str).map(normalize)
    )

    empty_resume_ids = int(review["resume_id"].eq("").sum())
    empty_entities = int(review["entity_norm"].eq("").sum())

    if empty_resume_ids:
        raise ValueError(
            f"Manual review file has {empty_resume_ids} rows with an empty "
            "resume_id."
        )

    if empty_entities:
        raise ValueError(
            f"Manual review file has {empty_entities} rows with empty " "entity_text."
        )

    valid_sources = {"v4", "gpt", "both"}
    invalid_sources = sorted(set(review["source_clean"]) - valid_sources)

    if invalid_sources:
        raise ValueError(
            f"Manual review file has invalid prediction sources: " f"{invalid_sources}"
        )

    valid_decisions = {"correct", "incorrect", "partial"}
    invalid_decisions = sorted(set(review["decision_clean"]) - valid_decisions)

    if invalid_decisions:
        raise ValueError(
            f"Manual review file has invalid review decisions: " f"{invalid_decisions}"
        )

    actual_review_resumes = review["resume_id"].nunique()

    if actual_review_resumes != expected_review_resumes:
        raise ValueError(
            f"Expected {expected_review_resumes} manually reviewed resumes, "
            f"but found {actual_review_resumes}."
        )

    duplicate_mask = review.duplicated(
        subset=[
            "resume_id",
            "entity_norm",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_preview = (
            review.loc[
                duplicate_mask,
                [
                    "resume_id",
                    "entity_text",
                    "prediction_source",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )
        raise ValueError(
            "Manual review file contains duplicate normalized predictions "
            "within a resume. This would distort the learned term counts:\n"
            f"{duplicate_preview}"
        )

    return review


def learn_deployment_blacklist(
    review: pd.DataFrame,
) -> pd.DataFrame:
    """
    Refit the exact cross-validated filtering rule on all reviewed data.

    As in evaluate_filtered_hybrid_cv_50.py:
    - learn from V4-only predictions;
    - require at least two reviewed examples of a normalized term;
    - blacklist only when incorrect examples outnumber valid examples;
    - count both "correct" and "partial" as valid when learning the filter.

    Cross-validation remains the unbiased performance estimate. This all-data
    refit is used only to deploy the already-validated rule on the separate
    1,020-resume corpus.
    """

    v4_only = review[review["source_clean"].eq("v4")].copy()

    term_quality = v4_only.groupby(
        "entity_norm",
        as_index=False,
    ).agg(
        training_total=("decision_clean", "size"),
        training_valid=(
            "decision_clean",
            lambda values: int(values.isin({"correct", "partial"}).sum()),
        ),
        training_incorrect=(
            "decision_clean",
            lambda values: int(values.eq("incorrect").sum()),
        ),
        reviewed_resumes=("resume_id", "nunique"),
    )

    term_quality["meets_minimum_frequency"] = term_quality["training_total"].ge(2)
    term_quality["incorrect_outnumbers_valid"] = term_quality["training_incorrect"].gt(
        term_quality["training_valid"]
    )
    term_quality["blacklisted"] = (
        term_quality["meets_minimum_frequency"]
        & term_quality["incorrect_outnumbers_valid"]
    )

    return term_quality.sort_values(
        [
            "blacklisted",
            "training_incorrect",
            "training_total",
            "entity_norm",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validate and filter the full raw hybrid
# ---------------------------------------------------------------------------


def prepare_raw_hybrid(
    raw: pd.DataFrame,
    metadata: pd.DataFrame,
    expected_raw_rows: int,
) -> pd.DataFrame:
    validate_required_columns(
        raw,
        RAW_HYBRID_COLUMNS,
        "Full raw-hybrid file",
    )

    if len(raw) != expected_raw_rows:
        raise ValueError(
            "The raw-hybrid file row count does not match the confirmed "
            f"full run: expected {expected_raw_rows:,}, found "
            f"{len(raw):,}."
        )

    raw = raw.copy()
    raw["resume_id"] = raw["resume_id"].fillna("").astype(str).str.strip()
    raw["dataset_group"] = raw["dataset_group"].fillna("").astype(str).str.strip()
    raw["entity_type"] = (
        raw["entity_type"].fillna("").astype(str).str.strip().str.upper()
    )
    raw["entity_text"] = raw["entity_text"].map(clean_display_text)
    raw["prediction_source"] = (
        raw["prediction_source"].fillna("").astype(str).str.strip()
    )
    raw["source_clean"] = raw["prediction_source"].map(normalize)
    raw["norm_clean"] = raw["norm"].fillna("").astype(str).map(normalize_raw_hybrid_key)
    raw["entity_norm_check"] = raw["entity_text"].map(normalize_raw_hybrid_key)

    if raw["resume_id"].eq("").any():
        raise ValueError("Full raw-hybrid file contains empty resume IDs.")

    if raw["entity_norm_check"].eq("").any():
        raise ValueError("Full raw-hybrid file contains empty skill text.")

    if not raw["entity_type"].eq("SKILL").all():
        invalid_types = sorted(set(raw["entity_type"]) - {"SKILL"})
        raise ValueError(
            f"Full raw-hybrid file contains non-SKILL types: " f"{invalid_types}"
        )

    valid_sources = {"v4", "gpt", "both"}
    invalid_sources = sorted(set(raw["source_clean"]) - valid_sources)

    if invalid_sources:
        raise ValueError(
            f"Full raw-hybrid file has invalid prediction sources: "
            f"{invalid_sources}"
        )

    norm_mismatches = raw[raw["norm_clean"].ne(raw["entity_norm_check"])]

    if not norm_mismatches.empty:
        mismatch_preview = norm_mismatches[
            [
                "resume_id",
                "entity_text",
                "norm",
            ]
        ].head(20)
        raise ValueError(
            "Full raw-hybrid file has normalized terms that do not match "
            "entity_text:\n" + mismatch_preview.to_string(index=False)
        )

    valid_resume_ids = set(metadata["resume_id"])
    raw_resume_ids = set(raw["resume_id"])
    unknown_ids = sorted(raw_resume_ids - valid_resume_ids)

    if unknown_ids:
        raise ValueError(
            "Full raw-hybrid file contains resume IDs outside the validated "
            "corpus:\n" + "\n".join(unknown_ids[:20])
        )

    expected_metadata = metadata[
        [
            "resume_id",
            "corpus_order",
            "dataset_group",
        ]
    ].rename(
        columns={
            "corpus_order": "expected_corpus_order",
            "dataset_group": "expected_dataset_group",
        }
    )

    raw = raw.merge(
        expected_metadata,
        on="resume_id",
        how="left",
        validate="many_to_one",
    )

    corpus_order_numeric = pd.to_numeric(
        raw["corpus_order"],
        errors="coerce",
    )

    metadata_mismatch = corpus_order_numeric.ne(raw["expected_corpus_order"]) | raw[
        "dataset_group"
    ].ne(raw["expected_dataset_group"])

    if metadata_mismatch.any():
        mismatch_preview = raw.loc[
            metadata_mismatch,
            [
                "resume_id",
                "corpus_order",
                "expected_corpus_order",
                "dataset_group",
                "expected_dataset_group",
            ],
        ].head(20)
        raise ValueError(
            "Full raw-hybrid metadata does not match the validated corpus:\n"
            + mismatch_preview.to_string(index=False)
        )

    duplicate_mask = raw.duplicated(
        subset=[
            "resume_id",
            "norm_clean",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_preview = raw.loc[
            duplicate_mask,
            [
                "resume_id",
                "entity_text",
                "prediction_source",
            ],
        ].head(20)
        raise ValueError(
            "Full raw-hybrid file contains duplicate normalized skills "
            "within a resume:\n" + duplicate_preview.to_string(index=False)
        )

    if len(raw) == EXPECTED_RAW_HYBRID_ROWS and len(metadata) == EXPECTED_CORPUS_SIZE:
        actual_source_counts = raw["prediction_source"].value_counts().to_dict()
        source_errors = []

        for source_name, expected_source_count in EXPECTED_RAW_SOURCE_COUNTS.items():
            actual_source_count = int(actual_source_counts.get(source_name, 0))

            if actual_source_count != expected_source_count:
                source_errors.append(
                    f"{source_name}: expected {expected_source_count:,}, "
                    f"found {actual_source_count:,}"
                )

        if source_errors:
            raise ValueError(
                "Raw-hybrid source-count validation failed:\n"
                + "\n".join(source_errors)
            )

    raw["corpus_order"] = corpus_order_numeric.astype(int)
    raw["norm"] = raw["norm_clean"]

    return raw.drop(
        columns=[
            "source_clean",
            "norm_clean",
            "entity_norm_check",
            "expected_corpus_order",
            "expected_dataset_group",
        ]
    )


def filter_raw_hybrid(
    raw: pd.DataFrame,
    blacklist_terms: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_clean = raw["prediction_source"].map(normalize)
    # Match the validated CV filter's lowercased entity-text normalization.
    norm_clean = raw["entity_text"].map(normalize)

    remove_mask = source_clean.eq("v4") & norm_clean.isin(blacklist_terms)

    filtered = raw.loc[~remove_mask].copy()
    removed = raw.loc[remove_mask].copy()

    # Method guard: the validated filter may remove V4-only rows only.
    if not removed.empty:
        removed_sources = set(removed["prediction_source"].map(normalize))

        if removed_sources != {"v4"}:
            raise AssertionError("Filtering removed a GPT or both-system prediction.")

    expected_filtered_rows = len(raw) - len(removed)

    if len(filtered) != expected_filtered_rows:
        raise AssertionError("Filtered row count does not reconcile with removed rows.")

    filtered = (
        filtered[FILTERED_HYBRID_COLUMNS]
        .sort_values(
            [
                "corpus_order",
                "norm",
            ]
        )
        .reset_index(drop=True)
    )

    removed = (
        removed[FILTERED_HYBRID_COLUMNS]
        .sort_values(
            [
                "corpus_order",
                "norm",
            ]
        )
        .reset_index(drop=True)
    )

    return filtered, removed


# ---------------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------------


def create_per_resume_counts(
    metadata: pd.DataFrame,
    raw: pd.DataFrame,
    filtered: pd.DataFrame,
    removed: pd.DataFrame,
) -> pd.DataFrame:
    raw_counts = (
        raw.groupby("resume_id").size().rename("raw_hybrid_total").reset_index()
    )

    removed_counts = (
        removed.groupby("resume_id")
        .size()
        .rename("removed_v4_only_count")
        .reset_index()
    )

    if filtered.empty:
        source_counts = pd.DataFrame(
            columns=[
                "resume_id",
                "V4",
                "GPT",
                "both",
            ]
        )
    else:
        source_counts = (
            filtered.groupby(
                [
                    "resume_id",
                    "prediction_source",
                ]
            )
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

    for source_name in ["V4", "GPT", "both"]:
        if source_name not in source_counts.columns:
            source_counts[source_name] = 0

    source_counts = source_counts.rename(
        columns={
            "V4": "filtered_v4_only_count",
            "GPT": "gpt_only_count",
            "both": "both_count",
        }
    )

    per_resume = (
        metadata.merge(
            raw_counts,
            on="resume_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            removed_counts,
            on="resume_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            source_counts[
                [
                    "resume_id",
                    "filtered_v4_only_count",
                    "gpt_only_count",
                    "both_count",
                ]
            ],
            on="resume_id",
            how="left",
            validate="one_to_one",
        )
    )

    count_columns = [
        "raw_hybrid_total",
        "removed_v4_only_count",
        "filtered_v4_only_count",
        "gpt_only_count",
        "both_count",
    ]
    per_resume[count_columns] = per_resume[count_columns].fillna(0).astype(int)
    per_resume["filtered_hybrid_total"] = per_resume[
        [
            "filtered_v4_only_count",
            "gpt_only_count",
            "both_count",
        ]
    ].sum(axis=1)

    reconciliation_ok = (
        per_resume["raw_hybrid_total"]
        .eq(per_resume["removed_v4_only_count"] + per_resume["filtered_hybrid_total"])
        .all()
    )

    if not reconciliation_ok:
        raise AssertionError(
            "Per-resume raw, removed, and filtered counts do not reconcile."
        )

    return per_resume[PER_RESUME_COLUMNS]


def create_source_summary(
    metadata: pd.DataFrame,
    raw: pd.DataFrame,
    filtered: pd.DataFrame,
    removed: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dataset_group in dataset_sequence():
        if dataset_group == "Combined":
            source_metadata = metadata
            source_raw = raw
            source_filtered = filtered
            source_removed = removed
        else:
            source_metadata = metadata[metadata["dataset_group"].eq(dataset_group)]
            source_raw = raw[raw["dataset_group"].eq(dataset_group)]
            source_filtered = filtered[filtered["dataset_group"].eq(dataset_group)]
            source_removed = removed[removed["dataset_group"].eq(dataset_group)]

        rows.append(
            {
                "dataset_group": dataset_group,
                "resumes": source_metadata["resume_id"].nunique(),
                "raw_hybrid_predictions": len(source_raw),
                "removed_v4_only": len(source_removed),
                "filtered_hybrid_predictions": len(source_filtered),
                "filtered_v4_only": int(
                    source_filtered["prediction_source"].eq("V4").sum()
                ),
                "gpt_only": int(source_filtered["prediction_source"].eq("GPT").sum()),
                "both": int(source_filtered["prediction_source"].eq("both").sum()),
                "percentage_raw_removed": (
                    round(
                        100.0 * len(source_removed) / len(source_raw),
                        3,
                    )
                    if len(source_raw)
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def create_distribution_summary(
    per_resume: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dataset_group in dataset_sequence():
        if dataset_group == "Combined":
            source_df = per_resume
        else:
            source_df = per_resume[per_resume["dataset_group"].eq(dataset_group)]

        values = source_df["filtered_hybrid_total"]

        rows.append(
            {
                "dataset_group": dataset_group,
                "resumes": len(source_df),
                "total_filtered_predictions": int(values.sum()),
                "mean_per_resume": round(float(values.mean()), 3),
                "median_per_resume": round(float(values.median()), 3),
                "minimum_per_resume": int(values.min()),
                "first_quartile": round(float(values.quantile(0.25)), 3),
                "third_quartile": round(float(values.quantile(0.75)), 3),
                "maximum_per_resume": int(values.max()),
                "resumes_with_zero_predictions": int(values.eq(0).sum()),
            }
        )

    return pd.DataFrame(rows)


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
    filtered: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    rows = []

    for dataset_group in dataset_sequence():
        if dataset_group == "Combined":
            source_df = filtered
            resume_count = metadata["resume_id"].nunique()
        else:
            source_df = filtered[filtered["dataset_group"].eq(dataset_group)]
            resume_count = metadata[metadata["dataset_group"].eq(dataset_group)][
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
            rows.append(
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

    return pd.DataFrame(rows)


def create_filter_summary(
    review: pd.DataFrame,
    term_quality: pd.DataFrame,
    raw: pd.DataFrame,
    filtered: pd.DataFrame,
    removed: pd.DataFrame,
) -> pd.DataFrame:
    blacklist_size = int(term_quality["blacklisted"].sum())
    v4_only_review_rows = int(review["source_clean"].eq("v4").sum())
    v4_only_raw_rows = int(raw["prediction_source"].map(normalize).eq("v4").sum())
    removed_resume_count = removed["resume_id"].nunique()

    return pd.DataFrame(
        [
            {
                "training_review_resumes": review["resume_id"].nunique(),
                "training_review_rows": len(review),
                "training_v4_only_rows": v4_only_review_rows,
                "training_unique_v4_only_terms": len(term_quality),
                "deployment_blacklist_terms": blacklist_size,
                "raw_hybrid_predictions": len(raw),
                "raw_v4_only_predictions": v4_only_raw_rows,
                "removed_v4_only_predictions": len(removed),
                "resumes_with_removed_predictions": removed_resume_count,
                "filtered_hybrid_predictions": len(filtered),
                "percentage_raw_predictions_removed": (
                    round(
                        100.0 * len(removed) / len(raw),
                        3,
                    )
                    if len(raw)
                    else 0.0
                ),
            }
        ]
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    require_file(args.review_file, "Completed manual review file")
    require_file(args.raw_hybrid_file, "Completed full raw-hybrid file")

    metadata = make_corpus_metadata(
        args.json_root,
        args.expected_count,
    )

    review_raw = pd.read_csv(
        args.review_file,
        keep_default_na=False,
    )
    raw_hybrid_input = pd.read_csv(
        args.raw_hybrid_file,
        keep_default_na=False,
    )

    review = prepare_review_file(
        review_raw,
        args.expected_review_resumes,
    )
    term_quality = learn_deployment_blacklist(review)
    blacklist_terms = set(
        term_quality.loc[
            term_quality["blacklisted"],
            "entity_norm",
        ]
    )
    raw_hybrid = prepare_raw_hybrid(
        raw_hybrid_input,
        metadata,
        args.expected_raw_rows,
    )
    filtered_hybrid, removed = filter_raw_hybrid(
        raw_hybrid,
        blacklist_terms,
    )

    print("FULL 1,020-RESUME FILTERED-HYBRID PREFLIGHT")
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
    print(f"Manual review resumes validated: {review['resume_id'].nunique():,}")
    print(f"Manual review prediction rows: {len(review):,}")
    print(
        "V4-only review rows used to learn the filter: "
        f"{int(review['source_clean'].eq('v4').sum()):,}"
    )
    print("Unique reviewed V4-only terms: " f"{len(term_quality):,}")
    print("Deployment blacklist terms learned: " f"{len(blacklist_terms):,}")
    print(f"Raw-hybrid rows validated: {len(raw_hybrid):,}")
    print("V4-only rows selected for removal: " f"{len(removed):,}")
    print("Filtered-hybrid rows previewed: " f"{len(filtered_hybrid):,}")
    print("GPT-only and both-system rows are always retained by this rule.")

    if args.check_only:
        print("\nREAD-ONLY PREFLIGHT COMPLETE")
        print("No output files or directories were changed.")
        return

    per_resume = create_per_resume_counts(
        metadata,
        raw_hybrid,
        filtered_hybrid,
        removed,
    )
    source_summary = create_source_summary(
        metadata,
        raw_hybrid,
        filtered_hybrid,
        removed,
    )
    distribution_summary = create_distribution_summary(per_resume)
    top_skills_summary = create_top_skills_summary(
        metadata,
        filtered_hybrid,
        args.top_skills,
    )
    filter_summary = create_filter_summary(
        review,
        term_quality,
        raw_hybrid,
        filtered_hybrid,
        removed,
    )

    args.private_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.summary_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filtered_output = (
        args.private_output_dir / "filtered_hybrid_skill_predictions_1020.csv"
    )
    removed_output = (
        args.private_output_dir / "filtered_hybrid_removed_v4_only_predictions_1020.csv"
    )
    per_resume_output = (
        args.private_output_dir / "filtered_hybrid_skill_counts_per_resume_1020.csv"
    )
    blacklist_output = (
        args.private_output_dir / "filtered_hybrid_deployment_term_quality_50.csv"
    )

    filter_summary_output = (
        args.summary_output_dir / "filtered_hybrid_filter_summary_1020.csv"
    )
    source_summary_output = (
        args.summary_output_dir / "filtered_hybrid_source_summary_1020.csv"
    )
    distribution_output = (
        args.summary_output_dir / "filtered_hybrid_distribution_1020.csv"
    )
    top_skills_output = (
        args.summary_output_dir / "filtered_hybrid_top_skills_by_source_1020.csv"
    )

    atomic_write_csv(filtered_hybrid, filtered_output)
    atomic_write_csv(removed, removed_output)
    atomic_write_csv(per_resume, per_resume_output)
    atomic_write_csv(term_quality, blacklist_output)
    atomic_write_csv(filter_summary, filter_summary_output)
    atomic_write_csv(source_summary, source_summary_output)
    atomic_write_csv(distribution_summary, distribution_output)
    atomic_write_csv(top_skills_summary, top_skills_output)

    print("\nFULL 1,020-RESUME FILTERED HYBRID COMPLETE")
    print(f"Raw-hybrid skill predictions: {len(raw_hybrid):,}")
    print(f"Deployment blacklist terms: {len(blacklist_terms):,}")
    print(f"V4-only predictions removed: {len(removed):,}")
    print("Filtered-hybrid skill predictions: " f"{len(filtered_hybrid):,}")
    print("\nSummary by dataset:")
    print(source_summary.to_string(index=False))
    print(
        "\nMethod note: 5-fold cross-validation remains the accuracy "
        "evaluation. These 1,020-resume outputs are a descriptive deployment "
        "of the validated filter refit on all 50 reviewed resumes; they are "
        "not new precision, recall, or F1 scores."
    )
    print(f"\nPrivate filtered predictions:\n{filtered_output}")
    print(f"\nPrivate removed V4-only predictions:\n{removed_output}")
    print(f"\nPrivate per-resume counts:\n{per_resume_output}")
    print(f"\nPrivate learned term-quality table:\n{blacklist_output}")
    print(f"\nAggregate filter summary:\n{filter_summary_output}")
    print(f"\nAggregate source summary:\n{source_summary_output}")
    print(f"\nAggregate distribution summary:\n{distribution_output}")
    print(f"\nAggregate top-skills summary:\n{top_skills_output}")


if __name__ == "__main__":
    main()
