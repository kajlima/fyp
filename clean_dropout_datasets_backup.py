from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def resolve_input_file(year: int) -> Path:
       candidates = [
        Path(f"dataset_{year}_hash.csv"),
    ]
    candidates.extend(sorted(Path(".").glob(f"dataset_{year}_hash*.csv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find dataset for {year}. Put dataset_{year}_hash.csv "
        f"or dataset_{year}_hash(...).csv in the working directory."
    )


INPUT_FILES: Dict[int, Path] = {
    2018: resolve_input_file(2018),
    2021: resolve_input_file(2021),
    2022: resolve_input_file(2022),
}
OUTPUT_DIR = Path("cleaned_dropout_outputs_lms_row_filtered_no_frac_keep_2018_blank_activity")

ID_COL = "dni_hash"
DEGREE_COL = "tit_hash"
COURSE_COL = "asi_hash"
TARGET_COL = "abandono_hash"
YEAR_COL = "caca"
START_YEAR_COL = "anyo_inicio_estudios"
GRADE_COL = "nota_asig_hash"

SEM_A_MONTHS = {9, 10, 11, 12, 1}
SEM_B_MONTHS = {2, 3, 4, 5, 6}

LMS_BASES = [
    "lms_events",
    "lms_assignment_submissions",
    "lms_test_submissions",
    "lms_total_minutes",
    "resource_events",
]
WIFI_BASES = ["attendance_days"]
ACTIVITY_BASES = LMS_BASES + WIFI_BASES
EVENT_BASES_TO_WINSORIZE = ["lms_events", "resource_events"]

# Columns used to decide whether a row has any LMS activity at all.
LMS_lms_ROW_PRESENCE_BASES = [
    "lms_events",
    "lms_visits",
    "lms_days_logged",
    "lms_assignment_submissions",
    "lms_test_submissions",
    "lms_total_minutes",
    "resource_events",
    "n_resource_days",
]

NON_NUMERIC_COLS = {
    ID_COL,
    DEGREE_COL,
    COURSE_COL,
    "tipo_ingreso",
    "campus_hash",
    "estudios_p_hash",
    "estudios_m_hash",
    "dedicacion",
    "desplazado_hash",
    TARGET_COL,
    "baja_fecha",
    "grupos_por_tipocredito_hash",
    "fecha_datos",
}

BASE_STUDENT_COLS = [
    TARGET_COL,
    YEAR_COL,
    "curso_mas_alto",
    "cred_mat_sem_a",
    "cred_mat_sem_b",
    "cred_mat_anu",
    "cred_mat_total",
    "cred_sup_sem_a",
    "cred_sup_sem_b",
    "cred_sup_anu",
    "cred_sup_total",
    START_YEAR_COL,
    "es_retitulado",
    "es_adaptado",
    "rend_total_ultimo",
    "rend_total_penultimo",
    "rend_total_antepenultimo",
    "_degree_size",
]

RENAME_FINAL = {
    ID_COL: "student_id_hash",
    YEAR_COL: "academic_year",
    "curso_mas_alto": "highest_course_year_enrolled",
    "cred_mat_sem_a": "credits_enrolled_semester_a",
    "cred_mat_sem_b": "credits_enrolled_semester_b",
    "cred_mat_total": "total_credits_enrolled_academic_year",
    "cred_sup_total": "total_credits_passed_academic_year",
    "rend_total_ultimo": "completion_rate_one_year_before",
    "rend_total_penultimo": "completion_rate_two_years_before",
    "rend_total_antepenultimo": "completion_rate_three_years_before",
    "es_retitulado": "obtained_new_degree_flag",
    "es_adaptado": "adapted_studies_flag",
    "_degree_size": "degree_size",
}

HISTORY_RENAME = {
    "rend_total_ultimo": "completion_rate_one_year_before",
    "rend_total_penultimo": "completion_rate_two_years_before",
    "rend_total_antepenultimo": "completion_rate_three_years_before",
}

# Raw / leakage columns that must not appear in the final output.
COLUMNS_THAT_MUST_NOT_SURVIVE = {
    DEGREE_COL,
    COURSE_COL,
    "cred_sup_tit",
    "cred_pend_sup_tit",
    "cred_sup_1o",
    "cred_sup_2o",
    "cred_sup_3o",
    "cred_sup_4o",
    "cred_sup_5o",
    "cred_sup_6o",
    "baja_fecha",
    "matricula_activa",
    "anyo_ingreso",
    "tipo_ingreso",
    "nota10_hash",
    "nota14_hash",
    "campus_hash",
    "estudios_p_hash",
    "estudios_m_hash",
    "dedicacion",
    "desplazado_hash",
    "preferencia_seleccion",
    "grupos_por_tipocredito_hash",
    "fecha_datos",
    "curso_mas_bajo",
    "cred_mat1",
    "cred_mat2",
    "cred_mat3",
    "cred_mat4",
    "cred_mat5",
    "cred_mat6",
    "cred_sup_normal",
    "cred_sup_espec",
    "cred_sup",
    "cred_mat_normal",
    "cred_mat_movilidad",
    "cred_ptes_acta",
    "cred_mat_practicas",
    "cred_mat_anu",
    "cred_sup_sem_a",
    "cred_sup_sem_b",
    "cred_sup_anu",
    "rendimiento_cuat_a",
    "rendimiento_cuat_b",
    "rendimiento_total",
    "exento_npp",
    START_YEAR_COL,
    "practicas",
    "actividades",
    "ajuste",
    "impagado_curso_mat",
    "asig1",
    "pract1",
    "activ1",
    "total1",
    "ajuste1",
    "lms_days_logged",
    "lms_visits",
    "n_resource_days",
}

DROP_FINAL_COLUMNS = {
    "frac_courses_with_lms",
    "frac_courses_without_grade",
}

# Feature sets for modelling.
NON_FEATURE_COLS = ["student_id_hash", "academic_year", "target"]

EARLY_FEATURES = [
    "degree_size",
    "seniority",
    "highest_course_year_enrolled",
    "adapted_studies_flag",
    "credits_enrolled_semester_a",
    "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year",
    "total_credits_enrolled_impossible_flag",
    "completion_rate_one_year_before",
    "completion_rate_two_years_before",
    "completion_rate_three_years_before",
    "completion_rate_one_year_before_missing_flag",
    "completion_rate_two_years_before_missing_flag",
    "completion_rate_three_years_before_missing_flag",
    "pass_ratio_sem_a",
    "pass_ratio_sem_a_and_zero_flag",
    "lms_events_sem_a",
    "lms_assignment_submissions_sem_a",
    "lms_test_submissions_sem_a",
    "lms_total_minutes_sem_a",
    "resource_events_sem_a",
    "attendance_days_sem_a",
    ]

FULL_ONLY_FEATURES = [
    "n_courses",
    "courses_mean",
    "total_credits_passed_academic_year",
    "pass_ratio_sem_b",
    "pass_ratio_sem_b_and_zero_flag",
    "total_performance",
    "total_performance_and_zero_flag",
    "obtained_new_degree_flag",
    "lms_events_sem_b",
    "lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_b",
    "lms_total_minutes_sem_b",
    "resource_events_sem_b",
    "attendance_days_sem_b",
]
FULL_FEATURES = EARLY_FEATURES + FULL_ONLY_FEATURES


def validate_feature_lists(final: pd.DataFrame, year: int, report: Dict[str, object]) -> List[str]:
    """Check every output column is either excluded or listed in FULL_FEATURES."""
    warnings: List[str] = []
    produced = set(final.columns)

    missing_early = [c for c in EARLY_FEATURES if c not in produced]
    missing_full = [c for c in FULL_FEATURES if c not in produced]
    if missing_early:
        warnings.append(f"{year}: EARLY_FEATURES missing from output: {missing_early}")
    if missing_full:
        warnings.append(f"{year}: FULL_FEATURES missing from output: {missing_full}")
    if not set(EARLY_FEATURES).issubset(set(FULL_FEATURES)):
        warnings.append(f"{year}: EARLY_FEATURES is not a subset of FULL_FEATURES")

    orphan = [
        c for c in produced
        if c not in set(FULL_FEATURES) | set(NON_FEATURE_COLS)
    ]
    if orphan:
        warnings.append(
            f"{year}: output columns assigned to NEITHER list nor excluded: {sorted(orphan)}"
        )

    report["n_early_features"] = len(EARLY_FEATURES)
    report["n_full_features"] = len(FULL_FEATURES)
    report["feature_list_warnings"] = warnings
    return warnings


def read_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        na_values=["", "NA", "NaN", "nan", "NULL", "null"],
        keep_default_na=True,
        low_memory=False,
    )
    # Force everything except the known text columns to numeric.
    for col in df.columns:
        if col not in NON_NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def mode_or_first(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    modes = non_null.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return non_null.iloc[0]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> Tuple[pd.Series, pd.Series]:
    # Ratio in %, with 0 where the denominator is missing or <= 0 (flagged).
    denom_bad = denominator.isna() | denominator.le(0)
    result = pd.Series(
        np.zeros(len(numerator), dtype="float64"),
        index=numerator.index,
    )
    ok = ~denom_bad
    result.loc[ok] = (numerator.loc[ok].fillna(0) / denominator.loc[ok]) * 100
    return result, denom_bad.astype("int8")


def parse_month_col(col: str) -> Tuple[str, int, int] | None:
    # Parse names like lms_events_2018_9 -> (base, year, month).
    match = re.match(r"^(?P<base>.+)_(?P<year>20\d{2})_(?P<month>\d{1,2})$", col)
    if not match:
        return None
    return match.group("base"), int(match.group("year")), int(match.group("month"))


def monthly_columns(
    df: pd.DataFrame,
    base: str,
    months: Iterable[int] | None = None,
) -> List[str]:
    cols: List[str] = []
    months_set = set(months) if months is not None else None
    for col in df.columns:
        parsed = parse_month_col(col)
        if parsed is None:
            continue
        parsed_base, _, parsed_month = parsed
        if parsed_base == base and (months_set is None or parsed_month in months_set):
            cols.append(col)
    return cols


def winsorize_columns_p99(df: pd.DataFrame, cols: List[str]) -> Dict[str, float | None]:
    caps: Dict[str, float | None] = {}
    for col in cols:
        if col not in df.columns:
            continue
        cap = df[col].quantile(0.99)
        if pd.isna(cap):
            caps[col] = None
            continue
        df[col] = df[col].clip(upper=cap)
        caps[col] = float(cap)
    return caps


def count_true(series: pd.Series) -> int:
    return int(series.fillna(False).astype(bool).sum())


def target_counts_by_student(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty or ID_COL not in df.columns or TARGET_COL not in df.columns:
        return {"dropout": 0, "non-dropout": 0}
    target_map = {
        "A": "dropout",
        "B": "non-dropout",
        "a": "dropout",
        "b": "non-dropout",
    }
    student_targets = (
        df.groupby(ID_COL, dropna=False)[TARGET_COL]
        .agg(mode_or_first)
        .map(target_map)
    )
    counts = student_targets.value_counts(dropna=False).to_dict()
    return {
        "dropout": int(counts.get("dropout", 0)),
        "non-dropout": int(counts.get("non-dropout", 0)),
    }


def lms_lms_presence_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for base in LMS_lms_ROW_PRESENCE_BASES:
        cols.extend(monthly_columns(df, base))
    wanted = set(cols)
    return [col for col in df.columns if col in wanted]


def drop_rows_without_any_lms_lms_activity(
    df: pd.DataFrame,
    report: Dict[str, object],
) -> pd.DataFrame:
    """Drop rows whose LMS monthly activity is entirely blank (a 0 counts as real)"""
    activity_cols = lms_lms_presence_columns(df)
    report["lms_lms_row_presence_columns_used"] = activity_cols
    report["lms_lms_row_presence_n_columns_used"] = int(len(activity_cols))
    if not activity_cols:
        raise ValueError("No LMS/lms monthly activity columns were found for row-level filtering.")

    rows_before = len(df)
    students_before = df[ID_COL].nunique(dropna=True)
    courses_before = df[COURSE_COL].nunique(dropna=True)
    target_counts_before = target_counts_by_student(df)

    keep_mask = df[activity_cols].notna().any(axis=1)
    removed_df = df.loc[~keep_mask].copy()
    kept_df = df.loc[keep_mask].copy()

    student_ids_before = set(df[ID_COL].dropna().unique())
    student_ids_after = set(kept_df[ID_COL].dropna().unique())
    fully_removed_student_ids = student_ids_before - student_ids_after
    fully_removed_students_df = df.loc[df[ID_COL].isin(fully_removed_student_ids)].copy()

    course_ids_before = set(df[COURSE_COL].dropna().unique())
    course_ids_after = set(kept_df[COURSE_COL].dropna().unique())

    report["rows_before_lms_lms_row_drop"] = int(rows_before)
    report["dropped_rows_blank_lms_lms_activity"] = int(rows_before - len(kept_df))
    report["rows_after_lms_lms_row_drop"] = int(len(kept_df))
    report["students_before_lms_lms_row_drop"] = int(students_before)
    report["students_after_lms_lms_row_drop"] = int(len(student_ids_after))
    report["students_removed_by_lms_lms_row_drop"] = int(len(fully_removed_student_ids))
    report["courses_before_lms_lms_row_drop"] = int(courses_before)
    report["courses_after_lms_lms_row_drop"] = int(len(course_ids_after))
    report["courses_removed_entirely_by_lms_lms_row_drop"] = int(len(course_ids_before - course_ids_after))
    report["student_target_counts_before_lms_lms_row_drop"] = target_counts_before
    report["student_target_counts_removed_by_lms_lms_row_drop"] = target_counts_by_student(
        fully_removed_students_df
    )
    report["student_target_counts_after_lms_lms_row_drop"] = target_counts_by_student(kept_df)
    report["students_with_at_least_one_dropped_blank_lms_lms_row"] = int(
        removed_df[ID_COL].nunique(dropna=True)
    )

    if kept_df.empty:
        raise ValueError("All rows were removed by the LMS/lms row-level filter.")
    return kept_df


def make_degree_size(df: pd.DataFrame) -> pd.Series:
    degree_total = df["cred_sup_tit"] + df["cred_pend_sup_tit"]
    tmp = pd.DataFrame(
        {
            DEGREE_COL: df[DEGREE_COL],
            "degree_total": degree_total,
        }
    )
    degree_size_map = (
        tmp.dropna(subset=[DEGREE_COL, "degree_total"])
        .groupby(DEGREE_COL)["degree_total"]
        .agg(mode_or_first)
    )
    return df[DEGREE_COL].map(degree_size_map)


def block_is_structurally_blank(df: pd.DataFrame, cols: List[str]) -> bool:
    """True when a feature block is blank for every row and month in this year."""
    if not cols:
        return True
    return int(df[cols].notna().sum().sum()) == 0


def build_semester_activity_features(
    df: pd.DataFrame,
    year: int,
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, object]]:
    """Build semester activity features and their missing flags."""
    value_features: List[str] = []
    flag_features: List[str] = []
    diagnostics: Dict[str, object] = {
        "structurally_blank_activity_blocks_kept_as_blank": [],
        "created_activity_value_features": [],
        "created_activity_flag_features": [],
    }
    new_cols: Dict[str, pd.Series] = {}

    for base in ACTIVITY_BASES:
        for sem_name, months in [("sem_a", SEM_A_MONTHS), ("sem_b", SEM_B_MONTHS)]:
            cols = monthly_columns(df, base, months)
            value_col = f"{base}_{sem_name}"
            flag_col = f"{base}_{sem_name}_missing_flag"

            # No source columns for this block: keep value + flag blank.
            if not cols:
                new_cols[value_col] = pd.Series(np.nan, index=df.index, dtype="float64")
                new_cols[flag_col] = pd.Series(pd.NA, index=df.index, dtype="Int64")
                value_features.append(value_col)
                flag_features.append(flag_col)
                diagnostics["structurally_blank_activity_blocks_kept_as_blank"].append(
                    {
                        "year": year,
                        "base": base,
                        "semester": sem_name,
                        "reason": "no_source_columns",
                    }
                )
                continue

            # Source columns exist but all blank (2018 case): keep value + flag blank.
            if block_is_structurally_blank(df, cols):
                new_cols[value_col] = pd.Series(np.nan, index=df.index, dtype="float64")
                new_cols[flag_col] = pd.Series(pd.NA, index=df.index, dtype="Int64")
                value_features.append(value_col)
                flag_features.append(flag_col)
                diagnostics["structurally_blank_activity_blocks_kept_as_blank"].append(
                    {
                        "year": year,
                        "base": base,
                        "semester": sem_name,
                        "reason": "all_source_values_blank",
                    }
                )
                continue

            missing_flag = df[cols].isna().any(axis=1).astype("int8")
            new_cols[flag_col] = missing_flag
            new_cols[value_col] = df[cols].fillna(0).sum(axis=1)
            value_features.append(value_col)
            flag_features.append(flag_col)
            diagnostics[f"rows_with_{flag_col}"] = int(missing_flag.sum())
            diagnostics["created_activity_value_features"].append(value_col)
            diagnostics["created_activity_flag_features"].append(flag_col)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df, value_features, flag_features, diagnostics


def validate_final(final: pd.DataFrame, year: int) -> List[str]:
    warnings: List[str] = []

    if final["student_id_hash"].duplicated().any():
        duplicated = int(final["student_id_hash"].duplicated().sum())
        warnings.append(f"{year}: duplicated student_id_hash rows = {duplicated}")

    target_values = set(final["target"].dropna().astype(str).unique().tolist())
    if not target_values.issubset({"dropout", "non-dropout"}):
        warnings.append(f"{year}: target contains unexpected values: {target_values}")

    # No monthly columns should survive into the student-level output.
    monthly_survivors = [
        c for c in final.columns
        if re.search(r"_20\d{2}_\d{1,2}$", c)
    ]
    if monthly_survivors:
        warnings.append(f"{year}: monthly columns survived: {monthly_survivors[:10]}")

    forbidden_survivors = sorted(
        [c for c in final.columns if c in COLUMNS_THAT_MUST_NOT_SURVIVE]
    )
    if forbidden_survivors:
        warnings.append(f"{year}: forbidden raw/drop columns survived: {forbidden_survivors}")

    removed_frac_cols = sorted(
        [c for c in final.columns if c in DROP_FINAL_COLUMNS]
    )
    if removed_frac_cols:
        warnings.append(f"{year}: removed frac columns survived: {removed_frac_cols}")

    suspicious_credit_names = [
        c for c in final.columns
        if c.startswith("cred_sup_") or c.startswith("cred_mat_")
    ]
    if suspicious_credit_names:
        warnings.append(f"{year}: suspicious raw credit names survived: {suspicious_credit_names}")
    return warnings


def clean_one_year(path: Path, year: int, output_dir: Path) -> Dict[str, object]:
    df = read_raw_csv(path)
    report: Dict[str, object] = {
        "year": year,
        "input_file": str(path),
        "input_rows": int(len(df)),
        "input_students": int(df[ID_COL].nunique(dropna=True)),
        "input_columns": int(df.shape[1]),
    }

    # Row-level credit sanity checks.
    row_error_mask = pd.Series(False, index=df.index)
    if {"cred_sup_normal", "cred_mat_normal"}.issubset(df.columns):
        mask = df["cred_sup_normal"].gt(df["cred_mat_normal"])
        row_error_mask |= mask.fillna(False)
        report["dropped_rows_cred_sup_normal_gt_cred_mat_normal"] = count_true(mask)

    credit_year_cols = [f"cred_mat{i}" for i in range(1, 7)]
    if set(credit_year_cols + ["cred_mat_total"]).issubset(df.columns):
        credit_sum = df[credit_year_cols].sum(axis=1, min_count=1)
        mismatch = (credit_sum - df["cred_mat_total"]).abs().gt(0.01)
        row_error_mask |= mismatch.fillna(False)
        report["dropped_rows_credit_identity_mismatch"] = count_true(mismatch)

    report["dropped_rows_total_before_student_drop"] = int(row_error_mask.sum())
    df = df.loc[~row_error_mask].copy()
    report["rows_after_row_drops"] = int(len(df))

    # Drop students who started studies after the academic year (negative seniority).
    seniority_raw = df[YEAR_COL] - df[START_YEAR_COL]
    anomalous_student_ids = (
        df.loc[seniority_raw.lt(0).fillna(False), ID_COL]
        .dropna()
        .unique()
    )
    report["dropped_students_start_year_gt_academic_year"] = int(len(anomalous_student_ids))
    if len(anomalous_student_ids) > 0:
        df = df.loc[~df[ID_COL].isin(anomalous_student_ids)].copy()
    report["rows_after_student_drops"] = int(len(df))
    report["students_after_student_drops"] = int(df[ID_COL].nunique(dropna=True))

    # degree_size must be computed before the leakage columns are dropped.
    df["_degree_size"] = make_degree_size(df)

    df = drop_rows_without_any_lms_lms_activity(df, report)

    # Drop rows with a blank course grade.
    students_before_blank_grade_drop = df[ID_COL].nunique(dropna=True)
    blank_grade_mask = df[GRADE_COL].isna()
    report["dropped_rows_blank_nota_asig_hash"] = int(blank_grade_mask.sum())
    df = df.loc[~blank_grade_mask].copy()
    students_after_blank_grade_drop = df[ID_COL].nunique(dropna=True)
    report["dropped_students_all_courses_blank_grade"] = int(
        students_before_blank_grade_drop - students_after_blank_grade_drop
    )
    report["rows_after_blank_grade_drop"] = int(len(df))
    report["students_after_blank_grade_drop"] = int(students_after_blank_grade_drop)
    if df.empty:
        raise ValueError(f"All rows were removed after dropping blank {GRADE_COL} for {year}.")

    # LMS minutes can come in negative; take absolute value.
    lms_minutes_cols = monthly_columns(df, "lms_total_minutes")
    report["negative_lms_total_minutes_cells_before_abs"] = (
        int((df[lms_minutes_cols] < 0).sum().sum()) if lms_minutes_cols else 0
    )
    for col in lms_minutes_cols:
        df[col] = df[col].abs()

    # p99 capping is deferred to train-only preprocessing.
    event_cols_to_cap: List[str] = []
    for base in EVENT_BASES_TO_WINSORIZE:
        event_cols_to_cap.extend(monthly_columns(df, base))
    report["p99_caps_monthly_lms_events_resource_events"] = "deferred_to_train_only"

    df, activity_value_cols, activity_flag_cols, activity_diag = build_semester_activity_features(
        df,
        year,
    )
    report.update(activity_diag)

    # Course-level aggregates per student.
    course_agg = (
        df.groupby(ID_COL, dropna=False)
        .agg(
            n_courses=(COURSE_COL, "nunique"),
            courses_mean=(GRADE_COL, "mean"),
        )
        .reset_index()
    )

    # One row per student for the base columns.
    agg_map = {
        col: first_non_null
        for col in BASE_STUDENT_COLS
        if col in df.columns
    }
    if TARGET_COL in agg_map:
        agg_map[TARGET_COL] = mode_or_first
    base_student = (
        df.groupby(ID_COL, dropna=False)
        .agg(agg_map)
        .reset_index()
    )
    final = base_student.merge(course_agg, on=ID_COL, how="left")

    # LMS values summed across courses; Wi-Fi (attendance) taken as max per student.
    wifi_value_cols = [
        col for col in activity_value_cols
        if any(col == f"{base}_{sem}" for base in WIFI_BASES for sem in ("sem_a", "sem_b"))
    ]
    lms_value_cols = [
        col for col in activity_value_cols
        if col not in wifi_value_cols
    ]

    if lms_value_cols:
        lms_values_student = (
            df.groupby(ID_COL, dropna=False)[lms_value_cols]
            .sum(min_count=1)
            .reset_index()
        )
        final = final.merge(lms_values_student, on=ID_COL, how="left")

    if wifi_value_cols:
        wifi_values_student = (
            df.groupby(ID_COL, dropna=False)[wifi_value_cols]
            .max()
            .reset_index()
        )
        final = final.merge(wifi_values_student, on=ID_COL, how="left")

    if activity_flag_cols:
        activity_flags_student = (
            df.groupby(ID_COL, dropna=False)[activity_flag_cols]
            .max()
            .reset_index()
        )
        final = final.merge(activity_flags_student, on=ID_COL, how="left")

    final["seniority"] = (final[YEAR_COL] - final[START_YEAR_COL]).clip(lower=0)

    # Pass ratios (%) per semester and overall.
    final["rendimiento_cuat_a"], final["pass_ratio_sem_a_and_zero_flag"] = safe_ratio(
        final["cred_sup_sem_a"],
        final["cred_mat_sem_a"],
    )
    final["rendimiento_cuat_b"], final["pass_ratio_sem_b_and_zero_flag"] = safe_ratio(
        final["cred_sup_sem_b"].fillna(0) + final["cred_sup_anu"].fillna(0),
        final["cred_mat_sem_b"].fillna(0) + final["cred_mat_anu"].fillna(0),
    )
    final["rendimiento_total"], final["total_performance_and_zero_flag"] = safe_ratio(
        final["cred_sup_total"],
        final["cred_mat_total"],
    )

    # Flag missing history, then fill with 0.
    for old_col, new_col in HISTORY_RENAME.items():
        if old_col in final.columns:
            final[f"{new_col}_missing_flag"] = final[old_col].isna().astype("int8")
            final[old_col] = final[old_col].fillna(0)

    for col in ["es_retitulado", "es_adaptado"]:
        if col in final.columns:
            final[col] = final[col].fillna(0)

    # Translate target labels.
    target_map = {
        "A": "dropout",
        "B": "non-dropout",
        "a": "dropout",
        "b": "non-dropout",
    }
    final["target"] = final[TARGET_COL].map(target_map)
    unexpected_targets = sorted(
        set(final.loc[final["target"].isna(), TARGET_COL].dropna().unique().tolist())
    )
    report["unexpected_target_values"] = unexpected_targets

    final = final.rename(columns=RENAME_FINAL)
    final = final.rename(
        columns={
            "rendimiento_cuat_a": "pass_ratio_sem_a",
            "rendimiento_cuat_b": "pass_ratio_sem_b",
            "rendimiento_total": "total_performance",
        }
    )

    # Cap implausible enrolled credits and flag them.
    credit_col = "total_credits_enrolled_academic_year"
    MAX_PLAUSIBLE_CREDITS = 120.0
    if credit_col in final.columns:
        impossible_mask = final[credit_col].gt(MAX_PLAUSIBLE_CREDITS)
        report["students_with_total_credits_enrolled_gt_120_before_cap"] = int(
            impossible_mask.sum()
        )
        final["total_credits_enrolled_impossible_flag"] = impossible_mask.astype("int8")
        final.loc[impossible_mask, credit_col] = MAX_PLAUSIBLE_CREDITS

    ratio_and_history_flags = [
        "pass_ratio_sem_a_and_zero_flag",
        "pass_ratio_sem_b_and_zero_flag",
        "total_performance_and_zero_flag",
        "total_credits_enrolled_impossible_flag",
        "completion_rate_one_year_before_missing_flag",
        "completion_rate_two_years_before_missing_flag",
        "completion_rate_three_years_before_missing_flag",
    ]

    final_base_order = [
        "student_id_hash",
        "academic_year",
        "target",
        "n_courses",
        "courses_mean",
        "degree_size",
        "seniority",
        "highest_course_year_enrolled",
        "credits_enrolled_semester_a",
        "credits_enrolled_semester_b",
        "total_credits_enrolled_academic_year",
        "total_credits_passed_academic_year",
        "pass_ratio_sem_a",
        "pass_ratio_sem_b",
        "total_performance",
        "obtained_new_degree_flag",
        "adapted_studies_flag",
        "completion_rate_one_year_before",
        "completion_rate_two_years_before",
        "completion_rate_three_years_before",
    ]

    final_activity_cols = [
        c for c in activity_value_cols
        if c in final.columns
    ]
    final_flag_cols = [
        c for c in activity_flag_cols
        if c in final.columns
    ]

    # Build final column order.
    ordered_cols = [c for c in final_base_order if c in final.columns]
    ordered_cols += [c for c in ratio_and_history_flags if c in final.columns]
    ordered_cols += final_activity_cols
    ordered_cols += final_flag_cols

    exclude = set(ordered_cols) | {TARGET_COL, START_YEAR_COL} | DROP_FINAL_COLUMNS
    derived_leftovers = [
        c for c in final.columns
        if c not in exclude
        and c not in COLUMNS_THAT_MUST_NOT_SURVIVE
        and not re.search(r"_20\d{2}_\d{1,2}$", c)
    ]
    derived_leftovers = [
        c for c in derived_leftovers
        if c not in BASE_STUDENT_COLS
    ]
    ordered_cols += derived_leftovers

    final = final[ordered_cols].copy()
    final = final.drop(columns=list(DROP_FINAL_COLUMNS), errors="ignore")

    still_exists = DROP_FINAL_COLUMNS.intersection(final.columns)
    if still_exists:
        raise AssertionError(
            f"Columns should have been removed but still exist: {sorted(still_exists)}"
        )

    # Cast integer-like columns.
    int_like_cols = [
        "academic_year",
        "n_courses",
        "seniority",
        "highest_course_year_enrolled",
        "obtained_new_degree_flag",
        "adapted_studies_flag",
    ] + [c for c in final.columns if c.endswith("_flag")]
    for col in int_like_cols:
        if col in final.columns:
            final[col] = pd.to_numeric(final[col], errors="coerce").round().astype("Int64")

    # Final safety filters (after aggregation, renaming, ordering and int casting).

    # Drop students if lms_events_sem_a > 4000 OR lms_events_sem_b > 4000.
    lms_events_drop_cols = ["lms_events_sem_a", "lms_events_sem_b"]
    lms_events_existing_cols = [
        c for c in lms_events_drop_cols
        if c in final.columns
    ]
    if set(lms_events_drop_cols).issubset(final.columns):
        lms_events_gt_4000_mask = final[lms_events_drop_cols].gt(4000).any(axis=1)
        report["lms_events_drop_threshold"] = 4000
        report["lms_events_drop_logic"] = (
            "drop_if_either_lms_events_sem_a_or_lms_events_sem_b_gt_4000"
        )
        report["lms_events_drop_columns_checked"] = lms_events_drop_cols
        report["dropped_rows_lms_events_either_semester_gt_4000_final"] = int(
            lms_events_gt_4000_mask.sum()
        )
        report["dropped_rows_lms_events_either_semester_gt_4000_by_column"] = {
            c: int(final[c].gt(4000).sum())
            for c in lms_events_drop_cols
        }
        if int(lms_events_gt_4000_mask.sum()) > 0 and "target" in final.columns:
            report["dropped_rows_lms_events_either_semester_gt_4000_by_target"] = {
                str(k): int(v)
                for k, v in final.loc[lms_events_gt_4000_mask, "target"]
                .value_counts(dropna=False)
                .items()
            }
        final = final.loc[~lms_events_gt_4000_mask].copy()
    else:
        report["lms_events_drop_threshold"] = 4000
        report["lms_events_drop_logic"] = "not_applied_missing_required_columns"
        report["lms_events_drop_columns_checked"] = lms_events_existing_cols
        report["dropped_rows_lms_events_either_semester_gt_4000_final"] = 0
        report["dropped_rows_lms_events_either_semester_gt_4000_by_column"] = {}

    # Drop students if any selected LMS activity feature > 5000.
    high_activity_drop_cols = [
        "lms_events_sem_a",
        "lms_events_sem_b",
        "lms_assignment_submissions_sem_a",
        "lms_assignment_submissions_sem_b",
        "lms_test_submissions_sem_a",
        "lms_test_submissions_sem_b",
        "resource_events_sem_a",
        "resource_events_sem_b",
    ]
    high_activity_existing_cols = [
        c for c in high_activity_drop_cols
        if c in final.columns
    ]
    if high_activity_existing_cols:
        high_activity_mask = final[high_activity_existing_cols].gt(5000).any(axis=1)
        report["high_activity_drop_threshold"] = 5000
        report["high_activity_drop_columns_checked"] = high_activity_existing_cols
        report["dropped_rows_high_activity_gt_5000_final"] = int(
            high_activity_mask.sum()
        )
        if int(high_activity_mask.sum()) > 0:
            report["dropped_rows_high_activity_gt_5000_by_column"] = {
                c: int(final[c].gt(5000).sum())
                for c in high_activity_existing_cols
            }
            if "target" in final.columns:
                report["dropped_rows_high_activity_gt_5000_by_target"] = {
                    str(k): int(v)
                    for k, v in final.loc[high_activity_mask, "target"]
                    .value_counts(dropna=False)
                    .items()
                }
        final = final.loc[~high_activity_mask].copy()
    else:
        report["high_activity_drop_threshold"] = 5000
        report["high_activity_drop_columns_checked"] = []
        report["dropped_rows_high_activity_gt_5000_final"] = 0

    # Hard check: no lms_events_sem_a/b > 4000 may survive.
    remaining_lms_events_gt_4000 = {
        c: int(final[c].gt(4000).sum())
        for c in lms_events_drop_cols
        if c in final.columns
    }
    report["remaining_lms_events_gt_4000_after_final_filter"] = (
        remaining_lms_events_gt_4000
    )
    if any(v > 0 for v in remaining_lms_events_gt_4000.values()):
        raise AssertionError(
            "lms events > 4000 still remain after final filter: "
            f"{remaining_lms_events_gt_4000}"
        )

    warnings = validate_final(final, year)
    warnings += validate_feature_lists(final, year, report)
    report["validation_warnings"] = warnings

    report["output_rows"] = int(final.shape[0])
    report["output_columns"] = int(final.shape[1])
    if "target" in final:
        valid_target = final["target"].dropna()
        report["target_dropout_rate"] = (
            float(valid_target.eq("dropout").mean()) if len(valid_target) else None
        )
        report["target_counts"] = {
            str(k): int(v)
            for k, v in final["target"].value_counts(dropna=False).items()
        }
    else:
        report["target_dropout_rate"] = None
        report["target_counts"] = {}

    report["remaining_missing_values_by_column"] = {
        c: int(v)
        for c, v in final.isna().sum().items()
        if int(v) > 0
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"cleaned_dataset_{year}_student_level.csv"
    final.to_csv(out_path, index=False)
    report["output_file"] = str(out_path)
    return report


def fit_transform_train_only(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit median imputation + p99 caps + StandardScaler on TRAIN, apply to TEST.

    Call this in the modelling step, after the train/test split.
    """
    from sklearn.preprocessing import StandardScaler

    train = train_df.copy()
    test = test_df.copy()

    do_not_scale = {"student_id_hash", "target"}
    numeric_cols = [
        c for c in train.columns
        if c not in do_not_scale and pd.api.types.is_numeric_dtype(train[c])
    ]

    medians = train[numeric_cols].median(numeric_only=True)
    train[numeric_cols] = train[numeric_cols].fillna(medians)
    test[numeric_cols] = test[numeric_cols].fillna(medians)

    p99_caps = train[numeric_cols].quantile(0.99)
    for col in numeric_cols:
        train[col] = train[col].clip(upper=p99_caps[col])
        test[col] = test[col].clip(upper=p99_caps[col])

    scaler = StandardScaler()
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])

    stats = {
        "median": medians.to_dict(),
        "p99_caps": p99_caps.to_dict(),
        "scaler": scaler,
    }
    return train, test, stats


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reports: List[Dict[str, object]] = []

    for year, path in INPUT_FILES.items():
        print(f"Cleaning {year}: {path}")
        report = clean_one_year(path, year, OUTPUT_DIR)
        reports.append(report)
        print(
            f"  -> {report['output_file']} | "
            f"rows={report['output_rows']} | "
            f"cols={report['output_columns']} | "
            f"dropout_rate={report['target_dropout_rate']:.4f}"
        )
        if report["validation_warnings"]:
            print("  WARNINGS:")
            for warning in report["validation_warnings"]:
                print(f"    - {warning}")

        per_year_report_path = (
            OUTPUT_DIR / f"report_{year}_lms_row_filtered_no_frac_no_structural_blank_flags.json"
        )
        with per_year_report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    report_path = OUTPUT_DIR / "cleaning_report_lms_row_filtered_no_frac_no_structural_blank_flags.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f"QA report saved to: {report_path}")


if __name__ == "__main__":
    main()
