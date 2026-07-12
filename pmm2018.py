#!/usr/bin/env python3
"""
Predictive Mean Matching (PMM) imputation of the six activity variables that are
systematically missing in 2018, using 2021 + 2022 as the donor sample.

Why PMM: it imputes by copying an OBSERVED donor value whose model-predicted mean is
closest to the recipient's predicted mean. Imputed values are therefore always real,
non-negative, integer counts, and preserve the donor distribution (no variance shrinkage).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

SEED = 42
N_DONORS = 5          # candidate donors per recipient (mice default is 5)

EARLY_FEATURES = ["degree_size","seniority","highest_course_year_enrolled","adapted_studies_flag",
    "credits_enrolled_semester_a","credits_enrolled_semester_b","total_credits_enrolled_academic_year",
    "total_credits_enrolled_impossible_flag","completion_rate_one_year_before","completion_rate_two_years_before",
    "completion_rate_three_years_before","completion_rate_one_year_before_missing_flag",
    "completion_rate_two_years_before_missing_flag","completion_rate_three_years_before_missing_flag",
    "pass_ratio_sem_a","pass_ratio_sem_a_den_zero_flag","lms_events_sem_a","lms_assignment_submissions_sem_a",
    "lms_test_submissions_sem_a","lms_total_minutes_sem_a","resource_events_sem_a","attendance_days_sem_a"]
FULL_ONLY = ["n_courses","courses_mean","total_credits_passed_academic_year","pass_ratio_sem_b",
    "pass_ratio_sem_b_den_zero_flag","total_performance","total_performance_den_zero_flag",
    "obtained_new_degree_flag","lms_events_sem_b","lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_b","lms_total_minutes_sem_b","resource_events_sem_b","attendance_days_sem_b"]
FULL_FEATURES = EARLY_FEATURES + FULL_ONLY

TARGETS_SEM_A = ["lms_assignment_submissions_sem_a","lms_test_submissions_sem_a","attendance_days_sem_a"]
TARGETS_SEM_B = ["lms_assignment_submissions_sem_b","lms_test_submissions_sem_b","attendance_days_sem_b"]
ALL_TARGETS = TARGETS_SEM_A + TARGETS_SEM_B


def predictors_for(context: list[str]) -> list[str]:
    # Predictors must be OBSERVED in 2018 -> exclude every systematically-missing target column.
    return [c for c in context if c not in ALL_TARGETS]


def _prep_X(df: pd.DataFrame, cols: list[str], med: pd.Series, mean: pd.Series, std: pd.Series) -> np.ndarray:
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(med)                 # donor-median fill for any missing predictor
    X = (X - mean) / std              # donor-fitted standardisation
    return X.values


def pmm_impute_column(recipient: pd.DataFrame, donor: pd.DataFrame,
                      pred_cols: list[str], y_col: str, rng: np.random.Generator) -> np.ndarray:
    d = donor[donor[y_col].notna()].copy()
    y = pd.to_numeric(d[y_col], errors="coerce").values

    med = d[pred_cols].apply(pd.to_numeric, errors="coerce").median()
    Xd_raw = d[pred_cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    mean, std = Xd_raw.mean(), Xd_raw.std(ddof=0).replace(0, 1).fillna(1)

    Xd = ((Xd_raw - mean) / std).values
    Xr = _prep_X(recipient, pred_cols, med, mean, std)

    model = Ridge(alpha=1.0, random_state=SEED).fit(Xd, y)
    yhat_d = model.predict(Xd).reshape(-1, 1)
    yhat_r = model.predict(Xr).reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=min(N_DONORS, len(y))).fit(yhat_d)
    _, idx = nn.kneighbors(yhat_r)                       # nearest donors by predicted mean
    pick = idx[np.arange(idx.shape[0]), rng.integers(0, idx.shape[1], size=idx.shape[0])]
    return y[pick]                                       # copy a real observed donor value


def holdout_validation(donor: pd.DataFrame, pred_cols: list[str], y_col: str,
                       frac: float = 0.10) -> dict:
    rng = np.random.default_rng(SEED)
    d = donor[donor[y_col].notna()].reset_index(drop=True)
    hold = rng.choice(d.index.values, size=int(frac * len(d)), replace=False)
    truth = pd.to_numeric(d.loc[hold, y_col]).astype(float).values

    masked = d.copy(); masked.loc[hold, y_col] = np.nan
    fit_pool = masked.drop(index=hold)
    pred = pmm_impute_column(masked.loc[hold], fit_pool, pred_cols, y_col, rng).astype(float)

    mae = float(np.mean(np.abs(truth - pred)))
    rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
    med = np.median(fit_pool[y_col].dropna().astype(float))
    base_mae = float(np.mean(np.abs(truth - med)))
    return {"column": y_col, "MAE_pmm": round(mae, 2), "RMSE_pmm": round(rmse, 2),
            "MAE_median_baseline": round(base_mae, 2),
            "std_truth": round(truth.std(), 2), "std_pmm": round(pred.std(), 2),
            "std_ratio": round(pred.std() / truth.std(), 2) if truth.std() else None}


def main():
    d18 = pd.read_csv("cleaned_dataset_2018_student_level.csv")
    d21 = pd.read_csv("cleaned_dataset_2021_student_level.csv")
    d22 = pd.read_csv("cleaned_dataset_2022_student_level.csv")
    donor = pd.concat([d21, d22], ignore_index=True)
    rng = np.random.default_rng(SEED)

    out = d18.copy()
    reports = []
    for targets, context in [(TARGETS_SEM_A, EARLY_FEATURES), (TARGETS_SEM_B, FULL_FEATURES)]:
        pred_cols = predictors_for(context)
        for col in targets:
            vals = pmm_impute_column(out, donor, pred_cols, col, rng)
            out[col] = pd.Series(vals, index=out.index).round().astype("Int64")
            reports.append(holdout_validation(donor, pred_cols, col))

    print("=== HOLD-OUT VALIDATION (within donor years 2021+2022) ===")
    print(pd.DataFrame(reports).to_string(index=False))
    print("\n=== VARIANCE CHECK: donor std vs imputed-2018 std ===")
    for col in ALL_TARGETS:
        s_don = pd.to_numeric(donor[col]).std(); s_imp = pd.to_numeric(out[col]).std()
        print(f"{col:34s} donor_std={s_don:6.2f}  imputed2018_std={s_imp:6.2f}  ratio={s_imp/s_don:.2f}")
    print(f"\nremaining NaN in 6 cols: {int(out[ALL_TARGETS].isna().sum().sum())}")
    out.to_csv("cleaned_dataset_2018_student_level_pmm.csv", index=False)
    pd.DataFrame(reports).to_csv("pmm_holdout_validation_report.csv", index=False)


if __name__ == "__main__":
    main()