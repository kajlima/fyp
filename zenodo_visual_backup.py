from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# CONFIG
# ============================================================
# all epsilon values are read from the result files, never hardcoded, so every
# label matches the data. one story per model: baseline -> under attack ->
# defended (clean) -> defended (under attack), for the test and validation splits

OUTPUT_DIR = Path("zenodo_adversarial_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_TO_PLOT: tuple[str, ...] = ("test", "validation")

# real output folder first, plain filename as a fallback
RF_ATTACK_CANDIDATES = [
    "fgsm_rf_full_results/rf_fgsm_attack_results.csv",
    "rf_fgsm_attack_results.csv",
]
RF_DEFENCE_CANDIDATES = [
    "rf_adversarial_training_full_results/rf_adversarial_training_results.csv",
    "rf_adversarial_training_results.csv",
]
MLP_ATTACK_CANDIDATES = [
    "pgd_mlp_full_results/mlp_pgd_attack_results.csv",
    "mlp_pgd_attack_results.csv",
]
MLP_DEFENCE_CANDIDATES = [
    "mlp_adversarial_training_full_results/mlp_adversarial_training_results.csv",
    "mlp_adversarial_training_results.csv",
]
PROX_SUMMARY_CANDIDATES = [
    "proximity_sensitivity_final_results/proximity_sensitivity_final_summary.csv",
    "proximity_sensitivity_final_summary.csv",
]
FEATURE_SUMMARY_CANDIDATES = [
    "proximity_sensitivity_final_results/top15_sensitive_features_final.csv",
    "proximity_sensitivity_final_results/feature_sensitivity_final_summary.csv",
    "top15_sensitive_features_final.csv",
    "feature_sensitivity_final_summary.csv",
]

# ============================================================
# THEME (colourblind-friendly, consistent across every figure)
# ============================================================
COL = {
    "rf": "#3B6EA5",          # Random Forest
    "mlp": "#E1812C",         # MLP
    "clean": "#2A9D8F",       # clean / safe
    "attack": "#D1495B",      # under attack
    "def_clean": "#8AB6D6",   # defended clean
    "def_attack": "#2E5E8C",  # defended under attack
    "text": "#222222",
    "muted": "#6B6B6B",
    "grid": "#E3E3E3",
}


def apply_theme():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COL["grid"],
        "grid.linewidth": 0.9,
        "xtick.color": COL["text"],
        "ytick.color": COL["text"],
        "text.color": COL["text"],
        "legend.frameon": False,
    })


def style_axis(ax, ygrid=True, xgrid=False, ymax=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", visible=ygrid)
    ax.grid(axis="x", visible=xgrid)
    if ymax is not None:
        ax.set_ylim(0, ymax)


def bar_value_labels(ax, bars, values, fmt="{:.2f}", dy=0.012, fontsize=9, color=None):
    # print each bar's value just above the bar
    valid = [v for v in values if pd.notna(v)]
    top = max(valid + [0])
    pad = dy if top <= 1.2 else top * dy
    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                fmt.format(value), ha="center", va="bottom",
                fontsize=fontsize, color=color or COL["text"])


def figure_caption(fig, text):
    fig.text(0.5, -0.02, text, ha="center", va="top",
             fontsize=8.5, color=COL["muted"], style="italic")


def save_plot(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# DATA LOADING
# ============================================================

def existing_paths(candidates: Iterable[str]) -> List[Path]:
    paths, seen = [], set()
    for c in candidates:
        p = Path(c)
        if p.exists() and p.resolve() not in seen:
            paths.append(p)
            seen.add(p.resolve())
    return paths


def read_all_csv(candidates: Iterable[str], required: bool, label: str) -> Optional[pd.DataFrame]:
    # read every candidate that exists and stack them, tagging each with its source
    # and priority (0 = the preferred folder) so later selection can prefer it
    paths = existing_paths(candidates)
    if not paths:
        if required:
            checked = "\n".join(str(Path(c)) for c in candidates)
            raise FileNotFoundError(f"required {label} not found. checked:\n{checked}")
        print(f"Skipped optional {label}: no matching file found.")
        return None
    frames = []
    for priority, path in enumerate(paths):
        df = pd.read_csv(path)
        df["_source_file"] = str(path)
        df["_source_priority"] = priority
        frames.append(df)
        print(f"Loaded {label}: {path}")
    return pd.concat(frames, ignore_index=True)


def normalize_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    # unify metric names and make sure epsilon/condition/split columns exist
    if df is None:
        return None
    df = df.copy()
    rename = {}
    if "f1_dropout" in df.columns and "f1_score" not in df.columns:
        rename["f1_dropout"] = "f1_score"
    if "precision_dropout" in df.columns and "precision" not in df.columns:
        rename["precision_dropout"] = "precision"
    if "recall_dropout" in df.columns and "recall" not in df.columns:
        rename["recall_dropout"] = "recall"
    df = df.rename(columns=rename)
    if "eval_epsilon" not in df.columns:
        df["eval_epsilon"] = df["epsilon"] if "epsilon" in df.columns else np.nan
    if "train_epsilon" not in df.columns:
        df["train_epsilon"] = np.nan
    if "epsilon" not in df.columns:
        df["epsilon"] = df["eval_epsilon"]
    if "condition" not in df.columns:
        df["condition"] = "Unknown"
    df["split_norm"] = df["split"].astype(str).str.lower() if "split" in df.columns else "unknown"
    df["condition_norm"] = df["condition"].astype(str).str.lower()
    return df


def metric(row: Optional[pd.Series], col: str) -> float:
    if row is None or col not in row.index or pd.isna(row[col]):
        return np.nan
    return float(row[col])


# ============================================================
# ROW SELECTION (condition-based, epsilon read from data)
# ============================================================

def _subset(df, split, include=None, exclude=None, clean_only=False, attacked_only=False):
    sub = df[df["split_norm"] == split.lower()].copy()
    if include is not None:
        keep = pd.Series(False, index=sub.index)
        for term in include:
            keep |= sub["condition_norm"].str.contains(term.lower(), na=False)
        sub = sub[keep]
    if exclude is not None:
        for term in exclude:
            sub = sub[~sub["condition_norm"].str.contains(term.lower(), na=False)]
    eval_eps = pd.to_numeric(sub["eval_epsilon"], errors="coerce").fillna(0.0)
    if clean_only:
        sub = sub[eval_eps == 0.0]
    if attacked_only:
        sub = sub[eval_eps > 0.0]
    return sub


def pick_clean_row(df, split, defended):
    if df is None or df.empty:
        return None
    if defended:
        sub = _subset(df, split, include=["defended clean", "defended", "clean"], clean_only=True)
        sub = sub[sub["condition_norm"].str.contains("defend", na=False)]
    else:
        sub = _subset(df, split, include=["clean"], exclude=["defend"], clean_only=True)
    if sub.empty:
        return None
    if "_source_priority" in sub.columns:
        sub = sub.sort_values("_source_priority")
    return sub.iloc[0]


def pick_attack_row(df, split, attack_keyword, defended):
    # pick the attacked row; when several epsilons exist, take the largest budget
    if df is None or df.empty:
        return None
    if defended:
        sub = _subset(df, split, include=[attack_keyword], attacked_only=True)
        sub = sub[sub["condition_norm"].str.contains("defend", na=False)]
    else:
        sub = _subset(df, split, include=[attack_keyword], exclude=["defend"], attacked_only=True)
    if sub.empty:
        return None
    sub = sub.copy()
    sub["_e"] = pd.to_numeric(sub["eval_epsilon"], errors="coerce")
    sub = sub.sort_values(["_e", "_source_priority"], ascending=[False, True])
    return sub.iloc[0]


def first_available(*rows):
    for r in rows:
        if r is not None:
            return r
    return None


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def build_performance_summary(split, rf_attack, rf_def, mlp_attack, mlp_def) -> pd.DataFrame:
    rows = []

    def add(model, stage, attack, row, note=""):
        if row is None:
            print(f"  Missing row: {model} | {stage} | split={split}")
            return
        rows.append({
            "split": split, "model": model, "stage": stage, "attack": attack,
            "train_epsilon": metric(row, "train_epsilon"),
            "eval_epsilon": metric(row, "eval_epsilon"),
            "accuracy": metric(row, "accuracy"),
            "precision": metric(row, "precision"),
            "recall": metric(row, "recall"),
            "f1_score": metric(row, "f1_score"),
            "attacked_rows_count": metric(row, "attacked_rows_count"),
            "dropout_to_nondropout_flips": metric(row, "dropout_to_nondropout_flips"),
            "source_file": row.get("_source_file", ""),
            "note": note,
        })

    # RF: baseline clean -> FGSM attack -> defended clean -> defended under attack
    add("Random Forest", "Baseline clean", "-",
        first_available(pick_clean_row(rf_attack, split, False), pick_clean_row(rf_def, split, False)))
    add("Random Forest", "Before defence", "FGSM", pick_attack_row(rf_attack, split, "fgsm", False))
    add("Random Forest", "Defended clean", "-", pick_clean_row(rf_def, split, True))
    add("Random Forest", "After defence", "FGSM", pick_attack_row(rf_def, split, "fgsm", True))
    # MLP: same four stages with PGD
    add("MLP", "Baseline clean", "-",
        first_available(pick_clean_row(mlp_attack, split, False), pick_clean_row(mlp_def, split, False)))
    add("MLP", "Before defence", "PGD", pick_attack_row(mlp_attack, split, "pgd", False))
    add("MLP", "Defended clean", "-", pick_clean_row(mlp_def, split, True))
    add("MLP", "After defence", "PGD", pick_attack_row(mlp_def, split, "pgd", True))

    df = pd.DataFrame(rows)
    if not df.empty:
        flips = pd.to_numeric(df["dropout_to_nondropout_flips"], errors="coerce")
        attacked = pd.to_numeric(df["attacked_rows_count"], errors="coerce")
        df["dropout_hiding_rate"] = flips / attacked
        df.loc[attacked.fillna(0) == 0, "dropout_hiding_rate"] = np.nan
    return df


def get_val(df, model, stage, col):
    sub = df[(df["model"] == model) & (df["stage"] == stage)]
    if sub.empty or col not in sub.columns or pd.isna(sub.iloc[0][col]):
        return np.nan
    return float(sub.iloc[0][col])


def round_to_display(value, decimals=2):
    # round to the same precision the bar labels show, so the delta arrows equal
    # the difference of the printed bar values (exact figures stay in the tables)
    return value if pd.isna(value) else round(float(value), decimals)


def attack_eps(df, model):
    sub = df[(df["model"] == model) & (df["stage"] == "Before defence")]
    if sub.empty or pd.isna(sub.iloc[0]["eval_epsilon"]):
        return None
    return float(sub.iloc[0]["eval_epsilon"])


def eps_str(value):
    return f"{value:g}" if value is not None and pd.notna(value) else "?"


# ============================================================
# FIGURE 01: F1 JOURNEY
# ============================================================

def plot_f1_journey(df, split):
    stages = ["Baseline clean", "Before defence", "After defence"]
    labels = ["Baseline\n(clean)", "Under\nattack", "Defended\nunder attack"]
    rf = [get_val(df, "Random Forest", s, "f1_score") for s in stages]
    mlp = [get_val(df, "MLP", s, "f1_score") for s in stages]
    if all(pd.isna(v) for v in rf) and all(pd.isna(v) for v in mlp):
        print(f"Skipped F1 journey ({split}).")
        return
    x = np.arange(len(stages))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 6))
    b1 = ax.bar(x - w / 2, np.nan_to_num(rf), w, label="Random Forest", color=COL["rf"])
    b2 = ax.bar(x + w / 2, np.nan_to_num(mlp), w, label="MLP", color=COL["mlp"])
    bar_value_labels(ax, b1, rf)
    bar_value_labels(ax, b2, mlp)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dropout F1-score")
    ax.set_title(f"Dropout-detection F1 across the robustness pipeline — {split} set", pad=34)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    style_axis(ax, ymax=1.0)
    rf_e, mlp_e = attack_eps(df, "Random Forest"), attack_eps(df, "MLP")
    figure_caption(fig, f"Attack settings read from data — RF: FGSM \u03b5={eps_str(rf_e)}; "
                        f"MLP: PGD \u03b5={eps_str(mlp_e)}. Higher is better.")
    save_plot(fig, f"01_{split}_f1_journey.png")


# ============================================================
# FIGURE 02: DROP AND RECOVERY (delta-annotated)
# ============================================================

def _drop_recovery_panel(ax, df, model, colour, attack_name):
    baseline = round_to_display(get_val(df, model, "Baseline clean", "f1_score"))
    under = round_to_display(get_val(df, model, "Before defence", "f1_score"))
    def_under = round_to_display(get_val(df, model, "After defence", "f1_score"))
    vals = [baseline, under, def_under]
    labels = ["Baseline\n(clean)", f"{attack_name}\nattack", f"{attack_name} after\ndefence"]
    colours = [COL["clean"], COL["attack"], COL["def_attack"]]
    bars = ax.bar(range(3), np.nan_to_num(vals), color=colours, width=0.62)
    bar_value_labels(ax, bars, vals)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dropout F1-score")
    ax.set_title(model, color=colour)
    style_axis(ax, ymax=1.05)
    # red arrow: F1 lost to the attack
    if pd.notna(baseline) and pd.notna(under):
        drop = under - baseline
        ax.annotate("", xy=(1, under + 0.03), xytext=(0, baseline + 0.03),
                    arrowprops=dict(arrowstyle="->", color=COL["attack"], lw=1.6))
        ax.text(0.5, max(baseline, under) + 0.09, f"{drop:+.2f}", ha="center",
                color=COL["attack"], fontsize=9, fontweight="bold")
    # blue arrow: F1 recovered by the defence
    if pd.notna(under) and pd.notna(def_under):
        rec = def_under - under
        ax.annotate("", xy=(2, def_under + 0.03), xytext=(1, under + 0.03),
                    arrowprops=dict(arrowstyle="->", color=COL["def_attack"], lw=1.6))
        ax.text(1.5, max(under, def_under) + 0.09, f"{rec:+.2f}", ha="center",
                color=COL["def_attack"], fontsize=9, fontweight="bold")


def plot_drop_recovery(df, split):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    _drop_recovery_panel(axes[0], df, "Random Forest", COL["rf"], "FGSM")
    _drop_recovery_panel(axes[1], df, "MLP", COL["mlp"], "PGD")
    fig.suptitle(f"F1 collapse under attack and recovery after adversarial training — {split} set",
                 fontsize=13, fontweight="bold")
    figure_caption(fig, "Red arrow: F1 lost to the attack. Blue arrow: F1 recovered by the defence.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_plot(fig, f"02_{split}_drop_and_recovery.png")


# ============================================================
# FIGURE 03: PRECISION / RECALL / F1 UNDER ATTACK
# ============================================================

def _prf_panel(ax, df, model):
    metrics = ["precision", "recall", "f1_score"]
    names = ["Precision", "Recall", "F1"]
    before = [get_val(df, model, "Before defence", m) for m in metrics]
    after = [get_val(df, model, "After defence", m) for m in metrics]
    x = np.arange(3)
    w = 0.38
    b1 = ax.bar(x - w / 2, np.nan_to_num(before), w, label="Under attack (before defence)", color=COL["attack"])
    b2 = ax.bar(x + w / 2, np.nan_to_num(after), w, label="Under attack (after defence)", color=COL["def_attack"])
    bar_value_labels(ax, b1, before, fontsize=8)
    bar_value_labels(ax, b2, after, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title(model)
    style_axis(ax, ymax=1.05)


def plot_prf_under_attack(df, split):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    _prf_panel(axes[0], df, "Random Forest")
    _prf_panel(axes[1], df, "MLP")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Precision, recall and F1 under attack — before vs after defence ({split} set)",
                 y=0.95, fontsize=13, fontweight="bold")
    figure_caption(fig, "Recall is the metric the targeted attack destroys most, and the metric the defence restores.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save_plot(fig, f"03_{split}_precision_recall_f1_under_attack.png")


# ============================================================
# FIGURE 04: DROPOUT HIDING RATE
# ============================================================

def plot_hiding_rate(df, split):
    models = ["Random Forest", "MLP"]
    before = [get_val(df, m, "Before defence", "dropout_hiding_rate") for m in models]
    after = [get_val(df, m, "After defence", "dropout_hiding_rate") for m in models]
    if all(pd.isna(v) for v in before) and all(pd.isna(v) for v in after):
        print(f"Skipped hiding rate ({split}).")
        return
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    b1 = ax.bar(x - w / 2, np.nan_to_num(before), w, label="Before defence", color=COL["attack"])
    b2 = ax.bar(x + w / 2, np.nan_to_num(after), w, label="After defence", color=COL["def_attack"])
    bar_value_labels(ax, b1, before, fmt="{:.2%}")
    bar_value_labels(ax, b2, after, fmt="{:.2%}")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("True dropouts hidden (flip rate)")
    ax.set_title(f"Dropout hiding rate under targeted attack — {split} set", pad=34)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    top = max([v for v in before + after if pd.notna(v)] + [0.1])
    style_axis(ax, ymax=min(1.0, top * 1.35))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    figure_caption(fig, "Share of correctly-detected dropout students the attack flips to 'non-dropout'. Lower is safer.")
    save_plot(fig, f"04_{split}_dropout_hiding_rate.png")


# ============================================================
# FIGURE 05: CLEAN-PERFORMANCE COST OF DEFENCE
# ============================================================

def plot_clean_cost(df, split):
    models = ["Random Forest", "MLP"]
    base = [get_val(df, m, "Baseline clean", "f1_score") for m in models]
    defc = [get_val(df, m, "Defended clean", "f1_score") for m in models]
    if all(pd.isna(v) for v in base) and all(pd.isna(v) for v in defc):
        print(f"Skipped clean cost ({split}).")
        return
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    b1 = ax.bar(x - w / 2, np.nan_to_num(base), w, label="Baseline (undefended)", color=COL["clean"])
    b2 = ax.bar(x + w / 2, np.nan_to_num(defc), w, label="Defended", color=COL["def_clean"])
    bar_value_labels(ax, b1, base)
    bar_value_labels(ax, b2, defc)
    # signed clean-F1 gap between baseline and defended
    for xi, (a, b) in enumerate(zip(base, defc)):
        if pd.notna(a) and pd.notna(b):
            ax.text(xi, max(a, b) + 0.06, f"{b - a:+.2f}", ha="center",
                    color=COL["muted"], fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Clean dropout F1-score")
    ax.set_title(f"Cost of the defence on clean data — {split} set", pad=34)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    style_axis(ax, ymax=1.0)
    figure_caption(fig, "Clean F1 before vs after adversarial training. A small gap means the defence keeps normal accuracy.")
    save_plot(fig, f"05_{split}_clean_cost_of_defence.png")


# ============================================================
# FIGURE 06: PROXIMITY / SENSITIVITY
# ============================================================

def filter_proximity(prox_df, split):
    # keep this split; prefer attacked-rows-only if that scope is present
    if prox_df is None or prox_df.empty:
        return None
    df = normalize_columns(prox_df)
    df = df[df["split_norm"] == split.lower()].copy()
    if df.empty:
        return None
    if "row_scope" in df.columns:
        attacked = df[df["row_scope"].astype(str).str.lower() == "attacked_rows_only"].copy()
        if not attacked.empty:
            df = attacked
    df["model_clean"] = df["model"].astype(str).apply(
        lambda n: "Random Forest" if ("random" in n.lower() or n.lower().strip() == "rf") else
                  ("MLP" if "mlp" in n.lower() else n))
    return df


def _prox_bar(ax, df, col, title, ylabel):
    if col not in df.columns:
        ax.set_visible(False)
        return False
    d = df.dropna(subset=[col]).copy()
    if d.empty:
        ax.set_visible(False)
        return False
    order = ["Random Forest", "MLP"]
    present = [m for m in order if m in set(d["model_clean"])]
    d = d.set_index("model_clean").reindex(present).reset_index()
    values = pd.to_numeric(d[col], errors="coerce")
    colours = [COL["rf"] if m == "Random Forest" else COL["mlp"] for m in d["model_clean"]]
    eps = pd.to_numeric(d["eval_epsilon"], errors="coerce")
    labels = [f"{m}\n\u03b5={e:g}" if pd.notna(e) else m for m, e in zip(d["model_clean"], eps)]
    bars = ax.bar(range(len(d)), values, color=colours, width=0.55)
    bar_value_labels(ax, bars, values, fmt="{:.2f}")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax, ymax=float(values.max()) * 1.25)
    return True


def plot_proximity(prox_df, split):
    df = filter_proximity(prox_df, split)
    if df is None:
        print(f"Skipped proximity ({split}): no data.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ok1 = _prox_bar(axes[0], df, "mean_l2_attackable", "Mean L2 distance", "Mean L2 (attackable features)")
    ok2 = _prox_bar(axes[1], df, "mean_sensitivity", "Mean sensitivity", "Mean sensitivity")
    if not (ok1 or ok2):
        plt.close(fig)
        print(f"Skipped proximity ({split}): required columns not found.")
        return
    fig.suptitle(f"How much perturbation the attack needed on the defended models — {split} set",
                 fontsize=13, fontweight="bold")
    figure_caption(fig, "Larger distance/sensitivity means the attack had to push features harder to fool the model.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_plot(fig, f"06_{split}_proximity_sensitivity.png")


# ============================================================
# FIGURE 07: TOP SENSITIVE FEATURES
# ============================================================

def _detect_column(df, options):
    for name in options:
        if name in df.columns:
            return name
    return None


def _top_features_panel(ax, feature_df, split, model_keyword, model_label, colour, top_n=10):
    df = normalize_columns(feature_df)
    df = df[df["split_norm"] == split.lower()].copy()
    feat_col = _detect_column(df, ["feature_name", "feature", "feature_col"])
    sens_col = _detect_column(df, ["mean_sensitivity", "sensitivity", "mean_abs_change"])
    if feat_col is None or sens_col is None or df.empty:
        ax.set_visible(False)
        return False
    if "is_attackable_feature" in df.columns:
        df = df[df["is_attackable_feature"].astype(str).str.lower().isin(["true", "1", "1.0"])]
    if "row_scope" in df.columns:
        attacked = df[df["row_scope"].astype(str).str.lower() == "attacked_rows_only"].copy()
        if not attacked.empty:
            df = attacked
    if "model" in df.columns:
        df = df[df["model"].astype(str).str.lower().str.contains(model_keyword.lower(), na=False)]
    if df.empty:
        ax.set_visible(False)
        return False
    top = (df.groupby(feat_col, as_index=False)
             .agg(v=(sens_col, "mean"))
             .sort_values("v", ascending=False).head(top_n)
             .sort_values("v", ascending=True))
    bars = ax.barh(top[feat_col], top["v"], color=colour)
    offset = max(top["v"].max() * 0.01, 0.001)
    for bar, value in zip(bars, top["v"]):
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                va="center", fontsize=8, color=COL["text"])
    ax.set_xlabel("Mean sensitivity")
    ax.set_title(model_label, color=colour)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    return True


def plot_top_features(feature_df, split):
    if feature_df is None or feature_df.empty:
        print(f"Skipped top features ({split}): no feature file.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ok1 = _top_features_panel(axes[0], feature_df, split, "random", "Random Forest (FGSM)", COL["rf"])
    ok2 = _top_features_panel(axes[1], feature_df, split, "mlp", "MLP (PGD)", COL["mlp"])
    if not (ok1 or ok2):
        plt.close(fig)
        print(f"Skipped top features ({split}): required columns not found for this split.")
        return
    fig.suptitle(f"Most-perturbed attackable features — {split} set", fontsize=13, fontweight="bold")
    figure_caption(fig, "Features the attack leaned on most to disguise a dropout student.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_plot(fig, f"07_{split}_top_sensitive_features.png")


# ============================================================
# TABLES
# ============================================================

def save_tables(combined, prox_df):
    csv_path = OUTPUT_DIR / "zenodo_performance_summary.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    excel_path = OUTPUT_DIR / "zenodo_visualization_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="Performance", index=False)
        used = [filter_proximity(prox_df, s) for s in SPLITS_TO_PLOT]
        used = [u for u in used if u is not None and not u.empty]
        if used:
            pd.concat(used, ignore_index=True).to_excel(writer, sheet_name="Proximity Sensitivity", index=False)
    print(f"Saved: {excel_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    apply_theme()
    print("=" * 80)
    print("ZENODO ADVERSARIAL ROBUSTNESS VISUALIZATION (thesis edition)")
    print("=" * 80)
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Splits           : {list(SPLITS_TO_PLOT)}\n")

    rf_attack = normalize_columns(read_all_csv(RF_ATTACK_CANDIDATES, False, "RF FGSM attack"))
    rf_def = normalize_columns(read_all_csv(RF_DEFENCE_CANDIDATES, True, "RF adversarial training"))
    mlp_attack = normalize_columns(read_all_csv(MLP_ATTACK_CANDIDATES, False, "MLP PGD attack"))
    mlp_def = normalize_columns(read_all_csv(MLP_DEFENCE_CANDIDATES, True, "MLP adversarial training"))
    prox = read_all_csv(PROX_SUMMARY_CANDIDATES, False, "proximity/sensitivity summary")
    feature = read_all_csv(FEATURE_SUMMARY_CANDIDATES, False, "feature sensitivity")

    all_perf = []
    for split in SPLITS_TO_PLOT:
        print(f"\n---- {split} ----")
        perf = build_performance_summary(split, rf_attack, rf_def, mlp_attack, mlp_def)
        if perf.empty:
            print(f"No rows for split '{split}'.")
            continue
        cols = ["model", "stage", "eval_epsilon", "accuracy", "precision", "recall",
                "f1_score", "dropout_hiding_rate"]
        print(perf[[c for c in cols if c in perf.columns]].to_string(index=False))
        plot_f1_journey(perf, split)
        plot_drop_recovery(perf, split)
        plot_prf_under_attack(perf, split)
        plot_hiding_rate(perf, split)
        plot_clean_cost(perf, split)
        plot_proximity(prox, split)
        plot_top_features(feature, split)
        all_perf.append(perf)

    if not all_perf:
        raise RuntimeError("no performance rows for any split. check the result folders.")
    save_tables(pd.concat(all_perf, ignore_index=True), prox)
    print("\nDone. Figures and tables saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
