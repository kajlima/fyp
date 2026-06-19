"""
Muat 100 baris pertama dari beberapa dataset dan periksa fitur-fitur pilihan.

Tujuan:
1. Memuat HANYA 100 baris pertama dari tiap dataset (hemat memori).
2. Mengambil hanya kolom/fitur yang dipilih.
3. Memeriksa apakah fitur tersebut benar-benar berisi nilai (0, 1, 2, ... dst)
   atau hanya berisi nilai kosong (blank / NaN).

Cara pakai:
    python load_and_check_features.py
atau arahkan ke folder lain:
    python load_and_check_features.py --data-dir "C:/path/ke/folder/csv"
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

# Dataset yang ada di screenshot.
DATASETS = [
    "dataset_2018_hash.csv",
    "dataset_2021_hash.csv",
    "dataset_2022_hash.csv",
]

# Fitur pilihan (sesuai screenshot kolom "feature").
SELECTED_FEATURES = [
    "cred_sup_total",
    "cred_sup_tit",
    "cred_sup",
    "cred_sup_espec",
    "cred_ptes_acta",
    "cred_sup_anu",
    "cred_mat_normal",
    "cred_mat_practicas",
    "cred_mat_movilidad",
    "cred_mat_anu",
]

N_ROWS = 100


def load_first_rows(path: str, features: list[str], n_rows: int = N_ROWS) -> pd.DataFrame:
    """Muat n_rows pertama dan hanya kolom yang ada di `features`.

    Kita baca dulu header agar tidak error kalau ada fitur yang namanya tak ada
    di file tersebut; fitur yang hilang akan dilewati (dan dilaporkan).
    """
    header = pd.read_csv(path, nrows=0)
    available = [c for c in features if c in header.columns]
    missing = [c for c in features if c not in header.columns]

    if missing:
        print(f"  [!] Fitur tidak ditemukan di file ini: {missing}")
    if not available:
        print("  [!] Tidak ada satu pun fitur pilihan di file ini. Dilewati.")
        return pd.DataFrame()

    # usecols -> hanya muat kolom yang dibutuhkan; nrows -> hanya 100 baris pertama.
    return pd.read_csv(path, usecols=available, nrows=n_rows)


def check_blank_or_values(df: pd.DataFrame) -> pd.DataFrame:
    """Buat ringkasan per fitur: apakah ada nilai nyata atau hanya blank."""
    summary_rows = []
    total = len(df)

    for col in df.columns:
        s = df[col]
        # Paksa jadi numerik agar bisa hitung 0/min/maks; teks/blank -> NaN.
        numeric = pd.to_numeric(s, errors="coerce")

        non_null = int(numeric.notna().sum())
        blank = total - non_null
        zeros = int((numeric == 0).sum())
        non_zero = int((numeric.fillna(0) != 0).sum())

        if non_null == 0:
            verdict = "SEMUA BLANK (tidak ada nilai)"
        elif non_zero == 0:
            verdict = "ADA NILAI, tapi semuanya 0"
        else:
            verdict = "ADA NILAI nyata (0..dst)"

        summary_rows.append(
            {
                "feature": col,
                "non_null": non_null,
                "blank/NaN": blank,
                "n_zeros": zeros,
                "n_nonzero": non_zero,
                "min": numeric.min(),
                "max": numeric.max(),
                "verdict": verdict,
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Folder tempat file CSV berada (default: folder saat ini).",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=N_ROWS,
        help=f"Jumlah baris pertama yang dimuat (default: {N_ROWS}).",
    )
    args = parser.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for name in DATASETS:
        path = os.path.join(args.data_dir, name)
        print("\n" + "=" * 70)
        print(f"DATASET: {name}")
        print("=" * 70)

        if not os.path.exists(path):
            print(f"  [!] File tidak ditemukan: {path} (lewati)")
            continue

        df = load_first_rows(path, SELECTED_FEATURES, n_rows=args.n_rows)
        if df.empty:
            continue

        print(f"\n100 baris pertama (bentuk: {df.shape[0]} baris x {df.shape[1]} kolom)")
        print(df.head(100).to_string(index=False))

        print("\nRINGKASAN: blank vs ada nilai")
        print(check_blank_or_values(df).to_string(index=False))


if __name__ == "__main__":
    main()
