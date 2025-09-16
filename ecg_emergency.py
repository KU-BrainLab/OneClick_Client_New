# Retry: Replace the 17th column of the second CSV with the 17th column of the first CSV
# and save the modified file. Provide a preview table.

import pandas as pd
from pathlib import Path
import os

first_path = Path("data/2025-09-02-1010.csv")
second_path = Path("data/2025-09-09-1644.csv")



def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, sep=None, engine="python")

df1 = read_csv_smart(first_path)
df2 = read_csv_smart(second_path)

if df1.shape[1] < 17 or df2.shape[1] < 17:
    diag = pd.DataFrame([{
        "first_csv_path": str(first_path),
        "second_csv_path": str(second_path),
        "first_csv_shape": df1.shape,
        "second_csv_shape": df2.shape,
    }])

    print("ERROR: One or both CSVs have fewer than 17 columns.")
else:
    # Save original for preview
    orig_second_col = df2.iloc[:, 16].copy()
    first_col = df1.iloc[:, 16].copy()

    n = min(len(df1), len(df2))
    df2.iloc[:n, 16] = first_col.iloc[:n].values

    preview_rows = min(12, n)
    preview = pd.DataFrame({
        "row_idx": list(range(preview_rows)),
        "first_csv_col17": first_col.iloc[:preview_rows].values,
        "second_csv_col17_before": orig_second_col.iloc[:preview_rows].values,
        "second_csv_col17_after": df2.iloc[:preview_rows, 16].values,
    })

    base, ext = os.path.splitext(second_path)
    out_path = f"{base}_ecg_dummyed{ext}"

    df2.to_csv(out_path, index=False)

    print("SUCCESS")
    print("Output file:", out_path)