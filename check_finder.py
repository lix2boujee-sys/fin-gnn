from pathlib import Path

import pandas as pd


def main() -> None:
    data_dir = Path("FinDER") / "data"
    parquet_files = sorted(data_dir.glob("*.parquet"))

    print("Found parquet files:")
    for path in parquet_files:
        print(path)

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    df = pd.read_parquet(parquet_files[0])
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print(df.head())


if __name__ == "__main__":
    main()
