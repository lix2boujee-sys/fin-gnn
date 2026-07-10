from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    zip_path = Path("FinDER") / "10-k.zip"
    output_dir = Path("10-k")

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing archive: {zip_path}")

    output_dir.mkdir(exist_ok=True)
    with ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    print(f"Extracted {zip_path} to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
