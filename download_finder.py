from huggingface_hub import snapshot_download


def main() -> None:
    snapshot_download(
        repo_id="Linq-AI-Research/FinDER",
        repo_type="dataset",
        local_dir="FinDER",
    )
    print("FinDER dataset downloaded to ./FinDER")


if __name__ == "__main__":
    main()
