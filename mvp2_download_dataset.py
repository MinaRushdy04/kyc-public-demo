from pathlib import Path

from huggingface_hub import snapshot_download

from modules.utils import simple_yaml_load


def main() -> None:
    cfg = simple_yaml_load("config_mvp2.yaml")
    repo = str(cfg["dataset_repo"])
    target = Path(str(cfg["raw_dataset_path"]))
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(target))
    print(f"Downloaded/verified MVP2 dataset at: {path}")


if __name__ == "__main__":
    main()
