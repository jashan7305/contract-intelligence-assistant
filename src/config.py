from pathlib import Path
import yaml

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("benchmark_path", "corpus_dir", "cache_dir"):
        cfg["data"][key] = str(Path(cfg["data"][key]))
    return cfg
