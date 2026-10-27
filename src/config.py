"""Path/config helpers shared by all modules."""
import os
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "experiment_config.yaml")

_CACHE = None


def load_config(reload: bool = False) -> dict:
    global _CACHE
    if _CACHE is None or reload:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _CACHE = yaml.safe_load(f)
    return _CACHE


def p(rel: str) -> str:
    """Resolve a repo-relative path to an absolute path."""
    return os.path.normpath(os.path.join(REPO_ROOT, rel))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
