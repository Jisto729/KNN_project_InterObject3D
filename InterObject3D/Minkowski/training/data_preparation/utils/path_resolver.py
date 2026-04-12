from pathlib import Path

def resolve_path_to_root() -> Path:
    return Path(__file__).resolve().parent.parent