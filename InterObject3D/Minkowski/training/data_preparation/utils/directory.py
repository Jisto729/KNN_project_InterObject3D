from pathlib import Path

def parse_directories(dir_path: Path) -> list:
    return sorted([p for p in dir_path.iterdir() if p.is_dir()])

def parse_files(dir_path: Path, suffix: str = None) -> list:
    if suffix:
        return sorted(dir_path.glob(f"*{suffix}"))
    return sorted([p for p in dir_path.iterdir() if p.is_file()])