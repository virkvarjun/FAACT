#!/usr/bin/env python3
"""Patch lerobot import_utils to handle packages with corrupted metadata (UnicodeDecodeError)."""
import sys
from pathlib import Path

LEROBOT_UTILS = Path(__file__).resolve().parent.parent / "lerobot" / "src" / "lerobot" / "utils" / "import_utils.py"

OLD = """    for dist in importlib.metadata.distributions():
        dist_name = dist.metadata.get("Name")
        if not dist_name:
            continue"""
NEW = """    for dist in importlib.metadata.distributions():
        try:
            dist_name = dist.metadata.get("Name")
        except (UnicodeDecodeError, OSError):
            continue
        if not dist_name:
            continue"""

def main():
    path = LEROBOT_UTILS
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    text = path.read_text()
    if "except (UnicodeDecodeError, OSError):" in text:
        print("Patch already applied.")
        return
    if OLD.strip() not in text:
        print("Error: Could not find target block to patch.", file=sys.stderr)
        sys.exit(1)
    new_text = text.replace(OLD, NEW)
    path.write_text(new_text)
    print("Patch applied successfully.")

if __name__ == "__main__":
    main()
