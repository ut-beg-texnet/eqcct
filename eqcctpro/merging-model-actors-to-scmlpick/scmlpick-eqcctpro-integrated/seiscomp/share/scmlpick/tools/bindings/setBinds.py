#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
from pathlib import Path
from typing import Tuple

SRC_DIR = Path("key")           
KEY_DIR = Path("/opt/seiscomp/etc/key")  
BACKUP_SUFFIX = ".bak"                   
DRY_RUN = False                          


def netsta_from_cfg_path(p: Path) -> Tuple[str, str]:
    base = p.stem
    if base.startswith("station_"):
        parts = base.split("_", 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid station key filename: {p.name}")
        net = parts[1]
        sta = parts[2]
    return net, sta


def ensure_scmlpick_line(text: str) -> Tuple[str, bool]:
    lines = text.splitlines()
    new_lines = []
    found_once = False
    changed = False
    pattern = re.compile(r'^\s*scmlpick\b', re.IGNORECASE)

    for line in lines:
        if pattern.match(line):
            if not found_once:
                if line.strip() != "scmlpick":
                    changed = True
                new_lines.append("scmlpick")
                found_once = True
            else:
                changed = True
        else:
            new_lines.append(line)

    if not found_once:
        new_lines.append("scmlpick")
        changed = True

    new_text = "\n".join(new_lines)
    if not new_text.endswith("\n"):
        new_text += "\n"
    return new_text, changed

def process_station_key(key_path: Path) -> None:
    if not key_path.exists():
        print(f"[NEW] {key_path}  -> creating with 'scmlpick'")
        if not DRY_RUN:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_text("scmlpick\n", encoding="utf-8")
        return

    original = key_path.read_text(encoding="utf-8")
    new_text, changed = ensure_scmlpick_line(original)

    if not changed:
        print(f"[OK ] {key_path}  -> already valid")
        return

    print(f"[FIX] {key_path}  -> updating to a single clean 'scmlpick' line")
    if not DRY_RUN:
        backup_path = key_path.with_suffix(key_path.suffix + BACKUP_SUFFIX)
        shutil.copy2(key_path, backup_path)
        key_path.write_text(new_text, encoding="utf-8")


def main():
    if not SRC_DIR.is_dir():
        raise SystemExit(f"[ERR] Source directory not found: {SRC_DIR}")

    count = 0
    for b in sorted(SRC_DIR.glob("station_*")):
        try:
            net, sta = netsta_from_cfg_path(b)
        except ValueError as e:
            print(f"[SKIP] {b.name}: {e}")
            continue

        key_name = f"station_{net}_{sta}"
        key_path = KEY_DIR / key_name
        process_station_key(key_path)
        count += 1

    print(f"[DONE] Processed {count} stations from: {SRC_DIR}")
    print(f"       Key directory: {KEY_DIR}")
    if DRY_RUN:
        print("       (dry-run mode: no files were modified)")

if __name__ == "__main__":
    main()
