#!/usr/bin/env python3
"""Set the LAVIS cache_root used by dataset configs."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update lavis/configs/default*.yml cache_root.")
    parser.add_argument("--cache-root", required=True, help="Directory for LAVIS datasets and checkpoints.")
    parser.add_argument("--no-backup", action="store_true", help="Do not write a timestamped backup.")
    return parser.parse_args()


def find_lavis_default_config() -> Path:
    import lavis

    lavis_dir = Path(lavis.__file__).resolve().parent
    candidates = [
        lavis_dir / "configs" / "default.yaml",
        lavis_dir / "configs" / "default.yml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    formatted = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find a LAVIS default config. Tried:\n{formatted}")


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    config_path = find_lavis_default_config()
    cfg = OmegaConf.load(config_path)

    if "env" not in cfg:
        cfg.env = {}
    cfg.env.cache_root = str(cache_root)

    if not args.no_backup:
        backup = config_path.with_suffix(config_path.suffix + f".bak.{int(time.time())}")
        shutil.copy2(config_path, backup)
        print(f"Backup: {backup}")

    OmegaConf.save(cfg, config_path)
    print(f"LAVIS cache_root set to: {cache_root}")
    print(f"Updated: {config_path}")


if __name__ == "__main__":
    main()
