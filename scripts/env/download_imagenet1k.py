#!/usr/bin/env python3
"""Download ImageNet-1k from Hugging Face and optionally upload it elsewhere.

ImageNet-1k is gated. Request access for your Hugging Face account first, then
authenticate with either `huggingface-cli login`, `HF_TOKEN`, or `--hf-token`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET = "imagenet-1k"
DEFAULT_CACHE_DIR = Path.cwd() / "data" / "ic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download gated ImageNet-1k and optionally upload the result with rsync."
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset name. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("IC_CACHE_DIR", str(DEFAULT_CACHE_DIR)),
        help="Hugging Face cache directory. Default: $IC_CACHE_DIR or ./data/ic",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Prefer HF_TOKEN or --hf-token-file on shared servers.",
    )
    parser.add_argument(
        "--hf-token-file",
        default=os.environ.get("HF_TOKEN_FILE"),
        help="Path to a file containing a Hugging Face token.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Dataset splits to download/export. Default: train validation",
    )
    parser.add_argument(
        "--export-imagefolder",
        default=None,
        help=(
            "Optional directory for a PyTorch ImageFolder export, e.g. "
            "/data/imagenet1k_imagefolder. Produces train/<class>/ and val/<class>/."
        ),
    )
    parser.add_argument(
        "--max-items-per-split",
        type=int,
        default=None,
        help="Debug option: export/download only the first N items of each split.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Hugging Face download and only run export/upload from existing data.",
    )
    parser.add_argument(
        "--upload-to",
        default=None,
        help="Optional rsync target, e.g. user@server:/data/imagenet1k/",
    )
    parser.add_argument(
        "--upload-source",
        default=None,
        help="Optional path to upload. Default: export directory if set, otherwise cache directory.",
    )
    parser.add_argument(
        "--delete-remote",
        action="store_true",
        help="Pass --delete to rsync so the remote mirror exactly matches the source.",
    )
    return parser.parse_args()


def read_token(args: argparse.Namespace) -> str | None:
    if args.hf_token:
        return args.hf_token.strip()

    if args.hf_token_file:
        token_path = Path(args.hf_token_file).expanduser()
        if not token_path.is_file():
            raise FileNotFoundError(f"HF token file does not exist: {token_path}")
        return token_path.read_text(encoding="utf-8").strip()

    return None


def load_imagenet(args: argparse.Namespace):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Python package 'datasets' is missing. Run `bash scripts/env/init_environment.sh` first."
        ) from exc

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = read_token(args)

    kwargs = {"cache_dir": str(cache_dir)}
    if token:
        kwargs["token"] = token

    split_arg: str | list[str]
    split_arg = args.splits[0] if len(args.splits) == 1 else args.splits

    try:
        dataset = load_dataset(args.dataset_name, split=split_arg, **kwargs)
    except TypeError as exc:
        if "token" not in str(exc) or not token:
            raise
        kwargs.pop("token", None)
        kwargs["use_auth_token"] = token
        dataset = load_dataset(args.dataset_name, split=split_arg, **kwargs)
    except Exception as exc:
        message = str(exc)
        access_words = ("gated", "access", "401", "403", "Unauthorized")
        if any(word.lower() in message.lower() for word in access_words):
            raise SystemExit(
                "Hugging Face denied access to ImageNet-1k.\n"
                "Request access at https://huggingface.co/datasets/imagenet-1k, then run:\n"
                "  HF_TOKEN=hf_xxx python scripts/env/download_imagenet1k.py"
            ) from exc
        raise

    if len(args.splits) == 1:
        dataset = {args.splits[0]: dataset}
    else:
        dataset = dict(zip(args.splits, dataset, strict=True))

    print(f"Cached {args.dataset_name} in {cache_dir}")
    for split_name, split_dataset in dataset.items():
        print(f"  {split_name}: {len(split_dataset)} examples")

    return dataset


def sanitize_class_name(name: str, label_id: int) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return cleaned or f"label_{label_id:04d}"


def split_output_name(split: str) -> str:
    return "val" if split == "validation" else split


def iter_limited(dataset, limit: int | None) -> Iterable[tuple[int, dict]]:
    total = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(total):
        yield index, dataset[index]


def export_imagefolder(dataset_by_split: dict, export_root: str, limit: int | None) -> Path:
    export_dir = Path(export_root).expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in dataset_by_split.items():
        output_split = export_dir / split_output_name(split_name)
        label_feature = split_dataset.features.get("label")
        label_names = getattr(label_feature, "names", None)

        for index, example in iter_limited(split_dataset, limit):
            label_id = int(example["label"])
            if label_names and 0 <= label_id < len(label_names):
                class_name = sanitize_class_name(label_names[label_id], label_id)
            else:
                class_name = f"label_{label_id:04d}"

            image = example["image"]
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            class_dir = output_split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            image.save(class_dir / f"{index:08d}.jpg", quality=95)

            if index and index % 10000 == 0:
                print(f"  exported {split_name}: {index} images")

        print(f"Exported {split_name} to {output_split}")

    return export_dir


def run_upload(source: str, target: str, delete_remote: bool) -> None:
    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Upload source does not exist: {source_path}")

    source_arg = str(source_path)
    if source_path.is_dir() and not source_arg.endswith("/"):
        source_arg += "/"

    command = ["rsync", "-a", "--info=progress2"]
    if delete_remote:
        command.append("--delete")
    command.extend([source_arg, target])

    print("Uploading with:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    dataset_by_split = None

    if not args.skip_download or args.export_imagefolder:
        dataset_by_split = load_imagenet(args)

    upload_source = args.upload_source
    if args.export_imagefolder:
        if dataset_by_split is None:
            raise RuntimeError("ImageFolder export requires a loaded dataset.")
        upload_source = str(
            export_imagefolder(dataset_by_split, args.export_imagefolder, args.max_items_per_split)
        )

    if args.upload_to:
        if upload_source is None:
            upload_source = args.cache_dir
        run_upload(upload_source, args.upload_to, args.delete_remote)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
