#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concept-layer metric wrapper for Habitat-CBM 3D outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from srcs.eval_habitat_cbm_concepts import evaluate_concepts  # noqa: E402


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Habitat-CBM 3D concept predictions.")
    parser.add_argument("--patient-concepts-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--scale", type=str, default="raw", choices=("raw", "std"))
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    evaluate_concepts(
        patient_concepts_csv=args.patient_concepts_csv,
        output_dir=args.output_dir,
        split_filter=args.split,
        use_scale=args.scale,
    )
    print("3D concept evaluation exported:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  split     : {args.split}")
    print(f"  scale     : {args.scale}")


if __name__ == "__main__":
    main()

