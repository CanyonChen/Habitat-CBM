#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patient-level concept intervention for Habitat-CBM 3D checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import srcs.intervene_habitat_cbm as intervene2d  # noqa: E402
from models.habitat_CBM_3D import HabitatCBM3D  # noqa: E402


class _HabitatCBM3DInterventionAdapter(HabitatCBM3D):
    """Adapter matching the 2D intervention loader constructor."""

    def __init__(
        self,
        in_channels: int,
        n_concepts: int,
        concept_hidden_dim: int = 256,
        label_hidden_dim: int = 32,
        dropout_p: float | None = None,
        concept_dropout_p: float | None = None,
        label_dropout_p: float | None = None,
        pretrained: bool = False,
        pretrain_path: str | Path | None = None,
    ) -> None:
        del pretrained
        super().__init__(
            in_channels=in_channels,
            n_concepts=n_concepts,
            concept_hidden_dim=concept_hidden_dim,
            label_hidden_dim=label_hidden_dim,
            dropout_p=dropout_p,
            concept_dropout_p=concept_dropout_p,
            label_dropout_p=label_dropout_p,
            pretrain_path=None,
        )


def run_intervention(
    checkpoint_path: Path,
    patient_predictions_csv: Path,
    patient_concepts_csv: Path,
    concept_scaler_json: Path,
    output_dir: Path,
    split: str,
    budgets_text: str,
    threshold: float,
    low_conf_margin: float,
    intervene_scope: str,
    ranking_policy: str,
    concept_whitelist_text: str,
    early_stop_mode: str,
    device_name: str,
    in_channels: int,
    n_concepts: int,
) -> None:
    previous_cls = intervene2d.HabitatCBM
    intervene2d.HabitatCBM = _HabitatCBM3DInterventionAdapter
    try:
        intervene2d.run_intervention(
            checkpoint_path=checkpoint_path,
            patient_predictions_csv=patient_predictions_csv,
            patient_concepts_csv=patient_concepts_csv,
            concept_scaler_json=concept_scaler_json,
            output_dir=output_dir,
            split=split,
            budgets_text=budgets_text,
            threshold=threshold,
            low_conf_margin=low_conf_margin,
            intervene_scope=intervene_scope,
            ranking_policy=ranking_policy,
            concept_whitelist_text=concept_whitelist_text,
            early_stop_mode=early_stop_mode,
            device_name=device_name,
            in_channels=in_channels,
            n_concepts=n_concepts,
        )
    finally:
        intervene2d.HabitatCBM = previous_cls


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run patient-level concept intervention for Habitat-CBM 3D.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patient-predictions-csv", type=Path, required=True)
    parser.add_argument("--patient-concepts-csv", type=Path, required=True)
    parser.add_argument("--concept-scaler-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--budgets", type=str, default="1,2,4,all")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--low-confidence-margin", type=float, default=0.1)
    parser.add_argument(
        "--intervene-scope",
        type=str,
        default=intervene2d._INTERVENE_SCOPE_CANDIDATES_ONLY,
        choices=(intervene2d._INTERVENE_SCOPE_ALL_PATIENTS, intervene2d._INTERVENE_SCOPE_CANDIDATES_ONLY),
    )
    parser.add_argument(
        "--ranking",
        type=str,
        default=intervene2d._RANKING_LOGIT_EFFECT,
        choices=(intervene2d._RANKING_ABS_ERROR, intervene2d._RANKING_LOGIT_EFFECT),
    )
    parser.add_argument("--concept-whitelist", type=str, default="")
    parser.add_argument(
        "--early-stop",
        type=str,
        default=intervene2d._EARLY_STOP_CROSS_THRESHOLD,
        choices=(intervene2d._EARLY_STOP_NONE, intervene2d._EARLY_STOP_CROSS_THRESHOLD),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--n-concepts", type=int, default=5)
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    run_intervention(
        checkpoint_path=args.checkpoint,
        patient_predictions_csv=args.patient_predictions_csv,
        patient_concepts_csv=args.patient_concepts_csv,
        concept_scaler_json=args.concept_scaler_json,
        output_dir=args.output_dir,
        split=args.split,
        budgets_text=args.budgets,
        threshold=args.threshold,
        low_conf_margin=args.low_confidence_margin,
        intervene_scope=args.intervene_scope,
        ranking_policy=args.ranking,
        concept_whitelist_text=args.concept_whitelist,
        early_stop_mode=args.early_stop,
        device_name=args.device,
        in_channels=args.in_channels,
        n_concepts=args.n_concepts,
    )
    print("3D intervention outputs written to:")
    print(f"  {args.output_dir}")


if __name__ == "__main__":
    main()

