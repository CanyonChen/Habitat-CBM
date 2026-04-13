#!/usr/bin/env python3
"""
胶质瘤 IDH 数据集患者级分层划分工具

功能说明:
    该脚本用于将胶质瘤 IDH 数据集按照患者级别进行分层划分，
    划分为训练集(train)、验证集(val)和测试集(test)。
    支持保持 IDH 突变型(mutant)和野生型(wild_type)的类别比例。

命令行运行示例:
    # 基本用法 - 使用默认参数 (训练集70%, 验证集10%, 测试集20%)
    python data_split.py --dataset-root /path/to/images --output-root /path/to/output

    # 自定义划分比例 (例如: 训练集80%, 验证集10%, 测试集10%)
    python data_split.py --dataset-root /path/to/images --output-root /path/to/output \
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

    # 使用复制模式而非符号链接 (将实际复制文件夹内容)
    python data_split.py --dataset-root /path/to/images --output-root /path/to/output \
        --link-mode copy

    # 只生成分配清单而不创建文件夹链接
    python data_split.py --dataset-root /path/to/images --output-root /path/to/output \
        --link-mode none

    # 指定不同的随机种子 (用于重现不同的划分结果)
    python data_split.py --dataset-root /path/to/images --output-root /path/to/output \
        --seed 123

    # 覆盖已存在的输出目录
    python data_split.py --dataset-root /path/to/images --output-root /path/to/output \
        --overwrite

    # 完整示例 - 自定义所有参数
    python data_split.py \
        --dataset-root /root/autodl-tmp/habitat_CBM/data/images \
        --output-root /root/autodl-tmp/habitat_CBM/data/split_output \
        --train-ratio 0.7 \
        --val-ratio 0.10 \
        --test-ratio 0.20 \
        --seed 42 \
        --csv-name idh.csv \
        --link-mode copy \
        --overwrite

预期输入目录结构:
    dataset_root/
    ├─ idh.csv                    # 患者信息CSV文件
    ├─ conventional/              # 常规成像分支
    │  ├─ mutant/                 # IDH突变型患者文件夹
    │  └─ wild_type/              # IDH野生型患者文件夹
    └─ functional/                # 功能性成像分支
       ├─ mutant/
       └─ wild_type/

输出目录结构:
    output_root/
    ├─ manifests/                 # 划分清单文件
    │  ├─ split_assignments.csv   # 完整的患者划分分配表
    │  ├─ train_ids.txt           # 训练集患者ID列表
    │  ├─ val_ids.txt             # 验证集患者ID列表
    │  ├─ test_ids.txt            # 测试集患者ID列表
    │  ├─ train.csv               # 训练集患者信息CSV
    │  ├─ val.csv                 # 验证集患者信息CSV
    │  └─ test.csv                # 测试集患者信息CSV
    ├─ train/                     # 训练集数据(符号链接或复制)
    ├─ val/                       # 验证集数据
    └─ test/                      # 测试集数据

注意事项:
    - 默认使用符号链接(--link-mode symlink)以避免重复复制大文件
    - 划分是按患者级别进行的，确保同一患者不会出现在不同集合中
    - 采用分层划分策略，保持各类别比例
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

SEED = 42
BRANCHES = ("conventional", "functional")
VOI_SOURCE_BRANCH = "functional"
LABEL_MAP = {"0": "wild_type", "1": "mutant"}
ALLOWED_EXTENSIONS = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")
DEFAULT_VOI_KEYWORDS = ("voi",)


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    idh_label: str
    idh_class: str
    raw: Dict[str, str]


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def is_supported_image(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def match_voi(path: Path) -> bool:
    name = normalize_name(path.name)
    return any(keyword in name for keyword in DEFAULT_VOI_KEYWORDS)


def score_voi_candidate(path: Path) -> int:
    name = normalize_name(path.name)
    if name.startswith("voi.") or name.startswith("voi_"):
        return 1000
    score = 0
    for rank, keyword in enumerate(DEFAULT_VOI_KEYWORDS):
        if keyword in name:
            score += 100 - rank
    return score


def collect_patient_image_files(patient_dirs: Dict[str, Path]) -> List[Path]:
    all_files: List[Path] = []
    for branch in BRANCHES:
        branch_dir = patient_dirs[branch]
        all_files.extend(
            [
                path
                for path in branch_dir.rglob("*")
                if path.is_file() and is_supported_image(path)
            ]
        )
    return all_files


def collect_branch_image_files(branch_dir: Path) -> List[Path]:
    return [
        path
        for path in branch_dir.rglob("*")
        if path.is_file() and is_supported_image(path)
    ]


def discover_voi_file(patient_dirs: Dict[str, Path]) -> Path:
    functional_dir = patient_dirs[VOI_SOURCE_BRANCH]
    all_files = collect_branch_image_files(functional_dir)
    matches = [path for path in all_files if match_voi(path)]
    if len(matches) == 0:
        raise FileNotFoundError(
            "Cannot find canonical VOI file under the functional branch: "
            f"{functional_dir}"
        )
    if len(matches) == 1:
        return matches[0]

    scored = sorted(matches, key=lambda path: score_voi_candidate(path), reverse=True)
    best_score = score_voi_candidate(scored[0])
    best_matches = [path for path in scored if score_voi_candidate(path) == best_score]
    if len(best_matches) > 1:
        match_str = "\n".join(str(path) for path in best_matches[:10])
        raise ValueError(
            f"Found multiple equally plausible VOI files under {functional_dir}. "
            "Please keep only one canonical functional/voi file or refine the rules.\n"
            f"{match_str}"
        )
    return best_matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a patient-level stratified split for the glioma IDH dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing idh.csv, conventional/, and functional/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory used to save split manifests and split folders.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio. Default: 0.7.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio. Default: 0.1.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio. Default: 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Global random seed used for the split. Default: 42.",
    )
    parser.add_argument(
        "--csv-name",
        default="idh.csv",
        help="CSV file name under dataset-root. Default: idh.csv.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy", "none"),
        default="copy",
        help="How to materialize split folders. Default: copy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-root before writing new results.",
    )
    parser.add_argument(
        "--require-voi",
        type=str2bool,
        default=True,
        help="Require every patient to have a discoverable canonical functional/voi file. Default: true.",
    )
    return parser.parse_args()


def ensure_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(r <= 0 for r in ratios):
        raise ValueError("train/val/test ratios must all be positive.")
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total:.6f}.")


def read_records(csv_path: Path) -> List[PatientRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required_columns = {"patient", "IDH"}
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    missing = required_columns - set(rows[0].keys())
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    records: List[PatientRecord] = []
    seen_ids = set()
    for row in rows:
        patient_id = f"{int(row['patient']):03d}"
        idh_label = str(row["IDH"]).strip()
        if idh_label not in LABEL_MAP:
            raise ValueError(f"Unsupported IDH label for patient {patient_id}: {idh_label}")
        if patient_id in seen_ids:
            raise ValueError(f"Duplicated patient id in CSV: {patient_id}")
        seen_ids.add(patient_id)
        records.append(
            PatientRecord(
                patient_id=patient_id,
                idh_label=idh_label,
                idh_class=LABEL_MAP[idh_label],
                raw=row,
            )
        )
    return records


def validate_layout(
    dataset_root: Path,
    records: Sequence[PatientRecord],
    require_voi: bool = True,
) -> None:
    missing_dirs: List[str] = []
    mismatched_dirs: List[str] = []
    missing_voi: List[str] = []
    ambiguous_voi: List[str] = []
    known_patients = {record.patient_id for record in records}
    patient_branch_dirs: Dict[str, Dict[str, Path]] = {}

    for record in records:
        patient_dirs: Dict[str, Path] = {}
        for branch in BRANCHES:
            patient_dir = dataset_root / branch / record.idh_class / record.patient_id
            if not patient_dir.is_dir():
                missing_dirs.append(str(patient_dir))
                continue
            patient_dirs[branch] = patient_dir
        patient_branch_dirs[record.patient_id] = patient_dirs

    for branch in BRANCHES:
        branch_root = dataset_root / branch
        if not branch_root.is_dir():
            missing_dirs.append(str(branch_root))
            continue
        for label_name in LABEL_MAP.values():
            label_root = branch_root / label_name
            if not label_root.is_dir():
                missing_dirs.append(str(label_root))
                continue
            for child in label_root.iterdir():
                if not child.is_dir():
                    continue
                if child.name not in known_patients:
                    mismatched_dirs.append(str(child))

    if require_voi:
        for record in records:
            patient_dirs = patient_branch_dirs.get(record.patient_id, {})
            if any(branch not in patient_dirs for branch in BRANCHES):
                continue
            try:
                discover_voi_file(patient_dirs)
            except FileNotFoundError as exc:
                missing_voi.append(str(exc))
            except ValueError as exc:
                ambiguous_voi.append(str(exc))

    if missing_dirs or mismatched_dirs or missing_voi or ambiguous_voi:
        problems = []
        if missing_dirs:
            preview = "\n".join(missing_dirs[:10])
            problems.append(f"Missing required directories (showing up to 10):\n{preview}")
        if mismatched_dirs:
            preview = "\n".join(mismatched_dirs[:10])
            problems.append(f"Orphan patient directories (showing up to 10):\n{preview}")
        if missing_voi:
            preview = "\n".join(missing_voi[:10])
            problems.append(f"Missing VOI masks (showing up to 10):\n{preview}")
        if ambiguous_voi:
            preview = "\n".join(ambiguous_voi[:10])
            problems.append(f"Ambiguous VOI masks (showing up to 10):\n{preview}")
        raise FileNotFoundError("\n\n".join(problems))


def stratified_split(
    records: Sequence[PatientRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[PatientRecord]]:
    del test_ratio  # implied by remaining samples after train and val sizes are computed

    rng = random.Random(seed)
    by_label: Dict[str, List[PatientRecord]] = {"0": [], "1": []}
    for record in records:
        by_label[record.idh_label].append(record)

    splits = {"train": [], "val": [], "test": []}
    for label, group in by_label.items():
        group = list(group)
        rng.shuffle(group)

        n_total = len(group)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))

        # Ensure each split remains valid after rounding.
        if n_total >= 3:
            n_train = min(max(n_train, 1), n_total - 2)
            n_val = min(max(n_val, 1), n_total - n_train - 1)
        n_test = n_total - n_train - n_val
        if n_test <= 0:
            raise ValueError(
                f"Label {label} does not leave any sample for test split. "
                f"Counts: total={n_total}, train={n_train}, val={n_val}, test={n_test}"
            )

        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    for split_name in splits:
        splits[split_name] = sorted(splits[split_name], key=lambda x: int(x.patient_id))
    return splits


def write_text_list(path: Path, rows: Iterable[str]) -> None:
    values = list(rows)
    suffix = "\n" if values else ""
    path.write_text("\n".join(values) + suffix, encoding="utf-8")


def write_subset_csv(path: Path, records: Sequence[PatientRecord], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.raw)


def write_manifest(
    manifest_path: Path,
    splits: Dict[str, Sequence[PatientRecord]],
    seed: int,
) -> None:
    fieldnames = ["patient_id", "idh_label", "idh_class", "split", "random_seed", "split_version"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            for record in splits[split_name]:
                writer.writerow(
                    {
                        "patient_id": record.patient_id,
                        "idh_label": record.idh_label,
                        "idh_class": record.idh_class,
                        "split": split_name,
                        "random_seed": seed,
                        "split_version": "v1",
                    }
                )


def link_or_copy_tree(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Destination already exists: {dst}")

    if link_mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
        return
    if link_mode == "copy":
        shutil.copytree(src, dst)
        return
    if link_mode == "none":
        return
    raise ValueError(f"Unsupported link mode: {link_mode}")


def materialize_split_dirs(
    dataset_root: Path,
    output_root: Path,
    splits: Dict[str, Sequence[PatientRecord]],
    link_mode: str,
) -> None:
    for split_name, records in splits.items():
        split_root = output_root / split_name
        for branch in BRANCHES:
            for label_name in LABEL_MAP.values():
                (split_root / branch / label_name).mkdir(parents=True, exist_ok=True)

        for record in records:
            for branch in BRANCHES:
                src = dataset_root / branch / record.idh_class / record.patient_id
                dst = split_root / branch / record.idh_class / record.patient_id
                link_or_copy_tree(src, dst, link_mode)


def summarise(splits: Dict[str, Sequence[PatientRecord]]) -> str:
    lines = []
    total = sum(len(rows) for rows in splits.values())
    lines.append(f"Total patients: {total}")
    for split_name in ("train", "val", "test"):
        records = splits[split_name]
        counter = Counter(record.idh_label for record in records)
        lines.append(
            f"{split_name}: total={len(records)}, "
            f"wild_type={counter.get('0', 0)}, mutant={counter.get('1', 0)}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    csv_path = dataset_root / args.csv_name

    records = read_records(csv_path)
    validate_layout(dataset_root, records, require_voi=args.require_voi)

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    splits = stratified_split(
        records=records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    fieldnames = list(records[0].raw.keys())
    write_manifest(manifests_root / "split_assignments.csv", splits, args.seed)
    for split_name in ("train", "val", "test"):
        write_text_list(
            manifests_root / f"{split_name}_ids.txt",
            [record.patient_id for record in splits[split_name]],
        )
        write_subset_csv(manifests_root / f"{split_name}.csv", splits[split_name], fieldnames)

    materialize_split_dirs(
        dataset_root=dataset_root,
        output_root=output_root,
        splits=splits,
        link_mode=args.link_mode,
    )

    print(f"Random seed: {args.seed}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output root: {output_root}")
    print(f"Link mode: {args.link_mode}")
    print(summarise(splits))


if __name__ == "__main__":
    main()
