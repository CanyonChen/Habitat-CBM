#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet-18 基线训练脚本，用于病人级别的 IDH 突变状态预测。

该脚本保持 ResNet-18 主干网络不变，完成一条完整的 baseline 流程：
1. 读取 train / val / test 三个数据划分下的 2.5D block 数据；
2. 使用多模态 2.5D 输入训练一个 ResNet-18 二分类模型，并可将 `functional/voi` 掩模一并输入；
3. 在验证集上选择表现最好的 checkpoint；
4. 将 block 级别的预测概率按病人聚合（均值池化）为 patient-level 结果；
5. 导出与实验协议对应的预测结果、ROC、混淆矩阵、指标汇总和训练日志。

================================================================================
命令行参数详细说明
================================================================================

【路径相关参数】

--split-base-root (Path)
    数据划分根目录，内部应包含 train/、val/、test/ 三个子目录。
    默认值: habitat_CBM/data/splited_data
    示例: --split-base-root /path/to/splited_data

--output-root (Path)
    结果输出根目录，保存预测表、指标表、训练历史、运行摘要等文件。
    默认值: habitat_CBM/results/baseline_ResNet18
    示例: --output-root /path/to/results

--checkpoint-root (Path)
    checkpoint 输出根目录，最佳模型会保存在该目录下对应 run_id 的子目录中。
    默认值: habitat_CBM/results/baseline_ResNet18/checkpoints
    示例: --checkpoint-root /path/to/checkpoints

--run-id (str | None)
    本次运行的标识符；若不传，则默认使用当前时间戳（格式：YYYYMMDD_HHMMSS）。
    默认值: None（自动使用当前时间戳）
    示例: --run-id exp_resnet18_seed42

【数据相关参数】

--modalities (str)
    输入模态列表，逗号分隔。
    可用选项: t1, t1ce, t2, t2flair, adc, cbf
    默认值: t1,t1ce,t2,t2flair,adc,cbf
    示例: --modalities t1,t1ce,t2,t2flair

--require-voi (bool)
    是否要求每位患者都有可用的 `functional/voi` 掩模。
    可用选项: true, false, 1, 0, yes, no, y, n
    默认值: true
    说明: 若设为 true，缺失 VOI 的患者会报错；若 append-voi-mask 或 mask-background-with-voi 为 true，则自动强制为 true。

--append-voi-mask (bool)
    是否把 `functional/voi` 掩模 block 作为额外输入通道拼接到网络输入。
    可用选项: true, false, 1, 0, yes, no, y, n
    默认值: true
    说明: 若开启，输入通道数 = len(modalities) * block_depth + block_depth。
          例如 6 模态 × 5 深度 + 5 = 35 通道。

--mask-background-with-voi (bool)
    是否用 `functional/voi` 将图像背景区域清零。
    可用选项: true, false, 1, 0, yes, no, y, n
    默认值: false
    说明: 若开启，VOI 区域外的像素值设为 0，只保留肿瘤区域信息。

--block-depth (int)
    2.5D block 的深度，即每个样本沿切片方向堆叠多少张相邻切片。
    可用选项: 任意正奇数（如 1, 3, 5, 7, ...）
    默认值: 5
    说明: 例如 5 表示以中心切片为锚点，取前后各 2 张切片（共 5 张）。

--slice-axis (int)
    切片轴，决定沿哪个体数据维度生成 2.5D block。
    可用选项: 0, 1, 2
    默认值: 2
    说明: 0 = 沿 X 轴（冠状面），1 = 沿 Y 轴（矢状面），2 = 沿 Z 轴（横断面/轴位）。

--intensity-norm (str)
    强度归一化方式。
    可用选项: zscore, minmax, none
    默认值: zscore
    说明: zscore = (x - μ) / σ，基于非零区域计算；
          minmax = (x - min) / (max - min)，缩放到 [0, 1]；
          none = 不归一化。

--min-nonzero-voxels (int)
    一个中心切片被保留为有效样本时，至少需要多少个前景体素。
    默认值: 32
    说明: 当启用 VOI 时，按 VOI 的非零体素数计算；否则按参考模态的非零体素数计算。
          默认值从 16 提升至 32，过滤 VOI 区域极小的低质量切片，提升训练样本信噪比。

--cache-volumes (bool)
    是否缓存体数据到内存，减少重复 IO。
    可用选项: true, false, 1, 0, yes, no, y, n
    默认值: true
    说明: 适合中小规模实验；若内存不足，可设为 false。

【训练相关参数】

--batch-size (int)
    训练和评估时 DataLoader 的 batch size。
    默认值: 32
    说明: 根据 GPU 显存调整，常见值: 4, 8, 16, 32。

--epochs (int)
    最大训练轮数。
    默认值: 200
    说明: 实际训练可能在达到此值前因早停而结束。

--lr (float)
    AdamW 优化器的学习率。
    默认值: 5e-6
    可用范围: 建议 1e-6 ~ 1e-4
    说明: 从 2e-5 进一步降低到 5e-6，减缓 fc 层参数更新速度，缓解 best_epoch=3 的极早期过拟合。
          配合 ReduceLROnPlateau，plateau 触发后 lr 可进一步降至 2.5e-6。

--weight-decay (float)
    AdamW 的权重衰减系数（L2 正则化）。
    默认值: 1e-3
    可用范围: 建议 1e-4 ~ 1e-2
    说明: 提升至 1e-3 可增强正则化强度，有效抑制过拟合。

--num-workers (int)
    DataLoader 使用的并行加载进程数。
    默认值: 16
    说明: 设为 0 表示在主进程加载；建议根据 CPU 核心数调整。

--seed (int)
    全局随机种子，用于保证实验可复现。
    默认值: 42

--pretrained (bool)
    是否使用 ImageNet 预训练权重初始化 ResNet-18。
    可用选项: true, false, 1, 0, yes, no, y, n
    默认值: true
    说明: 建议保持 true，小样本医学影像任务中预训练可显著提升性能。

--resize-height (int)
    输入网络前统一 resize 到的高度。
    默认值: 224
    说明: 若 <= 0，则不做 resize；ResNet-18 默认期望 224×224 输入。

--resize-width (int)
    输入网络前统一 resize 到的宽度。
    默认值: 224
    说明: 若 <= 0，则不做 resize。

--use-class-weights (bool)
    是否根据训练集类别频次自动计算类别权重，用于缓解类别不平衡。
    可用选项: true, false, 1, 0, yes, no, y, n
    默认值: true
    说明: 权重计算方式: weight = N / (n_classes × count)，按 patient case 统计而非 block。

--early-stop-patience (int)
    验证集 AUC 连续多少个 epoch 没有提升时触发早停。
    默认值: 20
    说明: 设为 0 可禁用早停；14 人验证集的 AUC 最小分辨率约 0.0156，
          patience=5 会过早停止，建议使用 15 ~ 30。

--early-stop-min-delta (float)
    判定为"有效提升"的最小阈值（验证集监控指标的提升幅度）。
    默认值: 0.005
    可用范围: [0.0, 1.0]
    说明: 只有当验证集 AUC 的提升 > min_delta 时，才认为是有效改进。
          14 人验证集 AUC 最小分辨率 ≈ 0.0156，将 delta 设为 0.005 相当于
          要求改进超过最小单位的 1/3，既能过滤随机波动又不过于严苛。
          设为 0 则任何提升都算改进。

--save-train-predictions (bool)
    是否额外导出训练集上的病人级预测结果。
    可用选项: true, false, 1, 0, yes, no, y, n
    默认值: true
    说明: 若设为 false，只导出 val 和 test 的结果。

--device (str)
    计算设备。
    默认值: cuda（若可用）否则 cpu
    可用选项: cuda, cpu, cuda:0, cuda:1 等
    示例: --device cuda:0

--threshold (float)
    病人级概率转为二分类标签时使用的阈值。
    默认值: 0.5
    可用范围: [0.0, 1.0]
    说明: 病人级概率 >= threshold 则预测为 mutant (1)，否则为 wild_type (0)。

--topk-pool (int)
    患者级聚合时保留的最高置信度 block 数量（Top-K 均值池化）。
    默认值: 0（使用全部 block，简单均值）
    说明: K > 0 时，按 |prob - 0.5| 降序排列，只对置信度最高的 K 个 block 取均值。
          推荐值：设为每位患者典型 block 数量的 1/3（约 15~20）。
          能有效过滤低质量/边缘切片对聚合概率的干扰，改善患者级预测准确性。

--threshold-search (bool)
    是否在训练结束后，基于验证集 Youden Index 自动搜索最优分类阈值。
    可用选项: true, false, 1, 0, yes, no, y, n
    默认值: true
    说明: 若开启，会在 --threshold 固定阈值评估之外，额外搜索使 (TPR - FPR) 最大的
          阈值，并输出使用最优阈值后的各 split 指标，以及测试集预测 CSV。
          不改变主要评估结果，只作为诊断和分析工具。

--log-to-file (bool)
    是否将终端输出同时保存到日志文件。
    可用选项: true, false, 1, 0, yes, no, y, n
    默认值: true
    说明: 若开启，终端所有输出将同时写入 output_dir/training_log_{MODEL_NAME}_{run_id}.txt。

--save-interval (int)
    每隔多少个 epoch 保存一个周期性 checkpoint。
    默认值: 50
    说明: 设为 0 可禁用周期性保存。最终评估仍使用 best checkpoint。

--dropout-p (float)
    ResNet-18 分类头前 Dropout 层的 dropout 概率。
    默认值: 0.7
    可用范围: [0.0, 1.0)
    说明: 设为 0.0 则不添加 Dropout 层；对于训练块数远多于验证患者数的情况
          （如约 6000 blocks vs 14 名患者），Dropout 可显著抑制过拟合。
          建议范围: 0.5 ~ 0.7。

--freeze-layers (str)
    冻结 ResNet-18 backbone 中哪些层（不参与反向传播），用逗号分隔。
    可用选项: none, conv1, layer1, layer2, layer3, layer4
    默认值: conv1,layer1,layer2,layer3,layer4
    说明: 冻结早期层可大幅减少可训练参数量，是小样本场景下最有效的正则化手段之一。
          - none: 不冻结任何层，全量微调（容易过拟合）；
          - conv1,layer1,layer2,layer3: 只训练 layer4 + fc，参数量 ~2.7M；
          - conv1,layer1,layer2,layer3,layer4（默认）: 只训练 fc，参数量 ~1K，
            最大程度抑制小样本过拟合；若 Val AUC 低于 0.75，可回退到 layer3。

--label-smoothing (float)
    CrossEntropyLoss 的 label smoothing 系数。
    默认值: 0.3
    可用范围: [0.0, 0.4)
    说明: 将 one-hot 标签软化为 (1 - ε) 和 ε / (C-1)，防止模型过于自信；
          从 0.2 提升至 0.3，进一步抑制 train AUC=0.9995 的极度自信预测。
          不建议超过 0.4，否则模型无法有效区分类别。设为 0.0 则使用标准交叉熵。

--lr-scheduler (str)
    学习率调度策略。
    默认值: cosine
    可用选项:
      none   - 不使用调度器，学习率全程固定。
      cosine - CosineAnnealingLR，将 lr 从初始值余弦退火至 eta_min=1e-6。
               适合大多数场景，训练后期平滑降低 lr 避免震荡。
      plateau - ReduceLROnPlateau(mode='max', factor=0.5, patience=5)，
                当验证 AUC 连续 5 epoch 没有提升时，将 lr 缩减为原来的 50%。
                适合训练曲线震荡较大的情况。

================================================================================
Checkpoint 保存策略
================================================================================

本脚本采用多模式 checkpoint 保存策略：

1. best.pt（验证集最优模型）
   - 保存时机: 验证集 AUC 提升时
   - 用途: 最终评估和部署
   - 特点: 覆盖式更新，始终保留最优版本

2. last.pt（最新模型）
   - 保存时机: 每个 epoch 结束后
   - 用途: 断点续训、训练状态恢复
   - 特点: 始终为最新状态

3. epoch_{N:03d}.pt（周期性模型）
   - 保存时机: 每隔 --save-interval 个 epoch
   - 用途: 训练过程分析、模型选择、集成学习
   - 特点: 不覆盖，保留历史版本

保存路径结构:
    checkpoint_root/
    └── {run_id}/
        ├── best.pt              # 最优模型
        ├── last.pt              # 最新模型
        ├── epoch_010.pt         # 周期性模型（示例）
        ├── epoch_020.pt
        └── ...

================================================================================
运行示例
================================================================================

1. 使用默认参数运行（单 GPU）：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py

2. 指定数据路径、输出路径和 GPU：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --split-base-root /path/to/splited_data \
       --output-root /path/to/results/01_resnet18_baseline \
       --checkpoint-root /path/to/checkpoints/resnet18_baseline \
       --device cuda

3. 指定模态、batch size、epoch 和 run id：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --modalities t1,t1ce,t2,t2flair,adc,cbf \
       --require-voi true \
       --append-voi-mask true \
       --batch-size 8 \
       --epochs 50 \
       --run-id exp_resnet18_seed42

4. 不使用 VOI 进行训练：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --require-voi false \
       --append-voi-mask false \
       --mask-background-with-voi false

5. 使用 VOI 掩码背景并作为额外通道：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --require-voi true \
       --append-voi-mask true \
       --mask-background-with-voi true \
       --min-nonzero-voxels 32 \
       --early-stop-patience 10 \
       --early-stop-min-delta 0.001

6. 小 batch size 配合更大学习率：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --batch-size 4 \
       --lr 5e-4 \
       --weight-decay 1e-5

7. 自定义所有参数（完整配置示例）：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --split-base-root /path/to/splited_data \
       --output-root /path/to/results \
       --checkpoint-root /path/to/checkpoints \
       --run-id exp_full_custom \
       --modalities t1,t1ce,t2,t2flair,adc,cbf \
       --require-voi true \
       --append-voi-mask true \
       --mask-background-with-voi true \
       --block-depth 5 \
       --slice-axis 2 \
       --intensity-norm zscore \
       --min-nonzero-voxels 16 \
       --cache-volumes true \
       --batch-size 16 \
       --epochs 100 \
       --lr 1e-4 \
       --weight-decay 1e-3 \
       --num-workers 4 \
       --seed 42 \
       --pretrained true \
       --resize-height 224 \
       --resize-width 224 \
       --use-class-weights true \
       --early-stop-patience 20 \
       --early-stop-min-delta 0.005 \
       --dropout-p 0.5 \
       --label-smoothing 0.1 \
       --lr-scheduler cosine \
       --save-train-predictions true \
       --device cuda \
       --threshold 0.5 \
       --log-to-file true \
       --save-interval 10

8. 保存所有 checkpoint（用于后续模型选择或集成）：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --save-interval 5 \
       --epochs 50 \
       --run-id save_all_checkpoints
   
   # 结果: 保存 best.pt, last.pt, epoch_005.pt, epoch_010.pt, ..., epoch_050.pt

9. 只保存 best 和 last（节省磁盘空间）：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --save-interval 0 \
       --run-id minimal_checkpoints

10. 严格早停（需要至少 0.5% 的 AUC 提升才算改进）：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --early-stop-patience 20 \
       --early-stop-min-delta 0.005 \
       --run-id strict_early_stop

11. 宽松早停（微小提升也算有效）：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --early-stop-patience 30 \
       --early-stop-min-delta 0.0 \
       --run-id loose_early_stop

12. 禁用正则化（对照实验）：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --dropout-p 0.0 \
       --label-smoothing 0.0 \
       --lr-scheduler none \
       --run-id ablation_no_regularization

13. 使用 ReduceLROnPlateau 调度器（适合震荡场景）：
   
   python habitat_CBM/repo/srcs/baseline_ResNet18.py \
       --lr-scheduler plateau \
       --lr 5e-4 \
       --run-id exp_plateau_scheduler
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TextIO

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = REPO_ROOT.parent  # 项目根目录: habitat_CBM/

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.resnet_18 import ResNet18Classifier
from srcs.data_loader import DEFAULT_MODALITIES, HabitatIDHBlockDataset, VOI_SOURCE_BRANCH
from srcs.monai_augmentation import MonaiAugmentConfig, build_monai_block_transforms

MODEL_NAME = "resnet18"
DEFAULT_SEED = 42


class TeeLogger:
    """同时将输出写入文件和终端的日志处理器。"""

    def __init__(self, filepath: Path, mode: str = "w") -> None:
        self.file: TextIO = open(filepath, mode, encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, message: str) -> None:
        self.file.write(message)
        self.stdout.write(message)
        self.flush()

    def flush(self) -> None:
        self.file.flush()
        self.stdout.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "TeeLogger":
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.close()


@dataclass
class EpochStats:
    """记录每个 epoch 的关键训练/验证指标，便于后续导出训练历史。"""

    epoch: int
    train_loss: float
    train_block_acc: float
    val_loss: float
    val_auc: float
    val_acc: float
    val_f1: float
    val_sen: float
    val_spe: float
    lr: float


def str2bool(value: str) -> bool:
    """将命令行字符串解析成布尔值。"""

    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_modalities(value: str) -> Sequence[str]:
    """解析逗号分隔的模态字符串，并统一转成小写元组。"""

    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("modalities cannot be empty")
    return tuple(items)


def build_argparser() -> argparse.ArgumentParser:
    """定义脚本支持的全部命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Train and evaluate the ResNet-18 IDH baseline with 2.5D MRI inputs."
    )
    parser.add_argument(
        "--split-base-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "splited_data",
        help="Directory containing train/, val/, and test/ split folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline_ResNet18",
        help="Directory used to save metrics, predictions, and logs.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline_ResNet18" / "checkpoints",
        help="Directory used to save model checkpoints.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional manual run id. Default uses local timestamp.",
    )
    parser.add_argument(
        "--modalities",
        type=parse_modalities,
        default=DEFAULT_MODALITIES,
        help="Comma-separated modality list. Default: t1,t1ce,t2,t2flair,adc,cbf",
    )
    parser.add_argument(
        "--require-voi",
        type=str2bool,
        default=True,
        help="Require a canonical functional/voi mask for every patient.",
    )
    parser.add_argument(
        "--append-voi-mask",
        type=str2bool,
        default=True,
        help="Append the functional/voi block as extra input channels.",
    )
    parser.add_argument(
        "--mask-background-with-voi",
        type=str2bool,
        default=False,
        help="Zero out image background outside the functional/voi mask.",
    )
    parser.add_argument("--block-depth", type=int, default=5, help="2.5D block depth.")
    parser.add_argument("--slice-axis", type=int, default=2, help="Slice axis.")
    parser.add_argument(
        "--intensity-norm",
        choices=("zscore", "minmax", "none"),
        default="zscore",
        help="Per-volume intensity normalization mode.",
    )
    parser.add_argument(
        "--min-nonzero-voxels",
        type=int,
        default=32,
        help=(
            "Minimum foreground voxels required for a center slice; when VOI is enabled this is "
            "counted on functional/voi. Default 32 (increased from 16) filters out near-empty slices "
            "with very small VOI regions that carry little tumor information, improving sample quality."
        ),
    )
    parser.add_argument("--cache-volumes", type=str2bool, default=True, help="Cache volumes.")
    parser.add_argument(
        "--use-monai-augmentation",
        type=str2bool,
        default=True,
        help="Enable the MONAI training augmentation pipeline on the train split.",
    )
    parser.add_argument(
        "--aug-affine-prob",
        type=float,
        default=0.9,
        help="Probability of applying the shared affine transform on image/mask.",
    )
    parser.add_argument(
        "--aug-rotate-deg",
        type=float,
        default=30.0,
        help="Maximum in-plane rotation angle in degrees for RandAffined.",
    )
    parser.add_argument(
        "--aug-translate-px",
        type=float,
        default=20.0,
        help="Maximum in-plane translation in pixels for RandAffined.",
    )
    parser.add_argument(
        "--aug-scale-range",
        type=float,
        default=0.25,
        help="Maximum isotropic scaling factor delta for RandAffined.",
    )
    parser.add_argument(
        "--aug-flip-prob",
        type=float,
        default=0.5,
        help="Probability of left-right flipping the 2D slice block.",
    )
    parser.add_argument(
        "--aug-intensity-scale-prob",
        type=float,
        default=0.5,
        help="Probability of applying intensity scaling on the image channels.",
    )
    parser.add_argument(
        "--aug-intensity-scale",
        type=float,
        default=0.25,
        help="Maximum intensity scaling factor used by RandScaleIntensityd.",
    )
    parser.add_argument(
        "--aug-intensity-shift-prob",
        type=float,
        default=0.5,
        help="Probability of applying intensity shifting on the image channels.",
    )
    parser.add_argument(
        "--aug-intensity-shift",
        type=float,
        default=0.25,
        help="Maximum std-based intensity shift used by RandStdShiftIntensityd.",
    )
    parser.add_argument(
        "--aug-gaussian-noise-prob",
        type=float,
        default=0.5,
        help="Probability of adding Gaussian noise to simulate MRI acquisition noise.",
    )
    parser.add_argument(
        "--aug-gaussian-noise-std",
        type=float,
        default=0.05,
        help="Standard deviation of the Gaussian noise added by RandGaussianNoised.",
    )
    parser.add_argument(
        "--aug-gibbs-noise-prob",
        type=float,
        default=0.3,
        help="Probability of applying Gibbs ringing noise to simulate MRI artifact.",
    )
    parser.add_argument(
        "--aug-gibbs-noise-alpha",
        type=float,
        default=0.5,
        help="Upper bound of the Gibbs noise alpha range [0, alpha] used by RandGibbsNoised.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help=(
            "AdamW learning rate. Default 5e-6 (reduced from 2e-5) further slows down learning to "
            "mitigate extreme early overfitting (best_epoch=3) on small medical image datasets. "
            "With layerwise lr_mult=0.1, backbone layer4 lr=5e-7, head lr=5e-6."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help=(
            "AdamW weight decay (L2 regularization). Default 1e-2 (increased from 1e-3) for "
            "stronger regularization on small datasets (~56 training patients)."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global random seed.")
    parser.add_argument(
        "--pretrained",
        type=str2bool,
        default=True,
        help="Whether to initialize ResNet-18 from ImageNet weights.",
    )
    parser.add_argument(
        "--freeze-layers",
        type=str,
        default="conv1,layer1,layer2,layer3,layer4",
        help=(
            "Comma-separated list of ResNet-18 layers to freeze (no gradient). "
            "Valid names: none, conv1, bn1, layer1, layer2, layer3, layer4. "
            "Use 'none' to disable freezing (full fine-tuning). "
            "Default 'conv1,layer1,layer2,layer3,layer4' freezes the entire backbone, only trains fc "
            "reducing trainable params from ~11M to ~1K — maximally suppresses overfitting for "
            "small datasets (~96 patients). If val AUC drops below 0.75, relax to layer3 only."
        ),
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=224,
        help="Resize height before feeding images to ResNet. <=0 disables resize.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=224,
        help="Resize width before feeding images to ResNet. <=0 disables resize.",
    )
    parser.add_argument(
        "--use-class-weights",
        type=str2bool,
        default=True,
        help="Use inverse-frequency class weights from the training split.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help=(
            "Stop if validation AUC does not improve for N epochs. "
            "Default 30 (increased from 20) gives the model more time to escape from early overfitting "
            "and find a better generalization point."
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help=(
            "Minimum change in validation score to qualify as an improvement for early stopping. "
            "Default 0.0 (reduced from 0.005): any improvement counts, since with strong regularization "
            "the model may improve only marginally per epoch."
        ),
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.7,
        help=(
            "Dropout probability applied before the final classification layer. "
            "0.0 disables Dropout. Default 0.7 (increased from 0.5) for stronger regularization "
            "when training blocks (~6000) greatly outnumber validation patients (~14)."
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.3,
        help=(
            "Label smoothing coefficient for CrossEntropyLoss (0.0 = standard cross-entropy). "
            "Default 0.3 (increased from 0.2) softens one-hot targets more aggressively "
            "to prevent overconfident predictions (train AUC=0.9995) on small datasets. "
            "Do not exceed 0.4 to avoid losing discriminative capacity."
        ),
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "cosine", "plateau"),
        default="plateau",
        help=(
            "Learning rate scheduler. Default 'plateau' (changed from 'cosine'): "
            "reduces lr when val AUC stagnates, more conservative than cosine for overfitting scenarios. "
            "'cosine': CosineAnnealingLR (T_max=epochs, eta_min=1e-6), smoothly decays lr. "
            "'none': constant lr throughout training."
        ),
    )
    parser.add_argument(
        "--save-train-predictions",
        type=str2bool,
        default=True,
        help="Whether to export patient-level train predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Patient-level probability threshold for class prediction.",
    )
    parser.add_argument(
        "--topk-pool",
        type=int,
        default=0,
        help=(
            "Top-K mean pooling for patient-level aggregation. "
            "0 = use all blocks (simple mean, default). "
            "K > 0 = keep only the K most confident blocks per patient "
            "(sorted by |prob - 0.5|, largest first) before averaging. "
            "Recommended: set K to roughly 1/3 of the typical block count per patient."
        ),
    )
    parser.add_argument(
        "--threshold-search",
        type=str2bool,
        default=True,
        help=(
            "Whether to automatically search for the optimal classification threshold on the "
            "validation set using Youden Index (max TPR - FPR) after training. "
            "If True, the best threshold is reported and used to re-evaluate the test set, "
            "while the fixed --threshold is still used for the primary evaluation. "
            "This helps diagnose and improve low Specificity without changing the training process."
        ),
    )
    parser.add_argument(
        "--log-to-file",
        type=str2bool,
        default=True,
        help="Whether to save terminal output to a log file in output directory.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save a checkpoint every N epochs. Set to 0 to disable periodic saving.",
    )
    return parser


def set_seed(seed: int) -> None:
    """固定常见随机源，尽量提升实验可复现性。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_run_id(run_id: str | None) -> str:
    """解析本次运行 ID；如果用户未指定，则使用时间戳。"""

    if run_id:
        return run_id
    return time.strftime("%Y%m%d_%H%M%S")


def build_augmentation_config(args: argparse.Namespace) -> MonaiAugmentConfig:
    """Convert CLI arguments into a MONAI augmentation config."""

    return MonaiAugmentConfig(
        enabled=args.use_monai_augmentation,
        affine_prob=args.aug_affine_prob,
        rotate_deg=args.aug_rotate_deg,
        translate_px=args.aug_translate_px,
        scale_range=args.aug_scale_range,
        flip_prob=args.aug_flip_prob,
        intensity_scale_prob=args.aug_intensity_scale_prob,
        intensity_scale=args.aug_intensity_scale,
        intensity_shift_prob=args.aug_intensity_shift_prob,
        intensity_shift=args.aug_intensity_shift,
        gaussian_noise_prob=args.aug_gaussian_noise_prob,
        gaussian_noise_std=args.aug_gaussian_noise_std,
        gibbs_noise_prob=args.aug_gibbs_noise_prob,
        gibbs_noise_alpha=args.aug_gibbs_noise_alpha,
    )


def build_datasets(args: argparse.Namespace) -> Dict[str, HabitatIDHBlockDataset]:
    """
    根据 train / val / test 目录构建数据集对象。

    这里依赖 HabitatIDHBlockDataset 完成：
    1. 多模态体数据读取；
    2. 按切片轴生成 2.5D block；
    3. 强度归一化；
    4. 返回病人元信息，供后续 block -> patient 聚合使用。
    """

    has_mask = args.require_voi or args.append_voi_mask or args.mask_background_with_voi
    augment_config = build_augmentation_config(args)
    transforms = build_monai_block_transforms(
        config=augment_config,
        has_mask=has_mask,
        spatial_size=(args.resize_height, args.resize_width),
    )

    datasets: Dict[str, HabitatIDHBlockDataset] = {}
    for split_name in ("train", "val", "test"):
        split_root = args.split_base_root / split_name
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Missing split directory: {split_root}. "
                "Please confirm data_split.py outputs or pass --split-base-root."
            )
        datasets[split_name] = HabitatIDHBlockDataset(
            split_root=split_root,
            modalities=args.modalities,
            require_voi=args.require_voi,
            append_voi_mask=args.append_voi_mask,
            mask_background_with_voi=args.mask_background_with_voi,
            block_depth=args.block_depth,
            slice_axis=args.slice_axis,
            intensity_norm=args.intensity_norm,
            min_nonzero_voxels=args.min_nonzero_voxels,
            cache_volumes=args.cache_volumes,
            return_metadata=True,
            transform=transforms[split_name],
        )
    return datasets


def build_dataloaders(
    datasets: Dict[str, HabitatIDHBlockDataset],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    """为三个数据划分分别创建 DataLoader。"""

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }


def maybe_resize(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """按需将输入 resize 到 ResNet-18 常用输入分辨率。"""

    if height <= 0 or width <= 0:
        return images
    if images.shape[-2:] == (height, width):
        return images
    return F.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)


def compute_input_channels(args: argparse.Namespace) -> int:
    """根据模态数和 VOI 配置计算模型输入通道数。"""

    modality_channels = len(args.modalities) * args.block_depth
    voi_channels = args.block_depth if args.append_voi_mask else 0
    return modality_channels + voi_channels


def compute_class_weights(dataset: HabitatIDHBlockDataset, device: torch.device) -> torch.Tensor:
    """
    根据训练集病人级类别频次计算反频率权重。

    这里按 patient case 统计，而不是按 block 统计，
    目的是避免切片数量较多的病人对类别权重产生额外偏置。
    """

    counts = np.zeros(2, dtype=np.float64)
    for case in dataset.patient_cases.values():
        counts[case.label_id] += 1.0
    if np.any(counts <= 0):
        raise ValueError(f"Invalid class counts for class-weight computation: {counts.tolist()}")
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    resize_height: int,
    resize_width: int,
) -> Dict[str, float]:
    """
    执行一个 epoch 的训练。

    返回的是 block 级别的平均 loss 和准确率，
    因为训练阶段的每个 batch 样本单位就是 2.5D block。
    """

    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for batch in progress:
        # 从 dataloader 中取出一个 batch，并搬运到目标设备。
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        labels = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)
        images = maybe_resize(images, resize_height, resize_width)

        # 标准的前向、反向与参数更新流程。
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())

        # tqdm 中显示的是到当前为止的累计平均损失和准确率。
        progress.set_postfix(
            loss=f"{(total_loss / max(total_samples, 1)):.4f}",
            acc=f"{(total_correct / max(total_samples, 1)):.4f}",
        )

    return {
        "loss": total_loss / max(total_samples, 1),
        "block_acc": total_correct / max(total_samples, 1),
    }


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """在标签只包含单一类别时返回 NaN，避免 roc_auc_score 报错。"""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def patient_level_from_block_predictions(
    records: List[Dict[str, object]],
    split_name: str,
    run_id: str,
    checkpoint_name: str,
    threshold: float,
    top_k: int = 0,
) -> List[Dict[str, object]]:
    """
    将 block 级别预测聚合为病人级别预测。

    聚合方式为：
    - 同一病人的所有 block 预测概率，若 top_k > 0 则取置信度最高的 top_k 个 block，
      否则使用全部 block；
    - 对选中的 block 概率取均值（Top-K 均值池化）；
    - 再与给定 threshold 比较，得到最终的二分类标签。

    top_k 置信度度量：以 abs(prob - 0.5) 为依据，数值越大表示预测越确定，
    能自然过滤靠近决策边界的低质量切片预测。
    """

    grouped: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {"y_true": None, "probs": []}
    )

    for record in records:
        # 将同一病人的所有 block 概率收集起来。
        patient_id = str(record["patient_id"])
        grouped[patient_id]["probs"].append(float(record["prob_idh_mut"]))
        if grouped[patient_id]["y_true"] is None:
            grouped[patient_id]["y_true"] = int(record["y_true"])

    rows: List[Dict[str, object]] = []
    for patient_id in sorted(grouped):
        y_true = int(grouped[patient_id]["y_true"])
        probs = grouped[patient_id]["probs"]

        # Top-K 均值池化：取置信度最高（距 0.5 最远）的 K 个 block。
        if top_k > 0 and len(probs) > top_k:
            probs = sorted(probs, key=lambda p: abs(p - 0.5), reverse=True)[:top_k]

        prob = float(np.mean(probs))
        pred_label = int(prob >= threshold)
        # 输出结构直接面向后续 CSV 导出，因此保留 run_id / checkpoint_name 等实验追踪字段。
        rows.append(
            {
                "patient_id": patient_id,
                "split": split_name,
                "y_true": y_true,
                "prob_idh_mut": prob,
                "pred_label": pred_label,
                "run_id": run_id,
                "checkpoint_name": checkpoint_name,
            }
        )

    return rows


def compute_patient_metrics(patient_rows: List[Dict[str, object]]) -> Dict[str, float]:
    """基于病人级预测结果计算 AUC、ACC、F1、敏感性、特异性和混淆矩阵。"""

    y_true = np.asarray([row["y_true"] for row in patient_rows], dtype=np.int64)
    y_prob = np.asarray([row["prob_idh_mut"] for row in patient_rows], dtype=np.float64)
    y_pred = np.asarray([row["pred_label"] for row in patient_rows], dtype=np.int64)

    auc = safe_auc(y_true, y_prob)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "sen": sen,
        "spe": spe,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    resize_height: int,
    resize_width: int,
    split_name: str,
    run_id: str,
    checkpoint_name: str,
    threshold: float,
    top_k: int = 0,
) -> Dict[str, object]:
    """
    在指定数据划分上执行评估，并返回 block 级与 patient 级结果。

    输出内容包括：
    - 平均 loss；
    - block 数、patient 数；
    - patient 级预测表；
    - patient 级指标；
    - block 级原始预测记录。
    """

    model.eval()
    total_loss = 0.0
    total_samples = 0
    block_records: List[Dict[str, object]] = []

    with torch.no_grad():
        progress = tqdm(dataloader, desc=f"eval-{split_name}", leave=False)
        for batch in progress:
            # 评估阶段不需要梯度，只做前向推理并记录概率。
            images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            labels = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)
            images = maybe_resize(images, resize_height, resize_width)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            progress.set_postfix(loss=f"{(total_loss / max(total_samples, 1)):.4f}")

            patient_ids = batch["patient_id"]
            slice_indices = batch["slice_index"]
            for i in range(batch_size):
                # 这里保存 block 级结果，是为了后续进行病人级均值池化聚合。
                block_records.append(
                    {
                        "patient_id": str(patient_ids[i]),
                        "slice_index": int(slice_indices[i]),
                        "y_true": int(labels[i].item()),
                        "prob_idh_mut": float(probs[i].item()),
                    }
                )

    patient_rows = patient_level_from_block_predictions(
        records=block_records,
        split_name=split_name,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        threshold=threshold,
        top_k=top_k,
    )
    patient_metrics = compute_patient_metrics(patient_rows)

    return {
        "loss": total_loss / max(total_samples, 1),
        "num_blocks": total_samples,
        "num_patients": len(patient_rows),
        "patient_rows": patient_rows,
        "metrics": patient_metrics,
        "block_records": block_records,
    }


def save_json(data: Dict[str, object], path: Path) -> None:
    """将字典以 UTF-8 JSON 格式保存到磁盘。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_json_safe(value: object) -> object:
    """递归处理 Path 等对象，保证其可被 json.dumps 序列化。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def save_yaml(data: Dict[str, object], path: Path) -> None:
    """将字典以 UTF-8 YAML 格式保存到磁盘。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            make_json_safe(data),
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def save_rows(rows: List[Dict[str, object]], path: Path, fieldnames: Sequence[str]) -> None:
    """将一组字典行保存成 CSV 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_prediction_tables(
    eval_outputs: Dict[str, Dict[str, object]],
    output_dir: Path,
    run_id: str,
    checkpoint_name: str,
    save_train_predictions: bool,
) -> None:
    """
    导出评估结果相关的表格文件。

    每个 split 会导出：
    - patient_predictions_*.csv：病人级预测结果；
    - roc_raw_*.csv：ROC 曲线原始点（若该 split 同时包含正负样本）；
    - confusion_matrix_*.csv：混淆矩阵。

    此外还会导出：
    - metrics_summary_*.csv：所有 split 的指标汇总。
    - metrics_*.csv：与 metrics_summary 内容相同的兼容别名。
    """

    metrics_fieldnames = [
        "split",
        "loss",
        "auc",
        "acc",
        "sen",
        "spe",
        "f1",
        "num_patients",
        "num_blocks",
        "run_id",
        "checkpoint_name",
    ]

    for split_name, output in eval_outputs.items():
        if split_name == "train" and not save_train_predictions:
            continue

        patient_rows = output["patient_rows"]
        pred_path = output_dir / f"patient_predictions_{MODEL_NAME}_{split_name}_{run_id}.csv"
        save_rows(
            patient_rows,
            pred_path,
            fieldnames=[
                "patient_id",
                "split",
                "y_true",
                "prob_idh_mut",
                "pred_label",
                "run_id",
                "checkpoint_name",
            ],
        )

        metrics = output["metrics"]
        y_true = np.asarray([row["y_true"] for row in patient_rows], dtype=np.int64)
        y_prob = np.asarray([row["prob_idh_mut"] for row in patient_rows], dtype=np.float64)

        # 只有同时存在正负样本时 ROC 才有意义。
        if len(np.unique(y_true)) >= 2:
            fpr, tpr, threshold_arr = roc_curve(y_true, y_prob)
            roc_rows = [
                {
                    "fpr": float(fpr_item),
                    "tpr": float(tpr_item),
                    "threshold": float(thr_item),
                    "model": MODEL_NAME,
                    "run_id": run_id,
                }
                for fpr_item, tpr_item, thr_item in zip(fpr, tpr, threshold_arr)
            ]
            save_rows(
                roc_rows,
                output_dir / f"roc_raw_{MODEL_NAME}_{split_name}_{run_id}.csv",
                fieldnames=["fpr", "tpr", "threshold", "model", "run_id"],
            )
            save_rows(
                roc_rows,
                output_dir / f"roc_points_{MODEL_NAME}_{split_name}_{run_id}.csv",
                fieldnames=["fpr", "tpr", "threshold", "model", "run_id"],
            )
        save_rows(
            [
                {
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tp": metrics["tp"],
                    "model": MODEL_NAME,
                    "run_id": run_id,
                }
            ],
            output_dir / f"confusion_matrix_{MODEL_NAME}_{split_name}_{run_id}.csv",
            fieldnames=["tn", "fp", "fn", "tp", "model", "run_id"],
        )

    metrics_rows: List[Dict[str, object]] = []
    for split_name, output in eval_outputs.items():
        if split_name == "train" and not save_train_predictions:
            continue
        metrics = output["metrics"]
        metrics_rows.append(
            {
                "split": split_name,
                "loss": output["loss"],
                "auc": metrics["auc"],
                "acc": metrics["acc"],
                "sen": metrics["sen"],
                "spe": metrics["spe"],
                "f1": metrics["f1"],
                "num_patients": output["num_patients"],
                "num_blocks": output["num_blocks"],
                "run_id": run_id,
                "checkpoint_name": checkpoint_name,
            }
        )
    save_rows(
        metrics_rows,
        output_dir / f"metrics_summary_{MODEL_NAME}_{run_id}.csv",
        fieldnames=metrics_fieldnames,
    )
    save_rows(
        metrics_rows,
        output_dir / f"metrics_{MODEL_NAME}_{run_id}.csv",
        fieldnames=metrics_fieldnames,
    )


def save_training_history(history: Iterable[EpochStats], output_dir: Path, run_id: str) -> None:
    """导出逐 epoch 的训练历史，便于后续画学习曲线或做实验记录。"""

    rows = [asdict(item) for item in history]
    fieldnames = [
        "epoch",
        "train_loss",
        "train_block_acc",
        "val_loss",
        "val_auc",
        "val_acc",
        "val_f1",
        "val_sen",
        "val_spe",
        "lr",
    ]
    save_rows(
        rows,
        output_dir / f"training_history_{MODEL_NAME}_{run_id}.csv",
        fieldnames=fieldnames,
    )
    save_rows(rows, output_dir / f"train_log_{MODEL_NAME}_{run_id}.csv", fieldnames=fieldnames)
    save_rows(rows, output_dir / "train_log.csv", fieldnames=fieldnames)


def export_wrong_cases(
    eval_outputs: Dict[str, Dict[str, object]],
    output_dir: Path,
    run_id: str,
    threshold: float,
) -> None:
    """导出所有误判病例，便于后续做错误分析和典型病例整理。"""

    wrong_rows: List[Dict[str, object]] = []
    for split_name, output in eval_outputs.items():
        for row in output["patient_rows"]:
            y_true = int(row["y_true"])
            pred_label = int(row["pred_label"])
            if y_true == pred_label:
                continue
            error_type = "FP" if y_true == 0 and pred_label == 1 else "FN"
            prob = float(row["prob_idh_mut"])
            wrong_rows.append(
                {
                    **row,
                    "error_type": error_type,
                    "prob_margin_to_threshold": abs(prob - threshold),
                    "is_low_confidence": int(abs(prob - threshold) < 0.1),
                    "num_patients_in_split": output["num_patients"],
                }
            )

    wrong_rows.sort(
        key=lambda item: (
            str(item["split"]),
            str(item["error_type"]),
            float(item["prob_margin_to_threshold"]),
            str(item["patient_id"]),
        )
    )
    save_rows(
        wrong_rows,
        output_dir / f"wrong_cases_{MODEL_NAME}_{run_id}.csv",
        fieldnames=[
            "patient_id",
            "split",
            "y_true",
            "prob_idh_mut",
            "pred_label",
            "run_id",
            "checkpoint_name",
            "error_type",
            "prob_margin_to_threshold",
            "is_low_confidence",
            "num_patients_in_split",
        ],
    )


def plot_roc_curve(patient_rows: List[Dict[str, object]], path: Path, title: str) -> bool:
    """根据病人级预测结果导出 ROC 曲线 PNG。"""

    y_true = np.asarray([row["y_true"] for row in patient_rows], dtype=np.int64)
    y_prob = np.asarray([row["prob_idh_mut"] for row in patient_rows], dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return False

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = safe_auc(y_true, y_prob)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=150)
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2.0, label=f"AUC = {auc:.3f}")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#999999", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_confusion_matrix(metrics: Dict[str, float], path: Path, title: str) -> None:
    """根据病人级混淆矩阵导出 PNG。"""

    matrix = np.asarray(
        [
            [int(metrics["tn"]), int(metrics["fp"])],
            [int(metrics["fn"]), int(metrics["tp"])],
        ],
        dtype=np.int64,
    )
    labels = (("TN", "FP"), ("FN", "TP"))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=150)
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(title)

    max_value = max(int(matrix.max()), 1)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            text_color = "white" if value > max_value / 2 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{labels[row_idx][col_idx]}\n{value}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def export_evaluation_figures(
    eval_outputs: Dict[str, Dict[str, object]],
    output_dir: Path,
    run_id: str,
) -> None:
    """导出测试口径需要的 ROC 与混淆矩阵 PNG 图。"""

    figure_dir = output_dir / "figures"
    for split_name, output in eval_outputs.items():
        patient_rows = output["patient_rows"]
        metrics = output["metrics"]

        roc_path = figure_dir / f"roc_curve_{MODEL_NAME}_{split_name}_{run_id}.png"
        cm_path = figure_dir / f"confusion_matrix_{MODEL_NAME}_{split_name}_{run_id}.png"
        wrote_roc = plot_roc_curve(
            patient_rows=patient_rows,
            path=roc_path,
            title=f"{MODEL_NAME.upper()} ROC ({split_name})",
        )
        plot_confusion_matrix(
            metrics=metrics,
            path=cm_path,
            title=f"{MODEL_NAME.upper()} Confusion Matrix ({split_name})",
        )

        if split_name == "test":
            if wrote_roc:
                plot_roc_curve(
                    patient_rows=patient_rows,
                    path=figure_dir / f"{MODEL_NAME}_roc.png",
                    title=f"{MODEL_NAME.upper()} ROC (test)",
                )
            plot_confusion_matrix(
                metrics=metrics,
                path=figure_dir / f"{MODEL_NAME}_cm.png",
                title=f"{MODEL_NAME.upper()} Confusion Matrix (test)",
            )


def aggregate_metric_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """将多次运行的指标表按 split 聚合为均值和标准差。"""

    if not rows:
        return []

    numeric_fields = ("loss", "auc", "acc", "sen", "spe", "f1", "num_patients", "num_blocks")
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for split_name in sorted(grouped):
        split_rows = grouped[split_name]
        summary_row: Dict[str, object] = {
            "split": split_name,
            "n_runs": len(split_rows),
        }
        for field in numeric_fields:
            values = np.asarray(
                [
                    float(item[field])
                    for item in split_rows
                    if item.get(field, "") not in {"", "nan", "NaN"}
                ],
                dtype=np.float64,
            )
            if values.size == 0:
                summary_row[f"{field}_mean"] = float("nan")
                summary_row[f"{field}_std"] = float("nan")
            else:
                summary_row[f"{field}_mean"] = float(np.mean(values))
                summary_row[f"{field}_std"] = float(np.std(values))
        summary_rows.append(summary_row)
    return summary_rows


def update_cross_run_metric_summaries(output_root: Path) -> None:
    """扫描输出根目录下所有运行结果，并更新跨运行指标汇总表。"""

    all_rows: List[Dict[str, str]] = []
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "checkpoints":
            continue

        for metrics_path in sorted(run_dir.glob(f"metrics_summary_{MODEL_NAME}_*.csv")):
            with metrics_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["source_file"] = str(metrics_path.relative_to(output_root))
                    all_rows.append(row)

    if not all_rows:
        return

    all_rows.sort(key=lambda item: (str(item.get("run_id", "")), str(item.get("split", ""))))
    save_rows(
        all_rows,
        output_root / f"metrics_{MODEL_NAME}_all_runs.csv",
        fieldnames=[
            "split",
            "loss",
            "auc",
            "acc",
            "sen",
            "spe",
            "f1",
            "num_patients",
            "num_blocks",
            "run_id",
            "checkpoint_name",
            "source_file",
        ],
    )

    summary_rows = aggregate_metric_rows(all_rows)
    save_rows(
        summary_rows,
        output_root / f"metrics_{MODEL_NAME}_summary.csv",
        fieldnames=[
            "split",
            "n_runs",
            "loss_mean",
            "loss_std",
            "auc_mean",
            "auc_std",
            "acc_mean",
            "acc_std",
            "sen_mean",
            "sen_std",
            "spe_mean",
            "spe_std",
            "f1_mean",
            "f1_std",
            "num_patients_mean",
            "num_patients_std",
            "num_blocks_mean",
            "num_blocks_std",
        ],
    )


def export_run_config_yaml(args: argparse.Namespace, output_dir: Path) -> None:
    """导出当前运行配置的 YAML 版本，便于锁定基线实验口径。"""

    config_payload = {
        "model": MODEL_NAME,
        "modalities": list(args.modalities),
        "split_base_root": args.split_base_root,
        "output_root": args.output_root,
        "checkpoint_root": args.checkpoint_root,
        "data": {
            "require_voi": args.require_voi,
            "append_voi_mask": args.append_voi_mask,
            "mask_background_with_voi": args.mask_background_with_voi,
            "block_depth": args.block_depth,
            "slice_axis": args.slice_axis,
            "intensity_norm": args.intensity_norm,
            "min_nonzero_voxels": args.min_nonzero_voxels,
            "cache_volumes": args.cache_volumes,
            "resize_height": args.resize_height,
            "resize_width": args.resize_width,
        },
        "train": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "pretrained": args.pretrained,
            "freeze_layers": args.freeze_layers,
            "use_class_weights": args.use_class_weights,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "dropout_p": args.dropout_p,
            "label_smoothing": args.label_smoothing,
            "lr_scheduler": args.lr_scheduler,
            "save_train_predictions": args.save_train_predictions,
            "save_interval": args.save_interval,
            "threshold": args.threshold,
            "device": args.device,
        },
        "augmentation": build_augmentation_config(args).to_dict(),
    }
    save_yaml(config_payload, output_dir / "configs" / "resnet18_base.yaml")



def choose_monitor_score(eval_output: Dict[str, object]) -> float:
    """
    定义 early stopping 与最佳模型选择时使用的监控分数。

    优先使用验证集 patient-level AUC；
    如果 AUC 无法计算（例如验证集只有单类），则退化为负的验证损失，
    这样分数越大仍然代表模型越好。
    """

    auc = float(eval_output["metrics"]["auc"])
    if not np.isnan(auc):
        return auc
    return -float(eval_output["loss"])


def search_optimal_threshold(
    patient_rows: List[Dict[str, object]],
) -> float:
    """基于验证集病人级预测，用 Youden Index 搜索最优分类阈值。

    Youden Index = Sensitivity + Specificity - 1 = TPR - FPR。
    在 ROC 曲线上找到使 (TPR - FPR) 最大的点对应的阈值。

    返回：
    - 最优阈值（float），若无法计算则返回 0.5。
    """
    y_true = np.asarray([row["y_true"] for row in patient_rows], dtype=np.int64)
    y_prob = np.asarray([row["prob_idh_mut"] for row in patient_rows], dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return float(thresholds[best_idx])


def evaluate_with_threshold(
    patient_rows: List[Dict[str, object]],
    threshold: float,
) -> Dict[str, object]:
    """用指定阈值重新对已有病人级概率做二分类，并计算指标。

    此函数不重新跑推理，仅重新应用阈值后计算指标，用于阈值搜索结果的快速评估。
    """
    new_rows = [
        {**row, "pred_label": int(float(row["prob_idh_mut"]) >= threshold)}
        for row in patient_rows
    ]
    metrics = compute_patient_metrics(new_rows)
    return {"patient_rows": new_rows, "metrics": metrics}


def write_run_summary(
    args: argparse.Namespace,
    datasets: Dict[str, HabitatIDHBlockDataset],
    best_epoch: int,
    checkpoint_path: Path,
    output_dir: Path,
    run_id: str,
    eval_outputs: Dict[str, Dict[str, object]],
) -> None:
    """将本次实验的核心配置、数据摘要和评估指标写入 JSON 摘要文件。"""

    summary = {
        "model": MODEL_NAME,
        "run_id": run_id,
        "seed": args.seed,
        "device": args.device,
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path),
        "modalities": list(args.modalities),
        "voi_source_branch": VOI_SOURCE_BRANCH,
        "require_voi": args.require_voi,
        "append_voi_mask": args.append_voi_mask,
        "mask_background_with_voi": args.mask_background_with_voi,
        "input_channels": compute_input_channels(args),
        "block_depth": args.block_depth,
        "slice_axis": args.slice_axis,
        "intensity_norm": args.intensity_norm,
        "threshold": args.threshold,
        "augmentation": build_augmentation_config(args).to_dict(),
        "datasets": {split: dataset.summary() for split, dataset in datasets.items()},
        "metrics": {
            split: {
                "loss": output["loss"],
                "num_patients": output["num_patients"],
                "num_blocks": output["num_blocks"],
                **output["metrics"],
            }
            for split, output in eval_outputs.items()
        },
    }
    save_json(summary, output_dir / f"run_summary_{MODEL_NAME}_{run_id}.json")


def main() -> None:
    """
    主入口：
    1. 解析参数并设置随机种子；
    2. 构建数据集、DataLoader、模型、损失函数与优化器；
    3. 训练并在验证集上选择最佳模型；
    4. 使用最佳模型在 train/val/test 上做最终评估；
    5. 导出 checkpoint、预测表格、指标汇总、训练历史与运行摘要。
    """

    args = build_argparser().parse_args()
    set_seed(args.seed)

    # 设置输出目录和日志文件
    run_id = resolve_run_id(args.run_id)
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    export_run_config_yaml(args, output_dir)

    # 若启用日志文件，使用 TeeLogger 同时输出到终端和文件
    if args.log_to_file:
        log_path = output_dir / f"training_log_{MODEL_NAME}_{run_id}.txt"
        with TeeLogger(log_path, mode="w"):
            _main_training_loop(args, run_id, output_dir)
    else:
        _main_training_loop(args, run_id, output_dir)


def _parse_freeze_layers(freeze_layers_str: str) -> list:
    """解析 --freeze-layers 参数，返回需要冻结的层名列表。

    参数:
        freeze_layers_str: 逗号分隔的层名字符串，如 'conv1,layer1,layer2,layer3'，
                           或 'none'/'' 表示不冻结。
    返回:
        层名列表，例如 ['conv1', 'layer1', 'layer2', 'layer3']；若不冻结则返回空列表。
    """
    if not freeze_layers_str or freeze_layers_str.strip().lower() == "none":
        return []
    return [name.strip() for name in freeze_layers_str.split(",") if name.strip() and name.strip().lower() != "none"]


def _main_training_loop(
    args: argparse.Namespace, run_id: str, output_dir: Path
) -> None:
    """实际训练流程（被 main 函数调用，支持日志重定向）。"""

    checkpoint_dir = args.checkpoint_root / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(args)
    dataloaders = build_dataloaders(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 输入通道数 = 模态数 × 2.5D block 深度 + 可选 VOI 通道。
    # 例如 6 个模态、block_depth=5，且追加 VOI，则输入通道为 35。
    device = torch.device(args.device)
    in_channels = compute_input_channels(args)
    model = ResNet18Classifier(
        in_channels=in_channels,
        num_classes=2,
        pretrained=args.pretrained,
        dropout_p=args.dropout_p,  # 分类头前 Dropout，用于抑制过拟合
    ).to(device)

    # 冻结指定的 backbone 层以减少可训练参数，缓解小样本过拟合。
    # 例如冻结 conv1,layer1,layer2,layer3 后，可训练参数从 ~11M 降至 ~2.7M。
    freeze_layer_names = _parse_freeze_layers(args.freeze_layers)
    if freeze_layer_names:
        frozen_params = 0
        for layer_name in freeze_layer_names:
            layer = getattr(model.model, layer_name, None)
            if layer is None:
                raise ValueError(
                    f"--freeze-layers: '{layer_name}' is not a valid ResNet-18 layer name. "
                    "Valid options: conv1, bn1, layer1, layer2, layer3, layer4."
                )
            for param in layer.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Frozen layers: {freeze_layer_names} "
            f"(frozen_params={frozen_params:,}, trainable_params={trainable_params:,})"
        )
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"No layers frozen (full fine-tuning, trainable_params={trainable_params:,})")

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(datasets["train"], device)
    # CrossEntropyLoss 增加 label_smoothing，软化目标标签防止模型过于自信；
    # 优化器为 AdamW，增大 weight_decay 至 1e-3 以加强 L2 正则化。
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # 使用分层差异化学习率：骨干浅层使用极小学习率以保护预训练特征，
    # 分类头使用完整 args.lr 以快速收敛。lr_mult=0.1 表示骨干最深层
    # 学习率为 args.lr × 0.1，浅层进一步缩小至 args.lr × 0.02。
    # 若模型未提供 get_param_groups()（如非 ResNet18Classifier 实例），
    # 则回退到全参数统一学习率，保持向后兼容。
    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(base_lr=args.lr, lr_mult=0.1)
        group_info = "  |  ".join(
            f"{g['name']}:lr={g['lr']:.2e}" for g in param_groups
        )
        print(f"Using layerwise param groups: {group_info}")
    else:
        param_groups = model.parameters()
        print(f"Using uniform lr={args.lr:.2e} for all parameters (no get_param_groups)")
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # 构建学习率调度器（可选）
    # - cosine: 余弦退火，将 lr 从初始值平滑衰减至 eta_min=1e-6，适合大多数场景。
    # - plateau: 当验证 AUC 停滞时将 lr 减半，适合训练曲线震荡场景。
    # - none: 不使用调度器，全程固定学习率。
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )
        print(f"LR scheduler: CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-6)")
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",       # 监控指标越大越好（val AUC）
            factor=0.5,       # lr 乘以 0.5
            patience=5,       # 连续 5 epoch 没有改进才降低 lr
            min_lr=1e-6,
        )
        print("LR scheduler: ReduceLROnPlateau (mode=max, factor=0.5, patience=5, min_lr=1e-6)")
    else:
        print("LR scheduler: none (constant learning rate)")

    history: List[EpochStats] = []
    best_score = -float("inf")
    best_epoch = -1
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # 先在训练集上跑一个 epoch。
        train_stats = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            resize_height=args.resize_height,
            resize_width=args.resize_width,
        )
        val_output = evaluate_model(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            resize_height=args.resize_height,
            resize_width=args.resize_width,
            split_name="val",
            run_id=run_id,
            checkpoint_name="best.pt",
            threshold=args.threshold,
            top_k=args.topk_pool,
        )

        # 验证集指标以病人级别结果为准，用于选择最佳模型。
        metrics = val_output["metrics"]
        current_score = choose_monitor_score(val_output)

        # 更新学习率调度器：
        # - plateau 调度器需要传入监控指标（val AUC），与 early stopping 共用同一分数；
        # - cosine 调度器直接按 epoch 步进即可；
        # - 在记录 current_lr 前调用，确保 history 中记录的是本 epoch 实际使用的 lr。
        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(current_score)
            else:
                scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            EpochStats(
                epoch=epoch,
                train_loss=train_stats["loss"],
                train_block_acc=train_stats["block_acc"],
                val_loss=val_output["loss"],
                val_auc=metrics["auc"],
                val_acc=metrics["acc"],
                val_f1=metrics["f1"],
                val_sen=metrics["sen"],
                val_spe=metrics["spe"],
                lr=current_lr,
            )
        )

        # 改进判断：提升幅度必须超过阈值才算有效改进
        improved = (current_score - best_score) > args.early_stop_min_delta
        delta_str = f"(+{current_score - best_score:.6f})" if improved else f"({current_score - best_score:+.6f})"
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"train_block_acc={train_stats['block_acc']:.4f} | "
            f"val_loss={val_output['loss']:.4f} | "
            f"val_auc={metrics['auc']:.4f} | "
            f"val_acc={metrics['acc']:.4f} | "
            f"val_f1={metrics['f1']:.4f} | "
            f"lr={current_lr:.2e} "
            f"delta={delta_str}"
        )

        # 构建 checkpoint 数据。
        # 注意：将 args 中所有 pathlib.Path 字段预转为 str，确保 checkpoint 不含
        # 非原生 Python 类型。PyTorch 2.6 默认以 weights_only=True 加载，而
        # Path 对象不在安全全局列表中，直接保存会导致后续 torch.load 失败。
        # 使用已有的 make_json_safe 工具函数统一处理，零冗余。
        safe_args = make_json_safe(vars(args))
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
            "current_score": current_score,
            "args": safe_args,
            "run_id": run_id,
        }

        # 1. 保存 last checkpoint（每个 epoch 都保存）
        torch.save(checkpoint_data, last_checkpoint_path)

        # 2. 保存 best checkpoint（验证集指标提升时）
        if improved:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"  -> New best checkpoint saved (score={best_score:.4f})")
        else:
            epochs_without_improvement += 1

        # 3. 定期保存 checkpoint（每隔 save_interval 个 epoch）
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            periodic_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint_data, periodic_path)
            print(f"  -> Periodic checkpoint saved: epoch_{epoch:03d}.pt")

        if epochs_without_improvement >= args.early_stop_patience:
            # 达到 patience 后提前结束训练，避免无效迭代。
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(patience={args.early_stop_patience}, "
                f"min_delta={args.early_stop_min_delta})."
            )
            break

    if best_epoch < 0 or not best_checkpoint_path.is_file():
        raise RuntimeError("Training finished without a valid best checkpoint.")

    # 使用验证集上最佳的权重重新加载模型，再做最终导出。
    # weights_only=True：PyTorch 2.6 的安全默认值，仅反序列化 tensor/storage 类型；
    # checkpoint 中的 args 已在保存时通过 make_json_safe 将 Path 转为 str，
    # 因此这里直接使用 weights_only=True 即可安全加载。
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_outputs: Dict[str, Dict[str, object]] = {}
    splits_to_export = ("train", "val", "test") if args.save_train_predictions else ("val", "test")
    for split_name in splits_to_export:
        # 这里导出的结果均基于同一个 best checkpoint，保证可比性。
        eval_outputs[split_name] = evaluate_model(
            model=model,
            dataloader=dataloaders[split_name],
            criterion=criterion,
            device=device,
            resize_height=args.resize_height,
            resize_width=args.resize_width,
            split_name=split_name,
            run_id=run_id,
            checkpoint_name=best_checkpoint_path.name,
            threshold=args.threshold,
            top_k=args.topk_pool,
        )

    export_prediction_tables(
        eval_outputs=eval_outputs,
        output_dir=output_dir,
        run_id=run_id,
        checkpoint_name=best_checkpoint_path.name,
        save_train_predictions=args.save_train_predictions,
    )
    save_training_history(history, output_dir, run_id)
    export_wrong_cases(
        eval_outputs=eval_outputs,
        output_dir=output_dir,
        run_id=run_id,
        threshold=args.threshold,
    )

    # C3: 动态阈值搜索 — 基于验证集 Youden Index 找最优阈值，并用其重新评估测试集。
    # 只需重新对已有的 patient_rows 做阈值应用，不重新跑推理，开销极低。
    if args.threshold_search and "val" in eval_outputs:
        optimal_thr = search_optimal_threshold(eval_outputs["val"]["patient_rows"])
        print(f"\n[Threshold Search] Fixed threshold={args.threshold:.3f} | "
              f"Youden optimal threshold={optimal_thr:.3f}")
        for split_name in eval_outputs:
            re_eval = evaluate_with_threshold(eval_outputs[split_name]["patient_rows"], optimal_thr)
            m = re_eval["metrics"]
            print(
                f"  [{split_name}] thr={optimal_thr:.3f} | "
                f"acc={m['acc']:.4f} | f1={m['f1']:.4f} | "
                f"sen={m['sen']:.4f} | spe={m['spe']:.4f} | "
                f"tn={m['tn']} fp={m['fp']} fn={m['fn']} tp={m['tp']}"
            )
        # 保存使用最优阈值后的测试集预测表
        if "test" in eval_outputs:
            test_rows_opt = evaluate_with_threshold(
                eval_outputs["test"]["patient_rows"], optimal_thr
            )["patient_rows"]
            opt_pred_path = output_dir / f"patient_predictions_{MODEL_NAME}_test_{run_id}_thr{optimal_thr:.3f}.csv"
            save_rows(
                test_rows_opt,
                opt_pred_path,
                fieldnames=["patient_id", "split", "y_true", "prob_idh_mut", "pred_label",
                            "run_id", "checkpoint_name"],
            )
            print(f"  Optimal-threshold test predictions saved: {opt_pred_path.name}")
        # 保存阈值搜索摘要
        thr_summary = {
            "fixed_threshold": args.threshold,
            "optimal_threshold_youden": optimal_thr,
            "splits": {},
        }
        for split_name in eval_outputs:
            re_eval = evaluate_with_threshold(eval_outputs[split_name]["patient_rows"], optimal_thr)
            thr_summary["splits"][split_name] = re_eval["metrics"]
        save_json(thr_summary, output_dir / f"threshold_search_{MODEL_NAME}_{run_id}.json")
    export_evaluation_figures(
        eval_outputs=eval_outputs,
        output_dir=output_dir,
        run_id=run_id,
    )
    write_run_summary(
        args=args,
        datasets=datasets,
        best_epoch=best_epoch,
        checkpoint_path=best_checkpoint_path,
        output_dir=output_dir,
        run_id=run_id,
        eval_outputs=eval_outputs,
    )

    # 额外保存本次运行的完整参数配置，便于复现实验。
    args_path = output_dir / f"run_config_{MODEL_NAME}_{run_id}.json"
    save_json(vars(args), args_path)
    update_cross_run_metric_summaries(args.output_root)
    print(f"Run completed. Best epoch: {best_epoch}.")
    print(f"Checkpoint: {best_checkpoint_path}")
    print(f"Outputs   : {output_dir}")


if __name__ == "__main__":
    main()
