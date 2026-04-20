---
name: user-guide-habitat-cbm
updated: 2026-04-20
description: Habitat-CBM 全流程使用说明（skill 风格）。面向用户与 LLM，覆盖概念标签构建、数据校验、三阶段训练、患者级评估、概念评估与概念干预。
---

# Habitat-CBM User Guide (Skill Style)

## Overview

本指南是 Habitat-CBM 的可执行上下文文档。目标是让人类用户与 LLM 在不读源码的前提下，按统一口径完成以下闭环：

1. 概念标签资产构建（`build_habitat_cbm_labels.py`）
2. 数据配置器校验（`data_loader_habitat_CBM.py`）
3. 三阶段训练（`train_habitat_CBM.py`）
4. 患者级主任务评估（`eval_habitat_CBM.py`）
5. 概念层评估（`eval_habitat_cbm_concepts.py`）
6. 患者级概念干预（`intervene_habitat_cbm.py`）

本指南默认与当前实现（2026-04-18）一致，输出命名对齐 `lab_timeline.md` 与 `paper.md` 的证据链要求。

## Use This Guide When

在以下场景使用本指南：

1. 你需要从 0 到 1 跑通 Habitat-CBM 全流程。
2. 你需要让 LLM 按既定协议生成可运行命令。
3. 你需要核对训练/评估/干预产物是否满足论文回填要求。
4. 你需要排查概念标签、患者 ID、split、checkpoint 对齐问题。

## Hard Contracts

### Contract 1: Patient-Level Protocol

1. 数据划分必须是患者级，`train/val/test` 不能患者泄漏。
2. 主任务指标统一按患者级统计：`AUC/ACC/SEN/SPE/F1`。
3. 概念评估统一按患者级统计：`MAE/RMSE/R2/Pearson r`。
4. 干预统一按患者级概念向量进行替换与重算。

### Contract 2: Concept Bottleneck

模型结构固定为硬瓶颈：

```text
x -> encoder -> c_hat -> dropout -> linear -> y_logit
```

`y` 仅依赖概念 `c_hat`（或干预后的 `c_after`）。

### Contract 3: Config-Driven

主配置文件固定为：

- `habitat_CBM/repo/srcs/args_train_habitat_CBM.json`

一级结构固定为：

1. `paths`
2. `data`
3. `model`
4. `train`
5. `eval`
6. `intervention`
7. `logging`

## Runtime Prerequisites

1. Python 环境包含：`torch`, `monai`, `nibabel`, `numpy`, `scikit-learn`。
2. 推荐从工程根目录执行命令：

```bash
cd /Users/yankeeschen/Documents/Research/bachelor_thesis/idh/codex
```

3. 若环境暂缺 `torch`，仅可做静态检查，不能完整训练/推理：

```bash
python -m py_compile \
  habitat_CBM/repo/srcs/build_habitat_cbm_labels.py \
  habitat_CBM/repo/srcs/data_loader_habitat_CBM.py \
  habitat_CBM/repo/srcs/train_habitat_CBM.py \
  habitat_CBM/repo/srcs/eval_habitat_CBM.py \
  habitat_CBM/repo/srcs/eval_habitat_cbm_concepts.py \
  habitat_CBM/repo/srcs/intervene_habitat_cbm.py
```

## End-to-End Workflow

### Step 0: Prepare Inputs

你至少需要三类输入资产：

1. 患者级划分目录：`habitat_CBM/dataset/splited_data/{train,val,test}`
2. 概念代理表：`concept_proxy_features_<run_id>.csv`
3. 训练配置：`habitat_CBM/repo/srcs/args_train_habitat_CBM.json`

### Step 1: Build Concept Assets

脚本：`srcs/build_habitat_cbm_labels.py`

功能：将代理特征表转换为 CBM 监督资产。

输入：

- `--concept-proxy-csv`

输出：

1. `concept_labels.csv`
2. `concept_statistics.csv`
3. `concept_scaler_stats.json`

命令：

```bash
python habitat_CBM/repo/srcs/build_habitat_cbm_labels.py \
  --concept-proxy-csv habitat_CBM/results/02_habitat/concept_proxy_features_run01.csv \
  --output-label-csv habitat_CBM/results/02_habitat/concept_labels.csv \
  --output-stats-csv habitat_CBM/results/02_habitat/concept_statistics.csv \
  --output-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --std-floor 1e-6
```

内置概念映射（源概念全集 8 维；当前默认训练子集为 `C1~C7`）：

1. `c1 <- c1_h1_t1ce_firstorder_mean`
2. `c2 <- c2_h23_t1ce_firstorder_mean`
3. `c3 <- c3_whole_tumor_shape_sphericity`
4. `c4 <- c4_whole_tumor_flair_ce_volume_ratio`
5. `c5 <- c5_h12_adc_10percentile`
6. `c6 <- c6_h1_cbf_95percentile`
7. `c7 <- c7_h1_volume_ratio`
8. `c8 <- c8_h2_volume_ratio`

关键约束：

1. scaler 只在 `split=train` 上拟合。
2. `std` 受 `std_floor` 下限保护，避免除零。
3. `concept_labels.csv` 要求 `patient_id` 唯一。

### Step 2: Validate Data Loader (Optional But Recommended)

脚本：`srcs/data_loader_habitat_CBM.py`

功能：验证数据集与概念资产是否对齐。

命令：

```bash
python habitat_CBM/repo/srcs/data_loader_habitat_CBM.py \
  --split-root habitat_CBM/dataset/splited_data/train \
  --concept-label-csv habitat_CBM/results/02_habitat/concept_labels.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --selected-concepts C1,C2,C3,C4,C5,C6,C7 \
  --block-depth 5 \
  --max-samples 3
```

启用概念模式后，`__getitem__` 会额外返回：

1. `concept_true_raw`（`[K]`）
2. `concept_true_std`（`[K]`）

强校验内容：

1. `dataset` 中所有 `patient_id` 必须在 `concept_labels.csv` 中存在。
2. `dataset label` 必须与 `concept_labels.csv:y_true` 一致。
3. split 必须一致（例如 train split 不可映射到 val 标签）。
4. 概念维度必须与 `selected_concepts` 长度一致；当前默认 `K=7`，可选全集为 `C1~C8`。
5. scaler 中每个概念 `std > 0`。

### Step 3: Train Habitat-CBM (Stage1 -> Stage2 -> Stage3)

脚本：`srcs/train_habitat_CBM.py`

核心能力：

1. JSON 主配置 + CLI 覆盖
2. 三阶段训练与 early stopping
3. 分阶段 best checkpoint 导出
4. 训练完成自动执行正式评估导出

最小运行命令：

```bash
python habitat_CBM/repo/srcs/train_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --run-id run01 \
  --device cuda:0
```

常用覆盖命令：

```bash
python habitat_CBM/repo/srcs/train_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --run-id run02 \
  --device cuda:0 \
  --epochs-stage1 20 \
  --epochs-stage2 20 \
  --epochs-stage3 30 \
  --batch-size 8
```

可用 CLI 覆盖参数：

1. `--config`
2. `--run-id`
3. `--output-root`
4. `--checkpoint-root`
5. `--device`
6. `--epochs-stage1`
7. `--epochs-stage2`
8. `--epochs-stage3`
9. `--batch-size`

阶段语义：

1. Stage1: `X -> C`（优化概念回归）
2. Stage2: `C_true_std -> Y`（Oracle concept 校准分类头）
3. Stage3: `X -> C -> Y`（联合微调）

监控与早停：

1. Stage1 监控 `val_concept_loss`（越小越好）。
2. Stage2 默认监控患者级 `val_label_loss`，并用 `val_auc` 作为 tie-break。
3. Stage3 默认监控患者级 `val_auc`；若 `val_auc` 不可计算（如单类），回退监控 `-val_total_loss`。

类别不平衡：

1. 训练脚本会始终统计训练集患者级 `pos_weight = N_negative / N_positive` 并写入 `run_train_summary_*.json` 的 `class_balance`。
2. 是否在 Stage2/3 的 BCE 中实际使用该权重，由 `loss.label.use_pos_weight` 和各 stage 的 `label_loss` override 决定；当前默认 Stage2/3 均关闭该项。

### Step 4: Formal Evaluation (Standalone or Auto After Training)

脚本：`srcs/eval_habitat_CBM.py`

说明：

1. 训练脚本结束后会自动调用该评估流程。
2. 你也可以单独执行评估（例如对不同 checkpoint 重评）。

命令示例：

```bash
python habitat_CBM/repo/srcs/eval_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --run-id run01 \
  --output-dir habitat_CBM/results/03_habitat_cbm/run01 \
  --device cuda:0 \
  --splits train,val,test \
  --threshold 0.5 \
  --topk-pool 0
```

评估 CLI 参数：

1. `--config`
2. `--checkpoint`（必填）
3. `--run-id`
4. `--output-dir`
5. `--device`
6. `--threshold`
7. `--topk-pool`
8. `--batch-size`
9. `--num-workers`
10. `--splits`（逗号分隔）

患者级聚合：

1. 分类概率默认对患者所有 block 做均值。
2. 若 `topk_pool > 0`，先按 `abs(prob - 0.5)` 选更自信的 top-k block 再均值。

### Step 5: Evaluate Concept Predictions

脚本：`srcs/eval_habitat_cbm_concepts.py`

输入：`patient_concepts_habitat_cbm_<run_id>.csv`

命令示例：

```bash
python habitat_CBM/repo/srcs/eval_habitat_cbm_concepts.py \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --output-dir habitat_CBM/results/04_concept_eval/run01 \
  --split test \
  --scale raw
```

参数说明：

1. `--split`: `train|val|test|all`
2. `--scale`: `raw|std`

输出指标：

1. `MAE`
2. `RMSE`
3. `R2`
4. `Pearson r`

不可计算场景处理：

1. `R2` 或 `Pearson` 不可算时写 `NaN`。
2. 同时输出原因字段（如 `insufficient_samples`, `zero_variance_true`, `zero_variance_pred`）。

### Step 6: Run Patient-Level Concept Intervention

脚本：`srcs/intervene_habitat_cbm.py`

逻辑：

1. 候选集 = 误判样本 OR 低置信度样本。
2. 对每位患者按 `|c_pred_std - c_true_std|` 从大到小排序。
3. 替换前 `k` 个概念（预算 `k=1,2,4,all`）。
4. 用 `forward_c_to_y` 重算干预后概率。
5. 同时输出全测试集口径与候选集口径指标。

命令示例：

```bash
python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --patient-predictions-csv habitat_CBM/results/03_habitat_cbm/run01/patient_predictions_habitat_cbm_run01.csv \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --output-dir habitat_CBM/results/05_intervention/run01 \
  --split test \
  --budgets 1,2,4,all \
  --threshold 0.5 \
  --low-confidence-margin 0.1 \
  --device cuda:0
```

关键参数：

1. `--split`: `train|val|test|all`
2. `--budgets`: 例如 `1,2,4,all`
3. `--low-confidence-margin`: 满足 `abs(prob_before - threshold) < margin` 视为低置信度

## Config Reference (args_train_habitat_CBM.json)

当前默认配置（节选）位于：

- `habitat_CBM/repo/srcs/args_train_habitat_CBM.json`

建议优先改 JSON，再用 CLI 做少量覆盖。

### paths

1. `split_base_root`: 划分数据根目录
2. `concept_label_csv`: 概念标签表
3. `concept_scaler_json`: 概念标准化统计
4. `runs_root`: 训练日志与 checkpoint 根目录
5. `results_root`: 评估产物根目录
6. `checkpoint_root`: 固定 checkpoint 输出目录；为 `null` 时默认使用 `{runs_root}/{run_id}/checkpoints`

### data

1. 模态列表、VOI 使用策略、2.5D 切块参数
2. 强度归一化与最小前景体素阈值
3. `resize_height/resize_width`

### model

1. `in_channels`
2. `selected_concepts`（默认 `C1~C7`，可选全集 `C1~C8`）
3. `n_concepts`（必须等于 `selected_concepts` 的长度；当前默认 7）
4. `concept_hidden_dim`
5. `label_hidden_dim`（兼容旧配置保留字段，当前单层 `label_head` 不再实际使用）
6. `concept_dropout_p` / `label_dropout_p`
7. `pretrained`

### train

1. 随机种子、设备、batch size、num workers
2. early stopping 参数
3. MONAI 增强参数

### stages

1. `stage1/2/3.epochs`: 三阶段 epoch 数
2. `stage1/2/3.checkpoint_name`: 阶段 best checkpoint 文件名
3. `stage1/2/3.log_csv`: 阶段训练日志文件名
4. 每个 stage 可选覆盖 `optimizer` / `scheduler`

### loss

1. `concept.name`: `mse|l1|smooth_l1`
2. `label.name`: `bce_with_logits|focal_bce_with_logits`
3. `label.use_pos_weight`: 是否使用训练集患者级 `pos_weight`
4. `joint.lambda_c` / `joint.lambda_y`

### optimizer

1. `name`: `adamw|adam|sgd|rmsprop`
2. `lr`, `lr_encoder`, `lr_concept_head`, `lr_label_head`
3. `weight_decay` 与优化器特有参数，如 `betas`, `momentum`, `eps`

### scheduler

1. `name`: `none|cosine_annealing|cosine_annealing_warm_restarts|step|multistep|exponential|reduce_on_plateau`
2. `enabled`: 是否启用 scheduler
3. 余弦退火默认每个 stage 单独按该 stage 的 epoch 数设置 `T_max`

### eval

1. `threshold`
2. `topk_pool`
3. `include_splits`

### intervention

1. `enable`（当前训练脚本不自动跑干预）
2. `budgets`
3. `low_confidence_margin`

### logging

1. `run_id`（可为空，空则自动时间戳）
2. `log_to_file`

## Output Artifacts (Evidence Chain)

### A. Training Artifacts

目录：`{runs_root}/{run_id}/`

1. `run_config_habitat_cbm_<run_id>.json`
2. `stage1_concept_log.csv`
3. `stage2_label_head_log.csv`
4. `stage3_joint_log.csv`

目录：`{checkpoint_root}`（默认 `{runs_root}/{run_id}/checkpoints`）

1. `stage1_best.pt`
2. `stage2_best.pt`
3. `stage3_best.pt`

### B. Main Task Artifacts (Patient-Level)

目录：`{results_root}/{run_id}/`

1. `patient_predictions_habitat_cbm_<run_id>.csv`
2. `metrics_habitat_cbm_<run_id>.csv`
3. `roc_points_habitat_cbm_<run_id>.csv`
4. `confusion_matrix_habitat_cbm_<run_id>.csv`
5. `wrong_cases_habitat_cbm_<run_id>.csv`
6. `run_summary_habitat_cbm_<run_id>.json`
7. `run_train_summary_habitat_cbm_<run_id>.json`

### C. Concept Artifacts

1. `patient_concepts_habitat_cbm_<run_id>.csv`
2. `concept_metrics_by_concept.csv`
3. `concept_metrics_overall.csv`
4. `concept_error_distribution.csv`
5. `concept_error_rank.csv`

同时会输出带后缀版本（如 `*_test_raw.csv`），便于多 split / 多口径并存。

### D. Intervention Artifacts

1. `intervention_case_list.csv`
2. `intervention_per_case.csv`
3. `intervention_metrics_by_budget.csv`
4. `intervention_correction_summary.csv`
5. `intervention_summary.csv`

## CSV Field Contracts

### patient_predictions_habitat_cbm_<run_id>.csv

必须包含：

1. `patient_id`
2. `split`
3. `y_true`
4. `prob_idh_mut`
5. `pred_label`
6. `run_id`
7. `checkpoint_name`

### patient_concepts_habitat_cbm_<run_id>.csv

基础字段：

1. `patient_id`
2. `split`
3. `y_true`
4. `run_id`
5. `checkpoint_name`

每个概念 `c1..c8` 扩展字段：

1. `cX_true_std`
2. `cX_pred_std`
3. `cX_abs_error_std`
4. `cX_true_raw`
5. `cX_pred_raw`
6. `cX_abs_error_raw`

### intervention_per_case.csv

关键字段：

1. `patient_id`
2. `budget_k`
3. `concepts_replaced`（1-based index，`|` 分隔）
4. `prob_before`
5. `prob_after`
6. `pred_before`
7. `pred_after`
8. `corrected_flag`
9. `run_id`
10. `checkpoint_name`

## Quick Smoke Test Protocol

按最小成本验证流程通路：

1. 先构建概念资产（Step 1）。
2. 用 data loader debug 查看 1-3 个样本（Step 2）。
3. 训练时把 `epochs_stage1/2/3` 设为 `1/1/1`（Step 3）。
4. 确认生成了 3 个 best checkpoint 与评估 CSV（Step 3/4）。
5. 概念评估用 `--split test --scale raw` 先跑一遍（Step 5）。
6. 干预用默认预算 `1,2,4,all` 跑一遍（Step 6）。

## Troubleshooting

### Case 1: "Patient ... missing in concept labels"

原因：`concept_labels.csv` 与 split 数据目录患者集合不一致。

处理：

1. 检查 `patient_id` 格式（前导零、空格）。
2. 确认 concept 标签是同一批 split 导出的。

### Case 2: "Label mismatch ... dataset label vs concept y_true"

原因：主标签与概念表标签冲突。

处理：

1. 回溯 `concept_proxy_features_<run_id>.csv` 的 `y_true` 来源。
2. 确认 `split` 与患者目录一致。

### Case 3: Stage2/3 AUC 出现 NaN

原因：验证集单类或样本过少。

处理：

1. 属于预期保护逻辑，脚本会自动回退到 loss 监控。
2. 若持续发生，优先检查 split 分层质量。

### Case 4: 干预脚本报概念列缺失

原因：`patient_concepts` 文件不完整或列名不匹配。

处理：

1. 确认输入为 `eval_habitat_CBM.py` 生成的概念表。
2. 至少要有 std 列或 raw 列成对存在。

### Case 5: 输出目录混乱

原因：CLI `--output-root` 会同时覆盖 `runs_root` 与 `results_root`。

处理：

1. 推荐通过 JSON 分别配置 `runs_root` 和 `results_root`。
2. 或者显式传 `--checkpoint-root` 保障 checkpoint 位置可控。

## LLM Usage Contract

当你把本文件提供给 LLM 作为上下文时，要求 LLM 遵守：

1. 先读 `args_train_habitat_CBM.json` 再给命令，不可臆造路径。
2. 所有主任务结果必须按患者级统计，不可用 block 级替代。
3. 概念训练使用 standardized 概念；论文报告优先 raw 口径并保留 std 口径。
4. 干预必须按患者级 `|c_pred_std-c_true_std|` 排序替换。
5. 输出检查必须覆盖 `run_id/split/patient_id/checkpoint_name` 可追溯字段。

建议 LLM 输出格式：

1. 先给可执行命令（按 Step 1-6）。
2. 再给每步产物检查清单。
3. 最后给失败排查路径（按本节 Troubleshooting）。

## One-Page Command Template

```bash
# 0) 进入工程目录
cd /Users/yankeeschen/Documents/Research/bachelor_thesis/idh/codex

# 1) 构建概念资产
python habitat_CBM/repo/srcs/build_habitat_cbm_labels.py \
  --concept-proxy-csv habitat_CBM/results/02_habitat/concept_proxy_features_run01.csv \
  --output-label-csv habitat_CBM/results/02_habitat/concept_labels.csv \
  --output-stats-csv habitat_CBM/results/02_habitat/concept_statistics.csv \
  --output-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json

# 2) 数据校验（可选）
python habitat_CBM/repo/srcs/data_loader_habitat_CBM.py \
  --split-root habitat_CBM/dataset/splited_data/train \
  --concept-label-csv habitat_CBM/results/02_habitat/concept_labels.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --max-samples 2

# 3) 三阶段训练（自动评估）
python habitat_CBM/repo/srcs/train_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --run-id run01 \
  --device cuda:0

# 4) 独立评估（可选）
python habitat_CBM/repo/srcs/eval_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --run-id run01 \
  --output-dir habitat_CBM/results/03_habitat_cbm/run01 \
  --device cuda:0 \
  --splits train,val,test

# 5) 概念评估
python habitat_CBM/repo/srcs/eval_habitat_cbm_concepts.py \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --output-dir habitat_CBM/results/04_concept_eval/run01 \
  --split test \
  --scale raw

# 6) 概念干预
python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --patient-predictions-csv habitat_CBM/results/03_habitat_cbm/run01/patient_predictions_habitat_cbm_run01.csv \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --output-dir habitat_CBM/results/05_intervention/run01 \
  --split test \
  --budgets 1,2,4,all \
  --threshold 0.5 \
  --low-confidence-margin 0.1 \
  --device cuda:0
```
