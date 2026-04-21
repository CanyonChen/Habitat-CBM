---
name: habitat-cbm-model
updated: 2026-04-21
description: Habitat-CBM 完整训练-评估-干预接口契约（JSON 配置驱动版）。
---

# Habitat-CBM 模型接口契约（可运行版）

## 1. 目标

当前 Habitat-CBM 代码已从“工具函数自检层”升级为完整闭环：

1. 标签资产构建：`build_habitat_cbm_labels.py`
2. 数据加载（含概念）：`data_loader_habitat_CBM.py`
3. 三阶段训练：`train_habitat_CBM.py`
4. 患者级评估：`eval_habitat_CBM.py`
5. 概念指标评估：`eval_habitat_cbm_concepts.py`
6. 患者级概念干预：`intervene_habitat_cbm.py`

---

## 2. 核心模型结构

文件：`repo/models/habitat_CBM.py`

结构契约：

```text
z = encoder(x)                           # [B, 512]
c_hat = concept_head(z)                  # [B, K]
y_logit = linear(dropout(c_hat))         # [B, 1]
```

其中 `K = len(selected_concepts)`；当前默认 `selected_concepts = [C1, C2, C3, C4, C6]`，可选概念全集仍为 `C1..C8`。

硬瓶颈约束：

1. `y` 仅由 `c_hat`（或干预后 `c_after`）生成。
2. 禁止 `z -> y` 旁路。

---

## 3. 配置契约

主配置文件：`repo/srcs/args_train_habitat_CBM.json`

固定一级结构：

1. `paths`
2. `data`
3. `model`
4. `train`
5. `eval`
6. `intervention`
7. `logging`

训练入口默认读取该文件，可用 CLI 覆盖关键参数。

---

## 4. 数据接口契约

文件：`repo/srcs/data_loader_habitat_CBM.py`

`HabitatIDHBlockDataset` 在原通用数据集基础上直接扩展，新增可选参数：

1. `concept_label_csv`
2. `concept_scaler_json`

启用概念模式后 `__getitem__` 输出包含：

1. `image`
2. `label`
3. `patient_id`
4. `slice_index`
5. `concept_true_raw`（shape `[K]`）
6. `concept_true_std`（shape `[K]`）

强校验：

1. 患者 ID 全覆盖
2. `label` 与 `concept_labels.csv:y_true` 一致
3. 概念维度必须与 `selected_concepts` 一致（当前默认 `K=5`）
4. scaler 统计合法（`std > 0`）

---

## 5. 训练契约

文件：`repo/srcs/train_habitat_CBM.py`

### 5.1 三阶段

1. Stage 1：`X -> C`（概念预训练）
2. Stage 2：`C -> Y`（Oracle concept 校准分类头）
3. Stage 3：`X -> C -> Y`（联合微调）

### 5.2 关键特性

1. JSON 主配置 + CLI 覆盖
2. 早停
3. 分阶段 best checkpoint：
   - `stage1_best.pt`
   - `stage2_best.pt`
   - `stage3_best.pt`
4. 分阶段日志：
   - `stage1_concept_log.csv`
   - `stage2_label_head_log.csv`
   - `stage3_joint_log.csv`
5. 标签 loss 的类别重加权由配置决定；当前默认 Stage2/3 通过 stage override 关闭 `pos_weight`，但仍记录患者级原始 `pos_weight` 供审计
6. 训练结束自动触发患者级评估导出

### 5.3 CLI 覆盖

```bash
python habitat_CBM/repo/srcs/train_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --run-id run01 \
  --output-root habitat_CBM/runs/03_habitat_cbm \
  --checkpoint-root habitat_CBM/runs/03_habitat_cbm/run01/checkpoints \
  --device cuda:0 \
  --epochs-stage1 20 \
  --epochs-stage2 20 \
  --epochs-stage3 30 \
  --batch-size 8
```

---

## 6. 评估契约

文件：`repo/srcs/eval_habitat_CBM.py`

保留并兼容原工具函数：

1. `predict_batch`
2. `prepare_intervention_order`
3. `forward_with_intervention`
4. `aggregate_patient_probabilities`
5. `aggregate_patient_concepts`

新增正式评估入口（checkpoint + 真实数据），输出：

1. `patient_predictions_habitat_cbm_<run_id>.csv`
2. `patient_concepts_habitat_cbm_<run_id>.csv`
3. `metrics_habitat_cbm_<run_id>.csv`
4. `roc_points_habitat_cbm_<run_id>.csv`
5. `confusion_matrix_habitat_cbm_<run_id>.csv`
6. `wrong_cases_habitat_cbm_<run_id>.csv`
7. `run_summary_habitat_cbm_<run_id>.json`

---

## 7. 概念评估与干预契约

### 7.1 概念评估

文件：`repo/srcs/eval_habitat_cbm_concepts.py`

输入：`patient_concepts_habitat_cbm_<run_id>.csv`

输出：

1. `concept_metrics_by_concept.csv`
2. `concept_metrics_overall.csv`
3. `concept_error_distribution.csv`
4. `concept_error_rank.csv`

指标：`MAE / RMSE / R2 / Pearson r`。

### 7.2 干预评估

文件：`repo/srcs/intervene_habitat_cbm.py`

预算：`k=1,2,4,all`。

逻辑：患者级 `|c_pred_std-c_true_std|` 排序后替换，再经 `forward_c_to_y` 重算。

输出：

1. `intervention_case_list.csv`
2. `intervention_per_case.csv`
3. `intervention_metrics_by_budget.csv`
4. `intervention_correction_summary.csv`
5. `intervention_summary.csv`

---

## 8. 标签资产构建契约

文件：`repo/srcs/build_habitat_cbm_labels.py`

输入：`concept_proxy_features_<run_id>.csv`

标签资产固定导出 `c1_true...c8_true` 全量列；训练/评估阶段可通过 `model.selected_concepts` 选择子集（当前默认 `C1/C2/C3/C4/C6`）。输出：

1. `concept_labels.csv`
2. `concept_statistics.csv`
3. `concept_scaler_stats.json`

约束：scaler 仅使用 `split=train` 拟合，并带 `std_floor` 防除零。

---

## 9. 运行顺序（推荐）

```bash
# 1) 构建概念标签资产
python habitat_CBM/repo/srcs/build_habitat_cbm_labels.py \
  --concept-proxy-csv habitat_CBM/results/02_habitat/concept_proxy_features_run01.csv \
  --output-label-csv habitat_CBM/results/02_habitat/concept_labels.csv \
  --output-stats-csv habitat_CBM/results/02_habitat/concept_statistics.csv \
  --output-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json

# 2) 三阶段训练 + 自动患者级评估
python habitat_CBM/repo/srcs/train_habitat_CBM.py --run-id run01 --device cuda:0

# 3) 概念评估
python habitat_CBM/repo/srcs/eval_habitat_cbm_concepts.py \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --output-dir habitat_CBM/results/04_concept_eval/run01 \
  --split test \
  --scale raw

# 4) 患者级概念干预
python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --patient-predictions-csv habitat_CBM/results/03_habitat_cbm/run01/patient_predictions_habitat_cbm_run01.csv \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --output-dir habitat_CBM/results/05_intervention/run01 \
  --split test \
  --budgets 1,2,4,all
```

---

## 10. 环境说明

当前本地环境若无 `torch`，可先做语法检查：

```bash
python -m py_compile habitat_CBM/repo/srcs/train_habitat_CBM.py
```

真实训练/推理/干预需在安装 `torch`、`monai`、`nibabel`、`scikit-learn` 的环境运行。
