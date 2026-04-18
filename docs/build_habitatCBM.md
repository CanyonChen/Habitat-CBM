---
name: build-habitat-cbm
updated: 2026-04-18
description: Habitat-CBM 当前代码实现的落地构建指南（JSON 配置 + 三阶段 + 患者级评估 + 概念评估 + 干预）。
---

# Habitat-CBM 构建指南（当前实现）

## 1. 前置输入

1. 患者级划分目录：`habitat_CBM/dataset/splited_data/{train,val,test}`
2. 概念代理表：`concept_proxy_features_<run_id>.csv`
3. 训练配置：`habitat_CBM/repo/srcs/args_train_habitat_CBM.json`

---

## 2. Step A：构建 concept 训练资产

脚本：`repo/srcs/build_habitat_cbm_labels.py`

```bash
python habitat_CBM/repo/srcs/build_habitat_cbm_labels.py \
  --concept-proxy-csv habitat_CBM/results/02_habitat/concept_proxy_features_run01.csv \
  --output-label-csv habitat_CBM/results/02_habitat/concept_labels.csv \
  --output-stats-csv habitat_CBM/results/02_habitat/concept_statistics.csv \
  --output-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json
```

输出：

1. `concept_labels.csv`
2. `concept_statistics.csv`
3. `concept_scaler_stats.json`

说明：

1. scaler 仅在 `train` split 拟合。
2. 内置 `std_floor` 防止标准差过小导致除零。

---

## 3. Step B：数据加载调试（可选）

脚本：`repo/srcs/data_loader_habitat_CBM.py`

```bash
python habitat_CBM/repo/srcs/data_loader_habitat_CBM.py \
  --split-root habitat_CBM/dataset/splited_data/train \
  --concept-label-csv habitat_CBM/results/02_habitat/concept_labels.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --max-samples 2
```

检查点：

1. `concept_true_raw` 与 `concept_true_std` 均返回。
2. patient label 与 `concept_labels.csv:y_true` 一致。

---

## 4. Step C：三阶段训练（主流程）

脚本：`repo/srcs/train_habitat_CBM.py`

### 4.1 默认配置运行

```bash
python habitat_CBM/repo/srcs/train_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --run-id run01 \
  --device cuda:0
```

### 4.2 配置覆盖运行

```bash
python habitat_CBM/repo/srcs/train_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --run-id run02 \
  --output-root habitat_CBM/runs/03_habitat_cbm \
  --checkpoint-root habitat_CBM/runs/03_habitat_cbm/run02/checkpoints \
  --device cuda:0 \
  --epochs-stage1 10 \
  --epochs-stage2 10 \
  --epochs-stage3 20 \
  --batch-size 4
```

### 4.3 训练输出

1. checkpoint：
   - `stage1_best.pt`
   - `stage2_best.pt`
   - `stage3_best.pt`
2. 日志：
   - `stage1_concept_log.csv`
   - `stage2_label_head_log.csv`
   - `stage3_joint_log.csv`
3. 运行配置快照：
   - `run_config_habitat_cbm_<run_id>.json`

说明：

1. Stage2/3 自动启用患者级类别不平衡 `pos_weight`。
2. 训练结束自动执行患者级评估导出。

---

## 5. Step D：患者级评估（可单独执行）

脚本：`repo/srcs/eval_habitat_CBM.py`

```bash
python habitat_CBM/repo/srcs/eval_habitat_CBM.py \
  --config habitat_CBM/repo/srcs/args_train_habitat_CBM.json \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --run-id run01 \
  --output-dir habitat_CBM/results/03_habitat_cbm/run01 \
  --device cuda:0 \
  --splits train,val,test
```

输出：

1. `patient_predictions_habitat_cbm_<run_id>.csv`
2. `patient_concepts_habitat_cbm_<run_id>.csv`
3. `metrics_habitat_cbm_<run_id>.csv`
4. `roc_points_habitat_cbm_<run_id>.csv`
5. `confusion_matrix_habitat_cbm_<run_id>.csv`
6. `wrong_cases_habitat_cbm_<run_id>.csv`
7. `run_summary_habitat_cbm_<run_id>.json`

---

## 6. Step E：概念层评估

脚本：`repo/srcs/eval_habitat_cbm_concepts.py`

```bash
python habitat_CBM/repo/srcs/eval_habitat_cbm_concepts.py \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --output-dir habitat_CBM/results/04_concept_eval/run01 \
  --split test \
  --scale raw
```

输出：

1. `concept_metrics_by_concept.csv`
2. `concept_metrics_overall.csv`
3. `concept_error_distribution.csv`
4. `concept_error_rank.csv`

---

## 7. Step F：患者级概念干预

脚本：`repo/srcs/intervene_habitat_cbm.py`

```bash
python habitat_CBM/repo/srcs/intervene_habitat_cbm.py \
  --checkpoint habitat_CBM/runs/03_habitat_cbm/run01/checkpoints/stage3_best.pt \
  --patient-predictions-csv habitat_CBM/results/03_habitat_cbm/run01/patient_predictions_habitat_cbm_run01.csv \
  --patient-concepts-csv habitat_CBM/results/03_habitat_cbm/run01/patient_concepts_habitat_cbm_run01.csv \
  --concept-scaler-json habitat_CBM/results/03_habitat_cbm/concept_scaler_stats.json \
  --output-dir habitat_CBM/results/05_intervention/run01 \
  --split test \
  --budgets 1,2,4,all \
  --threshold 0.5
```

输出：

1. `intervention_case_list.csv`
2. `intervention_per_case.csv`
3. `intervention_metrics_by_budget.csv`
4. `intervention_correction_summary.csv`
5. `intervention_summary.csv`

---

## 8. 与论文/时间表文件名对齐

当前实现已与 `lab_timeline.md` / `paper.md` 所需主产物对齐：

1. 患者级主任务：`metrics_*` / `roc_points_*` / `confusion_matrix_*`
2. 概念层：`patient_concepts_*` + `concept_metrics_*`
3. 干预层：`intervention_*`

---

## 9. 最小验收顺序

1. `build_habitat_cbm_labels.py` 产出 3 个标签资产。
2. `train_habitat_CBM.py` 跑完 3 阶段并生成 3 个 best checkpoint。
3. `eval_habitat_CBM.py` 导出患者级主任务与概念表。
4. `eval_habitat_cbm_concepts.py` 导出概念指标与偏差分布。
5. `intervene_habitat_cbm.py` 导出四个预算的干预统计。

---

## 10. 环境提示

若当前环境无 `torch`，可先做语法检查：

```bash
python -m py_compile habitat_CBM/repo/srcs/train_habitat_CBM.py
```

正式训练/推理需在安装 `torch`、`monai`、`nibabel`、`scikit-learn` 的环境执行。
