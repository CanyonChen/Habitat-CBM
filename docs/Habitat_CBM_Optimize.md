---
name: habitat-cbm-optimize
description: 历史版 Habitat-CBM 优化诊断与实现路线记录；当前实现请结合 user guide 与模型接口文档阅读。
---

# Habitat-CBM 优化诊断 Skill

> 更新说明（2026-04-20）：本文档保留了 Habitat-CBM 从“仅有骨架与工具函数”走向完整闭环的历史诊断过程。文中的 blocker、缺失项和部分“当前实现”表述对应的是早期阶段，不再等同于当前代码状态。当前实现请优先参考 `docs/user_guide_Habitat_CBM.md` 与 `docs/habitat_CBM_model.md`。

## Overview

本文件用于给用户、维护者和 LLM/Codex 提供 Habitat-CBM 优化上下文。阅读本文件后，应能判断当前 Habitat-CBM 实现为什么还不能支撑论文主实验，以及下一步应按什么顺序补齐训练、评估、概念学习和概念干预闭环。

历史结论（对应 pre-integration 阶段）：

1. `models/habitat_CBM.py` 的核心模型结构基本正确，已经形成 `X -> C -> Y` 的硬概念瓶颈。
2. `srcs/train_habitat_CBM.py` 和 `srcs/eval_habitat_CBM.py` 仍主要是工具函数与随机张量自检，不是完整实验 pipeline。
3. 论文和实验时间表需要的是患者级 Habitat-CBM 主结果、8 个概念的患者级回归评估，以及 `k=1,2,4,all` 的患者级 concept intervention；当前代码还不能直接产出这些结果。

本文档不替代 `build_habitatCBM.md` 和 `habitat_CBM_model.md`。它的作用是指出现状问题、优化方向和实施顺序，避免后续实现时只补局部函数而没有形成论文证据链。

## Historical Problem Overview

先看这一节即可快速把握当前 Habitat-CBM 的主要问题。后续章节会逐项展开原因、影响和优化方向。

1. 模型结构基本正确，但还只是 `X -> C -> Y` 的网络骨架，没有形成完整实验 pipeline。
2. `train_habitat_CBM.py` 目前不能直接训练真实数据，只能做随机张量 self-check。
3. 文档中的训练命令和实际 CLI 不一致，真实训练参数尚未实现。
4. 缺少 `concept_labels.csv`、`concept_scaler_stats.json` 和 train-only concept 标准化流程。
5. 缺少 CBM 专用 dataset wrapper，现有 dataloader 只返回 `image/label`，没有 `concept_true_raw/concept_true_std`。
6. Stage 1、Stage 2、Stage 3 尚未串成完整三阶段训练流程，也没有 checkpoint、early stopping 和训练日志。
7. 缺少患者级主任务评估导出，无法生成 Habitat-CBM 的 AUC、ACC、SEN、SPE、F1、ROC、混淆矩阵和错误病例表。
8. 缺少患者级 concept prediction 表和 8 个概念的 MAE、RMSE、R2、Pearson r 评估脚本。
9. 当前 concept intervention 是 block 级工具函数，不符合论文要求的患者级概念向量干预。
10. 分类损失尚未处理类别不平衡，和论文实验设置及 baseline 口径不一致。
11. raw concept 直接训练会有尺度主导风险，必须统一使用 standardized concept 训练，并在评估时明确 raw/standardized 口径。
12. 输入归一化口径与论文描述需要重新统一记录，避免方法描述和实际实验不一致。
13. 当前本地 Python 环境缺少 `torch`，只能做语法检查，不能完成真实深度学习 self-check。

## Use This Skill When

使用本文件作为上下文，当任务涉及以下目标：

1. 修复或补齐 Habitat-CBM 主模型训练流程。
2. 将 concept proxy features 接入 CBM 训练。
3. 实现三阶段训练：concept pretraining、label head calibration、joint finetuning。
4. 导出患者级 IDH 预测结果和主任务指标。
5. 评估 8 个概念的 MAE、RMSE、R2、Pearson r。
6. 实现患者级 concept intervention，并统计干预前后性能变化和误判纠正率。
7. 检查 Habitat-CBM 是否真正满足论文中的“影像 -> 概念 -> 预测”闭环。

不要把本文件用于大规模重写 ResNet-18 baseline 或 Radiomics + Logistic Regression baseline。除非为了复用其数据加载、患者级聚合、指标导出和 run artifact 组织方式，否则应尽量保持 baseline 不动。

## Ground Truth From Paper And Timeline

### 论文要求

论文中的 Habitat-CBM 不是普通 ResNet 分类器，而是面向胶质瘤 IDH 状态预测的可解释概念瓶颈模型。实现必须满足以下口径：

1. 数据划分单位是患者级，不能让同一患者的不同 2.5D block 跨越 `train/val/test`。
2. 输入是 2.5D 多模态 MRI block，六个模态为 `t1/t1ce/t2/t2flair/adc/cbf`，可按现有协议追加 `functional/voi` mask 通道。
3. 同一患者的所有有效 block 共享同一个患者级 `y_true` 和同一个 8 维 concept label。
4. 模型先预测概念向量 `c_hat`，再只通过 `c_hat` 预测 IDH 状态；分类头不能访问 encoder feature `z`。
5. 测试阶段先得到 block 级分类概率和 block 级概念预测，再按患者聚合，形成患者级分类概率和患者级概念输出。
6. 主任务在患者级统计 `AUC / ACC / SEN / SPE / F1`。
7. 概念层在患者级统计每个概念的 `MAE / RMSE / R2 / Pearson r`，并输出概念偏差分布。
8. 概念干预在患者级执行，按概念偏差排序，替换 `k = 1, 2, 4, all` 个概念，再将修正后的概念向量送入 `label_head`。
9. 干预评估需要报告干预前后性能变化、误判纠正率、平均干预概念数，并至少支持典型病例分析。

### 时间表要求

实验时间表对 Habitat-CBM 的硬要求是：

1. 在 Habitat-CBM 主模型阶段，跑通三阶段机制并检查概念维度、loss 和存盘逻辑。
2. Stage 1 记录 concept loss 曲线，检查是否有 concept 维度崩坏。
3. Stage 2 使用真实 concept 训练分类头，验证 `concept -> label` 映射是否基本可用。
4. Stage 3 联合微调，导出患者级分类结果和 concept 结果。
5. 至少完成多次正式运行，锁定最终主模型版本。
6. 计算 8 个 concept 的 `MAE / RMSE / R2 / Pearson r`。
7. 运行 `k = 1, 2, 4, all` 的 concept intervention。
8. 导出主表、主图、概念指标表、概念偏差图、intervention 结果和典型病例素材。

## Historical Implementation Snapshot

### `models/habitat_CBM.py`

当时的目标结构如下；当前代码已经进一步更新为可参数化概念子集 + 单层 `Dropout -> Linear` 的 `label_head`：

```text
z = encoder(x)                      # [B, 512]
c_hat = concept_head(z)             # [B, K]
y_logit = linear(dropout(c_hat))    # [B, 1]
```

已有优点：

1. `ResNet18Backbone` 复用现有 ResNet-18 第一层通道适配逻辑。
2. `HabitatCBM.forward_x_to_c()` 支持 `X -> C`。
3. `HabitatCBM.forward_c_to_y()` 支持 `C -> Y`。
4. `HabitatCBM.forward_x_to_cy()` 支持完整 `X -> C -> Y`。
5. `y_logit` 只依赖 `c_hat`，没有显式 `z -> y` 旁路，符合 hard bottleneck。

主要注意点：

1. 模型只定义结构，不负责训练、评估、标准化、干预统计或结果导出。
2. 当前代码中的 `label_head` 已简化为单层线性头；在 `model.eval()` 下，患者级推理/干预统一采用“先聚合患者级概念，再调用 `forward_c_to_y`”的口径，避免训练期/评估期写法分叉。

### `srcs/train_habitat_CBM.py`

当前文件是训练工具层，不是完整训练脚本。它已提供：

1. `set_train_stage(model, stage)`
2. `get_param_groups(...)`
3. `compute_concept_loss(...)`
4. `compute_label_loss(...)`
5. `compute_joint_loss(...)`
6. `stage1_train_step(...)`
7. `stage2_train_step(...)`
8. `stage3_train_step(...)`
9. 随机张量 self-check CLI。

缺失内容包括：

1. 真实数据集读取。
2. concept label 读取和标准化。
3. `DataLoader` 构建。
4. epoch loop。
5. validation loop。
6. early stopping。
7. checkpoint 保存。
8. 三阶段顺序执行和 stage 间权重加载。
9. 训练日志导出。
10. 最终患者级预测和指标导出。

### `srcs/eval_habitat_CBM.py`

当前文件是推理工具层，不是完整评估脚本。它已提供：

1. `predict_batch(model, x)`
2. `prepare_intervention_order(...)`
3. `forward_with_intervention(...)`
4. `aggregate_patient_probabilities(...)`
5. `aggregate_patient_concepts(...)`
6. 随机张量 self-check CLI。

缺失内容包括：

1. checkpoint 加载。
2. 真实 `DataLoader` 推理。
3. patient-level prediction table。
4. `AUC / ACC / SEN / SPE / F1` 计算。
5. ROC 原始点导出。
6. 混淆矩阵导出。
7. wrong cases 导出。
8. patient-level concept table。
9. concept metrics。
10. patient-level intervention summary。

### `srcs/data_loader.py`

当前 `HabitatIDHBlockDataset` 能返回：

1. `image`
2. `label`
3. `patient_id`
4. `slice_index`
5. `paths`

但 Habitat-CBM 训练还需要：

1. `concept_true_raw`: 8 维原始概念值。
2. `concept_true_std`: 8 维标准化概念值。
3. 严格校验 `dataset label` 与 `concept_labels.csv` 中 `y_true` 是否一致。

因此需要新增 dataset wrapper 或扩展专用 dataset，而不是直接把 CBM 逻辑硬塞回通用 data loader。

### `srcs/get_habitat_radiomics.py`

当前脚本能导出 `concept_proxy_features_{run_id}.csv`，并包含：

1. `patient_id`
2. `split`
3. `y_true`
4. `label_name`
5. `c1_h1_t1ce_firstorder_mean`
6. `c2_h23_t1ce_firstorder_mean`
7. `c3_whole_tumor_shape_sphericity`
8. `c4_whole_tumor_flair_ce_volume_ratio`
9. `c5_h12_adc_10percentile`
10. `c6_h1_cbf_95percentile`
11. `c7_h1_volume_ratio`
12. `c8_h2_volume_ratio`

但该文件仍是原始 concept proxy 表，还没有变成 CBM 训练专用的 `concept_labels.csv` 和 train-only scaler 资产。

## Historical Blocking Problems

> 注：本节记录的是早期实现阶段的 blocker 列表，其中多项已经在当前代码中解决。保留它们的目的是追溯设计决策，而不是描述当前仓库仍然缺什么。

### P0. `train_habitat_CBM.py` 无法直接训练真实数据

问题：

1. CLI 只支持 self-check 参数，例如 `--stage`、`--batch-size`、`--height`。
2. 文档中的真实训练参数尚未实现，例如 `--split-base-root`、`--concept-label-csv`、`--concept-scaler-json`、`--output-root`、`--run-id`。
3. 没有读取 `train/val/test`，没有 epoch loop，没有 validation，没有 checkpoint。

影响：

1. 无法产出 `stage1_best.pt`、`stage2_best.pt`、`stage3_best.pt`。
2. 无法生成 Habitat-CBM 主结果。
3. 无法支撑论文表 4.1、图 4.1、图 4.2。

优化方向：

1. 保留当前工具函数。
2. 在同文件中增加完整 CLI 和 `run_train_pipeline()`，或新增主训练脚本调用这些工具函数。
3. 复用 `baseline_ResNet18.py` 的训练日志、checkpoint、患者级评估和结果导出范式。

### P0. 缺少 concept label 构建与标准化闭环

问题：

1. `concept_proxy_features_{run_id}.csv` 只是原始特征表。
2. 训练需要统一列名 `c1_true ... c8_true`。
3. 训练 loss 需要 `c_true_std`，但目前没有 `concept_scaler_stats.json`。
4. 还没有 train-only scaler，容易误用 val/test 统计量造成数据泄漏。

影响：

1. Stage 1 和 Stage 3 没有真实 concept supervision。
2. raw concept 量纲差异很大，直接 MSE 会让高尺度概念主导 loss。
3. 概念评估无法可靠回到 raw scale。

优化方向：

1. 新增 `srcs/build_habitat_cbm_labels.py`。
2. 从 concept proxy 表映射为 `concept_labels.csv`。
3. 只用 `split == train` 拟合每个概念的 mean/std。
4. 导出 `concept_scaler_stats.json`，供 dataset、训练、评估、干预共同使用。

### P0. 缺少 CBM dataset wrapper

问题：

1. `HabitatIDHBlockDataset` 只返回 `image/label`。
2. `stage1_train_step` 和 `stage3_train_step` 需要 `c_true_std`。
3. `stage2_train_step` 需要真实 concept 作为输入。

影响：

1. 训练脚本无法把患者级 concept label 广播到同一患者所有 block。
2. 无法检查 `data_loader` 标签和 concept 表中标签是否一致。
3. 容易出现 patient ID 格式不一致、前导零丢失、标签错配等隐性错误。

优化方向：

1. 新增 `srcs/habitat_cbm_data.py`。
2. 实现 `HabitatCBMBlockDataset`，内部包装 `HabitatIDHBlockDataset`。
3. 在 `__getitem__` 中追加 `concept_true_raw` 和 `concept_true_std`。
4. 初始化时检查所有 patient 都有概念标签。
5. 初始化时检查 `label` 与 `concept_labels.csv/y_true` 一致。

### P0. 患者级评估导出缺失

问题：

1. `eval_habitat_CBM.py` 只有简单聚合工具。
2. 没有 Habitat-CBM 版本的 `patient_predictions_habitat_cbm_runXX.csv`。
3. 没有主任务 `metrics_habitat_cbm_runXX.csv`。
4. 没有 ROC、confusion matrix、wrong cases。

影响：

1. 无法与 ResNet-18 和 Radiomics LR 公平比较。
2. 无法填充论文第 4 章主任务表格与图。
3. 无法为 intervention 选择误判病例和低置信度病例。

优化方向：

1. 复用 `baseline_ResNet18.py` 中的 patient-level 聚合和指标计算思路。
2. 对每个 split 导出 patient-level predictions。
3. 保存 `AUC / ACC / SEN / SPE / F1`。
4. 导出 ROC 原始点和混淆矩阵。
5. 导出 wrong cases 和 low-confidence cases。

### P0. 概念评估脚本缺失

问题：

1. 没有 `eval_habitat_cbm_concepts.py`。
2. 没有从 patient-level concept table 计算 8 个概念的回归指标。
3. 没有 concept error distribution 和 error rank。

影响：

1. 无法证明 Habitat-CBM “确实学到了预定义概念”。
2. 无法填充论文表 4.2、概念散点图、概念偏差图。
3. 无法为 intervention 的 concept 排序提供患者级依据。

优化方向：

1. 评估脚本读取 `patient_concepts_habitat_cbm_runXX.csv`。
2. 对 `c1_true/c1_pred ... c8_true/c8_pred` 逐维计算 MAE、RMSE、R2、Pearson r。
3. 同时保存 overall mean 和 per-concept metrics。
4. 输出每位患者每个概念的 absolute error，供干预排序使用。

### P0. 当前 intervention 是 block 级，不符合论文患者级口径

问题：

1. `forward_with_intervention()` 对 batch 中每个 block 单独计算误差排序。
2. 当前逻辑替换 block-level `c_pred`，再对每个 block 得到 `y_after`。
3. 论文要求先得到患者级概念预测向量，再进行患者级概念替换。

影响：

1. block-level 干预结果与论文中 patient-level intervention 口径不一致。
2. 该歧义只在非线性 `label_head` 下成立；当前代码的线性 `label_head` 已消除这一点，但仍统一采用患者级概念聚合后再前向的实现口径。
3. 若不修正，干预表格的解释会不严谨。

优化方向：

1. 标准推理阶段保存 block-level `c_hat`。
2. 先按患者聚合 `c_pred_patient = mean(c_hat_block)`。
3. 取患者级 `c_true_std_patient`。
4. 按 `abs(c_pred_patient - c_true_std_patient)` 排序。
5. 对 `k=1,2,4,8` 替换患者级概念向量。
6. 使用 `model.forward_c_to_y(c_after_patient)` 得到干预后患者级概率。

### P1. 分类损失没有类别不平衡处理

问题：

1. 论文实验设置写明训练阶段采用类别权重。
2. 当前 `compute_label_loss()` 直接调用 `binary_cross_entropy_with_logits()`。
3. 没有 `pos_weight` 或 sample weight。

影响：

1. Habitat-CBM 与 baseline 的训练口径不一致。
2. 类别不平衡时，分类头可能偏向多数类。

优化方向：

1. 在训练 pipeline 中按 train patient 统计类别数。
2. 对 BCEWithLogits 使用 `pos_weight`，或在 loss 中支持 sample weight。
3. 确保 Stage 2 和 Stage 3 使用同一类不平衡策略。

### P1. 输入归一化口径需要统一记录

问题：

1. 论文方法描述包含线性归一化到 `[0,255]`。
2. 当前 `data_loader.py` 默认使用非零区域 z-score。
3. baseline 与 Habitat-CBM 如果使用不同归一化，会影响可比性。

影响：

1. 论文方法和代码实现可能不一致。
2. 不同模型间比较可能受输入尺度差异影响。

优化方向：

1. 冻结最终实验口径，优先与已跑通 baseline 保持一致。
2. 在 config 和 run summary 中明确记录 `intensity_norm`。
3. 论文中按最终实验口径回写，不要保留与实现不一致的描述。

### P1. 文档与代码状态不一致

问题：

1. `build_habitatCBM.md` 描述的是完整流程。
2. 实际脚本只实现了其中一部分。
3. 用户或 LLM 可能误以为命令已经可直接执行。

影响：

1. 后续 agent 可能按文档命令运行，遇到参数不存在错误。
2. 容易低估剩余工作量。

优化方向：

1. 在优化文档中明确当前脚本状态。
2. 后续实现完成后，再同步更新 `build_habitatCBM.md` 的“当前状态”和命令模板。
3. 不要在代码未实现前把工具函数称为完整训练脚本。

### P2. 当前本地 Python 环境缺 `torch`

问题：

1. 当前 shell 下运行 `train_habitat_CBM.py` 或 `eval_habitat_CBM.py` 会报 `ModuleNotFoundError: No module named 'torch'`。
2. `python -m py_compile` 可以通过，但不能验证真实前向。

影响：

1. 本地只能做语法检查，不能做 smoke test。
2. 训练实现完成后仍需要进入正确深度学习环境验证。

优化方向：

1. 在最终训练前激活包含 `torch`、`torchvision`、`monai`、`nibabel` 的环境。
2. 将环境信息写入 run summary。
3. 在文档中区分“语法检查通过”和“深度学习运行通过”。

## Optimization Blueprint

### Step 1: Build CBM Label Assets

新增脚本：`srcs/build_habitat_cbm_labels.py`

输入：

1. `concept_proxy_features_{run_id}.csv`

输出：

1. `results/02_habitat/concept_labels.csv`
2. `results/02_habitat/concept_statistics.csv`
3. `results/03_habitat_cbm/concept_scaler_stats.json`

必须实现：

1. 检查必需列是否存在。
2. 映射列名：
   - `c1_true <- c1_h1_t1ce_firstorder_mean`
   - `c2_true <- c2_h23_t1ce_firstorder_mean`
   - `c3_true <- c3_whole_tumor_shape_sphericity`
   - `c4_true <- c4_whole_tumor_flair_ce_volume_ratio`
   - `c5_true <- c5_h12_adc_10percentile`
   - `c6_true <- c6_h1_cbf_95percentile`
   - `c7_true <- c7_h1_volume_ratio`
   - `c8_true <- c8_h2_volume_ratio`
3. 检查 NaN、Inf、重复 patient、缺失 split。
4. 只用 train split 拟合 mean/std。
5. 对 std 过小的概念做保护，避免除零。
6. 保留 raw concept，并可派生 standardized concept。

建议 `concept_labels.csv` 字段：

```text
patient_id,split,y_true,label_name,
c1_true,c2_true,c3_true,c4_true,c5_true,c6_true,c7_true,c8_true
```

建议 `concept_scaler_stats.json` 结构：

```json
{
  "fit_split": "train",
  "concepts": {
    "c1": {"mean": 0.0, "std": 1.0, "source": "c1_h1_t1ce_firstorder_mean"},
    "c2": {"mean": 0.0, "std": 1.0, "source": "c2_h23_t1ce_firstorder_mean"}
  }
}
```

### Step 2: Add HabitatCBMBlockDataset

新增脚本：`srcs/habitat_cbm_data.py`

推荐接口：

```python
dataset = HabitatCBMBlockDataset(
    base_dataset=HabitatIDHBlockDataset(...),
    concept_label_csv=...,
    concept_scaler_json=...,
)
```

`__getitem__` 返回：

```text
image
label
concept_true_raw      # [8]
concept_true_std      # [8]
patient_id
slice_index
paths
```

必须检查：

1. 每个 patient_id 都能在 concept label 表中找到。
2. base dataset 的 `label` 与 concept 表 `y_true` 一致。
3. `concept_true_raw` 和 `concept_true_std` 都是 shape `[8]`。
4. patient_id 统一为字符串，保留前导零。

### Step 3: Expand Training Pipeline

扩展 `srcs/train_habitat_CBM.py` 或新增主入口脚本调用它。

推荐保留当前工具函数，并新增：

1. `build_argparser()` 的真实训练参数。
2. `build_datasets_and_loaders(args)`。
3. `train_one_epoch_stage1(...)`。
4. `train_one_epoch_stage2(...)`。
5. `train_one_epoch_stage3(...)`。
6. `evaluate_for_model_selection(...)`。
7. `save_checkpoint(...)`。
8. `run_train_pipeline(args)`。

三阶段策略：

1. Stage 1:
   - `set_train_stage(model, "stage1")`
   - 输入 image。
   - 优化 `encoder + concept_head`。
   - loss 为 `MSE(c_hat, c_true_std)`。
   - 监控 val concept loss。
   - 保存 `stage1_best.pt`。

2. Stage 2:
   - `set_train_stage(model, "stage2")`
   - 输入 `c_true_std`，不走 image。
   - 优化 `label_head`。
   - loss 为 weighted BCEWithLogits。
   - 监控 val AUC 或 val label loss。
   - 保存 `stage2_best.pt`。

3. Stage 3:
   - `set_train_stage(model, "stage3")`
   - 输入 image。
   - 优化全模型。
   - loss 为 `lambda_c * MSE + lambda_y * weighted BCEWithLogits`。
   - 监控 patient-level val AUC，AUC 不可用时退化为 val loss。
   - 保存 `stage3_best.pt`。

### Step 4: Export Patient-Level Main Task Results

在训练结束后，用 `stage3_best.pt` 对 `train/val/test` 推理。

输出：

1. `patient_predictions_habitat_cbm_runXX.csv`
2. `metrics_habitat_cbm_runXX.csv`
3. `roc_points_habitat_cbm_runXX.csv`
4. `confusion_matrix_habitat_cbm_runXX.csv`
5. `wrong_cases_habitat_cbm_runXX.csv`
6. `run_summary_habitat_cbm_runXX.json`

患者级分类聚合：

1. 对每个 block 输出 `prob_idh_mut = sigmoid(y_logit)`。
2. 同一患者内默认对全部 block 求均值。
3. 如使用 top-k pooling，必须在 config 中记录。
4. 默认阈值为 0.5；如果做阈值搜索，必须只用 val split 搜索，test 只报告应用后的结果。

### Step 5: Export Patient-Level Concept Predictions

推理时保存 block-level concept，再按患者聚合。

输出：

1. `patient_concepts_habitat_cbm_runXX.csv`

建议字段：

```text
patient_id,split,y_true,run_id,checkpoint_name,
c1_true,c1_pred,c1_abs_error,
c2_true,c2_pred,c2_abs_error,
...
c8_true,c8_pred,c8_abs_error
```

推荐口径：

1. 模型输出 `c_hat` 是 standardized scale。
2. 患者级 `c_pred_std = mean(c_hat_block)`。
3. 使用 scaler inverse transform 得到 `c_pred_raw`。
4. 论文表格优先报告 raw scale 指标。
5. 如果报告 standardized scale，必须在表名和说明中明确。

### Step 6: Evaluate Concepts

新增脚本：`srcs/eval_habitat_cbm_concepts.py`

输入：

1. `patient_concepts_habitat_cbm_runXX.csv`

输出：

1. `concept_metrics_by_concept.csv`
2. `concept_metrics_overall.csv`
3. `concept_error_distribution.csv`
4. `concept_error_rank.csv`

指标：

1. MAE
2. RMSE
3. R2
4. Pearson r

注意：

1. Pearson r 在样本数不足或概念真值方差为 0 时应返回 NaN，并记录原因。
2. R2 在概念真值方差为 0 时应返回 NaN。
3. 不要把 block 级样本当成独立患者计算概念指标。

### Step 7: Implement Patient-Level Intervention

新增脚本：`srcs/intervene_habitat_cbm.py`，或扩展 `eval_habitat_CBM.py` 的完整 CLI。

输入：

1. `stage3_best.pt`
2. `patient_concepts_habitat_cbm_runXX.csv`
3. `patient_predictions_habitat_cbm_runXX.csv`
4. `concept_scaler_stats.json`

预算：

```text
k = 1, 2, 4, all
```

流程：

1. 选候选病例：误判病例 + 低置信度病例。
2. 对每个患者计算 `abs(c_pred_std - c_true_std)`。
3. 按误差从大到小排序概念。
4. 替换前 k 个概念：`c_after_std = replace(c_pred_std, c_true_std, selected_indices)`。
5. 使用 `model.forward_c_to_y(c_after_std)` 得到 `y_logit_after`。
6. 比较 `y_before` 与 `y_after`。
7. 统计 corrected 标记、性能变化和纠正率。

输出：

1. `intervention_case_list.csv`
2. `intervention_per_case.csv`
3. `intervention_metrics_by_budget.csv`
4. `intervention_correction_summary.csv`
5. `intervention_summary.csv`

### Step 8: Keep Run Artifacts Traceable

每个正式 run 必须有：

1. `run_id`
2. config yaml/json
3. checkpoint 文件名
4. train/val/test split 信息
5. concept scaler 文件路径
6. git commit 或代码快照信息，如可用
7. 训练日志
8. 评估输出

推荐目录：

```text
runs/03_habitat_cbm/runXX/
  config_habitat_cbm_runXX.yaml
  stage1_concept_log.csv
  stage2_label_head_log.csv
  stage3_joint_log.csv
  checkpoints/
    stage1_best.pt
    stage2_best.pt
    stage3_best.pt
  patient_predictions_habitat_cbm_runXX.csv
  patient_concepts_habitat_cbm_runXX.csv
  metrics_habitat_cbm_runXX.csv
  roc_points_habitat_cbm_runXX.csv
  confusion_matrix_habitat_cbm_runXX.csv
  wrong_cases_habitat_cbm_runXX.csv
```

## Implementation Guardrails

### Data Leakage

1. 患者级划分一旦冻结，不允许训练中途重分。
2. scaler、阈值、类别权重只能基于 train split 拟合。
3. val split 用于模型选择和超参数选择。
4. test split 只能用于最终报告。
5. 不允许用 test 表现反向选择 checkpoint、阈值或 concept scaling。

### Hard Bottleneck

1. `y` 只能由 `c_hat` 或干预后的 `c_after` 推出。
2. 禁止新增 `encoder feature z -> y` 旁路。
3. 若做 ablation，可以单独命名为 no-bottleneck 或 hybrid，不得混入主模型。

### Concept Scale

1. 训练 loss 使用 standardized concept。
2. 论文概念指标建议用 raw scale。
3. 干预替换必须使用训练同口径的 standardized concept。
4. 保存 patient concept table 时同时保留 raw 和 standardized 口径会更安全。

### Patient-Level Reporting

1. 所有论文主表和主图以 patient 为统计单位。
2. block 级结果只能作为中间记录，不能作为最终报告单位。
3. patient-level concept prediction 应由同一患者 block-level `c_hat` 聚合得到。
4. patient-level intervention 应使用患者级概念向量。

### Compatibility With Existing Baselines

1. 不破坏 `baseline_ResNet18.py`。
2. 不破坏 `baseline_RadiomicsLR.py`。
3. 可复用 baseline 的工具函数，但要避免复制后产生口径漂移。
4. 任何复用的 threshold search、top-k pooling、class weights 都要在 Habitat-CBM config 中记录。

### Environment

1. 语法检查不等于深度学习运行通过。
2. 运行真实训练前必须进入包含 `torch` 的环境。
3. 自检至少包括：
   - import model
   - one batch forward
   - stage1 backward
   - stage2 backward
   - stage3 backward
   - checkpoint save/load
   - patient aggregation

## Acceptance Criteria

Habitat-CBM 优化完成必须满足以下条件。

### Training

1. 三阶段训练能完整运行。
2. 保存 `stage1_best.pt`、`stage2_best.pt`、`stage3_best.pt`。
3. 保存 `stage1_concept_log.csv`、`stage2_label_head_log.csv`、`stage3_joint_log.csv`。
4. Stage 1 能在 val concept loss 上选模。
5. Stage 2 能验证 oracle concept 到 label 的基本可用性。
6. Stage 3 能联合优化 concept loss 和 label loss。

### Main Task Evaluation

1. 产出 `patient_predictions_habitat_cbm_runXX.csv`。
2. 产出 `metrics_habitat_cbm_runXX.csv`。
3. 产出 ROC 原始点。
4. 产出混淆矩阵。
5. 产出 wrong cases。
6. 指标包含 `AUC / ACC / SEN / SPE / F1`。
7. 所有主任务指标均为患者级。

### Concept Evaluation

1. 产出 `patient_concepts_habitat_cbm_runXX.csv`。
2. 表中包含 `c1_true/c1_pred ... c8_true/c8_pred`。
3. 产出每个概念的 MAE、RMSE、R2、Pearson r。
4. 产出整体平均 concept metrics。
5. 产出 concept error distribution。
6. 产出 concept error rank。

### Intervention

1. 支持 `k=1,2,4,all`。
2. 干预在患者级概念向量上执行。
3. 产出 `intervention_per_case.csv`。
4. 产出 `intervention_metrics_by_budget.csv`。
5. 产出 `intervention_correction_summary.csv`。
6. 统计干预前后 `AUC / ACC / F1`。
7. 统计误判纠正率。
8. 能支持至少 2 个典型病例分析。

### Traceability

1. 每份结果包含 `run_id`。
2. 每份结果包含 split。
3. 每份患者级表包含 `patient_id`。
4. 每份预测表包含 checkpoint 名称。
5. 每次正式运行保存 config。
6. concept scaler 文件可追溯。

## Minimal Implementation Order

严格按以下顺序推进，避免先做干预或图表导致基础口径不稳：

1. 生成 `concept_labels.csv` 和 `concept_scaler_stats.json`。
2. 实现 `HabitatCBMBlockDataset`。
3. 跑一个 batch 的真实数据 smoke test。
4. 跑 Stage 1 debug。
5. 跑 Stage 2 debug。
6. 跑 Stage 3 debug。
7. 导出一次 debug 版 patient predictions 和 patient concepts。
8. 补齐主任务 metrics、ROC、confusion matrix。
9. 补齐 concept metrics。
10. 补齐 patient-level intervention。
11. 只在上述全部通过后跑正式 run。

## Common Failure Modes

### Patient ID mismatch

症状：

1. dataset 中 patient ID 找不到 concept label。
2. `001` 变成 `1`。

处理：

1. 所有 patient ID 都转成字符串。
2. 保留前导零。
3. 在 label 构建脚本和 dataset wrapper 中统一格式。

### Concept loss 被某一个概念主导

症状：

1. 总 concept loss 下降，但部分概念完全没学到。
2. 高量纲概念误差占据绝大多数 loss。

处理：

1. 确认训练用 `c_true_std`。
2. 检查 scaler 是否只用 train split。
3. 输出 per-concept MSE 监控。

### Stage 2 AUC 很低

症状：

1. 使用真实概念训练 label head 仍然不能预测 IDH。

处理：

1. 检查 concept label 和 y_true 是否错配。
2. 检查 concept 标准化是否错误。
3. 检查 label head 输入是否为 `c_true_std`。
4. 检查类别方向：`1` 是否稳定表示 mutant。

### Stage 3 后概念性能退化

症状：

1. 主任务 AUC 提升，但 concept MAE/RMSE 明显变差。
2. 干预结果不稳定。

处理：

1. 提高 `lambda_c`。
2. 降低 Stage 3 学习率。
3. 冻结部分 encoder 或缩短 Stage 3。
4. 同时监控 val AUC 和 val concept loss。

### Intervention 无提升

症状：

1. `k=all` 也几乎不能改变预测。

处理：

1. 检查 `label_head` 是否真的吃替换后的 concept。
2. 检查干预是否在 standardized scale 上执行。
3. 检查 patient-level 聚合口径。
4. 检查 Stage 2 是否学到了有效 `C -> Y` 映射。
5. 如果 oracle concept 本身预测力弱，应在论文中诚实报告概念集覆盖不足。

## What Not To Do

1. 不要把 `z` 拼接到 `c_hat` 后再分类，并仍称其为主 Habitat-CBM。
2. 不要用 test split 拟合 scaler、阈值或类别权重。
3. 不要用 block 级指标填论文患者级表格。
4. 不要只跑随机张量 self-check 就认为训练脚本完成。
5. 不要只保存 PNG 图而不保存 CSV 原始数据。
6. 不要跳过 concept evaluation 直接做 intervention。
7. 不要在干预时替换 raw concept 后直接送入使用 standardized concept 训练的 label head。

## Final Definition Of Done

当且仅当以下证据同时存在，才能认为 Habitat-CBM 主实验闭环完成：

1. `concept_labels.csv` 和 `concept_scaler_stats.json` 存在且通过 QC。
2. 三阶段训练日志和 checkpoint 存在。
3. 测试集患者级主任务指标存在。
4. 测试集患者级 concept prediction 表存在。
5. 8 个概念的 MAE、RMSE、R2、Pearson r 存在。
6. `k=1,2,4,all` 的 patient-level intervention 结果存在。
7. 误判纠正率存在。
8. 典型病例候选表存在。
9. 每个结果都能追溯到 run_id、checkpoint、split 和 patient_id。

---

## Implementation Status (2026-04-18)

以下状态基于当前仓库实现结果同步更新。

### 已实现

1. `srcs/build_habitat_cbm_labels.py`：已实现 concept proxy -> `concept_labels.csv` / `concept_statistics.csv` / `concept_scaler_stats.json`。
2. `srcs/data_loader_habitat_CBM.py`：已在 `HabitatIDHBlockDataset` 上直接扩展 concept 模式，返回 `concept_true_raw` 与 `concept_true_std`，并执行患者/标签一致性强校验。
3. `srcs/train_habitat_CBM.py`：已升级为完整三阶段训练主脚本（JSON 主配置 + CLI 覆盖 + 早停 + stage best checkpoint + stage 日志）。
4. Stage2/Stage3 类别不平衡：已按 train 患者级统计 `pos_weight` 接入 BCE。
5. `srcs/eval_habitat_CBM.py`：已升级为正式 checkpoint 评估脚本，输出患者级主任务表与患者级概念表。
6. `srcs/eval_habitat_cbm_concepts.py`：已实现概念层 `MAE/RMSE/R2/Pearson` 统计与偏差分布/排序导出。
7. `srcs/intervene_habitat_cbm.py`：已实现患者级 `k=1,2,4,all` 概念干预与候选集/全量双口径统计。
8. `srcs/args_train_habitat_CBM.json`：已填充为可运行默认模板，并固定为主配置源。

### 仍需在有 Torch 环境完成的运行验证

1. 三阶段真实训练 smoke test（stage1/2/3 各 1-2 epoch）。
2. checkpoint 加载与完整评估导出验证。
3. 干预预算全链路运行验证。

说明：当前仓库环境缺少 `torch`，本轮已完成语法级实现与静态检查，深度学习运行级验证需切换到具备 `torch` 的环境。
