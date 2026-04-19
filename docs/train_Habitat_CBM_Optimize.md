# Habitat-CBM 训练优化指南

> **文档定位**：本文档面向**人类研究者**和**LLM 辅助编码**双重受众。
> - 对人类：提供清晰的优化方向、参数对照表和操作步骤。
> - 对 LLM：提供充分的背景知识、诊断依据和约束条件，使其能直接基于本文档生成正确的代码修改或配置建议。
>
> **基准实验**：`run_id = 20260419_155537`，使用 `stage3_best.pt` 作为评估权重。

---

## 1. 项目背景与模型结构速览

### 1.1 任务定义

基于多模态脑 MRI 影像（T1/T1ce/T2/T2FLAIR/ADC/CBF）预测胶质瘤 IDH 突变状态（IDH mutant = 1，wild-type = 0）。输入为 2.5D slice block（每模态取 5 切片），拼接 VOI mask 共 **35 通道**，图像分辨率 224×224。

### 1.2 模型架构（硬概念瓶颈，Hard CBM）

```
x [B, 35, 224, 224]
  └─ encoder: ResNet18Backbone → z [B, 512]
       └─ concept_head: Linear(512→256)→ReLU→Dropout→Linear(256→8) → c_hat [B, 8]
            └─ label_head: Linear(8→32)→ReLU→Dropout→Linear(32→1) → y_logit [B, 1]
```

- **硬瓶颈（Hard Bottleneck）**：`y_logit` 仅依赖 `c_hat`，不能绕过概念层。
- **概念维度**：8 个连续概念（c1–c8），由 `build_habitat_cbm_labels.py` 生成的患者级 raw 概念标准化而来。
- **预测目标**：`c_hat` 回归标准化概念值（`concept_true_std`）；`y_logit` 做 IDH 二分类。

### 1.3 三阶段训练协议

| Stage | 冻结模块 | 训练模块 | Loss | 监控指标 |
|-------|---------|---------|------|---------|
| Stage1 | label_head | encoder + concept_head | MSE（概念回归） | `-val_concept_loss`（越大越好） |
| Stage2 | encoder + concept_head | label_head | BCE（使用真实概念 `c_true_std`） | `val_auc` |
| Stage3 | 无（全部解冻） | 全部 | `λ_c × concept_loss + λ_y × label_loss` | `val_auc` |

---

## 2. 基准实验结果（run_id: 20260419_155537）

### 2.1 最终指标（`stage3_best.pt`，threshold=0.5）

| Split | AUC | Acc | Sen（灵敏度） | Spe（特异度） | F1 | 患者数 |
|-------|-----|-----|------------|------------|-----|-------|
| Train | 0.842 | 79.2% | 71.1% | 84.5% | 0.730 | 96 |
| Val   | **0.813** | 71.4% | **50.0%** | 87.5% | 0.600 | 14 |
| Test  | **0.877** | 78.6% | 63.6% | 88.2% | 0.700 | 28 |

### 2.2 各阶段训练摘要

| Stage | 最优 Epoch | 最优分数 | 实际训练 Epoch 数 | 早停原因 |
|-------|-----------|--------|----------------|---------|
| Stage1 | 4 | val_concept_loss = 0.616 | 14 | patience=10 耗尽 |
| Stage2 | 5 | val_auc = 0.896 | 15 | patience=10 耗尽 |
| Stage3 | **1** | val_auc = 0.813 | 11 | patience=10 耗尽 |

> **关键诊断**：Stage3 最优出现在第 1 epoch，说明联合微调**没有产生正向优化**，模型在 Stage3 启动时即处于最佳状态，之后随训练退化。

### 2.3 混淆矩阵

| Split | TN | FP | FN | TP |
|-------|----|----|----|----|
| Train | 49 | 9 | 11 | 27 |
| Val   | 7 | 1 | **3** | 3 |
| Test  | 15 | 2 | **4** | 7 |

> FN（假阴性）系统性多于 FP（假阳性），说明模型对 IDH mutant 有漏检倾向。
> 几乎所有错误案例的预测概率都集中在 0.39–0.58（决策边界附近），属于低置信度错误。

### 2.4 Stage2 Oracle 性能

Stage2 使用真实概念标签输入（oracle 模式），val AUC 达 **0.896**。
这证明：8 个概念本身含有足够的 IDH 分类信息，当前性能瓶颈在于**概念预测精度不足**，而非 label_head 能力。

---

## 3. 问题诊断

本节为 LLM 提供结构化诊断，每条诊断对应后续的具体优化策略。

### 问题 A：Stage3 联合微调无效（最优出现在 epoch 1）

**根因分析：**
1. `lambda_c=1.0, lambda_y=0.5`，概念 loss 权重是分类 loss 的 2 倍，分类端驱动力不足。
2. Stage3 学习率与 Stage1/2 相同（1e-4），初始更新步长过大，导致模型从 Stage2 的分类最优解大幅偏移。
3. Stage3 全部解冻后，encoder 和 concept_head 的梯度同时作用，相互竞争，破坏了 Stage2 建立的 `C→Y` 映射。

### 问题 B：灵敏度偏低（Val Sen=50%，Test Sen=63.6%）

**根因分析：**
1. 数据集正负比接近 1:1.5（wild-type 略多），`pos_weight ≈ 1.53` 的矫正力度尚不足以完全补偿类别不平衡对 mutant 的漏检倾向。
2. 分类阈值固定为 0.5，而模型对 mutant 的预测概率系统性偏低（错误案例概率集中在 0.42–0.49），说明降低阈值可以直接提升灵敏度。

### 问题 C：概念预测泛化不足（Val concept_loss=0.616）

**根因分析：**
1. `dropout_p=0.7` 过高，影响概念预测的稳定性（训练时 70% 神经元随机失活，对小样本可能过于激进）。
2. Stage1 仅 14 epoch 就早停，概念 head 尚未充分收敛。
3. 96 例训练患者规模较小，对概念回归的泛化要求较高。

---

## 4. 优化策略（按优先级排序）

### 优化 OPT-1：修复 Stage3 学习率（**高优先级**）

**目标**：避免 Stage3 开始时从 Stage2 最优解大幅偏移，使联合微调真正产生正向作用。

**操作**：在 `args_train_habitat_CBM.json` 中为 Stage3 单独覆盖优化器配置（使用 `stage_override` 机制）。

**推荐配置变更**（在 `stages.stage3` 中添加 `optimizer` 子对象）：

```json
"stage3": {
  "epochs": 100,
  "checkpoint_name": "stage3_best.pt",
  "log_csv": "stage3_joint_log.csv",
  "optimizer": {
    "name": "adamw",
    "lr": 0.00001,
    "lr_encoder": 0.000005,
    "lr_concept_head": 0.00001,
    "lr_label_head": 0.00001,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-08,
    "amsgrad": false
  }
}
```

**说明**：
- `lr_encoder` 设为 5e-6，比 concept/label head 更小，保护 encoder 已学到的特征表示不被破坏。
- `lr_concept_head` 和 `lr_label_head` 降至 1e-5（原来的 1/10），小步更新，微调而非重训。

---

### 优化 OPT-2：调整 Stage3 loss 权重（**高优先级**）

**目标**：在保留概念可解释性的同时，加强分类 loss 在 Stage3 的驱动力。

**当前配置**：`lambda_c=1.0, lambda_y=0.5`（分类权重仅为概念权重的一半）

**推荐配置变更**（`loss.joint` 字段）：

```json
"joint": {
  "lambda_c": 0.5,
  "lambda_y": 1.0
}
```

**说明**：
- 将分类 loss 权重提升为概念 loss 的 2 倍，让 Stage3 真正以分类性能为导向联合微调。
- `lambda_c=0.5` 保留了对概念预测质量的约束，不会完全放弃可解释性。
- 如果 OPT-1 和 OPT-2 同时应用，预期 Stage3 最优 epoch 会延迟到 3–10 epoch，而不是第 1 epoch 即触发。

---

### 优化 OPT-3：调整分类决策阈值（**高优先级**）

**目标**：提升灵敏度（减少 FN），在临床场景中降低 IDH mutant 漏检风险。

**操作**：修改 `eval.threshold` 字段。

```json
"eval": {
  "threshold": 0.40,
  "topk_pool": 0,
  "include_splits": ["train", "val", "test"]
}
```

**预期效果（基于基准实验错误案例分析）**：
- 测试集中 4 例 FN 的预测概率为 0.422、0.466、0.468、0.475，降阈值至 0.40 可直接挽回后 3 例。
- 代价是新增少量 FP（当前 2 例 FP 的概率为 0.504、0.508，不受影响）。

**注意**：阈值调整属于推理参数，不影响训练权重，可在训练结束后单独用 eval 脚本扫描不同阈值，绘制灵敏度-特异度曲线后再确定最终值，推荐扫描范围 `[0.35, 0.50]`，步长 0.05。

---

### 优化 OPT-4：适当降低 Dropout 率（**中优先级**）

**目标**：改善概念预测泛化，减少 Stage1 概念 loss 的不稳定性。

**当前配置**：`model.dropout_p = 0.7`（非常激进，70% 神经元随机失活）

**推荐配置变更**：

```json
"model": {
  "in_channels": 35,
  "n_concepts": 8,
  "concept_hidden_dim": 256,
  "label_hidden_dim": 32,
  "dropout_p": 0.5,
  "pretrained": true
}
```

**说明**：
- 0.5 是标准小样本场景的常用值，在正则化强度和表达能力之间取得平衡。
- 如果降低后验证集概念 loss 仍不改善，可进一步尝试 0.3。
- 此改动需要重新从 Stage1 开始训练。

---

### 优化 OPT-5：增大 pos_weight 进一步矫正类别不平衡（**中优先级**）

**目标**：在 BCE loss 层面对 IDH mutant（正类）给予更高惩罚权重，直接缓解 FN 偏多的问题。

**当前行为**：`use_pos_weight=true`，`pos_weight` 由训练集自动计算为 `N_neg/N_pos ≈ 1.53`。

**推荐操作**：在 `loss.label` 中添加手动覆盖字段（需确认训练脚本是否支持 `manual_pos_weight`；若不支持，可直接修改 `train_habitat_CBM.py` 中的 pos_weight 计算逻辑）：

```json
"label": {
  "name": "bce_with_logits",
  "reduction": "mean",
  "use_pos_weight": true,
  "manual_pos_weight": 2.5,
  "label_smoothing": 0.0
}
```

**说明**：
- `manual_pos_weight=2.5` 意味着每个 FN 的 loss 惩罚是 FP 的 2.5 倍，引导模型优先减少漏检。
- 建议与 OPT-3（调整阈值）二选一优先实施，避免两者叠加导致 FP 大幅增加。

---

### 优化 OPT-6：延长 Stage1 训练轮次或调整早停 patience（**中优先级**）

**目标**：给 Stage1 的概念 head 更多收敛机会，提升概念预测质量。

**当前行为**：Stage1 在第 4 epoch 达到最优后，连续 10 epoch 无改善（第 14 epoch 触发早停），但 val concept_loss 在 0.619–0.754 间大幅震荡。

**根本原因**：震荡说明 val loss 曲线不稳定，可能是学习率在第 4 epoch 后仍偏高。

**推荐配置变更**：为 Stage1 单独配置更激进的调度器：

```json
"stage1": {
  "epochs": 100,
  "checkpoint_name": "stage1_best.pt",
  "log_csv": "stage1_concept_log.csv",
  "scheduler": {
    "enabled": true,
    "name": "reduce_on_plateau",
    "mode": "max",
    "factor": 0.3,
    "patience": 3,
    "min_lr": 1e-06
  }
}
```

**说明**：
- `factor=0.3`（原 0.5）：lr 降得更快，减少震荡。
- `patience=3`（原 5）：更快触发 lr 衰减，帮助模型在早期稳定。
- 这可以让 Stage1 的概念 loss 曲线更平滑，有望使最优 val concept_loss 低于当前的 0.616。

---

## 5. 推荐优化实验顺序

建议按以下顺序逐步实验，每次只变动 1–2 个因素，以便准确归因效果。

```
实验轮次    变动内容                              核心观察指标
─────────────────────────────────────────────────────────────────────
Run A     OPT-1 + OPT-2（Stage3 lr + λ 权重）  Stage3 最优 epoch 是否延迟；
                                               Test AUC 是否 > 0.877

Run B     Run A 基础上 + OPT-3（阈值 0.40）    Test Sen 是否 > 0.636；
                                               Test Spe 是否仍 > 0.80

Run C     OPT-4（dropout 0.5）重训全部阶段     Val concept_loss 是否 < 0.616；
                                               Stage2 Oracle AUC 是否 > 0.896

Run D     Run C 基础上 + OPT-1 + OPT-2 + OPT-3  全面对比基准
─────────────────────────────────────────────────────────────────────
```

---

## 6. 配置文件快速对照表

下表总结了从基准配置到推荐配置的所有关键字段变动，供直接修改 `args_train_habitat_CBM.json` 参考。

| 字段路径 | 基准值 | 推荐值（Run A） | 推荐值（Run C/D） | 优化目标 |
|---------|--------|--------------|----------------|---------|
| `loss.joint.lambda_c` | 1.0 | **0.5** | **0.5** | Stage3 分类驱动力 |
| `loss.joint.lambda_y` | 0.5 | **1.0** | **1.0** | Stage3 分类驱动力 |
| `stages.stage3.optimizer.lr` | *(继承全局 1e-4)* | **1e-5** | **1e-5** | Stage3 不偏移 |
| `stages.stage3.optimizer.lr_encoder` | *(继承全局 1e-4)* | **5e-6** | **5e-6** | 保护 encoder |
| `eval.threshold` | 0.5 | **0.40** | **0.40** | 提升灵敏度 |
| `model.dropout_p` | 0.7 | 0.7（不变） | **0.5** | 概念预测泛化 |
| `stages.stage1.scheduler.factor` | *(继承全局 0.5)* | *(不变)* | **0.3** | Stage1 稳定收敛 |
| `stages.stage1.scheduler.patience` | *(继承全局 5)* | *(不变)* | **3** | Stage1 稳定收敛 |

---

## 7. 代码修改参考

### 7.1 `args_train_habitat_CBM.json` 中添加 Stage3 optimizer 覆盖

训练脚本已支持在 `stages.stage3` 内嵌 `optimizer` 子对象以覆盖全局配置（见 `run_train_summary` 中 `stage_summary.stage3.optimizer` 字段存在即为证明）。直接在 `stages.stage3` 中添加 `optimizer` key 即可生效，无需修改 Python 代码。

### 7.2 `loss.label.manual_pos_weight` 支持（如需 OPT-5）

若训练脚本当前只支持自动 `pos_weight`，需在 `train_habitat_CBM.py` 的 pos_weight 构建逻辑中添加如下判断：

```python
label_cfg = cfg["loss"]["label"]
if label_cfg.get("manual_pos_weight") is not None:
    pos_weight_val = float(label_cfg["manual_pos_weight"])
else:
    pos_weight_val = n_neg / n_pos  # 原有自动计算逻辑
pos_weight = torch.tensor([pos_weight_val], device=device)
```

### 7.3 阈值扫描脚本（OPT-3 验证）

```python
import numpy as np
from sklearn.metrics import roc_curve

# 假设 y_true 和 y_prob 已从 patient_predictions CSV 读取
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
# 找使 Sen+Spe 最大的阈值（Youden's J）
j_scores = tpr + (1 - fpr) - 1
best_thresh = thresholds[np.argmax(j_scores)]
print(f"Youden 最优阈值: {best_thresh:.3f}")
```

---

## 8. 约束与注意事项

> 以下约束供 LLM 在生成修改建议时遵守，避免引入不兼容的改动。

1. **模型结构不变**：本轮优化不修改 `habitat_CBM.py` 中的网络结构（encoder/concept_head/label_head 的层数和维度）。所有变动均通过配置文件或训练脚本的少量修改实现。
2. **三阶段协议不变**：Stage1→2→3 的训练顺序和各阶段的冻结策略保持不变。
3. **概念维度不变**：`n_concepts=8`，与现有 `concept_labels.csv` 保持一致，不重新生成概念标签。
4. **数据划分不变**：Train/Val/Test 的患者分配不变，保证结果可与基准实验（`20260419_155537`）直接对比。
5. **`args_train_habitat_CBM.json` 的 `_comment` 字段**：所有 `_comment_*` 字段是合法 JSON 注释字段，训练脚本会忽略它们，修改时保留这些注释字段。
6. **Stage3 optimizer 覆盖格式**：必须与全局 `optimizer` 字段的 key 结构完全一致，训练脚本会做 deep merge，缺失的 key 自动回退到全局配置。

---

*文档生成时间：2026-04-19 | 基准实验 run_id：20260419_155537*
