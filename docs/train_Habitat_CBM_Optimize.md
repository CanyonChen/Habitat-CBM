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
| Stage1 | label_head | encoder + concept_head | MSE/SmoothL1（概念回归） | `-val_concept_loss`（越大越好） |
| Stage2 | encoder + concept_head | label_head | BCE（使用真实概念 `c_true_std`） | `val_auc` |
| Stage3 | 浅层 encoder（可配置） | layer4 + concept_head + label_head | `λ_c × concept_loss + λ_y × label_loss` | `val_auc` |

> **Stage3 冻结说明**：Stage3 默认冻结 `conv1/bn1/layer1/layer2/layer3`，与 Stage1 行为对称，防止联合微调时浅层特征大幅偏移。可通过 `stages.stage3.freeze_encoder_layers` 参数化控制。

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

## 2.5 已落地优化汇总（代码对齐，截至 2026-04-19）

以下条目已在当前代码中实现，通过默认配置直接生效：

| 编号 | 优化项 | 状态 | 当前默认值 | 代码入口 |
|------|--------|------|-----------|---------|
| OPT-1 | Stage3 学习率修复 | ✅ 已实现 | `stages.stage3.optimizer`: `lr=5e-6`，`lr_encoder=5e-6`，`lr_concept_head=1e-5`，`lr_label_head=5e-6` | `args_train_habitat_CBM.json` |
| OPT-2 | Stage3 loss 权重调整 | ✅ 已实现 | `loss.joint`: `lambda_c=0.5`，`lambda_y=1.0` | `args_train_habitat_CBM.json` |
| OPT-3 | 阈值：val Youden Index 动态选取 | ✅ 已实现 | 训练期监控用 `eval.threshold=0.5`；最终评估用 val 集 Youden Index 动态阈值 | `train_habitat_CBM.py` |
| OPT-5 | pos_weight 自动计算（不再手动固定） | ✅ 已实现 | `manual_pos_weight=null`，自动计算 `N_neg/N_pos ≈ 1.53` | `args_train_habitat_CBM.json` |
| OPT-6 | Stage1 调度器优化 | ✅ 已实现 | `stages.stage1.scheduler`: `factor=0.3`，`patience=3` | `args_train_habitat_CBM.json` |
| OPT-7 | 放宽早停 | ✅ 已实现 | `early_stop_patience=20`，`early_stop_min_delta=0.002` | `args_train_habitat_CBM.json` |
| OPT-8 | 患者均衡采样 | ✅ 已实现 | `train.patient_balanced_sampling=true`，使用 `WeightedRandomSampler` | `data_loader_habitat_CBM.py` |
| OPT-9 | Stage2 患者级 + 概念噪声（A+B） | ✅ 已实现 | `stages.stage2.patient_level=true`，`concept_noise_std=0.15` | `data_loader_habitat_CBM.py` + `train_habitat_CBM.py` |
| OPT-11 | 更鲁棒概念 loss | ✅ 已实现 | `loss.concept.name=smooth_l1`，`beta=0.5` | `args_train_habitat_CBM.json` |
| NEW-1 | Stage3 encoder 冻结层参数化 | ✅ 已实现 | `stages.stage3.freeze_encoder_layers=["conv1","bn1","layer1","layer2","layer3"]` | `args_train_habitat_CBM.json` + `train_habitat_CBM.py` |
| NEW-2 | PNG 结果图导出 | ✅ 已实现 | `eval.export_png=true`，`figure_dpi=300` | `eval_habitat_CBM.py` |

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
1. 数据集正负比接近 1:1.5（wild-type 略多），自动 `pos_weight ≈ 1.53` 的矫正力度可能尚不足以完全补偿类别不平衡对 mutant 的漏检倾向。
2. 固定阈值 0.5 偏高，模型对 mutant 的预测概率系统性偏低（错误案例概率集中在 0.42–0.49）。**当前已改为 val 集 Youden Index 动态选取阈值**，最终评估阈值随数据自适应调整。

### 问题 C：概念预测泛化不足（Val concept_loss=0.616）

**根因分析：**
1. `dropout_p=0.7` 过高，影响概念预测的稳定性（训练时 70% 神经元随机失活，对小样本可能过于激进）。
2. Stage1 仅 14 epoch 就早停，概念 head 尚未充分收敛。
3. 96 例训练患者规模较小，对概念回归的泛化要求较高。

### 问题 D：block 级训练与患者级标签存在权重错配

**根因分析：**
1. 当前 Dataset 将每位患者展开成多个 2.5D block，但 IDH 标签和 8 维概念标签都是患者级标签，会复制到该患者的所有 block。
2. `pos_weight` 按训练集患者数计算，但 BCE 实际按 block 聚合。若不同类别或不同患者的有效 block 数不均衡，患者级 `pos_weight` 不能完全修正 block 级训练偏置。
3. Stage2 用真实概念训练 label_head 时，理论上只有 96 个唯一训练点，但脚本会按 block 重复同一患者概念，等价于按患者 block 数给 `C→Y` 样本加权，容易让 label_head 过快过拟合。

### 问题 E：Stage2 与 Stage3 存在概念分布偏移

**根因分析：**
1. Stage2 的 label_head 只见过真实标准化概念 `c_true_std`，而 Stage3/推理时 label_head 接收的是 `c_hat`。
2. 由于 Stage1 的概念预测仍有明显误差，`c_hat` 的分布、方差和概念间相关结构不一定与 `c_true_std` 一致。
3. 这会导致 Stage2 的高 oracle AUC 无法完整转化为 Stage3 性能，即"真实概念可分，但预测概念输入下分类头不稳"。

### 问题 F：当前增强策略可能对概念回归过强

**根因分析：**
1. 当前增强组合较强：`affine_prob=0.9`、旋转 30°、平移 20 px、缩放 25%，并叠加强度扰动、高斯噪声和 Gibbs 噪声。
2. Habitat 概念来自强度、形状、体积比例等 radiomics proxy，过强的空间或强度扰动可能改变这些概念对应的可见影像证据，但标签仍保持不变。
3. 对 Stage1 的 `X→C` 概念回归而言，过强增强可能不是正则化，而是引入 label-preserving 假设不成立的噪声。

### 问题 G：早停与学习率调度节奏偏紧

**根因分析：**
1. 全局 `early_stop_patience=10`，而 ReduceLROnPlateau 默认 `patience=5`。如果前 5 个 epoch 无改进，学习率刚降低，通常只剩约 5 个 epoch 证明低学习率是否有效。
2. Val 只有 14 位患者，AUC 的离散粒度较粗，单个患者预测变化就会明显改变曲线；用单次 split 的 `val_auc` 早停很容易把随机波动当成最优点。
3. Stage3 最优在 epoch 1 不一定只代表"训练方向错误"，也可能包含"小验证集 + 高方差 AUC + 早停节奏过紧"的模型选择噪声。

---

## 4. 优化策略（按优先级排序）

### 优化 OPT-1：修复 Stage3 学习率（**高优先级**）

**目标**：避免 Stage3 开始时从 Stage2 最优解大幅偏移，使联合微调真正产生正向作用。

**实现状态**：✅ **已实现**。默认配置已写入 `stages.stage3.optimizer`，并通过 stage override 生效。

**当前配置**（`stages.stage3.optimizer`）：

```json
"stage3": {
  "optimizer": {
    "name": "adamw",
    "lr": 0.000005,
    "lr_encoder": 0.000005,
    "lr_concept_head": 0.00001,
    "lr_label_head": 0.000005,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-08,
    "amsgrad": false
  }
}
```

**说明**：
- `lr_encoder` 设为 5e-6，比 concept/label head 更小，保护 encoder 已学到的特征表示不被破坏。
- `lr_concept_head` 降至 1e-5，`lr_label_head` 降至 5e-6，小步更新，微调而非重训。

---

### 优化 OPT-2：调整 Stage3 loss 权重（**高优先级**）

**目标**：在保留概念可解释性的同时，加强分类 loss 在 Stage3 的驱动力。

**实现状态**：✅ **已实现**。

**当前配置**（`loss.joint`）：

```json
"joint": {
  "lambda_c": 0.5,
  "lambda_y": 1.0
}
```

**说明**：将分类 loss 权重提升为概念 loss 的 2 倍，让 Stage3 以分类性能为导向联合微调。`lambda_c=0.5` 保留了对概念预测质量的约束，不会完全放弃可解释性。

---

### 优化 OPT-3：阈值——val 集 Youden Index 动态选取（**高优先级**）

**目标**：移除人工固定阈值，自动在 val 集上选取使 Sensitivity + Specificity 最大的阈值，避免固定值掩盖模型漏检倾向。

**实现状态**：✅ **已实现**。

**机制**（`train_habitat_CBM.py`）：
1. Stage3 最优权重加载后，用 `collect_val_patient_probs()` 在 val 集收集患者级预测概率。
2. `find_youden_threshold()` 调用 `sklearn.metrics.roc_curve`，取使 `tpr - fpr`（Youden J）最大的阈值。
3. `run_full_evaluation` 使用该动态阈值 `eval_threshold`；`run_summary.json` 同时记录 `fixed_threshold`、`youden_threshold` 和 `eval_threshold_used`，便于溯源。

**训练期监控**（`eval.threshold`）：

```json
"eval": {
  "threshold": 0.5
}
```

训练期 val 监控使用中性阈值 0.5，仅用于每 epoch 输出 sen/spe/acc/f1 等中间指标，**不影响最终评估**。

**注意**：阈值选择必须基于 val split，不能基于 test FN 反向选择。本实现严格遵守此约束。

---

### 优化 OPT-4：适当降低 Dropout 率（**中优先级**）

**目标**：改善概念预测泛化，减少 Stage1 概念 loss 的不稳定性。

**当前配置**：`model.dropout_p = 0.4`（已从基准 0.7 降低）

**说明**：
- 0.4–0.5 是标准小样本场景的常用值，在正则化强度和表达能力之间取得平衡。
- 如果降低后验证集概念 loss 仍不改善，可进一步尝试 0.3。
- 此改动需要重新从 Stage1 开始训练。

---

### 优化 OPT-5：pos_weight 自动计算（**中优先级**）

**目标**：在 BCE loss 层面根据训练集实际分布自动矫正类别不平衡，不再依赖人工固定值。

**实现状态**：✅ **已实现**。

**当前行为**：`use_pos_weight=true`，`manual_pos_weight=null`，训练脚本自动计算 `N_neg / N_pos ≈ 1.53`。

**当前配置**（`loss.label`）：

```json
"label": {
  "name": "bce_with_logits",
  "reduction": "mean",
  "use_pos_weight": true,
  "manual_pos_weight": null,
  "label_smoothing": 0.0
}
```

**说明**：
- 不再人为放大 `pos_weight=2.5`，避免过度强调正类导致 Specificity 下降。
- 若确有需要提升灵敏度，应先通过 Youden 阈值调整（OPT-3 已实现）、OPT-8 均衡采样等更靠前的手段，再考虑调高 `manual_pos_weight`。
- 如需手动覆盖，将 `manual_pos_weight` 改为正数（如 `2.0`）即可，优先级高于自动计算。

---

### 优化 OPT-6：降低 Stage1 初始学习率 + 更激进调度（**高优先级**）

**目标**：消除 Stage1 val concept_loss 的大幅震荡，使概念预测稳定收敛。

**实现状态**：✅ **已实现**。

#### Val loss 震荡根因分析

实测观察到 Stage1 val_concept_loss 在各 epoch 间反复横跳（例如 0.376 → 0.433 → 0.387 → 0.416），没有持续下降趋势。这是以下几个因素的组合效应：

1. **验证集仅 14 例患者**：val loss 估计标准误 ≈ σ/√14，信号极其粗糙，单个患者概念预测的随机波动就能显著改变 val loss。
2. **初始学习率 1e-4 偏高**：每次参数更新步长足以让某几例患者的预测从"接近"跳到"偏差大"，导致 epoch 间 val loss 大幅跳动。
3. **ReduceLROnPlateau 被"假新低"重置**：每隔数轮随机出现一个偶发低点重置了 patience 计数器，使 lr 衰减迟迟不触发，震荡持续。
4. **aug_affine_prob=0.9 增强极激进**：训练集梯度方向每 epoch 都有较大随机性，进一步放大 val 上的震荡。

> **关键判断**：这不是模型在退步，而是**评估信号本身不稳定 + 学习率过高的组合**，导致模型无法稳定收敛。增加 epoch 上限无效，必须降低 lr 步长。

**当前配置**（`stages.stage1`）：

```json
"stage1": {
  "optimizer": {
    "name": "adamw",
    "lr": 0.00003,
    "lr_encoder": 0.00003,
    "lr_concept_head": 0.00003,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-08,
    "amsgrad": false
  },
  "scheduler": {
    "enabled": true,
    "name": "reduce_on_plateau",
    "mode": "min",
    "factor": 0.3,
    "patience": 3,
    "min_lr": 1e-06
  }
}
```

**说明**：
- **初始 lr 从 1e-4 降至 3e-5**（降低 3 倍）：直接减小每次更新步长，消除因步长过大导致的 val loss 跳动。
- `factor=0.3`（原 0.5）：触发衰减时 lr 降得更快（3e-5 → 9e-6 → 2.7e-6），越早进入细粒度搜索。
- `patience=3`（原 5）：减少假新低对 patience 计数器的重置窗口，更快触发 lr 衰减。
- 预期效果：val concept_loss 曲线从震荡形态变为单调下降或平台形态，最优 val concept_loss 有望低于基准的 0.616。

---

### 优化 OPT-7：放宽早停（**中优先级**）

**目标**：解决"实际训练 epoch 过少"的问题，避免学习率刚下降就触发早停。

**实现状态**：✅ **已实现**。

**当前配置**（`train`）：

```json
"train": {
  "early_stop_patience": 20,
  "early_stop_min_delta": 0.002
}
```

**说明**：
- `early_stop_patience` 建议至少大于 `scheduler.patience * 3`，让 ReduceLROnPlateau 降低学习率后仍有 8–12 个 epoch 的恢复窗口。
- `early_stop_min_delta=0.002` 可过滤极小浮动；对 Val AUC 这种小验证集离散指标，不建议设得过大。
- **早停机制**：判断条件为 `(current_score - best_score) > early_stop_min_delta`（严格大于）。`min_delta=0.0` 时只要有严格正提升即算改善，完全相等不算。`patience=0` 则直接禁用早停，训练跑满 `epochs` 轮。

---

### 优化 OPT-8：引入患者均衡的 block 采样（**高优先级**）

**目标**：让每位患者在每个 epoch 中贡献近似相同的梯度权重，避免"block 数多的患者支配训练"。

**实现状态**：✅ **已实现**。

**当前行为**：
- `HabitatIDHBlockDataset.sample_index` 是 block 级样本列表。
- `build_habitat_cbm_dataloaders()` 在 `patient_balanced_sampling=true` 时使用 `WeightedRandomSampler`。
- 采样权重：`weight(block_i) = 1 / (num_blocks_of_patient[pid] × num_patients_of_class[label])`。

**配置**（`train`）：

```json
"train": {
  "patient_balanced_sampling": true
}
```

**说明**：
- 第一项 `1 / num_blocks_of_patient` 保证每位患者总采样权重相近。
- 第二项 `1 / num_patients_of_class` 同时做患者级类别均衡，效果类似但比 `pos_weight` 作用更底层。

---

### 优化 OPT-9：将 Stage2 改为患者级训练，并加入概念噪声鲁棒性（**高优先级**）

**目标**：缓解 Stage2 oracle 到 Stage3 predicted concept 的分布偏移。

**实现状态**：✅ **已实现 A+B**。已接入 `PatientConceptDataset`（患者级 Stage2）与训练期概念噪声 `concept_noise_std`；未实现 Bridge Stage（C）。

**当前配置**（`stages.stage2`）：

```json
"stage2": {
  "patient_level": true,
  "concept_noise_std": 0.15
}
```

**方案说明**：
- **方案 A（患者级去重）**：Stage2 每位患者只出现一次，避免按 block 数加权的 `C→Y` 过拟合。
- **方案 B（概念噪声）**：训练期注入高斯噪声 `σ=0.15`，让 label_head 在噪声概念下保持鲁棒，缩小 Stage2 oracle 与 Stage3 `c_hat` 之间的分布差距；验证期不注入噪声。
- **方案 C（Bridge Stage，未实现）**：冻结 encoder+concept_head，用 `c_hat.detach()` 单独训练 label_head，让其适配 Stage1 的概念预测误差分布后再进入 Stage3。

---

### 优化 OPT-10：减弱增强策略，优先保护概念标签一致性（**中优先级**）

**目标**：降低增强对 radiomics proxy 概念的破坏，先让 `X→C` 学得稳定。

**推荐配置变更**：

```json
"train": {
  "use_monai_augmentation": true,
  "aug_affine_prob": 0.5,
  "aug_rotate_deg": 10.0,
  "aug_translate_px": 8.0,
  "aug_scale_range": 0.10,
  "aug_flip_prob": 0.5,
  "aug_intensity_scale_prob": 0.3,
  "aug_intensity_scale": 0.15,
  "aug_intensity_shift_prob": 0.3,
  "aug_intensity_shift": 0.15,
  "aug_gaussian_noise_prob": 0.2,
  "aug_gaussian_noise_std": 0.02,
  "aug_gibbs_noise_prob": 0.0,
  "aug_gibbs_noise_alpha": 0.0
}
```

**说明**：
- 对概念回归，强增强不一定等价于更好泛化。尤其是形状、体积、分区比例类概念，对空间变换更敏感。
- 建议先做一轮"弱增强"对照：若 Stage1 val concept_loss 明显下降，再逐步恢复强度增强。
- 当前脚本的增强是全阶段共享；若后续修改代码，建议支持 stage-specific augmentation：Stage1 弱增强，Stage2 无影像增强，Stage3 中等增强。

---

### 优化 OPT-11：使用更鲁棒的概念 loss（**中优先级**）

**目标**：避免少数噪声概念或异常值主导 Stage1/Stage3 的概念回归。

**实现状态**：✅ **已实现**。

**当前配置**（`loss.concept`）：

```json
"concept": {
  "name": "smooth_l1",
  "reduction": "mean",
  "beta": 0.5
}
```

**逐概念诊断建议**：
- 从 `patient_concepts_habitat_cbm_*.csv` 统计每个概念的 `abs_error_std` 均值、中位数和 90 分位。
- 计算每个 `c_i_true_std` 与 `y_true` 的单变量 AUC/Spearman 相关，以及 `c_i_pred_std` 与 `c_i_true_std` 的相关。
- 优先优化"对 IDH 有信息但预测误差大"的概念；对"标签信息弱且预测噪声大"的概念，不应让它在 joint loss 中占过高权重。

---

### 新增 NEW-1：Stage3 encoder 冻结层参数化（**已实现**）

**目标**：让 Stage3 的 encoder 冻结策略与 Stage1 对称，可通过配置灵活控制，防止联合微调时浅层特征大幅偏移。

**实现状态**：✅ **已实现**。

**当前配置**（`stages.stage3`）：

```json
"stage3": {
  "freeze_encoder_layers": ["conv1", "bn1", "layer1", "layer2", "layer3"]
}
```

**说明**：
- 可用层名与 Stage1 完全相同：`conv1`、`bn1`、`layer1`、`layer2`、`layer3`、`layer4`。
- 设为 `null` 或 `[]` 表示 Stage3 全量解冻（原始行为）。
- 默认值冻结浅层 5 层，只允许 `layer4` + `concept_head` + `label_head` 参与 Stage3 微调，避免浅层通用特征被破坏。
- 代码在 `train_habitat_CBM.py` 的通用函数 `_apply_encoder_freeze(model, freeze_layer_names, stage)` 中实现，Stage1 和 Stage3 均调用该函数，日志分别打印 `[stage1]` 和 `[stage3]` 前缀。

---

### 新增 NEW-2：PNG 结果图导出（**已实现**）

**目标**：在 CSV/JSON 指标基础上自动导出可直接用于实验记录和论文初稿的图像结果。

**当前实现**（`eval.export_png=true` 时生效）：
- 按 split（默认 `train/val/test`）导出：
  - ROC 曲线
  - PR 曲线
  - 混淆矩阵热力图
  - Calibration 曲线
  - 预测概率分布图
  - 概念绝对误差箱线图（std 尺度）
  - 概念 MAE 排名柱状图（std 尺度）
  - 概念 true-vs-pred 散点网格图（std 尺度）
- 图像目录：`results/.../<run_id>/figures/`
- `run_summary_habitat_cbm_<run_id>.json` 的 `files.figures` 会记录每张图路径。

**当前配置**（`eval`）：

```json
"eval": {
  "export_png": true,
  "figure_include_splits": ["train", "val", "test"],
  "figure_dpi": 300
}
```

---

### 优化 OPT-12：系统扫描患者级聚合与阈值（**中优先级**）

**目标**：区分"模型排序能力不足"和"患者级概率聚合/阈值不合适"。

**当前行为**：
- block 概率先按患者平均，再用 Youden Index 动态阈值判定（最终评估）。
- `topk_pool=0` 表示使用全部 block 平均；`topk_pool>0` 时选择离 0.5 最远的高置信 block。

**推荐扫描**：

```text
topk_pool: 0, 5, 10, 20
```

**说明**：
- 阈值已改为 Youden 动态选取（OPT-3），不再需要手动扫描阈值网格。
- `topk_pool` 可能提升信噪比，也可能放大少数过度自信的错误 block，需要和 Sen/Spe 一起看。
- 更稳健的长期方案是 logit 平均、按 VOI 面积加权平均，或 MIL/attention 聚合；这些需要评估脚本或模型结构支持。

---

### 优化 OPT-13：多 seed / 交叉验证确认优化是否真实有效（**中优先级**）

**目标**：降低 14 例 val split 带来的模型选择偶然性。

**推荐实验**：

```text
seed: 42, 3407, 2026
报告: mean ± std 的 Val/Test AUC、Sen、Spe、F1
```

**进一步建议**：
- 若训练成本允许，做患者级 stratified 5-fold cross validation。
- 最终只把 test 作为一次性外部评估；配置选择依据应来自 val 或 CV 均值。
- 对比优化是否有效时，不只看单次 Test AUC，还要看 Stage3 最优 epoch 是否后移、Val concept_loss 是否下降、FN 是否减少。

---

## 5. 推荐优化实验顺序

建议按以下顺序逐步实验，每次只变动 1–2 个因素，以便准确归因效果。

```
实验轮次    变动内容                              核心观察指标
─────────────────────────────────────────────────────────────────────
Run A     OPT-1 + OPT-2 + NEW-1（已默认生效）   Stage3 最优 epoch 是否延迟；
          Stage3 lr 降低 + λ 权重 + 浅层冻结     Val AUC/Val loss 是否改善

Run B     Run A 基础上 + OPT-7（已默认生效）     Stage3 是否能在低 lr 后继续改善；
          放宽早停                               Stage1/3 实际训练 epoch 是否增加

Run C     OPT-3 Youden 阈值（已默认生效）         Val Sen 是否提升；
                                               Spe 是否仍 > 0.80

Run D     OPT-4 + OPT-6 + OPT-10 重训全部阶段  Val concept_loss 是否 < 0.616；
          降 dropout + 弱增强                   Stage3 最优 epoch 是否后移

Run E     Run D 基础上 + OPT-11（已默认生效）    逐概念 abs_error_std 是否下降；
          SmoothL1                              Val AUC/Sen 是否稳定

Run F     OPT-8 均衡采样（已默认生效）           Train/Val loss gap 是否缩小；
                                               患者间预测是否更稳定

Run G     OPT-9（已默认生效 A+B）               Stage2 oracle 到 Stage3 的性能落差是否缩小；
          患者级 Stage2 + 概念噪声               Stage3 label_loss 是否下降

Run H     最优配置做 OPT-13（多 seed/CV）        Val/CV mean±std 是否优于基准；
                                               最终 test 是否同步提升
─────────────────────────────────────────────────────────────────────
```

**推荐执行原则**：
- Run A–C 的改动已全部内置为默认值，直接运行即可观察效果。
- 如果概念 loss 仍高，再做 Run D/E，集中优化 `X→C`。
- Run F/G 相关代码已落地，可直接做对照实验验证收益。

---

## 6. 配置文件快速对照表

下表总结了从基准配置到当前默认配置的所有关键字段，供核查 `args_train_habitat_CBM.json` 参考。

| 字段路径 | 基准值 | 当前默认值 | 状态 | 优化目标 |
|---------|--------|-----------|------|---------|
| `loss.joint.lambda_c` | 1.0 | **0.5** | ✅ 已生效 | Stage3 分类驱动力 |
| `loss.joint.lambda_y` | 0.5 | **1.0** | ✅ 已生效 | Stage3 分类驱动力 |
| `loss.concept.name` | mse | **smooth_l1** | ✅ 已生效 | 降低异常概念影响 |
| `loss.concept.beta` | *(无)* | **0.5** | ✅ 已生效 | SmoothL1 转折点 |
| `loss.label.manual_pos_weight` | 2.5 | **null（自动）** | ✅ 已生效 | 避免过度强调正类 |
| `stages.stage1.optimizer.lr` | *(继承全局 1e-4)* | **3e-5** | ✅ 已生效 | 消除 Stage1 震荡 |
| `stages.stage1.scheduler.factor` | *(继承全局 0.5)* | **0.3** | ✅ 已生效 | Stage1 快速降 lr |
| `stages.stage1.scheduler.patience` | *(继承全局 5)* | **3** | ✅ 已生效 | 减少假新低重置 |
| `stages.stage1.freeze_encoder_layers` | *(无)* | **["conv1","bn1","layer1","layer2","layer3"]** | ✅ 已生效 | 抑制浅层过拟合 |
| `stages.stage3.optimizer.lr` | *(继承全局 1e-4)* | **5e-6** | ✅ 已生效 | Stage3 不偏移 |
| `stages.stage3.optimizer.lr_encoder` | *(继承全局 1e-4)* | **5e-6** | ✅ 已生效 | 保护 encoder |
| `stages.stage3.optimizer.lr_concept_head` | *(继承全局 1e-4)* | **1e-5** | ✅ 已生效 | 小步更新概念层 |
| `stages.stage3.optimizer.lr_label_head` | *(继承全局 1e-4)* | **5e-6** | ✅ 已生效 | 小步更新分类头 |
| `stages.stage3.freeze_encoder_layers` | *(无，全解冻)* | **["conv1","bn1","layer1","layer2","layer3"]** | ✅ 已生效 | 防止浅层偏移 |
| `stages.stage2.patient_level` | *(不支持)* | **true** | ✅ 已生效 | Stage2 去 block 重复 |
| `stages.stage2.concept_noise_std` | *(不支持)* | **0.15** | ✅ 已生效 | label_head 适应 `c_hat` 误差 |
| `train.early_stop_patience` | 10 | **20** | ✅ 已生效 | 给 LR 衰减留出训练窗口 |
| `train.early_stop_min_delta` | 0.0 | **0.002** | ✅ 已生效 | 过滤小幅随机波动 |
| `train.patient_balanced_sampling` | *(不支持)* | **true** | ✅ 已生效 | 患者级梯度均衡 |
| `eval.threshold` | 0.4 | **0.5（训练期监控）** | ✅ 已生效 | 训练期中性阈值 |
| 最终评估阈值 | 固定 0.4 | **val 集 Youden Index 动态** | ✅ 已生效 | 自适应提升 Sen |
| `eval.topk_pool` | 0 | **0** | ⏳ 可扫描 | 优化患者级聚合 |
| `model.dropout_p` | 0.7 | **0.4** | ⏳ 建议验证 | 概念预测泛化 |
| `train.aug_rotate_deg` | 30.0 | 30.0（待调） | ⏳ 可尝试 10.0 | 减少空间标签噪声 |
| `train.aug_gibbs_noise_prob` | 0.3 | 0.3（待调） | ⏳ 可尝试关闭 | 先关闭强伪影增强 |

---

## 7. 代码实现对照（当前仓库）

### 7.1 Stage3 optimizer 覆盖机制

训练脚本已支持在 `stages.stage3` 内嵌 `optimizer` 子对象以覆盖全局配置。当前实现是 flat dict 合并：

```python
stage_optimizer_cfg = {
    **optimizer_cfg,
    **stage_cfg.get("optimizer", {}),
}
```

因此 `stages.stage3.optimizer` 中只需要写要覆盖的 key，未写字段会自动继承全局 `optimizer`。

### 7.2 `loss.label.manual_pos_weight` 支持（OPT-5）

```python
def _compute_pos_weight_from_train_patients(
    dataset: HabitatIDHBlockDataset,
    device: torch.device,
    label_loss_config: Mapping[str, object],
) -> Optional[torch.Tensor]:
    if not bool(label_loss_config.get("use_pos_weight", True)):
        return None
    if label_loss_config.get("manual_pos_weight") is not None:
        value = float(label_loss_config["manual_pos_weight"])
        return torch.tensor([value], dtype=torch.float32, device=device)
    # 自动计算：BCEWithLogits pos_weight = N_negative / N_positive
    ...
```

`manual_pos_weight=null` 时走自动计算分支，设为正数时优先使用手动值。

### 7.3 Youden Index 动态阈值（OPT-3）

```python
def find_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """在 val 集上用 Youden Index (tpr - fpr) 最大化选取最优阈值。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr  # Youden J = Sensitivity + Specificity - 1
    best_idx = int(np.argmax(j_scores))
    return float(np.clip(thresholds[best_idx], 1e-6, 1.0 - 1e-6))
```

在 Stage3 权重加载后调用，计算结果写入 `run_summary.json` 的 `threshold` 字段：
```json
"threshold": {
  "fixed_threshold": 0.5,
  "youden_threshold": <动态值>,
  "eval_threshold_used": <动态值>
}
```

### 7.4 Stage3 encoder 冻结（NEW-1）

```python
def _apply_encoder_freeze(
    model: HabitatCBM,
    freeze_layer_names: List[str],
    stage: str = "stage",
) -> None:
    """对 encoder 中指定层执行冻结，可用于任意训练阶段。"""
    if not freeze_layer_names:
        trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"[{stage}] encoder freeze: none (full fine-tune, trainable={trainable:,})")
        return
    frozen_params = 0
    for layer_name in freeze_layer_names:
        layer = getattr(model.encoder, layer_name, None)
        if layer is None:
            raise ValueError(f"freeze_encoder_layers: layer '{layer_name}' not found in encoder.")
        for p in layer.parameters():
            p.requires_grad = False
            frozen_params += p.numel()
    trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"[{stage}] encoder freeze: {freeze_layer_names} (frozen={frozen_params:,}, trainable={trainable:,})")
```

Stage1 和 Stage3 均调用此函数，参数来源均为各自 `stage_cfg.get("freeze_encoder_layers", None)`，结果记录在 `stage_summary[stage]["frozen_encoder_layers"]`。

### 7.5 患者均衡采样（OPT-8）

```python
def build_patient_balanced_sampler(dataset: HabitatIDHBlockDataset) -> WeightedRandomSampler:
    patient_block_counts = Counter(item.patient_id for item in dataset.sample_index)
    class_patient_counts = Counter(case.label_id for case in dataset.patient_cases.values())
    weights = []
    for item in dataset.sample_index:
        label = int(dataset.patient_cases[item.patient_id].label_id)
        w = 1.0 / (
            float(patient_block_counts[item.patient_id])
            * float(class_patient_counts[label])
        )
        weights.append(w)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
```

### 7.6 Stage2 患者级训练 + 概念噪声（OPT-9 A+B）

训练主循环当前实现：
- Stage1/3 使用 block loader；
- Stage2 在 `stages.stage2.patient_level=true` 时使用 patient loader；
- Stage2 训练期在 `concept_noise_std>0` 时注入高斯噪声，验证期不注入。

### 7.7 概念分布偏移诊断（OPT-9/OPT-11）

建议在正式评估后从 `patient_concepts_habitat_cbm_*.csv` 导出以下表：

```text
concept_id
mean_abs_error_std
median_abs_error_std
p90_abs_error_std
c_true_vs_y_auc
c_pred_vs_true_spearman
```

如果某个概念满足 `c_true_vs_y_auc` 高但 `c_pred_vs_true_spearman` 低，说明它是优先优化对象；如果二者都低，则它对当前 IDH 任务帮助有限，不应过度追求该概念的回归 loss。

---

## 8. 约束与注意事项

> 以下约束供 LLM 在生成修改建议时遵守，避免引入不兼容的改动。

1. **模型结构不变**：本轮优先优化不修改 `habitat_CBM.py` 中的网络结构（encoder/concept_head/label_head 的层数和维度）。所有高优先级变动优先通过配置文件或训练脚本的少量修改实现。
2. **三阶段协议不变**：Stage1→2→3 的训练顺序和各阶段的冻结策略保持不变。Stage3 的浅层冻结通过 `freeze_encoder_layers` 参数化控制，默认行为等价于冻结前 5 层。
3. **概念维度不变**：`n_concepts=8`，与现有 `concept_labels.csv` 保持一致，不重新生成概念标签。
4. **数据划分不变**：Train/Val/Test 的患者分配不变，保证结果可与基准实验（`20260419_155537`）直接对比。
5. **`args_train_habitat_CBM.json` 的 `_comment` 字段**：所有 `_comment_*` 字段是合法 JSON 注释字段，训练脚本会忽略它们，修改时保留这些注释字段。
6. **Stage optimizer 覆盖格式**：`stages.stage1/stage3.optimizer` 使用与全局 `optimizer` 相同的扁平 key；训练脚本会做浅层合并，缺失的 key 自动回退到全局配置。
7. **阈值选择原则（严格执行）**：Youden Index 阈值只基于 val split 计算，不参考 test 集任何信息；test 只用于最终一次性报告。
8. **区分配置优化和代码优化**：当前所有优化均已落地为默认配置或代码实现，后续重点是实验验证（多 seed、弱增强对照等）。

---

*文档更新时间：2026-04-19（覆盖 OPT-1/2/3/5/6/7/8/9/11 + NEW-1 Stage3 冻结参数化 + NEW-2 PNG 导出） | 基准实验 run_id：20260419_155537*
