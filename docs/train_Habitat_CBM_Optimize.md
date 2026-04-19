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

## 2.5 本轮已落地优化（代码对齐，2026-04-19）

以下条目已在当前代码中实现并可通过默认配置直接生效：

| 项目 | 状态 | 默认配置/行为 | 代码入口 |
|---|---|---|---|
| OPT-1 Stage3 学习率修复 | 已实现 | `stages.stage3.optimizer`: `lr=1e-5`, `lr_encoder=5e-6`, `lr_concept_head=1e-5`, `lr_label_head=1e-5` | `srcs/args_train_habitat_CBM.json` + `srcs/train_habitat_CBM.py` |
| OPT-3 阈值调整 | 已实现 | `eval.threshold=0.40` | `srcs/args_train_habitat_CBM.json` |
| OPT-5 手动 pos_weight | 已实现 | `loss.label.manual_pos_weight=2.5`，优先覆盖自动计算 | `srcs/train_habitat_CBM.py` |
| OPT-6 Stage1 调度器优化 | 已实现 | `stages.stage1.scheduler`: plateau + `factor=0.3`, `patience=3` | `srcs/args_train_habitat_CBM.json` |
| OPT-7 放宽早停 | 已实现 | `train.early_stop_patience=20`, `early_stop_min_delta=0.002` | `srcs/args_train_habitat_CBM.json` |
| OPT-8 患者均衡采样 | 已实现 | `train.patient_balanced_sampling=true`，使用 `WeightedRandomSampler(replacement=True)` | `srcs/data_loader_habitat_CBM.py` |
| OPT-9 Stage2 患者级 + 概念噪声 | 已实现（A+B） | `stages.stage2.patient_level=true`, `stages.stage2.concept_noise_std=0.15` | `srcs/data_loader_habitat_CBM.py` + `srcs/train_habitat_CBM.py` |
| OPT-11 更鲁棒概念 loss | 已实现 | `loss.concept.name=smooth_l1`, `loss.concept.beta=0.5` | `srcs/args_train_habitat_CBM.json` |
| 新增：PNG 结果图导出 | 已实现 | `eval.export_png=true`, `figure_include_splits=[train,val,test]`, `figure_dpi=150` | `srcs/eval_habitat_CBM.py` |

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

### 问题 D：block 级训练与患者级标签存在权重错配

**根因分析：**
1. 当前 Dataset 将每位患者展开成多个 2.5D block，但 IDH 标签和 8 维概念标签都是患者级标签，会复制到该患者的所有 block。
2. DataLoader 当前只是普通 `shuffle=True`，没有患者级或类别级均衡采样。因此，VOI 跨越切片更多的患者会在每个 epoch 中贡献更多梯度。
3. `pos_weight` 按训练集患者数计算，但 BCE 实际按 block 聚合。若不同类别或不同患者的有效 block 数不均衡，患者级 `pos_weight` 不能完全修正 block 级训练偏置。
4. Stage2 用真实概念训练 label_head 时，理论上只有 96 个唯一训练点，但脚本会按 block 重复同一患者概念，等价于按患者 block 数给 `C→Y` 样本加权，容易让 label_head 过快过拟合。

### 问题 E：Stage2 与 Stage3 存在概念分布偏移

**根因分析：**
1. Stage2 的 label_head 只见过真实标准化概念 `c_true_std`，而 Stage3/推理时 label_head 接收的是 `c_hat`。
2. 由于 Stage1 的概念预测仍有明显误差，`c_hat` 的分布、方差和概念间相关结构不一定与 `c_true_std` 一致。
3. 这会导致 Stage2 的高 oracle AUC 无法完整转化为 Stage3 性能，即“真实概念可分，但预测概念输入下分类头不稳”。

### 问题 F：当前增强策略可能对概念回归过强

**根因分析：**
1. 当前增强组合较强：`affine_prob=0.9`、旋转 30°、平移 20 px、缩放 25%，并叠加强度扰动、高斯噪声和 Gibbs 噪声。
2. Habitat 概念来自强度、形状、体积比例等 radiomics proxy，过强的空间或强度扰动可能改变这些概念对应的可见影像证据，但标签仍保持不变。
3. 对 Stage1 的 `X→C` 概念回归而言，过强增强可能不是正则化，而是引入 label-preserving 假设不成立的噪声。

### 问题 G：早停与学习率调度节奏偏紧

**根因分析：**
1. 全局 `early_stop_patience=10`，而 ReduceLROnPlateau 默认 `patience=5`。如果前 5 个 epoch 无改进，学习率刚降低，通常只剩约 5 个 epoch 证明低学习率是否有效。
2. Val 只有 14 位患者，AUC 的离散粒度较粗，单个患者预测变化就会明显改变曲线；用单次 split 的 `val_auc` 早停很容易把随机波动当成最优点。
3. Stage3 最优在 epoch 1 不一定只代表“训练方向错误”，也可能包含“小验证集 + 高方差 AUC + 早停节奏过紧”的模型选择噪声。

---

## 4. 优化策略（按优先级排序）

### 优化 OPT-1：修复 Stage3 学习率（**高优先级**）

**目标**：避免 Stage3 开始时从 Stage2 最优解大幅偏移，使联合微调真正产生正向作用。

**实现状态（2026-04-19）**：**已实现**。默认配置已写入 `stages.stage3.optimizer`，并通过 stage override 生效。

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

**实现状态（2026-04-19）**：**已实现**。默认 `eval.threshold` 已更新为 `0.40`。

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

**注意**：上述测试集 FN 概率只用于解释漏检机制，不能作为最终阈值选择依据。阈值调整属于推理参数，不影响训练权重，应在 val split 或 CV 上扫描不同阈值，绘制灵敏度-特异度曲线后再确定最终值，推荐扫描范围 `[0.35, 0.50]`，步长 0.05。

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

**实现状态（2026-04-19）**：**已实现**。训练脚本已支持 `loss.label.manual_pos_weight`，且在 `use_pos_weight=true` 时优先覆盖自动统计值。

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
- 本轮按需求与 OPT-3 同时启用（`threshold=0.40` + `manual_pos_weight=2.5`）。

---

### 优化 OPT-6：延长 Stage1 训练轮次或调整早停 patience（**中优先级**）

**目标**：给 Stage1 的概念 head 更多收敛机会，提升概念预测质量。

**实现状态（2026-04-19）**：**已实现**。Stage1 已新增独立 `scheduler` 覆盖，`factor=0.3`、`patience=3`。

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

### 优化 OPT-7：放宽早停，让 LR 衰减后有足够训练窗口（**中优先级**）

**目标**：解决“实际训练 epoch 过少”的问题，避免学习率刚下降就触发早停。

**实现状态（2026-04-19）**：**已实现**。全局早停参数已更新为 `early_stop_patience=20`、`early_stop_min_delta=0.002`。

**当前配置**：

```json
"train": {
  "early_stop_patience": 10,
  "early_stop_min_delta": 0.0
}
```

**推荐配置变更**：

```json
"train": {
  "early_stop_patience": 20,
  "early_stop_min_delta": 0.002
}
```

**说明**：
- `early_stop_patience` 建议至少大于 `scheduler.patience * 3`，让 ReduceLROnPlateau 降低学习率后仍有 8–12 个 epoch 的恢复窗口。
- `early_stop_min_delta=0.002` 可过滤极小浮动；对 Val AUC 这种小验证集离散指标，不建议设得过大。
- 当前脚本只支持全局早停参数；若后续修改代码，建议支持 stage 内覆盖：Stage1=20、Stage2=15、Stage3=20。

---

### 优化 OPT-8：引入患者均衡的 block 采样（**高优先级，需要代码支持**）

**目标**：让每位患者在每个 epoch 中贡献近似相同的梯度权重，避免“block 数多的患者支配训练”。

**实现状态（2026-04-19）**：**已实现**。`build_habitat_cbm_dataloaders()` 已支持 `patient_balanced_sampling` 开关并接入 `WeightedRandomSampler`。

**当前行为**：
- `HabitatIDHBlockDataset.sample_index` 是 block 级样本列表。
- `build_habitat_cbm_dataloaders()` 对 train split 使用普通 `shuffle=True`。
- 患者级标签被复制到所有 block，导致有效训练权重与患者 VOI 切片数成正比。

**推荐实现**：给每个 block 一个采样权重：

```text
weight(block_i) =
    1 / (num_blocks_of_patient[patient_id] * num_patients_of_class[label])
```

然后在 train DataLoader 中使用 `WeightedRandomSampler`：

```python
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True,
)
DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, ...)
```

**说明**：
- 第一项 `1 / num_blocks_of_patient` 保证每位患者总采样权重相近。
- 第二项 `1 / num_patients_of_class` 同时做患者级类别均衡。
- 应优先于继续增大 `manual_pos_weight`，因为它修正的是更底层的采样偏置。

---

### 优化 OPT-9：将 Stage2 改为患者级训练，并加入概念噪声鲁棒性（**高优先级，需要代码支持**）

**目标**：缓解 Stage2 oracle 到 Stage3 predicted concept 的分布偏移。

**实现状态（2026-04-19）**：**已实现 A+B**。已接入 `PatientConceptDataset`（患者级 Stage2）与训练期概念噪声 `concept_noise_std`；未实现 Bridge Stage（C）。

**推荐方案 A：Stage2 患者级去重**
- 为 Stage2 单独构建 `PatientConceptDataset`，每位患者只出现一次。
- 输入为患者级 `concept_true_std`，输出为患者级 `y_true`。
- 这样 label_head 学到的是患者级 `C→Y` 映射，而不是按 block 数加权后的映射。

**推荐方案 B：Stage2 加概念噪声**

```python
sigma = stage1_val_concept_rmse_per_concept  # 或统一取 0.10-0.20
c_input = c_true_std + torch.randn_like(c_true_std) * sigma
y_logit = model.forward_c_to_y(c_input)
```

**推荐方案 C：增加 Bridge Stage（预测概念适配）**
- 冻结 encoder + concept_head。
- 用 `c_hat.detach()` 训练 label_head，而不是只用 `c_true_std`。
- 这相当于先让 label_head 适应 Stage1 概念预测误差，再进入 Stage3 联合微调。

**说明**：
- Stage2 val AUC=0.896 证明真实概念可分，但不代表 label_head 对 `c_hat` 鲁棒。
- 如果 Stage3 继续出现 epoch 1 最优，优先排查 `c_true_std → c_hat` 分布偏移，而不是只调 Stage3 学习率。

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
- 建议先做一轮“弱增强”对照：若 Stage1 val concept_loss 明显下降，再逐步恢复强度增强。
- 当前脚本的增强是全阶段共享；若后续修改代码，建议支持 stage-specific augmentation：Stage1 弱增强，Stage2 无影像增强，Stage3 中等增强。

---

### 优化 OPT-11：使用更鲁棒的概念 loss，并做逐概念诊断（**中优先级**）

**目标**：避免少数噪声概念或异常值主导 Stage1/Stage3 的概念回归。

**实现状态（2026-04-19）**：**已实现**。默认概念 loss 已切换到 `smooth_l1(beta=0.5)`。

**推荐配置变更**：

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
- 优先优化“对 IDH 有信息但预测误差大”的概念；对“标签信息弱且预测噪声大”的概念，不应让它在 joint loss 中占过高权重。

**说明**：
- 当前标准化只解决尺度差异，不能解决异常值和概念标签噪声。
- 如果某一两个概念误差极大，可进一步实现 `concept_weights`，降低其 loss 权重；但这需要代码支持。

### 新增功能：输出 PNG 结果图（**已实现**）

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

**默认配置**：

```json
"eval": {
  "export_png": true,
  "figure_include_splits": ["train", "val", "test"],
  "figure_dpi": 150
}
```

---

### 优化 OPT-12：系统扫描患者级聚合与阈值（**中优先级**）

**目标**：区分“模型排序能力不足”和“患者级概率聚合/阈值不合适”。

**当前行为**：
- block 概率先按患者平均，再用固定 `threshold=0.5` 判定。
- `topk_pool=0` 表示使用全部 block 平均；`topk_pool>0` 时选择离 0.5 最远的高置信 block。

**推荐扫描**：

```text
threshold: 0.35, 0.40, 0.45, 0.50
topk_pool: 0, 5, 10, 20
```

**说明**：
- 阈值必须用 val split 选择，再一次性报告 test；不能根据 test FN 反向选择最终阈值。
- `topk_pool` 可能提升信噪比，也可能放大少数过度自信的错误 block，需要和阈值一起看 Sen/Spe。
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
Run A     OPT-1 + OPT-2（Stage3 lr + λ 权重）  Stage3 最优 epoch 是否延迟；
                                               Val AUC/Val loss 是否改善

Run B     Run A 基础上 + OPT-7（放宽早停）      Stage3 是否能在低 lr 后继续改善；
                                               Stage1/3 实际训练 epoch 是否增加

Run C     Run A 权重 + OPT-3/12（阈值和 topk） Val Sen 是否提升；
                                               Spe 是否仍 > 0.80

Run D     OPT-4 + OPT-6 + OPT-10 重训全部阶段  Val concept_loss 是否 < 0.616；
                                               Stage3 最优 epoch 是否后移

Run E     Run D 基础上 + OPT-11（SmoothL1）     逐概念 abs_error_std 是否下降；
                                               Val AUC/Sen 是否稳定

Run F     OPT-8（患者均衡采样，已实现）        Train/Val loss gap 是否缩小；
                                               患者间预测是否更稳定

Run G     OPT-9（患者级 Stage2 + 噪声，已实现A+B） Stage2 oracle 到 Stage3 的性能落差是否缩小；
                                               Stage3 label_loss 是否下降

Run H     最优配置做 OPT-13（多 seed/CV）       Val/CV mean±std 是否优于基准；
                                               最终 test 是否同步提升
─────────────────────────────────────────────────────────────────────
```

**推荐执行原则**：
- 先做 Run A/B/C，因为它们主要是配置和评估层面的改动，成本最低。
- 如果概念 loss 仍高，再做 Run D/E，集中优化 `X→C`。
- Run F/G 相关代码已落地，可直接做对照实验验证收益。

---

## 6. 配置文件快速对照表

下表总结了从基准配置到推荐配置的所有关键字段变动，供直接修改 `args_train_habitat_CBM.json` 参考。

| 字段路径 | 基准值 | 推荐值 | 生效范围 | 优化目标 |
|---------|--------|--------|----------|---------|
| `loss.joint.lambda_c` | 1.0 | **0.5** | Run A+ | Stage3 分类驱动力 |
| `loss.joint.lambda_y` | 0.5 | **1.0** | Run A+ | Stage3 分类驱动力 |
| `stages.stage3.optimizer.lr` | *(继承全局 1e-4)* | **1e-5** | Run A+ | Stage3 不偏移 |
| `stages.stage3.optimizer.lr_encoder` | *(继承全局 1e-4)* | **5e-6** | Run A+ | 保护 encoder |
| `stages.stage3.optimizer.lr_concept_head` | *(继承全局 1e-4)* | **1e-5** | Run A+ | 小步更新概念层 |
| `stages.stage3.optimizer.lr_label_head` | *(继承全局 1e-4)* | **1e-5** | Run A+ | 小步更新分类头 |
| `train.early_stop_patience` | 10 | **20** | Run B+ | 给 LR 衰减留出训练窗口 |
| `train.early_stop_min_delta` | 0.0 | **0.002** | Run B+ | 过滤小幅随机波动 |
| `eval.threshold` | 0.5 | **val 扫描 0.35–0.50** | Run C+ | 提升灵敏度 |
| `eval.topk_pool` | 0 | **0/5/10/20 扫描** | Run C+ | 优化患者级聚合 |
| `model.dropout_p` | 0.7 | **0.5** | Run D+ | 概念预测泛化 |
| `stages.stage1.scheduler.factor` | *(继承全局 0.5)* | **0.3** | Run D+ | Stage1 稳定收敛 |
| `stages.stage1.scheduler.patience` | *(继承全局 5)* | **3** | Run D+ | 更快降低 lr |
| `train.aug_rotate_deg` | 30.0 | **10.0** | Run D+ | 减少空间标签噪声 |
| `train.aug_translate_px` | 20.0 | **8.0** | Run D+ | 减少空间标签噪声 |
| `train.aug_scale_range` | 0.25 | **0.10** | Run D+ | 减少形状概念扰动 |
| `train.aug_gaussian_noise_std` | 0.05 | **0.02** | Run D+ | 减少强度概念扰动 |
| `train.aug_gibbs_noise_prob` | 0.3 | **0.0** | Run D+ | 先关闭强伪影增强 |
| `loss.concept.name` | mse | **smooth_l1** | Run E+ | 降低异常概念影响 |
| `loss.concept.beta` | *(无)* | **0.5** | Run E+ | SmoothL1 转折点 |
| `train.patient_balanced_sampling` | *(不支持)* | **true（已实现）** | Run F | 患者级梯度均衡 |
| `stages.stage2.patient_level` | *(不支持)* | **true（已实现）** | Run G | Stage2 去 block 重复 |
| `stages.stage2.concept_noise_std` | *(不支持)* | **0.10–0.20（已实现）** | Run G | label_head 适应 `c_hat` 误差 |

---

## 7. 代码实现对照（当前仓库）

### 7.1 `args_train_habitat_CBM.json` 中添加 Stage3 optimizer 覆盖

训练脚本已支持在 `stages.stage3` 内嵌 `optimizer` 子对象以覆盖全局配置。当前实现是 flat dict 合并：

```python
stage_optimizer_cfg = {
    **optimizer_cfg,
    **stage_cfg.get("optimizer", {}),
}
```

因此 `stages.stage3.optimizer` 中只需要写要覆盖的 key，未写字段会自动继承全局 `optimizer`。

### 7.2 `loss.label.manual_pos_weight` 支持（OPT-5，已实现）

当前 `train_habitat_CBM.py` 已支持手动覆盖与开关控制：

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

    # 原有逻辑：BCEWithLogits pos_weight = N_negative / N_positive
    ...
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

### 7.4 患者均衡采样（OPT-8，已实现）

可在 `data_loader_habitat_CBM.py` 中新增 train sampler 构建函数：

```python
from collections import Counter
from torch.utils.data import WeightedRandomSampler


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

然后在 `build_habitat_cbm_dataloaders()` 中通过配置切换：

```python
if patient_balanced_sampling:
    sampler = build_patient_balanced_sampler(datasets["train"])
    train_loader = DataLoader(..., sampler=sampler, shuffle=False)
else:
    train_loader = DataLoader(..., shuffle=True)
```

### 7.5 Stage2 患者级训练 + 概念噪声（OPT-9 A+B，已实现）

Stage2 不需要影像 block，可直接从 `dataset.patient_cases` 与 concept map 中构建患者级样本：

```python
class PatientConceptDataset(torch.utils.data.Dataset):
    def __init__(self, block_dataset: HabitatIDHBlockDataset) -> None:
        self.block_dataset = block_dataset
        self.patient_ids = sorted(block_dataset.patient_cases.keys())

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        pid = self.patient_ids[index]
        case = self.block_dataset.patient_cases[pid]
        concept_raw, concept_std = self.block_dataset._get_concepts(pid)
        return {
            "patient_id": pid,
            "concept_true_std": torch.as_tensor(concept_std, dtype=torch.float32),
            "label": torch.tensor(case.label_id, dtype=torch.long),
        }
```

训练主循环当前实现：
- Stage1/3 使用 block loader；
- Stage2 在 `stages.stage2.patient_level=true` 时使用 patient loader；
- Stage2 训练期在 `concept_noise_std>0` 时注入高斯噪声，验证期不注入。

### 7.6 概念分布偏移诊断（OPT-9/OPT-11）

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

### 7.7 PNG 结果图导出（新增功能，已实现）

`eval_habitat_CBM.py` 已支持图像导出，配置字段如下：

```json
"eval": {
  "export_png": true,
  "figure_include_splits": ["train", "val", "test"],
  "figure_dpi": 150
}
```

导出目录：`results/.../<run_id>/figures/`，并在 `run_summary` 的 `files.figures` 中记录路径。

---

## 8. 约束与注意事项

> 以下约束供 LLM 在生成修改建议时遵守，避免引入不兼容的改动。

1. **模型结构不变**：本轮优先优化不修改 `habitat_CBM.py` 中的网络结构（encoder/concept_head/label_head 的层数和维度）。所有高优先级变动优先通过配置文件或训练脚本的少量修改实现。
2. **三阶段协议不变**：Stage1→2→3 的训练顺序和各阶段的冻结策略保持不变。
3. **概念维度不变**：`n_concepts=8`，与现有 `concept_labels.csv` 保持一致，不重新生成概念标签。
4. **数据划分不变**：Train/Val/Test 的患者分配不变，保证结果可与基准实验（`20260419_155537`）直接对比。
5. **`args_train_habitat_CBM.json` 的 `_comment` 字段**：所有 `_comment_*` 字段是合法 JSON 注释字段，训练脚本会忽略它们，修改时保留这些注释字段。
6. **Stage3 optimizer 覆盖格式**：`stages.stage3.optimizer` 使用与全局 `optimizer` 相同的扁平 key；训练脚本会做浅层合并，缺失的 key 自动回退到全局配置。
7. **区分配置优化和代码优化**：OPT-1/2/3/4/6/7/10/11/12 可主要通过配置或评估脚本完成；OPT-8/9 已完成代码实现，后续重点是实验验证。
8. **Test 不参与调参**：阈值、topk、loss 权重、采样策略均应根据 train/val 或 CV 选择；test 只用于最终报告，避免把测试集错误案例反向用于调参。

---

*文档更新时间：2026-04-19（已覆盖 OPT-1/3/5/6/7/8/9/11 与 PNG 导出） | 基准实验 run_id：20260419_155537*
