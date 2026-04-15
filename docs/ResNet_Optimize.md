# ResNet-18 Baseline 训练结果分析与优化方案

> 基于实验 `20260414_171850` 的训练结果分析，针对 IDH 突变状态预测任务提出四套系统性优化方案。

---

## 一、训练结果现状

### 1.1 数据集概况

| 划分 | 患者数 | Block 数 | Mutant | Wild-type |
|------|--------|----------|--------|-----------|
| Train | 96 | 5,954 | 38 | 58 |
| Val | 14 | 685 | 6 | 8 |
| Test | 28 | 1,691 | 11 | 17 |

### 1.2 训练配置摘要

| 参数 | 值 |
|------|----|
| 主干网络 | ResNet-18（ImageNet 预训练） |
| 输入通道 | 35（6 模态 × 5 切片 + 5 VOI） |
| 冻结层 | conv1, layer1, layer2, layer3 |
| 可训练参数 | ~2.7M |
| 学习率 | 2e-5（分类头），分层差异化 |
| 权重衰减 | 1e-2 |
| Dropout | 0.7 |
| Label Smoothing | 0.2 |
| LR 调度器 | ReduceLROnPlateau |
| 早停 patience | 30 |
| 数据增强 | MONAI（仿射/翻转/强度变换） |

### 1.3 最终性能指标

| 指标 | Train | Val | Test |
|------|-------|-----|------|
| AUC | **0.9995** | 0.8333 | **0.8663** |
| ACC | 97.9% | 71.4% | 82.1% |
| F1 | 0.974 | 0.714 | 0.800 |
| Sensitivity | 97.4% | 83.3% | 90.9% |
| Specificity | 98.3% | 62.5% | 76.5% |
| Loss | 0.476 | 0.814 | 0.688 |

### 1.4 混淆矩阵

**验证集（14 人）**
```
              Pred 0  Pred 1
  True 0 (WT)    5      3    ← 3 个 FP
  True 1 (MT)    1      5    ← 1 个 FN
```

**测试集（28 人）**
```
              Pred 0  Pred 1
  True 0 (WT)   13      4    ← 4 个 FP
  True 1 (MT)    1     10    ← 1 个 FN
```

---

## 二、核心问题诊断

### 问题 1：严重的极早期过拟合（最关键）

- **Best epoch = 3**，仅训练 3 个 epoch 就达到验证集最优，随后连续 30 个 epoch 无提升触发早停
- Train AUC = 0.9995 vs Val AUC = 0.8333，差距 **0.166**
- Train ACC = 97.9% vs Val ACC = 71.4%，差距 **26.5 个百分点**
- Train Loss = 0.476 vs Val Loss = 0.814，差距 **0.338**

根本原因：训练集 5,954 个 block 对应 96 个患者，而验证集只有 14 个患者（685 个 block），训练信号远强于泛化信号。即使当前已有 dropout=0.7、weight_decay=0.01、label_smoothing=0.2、层冻结等措施，模型仍在极少轮次内记忆训练样本特征。

### 问题 2：验证集特异性偏低（Specificity = 62.5%）

- 3 个 Wild-type 患者被误判为 Mutant（FP）
- 尽管训练集已使用类别权重补偿（WT:58 vs MT:38），模型仍存在对 Mutant 的预测偏置
- 这一偏置在测试集同样存在（FP=4），是系统性问题

### 问题 3：验证集信号极度嘈杂

- 14 个患者的 AUC 分辨率 ≈ 1/C(14,2) ≈ 0.0110，最小可分辨 AUC 步长约 **0.0156**
- 单个患者的预测变化就会导致 AUC 大幅波动，验证指标不够稳定
- Best epoch=3 的选择本身可能存在随机性，并非真正的泛化最优点

---

## 三、优化方案

---

### 方案 A：解决极早期过拟合（优先级：★★★★★）

本方案直接针对 best_epoch=3 的根本原因，通过减缓学习速度和减少可训练参数来缓解极早期过拟合。

#### A1. 大幅降低学习率

**问题**：当前分类头 lr=2e-5，骨干 layer4 lr=2e-6。Best epoch=3 说明即使如此，模型仍然在极少轮次内发生参数偏移，泛化窗口极短。

**方案**：将学习率进一步降低 4 倍。

```bash
--lr 5e-6
```

效果预期：
- 分类头 lr 从 2e-5 → 5e-6，参数更新幅度减小
- Best epoch 预期从 3 提升至 10~30，模型有更多时间在泛化区间内收敛
- 配合 ReduceLROnPlateau，plateau 触发后 lr 可进一步降至 2.5e-6

#### A2. 进一步冻结 backbone，只训练分类头

**问题**：当前冻结 conv1+layer1+layer2+layer3，layer4（约 2.1M 参数）仍在微调。对于 96 个患者、仅 14 个验证患者的场景，layer4 的自由度仍然过大。

**方案**：完全冻结 backbone，只训练 fc 层（约 1K 参数）。

```bash
--freeze-layers conv1,layer1,layer2,layer3,layer4
```

可训练参数量对比：
| 冻结策略 | 可训练参数 |
|----------|-----------|
| 不冻结（全量微调） | ~11M |
| 冻结到 layer3（当前） | ~2.7M |
| 冻结到 layer4（方案 A2） | ~1K |

注意：若冻结全部 backbone 后验证 AUC 显著下降（低于 0.75），说明 layer4 的医学影像适配仍有必要，可回退到只冻结 conv1~layer2。

#### A3. 调整早停策略，给模型更多探索时间

**问题**：当前 patience=30、min_delta=0.0，策略本已宽松，但 best_epoch=3 说明模型在早期就找到了"局部最优"并停滞。

**方案**：
```bash
--early-stop-patience 50
--early-stop-min-delta 0.01
```

说明：
- patience 从 30 提升到 50，允许模型在更长时间内寻找更好的泛化点
- min_delta 从 0.0 提升到 0.01，要求 AUC 提升至少 0.01 才算有效改进（约 14 人验证集最小分辨率的 0.64 倍），过滤随机波动

#### A 方案完整命令

```bash
python repo/srcs/baseline_ResNet18.py \
  --lr 5e-6 \
  --weight-decay 0.01 \
  --dropout-p 0.7 \
  --label-smoothing 0.2 \
  --freeze-layers conv1,layer1,layer2,layer3,layer4 \
  --early-stop-patience 50 \
  --early-stop-min-delta 0.01 \
  --lr-scheduler plateau \
  --epochs 200 \
  --run-id exp_A_freeze_all_low_lr
```

---

### 方案 B：增强数据层面（优先级：★★★★☆）

本方案通过提升输入数据质量和增强多样性，从数据角度缓解过拟合，与方案 A 可叠加使用。

#### B1. 启用 VOI 背景掩码

**问题**：当前 `mask_background_with_voi=false`，肿瘤区域（VOI）外的脑组织和背景噪声被完整输入网络。这些无关区域不含 IDH 分类信号，但会增加模型需要忽略的无关特征，降低泛化能力。

**方案**：开启 VOI 背景掩码，只保留肿瘤区域信息。

```bash
--mask-background-with-voi true
```

效果预期：
- 输入信息更集中于肿瘤区域，减少背景噪声对特征提取的干扰
- 模型专注于学习 IDH 相关的肿瘤内部特征
- 可能略微降低 block 数量（VOI 外切片被过滤），但提升单个样本质量

**注意**：开启后 `require_voi` 会自动强制为 `true`，确保每位患者都有 VOI 掩模。

#### B2. 提高 min_nonzero_voxels 过滤低质量切片

**问题**：当前 `min_nonzero_voxels=16`，允许只有 16 个前景体素的切片进入训练集。这类切片几乎不包含有效的肿瘤信息，但会作为噪声样本参与训练，增加模型学习难度。

**方案**：
```bash
--min-nonzero-voxels 32
```

效果预期：
- 过滤掉 VOI 区域极小（≤32 体素）的低质量切片
- 减少约 5~15% 的训练 block，但每个 block 的信噪比更高
- 有助于减少模型对无效切片的过拟合

#### B3. 增强数据增强强度

**问题**：当前增强参数（旋转 ±20°、平移 15px、强度变换 ±0.15）相对保守，对于仅 96 个训练患者的小样本场景，增强多样性不足。

**方案**：
```bash
--aug-rotate-deg 30.0           # 旋转范围从 ±20° 提升到 ±30°
--aug-translate-px 20.0         # 平移范围从 15px 提升到 20px
--aug-scale-range 0.25          # 缩放范围从 0.2 提升到 0.25
--aug-intensity-scale 0.25      # 强度缩放从 ±0.15 提升到 ±0.25
--aug-intensity-shift 0.25      # 强度偏移从 ±0.15 提升到 ±0.25
--aug-affine-prob 0.9           # 仿射变换概率从 0.8 提升到 0.9
```

效果预期：
- 增大训练样本多样性，降低模型对特定方向/强度的过拟合
- 模拟更大范围的 MRI 采集差异，提升域泛化能力

#### B 方案完整命令（可与方案 A 叠加）

```bash
python repo/srcs/baseline_ResNet18.py \
  --mask-background-with-voi true \
  --min-nonzero-voxels 32 \
  --aug-rotate-deg 30.0 \
  --aug-translate-px 20.0 \
  --aug-scale-range 0.25 \
  --aug-intensity-scale 0.25 \
  --aug-intensity-shift 0.25 \
  --aug-affine-prob 0.9 \
  --run-id exp_B_stronger_aug_voi_mask
```

---

### 方案 C：损失函数与预测聚合策略优化（优先级：★★★☆☆）

本方案从训练目标和推理聚合两个角度优化，适合在 A、B 方案稳定后进一步提升性能。

#### C1. 进一步增大 Label Smoothing

**问题**：当前 `label_smoothing=0.2`，训练集 AUC=0.9995 说明模型仍然对训练样本过于自信。

**方案**：
```bash
--label-smoothing 0.3
```

原理：将 one-hot 标签从 `[1, 0]` 软化为 `[0.85, 0.15]`（ε=0.3 时为 `[0.85, 0.15]`），防止模型将 logits 推向极端值，增强正则化效果。

**注意**：label_smoothing 过大（>0.4）可能导致模型无法有效区分类别，建议不超过 0.3。

#### C2. 患者级聚合改为 Top-K 均值池化

**问题**：当前采用所有 block 预测概率的简单均值聚合。每位患者的切片质量参差不齐，低质量切片（VOI 极小、边缘切片）的预测可靠性低，会拉低整体聚合质量。

**方案**：修改 `patient_level_from_block_predictions` 函数，改为取 Top-K 高置信度 block 的均值。

```python
def patient_level_from_block_predictions(
    records: List[Dict[str, object]],
    split_name: str,
    run_id: str,
    checkpoint_name: str,
    threshold: float,
    top_k: int = 0,          # 0 表示使用全部 block（默认行为）
) -> List[Dict[str, object]]:
    grouped = defaultdict(lambda: {"y_true": None, "probs": []})

    for record in records:
        patient_id = str(record["patient_id"])
        grouped[patient_id]["probs"].append(float(record["prob_idh_mut"]))
        if grouped[patient_id]["y_true"] is None:
            grouped[patient_id]["y_true"] = int(record["y_true"])

    rows = []
    for patient_id in sorted(grouped):
        y_true = int(grouped[patient_id]["y_true"])
        probs = grouped[patient_id]["probs"]

        # Top-K 池化：只取置信度最高的 K 个 block
        if top_k > 0 and len(probs) > top_k:
            # 按离 0.5 的距离排序，取最确定的 K 个预测
            probs = sorted(probs, key=lambda p: abs(p - 0.5), reverse=True)[:top_k]

        prob = float(np.mean(probs))
        pred_label = int(prob >= threshold)
        rows.append({
            "patient_id": patient_id,
            "split": split_name,
            "y_true": y_true,
            "prob_idh_mut": prob,
            "pred_label": pred_label,
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
        })

    return rows
```

推荐参数：`top_k = max(1, len(probs) // 3)`，即取置信度最高的前 1/3 block。

#### C3. 动态调整分类阈值（针对 Specificity 偏低）

**问题**：当前阈值 threshold=0.5，验证集 Specificity 仅 62.5%（3 个 FP），对 Wild-type 识别能力不足。

**方案**：在验证集上搜索最优阈值，平衡 Sensitivity 和 Specificity。

```bash
--threshold 0.55    # 小幅提高阈值，降低 FP 率
```

或在训练结束后，基于验证集 ROC 曲线自动选择 Youden Index 最优阈值：

```python
# 基于验证集 ROC 找最优阈值（Youden Index: max(TPR - FPR)）
fpr, tpr, thresholds = roc_curve(y_true_val, y_prob_val)
best_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[best_idx]
```

#### C 方案完整命令

```bash
python repo/srcs/baseline_ResNet18.py \
  --label-smoothing 0.3 \
  --threshold 0.55 \
  --run-id exp_C_loss_tuning
```

---

### 方案 D：训练范式改进（优先级：★★★☆☆，长期方向）

本方案涉及架构调整或训练协议重构，适合作为后续进阶实验的方向。

#### D1. 替换为更轻量的主干网络

**问题**：ResNet-18 即使冻结到 layer3，layer4 仍有 ~2.1M 参数，对于 96 个训练患者而言过于冗余。

**推荐替代网络**：

| 网络 | 总参数量 | 说明 |
|------|----------|------|
| EfficientNet-B0 | ~5.3M | 效率更高，适合小数据集 |
| MobileNetV3-Small | ~2.5M | 极轻量，适合极小数据集 |
| ResNet-10（自定义） | ~6M | 比 ResNet-18 浅 2 个残差块 |

使用 EfficientNet-B0 示例（需修改模型定义文件）：

```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
# 替换第一层卷积以适配 35 通道输入
model.features[0][0] = nn.Conv2d(35, 32, kernel_size=3, stride=2, padding=1, bias=False)
# 替换分类头
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.7),
    nn.Linear(in_features, 2),
)
```

#### D2. 患者级别 K-Fold 交叉验证

**问题**：当前验证集仅 14 个患者，AUC 最小分辨率 ≈ 0.0156，验证信号极度嘈杂，Best epoch 的选择存在较大随机性。

**方案**：将 train+val（110 个患者）合并，进行 5-fold 患者级别交叉验证。

```
Fold 1: train(88人) + val(22人)
Fold 2: train(88人) + val(22人)
...
Fold 5: train(88人) + val(22人)
```

优势：
- 每折验证集 22 人，AUC 分辨率提升 1.5 倍
- 5 折均值 AUC 作为最终指标，大幅降低随机性
- 可基于 5 个模型集成进行最终预测，提升测试集稳定性

实施步骤：
1. 修改 `build_datasets()` 支持接收患者 ID 列表而非目录路径
2. 使用 `sklearn.model_selection.StratifiedKFold` 按 IDH 标签分层
3. 外循环 K 折，内循环保持现有训练逻辑不变

#### D3. 测试时间增强（TTA）

**方案**：推理阶段对每个 block 应用 N 次随机变换，取均值概率作为最终预测。

```python
def predict_with_tta(model, images, n_tta=10, transforms=None):
    """测试时间增强：对同一输入施加多次随机变换并平均预测概率。"""
    model.eval()
    all_probs = []
    with torch.no_grad():
        # 原始预测
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs)
        # TTA 变换预测
        for _ in range(n_tta - 1):
            aug_images = transforms(images)
            logits = model(aug_images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs)
    return torch.stack(all_probs).mean(dim=0)
```

效果预期：可将测试集 AUC 提升 0.01~0.03，尤其对预测概率接近阈值的边界病例有显著改善。

#### D4. Mixup 或 CutMix 数据增强

**方案**：在训练阶段对 block 进行 Mixup（线性插值）或 CutMix（区域替换）增强，进一步扩大训练分布，增强正则化效果。

```python
def mixup_data(x, y, alpha=0.2):
    """Mixup：对 batch 内样本进行线性插值。"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

推荐参数：`alpha=0.2`（轻度混合），避免过大 alpha 导致医学影像语义失真。

---

## 四、推荐实验路线图

### 第一步：快速验证（1-2 次实验）

将方案 A 的三个子方案全部叠加，验证是否能解决 best_epoch=3 的问题：

```bash
python repo/srcs/baseline_ResNet18.py \
  --lr 5e-6 \
  --weight-decay 0.01 \
  --dropout-p 0.7 \
  --label-smoothing 0.2 \
  --freeze-layers conv1,layer1,layer2,layer3,layer4 \
  --early-stop-patience 50 \
  --early-stop-min-delta 0.01 \
  --lr-scheduler plateau \
  --epochs 200 \
  --run-id exp_step1_freeze_all
```

**成功标准**：best_epoch ≥ 15，Val AUC ≥ 0.85

### 第二步：叠加数据增强（1-2 次实验）

在第一步基础上叠加方案 B：

```bash
python repo/srcs/baseline_ResNet18.py \
  --lr 5e-6 \
  --weight-decay 0.01 \
  --dropout-p 0.7 \
  --label-smoothing 0.2 \
  --freeze-layers conv1,layer1,layer2,layer3,layer4 \
  --mask-background-with-voi true \
  --min-nonzero-voxels 32 \
  --aug-rotate-deg 30.0 \
  --aug-translate-px 20.0 \
  --aug-scale-range 0.25 \
  --aug-intensity-scale 0.25 \
  --aug-intensity-shift 0.25 \
  --aug-affine-prob 0.9 \
  --early-stop-patience 50 \
  --early-stop-min-delta 0.01 \
  --run-id exp_step2_aug_voi
```

**成功标准**：Val AUC ≥ 0.87，Val Specificity ≥ 70%

### 第三步：精调损失函数与阈值（1 次实验）

```bash
python repo/srcs/baseline_ResNet18.py \
  # ... 继承第二步所有参数 ...
  --label-smoothing 0.3 \
  --threshold 0.55 \
  --run-id exp_step3_loss_tuning
```

### 第四步：引入交叉验证（长期）

修改训练脚本，实现患者级别 5-fold 交叉验证，得到更稳定可靠的性能估计。

---

## 五、优化方案汇总对照表

| 方案 | 子项 | 参数改动 | 预期收益 | 实施难度 |
|------|------|----------|----------|----------|
| **A1** | 降低学习率 | `--lr 5e-6` | best_epoch 延后，减缓过拟合 | 低 |
| **A2** | 冻结全部 backbone | `--freeze-layers ...layer4` | 可训练参数 2.7M→1K | 低 |
| **A3** | 调整早停 | `--early-stop-patience 50 --min-delta 0.01` | 给更多泛化探索时间 | 低 |
| **B1** | VOI 背景掩码 | `--mask-background-with-voi true` | 减少背景噪声干扰 | 低 |
| **B2** | 提高最小体素数 | `--min-nonzero-voxels 32` | 提升训练样本质量 | 低 |
| **B3** | 增强数据增强 | 旋转/平移/强度参数上调 | 增大训练分布多样性 | 低 |
| **C1** | 增大 label smoothing | `--label-smoothing 0.3` | 进一步防止模型过度自信 | 低 |
| **C2** | Top-K 聚合 | 修改聚合函数 | 提升聚合鲁棒性 | 中 |
| **C3** | 动态阈值 | `--threshold 0.55` 或 Youden | 改善 Specificity | 低-中 |
| **D1** | 轻量网络 | EfficientNet-B0 / MobileNetV3 | 减少模型容量，缓解过拟合 | 中 |
| **D2** | K-Fold 交叉验证 | 修改训练协议 | 验证信号更稳定可靠 | 高 |
| **D3** | TTA | 推理端增强 | 测试集 AUC +0.01~0.03 | 中 |
| **D4** | Mixup/CutMix | 训练端混合增强 | 进一步正则化 | 中 |
