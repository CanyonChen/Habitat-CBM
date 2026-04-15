# 多模态生理 MRI Habitat 掩膜说明

## 1. 先回答结论

对这篇论文

- `Interpretable Habitat Radiomics from Multimodal Physiological MRI for Grading and IDH Mutation Status Prediction of Adult-type Diffuse Gliomas.pdf`

来说，**真正最终进入后续 radiomics 建模的最优 habitat 掩膜是 `H1+2`**。

但要把这句话说完整，需要区分两个层次：

1. **候选 habitat 掩膜**
   - 论文先构造了 `7` 个 habitat 掩膜：
     - `H1`
     - `H2`
     - `H3`
     - `H1+2`
     - `H1+3`
     - `H2+3`
     - `H1+2+3`
   - 然后分别在这些 habitat 上提取 radiomics，比较性能。

2. **最终选中的 habitat 掩膜**
   - 论文后续“最佳 MRI 序列组合”分析所基于的 habitat，是 `H1+2`
   - 也就是说，**最终用于后续主线 radiomics 建模的 habitat mask 是 `H1+2`**

因此，如果你问的是：

- “论文里先试了哪些 habitat mask 去提 radiomics？”
  - 答案是 `7` 个 habitat 都试了。
- “论文最后选哪个 habitat 作为最优 mask 去继续做 radiomics 模型？”
  - 答案是 `H1+2`。

---

## 2. 为什么说最终是 `H1+2`

根据本地 PDF 抽取到的正文：

1. 论文在 `Habitat clustering and combination` 之后，明确说构造了：
   - `H1`
   - `H2`
   - `H3`
   - `H1+2`
   - `H1+3`
   - `H2+3`
   - `H1+2+3`

2. 在 `Feature extraction` 段，论文写的是：
   - radiomic features were extracted from **each combination of MRI sequence and habitat**

3. 在结果部分，论文又明确写到：
   - `Building on the optimal tumor habitat H1+2 ...`

4. 表 `Table S2` 的均值结果里，`H1+2` 的平均 AUC 也是最优：
   - grading mean AUC = `0.8853`
   - IDH mean AUC = `0.8645`

所以，方法上是“七个 habitat 都参与比较”，结果上是“`H1+2` 被选为最终最优 habitat”。

---

## 3. 在你当前数据集里，应该用哪些输入

根据 [dataset.md](../../dataset/dataset.md)，你这里的图像都已经完成预处理，因此**不需要再重做论文里的 DICOM 转换、N4、配准、重采样等步骤**。

从现在开始，你只需要使用这三个已经预处理好的对象：

1. `functional/<label>/<patient_id>/adc`
2. `functional/<label>/<patient_id>/cbf`
3. `functional/<label>/<patient_id>/voi`

其中：

- `functional/.../voi` 是 canonical VOI
- habitat 聚类和 habitat mask 生成，都应以这个 `VOI` 为准

如果需要做 overlay 检查，再额外读取：

4. `conventional/<label>/<patient_id>/t2flair`
5. `conventional/<label>/<patient_id>/t1`

---

## 4. 这篇论文里的三个 primary habitats 到底是什么

论文对三类主 habitat 的定义是：

- `H1`
  - high perfusion + high cellularity
  - 对应 **高 `CBF` + 低 `ADC`**
- `H2`
  - low perfusion + high cellularity
  - 对应 **低 `CBF` + 低 `ADC`**
- `H3`
  - low perfusion + low cellularity
  - 对应 **低 `CBF` + 高 `ADC`**

把它们翻译成更接近实现的形式，就是：

```text
H1 = high CBF + low ADC
H2 = low CBF + low ADC
H3 = low CBF + high ADC
```

注意：

- 这不是 `ADC-only` habitat
- 这是 `ADC + CBF` 的二维联合聚类

---

## 5. 最终用于 radiomics 的 mask 如何得到

因为你的数据已经预处理完成，所以从现在开始，流程可以直接从 `ADC / CBF / VOI` 出发。

## Step 1: 在 VOI 内取出配对体素

对每位患者：

1. 读取 `adc`
2. 读取 `cbf`
3. 读取 `voi`
4. 只保留 `VOI > 0` 的体素

得到：

```text
adc_vals = ADC[VOI > 0]
cbf_vals = CBF[VOI > 0]
```

然后按同一体素位置组成二维向量：

```text
x_i = (adc_i, cbf_i)
```

这一步不能错位。  
`ADC` 和 `CBF` 必须是一一对应的同体素配对。

---

## Step 2: 对 `ADC` 和 `CBF` 分别做 z-score

论文原文写的是：

- all paired voxels from both modalities were first normalized using z-score transformation

对你的患者级 habitat mask 任务，最合适的落地方式是：

- 在**单位患者自己的 VOI 内**
- 分别对 `ADC` 和 `CBF` 做 z-score

即：

```text
adc_z = (adc_vals - mean(adc_vals)) / std(adc_vals)
cbf_z = (cbf_vals - mean(cbf_vals)) / std(cbf_vals)
```

再组成聚类输入：

```text
X = [[adc_z_1, cbf_z_1],
     [adc_z_2, cbf_z_2],
     ...
     [adc_z_n, cbf_z_n]]
```

---

## Step 3: 在二维空间做 `K-means(k=3)`

对每位患者独立运行：

```python
KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)
```

输入：

```text
X.shape = [n_voxels, 2]
```

这里推荐默认采用**单患者独立聚类**，因为你最终需要的是每位患者自己的 habitat mask。

---

## Step 4: 把 raw cluster 映射成 `H1/H2/H3`

`K-means` 输出的 `cluster 0/1/2` 没有固定生理含义，必须重映射。

推荐按以下语义模板做匹配：

```text
H1 template = (-1, +1)   # low ADC, high CBF
H2 template = (-1, -1)   # low ADC, low CBF
H3 template = (+1, -1)   # high ADC, low CBF
```

做法：

1. 取 3 个 cluster centers
2. 在 z-score 空间中计算它们与三个模板的距离
3. 找到总距离最小的对应关系
4. 完成映射：
   - `H1 = 1`
   - `H2 = 2`
   - `H3 = 3`

映射后应满足的生理关系通常是：

- `CBF(H1)` 最高
- `ADC(H3)` 最高
- `ADC(H1)` 和 `ADC(H2)` 都偏低

---

## Step 5: 先生成三个 primary habitat masks

先回写三类 primary habitats：

- `H1`
- `H2`
- `H3`

在内存里你仍然可以维护一个三分类 `habitat3` 标签图，方便调试和可视化，但**不推荐作为最终磁盘输出格式**。

也就是说，可以逻辑上存在：

```text
{patient_id}_habitat3class.nii.gz
```

标签规则：

- background = `0`
- `H1 = 1`
- `H2 = 2`
- `H3 = 3`

同时导出三个二值掩膜：

- `{patient_id}_h1.nii.gz`
- `{patient_id}_h2.nii.gz`
- `{patient_id}_h3.nii.gz`

---

## Step 6: 再生成四个 composite habitat masks

论文后面比较的是 `7` 个 habitats，因此还要生成：

- `H1+2 = H1 OR H2`
- `H1+3 = H1 OR H3`
- `H2+3 = H2 OR H3`
- `H1+2+3 = H1 OR H2 OR H3`

其中：

```text
H123 = 原始整瘤 VOI
```

如果 `H123` 和 `VOI` 对不上，前面的 habitat 生成过程就有问题。

如果你只是为了复现论文最终主线结果，那么真正用于后续 radiomics 的核心 composite mask 是：

- `H1+2`

而：

- `H1+3`
- `H2+3`
- `H1+2+3`

可以只在方法说明或临时分析中保留，不必作为推荐输出文件长期保存。

结合你当前项目的落地需求，本地**推荐长期持久化保存**的仍然是四个文件：

- `H1`
- `H2`
- `H3`
- `H1+2`

---

## Step 7: 真正最终用于 radiomics 的 mask 是 `H1+2`

这一步是最关键的结论。

你要区分：

### 7.1 用于 habitat 比较阶段的掩膜

这一步要保留全部 `7` 个：

- `H1`
- `H2`
- `H3`
- `H1+2`
- `H1+3`
- `H2+3`
- `H1+2+3`

因为论文先是在这 `7` 个 habitat 上分别提取 radiomics，然后比较谁最好。

但如果你的目标是复现论文最终主线结果，而不是把论文整套对比实验原样重跑，那么本地**推荐持久化保存**的只需要下面四个：

- `H1`
- `H2`
- `H3`
- `H1+2`

### 7.2 用于最终主线 radiomics 模型的掩膜

这一步就是：

```text
H1+2
```

也就是把：

- `H1`（高灌注高细胞密度）
- `H2`（低灌注高细胞密度）

合并成一个最终 habitat mask。

从生物学解释上，论文认为 `H1+2` 代表的是：

- **高细胞密度区域**
- 同时覆盖活跃增殖区和浸润区

这也是它最终优于 `H1`、`H2`、`H3` 和 whole-tumor (`H1+2+3`) 的原因。

---

## 8. 一个最小可执行实现

```python
for patient in patients:
    adc = load_nifti(adc_path)
    cbf = load_nifti(cbf_path)
    voi = load_nifti(voi_path) > 0

    adc_vals = adc[voi]
    cbf_vals = cbf[voi]

    adc_z = (adc_vals - adc_vals.mean()) / adc_vals.std()
    cbf_z = (cbf_vals - cbf_vals.mean()) / cbf_vals.std()

    X = np.stack([adc_z, cbf_z], axis=1)

    km = KMeans(n_clusters=3, random_state=42, n_init=20)
    raw_labels = km.fit_predict(X)
    centers = km.cluster_centers_

    templates = np.array([
        [-1.0, +1.0],  # H1
        [-1.0, -1.0],  # H2
        [+1.0, -1.0],  # H3
    ])

    best_perm = None
    best_cost = None
    for perm in itertools.permutations([0, 1, 2]):
        cost = sum(np.linalg.norm(centers[perm[i]] - templates[i]) for i in range(3))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm

    mapping = {
        best_perm[0]: 1,  # H1
        best_perm[1]: 2,  # H2
        best_perm[2]: 3,  # H3
    }

    habitat3 = np.zeros_like(adc, dtype=np.uint8)
    habitat3[voi] = np.vectorize(mapping.get)(raw_labels)

    h1 = (habitat3 == 1).astype(np.uint8)
    h2 = (habitat3 == 2).astype(np.uint8)
    h3 = (habitat3 == 3).astype(np.uint8)
    h12 = ((habitat3 == 1) | (habitat3 == 2)).astype(np.uint8)
    h13 = ((habitat3 == 1) | (habitat3 == 3)).astype(np.uint8)   # 可选，不必持久化
    h23 = ((habitat3 == 2) | (habitat3 == 3)).astype(np.uint8)   # 可选，不必持久化
    h123 = (voi > 0).astype(np.uint8)                             # 可选，不必持久化
```

其中真正最终进入后续主线 radiomics 的掩膜，是：

```python
h12
```

---

## 9. 建议保存的文件

推荐目录：

```text
habitat_CBM/dataset/habitat_masks/
├── mutant/
│   ├── 005/
│   │   ├── h1.nii.gz
│   │   ├── h2.nii.gz
│   │   ├── h3.nii.gz
│   │   └── h12.nii.gz
│   └── ...
└── wild_type/
    ├── 003/
    │   ├── h1.nii.gz
    │   ├── h2.nii.gz
    │   ├── h3.nii.gz
    │   └── h12.nii.gz
    └── ...
```

这个组织方式参考了 [dataset](../../dataset) 的目录风格：

- 先按 `label` 分层：
  - `mutant/`
  - `wild_type/`
- 每位患者一个文件夹
- 患者文件夹下只保存四个 mask：
  - `h1.nii.gz`
  - `h2.nii.gz`
  - `h3.nii.gz`
  - `h12.nii.gz`

如果你的目标是和论文主线最接近，那么 radiomics 提取脚本实际优先读取的仍然应是：

```text
<label>/<patient_id>/h12.nii.gz
```

---

## 10. 你现在真正应该怎么做

既然 [dataset.md](../../dataset/dataset.md) 里的图像都已经完成预处理，那么现在的主线很简单：

1. 不要重做预处理
2. 直接读取 `ADC / CBF / functional VOI`
3. 生成 `H1/H2/H3`
4. 生成 `H1+2`
5. 推荐只保存四个 mask：
   - `H1`
   - `H2`
   - `H3`
   - `H1+2`
   保存格式参考 `dataset`：按 `label/patient_id/` 建目录
6. 如果是为了复现论文最终主结果，**把 `H1+2` 作为最终 radiomics 掩膜**

一句话总结：

```text
候选 habitat 掩膜一共有 7 个，但论文最后选中的最优 radiomics habitat mask 是 H1+2。
```

---

## 11. 参考

1. 数据集说明：[dataset.md](../../dataset/dataset.md)
2. 本地论文 PDF：
   - `Interpretable Habitat Radiomics from Multimodal Physiological MRI for Grading and IDH Mutation Status Prediction of Adult-type Diffuse Gliomas.pdf`
3. 本地抽取文本：
   - `/tmp/idh_habitat_paper.txt`
