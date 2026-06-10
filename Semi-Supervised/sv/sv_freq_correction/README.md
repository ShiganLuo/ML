# SV Frequency Correction — Model Training

将SV检测软件输出的突变频率(Freq)校正至接近ddPCR真实频率(ddPCR_AF)的回归模型。

## 数据

| 数据集 | 文件 | 行数 | 说明 |
|--------|------|------|------|
| 标签数据 | `SV_processed_ddPCR.tsv` | ~936 | 有ddPCR真实频率，约26个独立标签(原始编号) |
| 非标签数据 | `SV_processed_no_ddPCR.tsv` | ~22506 | 无ddPCR，占总数据96% |

两者共享列: `sampleID`, `FusionGene`, `FusionExon`, `BamPath`, `Pos1`, `Pos2`, `Freq`, `FusionType`  
标签数据额外有: `原始编号`, `ddPCR_AF`

核心挑战: 标签极少(26个unique ddPCR_AF)，非标签数据量是标签的24倍。

## 特征工程 (`features.py`)

`parser_table()` 从BAM文件提取突变级别特征，每个突变一行。

**BAM深度特征 (per breakpoint × 2):**
`depth_left`, `depth_right`, `depth_center`, `depth_sum`, `depth_diff`, `depth_asymmetry`

**突变级聚合:**
`bp_support_max`, `bp_depth_min`, `bp_depth_asymmetry_mean`

**Read证据特征:**
`read_total_unique`, `read_support_unique`, `read_support_fraction`, `read_mapq_mean`, `read_mapq_max`,  
`read_softclip_fraction`, `read_split_fraction`, `read_discordant_fraction`, `read_template_mean`, `read_template_std`,  
`read_breakpoint_balance`

**SV类型 one-hot (不参与标准化):**
`sv_type_CTX-1`, `sv_type_CTX-2`, `sv_type_DEL/ITX`, `sv_type_DUP/ITX`, `sv_type_INV`

**预处理流程 (`preprocess_features()`):**
1. StandardScaler 标准化 (跳过 sv_type_* one-hot列)
2. VarianceThreshold 过滤近常数特征
3. 相关性 > 0.95 的特征聚类，保留方差最大的代表

经预处理后保留约17个numeric特征 + 5个sv_type = 22个特征。

## 模型架构

### 基线模型 (`train.py`)

纯监督学习，仅用标签数据。12种sklearn回归器，通过 `ModelRegistry` 统一管理:

| 类别 | 模型 |
|------|------|
| 线性 | `ridge`, `elastic_net`, `lasso` |
| SVM | `svr_rbf`, `svr_linear`, `svr_poly` |
| 树集成 | `random_forest`, `gradient_boosting`, `extra_trees`, `adaboost` |
| 其他 | `knn`, `mlp` |

训练流程: GroupShuffleSplit分组划分 → GridSearchCV超参搜索 → 模型对比 → 保存最优模型

---

### 半监督方法

三种方法均通过 `_data.py` 加载标签+非标签数据，共享特征提取和预处理。
标签变换: 默认 logit (将(0,1)频率映射到实数域)。
低频加权: 反频率权重 `w = (1/freq)^power`，提升低频样本影响力。

#### 方法1: 伪标签自训练 (`self_training.py`)

**方法:** Self-Training，经典的半监督回归方法。

**回归模型:** 复用 `ModelRegistry`，默认 `gradient_boosting`，可选全部12种。

**流程:**
```
iter_0: 标签数据训练 → 评估test set
iter_1: 预测非标签数据 → 置信度过滤 → 合并训练 (伪标签权重×0.5)
iter_2: 重复...
iter_n: 无新伪标签时提前终止
```

**置信度过滤 (`--filter-strategy`):**
- `range`: 预测值在 (ε, 1-ε) 内
- `residual`: |预测 - Freq| < 分位数阈值 (Freq是已知的检测频率，残差小=预测更可信)
- `top_k`: 按残差排序取前k%
- `combined`: 三者交集（默认）

**安全机制:**
- 伪标签数量上限: `max_pseudo_fraction × 标签数据量` (默认50%)
- 伪标签样本权重: 标签数据的0.5倍
- 每轮保存 pseudo_labels.tsv 供人工审查

```bash
python self_training.py \
  --labeled-tsv .../SV_processed_ddPCR.tsv \
  --unlabeled-tsv .../SV_processed_no_ddPCR.tsv \
  -o output/self_training \
  --model-name gradient_boosting \
  --n-iterations 3 \
  --filter-strategy combined \
  --group-cols 原始编号
```

#### 方法2: 去噪自编码器 (`semi_supervised_ae.py`)

**方法:** 用全部数据(标签+非标签)训练去噪自编码器，提取latent特征，再用标签数据训练回归器。

**自编码器:** sklearn `MLPRegressor`，手动提取中间层激活作为latent representation。
```
n_features → [encoder_hidden] → latent_dim → [decoder_hidden] → n_features
```
- 输入: 原始特征 + 高斯噪声 (denoising)
- 目标: 重建原始特征
- latent提取: 手动用 `mlp.coefs_` / `mlp.intercepts_` 做encoder前向传播

**回归模型:** 同样是 `ModelRegistry` 的 `gradient_boosting`，在latent特征上训练。

**对比评估:** 同一回归模型在原始特征 vs latent特征上的表现差异，判断自编码器是否学到了有用的压缩表示。

```bash
python semi_supervised_ae.py \
  --labeled-tsv .../SV_processed_ddPCR.tsv \
  --unlabeled-tsv .../SV_processed_no_ddPCR.tsv \
  -o output/semi_ae \
  --latent-dim 8 \
  --noise-std 0.1 \
  --model-name gradient_boosting \
  --group-cols 原始编号
```

#### 方法3: 一致性正则化 (`consistency_reg.py`)

**方法:** 训练模型时同时优化两个loss:
```
loss = MSE(pred, label) + λ × MSE(pred(x), pred(x + noise))
```
第二项强制模型对输入扰动保持预测稳定——利用非标签数据的分布结构。

**运行模式 (自动检测PyTorch):**

| 模式 | 实现 | 说明 |
|------|------|------|
| 有PyTorch | `nn.Module` 自定义网络 | 真正的梯度级consistency loss。两层全连接 + ReLU + Dropout，Adam优化器，ReduceLROnPlateau调度 |
| 无PyTorch | `ModelRegistry` 集成 | **近似方案**：多轮在扰动特征上训练集成模型，用预测方差过滤伪标签。本质是bagging + pseudo-label，不是真正的consistency regularization |

```bash
python consistency_reg.py \
  --labeled-tsv .../SV_processed_ddPCR.tsv \
  --unlabeled-tsv .../SV_processed_no_ddPCR.tsv \
  -o output/consistency_reg \
  --consistency-weight 1.0 \
  --noise-std 0.1 \
  --group-cols 原始编号
```

---

### 三种方法对比

| 维度 | 方法1 Self-Training | 方法2 Autoencoder | 方法3 Consistency |
|------|-------------------|-------------------|-------------------|
| 利用非标签数据方式 | 扩充训练集(伪标签) | 学习特征表示 | 正则化约束 |
| 回归器 | ModelRegistry (12种) | ModelRegistry (12种) | PyTorch MLP / ModelRegistry |
| 需要额外依赖 | 无 | 无 | PyTorch(可选) |
| 核心风险 | 伪标签噪声累积 | MLPRegressor提取latent粗糙 | 无PyTorch时退化为bagging |
| 适合场景 | 标签模型较可靠时 | 特征冗余/高维时 | 特征空间平滑假设成立时 |

**共同局限:** 三种方法的回归器本质都是sklearn传统模型，对小标签(~26 unique)场景改进空间有限。非标签数据的价值主要体现在:
- 方法1: 扩充训练样本数 (但伪标签有噪声)
- 方法2: 学习更好的特征压缩 (但sklearn MLP能力有限)
- 方法3: 利用特征空间的平滑性 (但tabular数据平滑性假设不一定成立)

**可能的进阶方向:**
- TabNet / SAINT 等tabular专用架构做半监督预训练
- 对比学习 (SimCLR/MoCo思想) 学特征表示
- VAE 替代普通自编码器，学习频率的分布而非点重建

## 公共参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--labeled-tsv` | (必填) | 标签数据TSV |
| `--unlabeled-tsv` | (必填) | 非标签数据TSV |
| `-o, --out-dir` | (必填) | 输出目录 |
| `--probe-infile` | None | 探针序列BED文件 |
| `--feature-dir` | None | 共享BAM特征缓存目录 |
| `--group-cols` | `原始编号` | 分组列 (repeatable) |
| `--target-transform` | `logit` | 标签变换: `none` 或 `logit` |
| `--clip-epsilon` | 1e-6 | logit变换裁剪边界 |
| `--test-size` | 0.2 | 测试集比例 |
| `--random-state` | 42 | 随机种子 |
| `--weight-low-af` | True | 低频样本加权 |
| `--enable-cv` | True | 启用交叉验证调参 |
| `--cv-folds` | 10/5 | CV折数 |
| `--force-extract` | False | 强制重新提取BAM特征 |

## 特征缓存共享

BAM特征提取耗时较长(~22500行)。三个半监督脚本通过 `--feature-dir` 共享缓存:

```bash
# 步骤1: 首次运行提取特征 (耗时)
python self_training.py --labeled-tsv ... --unlabeled-tsv ... -o output/self_training

# 步骤2: 后续脚本复用特征缓存 (秒级)
python semi_supervised_ae.py ... -o output/semi_ae \
  --feature-dir output/self_training/features

python consistency_reg.py ... -o output/consistency_reg \
  --feature-dir output/self_training/features
```

缓存结构:
```
features/
├── labeled/
│   └── labeled_raw_features.tsv
├── unlabeled/
│   └── unlabeled_raw_features.tsv
├── preprocessing_scaler.joblib
├── preprocessing_metadata.json
└── preprocessing_diagnostics.json
```

## 文件结构

```
sv_freq_correction/
├── features.py              # 特征提取(parser_table) + 预处理(preprocess_features)
├── train.py                 # 基线监督训练 (ModelRegistry, 12种回归器)
├── _data.py                 # 半监督数据加载 (标签+非标签合并, 共享预处理)
├── self_training.py         # 方法1: 伪标签自训练
├── semi_supervised_ae.py    # 方法2: 去噪自编码器
├── consistency_reg.py       # 方法3: 一致性正则化
├── predict.py               # 预测推理
├── diagnose_split.py        # 训练/测试集诊断
└── report_correlated.py     # 相关性特征报告
```

## 输出

每个半监督方法输出 `summary.json`，包含:
- baseline指标 (仅标签数据): MSE, MAE, R², Pearson r
- 半监督指标 (纳入非标签数据)
- 改善幅度: MSE reduction, R² gain

三个方法可横向对比。若改善不显著，说明当前回归器能力不足以从非标签数据中获益，需考虑进阶方向。
