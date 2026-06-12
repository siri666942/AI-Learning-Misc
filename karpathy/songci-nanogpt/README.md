# Poetry-Nano 🎋

基于 [nanoGPT](https://github.com/karpathy/nanoGPT) 重构的**中国古典诗词生成模型**。
8GB 显存友好，字级分词，专为五言/七言格律精准控制设计。

> 架构参数经论文验证：[Muennighoff et al. (2023)](https://arxiv.org/abs/2305.16264) 数据受限 scaling law、
> [Krajewski et al. (2024)](https://arxiv.org/abs/2402.07871) MoE scaling law、
> [Zhou et al. (2024)](https://arxiv.org/abs/2404.10102) 多 epoch 过拟合研究。

## 核心特性

- **字级分词**: 1 Token = 1 汉字，格律字数绝对精准
- **领域标签隔离**: `[POEM]` `[CLASSICAL]` `[MODERN]` 防止文白混杂
- **Dense 架构**: 论文验证 50M 参数级 Dense 优于 MoE，训练更稳定 + torch.compile 可用
- **混合数据配比**: 80% 诗词 / 20% 古典文（可选加入现代文）
- **消费级 GPU**: RTX 4060 8GB 即可从零训练（峰值显存 <2GB）

## 项目结构

```
songci-nanogpt/
├── char_tokenizer.py       # 字级分词器
├── prepare_poetry_data.py  # 数据流水线 (抽取→清洗→编码→.npy)
├── inspect_data.py         # 数据分布检查工具
├── train.ipynb             # GPT 模型定义 + 训练循环
├── vocab.json              # 词表 (prepare 后生成)
├── poetry_data/            # .npy 训练数据 (prepare 后生成)
│   ├── poetry_train_*.npy
│   └── poetry_val_*.npy
└── log/                    # 训练日志 & checkpoint
```

## 模型配置

| 参数 | 值 | 说明 |
|:---|:---|:---|
| block_size | 256 | 上下文窗口（诗词平均 77 字，中位 50 字） |
| vocab_size | 13,312 | 7 特殊 + 13,305 汉字，对齐 128 倍数触发 Tensor Core |
| n_layer | 10 | Muennighoff: 数据受限时降深度优先于宽度 |
| n_head | 9 | 576 ÷ 9 = 64 per head (GPT-2 标准) |
| n_embd | 576 | Shakespeare(384) 与 GPT2-small(768) 之间 |
| MLP 扩展比 | 4× | c_fc: 576→2304 |
| 激活函数 | GELU (tanh approx) | |
| dropout | 0.2 | residual dropout（Zhou et al.: 多 epoch 最有效正则化） |
| 架构 | **Dense** | Krajewski(2024): <100M 参数级 MoE 无优势 |
| Linear bias | False | 节省 ~10M 参数，降低过拟合风险 |
| weight tying | ✅ | lm_head 与 wte 共享权重 |
| 参数量 | **~48M** | 参数:token 比 ≈ 1.5:1 |

## 训练超参数

| 参数 | 值 | 说明 |
|:---|:---|:---|
| total_batch_size | 524,288 tokens | 每次参数更新消耗的 token 数 |
| B (micro batch) | 64 | 单步 GPU batch size |
| T (seq len) | 256 | 序列长度 = block_size |
| grad_accum_steps | 32 | 梯度累积步数 |
| max_lr | 8e-4 | 小模型更高 LR (Shakespeare 用 1e-3) |
| min_lr | 8e-5 | 学习率终值 (= max_lr × 0.1) |
| warmup_steps | 100 | 线性 warmup（5% of max_steps） |
| max_steps | 2,000 | 总训练步数（~32 epochs） |
| lr schedule | cosine decay | warmup → cosine → min_lr |
| optimizer | AdamW | β₁=0.9, **β₂=0.99**（小 batch 更稳定） |
| weight_decay | 0.2 | 仅对 2D+ 参数 (Linear.weight, Embedding) |
| grad_clip | 1.0 | 梯度裁剪 max_norm |
| mixed precision | bfloat16 | torch.autocast |
| matmul precision | high | TF32 加速 (Ampere+ GPU) |
| **torch.compile** | **✅ 开启** | Dense 模型可用，训练加速 30-50% |
| early stopping | patience=5 | warmup 后，val loss 连续 5 次不降则停 |
| checkpoint | val-best | 仅 val loss 改善时保存 |
| random seed | 114 | |

### epoch 计算

```
epochs = max_steps × total_batch_size / total_tokens
       = 2000 × 524288 / 32729002
       ≈ 32 epochs
```

> 最多 32 epochs，配合 early stopping (patience=5) 通常在 15~25 epochs 自动停下。
> Dense 架构 + torch.compile 训练加速 30-50%，RTX 4060 8GB 显存占用 <2GB。

## 词表设计

| ID | 符号 | 用途 |
|:---|:---|:---|
| 0 | [PAD] | 填充 (保留，当前未使用) |
| 1 | [BOS] | 句首 (保留，当前未使用) |
| 2 | [EOS] | 句尾/文档分隔符 |
| 3 | [POEM] | 诗词领域标签 |
| 4 | [CLASSICAL] | 古典文领域标签 |
| 5 | [MODERN] | 现代文领域标签（当前未启用） |
| 6 | [UNK] | 未知字符 |
| 7~13,311 | 汉字 | 按频率降序，共 13,305 个 |

实际字符总数 16,931，超出容量的 3,626 个低频字被截断。

## 数据概览

### 实际统计 (`python inspect_data.py` 输出)

| 领域 | token 数 | token 占比 |
|:---|---:|---:|
| [POEM] 诗词 | 22,959,948 | 80.0% |
| [CLASSICAL] 古典文 | 5,742,740 | 20.0% |
| [MODERN] 现代文 | 0 | 0%（默认关闭，改 `RATIO_MODERN` 启用） |
| **合计** | **29,456,101** | |

### 分词后

| 指标 | 值 |
|:---|:---|
| 总 token 数 | 32,729,002 |
| 训练集 | 29,456,101 tokens (90%) |
| 验证集 | 3,272,901 tokens (10%) |
| 词表实际汉字 | 13,305 |
| UNK 率 (采样) | 0.002% |

### 序列长度分布 (EOS 间隔)

| 区间 | 占比 |
|:---|---:|
| 0-20 | 2.5% |
| 20-50 | 33.3% |
| 50-100 | 47.8% |
| 100-200 | 9.3% |
| 200+ | 7.2% |
| 平均 | 77 字 |
| 中位 | 50 字 |

### 数据来源

| 领域 | 来源 |
|:---|:---|
| 诗词 | [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) — 全唐诗、宋词、元曲、诗经等 |
| 古典文 | chinese-poetry (论语/四书五经/幽梦影/蒙学) + [xmj2002/Chinese_modern_classical](https://huggingface.co/datasets/xmj2002/Chinese_modern_classical) (二十四史等) |
| 现代文 | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 20231101.zh（可选，设 `RATIO_MODERN=0.15` 启用） |

## 快速开始

### 1. 环境
```bash
pip install torch numpy zhconv datasets
```

### 2. 数据准备
```bash
# 完整运行 (首次约需 2 分钟)
python prepare_poetry_data.py

# 只看统计，不生成
python prepare_poetry_data.py --dry-run

# 检查数据分布
python inspect_data.py
```

### 3. 数据格式
```
[POEM]床前明月光疑是地上霜[EOS]
[CLASSICAL]子曰学而时习之不亦说乎[EOS]
[MODERN]人工智能是计算机科学的一个分支[EOS]
```

### 4. 训练
在 VS Code 中打开 `train.ipynb`，按顺序执行所有 cell。

## 训练数据流

```
D:\chinese-poetry/*.json          HuggingFace (古典文/现代文)
       │                                    │
       ├─ extract_texts()                   ├─ load_dataset()
       ├─ 繁转简 (zhconv)                    ├─ 随机采样
       ├─ 按目录打标签 [POEM]/[CLASSICAL]    └─ 打标签 [CLASSICAL]/[MODERN]
       │                                    │
       └────────────┬───────────────────────┘
                    ▼
          加 [EOS] → 合并 → 打乱
                    │
                    ▼
          CharTokenizer.build_vocab()
          CharTokenizer.encode()
                    │
                    ▼
           np.save() → poetry_data/*.npy
```

## 与 nanoGPT 的区别

| | nanoGPT | Poetry-Nano |
|:---|:---|:---|
| 分词器 | tiktoken BPE (50,257) | Char-level (13,312) |
| 1 个汉字 | 1~N 个 token | **严格 1 token** |
| 上下文 | 1024 | 256 |
| 层数/头数/维度 | 12/12/768 | **10/9/576**（数据受限优化） |
| 训练数据 | FineWeb 英文 | 诗词 + 古文 + 维基百科(可选) |
| 特殊 token | 1 (`<\|endoftext\|>`) | 7 (含领域标签) |
| 架构 | Dense | **Dense**（Krajewski 验证小模型 MoE 无优势） |
| torch.compile | ✅ | ✅（Dense 可用） |
| 参数 | 124M (GPT-2) | **~48M** |
| 混合精度 | bfloat16 | bfloat16 (TF32 matmul) |

## 配置配比

在 `prepare_poetry_data.py` 顶部修改：

```python
RATIO_POEM = 0.65       # 诗词
RATIO_CLASSICAL = 0.20   # 古典文
RATIO_MODERN = 0.15      # 现代文（需联网下载 Wikipedia）
```

三者之和必须 = 1.0。设为 0 则跳过该数据源。

## 许可

MIT
