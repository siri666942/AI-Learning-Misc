# 知识地图：训练一个小型语言模型涉及的学科板块

## 七个板块全景

```
┌─────────────────────────────────────────────────────────────┐
│                    训练一个小型 LM                            │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Scaling  │ │ 模型架构  │ │ 分词与   │ │ 优化与   │       │
│  │  Laws    │ │   设计    │ │ 数据处理  │ │ 正则化   │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│       │             │            │            │             │
│       └─────────────┼────────────┼────────────┘             │
│                     │            │                          │
│       ┌─────────────┼────────────┼────────────┐             │
│       │             │            │            │             │
│  ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐       │
│  │  GPU硬件 │ │ 实验工程 │ │ 学术研究 │ │          │       │
│  │  与系统  │ │ 与调试   │ │   方法   │ │   ...    │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 板块一：Scaling Laws（规模规律）

**属于**：机器学习理论 / 经验科学

核心问题：**多大模型配多少数据，多少算力，训练多久。**

| 本项目涉及的子话题 | 对应知识点 |
|-------------------|-----------|
| 306 epochs 为什么太多了 | Data-constrained scaling law（Muennighoff et al. 2023） |
| 48M 参数配 32M tokens 是否合理 | Chinchilla 最优比例 20:1（Hoffmann et al. 2022） |
| 为什么数据不足时仍要多训几个 epoch | 有效数据公式 D' = U + U·R*(1-e^(-R/R*)) |
| 什么时候过拟合从"可接受"变成"有害" | 参数 > 唯一 token 数时，>10 epoch 开始退化 |
| 不同 epoch 数收益递减的具体数字 | ≤4 epoch 无损，4-16 递减，>16 无效 |

**前置知识**：概率统计、信息论基础、损失函数

---

## 板块二：模型架构设计（Model Architecture）

**属于**：深度学习 / NLP 模型设计

核心问题：**Transformer 的各个维度怎么定。**

| 本项目涉及的子话题 | 对应知识点 |
|-------------------|-----------|
| n_layer / n_head / n_embd 怎么选 | Transformer 架构原理（Vaswani et al. 2017） |
| n_embd / n_head = 64 是标准 | GPT-2 架构约定（Radford et al. 2019） |
| MoE 在什么规模有意义 | Fine-Grained MoE Scaling Law（Krajewski et al. 2024） |
| Weight Tying 的作用 | Press & Wolf (2017) — 输入输出 embedding 共享 |
| MLP 扩展比 4× vs 8/3× | GELU vs SwiGLU 激活函数差异（Shazeer 2020） |
| Linear bias 能不能省 | 参数分解：bias 对梯度流和 weight decay 的影响 |
| 参数量手算公式 | 每层 ≈ 12d²(attn) + 8d²(MLP) + embedding |

**前置知识**：线性代数、注意力机制、前馈网络

---

## 板块三：分词与数据处理（Tokenization & Data）

**属于**：自然语言处理 / 数据工程

核心问题：**文本怎么表示成数字，不同表示法有什么优劣。**

| 本项目涉及的子话题 | 对应知识点 |
|-------------------|-----------|
| Char-level 为什么适合诗词 | 中文 NLP 分词方案对比（字/词/子词） |
| BPE 的原理和中文适配问题 | Subword tokenization（Sennrich et al. 2016） |
| Char-level 的 OOV 优势 | Sun & Hewitt (2023) — Chinese Backpack LMs |
| 词表大小怎么设计 | 对齐 CUDA 友好维度（128 倍数触发 Tensor Core） |
| 领域标签 `[POEM]` 等怎么用 | CTRL-style domain control tokens |
| 数据配比 80:20 怎么定 | 域适应、数据混合策略 |

**前置知识**：中文语言学基础、编码/解码、数据处理 pipeline

---

## 板块四：优化与正则化（Optimization & Regularization）

**属于**：优化理论 / 深度学习训练技巧

核心问题：**怎么让模型快且稳地收敛，且不背答案。**

| 本项目涉及的子话题 | 对应知识点 |
|-------------------|-----------|
| AdamW 的 β₁/β₂ 怎么选 | 自适应优化器原理（Kingma & Ba 2014, Loshchilov & Hutter 2019） |
| 小模型用小 β₂ | nanoGPT 社区实践：β₂=0.99 对 batch token 少更稳定 |
| Weight Decay 只对 2D 参数 | 参数分组衰减策略（GPT-2/3 标准） |
| Dropout 放哪里、放多少 | 多 epoch 训练最优正则化（Zhou et al. 2024） |
| Cosine warmup + decay | 学习率调度理论 |
| 小模型更高 LR | LR 随模型规模的 scaling 经验：1e-3(14M)→6e-4(124M)→3e-4(1B+) |
| Gradient Clipping | 梯度爆炸的预防，max_norm=1.0 是 GPT 标准 |

**前置知识**：梯度下降、动量、L1/L2 正则化

---

## 板块五：GPU 硬件与系统（Hardware & Systems）

**属于**：计算机体系结构 / 系统性能

核心问题：**GPU 内部发生了什么，显存怎么管。**

| 本项目涉及的子话题 | 对应知识点 |
|-------------------|-----------|
| 100%利用率 但 34W | GPU 架构：CUDA Cores vs Tensor Cores 功耗差异 |
| Memory-bound vs Compute-bound | 算术强度（FLOPs/byte）决定瓶颈在哪 |
| Ada Lovelace 架构特点 | RTX 40 系：Tensor Cores FP8/bf16、更大的 L2 cache |
| 显存布局 | 权重 + 梯度 + 优化器状态 + 激活 + 临时张量 |
| WDDM 换页 | Windows GPU 虚拟内存机制，>90% VRAM 触发换页 |
| torch.compile 为什么 Windows 上不可用 | Triton 编译器只支持 Linux，Windows 缺工具链 |
| nvidia-smi 怎么读 | util / clocks / power.draw / power.limit 四个关键指标 |
| bf16 混合精度原理 | 前向 bf16（快省显存）、梯度 fp32（精度够）、权重 bf16 |

**前置知识**：GPU 计算模型、显存层次、CUDA 基础概念

---

## 板块六：实验工程与调试（ML Engineering & Debugging）

**属于**：软件工程 / MLOps / 实验方法

核心问题：**怎么确认训练在正确进行，出问题怎么定位。**

| 本项目涉及的子话题 | 对应知识点 |
|-------------------|-----------|
| 初始 loss 校验 | 随机 baseline：loss ≈ ln(vocab_size) |
| 过拟合诊断 | train loss↓ + val loss↑ = 过拟合 |
| Early Stopping 怎么设 | patience = N × eval_interval |
| benchmark 为什么不准 | 不能和训练进程抢 GPU，要模拟真实负载 |
| 代码改完先跑语法+初始化 | compile(code) + model = GPT(config) 快速验证 |
| 命名一致性 | Python 无编译期检查，dataclass 字段和 config.xxx 要一致 |
| log 设计 | 每步记录 step / train loss / val loss / lr，事后可画图 |

**前置知识**：Python 调试、shell 基础、实验记录习惯

---

## 板块七：学术研究方法（Research Methodology）

**属于**：科研方法 / 元技能

核心问题：**面对不确定的参数，怎么高效找到可靠答案。**

| 本项目涉及的子话题 | 对应知识点 |
|-------------------|-----------|
| 论文检索优先级 | 先 scaling law / 综述，再 benchmark / 消融实验，最后单点优化 |
| 论文 vs 社区经验 | 论文给边界（4 epoch 安全），社区给锚点（Shakespeare config） |
| 可复现的锚点 | GPT-2 的 64/head、nanoGPT 的 Shakespeare config |
| Trade-off 思维 | 速度 vs 显存（batch size）、容量 vs 过拟合（模型大小）、稳定性 vs 速度（MoE vs Dense） |
| 外推风险 | 175B 结论未必适用 50M，要验证或找针对性研究 |
| 复现 vs 创新决策 | 小项目先对齐已知最佳实践，再调一个变量 |

---

## 知识依赖关系

```
                    ┌──────────────────┐
                    │  学术研究方法 (7)  │  ← 元层，指导其他六个
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────┴─────┐        ┌────┴─────┐        ┌────┴─────┐
   │ ML 理论  │        │ 系统工程  │        │ 工程习惯 │
   ├──────────┤        ├──────────┤        ├──────────┤
   │ Scaling  │        │ GPU 硬件 │        │ 实验工程 │
   │  Laws(1) │        │ 与系统(5)│        │ 与调试(6)│
   ├──────────┤        └──────────┘        └──────────┘
   │ 模型架构  │
   │   设计(2) │
   ├──────────┤
   │ 分词处理  │
   │ 与数据(3) │
   ├──────────┤
   │ 优化正则  │
   │   化(4)   │
   └──────────┘
```

- **(1)-(4)** 是 ML 核心能力：理解模型和数据
- **(5)** 是工程底座：理解硬件
- **(6)** 是工程习惯：保证正确性
- **(7)** 是无技能：决定决策质量

---

## 如果想系统补某个板块

| 板块 | 入门资源 |
|------|---------|
| Scaling Laws | [Chinchilla 论文](https://arxiv.org/abs/2203.15556) + Muennighoff 的 [Data-Constrained](https://arxiv.org/abs/2305.16264) |
| 模型架构 | [GPT-2 论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) + Karpathy 的 [nanoGPT 视频](https://youtu.be/kCc8FmEb1nY) |
| 分词 | [BPE 论文](https://arxiv.org/abs/1508.07909) + tiktoken 源码 |
| 优化正则化 | [AdamW 论文](https://arxiv.org/abs/1711.05101) + [Zhou et al. 2024](https://arxiv.org/abs/2404.10102) |
| GPU 硬件 | [NVIDIA Ada 白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) + CUDA C++ Programming Guide |
| 实验工程 | Andrej Karpathy 的 [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) |
