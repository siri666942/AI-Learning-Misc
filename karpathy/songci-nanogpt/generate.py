"""
宋词生成脚本
============
用法:
  python generate.py                          # 默认: [POEM] 前缀, 生成 4 首
  python generate.py --prompt "明月几时有"      # 自定义前缀
  python generate.py -n 10                     # 生成 10 首
  python generate.py --prompt "春风" -n 3       # 自定义前缀 + 数量
  python generate.py --prompt "大江东去" --temperature 0.9 --top-k 80
  python generate.py --help
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ============================================================
# 模型定义 (与 train.ipynb 完全一致)
# ============================================================

from dataclasses import dataclass
import math
import torch.nn as nn


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 13312
    n_layer: int = 10
    n_head: int = 9
    n_embd: int = 576
    dropout: float = 0.2
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.attn_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ============================================================
# 分词器
# ============================================================

from char_tokenizer import CharTokenizer


# ============================================================
# 核心逻辑
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# 特殊 token ID
EOS = 2
POEM = 3


def load_model(checkpoint_path: str, device: str):
    """加载模型和配置。"""
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    # 兼容旧 checkpoint: config 可能是 dict 不是 GPTConfig
    if isinstance(config, dict):
        cfg = GPTConfig(**config)
    else:
        cfg = config

    print(f"  配置: {cfg.n_layer}层 / {cfg.n_head}头 / {cfg.n_embd}维 "
          f"| block_size={cfg.block_size} | vocab={cfg.vocab_size}")

    model = GPT(cfg)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    step = checkpoint.get('step', '?')
    val_loss = checkpoint.get('val_loss', float('nan'))
    print(f"  step={step} | val_loss={val_loss:.4f}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model, cfg


@torch.no_grad()
def generate(
    model: GPT,
    enc: CharTokenizer,
    prompt: str,
    num_return_sequences: int = 4,
    max_new_tokens: int = 96,
    temperature: float = 0.5,
    top_k: int = 50,
    device: str = "cuda",
    seed: int = None,
) -> list[str]:
    """
    从 prompt 开始，自回归生成宋词。

    参数:
      prompt:     前缀字符串 (如 "明月几时有", 不含 [POEM] 标签)
      num_return_sequences: 生成几首
      max_new_tokens: 每首最多生成多少个 token (字)
      temperature: 温度，<1.0 更保守/工整，>1.0 更多样
      top_k:       top-k 采样
      device:      "cuda" 或 "cpu"
      seed:        随机种子，None 则每次不同
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # 构建带标签的完整 prompt
    # 训练数据格式: [POEM]诗句[EOS] 或 [CLASSICAL]文言[EOS]
    # 如果用户 prompt 以 [POEM] 或 [CLASSICAL] 开头，直接用；否则加 [POEM]
    if not any(prompt.startswith(tag) for tag in ['[POEM]', '[CLASSICAL]', '[MODERN]']):
        full_prompt = f"[POEM]{prompt}"
    else:
        full_prompt = prompt

    tokens = enc.encode(full_prompt)
    prompt_len = len(tokens)
    print(f"\n前缀: \"{full_prompt}\" → {prompt_len} tokens")

    if prompt_len >= model.config.block_size:
        print(f"⚠ 前缀长度 {prompt_len} 超过 block_size {model.config.block_size}，截断尾部")
        tokens = tokens[:model.config.block_size]
        prompt_len = len(tokens)

    x = torch.tensor(tokens, dtype=torch.long, device=device)
    x = x.unsqueeze(0).repeat(num_return_sequences, 1)

    # 生成
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    finished = torch.zeros(num_return_sequences, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        if finished.all():
            break

        # 截断上下文到 block_size
        x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x_cond)
        logits = logits[:, -1, :] / temperature  # 温度缩放
        probs = F.softmax(logits, dim=-1)

        # top-k 采样
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=rng)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

        # 标记已生成 EOS 的序列
        finished = finished | (xcol.squeeze(-1) == EOS)

    # 解码
    results = []
    for i in range(num_return_sequences):
        # 只取新生成的部分 (去 prompt)，到第一个 EOS 为止
        ids = x[i, prompt_len:].tolist()
        if EOS in ids:
            ids = ids[:ids.index(EOS)]
        text = enc.decode(ids)
        results.append(text)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="宋词生成器 — 基于 nanoGPT 字符级模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate.py
  python generate.py --prompt "明月几时有"
  python generate.py -n 6 --temperature 0.7
  python generate.py --prompt "大江东去" --top-k 80 --max-tokens 128
  python generate.py --device cpu
        """
    )
    parser.add_argument(
        "--prompt", "-p", type=str, default="",
        help="生成前缀 (不含标签，脚本会自动加 [POEM])"
    )
    parser.add_argument(
        "--num", "-n", type=int, default=4,
        help="生成几首 (默认: 4)"
    )
    parser.add_argument(
        "--max-tokens", "-m", type=int, default=96,
        help="每首最多生成字数 (默认: 96)"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.5,
        help="温度，<1 保守工整，>1 多样 (默认: 0.5)"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=50,
        help="top-k 采样 (默认: 50)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None,
        help="随机种子 (默认: 随机)"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备 (默认: cuda 或 cpu)"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str,
        default=str(SCRIPT_DIR / "log" / "model_best.pt"),
        help="模型 checkpoint 路径"
    )
    parser.add_argument(
        "--vocab", "-v", type=str,
        default=str(SCRIPT_DIR / "vocab.json"),
        help="词表路径"
    )
    args = parser.parse_args()

    # ---- 检查 ----
    if not Path(args.checkpoint).exists():
        print(f"❌ checkpoint 不存在: {args.checkpoint}")
        print(f"   请确保模型已训练并保存在 log/model_best.pt")
        sys.exit(1)

    if not Path(args.vocab).exists():
        print(f"❌ 词表不存在: {args.vocab}")
        sys.exit(1)

    # ---- 加载 ----
    device = args.device
    model, cfg = load_model(args.checkpoint, device)

    enc = CharTokenizer(vocab_size=cfg.vocab_size)
    enc.load(args.vocab)

    # ---- 生成 ----
    prompt = args.prompt or ""  # 空字符串 → 纯 [POEM] 开头
    poems = generate(
        model=model,
        enc=enc,
        prompt=prompt,
        num_return_sequences=args.num,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        seed=args.seed,
    )

    # ---- 输出 ----
    print()
    print("=" * 55)
    for i, poem in enumerate(poems, 1):
        print(f"  [{i}]  {poem}")
        print()
    print("=" * 55)
    print(f"温度={args.temperature} | top_k={args.top_k} | "
          f"max_tokens={args.max_tokens} | 种子={args.seed}")


if __name__ == '__main__':
    main()
