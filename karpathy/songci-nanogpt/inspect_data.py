"""
数据分布检查工具
================
用法:
  python inspect_data.py                    # 检查 poetry_data/ 下所有 .npy
  python inspect_data.py --vocab vocab.json # 加载词表以显示汉字
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def load_tokenizer(vocab_path):
    if not Path(vocab_path).exists():
        return None
    from char_tokenizer import CharTokenizer
    tk = CharTokenizer()
    tk.load(vocab_path)
    return tk


def inspect(data_dir, tokenizer=None):
    npy_files = sorted(Path(data_dir).glob("*.npy"))
    if not npy_files:
        print(f"❌ {data_dir}/ 下无 .npy 文件")
        return

    train_files = [f for f in npy_files if "train" in f.name]
    val_files = [f for f in npy_files if "val" in f.name]

    total_tokens = sum(len(np.load(f)) for f in npy_files)
    train_tokens = sum(len(np.load(f)) for f in train_files)
    val_tokens = sum(len(np.load(f)) for f in val_files)

    # ======== 1. 总览 ========
    print("=" * 55)
    print("📊 总体统计")
    print("=" * 55)
    print(f"  训练集: {len(train_files)} 文件, {train_tokens:,} tokens "
          f"({train_tokens/total_tokens*100:.1f}%)")
    print(f"  验证集: {len(val_files)} 文件, {val_tokens:,} tokens "
          f"({val_tokens/total_tokens*100:.1f}%)")
    print(f"  总计:   {total_tokens:,} tokens  "
          f"(≈{total_tokens/total_tokens*1:.0f} epochs for {total_tokens//256} blocks)")

    # ======== 2. 领域分布 (按 token 数，非标签数) ========
    print(f"\n{'='*55}")
    print("🏷️  领域分布 (训练 token 占比)")
    print("=" * 55)
    TAG_ID = {3: '[POEM]', 4: '[CLASSICAL]', 5: '[MODERN]'}
    EOS = 2

    domain_tokens = {'[POEM]': 0, '[CLASSICAL]': 0, '[MODERN]': 0}
    # 统计每个标签到下一个标签/EOS 之间的 token 数
    for f in train_files:
        data = np.load(f)
        current_tag = None
        count = 0
        for tid in data:
            if tid in TAG_ID:
                if current_tag is not None:
                    domain_tokens[current_tag] += count
                current_tag = TAG_ID[tid]
                count = 0
            elif tid == EOS:
                if current_tag is not None:
                    domain_tokens[current_tag] += count
                current_tag = None
                count = 0
            elif current_tag is not None:
                count += 1

    total_domain = sum(domain_tokens.values())
    for name in ['[POEM]', '[CLASSICAL]', '[MODERN]']:
        c = domain_tokens[name]
        pct = c / total_domain * 100 if total_domain > 0 else 0
        bar = '█' * int(pct)
        print(f"  {name:12s} {c:>12,} tokens  ({pct:5.1f}%) {bar}")

    # ======== 3. Token ID 分布 ========
    print(f"\n{'='*55}")
    print("🔢 Token 统计 (采样 500K)")
    print("=" * 55)

    sample_size = 500_000
    id_counter = Counter()
    for f in npy_files:
        data = np.load(f)
        n_sample = min(len(data), sample_size // len(npy_files))
        idx = np.random.choice(len(data), n_sample, replace=False)
        id_counter.update(data[idx].tolist())

    print(f"  ID 范围: {min(id_counter.keys())} ~ {max(id_counter.keys())}")
    print(f"  不同 token: {len(id_counter):,}")

    print(f"\n  特殊 token:")
    for tid, name in [(0,'[PAD]'),(1,'[BOS]'),(2,'[EOS]'),(3,'[POEM]'),
                       (4,'[CLASSICAL]'),(5,'[MODERN](未用)'),(6,'[UNK]')]:
        c = id_counter.get(tid, 0)
        print(f"    id={tid} {name:12s}: {c:>8,} ({c/sample_size*100:.3f}%)")

    if tokenizer:
        print(f"\n  Top-20 汉字:")
        normal = {k: v for k, v in id_counter.items() if k >= 7}
        for tid, cnt in Counter(normal).most_common(20):
            char = tokenizer.decode([tid])
            print(f"    {char} (id={tid:>5}): {cnt:>8,}")

    # ======== 4. EOS 间隔分布 ========
    print(f"\n{'='*55}")
    print("📏 序列长度分布 (EOS-to-EOS)")
    print("=" * 55)
    EOS = 2
    lengths = []
    for f in train_files[:1]:
        data = np.load(f)
        pos = np.where(data == EOS)[0]
        if len(pos) > 1:
            lengths = np.diff(pos)

    if len(lengths) > 0:
        print(f"  样本数: {len(lengths):,}")
        print(f"  最短/最长/平均/中位: "
              f"{lengths.min()}/{lengths.max()}/{lengths.mean():.0f}/{np.median(lengths):.0f}")
        for lo, hi, label in [(0,20,'0-20'),(20,50,'20-50'),(50,100,'50-100'),
                               (100,200,'100-200'),(200,99999,'200+')]:
            c = ((lengths >= lo) & (lengths < hi)).sum()
            pct = c / len(lengths) * 100
            print(f"  {label:>8s}: {c:>6,} ({pct:5.1f}%) {'█'*int(pct)}")

    # ======== 5. Shard 列表 ========
    print(f"\n{'='*55}")
    print("📁 Shard 文件")
    print("=" * 55)
    for f in npy_files:
        data = np.load(f)
        tag = "🏋️" if "train" in f.name else "🧪"
        mb = f.stat().st_size / 1024**2
        print(f"  {tag} {f.name:40s} {len(data):>12,} tokens  {mb:6.1f} MB")

    print(f"\n✅ 检查完成")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(SCRIPT_DIR / "poetry_data"))
    parser.add_argument("--vocab", default=str(SCRIPT_DIR / "vocab.json"))
    args = parser.parse_args()
    inspect(args.data_dir, load_tokenizer(args.vocab))
