"""
诗词数据预处理流水线
====================
三类数据 → 繁转简 → 加领域标签 → CharTokenizer 编码 → .npy shards

数据来源:
  诗词 (80%): D:\chinese-poetry JSON 文件
  古典文 (20%): chinese-poetry 中的论语/四书五经/幽梦影/蒙学
              + HuggingFace xmj2002/Chinese_modern_classical (二十四史等)

用法:
  python prepare_poetry_data.py            # 完整运行
  python prepare_poetry_data.py --dry-run  # 只看统计
"""

import json
import os
import sys
import random
import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import zhconv

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from char_tokenizer import CharTokenizer

# ============================================================
# 配置
# ============================================================
POETRY_ROOT = Path(r"D:\chinese-poetry")
OUTPUT_DIR = SCRIPT_DIR / "poetry_data"
VOCAB_PATH = SCRIPT_DIR / "vocab.json"

VOCAB_SIZE = 13312
TRAIN_RATIO = 0.9
TOKENS_PER_SHARD = 2_000_000
SEED = 42

# 数据配比 (诗词为锚，其余补齐到目标比例)
# 设为 0 则跳过该数据源
RATIO_POEM = 0.80
RATIO_CLASSICAL = 0.20
RATIO_MODERN = 0.00  # Wikipedia 现代文，0=跳过

# 目录 → 默认标签
DIR_TO_TAG = {
    '全唐诗': '[POEM]', '宋词': '[POEM]', '元曲': '[POEM]',
    '五代诗词': '[POEM]', '诗经': '[POEM]', '纳兰性德': '[POEM]',
    '曹操诗集': '[POEM]', '楚辞': '[POEM]', '水墨唐诗': '[POEM]',
    '御定全唐詩': '[POEM]',
    '论语': '[CLASSICAL]', '四书五经': '[CLASSICAL]',
    '幽梦影': '[CLASSICAL]', '蒙学': '[CLASSICAL]',
}
EXCLUDE_DIRS = {'loader', 'rank', 'strains', 'images'}
EXCLUDE_NAMES = {'intro.json', 'authors.json', 'author.song.json',
                 'authors.tang.json', '表面结构字.json', 'README.md'}

# 蒙学目录中实际是诗词的文件 (应标 [POEM])
MENGXUE_POEM_FILES = {'qianjiashi.json', 'shenglvqimeng.json', 'tangshisanbaishou.json'}


# ============================================================
# 文本提取
# ============================================================

def extract_texts(item):
    """递归提取 JSON 中的诗文文本。"""
    if isinstance(item, str):
        return [item]
    if isinstance(item, list):
        result = []
        for x in item:
            result.extend(extract_texts(x))
        return result
    if isinstance(item, dict):
        for key in ('paragraphs', 'content', 'para'):
            if key in item:
                lines = extract_texts(item[key])
                if lines:
                    return [''.join(lines)]
        for key in item:
            if key in ('paragraphs', 'content', 'para'):
                continue
            result = extract_texts(item[key])
            if result:
                return result
    return []


# ============================================================
# 数据源 1: 诗词 (本地 JSON)
# ============================================================

def load_poetry() -> list[str]:
    """从 chinese-poetry 提取诗词正文。蒙学中的唐诗/千家诗/声律启蒙也归入此类。"""
    print("[1/3] 加载诗词...")
    texts = []
    for f in POETRY_ROOT.glob("**/*.json"):
        top = f.relative_to(POETRY_ROOT).parts[0]
        if top in EXCLUDE_DIRS or f.name in EXCLUDE_NAMES or 'error' in f.parts:
            continue

        # 判断标签
        tag = DIR_TO_TAG.get(top, '[POEM]')
        if f.name in MENGXUE_POEM_FILES:
            tag = '[POEM]'

        if tag != '[POEM]':
            continue

        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception:
            continue
        texts.extend(t for t in extract_texts(data) if t and t.strip())

    texts = [zhconv.convert(t, 'zh-cn') for t in texts]
    print(f"  诗词: {len(texts):,} 首, {sum(len(t) for t in texts):,} 字符")
    return texts


# ============================================================
# 数据源 2a: 古典文 (本地 JSON)
# ============================================================

def load_classical_local() -> list[str]:
    """从 chinese-poetry 提取古典文 (论语/四书五经/幽梦影/蒙学中非诗词部分)。"""
    print("[2/2] 加载古典文 (本地)...")
    texts = []
    for f in POETRY_ROOT.glob("**/*.json"):
        top = f.relative_to(POETRY_ROOT).parts[0]
        if top in EXCLUDE_DIRS or f.name in EXCLUDE_NAMES or 'error' in f.parts:
            continue

        tag = DIR_TO_TAG.get(top, '[POEM]')
        # 蒙学中的诗词文件跳过
        if f.name in MENGXUE_POEM_FILES:
            continue

        if tag != '[CLASSICAL]':
            continue

        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception:
            continue
        texts.extend(t for t in extract_texts(data) if t and t.strip())

    texts = [zhconv.convert(t, 'zh-cn') for t in texts]
    print(f"  古典文(本地): {len(texts):,} 篇, {sum(len(t) for t in texts):,} 字符")
    return texts


# ============================================================
# 数据源 2b: 古典文 (HuggingFace 二十四史等)
# ============================================================

def load_classical_hf(target_chars: int) -> list[str]:
    """
    从 xmj2002/Chinese_modern_classical 提取古文原文。
    按来源分组 → 拼接同源句 → 切分成 200~500 字的段。
    """
    from datasets import load_dataset

    print(f"  从 HuggingFace 加载古典文 (xmj2002/Chinese_modern_classical)...")
    ds = load_dataset("xmj2002/Chinese_modern_classical", split="train")
    print(f"  共 {len(ds):,} 句，正在按来源分组...")

    # 按 info 分组，保持原始顺序
    groups = OrderedDict()
    for row in ds:
        info = row["info"]
        groups.setdefault(info, []).append(row["classical"])

    print(f"  共 {len(groups):,} 个来源")

    # 拼接 + 切分
    texts, total = [], 0
    random.seed(SEED)
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    for key in group_keys:
        sentences = groups[key]
        # 过滤过短句
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 4]
        if not sentences:
            continue
        full = "".join(sentences)
        if len(full) < 50:
            continue

        # 切分成 200~500 字的小段
        chunk_size = random.randint(200, 500)
        for i in range(0, len(full), chunk_size):
            chunk = full[i:i + chunk_size]
            if len(chunk) < 50:
                continue
            texts.append(chunk)
            total += len(chunk)
            if total >= target_chars:
                break
        if total >= target_chars:
            break

    texts = [zhconv.convert(t, 'zh-cn') for t in texts]
    print(f"  古典文(HF): {len(texts):,} 篇, {total:,} 字符\n")
    return texts



# ============================================================
# 数据源 3: 现代文 (Wikipedia 中文)
# ============================================================

def load_modern_wikipedia(target_chars: int) -> list[str]:
    """
    从 wikimedia/wikipedia 20231101.zh 随机采样现代中文文本。
    按目标字符数截断。
    """
    from datasets import load_dataset

    print(f"  从 HuggingFace 加载现代文 (wikimedia/wikipedia 20231101.zh)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")
    print(f"  共 {len(ds):,} 篇")

    indices = list(range(len(ds)))
    random.seed(SEED)
    random.shuffle(indices)

    texts, total = [], 0
    for idx in indices:
        text = ds[idx]["text"].strip()
        if len(text) < 50:
            continue
        texts.append(text)
        total += len(text)
        if total >= target_chars:
            break

    texts = [zhconv.convert(t, 'zh-cn') for t in texts]
    print(f"  现代文(Wikipedia): {len(texts):,} 篇, {total:,} 字符\n")
    return texts


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)

    # ==== 收集数据 ====
    poems = load_poetry()
    classical_local = load_classical_local()

    p_chars = sum(len(t) for t in poems)
    c_local_chars = sum(len(t) for t in classical_local)

    # 以诗词为锚，按配比计算各类数据需要多少字符
    assert RATIO_POEM > 0, "RATIO_POEM must be > 0"
    assert abs(RATIO_POEM + RATIO_CLASSICAL + RATIO_MODERN - 1.0) < 0.001, \
        f"Ratios must sum to 1.0, got {RATIO_POEM + RATIO_CLASSICAL + RATIO_MODERN}"

    total_target = int(p_chars / RATIO_POEM)
    classical_target = int(total_target * RATIO_CLASSICAL)
    modern_target = int(total_target * RATIO_MODERN)

    # 古典文
    classical_hf_needed = max(0, classical_target - c_local_chars)
    classical_hf = load_classical_hf(classical_hf_needed) if classical_hf_needed > 0 else []
    c_hf_chars = sum(len(t) for t in classical_hf)
    classical_all = classical_local + classical_hf
    c_chars = c_local_chars + c_hf_chars

    # 现代文 (Wikipedia)
    modern_all = load_modern_wikipedia(modern_target) if modern_target > 0 else []
    m_chars = sum(len(t) for t in modern_all)

    # ==== 统计 ====
    total = p_chars + c_chars + m_chars
    print(f"\n{'='*55}")
    print(f"数据统计 (繁转简后)")
    print(f"{'='*55}")
    print(f"  [POEM]      {len(poems):>8,} 首  {p_chars:>12,} 字符  ({p_chars/total*100:.1f}%)")
    if c_chars > 0:
        print(f"  [CLASSICAL] {len(classical_all):>8,} 篇  {c_chars:>12,} 字符  ({c_chars/total*100:.1f}%)")
    if m_chars > 0:
        print(f"  [MODERN]    {len(modern_all):>8,} 篇  {m_chars:>12,} 字符  ({m_chars/total*100:.1f}%)")
    print(f"  总计                    {total:>12,} 字符")

    if args.dry_run:
        print("\n[dry-run] 跳过 .npy 生成。")
        return

    # ==== 加标签 & 打乱 ====
    print(f"\n加标签 & 编码...")
    tagged = []
    tagged.extend(f"[POEM]{t}[EOS]" for t in poems if t.strip())
    tagged.extend(f"[CLASSICAL]{t}[EOS]" for t in classical_all if t.strip())
    tagged.extend(f"[MODERN]{t}[EOS]" for t in modern_all if t.strip())
    random.shuffle(tagged)

    # ==== 构建词表 ====
    clean = [t.replace('[POEM]', '').replace('[CLASSICAL]', '').replace('[MODERN]', '')
              .replace('[EOS]', '') for t in tagged]
    clean = [t for t in clean if t.strip()]

    tokenizer = CharTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(clean)
    tokenizer.save(str(VOCAB_PATH))

    # ==== 编码 ====
    all_ids = []
    for text in tagged:
        all_ids.extend(tokenizer.encode(text))
    all_ids = np.array(all_ids, dtype=np.uint16)
    print(f"总 token 数: {len(all_ids):,}")

    # ==== 切分 train/val ====
    split = int(len(all_ids) * TRAIN_RATIO)
    split = max(1024, min(split, len(all_ids) - 1024))
    train = all_ids[:split]
    val = all_ids[split:]
    print(f"训练集: {len(train):,} tokens ({len(train)/len(all_ids)*100:.1f}%)")
    print(f"验证集: {len(val):,} tokens ({len(val)/len(all_ids)*100:.1f}%)")

    # ==== 保存 shards ====
    OUTPUT_DIR.mkdir(exist_ok=True)

    def save_shards(tokens, prefix):
        n = (len(tokens) + TOKENS_PER_SHARD - 1) // TOKENS_PER_SHARD
        for i in range(n):
            s, e = i * TOKENS_PER_SHARD, min((i+1) * TOKENS_PER_SHARD, len(tokens))
            path = OUTPUT_DIR / f"{prefix}_{i:06d}.npy"
            np.save(path, tokens[s:e])
            print(f"  → {path.name} ({e-s:,} tokens)")

    print("\n训练集 shards:")
    save_shards(train, "poetry_train")
    print("\n验证集 shards:")
    save_shards(val, "poetry_val")

    # ==== 完成 ====
    train_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("poetry_train_*.npy")) / 1024**2
    val_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("poetry_val_*.npy")) / 1024**2
    print(f"\n{'='*55}")
    print("✅ 数据准备完成！")
    print(f"{'='*55}")
    print(f"  词表: {VOCAB_PATH} ({len(tokenizer.char_to_id)} 汉字)")
    print(f"  训练: {OUTPUT_DIR}/poetry_train_*.npy ({train_mb:.0f} MB)")
    print(f"  验证: {OUTPUT_DIR}/poetry_val_*.npy ({val_mb:.0f} MB)")
    print(f"  配比: {p_chars/total*100:.1f}% [POEM] / "
          f"{c_chars/total*100:.1f}% [CLASSICAL]"
          + (f" / {m_chars/total*100:.1f}% [MODERN]" if m_chars > 0 else ""))
    print(f"\n  → 下一步: python inspect_data.py")


if __name__ == '__main__':
    main()
