"""
字级分词器 (Character-level Tokenizer)
======================================
1 Token = 1 汉字，专为中国古典诗词生成设计。

特殊 Token (id 0~6):
  [PAD]=0  [BOS]=1  [EOS]=2  [POEM]=3  [CLASSICAL]=4  [MODERN]=5  [UNK]=6

普通汉字从 id=7 开始，按频率降序分配。

接口对齐 tiktoken (encode / encode_ordinary / decode)，
训练脚本改 import 即可，其余调用不变。
"""

import json
import os
from collections import Counter


class CharTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    POEM = 3
    CLASSICAL = 4
    MODERN = 5
    UNK = 6
    NUM_SPECIAL = 7  # 普通汉字 id 从这里开始

    SPECIAL_TOKENS = {
        '[PAD]': 0, '[BOS]': 1, '[EOS]': 2,
        '[POEM]': 3, '[CLASSICAL]': 4, '[MODERN]': 5, '[UNK]': 6,
    }
    ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}

    def __init__(self, vocab_size: int = 13312):
        self.vocab_size = vocab_size
        self.max_chars = vocab_size - self.NUM_SPECIAL

        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._is_built = False

    # ============================================================
    # 构建 / 持久化
    # ============================================================

    def build_vocab(self, texts: list[str]) -> None:
        """扫描纯中文文本(不含标签)，按频率构建 char↔id 映射。"""
        counter = Counter()
        for text in texts:
            counter.update(text)

        print(f"[CharTokenizer] 扫描 {len(counter):,} 个不同字符 "
              f"(容量 {self.max_chars:,})")

        sorted_chars = [c for c, _ in counter.most_common()]

        if len(sorted_chars) > self.max_chars:
            dropped = len(sorted_chars) - self.max_chars
            print(f"[CharTokenizer] ⚠ 截断 {dropped:,} 个低频字符")
            sorted_chars = sorted_chars[:self.max_chars]

        self.char_to_id = {
            c: i + self.NUM_SPECIAL for i, c in enumerate(sorted_chars)
        }
        self.id_to_char = {
            i + self.NUM_SPECIAL: c for i, c in enumerate(sorted_chars)
        }

        # 统一映射表：特殊 token + 汉字
        self.token_to_id = dict(self.SPECIAL_TOKENS)
        self.token_to_id.update(self.char_to_id)
        self.id_to_token = dict(self.ID_TO_SPECIAL)
        self.id_to_token.update(self.id_to_char)

        self._is_built = True
        print(f"[CharTokenizer] 词表: {len(self.SPECIAL_TOKENS)} 特殊 + "
              f"{len(self.char_to_id):,} 汉字 = {len(self.token_to_id):,} total")

    def save(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'char_to_id': self.char_to_id,
            }, f, ensure_ascii=False, indent=2)
        print(f"[CharTokenizer] 词表 → {filepath}")

    def load(self, filepath: str) -> None:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.char_to_id = data['char_to_id']
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}

        self.token_to_id = dict(self.SPECIAL_TOKENS)
        self.token_to_id.update(self.char_to_id)
        self.id_to_token = dict(self.ID_TO_SPECIAL)
        self.id_to_token.update(self.id_to_char)

        self._is_built = True
        print(f"[CharTokenizer] 词表 ← {filepath} ({len(self.char_to_id)} 汉字)")

    # ============================================================
    # 核心接口
    # ============================================================

    def encode(self, text: str) -> list[int]:
        """字符串 → token id 列表。先匹配多字符标签，再逐字编码。"""
        assert self._is_built, "请先 build_vocab() 或 load()"
        ids = []
        i = 0
        while i < len(text):
            matched = False
            # 优先匹配特殊标签 (如 [POEM]), 按长度降序防止 [EOS] 被 [E 误匹配
            for tag in sorted(self.SPECIAL_TOKENS, key=len, reverse=True):
                if text.startswith(tag, i):
                    ids.append(self.SPECIAL_TOKENS[tag])
                    i += len(tag)
                    matched = True
                    break
            if matched:
                continue
            ids.append(self.char_to_id.get(text[i], self.UNK))
            i += 1
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """tiktoken 兼容别名。"""
        return self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """token id 列表 → 字符串。"""
        assert self._is_built, "请先 build_vocab() 或 load()"
        return ''.join(self.id_to_token.get(tid, '[UNK]') for tid in ids)

    def __repr__(self):
        if self._is_built:
            return (f"CharTokenizer(vocab={self.vocab_size}, "
                    f"chars={len(self.char_to_id):,})")
        return f"CharTokenizer(vocab={self.vocab_size}, <not built>)"


# ============================================================
# 自测
# ============================================================
if __name__ == '__main__':
    samples = [
        "床前明月光疑是地上霜",
        "举头望明月低头思故乡",
        "春眠不觉晓处处闻啼鸟",
    ]
    tk = CharTokenizer(13312)
    tk.build_vocab(samples)

    # 1) 纯中文
    t = "床前明月光"
    ids = tk.encode(t)
    dec = tk.decode(ids)
    assert t == dec, f"'{t}' != '{dec}'"
    assert len(ids) == len(t), f"char≠token: {len(t)} vs {len(ids)}"

    # 2) 带标签
    full = "[POEM]床前明月光疑是地上霜[EOS]"
    ids = tk.encode(full)
    dec = tk.decode(ids)
    assert full == dec, f"'{full}' != '{dec}'"
    assert len(ids) == 12, f"token数={len(ids)} != 12  ([POEM]+10字+[EOS])"

    # 3) 未知字符
    ids3 = tk.encode("床💡前")
    print(f"床💡前 → {ids3} → '{tk.decode(ids3)}'")

    # 4) 保存/加载
    tk.save("_test_vocab.json")
    t2 = CharTokenizer()
    t2.load("_test_vocab.json")
    assert t2.encode(full) == ids
    os.remove("_test_vocab.json")

    print("\n✅ 自测全部通过")
