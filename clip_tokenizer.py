"""Minimal CLIP byte-level BPE tokenizer (needed to turn text prompts into the
token ids that OWLv2's text encoder expects).

Written from the published CLIP algorithm (OpenAI CLIP simple_tokenizer):
  lower-case, split into words with a regex, map UTF-8 bytes to printable
  unicode, apply BPE merges with '</w>' marking the end of a word, wrap with
  <|startoftext|> ... <|endoftext|>, pad to 16 tokens.
Reads only two local files shipped with the model: vocab.json, merges.txt.

Differences from the reference (irrelevant for plain English prompts):
  * no `ftfy` text repair; html.unescape + whitespace cleanup only
  * Python `re` instead of the `regex` module, so \\p{L}/\\p{N} are
    approximated by [^\\W\\d_] and \\d
`ClipTokenizer.self_check()` verifies the ids of a known sentence.
"""

import html
import json
import os
import re

_PAT = re.compile(
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+|_+",
    re.IGNORECASE)


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _get_pairs(word):
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class ClipTokenizer:
    def __init__(self, vocab_json, merges_txt, special_tokens_map=None, max_length=16):
        with open(vocab_json, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        with open(merges_txt, "r", encoding="utf-8") as f:
            merges = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#version")]
        self.ranks = {tuple(m.split()): i for i, m in enumerate(merges) if len(m.split()) == 2}
        self.byte_encoder = _bytes_to_unicode()
        self.cache = {}
        self.bos = self.encoder["<|startoftext|>"]
        self.eos = self.encoder["<|endoftext|>"]
        pad_token = "!"  # OWL-ViT / OWLv2 tokenizers pad with "!" (id 0)
        if special_tokens_map and os.path.isfile(special_tokens_map):
            with open(special_tokens_map, "r", encoding="utf-8") as f:
                stm = json.load(f)
            p = stm.get("pad_token")
            pad_token = p.get("content", pad_token) if isinstance(p, dict) else (p or pad_token)
        self.pad = self.encoder.get(pad_token, 0)
        self.max_length = int(max_length)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if bigram not in self.ranks:
                break
            first, second = bigram
            new_word, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text):
        text = html.unescape(html.unescape(text)).strip()
        text = re.sub(r"\s+", " ", text).strip().lower()
        ids = []
        for tok in _PAT.findall(text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self.bpe(tok).split(" "))
        return ids

    def batch(self, texts):
        """Returns (input_ids, attention_mask) as lists of lists, length max_length."""
        all_ids, all_mask, truncated = [], [], []
        for t in texts:
            ids = [self.bos] + self.encode(t) + [self.eos]
            if len(ids) > self.max_length:
                truncated.append(t)
                ids = ids[:self.max_length - 1] + [self.eos]
            mask = [1] * len(ids) + [0] * (self.max_length - len(ids))
            ids = ids + [self.pad] * (self.max_length - len(ids))
            all_ids.append(ids)
            all_mask.append(mask)
        return all_ids, all_mask, truncated

    def self_check(self):
        """Known CLIP ids for 'a photo of a cat'."""
        expected = [49406, 320, 1125, 539, 320, 2368, 49407]
        got = [self.bos] + self.encode("a photo of a cat") + [self.eos]
        return got == expected, got
