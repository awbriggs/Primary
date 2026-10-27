#!/usr/bin/env python3
"""Build empirical distributions from the Gutenberg Dialogue Dataset.

Input format (train.txt / dev.txt / test.txt): one utterance per line,
dialogues separated by a blank line.

Distributions written to --out as JSON, each with its Shannon entropy:
  utterance_length   tokens per utterance
  dialogue_length    utterances per dialogue
  unigram            token frequencies
  first_token        first token of each utterance
  bigram             P(next token | token), top --top-k successors per token
  char               character frequencies

Optionally samples dialogues from the bigram model with --sample N.

Usage:
  python dialogue_dists.py train.txt --out dists --max-dialogues 200000
  python dialogue_dists.py train.txt --out dists --sample 5 --seed 1
"""
import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")
BOS, EOS = "<s>", "</s>"


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def read_dialogues(path, max_dialogues=None):
    dialogue = []
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dialogue.append(line)
                continue
            if dialogue:
                yield dialogue
                dialogue = []
                n += 1
                if max_dialogues and n >= max_dialogues:
                    return
    if dialogue:
        yield dialogue


def entropy_bits(counter):
    total = sum(counter.values())
    return -sum(c / total * math.log2(c / total) for c in counter.values() if c)


def normalize(counter):
    total = sum(counter.values())
    return {str(k): v / total for k, v in sorted(counter.items(), key=lambda kv: -kv[1])}


def build(path, max_dialogues):
    utt_len, dlg_len = Counter(), Counter()
    unigram, first, chars = Counter(), Counter(), Counter()
    bigram = defaultdict(Counter)

    for dialogue in read_dialogues(path, max_dialogues):
        dlg_len[len(dialogue)] += 1
        for utt in dialogue:
            toks = tokenize(utt)
            if not toks:
                continue
            utt_len[len(toks)] += 1
            first[toks[0]] += 1
            unigram.update(toks)
            chars.update(utt.lower())
            seq = [BOS] + toks + [EOS]
            for a, b in zip(seq, seq[1:]):
                bigram[a][b] += 1
    return utt_len, dlg_len, unigram, first, chars, bigram


def write_json(out_dir, name, counter, extra=None):
    doc = {
        "support_size": len(counter),
        "total": sum(counter.values()),
        "entropy_bits": entropy_bits(counter),
        "probs": normalize(counter),
    }
    if extra:
        doc.update(extra)
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return doc["entropy_bits"], doc["support_size"]


def write_bigram(out_dir, bigram, top_k):
    table, ent_weighted, total = {}, 0.0, 0
    for tok, nxt in bigram.items():
        n = sum(nxt.values())
        ent_weighted += n * entropy_bits(nxt)
        total += n
        table[tok] = {
            "count": n,
            "entropy_bits": entropy_bits(nxt),
            "next": {k: v / n for k, v in nxt.most_common(top_k)},
        }
    cond_entropy = ent_weighted / total
    with open(os.path.join(out_dir, "bigram.json"), "w", encoding="utf-8") as f:
        json.dump({"conditional_entropy_bits": cond_entropy, "table": table}, f)
    return cond_entropy, len(table)


def sample_dialogues(bigram, dlg_len, n, rng, max_tokens=60):
    def pick(counter):
        keys = list(counter)
        return rng.choices(keys, weights=[counter[k] for k in keys])[0]

    out = []
    for _ in range(n):
        utts = []
        for _ in range(pick(dlg_len)):
            tok, words = BOS, []
            while len(words) < max_tokens:
                tok = pick(bigram[tok])
                if tok == EOS:
                    break
                words.append(tok)
            utts.append(" ".join(words))
        out.append(utts)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="path to train.txt (or dev/test)")
    ap.add_argument("--out", default="dists", help="output directory")
    ap.add_argument("--max-dialogues", type=int, default=None, help="stop after N dialogues")
    ap.add_argument("--top-k", type=int, default=50, help="successors kept per token in bigram.json")
    ap.add_argument("--sample", type=int, default=0, help="print N dialogues sampled from the bigram model")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    utt_len, dlg_len, unigram, first, chars, bigram = build(args.input, args.max_dialogues)
    if not dlg_len:
        raise SystemExit(f"no dialogues found in {args.input}")

    for name, counter in [
        ("utterance_length", utt_len),
        ("dialogue_length", dlg_len),
        ("unigram", unigram),
        ("first_token", first),
        ("char", chars),
    ]:
        h, k = write_json(args.out, name, counter)
        print(f"{name:18s} support={k:>9,d}  entropy={h:6.3f} bits")
    h, k = write_bigram(args.out, bigram, args.top_k)
    print(f"{'bigram':18s} contexts={k:>9,d}  H(next|tok)={h:6.3f} bits")

    if args.sample:
        rng = random.Random(args.seed)
        for i, d in enumerate(sample_dialogues(bigram, dlg_len, args.sample, rng), 1):
            print(f"\n--- sample {i} ---")
            for j, u in enumerate(d):
                print(f"{'AB'[j % 2]}: {u}")


if __name__ == "__main__":
    main()
