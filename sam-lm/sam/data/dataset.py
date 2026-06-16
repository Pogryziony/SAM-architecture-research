"""Tokenizer, torch Dataset, and KB tensor helpers for the SAM POC."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .synthetic_facts import TOKEN_RE, SPECIAL_TOKENS


def _tok(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


class Tokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv = {i: t for t, i in vocab.items()}
        self.pad = vocab["<pad>"]
        self.bos = vocab["<bos>"]
        self.eos = vocab["<eos>"]
        self.ans = vocab["<ans>"]
        self.q = vocab["<q>"]
        self.fact = vocab["<fact>"]
        self.unk = vocab["<unk>"]

    @classmethod
    def from_dir(cls, data_dir: str) -> "Tokenizer":
        with open(os.path.join(data_dir, "vocab.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(t, self.unk) for t in _tok(text)]

    def decode(self, ids: List[int]) -> str:
        toks = [self.inv.get(int(i), "<unk>") for i in ids
                if int(i) not in (self.pad, self.bos, self.eos)]
        return " ".join(toks)


def _slot_int(slot_str: str) -> int:
    # "slot_000123" -> 123
    return int(slot_str.split("_")[1])


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class QADataset(Dataset):
    """Closed-book (default) or open-book QA, plus a 'fact' knowledge-injection kind.

    kind:
      'qa'   -> question -> answer sequences (closed or open book)
      'fact' -> raw fact text sequences (used to let a dense model memorise the KB)
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer: Tokenizer,
        kind: str = "qa",
        open_book: bool = False,
        oracle_text: bool = False,
        max_seq_len: int = 128,
    ):
        self.tok = tokenizer
        self.kind = kind
        self.open_book = open_book
        self.oracle_text = oracle_text
        self.max_seq_len = max_seq_len
        if kind == "qa":
            self.examples = load_jsonl(os.path.join(data_dir, f"{split}.jsonl"))
        elif kind == "fact":
            self.examples = load_jsonl(os.path.join(data_dir, "kb.jsonl"))
        else:
            raise ValueError(kind)

    def __len__(self) -> int:
        return len(self.examples)

    # -- sequence builders ---------------------------------------------------
    def _qa_ids(self, ex: Dict):
        t = self.tok
        q_ids = t.encode(ex["question"])
        a_ids = t.encode(ex["answer"])
        if self.open_book:
            fact_ids: List[int] = []
            for fo in ex["facts"]:
                fact_ids += [t.fact] + t.encode(fo["text"])
            prompt = [t.bos] + fact_ids + [t.q] + q_ids + [t.ans]
        elif self.oracle_text:
            fact_ids: List[int] = []
            for fo in ex["facts"]:
                fact_ids += [t.fact] + t.encode(fo["text"])
            prompt = [t.bos] + fact_ids + [t.q] + q_ids + [t.ans]
        else:
            prompt = [t.bos] + q_ids + [t.ans]
        full = prompt + a_ids + [t.eos]
        full = full[: self.max_seq_len]
        prompt_len = min(len(prompt), len(full))
        return full, prompt_len, a_ids

    def _fact_ids(self, rec: Dict):
        t = self.tok
        seq = [t.bos, t.fact] + t.encode(rec["text"]) + [t.eos]
        seq = seq[: self.max_seq_len]
        return seq, 1  # mask only <bos>

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        if self.kind == "qa":
            full, prompt_len, a_ids = self._qa_ids(ex)
            required = [_slot_int(s) for s in ex["required_slots"]]
            addressable = [_slot_int(s) for s in ex.get("addressable_slots", [])]
            hops = int(ex["reasoning_hops"])
            task_type = ex["task_type"]
        else:  # fact (kb.jsonl stores slot_id as an int)
            full, prompt_len = self._fact_ids(ex)
            a_ids = []
            sid = ex["slot_id"]
            sid = sid if isinstance(sid, int) else _slot_int(sid)
            required = [sid]
            addressable = required
            hops = 0
            task_type = "fact"

        input_ids = torch.tensor(full, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": prompt_len,
            "required_slots": torch.tensor(required, dtype=torch.long),
            "addressable_slots": torch.tensor(addressable, dtype=torch.long),
            "answer_ids": torch.tensor(a_ids, dtype=torch.long),
            "hops": hops,
            "task_type": task_type,
        }


def collate_qa(batch: List[Dict], pad_id: int) -> Dict:
    maxlen = max(b["input_ids"].size(0) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    labels = torch.full((B, maxlen), -100, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.bool)
    max_req = max(1, max(b["required_slots"].numel() for b in batch))
    required = torch.full((B, max_req), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attn[i, :L] = True
        r = b["required_slots"]
        required[i, : r.numel()] = r
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
        "required_slots": required,
        "prompt_len": [b["prompt_len"] for b in batch],
        "hops": [b["hops"] for b in batch],
        "task_type": [b["task_type"] for b in batch],
    }


def build_kb_tensors(data_dir: str, total_slots: int, tokenizer: Tokenizer):
    """Return (slot_value_token [max(total_slots, num_live)] long, num_live).

    slot_value_token[s] = vocab id of slot s's ``embedding_target`` (object token),
    or -1 for dead/empty slots.
    """
    kb = load_jsonl(os.path.join(data_dir, "kb.jsonl"))
    num_live = len(kb)
    # Auto-size: use at least num_live slots so all KB records fit
    effective_slots = max(total_slots, num_live)
    slot_value_token = torch.full((effective_slots,), -1, dtype=torch.long)
    for rec in kb:
        sid = rec["slot_id"]
        # If sid >= effective_slots, skip (shouldn't happen with contiguous IDs)
        if sid >= effective_slots:
            continue
        target = rec["embedding_target"]
        ids = tokenizer.encode(target)
        slot_value_token[sid] = ids[0] if ids else tokenizer.unk
    return slot_value_token, num_live
