"""Deterministic synthetic fact/question generator for the SAM POC.

Design goals (see docs/experiment-0.md):
  * Knowledge lives in a fixed set of FACTS. Each unique fact == one memory slot.
  * Questions compose facts in chains of 1-3 hops.
  * For multi-hop questions the *bridge* entity is intentionally NOT named in the
    question, so single-shot retrieval keyed on the question cannot reach the
    answer-bearing fact. This is what makes the multi-hop gate decisive: only
    chaining (oracle injection, or future adaptive re-query) can solve it.
  * Each fact's ``embedding_target`` is the OBJECT token of the fact. The answer
    of every example is a single vocabulary token, so accuracy is clean
    exact-match. For multi-hop the intermediate ("bridge") objects act as
    distractors the model must NOT emit.
  * Train/test entity separation: the noun roots used to build test entities are
    disjoint from train (``--entity-separation soft|hard|none``).

Every example matches the schema requested in the task:

    {
      "facts": [{"slot_id": "...", "text": "...", "embedding_target": "..."}],
      "question": "...",
      "answer": "...",
      "required_slots": ["slot_001", "slot_002"],
      "reasoning_hops": 2,
      "task_type": "two_hop_reasoning"
    }

plus a few extra diagnostic fields (split, answer_token, distractor_tokens,
addressable_slots).

CLI:
    python -m sam.data.synthetic_facts --output data/synthetic \
        --train 10000 --val 1000 --test 1000 --seed 42
    python -m sam.data.synthetic_facts --tiny     # 500/100/100 fast preset
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import OrderedDict
from typing import Dict, List, Tuple

TASK_TYPES = [
    "single_fact_recall",
    "two_hop_reasoning",
    "three_hop_reasoning",
    "api_usage_reasoning",
    "code_symbol_reasoning",
]

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<ans>", "<q>", "<fact>", "<unk>"]

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[?.,:;/()]")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------
VERBS = [
    "create", "fetch", "delete", "update", "build", "parse", "render", "submit",
    "validate", "resolve", "encode", "decode", "load", "store", "sync", "queue",
    "dispatch", "cancel", "refund", "archive", "import", "export", "verify", "lock",
]

NOUNS = [
    "Order", "User", "Payment", "Invoice", "Session", "Account", "Profile", "Cart",
    "Shipment", "Ticket", "Report", "Job", "Task", "Message", "Record", "Asset",
    "Policy", "Wallet", "Coupon", "Refund", "Device", "Batch", "Token", "Channel",
    "Vendor", "Catalog", "Booking", "Contract", "License", "Webhook",
]

VALUE_SUFFIXES = ["Id", "Ref", "Key", "Code", "Hash", "Uid"]
STATUS_WORDS = [
    "Conflict", "NotFound", "Timeout", "Forbidden", "Invalid",
    "ServerError", "RateLimited", "Unauthorized",
]


class NamePools:
    """Holds split-specific noun pools so train/test entities can be disjoint."""

    def __init__(self, separation: str):
        self.separation = separation
        n = len(NOUNS)
        cut = int(round(n * 0.7))
        self.train_nouns = NOUNS[:cut]
        self.test_nouns = NOUNS[cut:]
        if separation == "none":
            self.train_nouns = NOUNS
            self.test_nouns = NOUNS
        # verbs only split under "hard"
        if separation == "hard":
            vcut = int(round(len(VERBS) * 0.7))
            self.train_verbs = VERBS[:vcut]
            self.test_verbs = VERBS[vcut:]
        else:
            self.train_verbs = VERBS
            self.test_verbs = VERBS

    def nouns(self, split: str) -> List[str]:
        return self.test_nouns if split == "test" else self.train_nouns

    def verbs(self, split: str) -> List[str]:
        return self.test_verbs if split == "test" else self.train_verbs


# entity builders (each returns a single vocab token) --------------------------
def func_name(rng, verbs, noun) -> str:
    return rng.choice(verbs) + noun


def value_name(rng, noun) -> str:
    return noun + rng.choice(VALUE_SUFFIXES)


def api_name(noun) -> str:
    return "api" + noun


def endpoint_name(noun) -> str:
    return noun[0].lower() + noun[1:] + "Endpoint"


def header_name(noun) -> str:
    return noun + "Header"


def token_name(noun) -> str:
    return noun + "Token"


def error_name(noun) -> str:
    return noun + "Error"


def status_name(rng) -> str:
    return "Status" + rng.choice(STATUS_WORDS)


# ---------------------------------------------------------------------------
# KB: maps canonical fact text -> slot id (stable, shared across splits)
# ---------------------------------------------------------------------------
class KB:
    def __init__(self):
        self.text_to_id: "OrderedDict[str, int]" = OrderedDict()
        self.records: List[Dict] = []  # slot_id -> {text, embedding_target, subject}

    def add(self, text: str, subject: str, obj: str) -> int:
        if text in self.text_to_id:
            return self.text_to_id[text]
        sid = len(self.records)
        self.text_to_id[text] = sid
        self.records.append({"slot_id": sid, "text": text,
                             "embedding_target": obj, "subject": subject})
        return sid


# ---------------------------------------------------------------------------
# Task generators. Each returns:
#   facts: list of (text, subject, obj)
#   answer, hops, addressable_subjects
# ---------------------------------------------------------------------------
def gen_single(rng, pools, split):
    nouns, verbs = pools.nouns(split), pools.verbs(split)
    na = rng.choice(nouns)
    fn = func_name(rng, verbs, na)
    v = value_name(rng, na)
    facts = [(f"Function {fn} returns {v} .", fn, v)]
    return facts, v, 1, [fn]

def gen_two_hop(rng, pools, split):
    nouns, verbs = pools.nouns(split), pools.verbs(split)
    na, nb = rng.sample(nouns, 2)
    fn = func_name(rng, verbs, na)
    b = value_name(rng, na)
    v = value_name(rng, nb)
    facts = [
        (f"Function {fn} returns {b} .", fn, b),
        (f"Value {b} is wrapped as {v} .", b, v),
    ]
    return facts, v, 2, [fn]

def gen_three_hop(rng, pools, split):
    nouns, verbs = pools.nouns(split), pools.verbs(split)
    na, nb, nc = rng.sample(nouns, 3)
    api = api_name(na)
    x = na + "Id"
    y = nb + "Ref"
    z = nc + "Key"
    facts = [
        (f"API {api} returns {x} .", api, x),
        (f"Mapper converts {x} into {y} .", x, y),
        (f"Adapter converts {y} into {z} .", y, z),
    ]
    return facts, z, 3, [api]

def gen_api_usage(rng, pools, split):
    nouns = pools.nouns(split)
    na, nb, nc = rng.sample(nouns, 3)
    ep = endpoint_name(na)
    h = header_name(nb)
    tok = token_name(nc)
    facts = [
        (f"Endpoint {ep} requires header {h} .", ep, h),
        (f"Header {h} carries token {tok} .", h, tok),
    ]
    return facts, tok, 2, [ep]

def gen_code_symbol(rng, pools, split):
    nouns, verbs = pools.nouns(split), pools.verbs(split)
    na, nb = rng.sample(nouns, 2)
    fn = func_name(rng, verbs, na)
    err = error_name(nb)
    code = status_name(rng)
    facts = [
        (f"Function {fn} throws {err} on failure .", fn, err),
        (f"Error {err} maps to {code} .", err, code),
    ]
    return facts, code, 2, [fn]


GENERATORS = {
    "single_fact_recall": gen_single,
    "two_hop_reasoning": gen_two_hop,
    "three_hop_reasoning": gen_three_hop,
    "api_usage_reasoning": gen_api_usage,
    "code_symbol_reasoning": gen_code_symbol,
}


# ---------------------------------------------------------------------------
# Fact pool: pre-generate facts once, then reuse across many questions
# ---------------------------------------------------------------------------
def build_fact_pool(rng, pools, kb, split, pool_size: int = 800):
    """Pre-generate a pool of fact chains that will be reused across questions.
    
    Returns a list of (fact_objects, answer, hops, addressable_subjects) 
    ready to be composed into questions.
    """
    pool = []
    seen_fact_sets = set()
    attempts = 0
    nouns = pools.nouns(split)
    verbs = pools.verbs(split)
    
    while len(pool) < pool_size and attempts < pool_size * 20:
        attempts += 1
        task_type = TASK_TYPES[len(pool) % len(TASK_TYPES)]
        
        gen = GENERATORS[task_type]
        facts, answer, hops, addressable_subjects = gen(rng, pools, split)
        
        # Register facts in KB (but don't count as examples yet)
        fact_objs = []
        required_slots = []
        for text, subject, obj in facts:
            sid = kb.add(text, subject, obj)
            fact_objs.append({"slot_id": f"slot_{sid:06d}", "text": text,
                              "embedding_target": obj, "subject": subject})
            required_slots.append(f"slot_{sid:06d}")
        
        fact_key = tuple(f["text"] for f in fact_objs)
        if fact_key in seen_fact_sets:
            continue
        seen_fact_sets.add(fact_key)
        
        addressable = [fo["slot_id"] for fo in fact_objs
                       if fo["subject"] in addressable_subjects]
        
        pool.append({
            "fact_objs": fact_objs,
            "required_slots": required_slots,
            "addressable_slots": addressable,
            "answer": answer,
            "hops": hops,
            "addressable_subjects": addressable_subjects,
            "task_type": task_type,
        })
    
    return pool


def generate_split(rng, pools, kb, split, n) -> List[Dict]:
    """Generate examples by reusing facts from a fixed pool.
    
    Each fact gets reused across many questions, ensuring 10+ examples per slot.
    """
    # Build a fact pool (~800-1000 facts, ensuring ~20 examples per fact for 20K total)
    pool_size = min(n // 5, 4000)  # larger pool for more question variety
    pool = build_fact_pool(rng, pools, kb, split, pool_size)
    
    examples = []
    seen_signatures = set()
    attempts = 0
    max_attempts = n * 10  # more attempts for template coverage
    
    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        # Randomly pick a fact chain from the pool
        chain = pool[rng.randint(0, len(pool) - 1)]
        task_type = chain["task_type"]
        template_idx = rng.randint(0, 9)  # random template variant
        
        fact_objs = chain["fact_objs"]
        answer = chain["answer"]
        hops = chain["hops"]
        
        # Build a fact key for deduplication
        fact_key = tuple(fo["text"] for fo in fact_objs)
        
        question = _question_for(task_type,
                                 [(fo["text"], fo["subject"], fo["embedding_target"]) for fo in fact_objs],
                                 chain["addressable_subjects"],
                                 template_idx)
        
        distractors = [fo["embedding_target"] for fo in fact_objs[:-1]]
        
        # Lightweight deduplication: only skip exact (question, answer, fact_texts) repeats
        # This allows many template variants with slightly different wordings
        sig = (question, answer, fact_key)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        
        examples.append({
            "facts": fact_objs,
            "question": question,
            "answer": answer,
            "answer_token": answer,
            "required_slots": chain["required_slots"],
            "addressable_slots": chain["addressable_slots"],
            "distractor_tokens": distractors,
            "reasoning_hops": hops,
            "task_type": task_type,
            "split": split,
        })
    
    rng.shuffle(examples)
    return examples


def generate_split_from_pool(rng, pool, split, n) -> List[Dict]:
    """Generate examples by sampling from a pre-built fact pool."""
    examples = []
    seen = {}  # sig -> count
    attempts = 0
    while len(examples) < n and attempts < n * 20:
        attempts += 1
        chain = pool[rng.randint(0, len(pool) - 1)]
        task_type = chain["task_type"]
        template_idx = rng.randint(0, 9)
        fact_objs = chain["fact_objs"]
        answer = chain["answer"]
        question = _question_for(task_type,
                                 [(fo["text"], fo["subject"], fo["embedding_target"]) for fo in fact_objs],
                                 chain["addressable_subjects"], template_idx)
        fact_key = tuple(fo["text"] for fo in fact_objs)
        sig = (question, answer, fact_key)
        if split == "train":
            # Allow up to 5 repeats per exact (question, answer, facts) combo for density
            count = seen.get(sig, 0)
            if count > 4:
                continue
            seen[sig] = count + 1
        else:
            if sig in seen:
                continue
            seen[sig] = 1
        examples.append({
            "facts": fact_objs, "question": question, "answer": answer,
            "answer_token": answer, "required_slots": chain["required_slots"],
            "addressable_slots": chain["addressable_slots"],
            "distractor_tokens": [fo["embedding_target"] for fo in fact_objs[:-1]],
            "reasoning_hops": chain["hops"], "task_type": task_type, "split": split,
        })
    rng.shuffle(examples)
    return examples


def _question_for(task_type, facts, addressable_subjects, template_idx=0):
    fn = facts[0][1]
    if task_type == "single_fact_recall":
        templates = [
            f"What does {fn} return ?",
            f"The function {fn} returns which value ?",
            f"Which value is the output of {fn} ?",
            f"After executing {fn} , what is returned ?",
            f"What is the return value of {fn} ?",
        ]
        return templates[template_idx % len(templates)]
    if task_type == "two_hop_reasoning":
        templates = [
            f"Calling {fn} returns a value that is then wrapped . What is the wrapped value ?",
            f"The result of {fn} is further processed . What is the final value ?",
            f"After calling {fn} and transforming the result , what value do you get ?",
            f"The output of {fn} goes through a conversion . What is the converted value ?",
            f"{fn} produces a value that is then transformed . What does it become ?",
        ]
        return templates[template_idx % len(templates)]
    if task_type == "three_hop_reasoning":
        api = facts[0][1]
        templates = [
            f"What is {api} 's result after the mapper and adapter convert it ?",
            f"After {api} 's output passes through two transformations , what is the result ?",
            f"The API {api} returns a value that is mapped and adapted . What is the final result ?",
        ]
        return templates[template_idx % len(templates)]
    if task_type == "api_usage_reasoning":
        ep = facts[0][1]
        templates = [
            f"Which token must be sent to call {ep} ?",
            f"To invoke {ep} , which authentication token is required ?",
            f"What token does the endpoint {ep} expect ?",
        ]
        return templates[template_idx % len(templates)]
    if task_type == "code_symbol_reasoning":
        fn = facts[0][1]
        templates = [
            f"Which status results when {fn} fails ?",
            f"When {fn} encounters an error , what status is returned ?",
            f"If {fn} throws an exception , which status code should be expected ?",
        ]
        return templates[template_idx % len(templates)]
    raise ValueError(task_type)


def build_vocab(examples_by_split, kb) -> "OrderedDict[str, int]":
    vocab: "OrderedDict[str, int]" = OrderedDict()
    for tok in SPECIAL_TOKENS:
        vocab[tok] = len(vocab)
    content = set()
    for split_examples in examples_by_split.values():
        for ex in split_examples:
            for t in tokenize(ex["question"]):
                content.add(t)
            content.add(ex["answer"])
            for fo in ex["facts"]:
                for t in tokenize(fo["text"]):
                    content.add(t)
                content.add(fo["embedding_target"])
    for rec in kb.records:
        for t in tokenize(rec["text"]):
            content.add(t)
        content.add(rec["embedding_target"])
    for tok in sorted(content):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Generate SAM synthetic fact dataset.")
    ap.add_argument("--output", default="data/synthetic")
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--entity-separation", choices=["soft", "hard", "none"],
                    default="soft")
    ap.add_argument("--tiny", action="store_true",
                    help="fast preset: 500/100/100 into data/synthetic_tiny")
    args = ap.parse_args()

    if args.tiny:
        if args.output == "data/synthetic":
            args.output = "data/synthetic_tiny"
        if args.train == 10000:
            args.train, args.val, args.test = 500, 100, 100

    os.makedirs(args.output, exist_ok=True)
    pools = NamePools(args.entity_separation)
    kb = KB()

    rng_train = random.Random(args.seed)
    rng_val = random.Random(args.seed + 1)
    rng_test = random.Random(args.seed + 2)

    # Build a single shared fact pool so val/test reuse train slots
    shared_pool = build_fact_pool(rng_train, pools, kb, "train", pool_size=min(args.train // 30, 1000))

    train = generate_split_from_pool(rng_train, shared_pool, "train", args.train)
    val = generate_split_from_pool(rng_val, shared_pool, "val", args.val)
    test = generate_split_from_pool(rng_test, shared_pool, "test", args.test)

    examples_by_split = {"train": train, "val": val, "test": test}
    vocab = build_vocab(examples_by_split, kb)

    write_jsonl(os.path.join(args.output, "train.jsonl"), train)
    write_jsonl(os.path.join(args.output, "val.jsonl"), val)
    write_jsonl(os.path.join(args.output, "test.jsonl"), test)
    write_jsonl(os.path.join(args.output, "kb.jsonl"), kb.records)

    with open(os.path.join(args.output, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    meta = {
        "seed": args.seed,
        "entity_separation": args.entity_separation,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "num_slots": len(kb.records),
        "vocab_size": len(vocab),
        "task_types": TASK_TYPES,
        "special_tokens": {t: vocab[t] for t in SPECIAL_TOKENS},
        "hops_by_task": {
            "single_fact_recall": 1, "two_hop_reasoning": 2,
            "three_hop_reasoning": 3, "api_usage_reasoning": 2,
            "code_symbol_reasoning": 2,
        },
    }
    with open(os.path.join(args.output, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[synthetic_facts] wrote {len(train)}/{len(val)}/{len(test)} "
          f"examples, {len(kb.records)} slots, vocab {len(vocab)} -> {args.output}")


if __name__ == "__main__":
    main()
