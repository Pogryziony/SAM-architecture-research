"""Basic smoke tests for the SAM POC.
Run with: pytest tests/ -v
"""
import os
import sys
import tempfile

import pytest
import torch

# Ensure sam-lm is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDataGeneration:
    def test_generate_tiny_dataset(self):
        """Verify the tiny dataset generator runs and produces expected files."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run([
                sys.executable, "-m", "sam.data.synthetic_facts",
                "--output", tmpdir, "--train", "100", "--val", "50", "--test", "50",
                "--seed", "42",
            ], capture_output=True, text=True, cwd=os.path.dirname(__file__) + "/..")
            assert result.returncode == 0, result.stderr
            assert os.path.exists(os.path.join(tmpdir, "train.jsonl"))
            assert os.path.exists(os.path.join(tmpdir, "vocab.json"))
            assert os.path.exists(os.path.join(tmpdir, "kb.jsonl"))


class TestTokenizer:
    def test_tokenizer(self):
        from sam.data.dataset import Tokenizer
        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<ans>": 3,
                 "<q>": 4, "<fact>": 5, "<unk>": 6, "hello": 7, "world": 8}
        tok = Tokenizer(vocab)
        assert tok.vocab_size == 9
        ids = tok.encode("hello world")
        assert ids == [7, 8]
        text = tok.decode([7, 8])
        assert "hello" in text and "world" in text


class TestDenseTransformer:
    def test_forward_pass(self):
        from sam.model.transformer import DenseTransformer
        model = DenseTransformer(
            vocab_size=100, d_model=64, n_layers=2, n_heads=2, d_ff=128,
            max_seq_len=32, pad_id=0,
        )
        x = torch.randint(0, 100, (2, 16))
        labels = x.clone()
        logits, loss = model(x, labels=labels)
        assert logits.shape == (2, 16, 100)
        assert loss is not None and loss.item() > 0

    def test_generate(self):
        from sam.model.transformer import DenseTransformer
        model = DenseTransformer(
            vocab_size=100, d_model=64, n_layers=2, n_heads=2, d_ff=128,
            max_seq_len=32, pad_id=0,
        )
        prompt = torch.randint(1, 100, (8,))
        generated = model.generate(prompt, max_new_tokens=4, eos_id=2)
        assert generated.numel() <= 4


class TestProductKeyMemory:
    def test_forward(self):
        from sam.model.product_key_memory import ProductKeyMemory
        pkm = ProductKeyMemory(
            num_subkeys=64, key_dim=32, value_dim=48,
            top_a=8, top_b=8, top_k=4,
        )
        query = torch.randn(2, 64)
        # Without value_emb, returns None for memory
        mem, slots, weights = pkm(query)
        assert mem is None
        assert slots.shape == (2, 4)
        assert weights.shape == (2, 4)

    def test_with_values(self):
        from sam.model.product_key_memory import ProductKeyMemory
        pkm = ProductKeyMemory(
            num_subkeys=64, key_dim=32, value_dim=48,
            top_a=8, top_b=8, top_k=4,
        )
        # Set up slot value tokens
        pkm.slot_value_token[:100] = torch.randint(0, 50, (100,))
        value_emb = torch.randn(50, 48)
        query = torch.randn(2, 64)
        mem, slots, weights = pkm(query, value_emb_weight=value_emb)
        assert mem is not None
        assert mem.shape == (2, 48)

    def test_read_slot_values(self):
        from sam.model.product_key_memory import ProductKeyMemory
        pkm = ProductKeyMemory(
            num_subkeys=16, key_dim=16, value_dim=32,
            top_a=4, top_b=4, top_k=4,
        )
        pkm.slot_value_token[:10] = torch.randint(0, 20, (10,))
        value_emb = torch.randn(20, 32)
        slots = torch.tensor([[0, 1, 5, -1]])
        vals = pkm.read_slot_values(slots, value_emb_weight=value_emb)
        assert vals.shape == (1, 32)


class TestSamModel:
    def test_core_only(self):
        from sam.model.sam_core import SamModel
        model = SamModel(
            vocab_size=100, d_model=64, n_layers=3, n_heads=2, d_ff=128,
            max_seq_len=32, memory_every=2,
            memory_cfg={"num_subkeys": 16, "key_dim": 16, "value_dim": 32,
                        "top_a": 4, "top_b": 4, "top_k": 4},
            pad_id=0,
        )
        x = torch.randint(0, 100, (2, 16))
        labels = x.clone()
        logits, loss, aux = model(x, labels=labels, mode="core_only")
        assert logits.shape == (2, 16, 100)
        assert loss is not None

    def test_oracle_memory(self):
        from sam.model.sam_core import SamModel
        model = SamModel(
            vocab_size=100, d_model=64, n_layers=3, n_heads=2, d_ff=128,
            max_seq_len=32, memory_every=2,
            memory_cfg={"num_subkeys": 16, "key_dim": 16, "value_dim": 32,
                        "top_a": 4, "top_b": 4, "top_k": 4},
            pad_id=0,
        )
        x = torch.randint(0, 100, (2, 16))
        labels = x.clone()
        required = torch.tensor([[0, 1], [2, 3]])
        logits, loss, aux = model(x, labels=labels, required_slots=required,
                                  mode="oracle_memory")
        assert logits.shape == (2, 16, 100)
        assert loss is not None

    def test_retrieved_memory(self):
        from sam.model.sam_core import SamModel
        model = SamModel(
            vocab_size=100, d_model=64, n_layers=3, n_heads=2, d_ff=128,
            max_seq_len=32, memory_every=2, memory_query="sequence",
            memory_cfg={"num_subkeys": 16, "key_dim": 16, "value_dim": 32,
                        "top_a": 4, "top_b": 4, "top_k": 4},
            pad_id=0,
        )
        # Set up some live slots
        model.pkm.slot_value_token[:50] = torch.randint(0, 100, (50,))
        x = torch.randint(0, 100, (2, 16))
        labels = x.clone()
        plens = torch.tensor([16, 16])
        logits, loss, aux = model(x, labels=labels, prompt_lens=plens,
                                  mode="retrieved_memory")
        assert logits.shape == (2, 16, 100)
        assert loss is not None

    def test_random_memory(self):
        from sam.model.sam_core import SamModel
        model = SamModel(
            vocab_size=100, d_model=64, n_layers=3, n_heads=2, d_ff=128,
            max_seq_len=32, memory_every=2,
            memory_cfg={"num_subkeys": 16, "key_dim": 16, "value_dim": 32,
                        "top_a": 4, "top_b": 4, "top_k": 4},
            pad_id=0,
        )
        model.pkm.slot_value_token[:50] = torch.randint(0, 100, (50,))
        model.live_slot_ids = torch.arange(50)
        x = torch.randint(0, 100, (2, 16))
        labels = x.clone()
        logits, loss, aux = model(x, labels=labels, mode="random_memory")
        assert logits.shape == (2, 16, 100)
        assert loss is not None


class TestEvalMetrics:
    def test_accuracy_by_hop(self):
        from sam.eval.metrics import accuracy_by_hop
        # This test requires a full dataloader; tested implicitly through
        # the smoke training pipeline
        pass

    def test_compute_derived_metrics(self):
        from sam.eval.metrics import compute_derived_metrics
        dense = {"accuracy_overall": 0.5, "accuracy_single_hop": 0.8,
                 "accuracy_two_hop": 0.4, "accuracy_three_hop": 0.2}
        core = {"accuracy_overall": 0.3, "accuracy_single_hop": 0.5,
                "accuracy_two_hop": 0.2, "accuracy_three_hop": 0.05}
        oracle = {"accuracy_overall": 0.7, "accuracy_single_hop": 0.9,
                  "accuracy_two_hop": 0.65, "accuracy_three_hop": 0.4}
        retrieved = {"accuracy_overall": 0.55, "accuracy_single_hop": 0.85,
                     "accuracy_two_hop": 0.5, "accuracy_three_hop": 0.25}
        recall = {"recall_at_8": 0.75, "recall_at_32": 0.85}
        derived = compute_derived_metrics(dense, core, oracle, retrieved, recall)
        assert derived["oracle_gap"] == pytest.approx(0.7 - 0.55)
        assert derived["memory_gain"] == pytest.approx(0.55 - 0.3)
        assert derived["dense_gap"] == pytest.approx(0.55 - 0.5)

    def test_evaluate_gates(self):
        from sam.eval.metrics import evaluate_gates
        recall = {"recall_at_8": 0.85}
        core = {"accuracy_overall": 0.3}
        oracle = {"accuracy_overall": 0.7, "accuracy_single_hop": 0.9,
                  "accuracy_two_hop": 0.65, "accuracy_three_hop": 0.4}
        retrieved = {"accuracy_overall": 0.55}
        dense = {"accuracy_overall": 0.5}
        gates = evaluate_gates(recall, core, oracle, retrieved, dense)
        assert gates["gate_1_retrieval"]["passed"]
        assert gates["gate_2_memory_usefulness"]["passed"]
        assert gates["gate_3_retrieval_gap"]["passed"]  # gap = 0.15 < 0.20


class TestConfig:
    def test_load_config(self):
        from sam.utils.config import load_config, Config
        cfg = load_config("configs/dense_smoke.yaml")
        assert cfg.model.d_model == 64
        assert cfg.train.epochs == 1

    def test_merge_override(self):
        from sam.utils.config import load_config
        cfg = load_config("configs/dense_smoke.yaml",
                          overrides={"train.epochs": 5})
        assert cfg.train.epochs == 5
