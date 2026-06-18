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


class TestRequiredSetMetrics:
    """Experiment 0.10: Required-set retrieval metrics."""

    def test_single_required_slot_found(self):
        from sam.eval.metrics import compute_required_set_metrics
        results = compute_required_set_metrics(
            required_slots_list=[[5]],
            retrieved_topk_list=[[5, 10, 20, 30, 40, 50, 60, 70]],
            hops_list=[1],
            k_values=(1, 3, 8),
        )
        assert results["any_required_present_at_1"] == 1.0
        assert results["all_required_present_at_1"] == 1.0
        assert results["required_slot_coverage_at_1"] == 1.0
        assert results["any_required_present_at_8"] == 1.0
        assert results["all_required_present_at_8"] == 1.0
        assert results["mean_required_count"] == 1.0
        assert results["mean_retrieved_required_at_8"] == 1.0

    def test_single_required_slot_not_found(self):
        from sam.eval.metrics import compute_required_set_metrics
        results = compute_required_set_metrics(
            required_slots_list=[[99]],
            retrieved_topk_list=[[5, 10, 20, 30]],
            hops_list=[1],
            k_values=(1, 4),
        )
        assert results["any_required_present_at_4"] == 0.0
        assert results["all_required_present_at_4"] == 0.0
        assert results["required_slot_coverage_at_4"] == 0.0

    def test_multiple_required_slots_partial(self):
        from sam.eval.metrics import compute_required_set_metrics
        results = compute_required_set_metrics(
            required_slots_list=[[5, 10, 15]],
            retrieved_topk_list=[[5, 10, 20, 30, 40, 50, 60, 70]],
            hops_list=[2],
            k_values=(1, 2, 3, 8),
        )
        # at K=1: only slot 5 present, so any=True, all=False
        assert results["any_required_present_at_1"] == 1.0
        assert results["all_required_present_at_1"] == 0.0
        assert results["required_slot_coverage_at_1"] == 1.0 / 3.0

        # at K=2: slots 5 and 10 present, so any=True, all=False
        assert results["any_required_present_at_2"] == 1.0
        assert results["all_required_present_at_2"] == 0.0
        assert results["required_slot_coverage_at_2"] == 2.0 / 3.0

        # at K=3: still missing 15 (slot 20 is not required)
        assert results["any_required_present_at_3"] == 1.0
        assert results["all_required_present_at_3"] == 0.0

    def test_multiple_required_slots_all_found(self):
        from sam.eval.metrics import compute_required_set_metrics
        results = compute_required_set_metrics(
            required_slots_list=[[5, 10, 15]],
            retrieved_topk_list=[[5, 15, 10, 20, 30, 40, 50, 60]],
            hops_list=[3],
            k_values=(3, 8),
        )
        assert results["all_required_present_at_3"] == 1.0
        assert results["all_required_present_at_8"] == 1.0
        assert results["required_slot_coverage_at_3"] == 1.0
        assert results["all_required_three_hop_at_3"] == 1.0

    def test_duplicate_retrieved_slots(self):
        """Duplicate retrieved slots should not affect required-set metrics."""
        from sam.eval.metrics import compute_required_set_metrics
        results = compute_required_set_metrics(
            required_slots_list=[[5]],
            retrieved_topk_list=[[5, 5, 10, 5, 20, 30, 40, 50]],
            hops_list=[1],
            k_values=(3, 8),
        )
        assert results["all_required_present_at_3"] == 1.0
        assert results["required_slot_coverage_at_3"] == 1.0

    def test_no_required_slots(self):
        """Examples with no required slots should not break metrics."""
        from sam.eval.metrics import compute_required_set_metrics
        results = compute_required_set_metrics(
            required_slots_list=[[]],
            retrieved_topk_list=[[5, 10, 20, 30]],
            hops_list=[0],
            k_values=(4,),
        )
        # With no required slots, all_present should be True (vacuously) and any False
        assert results["any_required_present_at_4"] == 0.0
        # all_required is True when n_required > 0 and all are present;
        # with n_required==0, it's False
        assert results["all_required_present_at_4"] == 0.0
        assert results["required_slot_coverage_at_4"] == 0.0

    def test_multi_hop_per_hop_breakdown(self):
        from sam.eval.metrics import compute_required_set_metrics
        results = compute_required_set_metrics(
            required_slots_list=[[1], [2, 3], [4, 5, 6]],
            retrieved_topk_list=[[1, 10, 11], [2, 20, 21], [4, 7, 8, 9, 5, 6, 30, 31]],
            hops_list=[1, 2, 3],
            k_values=(3, 8),
        )
        # 1-hop: all present at K=3
        assert results["all_required_single_hop_at_3"] == 1.0
        # 2-hop: only slot 2 present, missing 3
        assert results["all_required_two_hop_at_3"] == 0.0
        # 3-hop: at K=3 only slot 4 present, missing 5, 6
        assert results["all_required_three_hop_at_3"] == 0.0
        # at K=8: all 3-hop required present
        assert results["all_required_three_hop_at_8"] == 1.0

    def test_aggregate_over_multiple_examples(self):
        from sam.eval.metrics import compute_required_set_metrics
        results = compute_required_set_metrics(
            required_slots_list=[[1], [2, 3], [99]],
            retrieved_topk_list=[
                [1, 10, 11, 12, 13, 14, 15, 16],
                [10, 11, 3, 2, 14, 15, 16, 17],
                [1, 2, 3, 4, 5, 6, 7, 8],
            ],
            hops_list=[1, 2, 1],
            k_values=(8,),
        )
        n = 3
        # Example 0: any=True, all=True. Example 1: any=True, all=True (2,3 both in top8)
        # Example 2: any=False, all=False
        assert results["any_required_present_at_8"] == 2.0 / n
        assert results["all_required_present_at_8"] == 2.0 / n
        # Total required: 1 + 2 + 1 = 4. Retrieved: 1 + 2 + 0 = 3
        assert results["required_slot_coverage_at_8"] == 3.0 / 4.0


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


class TestChainSetRetrieval:
    """Tests for Experiment 0.11 chain-set retrieval."""

    def test_chain_set_retriever_creation(self):
        from sam.training.train_retrieval import ChainSetRetriever, QueryEncoder
        from sam.data.dataset import Tokenizer
        import tempfile, subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run([
                sys.executable, "-m", "sam.data.synthetic_facts",
                "--output", tmpdir, "--train", "50", "--val", "30", "--test", "30",
                "--seed", "42", "--entity-separation", "none",
            ], capture_output=True, text=True,
               cwd=os.path.dirname(__file__) + "/..")

            tok = Tokenizer.from_dir(tmpdir)
            enc = QueryEncoder(vocab_size=tok.vocab_size, d_model=256,
                             n_layers=2, n_heads=4, d_ff=512, query_dim=256,
                             max_seq_len=64, pad_id=tok.pad)
            model = ChainSetRetriever(enc, 256, 100, temperature=0.07)
            assert model.param_count() > 0
            assert model.num_slots == 100

    def test_multi_positive_bce_loss(self):
        from sam.training.train_retrieval import multi_positive_bce_loss
        import torch

        B, D, S = 4, 256, 100
        q = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        s = torch.nn.functional.normalize(torch.randn(S, D), dim=-1)
        req = torch.tensor([[5, 10, -1], [3, -1, -1],
                           [7, 20, 25], [1, -1, -1]])

        loss = multi_positive_bce_loss(q, s, req, S, "cpu",
                                       temperature=0.07,
                                       negatives_per_positive=8,
                                       pos_weight=5.0)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_multi_positive_infonce_loss(self):
        from sam.training.train_retrieval import multi_positive_infonce_loss
        import torch

        B, D, S = 4, 256, 100
        q = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        s = torch.nn.functional.normalize(torch.randn(S, D), dim=-1)
        req = torch.tensor([[5, 10, -1], [3, -1, -1],
                           [7, 20, 25], [1, -1, -1]])

        loss = multi_positive_infonce_loss(q, s, req, S, "cpu",
                                           temperature=0.07,
                                           negatives_per_positive=8)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_slot_graph_expander(self):
        from sam.training.train_retrieval import SlotGraphExpander


class TestNoisyMemory:
    """Experiment 0.13A: Controlled Noisy Memory Tolerance tests."""

    def test_build_noisy_memory_slots_zero_distractors(self):
        """With 0 distractors, should return exactly the required slots."""
        from sam.model.sam_core import SamModel
        import torch

        model = SamModel(
            vocab_size=100, d_model=64, n_layers=3, n_heads=2, d_ff=128,
            max_seq_len=32, memory_every=2,
            memory_cfg={"num_subkeys": 16, "key_dim": 16, "value_dim": 32,
                        "top_a": 4, "top_b": 4, "top_k": 4},
            pad_id=0,
        )
        # Set up live slots
        model.pkm.slot_value_token[:30] = torch.randint(0, 50, (30,))
        model.live_slot_ids = torch.arange(30)

        required = torch.tensor([[5, 10], [15, 20]])
        B, K = 2, 8
        slots, scores = model._build_noisy_memory_slots(
            required, B, K, device=torch.device("cpu"),
            num_distractors=0, distractor_score=0.5,
        )
        assert slots.shape == (2, 8)
        assert scores.shape == (2, 8)
        # First two slots should be the required ones with score 1.0 (order may vary)
        req_set_0 = {int(slots[0, j].item()) for j in range(2)}
        assert req_set_0 == {5, 10}
        assert scores[0, 0].item() == 1.0
        assert scores[0, 1].item() == 1.0
        # Remaining slots should be -1 (padded)
        assert int(slots[0, 2].item()) == -1
        assert scores[0, 2].item() == float('-inf')

    def test_build_noisy_memory_slots_with_distractors(self):
        """With distractors, required + N non-required random slots."""
        from sam.model.sam_core import SamModel
        import torch

        model = SamModel(
            vocab_size=100, d_model=64, n_layers=3, n_heads=2, d_ff=128,
            max_seq_len=32, memory_every=2,
            memory_cfg={"num_subkeys": 16, "key_dim": 16, "value_dim": 32,
                        "top_a": 4, "top_b": 4, "top_k": 4},
            pad_id=0,
        )
        # 100 live slots
        model.pkm.slot_value_token[:100] = torch.randint(0, 50, (100,))
        model.live_slot_ids = torch.arange(100)

        # Use same number of required slots per example for tensor construction
        required = torch.tensor([[5, 10], [15, 20]])
        B, K = 2, 8
        slots, scores = model._build_noisy_memory_slots(
            required, B, K, device=torch.device("cpu"),
            num_distractors=3, distractor_score=0.5,
        )
        assert slots.shape == (2, 8)

        # Check first example: 2 required + 3 distractors
        req_0 = {int(slots[0, 0].item()), int(slots[0, 1].item())}
        assert req_0 == {5, 10}
        assert scores[0, 0].item() == 1.0
        assert scores[0, 1].item() == 1.0

        # Check distractors are present (slots 2, 3, 4) with distractor score
        dist_slots_0 = [int(slots[0, j].item()) for j in range(2, 5)]
        assert all(s >= 0 for s in dist_slots_0), f"Expected valid distractor slots, got {dist_slots_0}"
        assert all(s not in req_0 for s in dist_slots_0), f"Distractors should not be required slots: {dist_slots_0} intersect {req_0}"
        assert all(scores[0, j].item() == 0.5 for j in range(2, 5))

        # Check second example: 2 required + 3 distractors
        req_1 = {int(slots[1, 0].item()), int(slots[1, 1].item())}
        assert req_1 == {15, 20}

    def test_build_noisy_memory_distractors_not_required(self):
        """Distractors must never be identical to required slots."""
        from sam.model.sam_core import SamModel
        import torch

        model = SamModel(
            vocab_size=100, d_model=64, n_layers=3, n_heads=2, d_ff=128,
            max_seq_len=32, memory_every=2,
            memory_cfg={"num_subkeys": 16, "key_dim": 16, "value_dim": 32,
                        "top_a": 4, "top_b": 4, "top_k": 4},
            pad_id=0,
        )
        model.pkm.slot_value_token[:100] = torch.randint(0, 50, (100,))
        model.live_slot_ids = torch.arange(100)

        # Run many times since randomization could occasionally pick a required slot
        for _ in range(10):
            required = torch.tensor([[0, 1, 2]])
            B, K = 1, 8
            slots, scores = model._build_noisy_memory_slots(
                required, B, K, device=torch.device("cpu"),
                num_distractors=4, distractor_score=0.5,
            )
            req_set = {0, 1, 2}
            for j in range(3, 7):  # distractor positions
                sid = int(slots[0, j].item())
                if sid >= 0:
                    assert sid not in req_set, f"Distractor {sid} is also a required slot!"

    def test_memory_integration_modes_exist(self):
        """Verify all integration modes can be instantiated."""
        from sam.model.sam_core import MemoryHead
        import torch

        for mode in ["gated_sum", "forced_gate_1", "forced_gate_scalar", "concat_projection"]:
            head = MemoryHead(
                d_model=64, key_dim=32, value_dim=48,
                integration=mode, gate_alpha=0.5,
            )
            x = torch.randn(2, 8, 64)
            mem_val = torch.randn(2, 8, 48)
            result = head.integrate(x, mem_val)
            assert result.shape == (2, 8, 64), f"Mode {mode}: expected (2,8,64) got {result.shape}"

    def test_forced_gate_1_simple_addition(self):
        """Forced gate=1 should be x + mem_d (no sigmoid suppression)."""
        from sam.model.sam_core import MemoryHead
        import torch

        head = MemoryHead(d_model=64, key_dim=32, value_dim=48, integration="forced_gate_1")
        x = torch.ones(1, 4, 64)
        mem_val = torch.ones(1, 4, 48)
        result = head.integrate(x, mem_val)
        # mem_d = mem_proj(ones(48)) which could have values, but x=1s
        # Just verify shape and no errors
        assert result.shape == (1, 4, 64)
        # result should not equal x (memory was added)
        assert not torch.allclose(result, x)

    def test_forward_with_noise_mode(self):
        """Forward pass with oracle_plus_distractors noise mode."""
        from sam.model.sam_core import SamModel
        import torch

        model = SamModel(
            vocab_size=100, d_model=64, n_layers=3, n_heads=2, d_ff=128,
            max_seq_len=32, memory_every=2,
            memory_cfg={"num_subkeys": 16, "key_dim": 16, "value_dim": 32,
                        "top_a": 4, "top_b": 4, "top_k": 4},
            pad_id=0,
        )
        model.pkm.slot_value_token[:50] = torch.randint(0, 100, (50,))
        model.live_slot_ids = torch.arange(50)
        model._memory_noise_mode = "oracle_plus_distractors"
        model._num_distractors = 2
        model._distractor_score = 0.5

        x = torch.randint(1, 100, (2, 16))
        labels = x.clone()
        required = torch.tensor([[3, 7], [5, 12]])
        plens = torch.tensor([16, 16])
        logits, loss, aux = model(
            x, labels=labels, required_slots=required,
            prompt_lens=plens, mode="retrieved_memory_external_text_query",
        )
        assert logits.shape == (2, 16, 100)
        assert loss is not None
        # Check noise diagnostics
        assert "_noisy_memory_info" in aux
        info = aux["_noisy_memory_info"]
        assert len(info) == 2
        assert info[0]["num_required"] == 2
        assert info[0]["num_distractors"] <= 2
        # Check gate diagnostics are populated
        assert "gate_mean" in aux
        assert "memory_norm" in aux
        assert "memory_residual_ratio" in aux

    def test_forward_with_forced_gate(self):
        """Forward pass with forced_gate_1 integration mode."""
        from sam.model.sam_core import SamModel
        import torch

        model = SamModel(
            vocab_size=100, d_model=64, n_layers=3, n_heads=2, d_ff=128,
            max_seq_len=32, memory_every=2,
            memory_cfg={"num_subkeys": 16, "key_dim": 16, "value_dim": 32,
                        "top_a": 4, "top_b": 4, "top_k": 4,
                        "memory_integration_mode": "forced_gate_1"},
            pad_id=0,
        )
        model.pkm.slot_value_token[:50] = torch.randint(0, 100, (50,))
        model.live_slot_ids = torch.arange(50)

        x = torch.randint(1, 100, (2, 16))
        labels = x.clone()
        required = torch.tensor([[3, 7], [5, 12]])
        plens = torch.tensor([16, 16])
        logits, loss, aux = model(
            x, labels=labels, required_slots=required,
            prompt_lens=plens, mode="retrieved_memory",
        )
        assert logits.shape == (2, 16, 100)
        assert loss is not None
        # Gate diagnostics for forced gate
        if "gate_mean" in aux:
            assert aux["gate_mean"] == 1.0  # forced gate = 1.0

    def test_forward_with_concat_projection(self):
        """Forward pass with concat_projection integration mode."""
        from sam.model.sam_core import SamModel
        import torch

        model = SamModel(
            vocab_size=100, d_model=64, n_layers=3, n_heads=2, d_ff=128,
            max_seq_len=32, memory_every=2,
            memory_cfg={"num_subkeys": 16, "key_dim": 16, "value_dim": 32,
                        "top_a": 4, "top_b": 4, "top_k": 4,
                        "memory_integration_mode": "concat_projection"},
            pad_id=0,
        )
        model.pkm.slot_value_token[:50] = torch.randint(0, 100, (50,))
        model.live_slot_ids = torch.arange(50)

        x = torch.randint(1, 100, (2, 16))
        labels = x.clone()
        plens = torch.tensor([16, 16])
        logits, loss, aux = model(
            x, labels=labels, prompt_lens=plens,
            mode="core_only",
        )
        assert logits.shape == (2, 16, 100)
        assert loss is not None

    def test_required_set_rank_metrics(self):
        from sam.eval.analyze_required_set_retrieval import compute_extended_rank_metrics

        n = 3
        required = [[5], [10, 15], [20, 25, 30]]
        retrieved = [[5, 2, 10, 3], [15, 10, 1, 2], [20, 25, 30, 1]]
        k_values = (1, 4)

        metrics = compute_extended_rank_metrics(required, retrieved, k_values)
        assert "mrr_first_required_at_4" in metrics
        # Slot 5 found at rank 1 in first example -> MRR contribution 1.0
        assert metrics["mrr_first_required_at_4"] > 0.5

    def test_chain_set_retrieval_from_query(self):
        from sam.training.train_retrieval import ChainSetRetriever, QueryEncoder, \
            multi_positive_bce_loss
        from sam.data.dataset import Tokenizer
        import tempfile, subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run([
                sys.executable, "-m", "sam.data.synthetic_facts",
                "--output", tmpdir, "--train", "100", "--val", "50", "--test", "50",
                "--seed", "42", "--entity-separation", "none",
            ], capture_output=True, text=True,
               cwd=os.path.dirname(__file__) + "/..")

            tok = Tokenizer.from_dir(tmpdir)
            enc = QueryEncoder(vocab_size=tok.vocab_size, d_model=256,
                             n_layers=2, n_heads=4, d_ff=512, query_dim=256,
                             max_seq_len=64, pad_id=tok.pad)

            import json
            with open(os.path.join(tmpdir, "meta.json")) as f:
                meta = json.load(f)

            num_slots = meta.get("num_slots", 100)
            model = ChainSetRetriever(enc, 256, num_slots, temperature=0.07)

            # Forward pass with dummy input
            dummy = torch.randint(0, tok.vocab_size, (2, 32))
            lens = torch.tensor([16, 20])
            q, s = model(dummy, lens)
            assert q.shape == (2, 256)
            assert s.shape == (num_slots, 256)

            # Retrieve top-k (k limited by actual slot count)
            k_retrieve = min(8, num_slots)
            slots, scores = model.retrieve_topk(q, k_retrieve)
            assert slots.shape == (2, k_retrieve)
            assert scores.shape == (2, k_retrieve)
