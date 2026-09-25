"""Tests for the prefill capture: masks, pooling, and hooks that keep only the prefill."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.extract.prefill import (  # noqa: E402
    POSITIONS, PrefillCapture, build_pool_masks, decoder_layers, pool,
)

IMG, SPECIAL = 9, 0
HIDDEN = 4


def masks_for(ids, attention=None):
    ids = torch.tensor([ids])
    attention = torch.ones_like(ids) if attention is None else torch.tensor([attention])
    return build_pool_masks(ids, attention, IMG, [SPECIAL])


class TestMasks:
    def test_vision_text_and_last(self):
        m = masks_for([SPECIAL, 5, IMG, IMG, IMG, 6, SPECIAL])
        assert m.vision.tolist() == [False, False, True, True, True, False, False]
        assert m.text.tolist() == [False, True, False, False, False, True, False]
        assert m.last == 6 and m.n_vision == 3 and m.seq_len == 7

    def test_last_follows_the_attention_mask(self):
        m = masks_for([5, IMG, 6, 7], attention=[1, 1, 1, 0])
        assert m.last == 2
        assert not m.text[3]

    def test_no_image_tokens_raises(self):
        with pytest.raises(ValueError, match="no image tokens"):
            masks_for([SPECIAL, 5, 6])

    def test_no_text_tokens_raises(self):
        with pytest.raises(ValueError, match="no text tokens"):
            masks_for([SPECIAL, IMG, IMG])

    def test_batches_are_refused(self):
        with pytest.raises(ValueError, match="one unpadded item"):
            build_pool_masks(torch.tensor([[5, IMG], [5, IMG]]), torch.ones(2, 2), IMG, [SPECIAL])


class TestPool:
    def test_each_position_reads_its_own_tokens(self):
        m = masks_for([SPECIAL, 5, IMG, IMG, 6])
        h = torch.arange(5 * HIDDEN, dtype=torch.float16).reshape(1, 5, HIDDEN)
        out = pool(h, m)
        x = h[0].float()
        expected = {
            "vision_mean": x[2:4].mean(0),
            "vision_max": x[2:4].amax(0),
            "query_last": x[4],
            "query_mean": x[[1, 4]].mean(0),
        }
        assert out.shape == (len(POSITIONS), HIDDEN) and out.dtype == torch.float16
        for k, pos in enumerate(POSITIONS):
            assert torch.equal(out[k], expected[pos].half())

    def test_overflow_to_fp16_is_refused(self):
        m = masks_for([5, IMG, 6])
        h = torch.full((1, 3, HIDDEN), 1e6)            # finite in fp32, inf in fp16
        with pytest.raises(FloatingPointError):
            pool(h, m)


# ---------------------------------------------------------------------------
# Hooks, against a fake decoder that is called the way generate() calls it
# ---------------------------------------------------------------------------

class FakeBlock(nn.Module):
    """Returns (hidden_states,) like Qwen2_5_VLDecoderLayer on transformers 4.57.6."""
    def forward(self, h):
        return (h + 1,)


def fake_generate(layers, seq_len, decode_steps=3):
    """A prefill over the whole prompt, then one-token decode steps."""
    h = torch.zeros(1, seq_len, HIDDEN)
    for block in layers:
        h = block(h)[0]
    x = h[:, -1:]
    for _ in range(decode_steps):
        for block in layers:
            x = block(x)[0]


class TestPrefillCapture:
    def setup_method(self):
        self.layers = nn.ModuleList(FakeBlock() for _ in range(3))
        self.masks = masks_for([SPECIAL, 5, IMG, IMG, 6])

    def test_keeps_the_prefill_and_ignores_decode_steps(self):
        with PrefillCapture(self.layers, [0, 2], HIDDEN) as cap:
            cap.arm(self.masks)
            fake_generate(self.layers, self.masks.seq_len)
            acts = cap.collect()
        # block i outputs i+1 everywhere during the prefill; decode steps would be larger
        assert acts.shape == (2, len(POSITIONS), HIDDEN)
        assert torch.all(acts[0] == 1) and torch.all(acts[1] == 3)

    def test_state_does_not_leak_between_items(self):
        with PrefillCapture(self.layers, [0], HIDDEN) as cap:
            for _ in range(2):
                cap.arm(self.masks)
                fake_generate(self.layers, self.masks.seq_len)
                assert torch.all(cap.collect() == 1)

    def test_missing_layer_raises(self):
        with PrefillCapture(self.layers, [0, 1], HIDDEN) as cap:
            cap.arm(self.masks)
            self.layers[0](torch.zeros(1, self.masks.seq_len, HIDDEN))
            with pytest.raises(RuntimeError, match=r"layers \[1\]"):
                cap.collect()

    def test_decoder_call_while_disarmed_raises(self):
        with PrefillCapture(self.layers, [0], HIDDEN):
            with pytest.raises(RuntimeError, match="not armed"):
                fake_generate(self.layers, self.masks.seq_len)

    def test_arming_twice_raises(self):
        with PrefillCapture(self.layers, [0], HIDDEN) as cap:
            cap.arm(self.masks)
            with pytest.raises(RuntimeError, match="already armed"):
                cap.arm(self.masks)

    def test_first_call_shorter_than_prompt_raises(self):
        with PrefillCapture(self.layers, [0], HIDDEN) as cap:
            cap.arm(self.masks)
            with pytest.raises(ValueError, match="sequence length"):
                self.layers[0](torch.zeros(1, 1, HIDDEN))

    def test_wrong_hidden_size_raises(self):
        with PrefillCapture(self.layers, [0], HIDDEN + 1) as cap:
            cap.arm(self.masks)
            with pytest.raises(ValueError, match="unexpected hidden shape"):
                fake_generate(self.layers, self.masks.seq_len)

    def test_unexpected_output_type_raises(self):
        class Bare(nn.Module):
            def forward(self, h):
                return h
        layers = nn.ModuleList([Bare()])
        with PrefillCapture(layers, [0], HIDDEN) as cap:
            cap.arm(self.masks)
            with pytest.raises(TypeError, match="1-tuple"):
                layers[0](torch.zeros(1, self.masks.seq_len, HIDDEN))

    def test_hooks_are_removed_on_exit(self):
        with PrefillCapture(self.layers, [0, 1, 2], HIDDEN):
            pass
        assert all(not block._forward_hooks for block in self.layers)


class TestDecoderLayers:
    def model(self, n):
        return SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(layers=nn.ModuleList(FakeBlock() for _ in range(n)))))

    def test_finds_the_blocks(self):
        assert len(decoder_layers(self.model(28), 28)) == 28

    def test_wrong_count_raises(self):
        with pytest.raises(RuntimeError, match="expected 28"):
            decoder_layers(self.model(27), 28)


def test_positions_match_what_the_probes_read():
    from src.probes.dataset import POSITIONS as PROBE_POSITIONS
    assert POSITIONS == PROBE_POSITIONS
