# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the torch-dependent recording helpers of ``gr00t.explainer.capture``.

The real capture needs a 3B checkpoint and a GPU, but its building blocks do not: this
module instantiates a tiny :class:`Gr00tN1d7ActionHead` (an ``AlternateVLDiT`` with two
blocks plus a one-layer VL self-attention transformer) and checks the two properties the
explainer trace depends on.

* :class:`RecordingAttnProcessor` is a *faithful* replacement for the default diffusers
  processor -- swapping it in must not change the model output, only expose the softmax
  probabilities. It must also reproduce the block routing of ``AlternateVLDiT``: cross
  blocks attend over the vision-language sequence with the image/text mask applied, self
  blocks over the ``[state, actions]`` tokens.
* :func:`run_denoising_with_trace` is a faithful re-implementation of the Euler loop in
  ``Gr00tN1d7ActionHead.get_action_with_features``: under the same seed both must produce
  the same action chunk, and the recorded per-step intermediates must be self-consistent.
"""

from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config  # noqa: E402
from gr00t.explainer.capture import (  # noqa: E402
    AttentionRecorder,
    RecordingAttnProcessor,
    _action_head_blocks,
    bool_mask_to_additive,
    recording_attn_processors,
    run_denoising_with_trace,
)
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead  # noqa: E402
from transformers.feature_extraction_utils import BatchFeature  # noqa: E402


SEQ_LEN = 10
"""Length of the synthetic vision-language sequence handed to the action head."""

IMAGE_POSITIONS = (2, 3, 4, 5)
"""Positions of the (fake) image tokens inside that sequence."""

EMBED_DIM = 64
STATE_DIM = 7
ACTION_HORIZON = 4
SA_LEN = 1 + ACTION_HORIZON  # one state token + the action tokens
DIT_HEADS = 2


def _small_config(**overrides) -> Gr00tN1d7Config:
    """The tiny action-head config of ``tests/gr00t/model/test_action_head.py``.

    Two differences: ``use_alternate_vl_dit`` is on (the explainer only traces the
    alternating vision-language DiT) and a one-layer VL self-attention transformer is
    configured so ``head.vl_self_attention`` is a real module rather than ``Identity``.
    """
    defaults = dict(
        backbone_embedding_dim=EMBED_DIM,
        hidden_size=EMBED_DIM,
        input_embedding_dim=EMBED_DIM,
        max_state_dim=STATE_DIM,
        max_action_dim=STATE_DIM,
        action_horizon=ACTION_HORIZON,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=True,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": DIT_HEADS,
            "attention_head_dim": EMBED_DIM // DIT_HEADS,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": EMBED_DIM,
            "interleave_self_attention": True,
        },
        vl_self_attention_cfg={
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": EMBED_DIM // 2,
            "dropout": 0.0,
            "final_dropout": False,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


@pytest.fixture
def head_and_config():
    config = _small_config()
    torch.manual_seed(0)
    head = Gr00tN1d7ActionHead(config)
    head.eval()
    return head, config


@pytest.fixture
def synthetic_inputs():
    """A fake backbone output plus the matching action input.

    ``make_backbone_output`` returns a *fresh* :class:`BatchFeature` around the same
    tensors on every call: ``Gr00tN1d7ActionHead.process_backbone_output`` overwrites
    ``backbone_features`` in place, so a single instance cannot be reused across the
    traced run and the reference ``get_action`` call.
    """
    torch.manual_seed(123)
    features = torch.randn(1, SEQ_LEN, EMBED_DIM)
    attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.bool)
    image_mask = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    image_mask[0, list(IMAGE_POSITIONS)] = True
    action_input = BatchFeature(
        data={"state": torch.randn(1, 1, STATE_DIM), "embodiment_id": torch.tensor([1])}
    )

    def make_backbone_output() -> BatchFeature:
        return BatchFeature(
            data={
                "backbone_features": features,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )

    return make_backbone_output, action_input


def test_bool_mask_to_additive_shapes():
    """Boolean keep-masks become ``(B*H, 1, S)`` additive masks; floats pass through."""
    like = torch.zeros(1, dtype=torch.float32)
    neg = torch.finfo(torch.float32).min

    keep = torch.ones(1, SEQ_LEN, dtype=torch.bool)
    keep[0, 3] = False
    keep[0, 7] = False
    additive = bool_mask_to_additive(keep, 1, DIT_HEADS, SEQ_LEN, like)
    assert additive.shape == (DIT_HEADS, 1, SEQ_LEN)
    assert additive.dtype == torch.float32

    expected = torch.zeros(1, SEQ_LEN)
    expected[0, 3] = neg
    expected[0, 7] = neg
    for h in range(DIT_HEADS):
        assert torch.equal(additive[h], expected)

    assert bool_mask_to_additive(None, 1, DIT_HEADS, SEQ_LEN, like) is None

    float_mask = torch.randn(1, 1, SEQ_LEN)
    passed = bool_mask_to_additive(float_mask, 1, DIT_HEADS, SEQ_LEN, like)
    assert passed.shape == (DIT_HEADS, 1, SEQ_LEN)
    for h in range(DIT_HEADS):
        assert torch.equal(passed[h], float_mask[0])


def test_recording_processor_matches_default(head_and_config, synthetic_inputs):
    """Recording must be transparent: same DiT output, plus usable attention maps."""
    head, config = head_and_config
    make_backbone_output, _ = synthetic_inputs
    backbone_output = make_backbone_output()

    torch.manual_seed(1)
    dit_kwargs = dict(
        hidden_states=torch.randn(1, SA_LEN, EMBED_DIM),
        encoder_hidden_states=backbone_output.backbone_features,
        timestep=torch.tensor([250]),
        image_mask=backbone_output.image_mask,
        backbone_attention_mask=backbone_output.backbone_attention_mask,
    )

    dit_blocks, _ = _action_head_blocks(head)
    assert [name for name, _ in dit_blocks] == ["dit.block0", "dit.block1"]
    default_processors = [block.attn1.processor for _, block in dit_blocks]

    recorder = AttentionRecorder()
    with torch.no_grad():
        reference = head.model(**dit_kwargs)
        with recording_attn_processors(dit_blocks, recorder, lambda: ""):
            assert all(
                isinstance(block.attn1.processor, RecordingAttnProcessor) for _, block in dit_blocks
            )
            recorded = head.model(**dit_kwargs)

    assert torch.allclose(reference, recorded, atol=1e-4)
    # The originals are put back on exit, so later untraced calls are unaffected.
    assert [block.attn1.processor for _, block in dit_blocks] == default_processors

    assert set(recorder.mean) == {"dit.block0", "dit.block1"}

    # Block 1 is a self block: the [state, actions] tokens attend to themselves.
    self_attn = recorder.get_mean("dit.block1")
    assert self_attn.shape == (SA_LEN, SA_LEN)
    assert np.allclose(self_attn.sum(axis=-1), 1.0, atol=1e-5)

    # Block 0 is a cross_text block (idx % (2 * attend_text_every_n_blocks) == 0): it
    # attends over the vision-language sequence with the image tokens masked out.
    assert config.attend_text_every_n_blocks == 2
    cross_attn = recorder.get_mean("dit.block0")
    assert cross_attn.shape == (SA_LEN, SEQ_LEN)
    assert np.all(cross_attn[:, list(IMAGE_POSITIONS)] == 0.0)
    assert np.allclose(cross_attn.sum(axis=-1), 1.0, atol=1e-5)


def test_run_denoising_matches_get_action(head_and_config, synthetic_inputs):
    """The traced Euler loop reproduces ``get_action`` and records every step."""
    head, config = head_and_config
    make_backbone_output, action_input = synthetic_inputs

    dit_blocks, vlsa_blocks = _action_head_blocks(head)
    assert [name for name, _ in vlsa_blocks] == ["vlsa.layer0"], (
        "the fixture config must build a real SelfAttentionTransformer"
    )

    recorder = AttentionRecorder()
    step_holder = {"step": -1}

    def prefix() -> str:
        # The VL self-attention runs once, before the loop, while step == -1.
        return "" if step_holder["step"] < 0 else f"step{step_holder['step']}."

    with torch.no_grad():
        with recording_attn_processors(dit_blocks + vlsa_blocks, recorder, prefix):
            traced = run_denoising_with_trace(
                head,
                make_backbone_output(),
                action_input,
                recorder,
                step_holder,
                seed=7,
            )

        # ``get_action`` draws its noise with ``torch.randn`` from the global RNG, exactly
        # like the traced loop does after its own ``torch.manual_seed(seed)``.
        torch.manual_seed(7)
        expected = head.get_action(make_backbone_output(), action_input)["action_pred"]

    assert torch.allclose(traced["actions"], expected, atol=1e-4)

    n_steps = config.num_inference_timesteps
    assert len(traced["steps"]) == n_steps
    dt = 1.0 / n_steps
    for i, step in enumerate(traced["steps"]):
        assert step["actions_in"].shape == (1, ACTION_HORIZON, STATE_DIM)
        assert torch.allclose(
            step["actions_out"], step["actions_in"] + dt * step["velocity"], atol=1e-6
        ), f"step {i} is not one Euler update"
    # ``dit_hidden`` is the input plus the output of every block.
    assert len(traced["steps"][0]["dit_hidden"]) == config.diffusion_model_cfg["num_layers"] + 1

    for key in ("step0.dit.block0", "step0.dit.block1", "step1.dit.block0", "step1.dit.block1"):
        assert recorder.has(key), f"missing recorded attention for {key}"
    assert recorder.has("vlsa.layer0"), "VL self-attention keys carry no step prefix"

    assert traced["vl_embs"].shape == (1, SEQ_LEN, EMBED_DIM)
    assert traced["state_features"].shape == (1, 1, EMBED_DIM)


def test_recorder_per_head_filter():
    """``keep_per_head`` selects which keys keep their per-head maps; repeats stack."""
    recorder = AttentionRecorder(keep_per_head=lambda key: key.endswith("block1"))
    torch.manual_seed(3)
    probs_a = torch.rand(1, 2, 3, 3)
    probs_b = torch.rand(1, 2, 3, 3)
    recorder.record("a.block0", probs_a)
    recorder.record("a.block1", probs_b)

    assert recorder.get_per_head("a.block0") is None
    per_head = recorder.get_per_head("a.block1")
    assert per_head.shape == (2, 3, 3)
    assert np.allclose(per_head, probs_b[0].numpy(), atol=1e-6)

    mean = recorder.get_mean("a.block1")
    assert mean.shape == (3, 3)
    assert np.allclose(mean, per_head.mean(axis=0), atol=1e-6)
    assert np.allclose(recorder.get_mean("a.block0"), probs_a[0].numpy().mean(axis=0), atol=1e-6)

    # A second call under the same key (the vision blocks do this, once per image) stacks.
    recorder.record("a.block0", probs_b)
    assert recorder.get_mean("a.block0").shape == (2, 3, 3)
