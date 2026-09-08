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

"""CPU tests for the numpy-only synthetic ``groot-trace/1`` generator.

These cover the contract the explainer website depends on: the tensor inventory of
:func:`gr00t.explainer.schema.expected_tensor_specs` is what actually lands on disk, the
structural masks (causal LLM attention, text-only / image-only DiT cross blocks) are
honored, the Euler bookkeeping is self-consistent, the validator catches corruption, and
none of it needs torch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from gr00t.explainer.schema import (
    dit_schedule,
    expected_tensor_specs,
    load_manifest,
    quantize_u8,
    read_tensor,
    validate_trace,
)
from gr00t.explainer.synthetic import build_synthetic_trace, write_trace
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
CAPTURE_SCRIPT = "scripts/explainer/capture_trace.py"

# gr00t/__init__.py patches transformers (and therefore imports torch) when it sees
# pytest's env markers; strip them so the subprocesses below import only what the module
# under test imports.
_STRIPPED_ENV = ("PYTEST_CURRENT_TEST", "GROOT_HF_LOCAL_FIRST", "GROOT_PATCH_MISTRAL")


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}


@pytest.fixture(scope="module")
def lite_trace() -> tuple[dict, dict[str, np.ndarray]]:
    """A lite synthetic trace, built once and only ever read by the tests."""
    return build_synthetic_trace(profile="lite", seed=0)


def test_lite_trace_round_trip(tmp_path: Path, lite_trace) -> None:
    manifest, tensors = lite_trace
    write_trace(tmp_path, manifest, tensors)
    assert validate_trace(tmp_path) == []

    written = load_manifest(tmp_path)
    assert written["schema_version"] == "groot-trace/1"
    assert written["source"] == "synthetic"
    assert written["profile"] == "lite"

    specs = expected_tensor_specs(manifest["meta"], "lite")
    assert specs, "the lite inventory must not be empty"
    for name, (shape, dtype) in specs.items():
        spec = written["tensors"].get(name)
        assert spec is not None, f"tensor {name!r} missing from the written manifest"
        assert tuple(spec["shape"]) == tuple(shape), name
        assert spec["dtype"] == dtype, name

    f16_names = ["llm.inputs_embeds", "llm.layer0.norms", "step0.actions_in", "vlsa.vl_embs"]
    for name in f16_names:
        assert specs[name][1] == "f16"
        got = read_tensor(tmp_path, written, name).astype(np.float32)
        assert np.allclose(got, tensors[name], rtol=2e-3, atol=1e-3), name

    u8_names = ["llm.layer0.attn_mean", "vit.block0.attn_mean", "step0.dit.block1.attn_mean"]
    for name in u8_names:
        assert specs[name][1] == "u8"
        got = read_tensor(tmp_path, written, name)
        assert np.array_equal(got, quantize_u8(tensors[name])), name


def test_full_profile_inventory() -> None:
    manifest, tensors = build_synthetic_trace(profile="full", full_head_steps=(3,), seed=0)
    meta = manifest["meta"]
    specs = expected_tensor_specs(meta, "full", full_head_steps=(3,))
    seq = int(meta["llm"]["seq_len"])
    sa_len = 1 + int(meta["action"]["horizon_model"])

    assert specs["llm.layer0.attn"] == ((int(meta["llm"]["heads"]), seq, seq), "u8")
    assert tensors["llm.layer0.attn"].shape == (16, seq, seq)
    assert specs["vlsa.layer0.attn"] == ((int(meta["vlsa"]["heads"]), seq, seq), "u8")

    dit_heads = int(meta["dit"]["heads"])
    assert "step3.dit.block0.attn" in specs
    assert tensors["step3.dit.block0.attn"].shape == (dit_heads, sa_len, seq)
    assert "step3.dit.block1.attn" in specs
    assert tensors["step3.dit.block1.attn"].shape == (dit_heads, sa_len, sa_len)
    assert "step0.dit.block0.attn" not in specs
    assert "step0.dit.block0.attn" not in tensors
    assert "step0.dit.block0.attn_mean" in specs

    # The lite summaries are exactly the head-mean of the per-head maps.
    for mean_name in (
        "llm.layer0.attn_mean",
        "vlsa.layer0.attn_mean",
        "step3.dit.block0.attn_mean",
    ):
        per_head = tensors[mean_name[: -len("_mean")]]
        assert np.allclose(tensors[mean_name], per_head.mean(0), atol=1e-5), mean_name


def test_llm_attention_is_causal(lite_trace) -> None:
    manifest, tensors = lite_trace
    n_llm = int(manifest["meta"]["llm"]["select_layer"])
    for i in range(n_llm):
        attn = tensors[f"llm.layer{i}.attn_mean"]
        assert np.all(np.triu(attn, k=1) == 0.0), f"layer {i} attends to the future"
        assert np.allclose(attn.sum(axis=-1), 1.0, atol=1e-4), f"layer {i} rows are unnormalized"


def test_dit_cross_attention_respects_masks(lite_trace) -> None:
    manifest, tensors = lite_trace
    meta = manifest["meta"]
    image_mask = np.asarray(manifest["tokens"]["image_mask"], dtype=bool)
    seq = int(meta["llm"]["seq_len"])
    sa_len = 1 + int(meta["action"]["horizon_model"])
    n_steps = int(meta["flow"]["num_inference_timesteps"])

    for step in (0, n_steps - 1):
        for entry in meta["dit"]["schedule"]:
            attn = tensors[f"step{step}.dit.block{entry['idx']}.attn_mean"]
            where = f"step{step} block{entry['idx']} ({entry['kind']})"
            if entry["kind"] == "self":
                assert attn.shape == (sa_len, sa_len), where
                continue
            assert attn.shape == (sa_len, seq), where
            forbidden = image_mask if entry["kind"] == "cross_text" else ~image_mask
            assert np.all(attn[:, forbidden] == 0.0), where
            assert np.all(attn[:, ~forbidden].sum(axis=-1) > 0.0), where


def test_euler_consistency(lite_trace) -> None:
    manifest, tensors = lite_trace
    meta = manifest["meta"]
    dt = float(meta["flow"]["dt"])
    n_steps = int(meta["flow"]["num_inference_timesteps"])
    h_valid = int(meta["action"]["horizon_valid"])
    d_valid = int(meta["action"]["dim_valid"])
    steps = manifest["denoising"]["steps"]
    assert len(steps) == n_steps

    for s in range(n_steps):
        a_in = tensors[f"step{s}.actions_in"]
        a_out = tensors[f"step{s}.actions_out"]
        velocity = tensors[f"step{s}.velocity"]
        assert np.allclose(a_out, a_in + dt * velocity), f"step {s} is not an Euler step"
        if s + 1 < n_steps:
            assert np.array_equal(tensors[f"step{s + 1}.actions_in"], a_out), f"gap after step {s}"
        inline = np.asarray(steps[s]["actions_out_norm_valid"], dtype=np.float32)
        assert inline.shape == (h_valid, d_valid)
        assert np.allclose(inline, a_out[:h_valid, :d_valid]), f"step {s} inline chunk drifted"


def test_token_layout(lite_trace) -> None:
    manifest, _ = lite_trace
    meta = manifest["meta"]
    tokens = manifest["tokens"]
    strings = tokens["strings"]
    seq = int(meta["llm"]["seq_len"])

    expected_image_tokens = int(meta["vit"]["num_images"]) * int(meta["vit"]["tokens_per_image"])
    assert sum(tokens["image_mask"]) == expected_image_tokens
    assert seq == len(strings)
    assert len(tokens["input_ids"]) == seq
    assert len(tokens["attention_mask"]) == seq

    cursor = 0
    image_indices = []
    for group in tokens["groups"]:
        assert group["start"] == cursor, "token groups must be contiguous"
        assert group["end"] > group["start"]
        cursor = group["end"]
        if group["kind"] == "image":
            image_indices.append(group["image_index"])
    assert cursor == seq, "token groups must cover the whole sequence"
    assert image_indices == [0, 1]

    position_ids = tokens["position_ids"]
    assert len(position_ids) == 3
    assert all(len(row) == seq for row in position_ids)

    schedule = dit_schedule(32, 2)
    kinds = [entry["kind"] for entry in schedule]
    assert [entry["idx"] for entry in schedule] == list(range(32))
    assert kinds[:5] == ["cross_text", "self", "cross_image", "self", "cross_text"]
    assert kinds.count("self") == 16


def test_validate_detects_corruption(tmp_path: Path, lite_trace) -> None:
    manifest, tensors = lite_trace

    bad_shape_dir = tmp_path / "bad_shape"
    write_trace(bad_shape_dir, manifest, tensors)
    manifest_path = bad_shape_dir / "manifest.json"
    on_disk = json.loads(manifest_path.read_text())
    on_disk["tensors"]["llm.inputs_embeds"]["shape"] = [1, 1]
    manifest_path.write_text(json.dumps(on_disk))
    problems = validate_trace(bad_shape_dir)
    assert any("llm.inputs_embeds" in p for p in problems), problems

    missing_part_dir = tmp_path / "missing_part"
    write_trace(missing_part_dir, manifest, tensors)
    assert validate_trace(missing_part_dir) == []
    (missing_part_dir / "part0.bin").unlink()
    problems = validate_trace(missing_part_dir)
    assert problems
    assert any("part0.bin" in p for p in problems), problems


def test_no_torch_import() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, gr00t.explainer.synthetic, gr00t.explainer.schema; "
            "assert 'torch' not in sys.modules",
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        check=True,
    )


def test_cli_synthetic_and_validate(tmp_path: Path) -> None:
    out_dir = tmp_path / "t"
    env = _clean_env()
    subprocess.run(
        [
            sys.executable,
            CAPTURE_SCRIPT,
            "--synthetic",
            "--profile",
            "lite",
            "--output-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    assert (out_dir / "manifest.json").exists()
    assert validate_trace(out_dir) == []

    validated = subprocess.run(
        [sys.executable, CAPTURE_SCRIPT, "--validate", str(out_dir)], cwd=REPO_ROOT, env=env
    )
    assert validated.returncode == 0

    dry = subprocess.run(
        [sys.executable, CAPTURE_SCRIPT, "--synthetic", "--profile", "lite", "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    total_lines = [ln for ln in dry.stdout.splitlines() if ln.startswith("total")]
    assert len(total_lines) == 1, dry.stdout[-2000:]
