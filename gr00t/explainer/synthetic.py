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

"""Shape-correct synthetic traces (numpy only, no torch).

The website must render before anyone has run the real model. This module produces a
trace whose *structure* is exactly what the real capture emits (same tensor names, shapes,
token layout, DiT schedule, Euler bookkeeping) while the *values* are deliberately
meaningless:

- attention maps are softmaxes of Gaussian noise that only honor the structural masks
  (causal for the LLM, text-only / image-only columns for the DiT cross blocks);
- activations are Gaussian;
- the denoising trajectory follows the ideal rectified-flow straight line to a smooth
  random target, so ``actions_out == actions_in + dt * velocity`` holds exactly.

Nothing here should be read as a property of the model. The manifest carries
``source: "synthetic"`` and the website shows a badge for it.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.explainer.schema import (
    EMBODIMENT_W_PREVIEW_COLS,
    TraceWriter,
    action_dim_state,
    expected_tensor_specs,
    finalize_meta,
    vector_norms,
)


# Mirror of ``EMBODIMENT_TAG_TO_PROJECTOR_INDEX`` in
# ``gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`` (that module imports torch, which this
# module must not). Keep in sync when adding embodiments.
EMBODIMENT_TAG_TO_ID: dict[str, int] = {
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
    "robocasa_panda_omron": 10,
    "robocasa_gr1_tabletop": 10,
    "unitree_g1_sonic": 11,
    "oxe_droid_relative_eef_relative_joint": 24,
    "real_g1_relative_eef_relative_joints": 25,
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
}

DEFAULT_INSTRUCTION = (
    "put the white mug on the left plate and put the yellow and white mug on the right plate"
)

# Qwen3-VL special tokens (Cosmos-Reason2-2B shares the Qwen3-VL tokenizer family).
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
VISION_START, VISION_END, IMAGE_PAD = "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"

# A valid 1x1 white JPEG, used when Pillow is unavailable.
_PLACEHOLDER_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJ"
    "DRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ"
    "EBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEA"
    "AAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhED"
    "EQA/AKpgA//Z"
)


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


def default_libero_meta(num_images: int = 2) -> dict[str, Any]:
    """Meta block matching the shipped ``nvidia/GR00T-N1.7-3B`` checkpoint on LIBERO.

    Numbers come from the checkpoint ``config.json`` (DiT 32 x 32 x 48, ``select_layer``
    16, VL self-attention 4 x 32 x 64) and the Cosmos-Reason2-2B / Qwen3-VL-2B vision
    config (24 blocks, patch 16, 2 x 2 spatial merge, deepstack taps at blocks 5/11/17),
    not from the dataclass defaults in ``gr00t/configs/model/gr00t_n1d7.py``.
    """
    meta: dict[str, Any] = {
        "model_name": "nvidia/GR00T-N1.7-3B",
        "params_total": 3_144_016_000,
        "embodiment_tag": "libero_sim",
        "embodiment_id": EMBODIMENT_TAG_TO_ID["libero_sim"],
        "num_embodiments": 32,
        "embodiment_tag_to_id": dict(EMBODIMENT_TAG_TO_ID),
        "llm": {
            "hidden": 2048,
            "num_layers_total": 28,
            "select_layer": 16,
            "heads": 16,
            "kv_heads": 8,
            "head_dim": 128,
            "deepstack_layers": [0, 1, 2],
            "image_token_id": 151655,
            "seq_len": None,  # filled from the token sequence
        },
        "vit": {
            "depth": 24,
            "hidden": 1024,
            "heads": 16,
            "patch": 16,
            "merge": 2,
            "grid": [16, 16],
            "num_images": num_images,
            "featured_blocks": [0, 5, 11, 17, 23],
            "deepstack_blocks": [5, 11, 17],
        },
        "vlsa": {"num_layers": 4, "heads": 32, "head_dim": 64},
        "dit": {
            "num_layers": 32,
            "heads": 32,
            "head_dim": 48,
            "output_dim": 1024,
            "attend_text_every_n_blocks": 2,
        },
        "action": {
            "horizon_model": 40,
            "horizon_valid": 16,
            "dim_model": 132,
            "dim_valid": 7,
            "keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            "dims": [1, 1, 1, 1, 1, 1, 1],
        },
        "state": {
            "dim_model": 132,
            "dim_valid": 8,
            "history_length": 1,
            "keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            "dims": [1, 1, 1, 1, 1, 1, 2],
        },
        "flow": {
            "num_inference_timesteps": 4,
            "num_timestep_buckets": 1000,
            "beta_alpha": 1.5,
            "beta_beta": 1.0,
            "noise_s": 0.999,
        },
        "image_preproc": {"shortest_edge": 256, "crop_fraction": 0.95, "target": [256, 256]},
    }
    return finalize_meta(meta)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def synthetic_token_strings(instruction: str, num_images: int, tokens_per_image: int) -> list[str]:
    """Approximate the Qwen3-VL chat template for one user turn with images then text.

    Real tokenization splits words into sub-word pieces and may differ by a few tokens;
    the template scaffolding and the image-token blocks are exact.
    """
    toks = [IM_START, "user", "\n"]
    for _ in range(num_images):
        toks += [VISION_START] + [IMAGE_PAD] * tokens_per_image + [VISION_END]
    toks += instruction.split()
    toks += [IM_END, "\n"]
    return toks


def token_groups(strings: list[str]) -> list[dict[str, Any]]:
    """Split the token sequence into runs of ``template`` / ``image`` / ``text`` / ``end``."""
    kinds = []
    for i, s in enumerate(strings):
        if s == IMAGE_PAD:
            kinds.append("image")
        elif s == IM_END or (s == "\n" and i == len(strings) - 1):
            kinds.append("end")
        elif s.startswith("<|") or s in ("user", "system", "assistant", "\n"):
            kinds.append("template")
        else:
            kinds.append("text")
    groups: list[dict[str, Any]] = []
    image_index = 0
    start = 0
    for i in range(1, len(kinds) + 1):
        if i == len(kinds) or kinds[i] != kinds[start]:
            group: dict[str, Any] = {"kind": kinds[start], "start": start, "end": i}
            if kinds[start] == "image":
                group["image_index"] = image_index
                image_index += 1
            groups.append(group)
            start = i
    return groups


def mrope_position_ids(strings: list[str], grid_hw: tuple[int, int]) -> list[list[int]]:
    """Approximate Qwen-VL ``get_rope_index`` (3-D M-RoPE positions) for one sequence.

    Text tokens advance all three of (t, h, w) together. Each image block keeps ``t``
    fixed at the block's start index and lays ``h``/``w`` out on the merged patch grid;
    the next text token resumes at ``start + max(grid) ``.
    """
    rows_h, cols_w = grid_hw
    t_row, h_row, w_row = [], [], []
    cur = 0
    i = 0
    n = len(strings)
    while i < n:
        if strings[i] == IMAGE_PAD:
            j = i
            while j < n and strings[j] == IMAGE_PAD:
                j += 1
            count = j - i
            for k in range(count):
                t_row.append(cur)
                h_row.append(cur + k // cols_w)
                w_row.append(cur + k % cols_w)
            cur = cur + max(rows_h, cols_w, 1)
            i = j
        else:
            t_row.append(cur)
            h_row.append(cur)
            w_row.append(cur)
            cur += 1
            i += 1
    return [t_row, h_row, w_row]


# ---------------------------------------------------------------------------
# Random tensor builders
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def random_attention(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    allowed_cols: np.ndarray | None = None,
    causal: bool = False,
    temperature: float = 1.5,
) -> np.ndarray:
    """Softmax of Gaussian noise over the last axis, honoring structural masks only."""
    logits = rng.normal(size=shape).astype(np.float32) * temperature
    rows, cols = shape[-2], shape[-1]
    mask = np.ones((rows, cols), dtype=bool)
    if causal:
        mask &= np.tril(np.ones((rows, cols), dtype=bool))
    if allowed_cols is not None:
        mask &= np.asarray(allowed_cols, dtype=bool)[None, :]
    logits = np.where(mask, logits, -np.inf)
    probs = _softmax(logits)
    # Rows with no allowed column are all -inf -> nan; make them zero.
    return np.nan_to_num(probs, nan=0.0)


def smooth_trajectory(rng: np.random.Generator, horizon: int, dim: int) -> np.ndarray:
    """A smooth random target trajectory in [-1, 1] (sum of two low-frequency sinusoids)."""
    t = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[:, None]
    phase = rng.uniform(0, 2 * np.pi, size=(1, dim)).astype(np.float32)
    freq = rng.uniform(0.5, 1.5, size=(1, dim)).astype(np.float32)
    amp = rng.uniform(0.2, 0.6, size=(1, dim)).astype(np.float32)
    offset = rng.uniform(-0.3, 0.3, size=(1, dim)).astype(np.float32)
    traj = (
        offset + amp * np.sin(2 * np.pi * freq * t + phase) + 0.15 * np.sin(6 * np.pi * t + phase)
    )
    return np.clip(traj, -1.0, 1.0).astype(np.float32)


def synthetic_image(rng: np.random.Generator, size: int, seed_hue: float) -> np.ndarray:
    """A 256x256 RGB gradient with noise, clearly not a photo."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)
    r = 0.5 + 0.5 * np.sin(2 * np.pi * (xx + seed_hue))
    g = 0.5 + 0.5 * np.cos(2 * np.pi * (yy - seed_hue))
    b = 0.5 + 0.5 * np.sin(2 * np.pi * (xx * yy + seed_hue))
    img = np.stack([r, g, b], axis=-1)
    img += rng.normal(scale=0.05, size=img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def encode_jpeg_b64(image_hwc: np.ndarray, quality: int = 85) -> str:
    """Encode an HxWx3 uint8 array as base64 JPEG; falls back to a 1x1 placeholder."""
    try:
        from PIL import Image
    except ImportError:
        return _PLACEHOLDER_JPEG_B64
    buf = io.BytesIO()
    Image.fromarray(image_hwc).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def unnormalize_minmax(u: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Inverse of the [-1, 1] min-max normalization used for states and actions."""
    return (u + 1.0) / 2.0 * (q99 - q01) + q01


# ---------------------------------------------------------------------------
# Trace assembly
# ---------------------------------------------------------------------------


def build_synthetic_trace(
    meta: dict[str, Any] | None = None,
    profile: str = "lite",
    seed: int = 0,
    instruction: str = DEFAULT_INSTRUCTION,
    full_head_steps: tuple[int, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return ``(manifest_without_tensors_index, {tensor_name: array})``."""
    rng = np.random.default_rng(seed)
    meta = default_libero_meta() if meta is None else finalize_meta(meta)
    n_img = meta["vit"]["num_images"]
    tokens_per_image = meta["vit"]["tokens_per_image"]
    merge_grid = int(round(tokens_per_image**0.5))

    # --- tokens ------------------------------------------------------------------
    strings = synthetic_token_strings(instruction, n_img, tokens_per_image)
    seq = len(strings)
    meta["llm"]["seq_len"] = seq
    image_mask = np.array([s == IMAGE_PAD for s in strings], dtype=bool)
    attention_mask = np.ones(seq, dtype=bool)
    text_cols = ~image_mask & attention_mask
    image_cols = image_mask & attention_mask
    tokens = {
        "input_ids": [
            meta["llm"]["image_token_id"] if m else 1000 + i for i, m in enumerate(image_mask)
        ],
        "strings": strings,
        "groups": token_groups(strings),
        "image_mask": image_mask.tolist(),
        "attention_mask": attention_mask.astype(int).tolist(),
        "position_ids": mrope_position_ids(strings, (merge_grid, merge_grid)),
    }

    # --- observation ---------------------------------------------------------------
    state_keys = meta["state"]["keys"]
    state_dims = meta["state"]["dims"]
    state_center = {
        "x": -0.05,
        "y": 0.0,
        "z": 0.95,
        "roll": 3.1,
        "pitch": 0.0,
        "yaw": 0.0,
        "gripper": 0.0,
    }
    state_raw: dict[str, list[float]] = {}
    state_norm_params: dict[str, dict[str, list[float]]] = {}
    state_normalized: list[float] = []
    for key, dim in zip(state_keys, state_dims):
        c = state_center.get(key, 0.0)
        q01 = np.full(dim, c - 0.25, dtype=np.float32)
        q99 = np.full(dim, c + 0.25, dtype=np.float32)
        raw = (c + rng.uniform(-0.15, 0.15, size=dim)).astype(np.float32)
        norm = np.clip(2.0 * (raw - q01) / (q99 - q01) - 1.0, -1.0, 1.0)
        state_raw[key] = raw.tolist()
        state_norm_params[key] = {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "min": (q01 - 0.1).tolist(),
            "max": (q99 + 0.1).tolist(),
        }
        state_normalized += norm.tolist()

    action_keys = meta["action"]["keys"]
    action_dims = meta["action"]["dims"]
    action_span = {
        "x": 0.9375,
        "y": 0.9375,
        "z": 0.9375,
        "roll": 0.375,
        "pitch": 0.375,
        "yaw": 0.375,
        "gripper": 1.0,
    }
    action_norm_params = {
        key: {"q01": [-action_span.get(key, 1.0)] * dim, "q99": [action_span.get(key, 1.0)] * dim}
        for key, dim in zip(action_keys, action_dims)
    }
    q01_action = np.concatenate(
        [np.asarray(action_norm_params[k]["q01"], dtype=np.float32) for k in action_keys]
    )
    q99_action = np.concatenate(
        [np.asarray(action_norm_params[k]["q99"], dtype=np.float32) for k in action_keys]
    )

    image_size = int(meta["image_preproc"]["target"][0])
    images = []
    for i in range(n_img):
        key = ["image", "wrist_image"][i] if i < 2 else f"image_{i}"
        img = synthetic_image(rng, image_size, seed_hue=0.17 * (i + 1))
        images.append(
            {
                "key": key,
                "model_input_jpeg_b64": encode_jpeg_b64(img),
                "grid_thw": [1, meta["vit"]["grid"][0], meta["vit"]["grid"][1]],
            }
        )

    def decode(valid_norm: np.ndarray) -> dict[str, list[list[float]]]:
        decoded = unnormalize_minmax(valid_norm, q01_action, q99_action)
        out, start = {}, 0
        for key, dim in zip(action_keys, action_dims):
            out[key] = decoded[:, start : start + dim].tolist()
            start += dim
        return out

    # --- denoising bookkeeping ------------------------------------------------------
    flow = meta["flow"]
    n_steps = int(flow["num_inference_timesteps"])
    dt = float(flow["dt"])
    horizon, action_dim = meta["action"]["horizon_model"], meta["action"]["dim_model"]
    h_valid, d_valid = meta["action"]["horizon_valid"], meta["action"]["dim_valid"]
    target = np.zeros((horizon, action_dim), dtype=np.float32)
    target[:h_valid, :d_valid] = smooth_trajectory(rng, h_valid, d_valid)
    noise0 = rng.normal(size=(horizon, action_dim)).astype(np.float32)
    velocity = target - noise0  # the ideal rectified-flow velocity is constant along the line
    actions_by_step = [noise0]
    for _ in range(n_steps):
        actions_by_step.append(actions_by_step[-1] + dt * velocity)
    steps_inline = []
    for s in range(n_steps):
        a_in, a_out = actions_by_step[s], actions_by_step[s + 1]
        steps_inline.append(
            {
                "t_cont": flow["t_cont"][s],
                "t_disc": flow["t_disc"][s],
                "actions_in_norm_valid": a_in[:h_valid, :d_valid].tolist(),
                "velocity_valid": velocity[:h_valid, :d_valid].tolist(),
                "actions_out_norm_valid": a_out[:h_valid, :d_valid].tolist(),
                "actions_out_decoded": decode(a_out[:h_valid, :d_valid]),
            }
        )
    final_valid = actions_by_step[-1][:h_valid, :d_valid]
    ground_truth = decode(target[:h_valid, :d_valid])

    t_train = float(rng.beta(flow["beta_alpha"], flow["beta_beta"]))
    t_train = (1.0 - t_train) * float(flow["noise_s"])
    train_noise = rng.normal(size=(h_valid, d_valid)).astype(np.float32)
    training_example = {
        "t": t_train,
        "noise_valid": train_noise.tolist(),
        "noisy_valid": (
            (1 - t_train) * train_noise + t_train * target[:h_valid, :d_valid]
        ).tolist(),
        "target_velocity_valid": (target[:h_valid, :d_valid] - train_noise).tolist(),
    }

    n_emb = int(meta["num_embodiments"])
    embodiment_bank = {
        "state_encoder_l1_norm": rng.uniform(5, 20, size=n_emb).tolist(),
        "action_encoder_w1_norm": rng.uniform(5, 20, size=n_emb).tolist(),
        "action_decoder_l2_norm": rng.uniform(5, 20, size=n_emb).tolist(),
        "selected": int(meta["embodiment_id"]),
    }

    # --- large tensors, driven by the schema inventory ---------------------------------
    specs = expected_tensor_specs(meta, profile, full_head_steps=full_head_steps)
    tensors: dict[str, np.ndarray] = {}
    schedule_kind = {e["idx"]: e["kind"] for e in meta["dit"]["schedule"]}
    sa_len = 1 + horizon

    def attention_for(name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name.startswith("llm."):
            return random_attention(rng, shape, causal=True)
        if name.startswith("vlsa.") or name.startswith("vit."):
            return random_attention(rng, shape)
        # step{s}.dit.block{b}.attn[_mean]
        block = int(name.split(".dit.block")[1].split(".")[0])
        kind = schedule_kind[block]
        if kind == "self":
            return random_attention(rng, shape, temperature=1.0)
        allowed = text_cols if kind == "cross_text" else image_cols
        return random_attention(rng, shape, allowed_cols=allowed)

    for name, (shape, dtype) in specs.items():
        if name in tensors:
            continue
        if name.endswith(".attn_mean"):
            per_head_name = name[: -len("_mean")]
            if per_head_name in specs:
                per_head = attention_for(per_head_name, specs[per_head_name][0])
                tensors[per_head_name] = per_head
                tensors[name] = per_head.mean(axis=0)
            else:
                tensors[name] = attention_for(name, shape)
        elif name.endswith(".attn"):
            tensors[name] = attention_for(name, shape)
        elif name.endswith(".attn_received"):
            received = rng.gamma(2.0, 1.0, size=shape).astype(np.float32)
            tensors[name] = received / received.sum(axis=-1, keepdims=True) * shape[-1]
        elif name.endswith(".norms") or name.endswith(".delta_norms"):
            scale = 3.0 if ".delta_norms" in name else 30.0
            tensors[name] = np.abs(rng.normal(loc=scale, scale=scale / 4, size=shape)).astype(
                np.float32
            )
        elif name.endswith(".actions_in"):
            s = int(name.split(".")[0][len("step") :])
            tensors[name] = actions_by_step[s]
        elif name.endswith(".actions_out"):
            s = int(name.split(".")[0][len("step") :])
            tensors[name] = actions_by_step[s + 1]
        elif name.endswith(".velocity"):
            tensors[name] = velocity
        elif name.endswith(".pred"):
            pred = np.zeros((sa_len, action_dim), dtype=np.float32)
            pred[0] = rng.normal(scale=0.1, size=action_dim)
            pred[1:] = velocity
            tensors[name] = pred
        elif name == "head.state_input":
            state_input = np.zeros(action_dim_state(meta), dtype=np.float32)
            state_input[: len(state_normalized)] = state_normalized
            tensors[name] = state_input
        elif name == "head.embodiment_w_preview":
            tensors[name] = (
                0.02 * rng.normal(size=(action_dim_state(meta), EMBODIMENT_W_PREVIEW_COLS))
            ).astype(np.float32)
        else:
            tensors[name] = rng.normal(size=shape).astype(np.float32)

    # Make the LLM per-token norms tell a consistent story with the wide activations.
    tensors["llm.layer0.norms"] = vector_norms(tensors["llm.inputs_embeds"])
    tensors[f"llm.layer{meta['llm']['select_layer']}.norms"] = vector_norms(
        tensors["llm.backbone_features"]
    )

    manifest: dict[str, Any] = {
        "source": "synthetic",
        "profile": profile,
        "created_at": None,
        "env": {"numpy": np.__version__, "generator": "gr00t.explainer.synthetic", "seed": seed},
        "full_head_steps": list(full_head_steps) if full_head_steps else [n_steps - 1],
        "meta": meta,
        "observation": {
            "instruction_raw": instruction,
            "instruction_formalized": instruction.lower(),
            "images": images,
            "state_raw": state_raw,
            "state_norm_params": state_norm_params,
            "state_normalized": state_normalized,
            "action_norm_params": action_norm_params,
            "ground_truth_action": ground_truth,
        },
        "tokens": tokens,
        "denoising": {"steps": steps_inline, "final_decoded": decode(final_valid)},
        "training_example": training_example,
        "embodiment_bank": embodiment_bank,
    }
    return manifest, tensors


def write_trace(
    out_dir: str | Path, manifest: dict[str, Any], tensors: dict[str, np.ndarray]
) -> Path:
    """Write a manifest + tensors pair produced by :func:`build_synthetic_trace` (or the real
    capture) to ``out_dir`` and return the manifest path."""
    specs = expected_tensor_specs(
        manifest["meta"],
        manifest["profile"],
        full_head_steps=tuple(manifest.get("full_head_steps", ())),
    )
    writer = TraceWriter(out_dir)
    # Schema order first (stable layout), then any extras the capture added.
    for name, (_, dtype) in specs.items():
        writer.add(name, tensors[name], dtype)
    for name, arr in tensors.items():
        if name not in specs:
            dtype = "u8" if arr.dtype == np.uint8 else "f16"
            writer.add(name, arr, dtype)
    return writer.finish(manifest)
