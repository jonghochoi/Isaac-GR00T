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

"""The ``groot-trace/1`` on-disk format.

A trace directory contains ``manifest.json`` and one or more ``part{k}.bin`` files.
Small values (config, token strings, the 16x7 action chunks, ...) live inline in the
manifest. Large arrays are stored raw, little-endian, in the part files and indexed by
``manifest["tensors"][name] = {dtype, shape, part, offset, byte_length}``.

Encodings:

- ``f16``: IEEE half floats (vectors, norms).
- ``u8``: probabilities in ``[0, 1]`` scaled by 255 and rounded (attention maps).
- ``f32`` / ``i32``: used only when precision matters.

:func:`expected_tensor_specs` is the single source of truth for which tensors a trace of a
given ``profile`` must contain and their shapes. The synthetic generator, the real capture,
the validator and the tests all derive from it, so the website can rely on it too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


SCHEMA_VERSION = "groot-trace/1"
PROFILES = ("full", "lite")
DTYPES: dict[str, np.dtype] = {
    "f16": np.dtype("<f2"),
    "u8": np.dtype("u1"),
    "f32": np.dtype("<f4"),
    "i32": np.dtype("<i4"),
}
DEFAULT_PART_BYTES = 8 << 20
EMBODIMENT_W_PREVIEW_COLS = 64

# Manifest keys every trace must carry (checked by ``validate_trace``).
REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "source",
    "profile",
    "meta",
    "observation",
    "tokens",
    "denoising",
    "embodiment_bank",
    "tensors",
)
REQUIRED_META_KEYS = ("llm", "vit", "vlsa", "dit", "action", "state", "flow", "embodiment_id")


# ---------------------------------------------------------------------------
# Model-structure helpers (pure functions of the config)
# ---------------------------------------------------------------------------


def dit_schedule(num_layers: int, attend_text_every_n_blocks: int) -> list[dict[str, Any]]:
    """Replicate the block routing of ``AlternateVLDiT.forward``.

    Odd blocks run self-attention over the [state, actions] tokens. Even blocks run
    cross-attention into the vision-language tokens, alternating between the *text*
    (non-image) tokens and the *image* tokens.

    Mirrors ``gr00t/model/modules/dit.py`` (``AlternateVLDiT.forward``): keep the two in
    sync if the routing ever changes.
    """
    schedule = []
    for idx in range(num_layers):
        if idx % 2 == 1:
            kind = "self"
        elif idx % (2 * attend_text_every_n_blocks) == 0:
            kind = "cross_text"
        else:
            kind = "cross_image"
        schedule.append({"idx": idx, "kind": kind})
    return schedule


def flow_timesteps(num_inference_timesteps: int, num_timestep_buckets: int) -> dict[str, Any]:
    """The continuous and discretized timesteps visited by the Euler denoising loop."""
    t_cont = [t / float(num_inference_timesteps) for t in range(num_inference_timesteps)]
    t_disc = [int(t * num_timestep_buckets) for t in t_cont]
    return {"dt": 1.0 / num_inference_timesteps, "t_cont": t_cont, "t_disc": t_disc}


def finalize_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Fill in the derived fields (DiT schedule, timesteps, inner dims) of a meta dict."""
    dit = meta["dit"]
    dit["schedule"] = dit_schedule(dit["num_layers"], dit["attend_text_every_n_blocks"])
    dit.setdefault("inner_dim", dit["heads"] * dit["head_dim"])
    flow = meta["flow"]
    flow.update(flow_timesteps(flow["num_inference_timesteps"], flow["num_timestep_buckets"]))
    vit = meta["vit"]
    vit.setdefault("patches_per_image", (vit["grid"][0] * vit["grid"][1]))
    vit.setdefault("tokens_per_image", vit["patches_per_image"] // (vit["merge"] ** 2))
    return meta


# ---------------------------------------------------------------------------
# Tensor inventory
# ---------------------------------------------------------------------------


def expected_tensor_specs(
    meta: dict[str, Any], profile: str, full_head_steps: tuple[int, ...] | None = None
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Return ``{tensor_name: (shape, dtype)}`` for a trace of the given profile.

    ``lite`` is what the committed synthetic fixture ships (a few MB); ``full`` adds
    per-head attention maps and the wide intermediate activations of every denoising
    step. ``full_head_steps`` selects which denoising steps get per-head DiT maps in the
    ``full`` profile (default: the last step only, they are the largest tensors).
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    full = profile == "full"

    llm, vit, vlsa, dit = meta["llm"], meta["vit"], meta["vlsa"], meta["dit"]
    seq = int(llm["seq_len"])
    hidden = int(llm["hidden"])
    n_llm = int(llm["select_layer"])
    n_img = int(vit["num_images"])
    patches = int(vit["patches_per_image"])
    tokens_per_image = int(vit["tokens_per_image"])
    horizon = int(meta["action"]["horizon_model"])
    action_dim = int(meta["action"]["dim_model"])
    sa_len = 1 + horizon
    inner = int(dit["inner_dim"])
    out_dim = int(dit["output_dim"])
    n_steps = int(meta["flow"]["num_inference_timesteps"])
    if full_head_steps is None:
        full_head_steps = (n_steps - 1,)

    specs: dict[str, tuple[tuple[int, ...], str]] = {}

    # --- Vision encoder ------------------------------------------------------
    for b in vit["featured_blocks"]:
        specs[f"vit.block{b}.attn_mean"] = ((n_img, patches, patches), "u8")
    for b in range(int(vit["depth"])):
        specs[f"vit.block{b}.attn_received"] = ((n_img, patches), "f16")
    specs["vit.image_embeds"] = ((n_img * tokens_per_image, hidden), "f16")

    # --- Truncated LLM -------------------------------------------------------
    specs["llm.inputs_embeds"] = ((seq, hidden), "f16")
    for i in range(n_llm + 1):
        specs[f"llm.layer{i}.norms"] = ((seq,), "f16")
    for i in range(1, n_llm + 1):
        specs[f"llm.layer{i}.delta_norms"] = ((seq,), "f16")
    for i in range(n_llm):
        specs[f"llm.layer{i}.attn_mean"] = ((seq, seq), "u8")
        if full:
            specs[f"llm.layer{i}.attn"] = ((int(llm["heads"]), seq, seq), "u8")
    specs["llm.backbone_features"] = ((seq, hidden), "f16")

    # --- VL LayerNorm + self-attention ----------------------------------------
    if full:
        specs["vlsa.vlln_out"] = ((seq, hidden), "f16")
    for i in range(int(vlsa["num_layers"])):
        specs[f"vlsa.layer{i}.attn_mean"] = ((seq, seq), "u8")
        if full:
            specs[f"vlsa.layer{i}.attn"] = ((int(vlsa["heads"]), seq, seq), "u8")
    specs["vlsa.vl_embs"] = ((seq, hidden), "f16")

    # --- Action head, step independent ----------------------------------------
    specs["head.state_input"] = ((action_dim_state(meta),), "f16")
    specs["head.state_features"] = ((inner,), "f16")
    specs["head.embodiment_w_preview"] = (
        (action_dim_state(meta), EMBODIMENT_W_PREVIEW_COLS),
        "f16",
    )
    if full:
        specs["head.position_embedding"] = ((horizon, inner), "f16")

    # --- Denoising steps -------------------------------------------------------
    for s in range(n_steps):
        p = f"step{s}"
        specs[f"{p}.actions_in"] = ((horizon, action_dim), "f16")
        specs[f"{p}.velocity"] = ((horizon, action_dim), "f16")
        specs[f"{p}.actions_out"] = ((horizon, action_dim), "f16")
        specs[f"{p}.pred"] = ((sa_len, action_dim), "f16")
        specs[f"{p}.temb"] = ((inner,), "f16")
        wide = full or s == n_steps - 1
        if full:
            specs[f"{p}.action_features"] = ((horizon, inner), "f16")
        if wide:
            specs[f"{p}.sa_embs"] = ((sa_len, inner), "f16")
            specs[f"{p}.dit.output"] = ((sa_len, out_dim), "f16")
        for b in range(int(dit["num_layers"]) + 1):
            specs[f"{p}.dit.block{b}.norms"] = ((sa_len,), "f16")
        for entry in dit["schedule"]:
            b, kind = entry["idx"], entry["kind"]
            cols = sa_len if kind == "self" else seq
            specs[f"{p}.dit.block{b}.attn_mean"] = ((sa_len, cols), "u8")
            if full and s in full_head_steps:
                specs[f"{p}.dit.block{b}.attn"] = ((int(dit["heads"]), sa_len, cols), "u8")
    return specs


def action_dim_state(meta: dict[str, Any]) -> int:
    """Flattened state input dim: ``max_state_dim * state_history_length``."""
    return int(meta["state"]["dim_model"]) * int(meta["state"].get("history_length", 1))


def estimate_bytes(specs: dict[str, tuple[tuple[int, ...], str]]) -> int:
    return sum(int(np.prod(shape)) * DTYPES[dtype].itemsize for shape, dtype in specs.values())


def format_spec_table(specs: dict[str, tuple[tuple[int, ...], str]]) -> str:
    """Human-readable inventory, one line per tensor (used by ``--dry-run``)."""
    lines = [f"{'tensor':48s} {'shape':>22s} {'dtype':>5s} {'bytes':>12s}"]
    for name, (shape, dtype) in specs.items():
        nbytes = int(np.prod(shape)) * DTYPES[dtype].itemsize
        lines.append(f"{name:48s} {str(tuple(shape)):>22s} {dtype:>5s} {nbytes:>12,d}")
    lines.append(f"{'total':48s} {'':>22s} {'':>5s} {estimate_bytes(specs):>12,d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def quantize_u8(x: np.ndarray) -> np.ndarray:
    """Map probabilities in ``[0, 1]`` to ``uint8`` (``round(p * 255)``)."""
    x = np.asarray(x, dtype=np.float32)
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def cast_for_dtype(x: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "u8":
        arr = np.asarray(x)
        if arr.dtype != np.uint8:
            arr = quantize_u8(arr)
        return np.ascontiguousarray(arr, dtype=DTYPES["u8"])
    return np.ascontiguousarray(np.asarray(x), dtype=DTYPES[dtype])


def vector_norms(x: np.ndarray) -> np.ndarray:
    """L2 norm over the last axis (``(..., D) -> (...)``) in float32."""
    return np.linalg.norm(np.asarray(x, dtype=np.float32), axis=-1)


# ---------------------------------------------------------------------------
# Writer / reader
# ---------------------------------------------------------------------------


class TraceWriter:
    """Append tensors to ``part{k}.bin`` files and finish with ``manifest.json``.

    Example::

        writer = TraceWriter(out_dir)
        writer.add("llm.inputs_embeds", embeds, "f16")
        writer.add_attn("llm.layer0.attn_mean", probs)
        writer.finish(manifest)
    """

    def __init__(self, out_dir: str | Path, part_bytes: int = DEFAULT_PART_BYTES):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.part_bytes = int(part_bytes)
        self.tensors: dict[str, dict[str, Any]] = {}
        self._part_idx = 0
        self._part_buffers: list[bytearray] = [bytearray()]

    @property
    def names(self) -> list[str]:
        return list(self.tensors)

    def add(self, name: str, array: np.ndarray, dtype: str) -> None:
        if name in self.tensors:
            raise ValueError(f"tensor {name!r} was already added")
        arr = cast_for_dtype(array, dtype)
        payload = arr.tobytes(order="C")
        buf = self._part_buffers[-1]
        if len(buf) > 0 and len(buf) + len(payload) > self.part_bytes:
            self._part_buffers.append(bytearray())
            buf = self._part_buffers[-1]
        part = len(self._part_buffers) - 1
        offset = len(buf)
        buf.extend(payload)
        self.tensors[name] = {
            "dtype": dtype,
            "shape": [int(d) for d in arr.shape],
            "part": f"part{part}.bin",
            "offset": offset,
            "byte_length": len(payload),
        }

    def add_attn(self, name: str, probs: np.ndarray) -> None:
        """Store an attention probability map as ``u8``."""
        self.add(name, quantize_u8(probs), "u8")

    def finish(self, manifest: dict[str, Any]) -> Path:
        for i, buf in enumerate(self._part_buffers):
            (self.out_dir / f"part{i}.bin").write_bytes(bytes(buf))
        manifest = dict(manifest)
        manifest["schema_version"] = SCHEMA_VERSION
        manifest["tensors"] = self.tensors
        manifest["parts"] = [f"part{i}.bin" for i in range(len(self._part_buffers))]
        path = self.out_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=None, separators=(",", ":")))
        return path


def load_manifest(trace_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(trace_dir) / "manifest.json").read_text())


def read_tensor(trace_dir: str | Path, manifest: dict[str, Any], name: str) -> np.ndarray:
    """Read one tensor back from a trace directory (used by tests and the validator)."""
    spec = manifest["tensors"][name]
    data = (Path(trace_dir) / spec["part"]).read_bytes()
    chunk = data[spec["offset"] : spec["offset"] + spec["byte_length"]]
    arr = np.frombuffer(chunk, dtype=DTYPES[spec["dtype"]])
    return arr.reshape(spec["shape"])


def iter_tensors(
    trace_dir: str | Path, manifest: dict[str, Any]
) -> Iterator[tuple[str, np.ndarray]]:
    for name in manifest["tensors"]:
        yield name, read_tensor(trace_dir, manifest, name)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_trace(trace_dir: str | Path, sample_rows: int = 8) -> list[str]:
    """Check a trace directory against the schema. Returns a list of problems (empty = ok)."""
    trace_dir = Path(trace_dir)
    problems: list[str] = []
    manifest_path = trace_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"manifest.json is not valid JSON: {exc}"]

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in manifest:
            problems.append(f"manifest missing top-level key {key!r}")
    if problems:
        return problems
    if manifest["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version {manifest['schema_version']!r} != {SCHEMA_VERSION!r}")
    if manifest["source"] not in ("real", "synthetic"):
        problems.append(f"source must be 'real' or 'synthetic', got {manifest['source']!r}")
    if manifest["profile"] not in PROFILES:
        problems.append(f"profile must be one of {PROFILES}, got {manifest['profile']!r}")
        return problems
    meta = manifest["meta"]
    for key in REQUIRED_META_KEYS:
        if key not in meta:
            problems.append(f"meta missing key {key!r}")
    if problems:
        return problems

    # Token block consistency.
    tokens = manifest["tokens"]
    seq = int(meta["llm"]["seq_len"])
    for key in ("input_ids", "strings", "image_mask", "attention_mask"):
        if key not in tokens:
            problems.append(f"tokens missing {key!r}")
        elif len(tokens[key]) != seq:
            problems.append(f"tokens.{key} has length {len(tokens[key])}, expected {seq}")
    pos = tokens.get("position_ids")
    if pos is None or len(pos) != 3 or any(len(row) != seq for row in pos):
        problems.append("tokens.position_ids must be a 3 x seq_len nested list")
    n_img_tokens = sum(1 for m in tokens.get("image_mask", []) if m)
    expected_img_tokens = int(meta["vit"]["num_images"]) * int(meta["vit"]["tokens_per_image"])
    if n_img_tokens != expected_img_tokens:
        problems.append(
            f"tokens.image_mask marks {n_img_tokens} image tokens, expected {expected_img_tokens}"
        )

    # Denoising block consistency.
    steps = manifest["denoising"].get("steps", [])
    if len(steps) != int(meta["flow"]["num_inference_timesteps"]):
        problems.append(
            f"denoising.steps has {len(steps)} entries, expected "
            f"{meta['flow']['num_inference_timesteps']}"
        )

    # Tensor inventory.
    full_head_steps = tuple(manifest.get("full_head_steps", ()))
    expected = expected_tensor_specs(
        meta, manifest["profile"], full_head_steps=full_head_steps or None
    )
    tensors = manifest["tensors"]
    for name, (shape, dtype) in expected.items():
        spec = tensors.get(name)
        if spec is None:
            problems.append(f"missing tensor {name!r}")
            continue
        if tuple(spec["shape"]) != tuple(shape):
            problems.append(f"tensor {name!r} shape {spec['shape']} != expected {list(shape)}")
        if spec["dtype"] != dtype:
            problems.append(f"tensor {name!r} dtype {spec['dtype']!r} != expected {dtype!r}")

    # Byte ranges and content sanity.
    part_sizes: dict[str, int] = {}
    for name, spec in tensors.items():
        part = spec["part"]
        if part not in part_sizes:
            path = trace_dir / part
            part_sizes[part] = path.stat().st_size if path.exists() else -1
        size = part_sizes[part]
        if size < 0:
            problems.append(f"tensor {name!r} references missing part file {part}")
            continue
        end = spec["offset"] + spec["byte_length"]
        if end > size:
            problems.append(f"tensor {name!r} byte range [{spec['offset']}, {end}) exceeds {part}")
            continue
        n_expected = int(np.prod(spec["shape"])) * DTYPES[spec["dtype"]].itemsize
        if n_expected != spec["byte_length"]:
            problems.append(
                f"tensor {name!r} byte_length {spec['byte_length']} != shape*itemsize {n_expected}"
            )
            continue
        if len(problems) > 50:
            problems.append("... too many problems, stopping early")
            break
        arr = read_tensor(trace_dir, manifest, name)
        if spec["dtype"] == "f16" and not np.all(np.isfinite(arr.astype(np.float32))):
            problems.append(f"tensor {name!r} contains non-finite values")
        if spec["dtype"] == "u8" and name.endswith((".attn_mean", ".attn")) and arr.ndim >= 2:
            rows = arr.reshape(-1, arr.shape[-1])[:sample_rows].astype(np.int32)
            sums = rows.sum(axis=1)
            # Rows attend over a masked subset; padded query rows may legitimately be
            # all-zero, so only non-empty rows are checked. u8 rounding drops mass from
            # wide rows with many tiny entries, so the band is deliberately loose: it is
            # meant to catch un-normalized maps (raw logits), not rounding.
            bad = [int(s) for s in sums if s != 0 and not (180 <= s <= 270)]
            if bad:
                problems.append(f"tensor {name!r} attention rows do not sum to ~255: {bad[:4]}")
    return problems
