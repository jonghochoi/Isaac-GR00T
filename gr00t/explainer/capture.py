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

"""Record a real GR00T N1.7 forward pass as a ``groot-trace/1`` directory.

How the recording works (nothing in the model is modified permanently; every patch is
undone in a ``finally``):

* **Backbone (Qwen3-VL).** The vision and text attention modules call the module-level
  function ``eager_attention_forward`` when the attention implementation is ``"eager"``.
  We switch the backbone to eager, wrap that function so it hands us the softmax
  probabilities, and identify the calling block by the ``module`` argument it receives.
  Per-layer hidden-state norms come from ordinary forward hooks on the decoder layers.
* **Action head (diffusers ``Attention``).** Every ``BasicTransformerBlock.attn1`` in the
  DiT and in the VL self-attention transformer gets a :class:`RecordingAttnProcessor`
  that computes attention the classic way (``softmax(QK^T * scale + mask)``) and records
  the probabilities before applying them to ``V``.
* **Denoising loop.** Instead of calling ``Gr00tN1d7ActionHead.get_action`` we re-run the
  same Euler loop step by step (see ``get_action_with_features`` in
  ``gr00t/model/gr00t_n1d7/gr00t_n1d7.py``) so that every intermediate tensor is in hand.
  ``verify=True`` re-runs the untouched ``get_action`` under the same seed and checks that
  both produce the same actions.

This module needs torch, transformers, diffusers and a checkpoint; it is imported lazily by
``scripts/explainer/capture_trace.py`` so the synthetic path stays torch-free.
"""

from __future__ import annotations

import base64
import contextlib
import datetime as _dt
import io
import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterator

import numpy as np
import torch

from gr00t.explainer.schema import (
    EMBODIMENT_W_PREVIEW_COLS,
    expected_tensor_specs,
    finalize_meta,
    vector_norms,
)
from gr00t.explainer.synthetic import encode_jpeg_b64, token_groups, write_trace


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention recording
# ---------------------------------------------------------------------------


class AttentionRecorder:
    """Collects head-mean (and optionally per-head) attention maps keyed by tensor name.

    ``record(key, probs)`` takes ``probs`` of shape ``(B, H, T, S)`` (batch 0 is used).
    Vision blocks call once per image chunk; those calls are accumulated under the same
    key and stacked to ``(num_images, P, P)`` on read.
    """

    def __init__(self, keep_per_head: Callable[[str], bool] | None = None):
        self.keep_per_head = keep_per_head or (lambda key: False)
        self.mean: dict[str, list[np.ndarray]] = {}
        self.per_head: dict[str, list[np.ndarray]] = {}
        self.enabled = True

    def record(self, key: str, probs: torch.Tensor) -> None:
        if not self.enabled:
            return
        p = probs[0].detach().to(torch.float32).cpu().numpy()  # (H, T, S)
        self.mean.setdefault(key, []).append(p.mean(axis=0))
        if self.keep_per_head(key):
            self.per_head.setdefault(key, []).append(p)

    def get_mean(self, key: str) -> np.ndarray:
        chunks = self.mean[key]
        return chunks[0] if len(chunks) == 1 else np.stack(chunks, axis=0)

    def get_per_head(self, key: str) -> np.ndarray | None:
        chunks = self.per_head.get(key)
        if not chunks:
            return None
        return chunks[0] if len(chunks) == 1 else np.stack(chunks, axis=0)

    def has(self, key: str) -> bool:
        return key in self.mean

    def clear(self, prefix: str | None = None) -> None:
        for store in (self.mean, self.per_head):
            for k in [k for k in store if prefix is None or k.startswith(prefix)]:
                del store[k]


class RecordingAttnProcessor:
    """A diffusers ``Attention`` processor that records softmax probabilities.

    Behaves like ``diffusers.models.attention_processor.AttnProcessor`` (query/key/value
    projections, ``softmax(QK^T * scale + mask)``, ``V`` weighting, output projection) with
    two differences: a boolean ``(B, S)`` mask -- what ``AlternateVLDiT`` passes -- is turned
    into an additive float mask, and the probabilities are computed in float32 and handed
    to ``recorder.record(key_fn(), probs)`` with shape ``(B, H, T, S)``.
    """

    def __init__(self, recorder: AttentionRecorder, key_fn: Callable[[], str]):
        self.recorder = recorder
        self.key_fn = key_fn

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        batch_size, query_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key_len = encoder_hidden_states.shape[1]
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)  # (B*H, T, d)
        key = attn.head_to_batch_dim(key)  # (B*H, S, d)
        value = attn.head_to_batch_dim(value)

        scores = torch.baddbmm(
            torch.zeros(1, 1, 1, dtype=torch.float32, device=query.device),
            query.float(),
            key.float().transpose(-1, -2),
            beta=0,
            alpha=attn.scale,
        )  # (B*H, T, S) in float32
        additive = bool_mask_to_additive(attention_mask, batch_size, attn.heads, key_len, scores)
        if additive is not None:
            scores = scores + additive
        probs = scores.softmax(dim=-1)
        self.recorder.record(self.key_fn(), probs.view(batch_size, attn.heads, query_len, key_len))

        hidden_states = torch.bmm(probs.to(value.dtype), value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def bool_mask_to_additive(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    heads: int,
    key_len: int,
    like: torch.Tensor,
) -> torch.Tensor | None:
    """Turn a ``(B, S)`` boolean keep-mask (or an additive float mask) into a float32
    ``(B*H, 1, S)`` additive mask broadcastable against ``(B*H, T, S)`` scores."""
    if attention_mask is None:
        return None
    if attention_mask.dtype == torch.bool:
        additive = torch.zeros(
            attention_mask.shape, dtype=torch.float32, device=attention_mask.device
        )
        additive = additive.masked_fill(~attention_mask, torch.finfo(torch.float32).min)
    else:
        additive = attention_mask.to(torch.float32)
    additive = additive.reshape(batch_size, -1, additive.shape[-1])  # (B, 1 or T, S)
    if additive.shape[-1] != key_len:
        raise ValueError(f"mask length {additive.shape[-1]} != key length {key_len}")
    additive = additive.repeat_interleave(heads, dim=0)  # (B*H, 1 or T, S)
    return additive.to(like.device)


@contextlib.contextmanager
def recording_attn_processors(
    blocks: list[tuple[str, torch.nn.Module]],
    recorder: AttentionRecorder,
    prefix_fn: Callable[[], str],
) -> Iterator[None]:
    """Swap ``attn1.processor`` of each ``(name, BasicTransformerBlock)`` for a recorder and
    restore the originals on exit. Keys are ``f"{prefix_fn()}{name}"``."""
    saved = []
    try:
        for name, block in blocks:
            attn = block.attn1
            saved.append((attn, attn.processor))
            attn.set_processor(
                RecordingAttnProcessor(recorder, key_fn=(lambda n=name: f"{prefix_fn()}{n}"))
            )
        yield
    finally:
        for attn, processor in saved:
            attn.set_processor(processor)


@contextlib.contextmanager
def recording_qwen_attention(
    qwen_model: torch.nn.Module, recorder: AttentionRecorder, module_to_key: dict[int, str]
) -> Iterator[None]:
    """Force eager attention on a Qwen3-VL model and wrap its ``eager_attention_forward``
    so every call records ``attn_weights`` under ``module_to_key[id(module)]``."""
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen_mod

    original_fn = qwen_mod.eager_attention_forward
    previous_impl = _set_attn_implementation(qwen_model, "eager")

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        attn_output, attn_weights = original_fn(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
        )
        tensor_key = module_to_key.get(id(module))
        if tensor_key is not None and attn_weights is not None:
            recorder.record(tensor_key, attn_weights)
        return attn_output, attn_weights

    qwen_mod.eager_attention_forward = wrapped
    try:
        yield
    finally:
        qwen_mod.eager_attention_forward = original_fn
        if previous_impl is not None:
            _set_attn_implementation(qwen_model, previous_impl)


def _set_attn_implementation(model: torch.nn.Module, impl: str) -> str | None:
    """Switch the attention implementation on a HF model and all sub-configs.

    Returns the previous implementation (best effort) so it can be restored.
    """
    previous = getattr(model.config, "_attn_implementation", None)
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation(impl)
        except Exception as exc:  # pragma: no cover - depends on transformers version
            logger.warning(f"set_attn_implementation failed ({exc}); patching configs directly")
    # Attention modules read ``self.config._attn_implementation`` from the sub-config
    # object they were built with; set it on every config we can find.
    seen = set()
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        try:
            cfg._attn_implementation = impl
        except Exception:
            try:
                cfg._attn_implementation_internal = impl
            except Exception:
                pass
    return previous


# ---------------------------------------------------------------------------
# Hooks on the backbone
# ---------------------------------------------------------------------------


class BackboneCapture:
    """Forward hooks recording per-layer LLM hidden states, ViT outputs and LLM inputs."""

    def __init__(self, qwen_model: torch.nn.Module):
        self.qwen = qwen_model
        self.visual = qwen_model.model.visual
        self.language_model = qwen_model.model.language_model
        self.llm_layer_inputs: dict[int, torch.Tensor] = {}
        self.llm_last_output: torch.Tensor | None = None
        self.image_embeds: torch.Tensor | None = None
        self.deepstack: list[torch.Tensor] = []
        self.inputs_embeds: torch.Tensor | None = None
        self.position_ids: torch.Tensor | None = None
        self.visual_pos_masks: torch.Tensor | None = None
        self._handles: list[Any] = []

    def __enter__(self) -> "BackboneCapture":
        layers = self.language_model.layers
        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_pre_hook(self._make_input_hook(i)))
        self._handles.append(layers[-1].register_forward_hook(self._last_output_hook))
        self._handles.append(self.visual.register_forward_hook(self._visual_hook))
        self._handles.append(
            self.language_model.register_forward_pre_hook(self._llm_input_hook, with_kwargs=True)
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_input_hook(self, idx: int):
        def hook(module, args):
            hs = args[0] if args else None
            if hs is not None:
                # clone: deepstack injection mutates layer outputs in place.
                self.llm_layer_inputs[idx] = hs.detach().clone()

        return hook

    def _last_output_hook(self, module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        self.llm_last_output = out.detach().clone()

    def _visual_hook(self, module, args, output):
        if isinstance(output, tuple):
            self.image_embeds = output[0].detach().clone()
            if len(output) > 1 and output[1] is not None:
                self.deepstack = [d.detach().clone() for d in output[1]]
        else:
            self.image_embeds = output.detach().clone()

    def _llm_input_hook(self, module, args, kwargs):
        ie = kwargs.get("inputs_embeds")
        if ie is None and args:
            ie = args[0]
        if ie is not None:
            self.inputs_embeds = ie.detach().clone()
        for name in ("position_ids", "visual_pos_masks"):
            v = kwargs.get(name)
            if v is not None:
                setattr(self, name, v.detach().clone())


# ---------------------------------------------------------------------------
# Meta from a loaded model / a config.json
# ---------------------------------------------------------------------------


def _qwen_dims(qwen_config: Any) -> dict[str, Any]:
    text = getattr(qwen_config, "text_config", qwen_config)
    vision = getattr(qwen_config, "vision_config", None)
    heads = int(text.num_attention_heads)
    return {
        "llm": {
            "hidden": int(text.hidden_size),
            "num_layers_total": int(text.num_hidden_layers),
            "heads": heads,
            "kv_heads": int(getattr(text, "num_key_value_heads", heads)),
            "head_dim": int(getattr(text, "head_dim", text.hidden_size // heads)),
            "image_token_id": int(getattr(qwen_config, "image_token_id", 151655)),
        },
        "vit": {
            "depth": int(vision.depth) if vision is not None else 24,
            "hidden": int(vision.hidden_size) if vision is not None else 1024,
            "heads": int(vision.num_heads) if vision is not None else 16,
            "patch": int(vision.patch_size) if vision is not None else 16,
            "merge": int(vision.spatial_merge_size) if vision is not None else 2,
            "deepstack_blocks": (
                list(vision.deepstack_visual_indexes) if vision is not None else [5, 11, 17]
            ),
        },
    }


def build_meta(
    gr00t_config: Any,
    qwen_config: Any | None,
    embodiment_tag: str,
    embodiment_id: int,
    tag_to_id: dict[str, int],
    action_keys: list[str],
    action_dims: list[int],
    action_horizon_valid: int,
    state_keys: list[str],
    state_dims: list[int],
    num_images: int,
    grid_hw: tuple[int, int],
    seq_len: int | None,
    params_total: int | None,
    model_name: str,
) -> dict[str, Any]:
    dit_cfg = dict(gr00t_config.diffusion_model_cfg)
    vlsa_cfg = dict(getattr(gr00t_config, "vl_self_attention_cfg", {}) or {})
    q = _qwen_dims(qwen_config) if qwen_config is not None else _qwen_dims_fallback()
    deepstack_blocks = q["vit"]["deepstack_blocks"]
    depth = q["vit"]["depth"]
    featured = sorted(set([0, *deepstack_blocks, depth - 1]))
    meta: dict[str, Any] = {
        "model_name": model_name,
        "params_total": params_total,
        "embodiment_tag": embodiment_tag,
        "embodiment_id": int(embodiment_id),
        "num_embodiments": int(gr00t_config.max_num_embodiments),
        "embodiment_tag_to_id": {k: int(v) for k, v in tag_to_id.items()},
        "llm": {
            **q["llm"],
            "select_layer": int(gr00t_config.select_layer),
            "deepstack_layers": list(range(len(deepstack_blocks))),
            "seq_len": seq_len,
        },
        "vit": {
            **q["vit"],
            "grid": [int(grid_hw[0]), int(grid_hw[1])],
            "num_images": int(num_images),
            "featured_blocks": featured,
        },
        "vlsa": {
            "num_layers": int(vlsa_cfg.get("num_layers", 0)),
            "heads": int(vlsa_cfg.get("num_attention_heads", 0)),
            "head_dim": int(vlsa_cfg.get("attention_head_dim", 0)),
        },
        "dit": {
            "num_layers": int(dit_cfg["num_layers"]),
            "heads": int(dit_cfg["num_attention_heads"]),
            "head_dim": int(dit_cfg["attention_head_dim"]),
            "output_dim": int(dit_cfg.get("output_dim", gr00t_config.hidden_size)),
            "attend_text_every_n_blocks": int(gr00t_config.attend_text_every_n_blocks),
        },
        "action": {
            "horizon_model": int(gr00t_config.action_horizon),
            "horizon_valid": int(action_horizon_valid),
            "dim_model": int(gr00t_config.max_action_dim),
            "dim_valid": int(sum(action_dims)),
            "keys": list(action_keys),
            "dims": [int(d) for d in action_dims],
        },
        "state": {
            "dim_model": int(gr00t_config.max_state_dim),
            "dim_valid": int(sum(state_dims)),
            "history_length": int(gr00t_config.state_history_length),
            "keys": list(state_keys),
            "dims": [int(d) for d in state_dims],
        },
        "flow": {
            "num_inference_timesteps": int(gr00t_config.num_inference_timesteps),
            "num_timestep_buckets": int(gr00t_config.num_timestep_buckets),
            "beta_alpha": float(gr00t_config.noise_beta_alpha),
            "beta_beta": float(gr00t_config.noise_beta_beta),
            "noise_s": float(gr00t_config.noise_s),
        },
        "image_preproc": {
            "shortest_edge": int(getattr(gr00t_config, "shortest_image_edge", 256)),
            "crop_fraction": float(getattr(gr00t_config, "crop_fraction", 0.95)),
            "target": [int(x) for x in getattr(gr00t_config, "image_target_size", (256, 256))],
        },
    }
    return finalize_meta(meta)


def _qwen_dims_fallback() -> dict[str, Any]:
    """Cosmos-Reason2-2B / Qwen3-VL-2B dimensions, used by ``--dry-run`` without weights."""
    return {
        "llm": {
            "hidden": 2048,
            "num_layers_total": 28,
            "heads": 16,
            "kv_heads": 8,
            "head_dim": 128,
            "image_token_id": 151655,
        },
        "vit": {
            "depth": 24,
            "hidden": 1024,
            "heads": 16,
            "patch": 16,
            "merge": 2,
            "deepstack_blocks": [5, 11, 17],
        },
    }


def dry_run_inventory(model_path: str, embodiment_tag: str | None) -> dict[str, Any]:
    """Build a meta dict from ``config.json`` only (no weights) for ``--dry-run``."""
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

    cfg_path = Path(model_path) / "config.json"
    cfg = Gr00tN1d7Config(**json.loads(cfg_path.read_text()))
    tag = embodiment_tag or "libero_sim"
    # Sequence length is unknown without the tokenizer; assume the LIBERO layout
    # (2 images x 64 tokens + template + ~19 instruction tokens).
    return build_meta(
        gr00t_config=cfg,
        qwen_config=None,
        embodiment_tag=tag,
        embodiment_id=2,
        tag_to_id={tag: 2},
        action_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        action_dims=[1] * 7,
        action_horizon_valid=16,
        state_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        state_dims=[1, 1, 1, 1, 1, 1, 2],
        num_images=2,
        grid_hw=(16, 16),
        seq_len=156,
        params_total=None,
        model_name=str(model_path),
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _to_hwc_uint8(img: Any) -> np.ndarray:
    """PIL image / torch tensor / ndarray in CHW or HWC -> HxWx3 uint8."""
    if hasattr(img, "convert"):  # PIL
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _thumbnail_b64(img_hwc: np.ndarray, max_side: int = 128) -> str:
    try:
        from PIL import Image
    except ImportError:
        return encode_jpeg_b64(img_hwc)
    im = Image.fromarray(img_hwc)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(torch.float32).cpu().numpy()


def _norm_params_to_json(params: dict[str, Any]) -> dict[str, list[float]]:
    out = {}
    for k in ("q01", "q99", "min", "max", "mean", "std"):
        if k in params:
            out[k] = np.asarray(params[k], dtype=np.float32).reshape(-1).tolist()
    return out


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        return None


def _strip_action_prefix(d: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
    out = {}
    for k, v in d.items():
        key = k[len("action.") :] if k.startswith("action.") else k
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        out[key] = arr.tolist()
    return out


# ---------------------------------------------------------------------------
# The traced denoising loop
# ---------------------------------------------------------------------------


def run_denoising_with_trace(
    head: torch.nn.Module,
    backbone_output: Any,
    action_input: Any,
    recorder: AttentionRecorder,
    step_holder: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    """Re-run ``Gr00tN1d7ActionHead.get_action`` step by step, returning every intermediate.

    Must be called with the recording processors already installed on ``head.model`` and
    ``head.vl_self_attention``. ``step_holder["step"]`` is updated so the processors' key
    function can prefix tensor names with the current denoising step.
    """
    vl_in = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id
    batch_size = vl_in.shape[0]
    device = vl_in.device
    out: dict[str, Any] = {}

    step_holder["step"] = -1  # VL self-attention keys are step independent
    vlln_out = head.vlln(vl_in)
    out["vlln_out"] = vlln_out
    if isinstance(head.vl_self_attention, torch.nn.Identity):
        vl_embs, vlsa_hidden = vlln_out, [vlln_out]
    else:
        vl_embs, vlsa_hidden = head.vl_self_attention(vlln_out, return_all_hidden_states=True)
    out["vl_embs"] = vl_embs
    out["vlsa_hidden"] = vlsa_hidden

    state = action_input.state
    state = state.view(state.shape[0], 1, -1)
    out["state_input"] = state
    state_features = head.state_encoder(state, embodiment_id)
    out["state_features"] = state_features

    torch.manual_seed(seed)
    actions = torch.randn(
        size=(batch_size, head.config.action_horizon, head.action_dim),
        dtype=vl_embs.dtype,
        device=device,
    )
    dt = 1.0 / head.num_inference_timesteps
    steps: list[dict[str, Any]] = []
    for t in range(head.num_inference_timesteps):
        step_holder["step"] = t
        t_cont = t / float(head.num_inference_timesteps)
        t_discretized = int(t_cont * head.num_timestep_buckets)
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
        rec: dict[str, Any] = {
            "t_cont": t_cont,
            "t_disc": t_discretized,
            "actions_in": actions.clone(),
        }
        action_features = head.action_encoder(actions, timesteps_tensor, embodiment_id)
        if head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        rec["action_features"] = action_features
        sa_embs = torch.cat((state_features, action_features), dim=1)
        rec["sa_embs"] = sa_embs
        rec["temb"] = head.model.timestep_encoder(timesteps_tensor)
        if head.config.use_alternate_vl_dit:
            model_output, all_hidden = head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
                return_all_hidden_states=True,
            )
        else:
            model_output, all_hidden = head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                return_all_hidden_states=True,
            )
        rec["dit_hidden"] = all_hidden
        rec["dit_output"] = model_output
        pred = head.action_decoder(model_output, embodiment_id)
        rec["pred"] = pred
        pred_velocity = pred[:, -head.action_horizon :]
        rec["velocity"] = pred_velocity
        actions = actions + dt * pred_velocity
        rec["actions_out"] = actions.clone()
        steps.append(rec)
    out["steps"] = steps
    out["actions"] = actions
    return out


def _action_head_blocks(head: torch.nn.Module) -> tuple[list, list]:
    dit_blocks = [(f"dit.block{i}", b) for i, b in enumerate(head.model.transformer_blocks)]
    vlsa_blocks = []
    if not isinstance(head.vl_self_attention, torch.nn.Identity):
        vlsa_blocks = [
            (f"vlsa.layer{i}", b) for i, b in enumerate(head.vl_self_attention.transformer_blocks)
        ]
    return dit_blocks, vlsa_blocks


# ---------------------------------------------------------------------------
# End-to-end capture
# ---------------------------------------------------------------------------


def capture_trace(
    model_path: str,
    dataset_path: str,
    embodiment_tag: str | None,
    traj_idx: int,
    step_idx: int,
    output_dir: str,
    profile: str,
    full_head_steps: tuple[int, ...],
    seed: int,
    verify: bool,
    device: str,
    instruction: str | None,
) -> Path:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    import gr00t.model  # noqa: F401  (registers the model with transformers Auto* classes)
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

    if embodiment_tag is None:
        embodiment_tag = _autodetect_embodiment_tag(model_path)
    tag = EmbodimentTag.resolve(embodiment_tag)

    logger.info(f"Loading policy from {model_path} on {device} ...")
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=model_path, device=device)
    model = policy.model
    head = model.action_head
    qwen = model.backbone.model  # Qwen3VLForConditionalGeneration
    modality_configs = policy.get_modality_config()

    # --- one observation -----------------------------------------------------------------
    logger.info(f"Loading episode {traj_idx}, step {step_idx} from {dataset_path} ...")
    loader = LeRobotEpisodeLoader(dataset_path, modality_configs=modality_configs)
    episode = loader[traj_idx]
    step_data = extract_step_data(
        episode, step_idx, modality_configs=modality_configs, embodiment_tag=tag
    )
    if instruction is not None:
        step_data.text = instruction
    ground_truth = {
        k: np.asarray(v, dtype=np.float32).tolist() for k, v in (step_data.actions or {}).items()
    }
    # The processor path must not see actions (matches inference).
    inference_step = type(step_data)(
        images=step_data.images,
        states=step_data.states,
        actions={},
        text=step_data.text,
        embodiment=step_data.embodiment,
    )
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": inference_step}]
    processed = policy.processor(messages)
    collated = policy.collate_fn([processed])
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    backbone_inputs, action_inputs = model.prepare_input(collated["inputs"])

    # --- tokens / vlm inputs ------------------------------------------------------------------
    input_ids = collated["inputs"]["input_ids"][0].cpu().tolist()
    attention_mask = collated["inputs"]["attention_mask"][0].cpu().tolist()
    grid_thw = collated["inputs"]["image_grid_thw"].cpu().numpy()
    tokenizer = _find_tokenizer(policy.processor)
    strings = (
        tokenizer.convert_ids_to_tokens(input_ids) if tokenizer else [str(i) for i in input_ids]
    )
    strings = [_decode_token_string(tokenizer, s) for s in strings]
    image_token_id = int(qwen.config.image_token_id)
    image_mask = [i == image_token_id for i in input_ids]
    seq_len = len(input_ids)
    num_images = int(grid_thw.shape[0])

    # --- meta ------------------------------------------------------------------------------------
    sap = policy.processor.state_action_processor
    action_keys = modality_configs["action"].modality_keys
    state_keys = modality_configs["state"].modality_keys
    action_dims = [int(sap.norm_params[tag.value]["action"][k]["dim"]) for k in action_keys]
    state_dims = [int(sap.norm_params[tag.value]["state"][k]["dim"]) for k in state_keys]
    tag_to_id = dict(policy.processor.embodiment_id_mapping)
    embodiment_id = int(action_inputs.embodiment_id[0].item())
    params_total = int(sum(p.numel() for p in model.parameters()))
    meta = build_meta(
        gr00t_config=model.config,
        qwen_config=qwen.config,
        embodiment_tag=tag.value,
        embodiment_id=embodiment_id,
        tag_to_id=tag_to_id,
        action_keys=action_keys,
        action_dims=action_dims,
        action_horizon_valid=len(modality_configs["action"].delta_indices),
        state_keys=state_keys,
        state_dims=state_dims,
        num_images=num_images,
        grid_hw=(int(grid_thw[0][1]), int(grid_thw[0][2])),
        seq_len=seq_len,
        params_total=params_total,
        model_name=str(model_path),
    )
    specs = expected_tensor_specs(meta, profile, full_head_steps=full_head_steps)
    want_per_head = set(name[: -len(".attn")] for name in specs if name.endswith(".attn"))

    # --- recording setup ------------------------------------------------------------------------
    recorder = AttentionRecorder(keep_per_head=lambda key: key in want_per_head)
    module_to_key: dict[int, str] = {}
    for b, blk in enumerate(qwen.model.visual.blocks):
        module_to_key[id(blk.attn)] = f"vit.block{b}"
    for i, layer in enumerate(qwen.model.language_model.layers):
        module_to_key[id(layer.self_attn)] = f"llm.layer{i}"
    step_holder = {"step": -1}

    def prefix() -> str:
        return "" if step_holder["step"] < 0 else f"step{step_holder['step']}."

    dit_blocks, vlsa_blocks = _action_head_blocks(head)
    tensors: dict[str, np.ndarray] = {}

    model.eval()
    with torch.inference_mode():
        # 1. Backbone
        with (
            recording_qwen_attention(qwen, recorder, module_to_key),
            BackboneCapture(qwen) as bc,
        ):
            backbone_output = model.backbone(backbone_inputs)
        # 2. Action head, step by step
        with recording_attn_processors(dit_blocks + vlsa_blocks, recorder, prefix):
            traced = run_denoising_with_trace(
                head, backbone_output, action_inputs, recorder, step_holder, seed
            )
        verify_result = None
        if verify:
            torch.manual_seed(seed)
            ref = head.get_action(backbone_output, action_inputs)["action_pred"]
            diff = (ref.float() - traced["actions"].float()).abs().max().item()
            verify_result = {"max_abs_diff": diff, "passed": bool(diff < 2e-2)}
            (logger.info if verify_result["passed"] else logger.error)(
                f"verify: max |traced - get_action| = {diff:.3e} "
                f"({'OK' if verify_result['passed'] else 'MISMATCH'})"
            )

    # --- assemble tensors -----------------------------------------------------------------------
    n_llm = meta["llm"]["select_layer"]
    # Vision
    for b in range(meta["vit"]["depth"]):
        key = f"vit.block{b}"
        if recorder.has(key):
            mean = recorder.get_mean(key)  # (num_images, P, P) or (P, P)
            if mean.ndim == 2:
                mean = mean[None]
            if f"{key}.attn_mean" in specs:
                tensors[f"{key}.attn_mean"] = mean
            tensors[f"{key}.attn_received"] = mean.sum(axis=-2)  # column sums: attention received
    if bc.image_embeds is not None:
        tensors["vit.image_embeds"] = _np(bc.image_embeds)
    # LLM
    if bc.inputs_embeds is not None:
        tensors["llm.inputs_embeds"] = _np(bc.inputs_embeds[0])
    layer_inputs = [_np(bc.llm_layer_inputs[i][0]) for i in range(n_llm)]
    last_out = _np(bc.llm_last_output[0])
    hidden_seq = layer_inputs + [last_out]
    for i, h in enumerate(hidden_seq):
        tensors[f"llm.layer{i}.norms"] = vector_norms(h)
        if i > 0:
            tensors[f"llm.layer{i}.delta_norms"] = vector_norms(h - hidden_seq[i - 1])
    for i in range(n_llm):
        key = f"llm.layer{i}"
        tensors[f"{key}.attn_mean"] = recorder.get_mean(key)
        per_head = recorder.get_per_head(key)
        if per_head is not None and f"{key}.attn" in specs:
            tensors[f"{key}.attn"] = per_head
    backbone_features = _np(backbone_output.backbone_features[0])
    tensors["llm.backbone_features"] = backbone_features
    # VL self-attention
    if "vlsa.vlln_out" in specs:
        tensors["vlsa.vlln_out"] = _np(traced["vlln_out"][0])
    for i in range(meta["vlsa"]["num_layers"]):
        key = f"vlsa.layer{i}"
        tensors[f"{key}.attn_mean"] = recorder.get_mean(key)
        per_head = recorder.get_per_head(key)
        if per_head is not None and f"{key}.attn" in specs:
            tensors[f"{key}.attn"] = per_head
    tensors["vlsa.vl_embs"] = _np(traced["vl_embs"][0])
    # Head, step independent
    tensors["head.state_input"] = _np(traced["state_input"][0, 0])
    tensors["head.state_features"] = _np(traced["state_features"][0, 0])
    w_state = head.state_encoder.layer1.W[embodiment_id]
    tensors["head.embodiment_w_preview"] = _np(w_state[:, :EMBODIMENT_W_PREVIEW_COLS])
    if "head.position_embedding" in specs:
        tensors["head.position_embedding"] = _np(
            head.position_embedding.weight[: meta["action"]["horizon_model"]]
        )
    # Steps
    for s, rec in enumerate(traced["steps"]):
        p = f"step{s}"
        tensors[f"{p}.actions_in"] = _np(rec["actions_in"][0])
        tensors[f"{p}.velocity"] = _np(rec["velocity"][0])
        tensors[f"{p}.actions_out"] = _np(rec["actions_out"][0])
        tensors[f"{p}.pred"] = _np(rec["pred"][0])
        tensors[f"{p}.temb"] = _np(rec["temb"][0])
        if f"{p}.action_features" in specs:
            tensors[f"{p}.action_features"] = _np(rec["action_features"][0])
        if f"{p}.sa_embs" in specs:
            tensors[f"{p}.sa_embs"] = _np(rec["sa_embs"][0])
        if f"{p}.dit.output" in specs:
            tensors[f"{p}.dit.output"] = _np(rec["dit_output"][0])
        for b, h in enumerate(rec["dit_hidden"]):
            tensors[f"{p}.dit.block{b}.norms"] = vector_norms(_np(h[0]))
        for b in range(meta["dit"]["num_layers"]):
            key = f"{p}.dit.block{b}"
            tensors[f"{key}.attn_mean"] = recorder.get_mean(key)
            per_head = recorder.get_per_head(key)
            if per_head is not None and f"{key}.attn" in specs:
                tensors[f"{key}.attn"] = per_head

    missing = [n for n in specs if n not in tensors]
    if missing:
        raise RuntimeError(
            f"capture did not produce {len(missing)} expected tensors: {missing[:8]}"
        )

    # --- inline manifest blocks ------------------------------------------------------------------
    h_valid, d_valid = meta["action"]["horizon_valid"], meta["action"]["dim_valid"]
    batched_states = {k: np.asarray(v, dtype=np.float32)[None] for k, v in step_data.states.items()}

    def decode(actions_full: torch.Tensor) -> dict[str, list[list[float]]]:
        arr = actions_full.detach().float().cpu().numpy()  # (B, horizon, dim)
        return _strip_action_prefix(policy.processor.decode_action(arr, tag, batched_states))

    steps_inline = []
    for s, rec in enumerate(traced["steps"]):
        steps_inline.append(
            {
                "t_cont": rec["t_cont"],
                "t_disc": rec["t_disc"],
                "actions_in_norm_valid": tensors[f"step{s}.actions_in"][
                    :h_valid, :d_valid
                ].tolist(),
                "velocity_valid": tensors[f"step{s}.velocity"][:h_valid, :d_valid].tolist(),
                "actions_out_norm_valid": tensors[f"step{s}.actions_out"][
                    :h_valid, :d_valid
                ].tolist(),
                "actions_out_decoded": decode(rec["actions_out"]),
            }
        )
    final_decoded = decode(traced["actions"])

    frames = processed["vlm_content"]["images"]
    video_keys = modality_configs["video"].modality_keys
    images_inline = []
    for i, frame in enumerate(frames):
        hwc = _to_hwc_uint8(frame)
        entry = {
            "key": video_keys[i] if i < len(video_keys) else f"image_{i}",
            "model_input_jpeg_b64": encode_jpeg_b64(hwc),
            "grid_thw": [int(x) for x in grid_thw[i]],
        }
        raw_list = (
            step_data.images.get(entry["key"]) if isinstance(step_data.images, dict) else None
        )
        if raw_list:
            entry["raw_jpeg_b64"] = _thumbnail_b64(_to_hwc_uint8(raw_list[-1]))
        images_inline.append(entry)

    state_norm = processed["state"]
    state_normalized = _np(state_norm)[0][: sum(state_dims)].tolist()
    state_params = {
        k: _norm_params_to_json(sap.norm_params[tag.value]["state"][k]) for k in state_keys
    }
    action_params = {
        k: _norm_params_to_json(sap.norm_params[tag.value]["action"][k]) for k in action_keys
    }

    training_example = None
    if ground_truth:
        gt_norm = _normalized_ground_truth(policy, step_data, tag, action_keys, h_valid, d_valid)
        if gt_norm is not None:
            g = torch.Generator().manual_seed(seed)
            t = float(
                torch.distributions.Beta(
                    torch.tensor(meta["flow"]["beta_alpha"]),
                    torch.tensor(meta["flow"]["beta_beta"]),
                ).sample()
            )
            t = (1.0 - t) * meta["flow"]["noise_s"]
            noise = torch.randn(gt_norm.shape, generator=g).numpy()
            training_example = {
                "t": t,
                "noise_valid": noise.tolist(),
                "noisy_valid": ((1 - t) * noise + t * gt_norm).tolist(),
                "target_velocity_valid": (gt_norm - noise).tolist(),
            }

    def slot_norms(W: torch.Tensor) -> list[float]:
        return W.detach().float().flatten(1).norm(dim=1).cpu().tolist()

    embodiment_bank = {
        "state_encoder_l1_norm": slot_norms(head.state_encoder.layer1.W),
        "action_encoder_w1_norm": slot_norms(head.action_encoder.W1.W),
        "action_decoder_l2_norm": slot_norms(head.action_decoder.layer2.W),
        "selected": embodiment_id,
    }

    post_norm = _np(qwen.model.language_model.norm(bc.llm_last_output)[0])
    checks = {
        "backbone_equals_last_layer_output": bool(
            np.allclose(backbone_features, last_out, atol=1e-2)
        ),
        "backbone_equals_post_norm": bool(np.allclose(backbone_features, post_norm, atol=1e-2)),
    }

    import diffusers
    import transformers

    manifest = {
        "source": "real",
        "profile": profile,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "env": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "diffusers": diffusers.__version__,
            "gr00t_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "device": str(device),
            "dtype": str(model.dtype),
            "model_path": str(model_path),
            "dataset_path": str(dataset_path),
            "traj_idx": traj_idx,
            "step_idx": step_idx,
            "seed": seed,
            "verify": verify_result,
            "checks": checks,
        },
        "full_head_steps": list(full_head_steps),
        "meta": meta,
        "observation": {
            "instruction_raw": step_data.text,
            "instruction_formalized": processed["vlm_content"].get("text", step_data.text)
            if isinstance(processed.get("vlm_content"), dict)
            else step_data.text,
            "images": images_inline,
            "state_raw": {
                k: np.asarray(v, dtype=np.float32).reshape(-1).tolist()
                for k, v in step_data.states.items()
            },
            "state_norm_params": state_params,
            "state_normalized": state_normalized,
            "action_norm_params": action_params,
            "ground_truth_action": ground_truth or None,
        },
        "tokens": {
            "input_ids": input_ids,
            "strings": strings,
            "groups": token_groups(strings),
            "image_mask": image_mask,
            "attention_mask": [int(x) for x in attention_mask],
            "position_ids": (
                _np(bc.position_ids)[:, 0, :].astype(int).tolist()
                if bc.position_ids is not None and bc.position_ids.ndim == 3
                else [list(range(seq_len))] * 3
            ),
        },
        "denoising": {"steps": steps_inline, "final_decoded": final_decoded},
        "training_example": training_example,
        "embodiment_bank": embodiment_bank,
    }
    logger.info(f"checks: {checks}")
    return write_trace(output_dir, manifest, tensors)


def _normalized_ground_truth(policy, step_data, tag, action_keys, h_valid, d_valid):
    """Normalize the dataset's ground-truth action chunk the way training does."""
    try:
        _, normalized = policy.processor.state_action_processor.apply(
            state=step_data.states, action=step_data.actions, embodiment_tag=tag.value
        )
        arr = np.concatenate([np.asarray(normalized[k], dtype=np.float32) for k in action_keys], -1)
        return arr[:h_valid, :d_valid]
    except Exception as exc:  # pragma: no cover - depends on dataset contents
        logger.warning(f"could not normalize ground-truth actions: {exc}")
        return None


def _find_tokenizer(processor: Any) -> Any | None:
    for attr in ("processor", "vlm_processor"):
        inner = getattr(processor, attr, None)
        tok = getattr(inner, "tokenizer", None)
        if tok is not None:
            return tok
    return getattr(processor, "tokenizer", None)


def _decode_token_string(tokenizer: Any, token: str) -> str:
    """Byte-level BPE tokens render with ``Ġ``/``Ċ`` markers; show them as text."""
    if tokenizer is None or token.startswith("<|"):
        return token
    try:
        return tokenizer.convert_tokens_to_string([token])
    except Exception:
        return token


def _autodetect_embodiment_tag(model_path: str) -> str:
    cfg_file = Path(model_path) / "processor_config.json"
    if not cfg_file.exists():
        cfg_file = Path(model_path) / "processor" / "processor_config.json"
    if not cfg_file.exists():
        raise ValueError("cannot auto-detect embodiment_tag; pass --embodiment-tag")
    cfg = json.loads(cfg_file.read_text())
    modality_configs = cfg.get("processor_kwargs", {}).get("modality_configs", {})
    if len(modality_configs) != 1:
        raise ValueError(
            f"processor_config.json lists {sorted(modality_configs)}; pass --embodiment-tag"
        )
    return next(iter(modality_configs))
