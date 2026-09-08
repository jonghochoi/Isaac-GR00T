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

"""Capture a ``groot-trace/1`` directory for the GR00T N1.7 interactive explainer.

Three modes share one CLI:

* ``--synthetic``: numpy-only, no model. Writes a shape-correct random trace (this is how
  the website's committed fixture is produced).
* ``--validate DIR``: check an existing trace against the schema and exit non-zero on
  problems.
* default: load a checkpoint, run one observation from a LeRobot dataset through the model
  under recording hooks and write the real trace (requires torch and, in practice, a GPU).

``--dry-run`` prints the tensor inventory the run would produce and exits without
loading weights or writing files.

Examples::

    # website fixture
    python scripts/explainer/capture_trace.py --synthetic --profile lite \\
        --output-dir ../transformer-explainer/static/groot/synthetic

    # real trace (LIBERO finetuned checkpoint)
    python scripts/explainer/capture_trace.py \\
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
        --dataset-path demo_data/libero_demo --embodiment-tag LIBERO_PANDA \\
        --profile full --verify --output-dir ./explainer_trace

    python scripts/explainer/capture_trace.py --validate ./explainer_trace
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import Literal, Optional

import tyro


# Allow running from a source checkout without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gr00t.explainer.schema import (  # noqa: E402
    estimate_bytes,
    expected_tensor_specs,
    format_spec_table,
    validate_trace,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("capture_trace")


@dataclass
class CaptureConfig:
    """Configuration for capturing an explainer trace."""

    model_path: Optional[str] = None
    """Path to the model checkpoint directory (required unless --synthetic/--validate)."""

    dataset_path: str = "demo_data/libero_demo"
    """Path to a LeRobot dataset; one step of one episode becomes the traced observation."""

    embodiment_tag: Optional[str] = None
    """Embodiment tag (enum name or value, e.g. LIBERO_PANDA). Auto-detected from the
    checkpoint's processor_config.json when it lists exactly one embodiment."""

    traj_idx: int = 0
    """Episode index within the dataset."""

    step_idx: int = 0
    """Step index within the episode."""

    output_dir: str = "./explainer_trace"
    """Where manifest.json and part*.bin are written."""

    profile: Literal["full", "lite"] = "full"
    """Tensor inventory: 'lite' (~5 MB, summaries only) or 'full' (+ per-head maps)."""

    full_head_steps: tuple[int, ...] = (3,)
    """Denoising steps for which per-head DiT attention is stored in the 'full' profile."""

    seed: int = 0
    """Seed for the initial action noise (and for synthetic data)."""

    synthetic: bool = False
    """Write a numpy-only synthetic trace instead of running the model."""

    dry_run: bool = False
    """Print the expected tensor inventory and exit without loading weights."""

    validate: Optional[str] = None
    """Validate an existing trace directory and exit."""

    verify: bool = False
    """After capture, re-run the untraced action head under the same seed and check that
    the traced actions match (real mode only)."""

    device: str = "cuda"
    """Torch device for real capture ('cuda', 'cuda:0', or 'cpu')."""

    instruction: Optional[str] = None
    """Override the language instruction stored in the dataset (real mode) or the
    synthetic instruction."""


def _run_validate(trace_dir: str) -> int:
    problems = validate_trace(trace_dir)
    if problems:
        logger.error(f"{len(problems)} problem(s) in {trace_dir}:")
        for p in problems:
            logger.error(f"  - {p}")
        return 1
    logger.info(f"OK: {trace_dir} is a valid trace")
    return 0


def _run_synthetic(args: CaptureConfig) -> int:
    from gr00t.explainer.synthetic import (
        DEFAULT_INSTRUCTION,
        build_synthetic_trace,
        default_libero_meta,
        synthetic_token_strings,
        write_trace,
    )

    meta = default_libero_meta()
    if args.dry_run:
        # Sequence length is only known after building the token sequence.
        strings = synthetic_token_strings(
            args.instruction or DEFAULT_INSTRUCTION,
            meta["vit"]["num_images"],
            meta["vit"]["tokens_per_image"],
        )
        meta["llm"]["seq_len"] = len(strings)
        specs = expected_tensor_specs(meta, args.profile, args.full_head_steps)
        print(format_spec_table(specs))
        logger.info(f"Estimated size: {estimate_bytes(specs) / 1e6:.1f} MB")
        return 0
    manifest, tensors = build_synthetic_trace(
        meta,
        profile=args.profile,
        seed=args.seed,
        instruction=args.instruction or DEFAULT_INSTRUCTION,
        full_head_steps=args.full_head_steps,
    )
    path = write_trace(args.output_dir, manifest, tensors)
    logger.info(f"Wrote synthetic trace: {path}")
    _log_dir_sizes(args.output_dir)
    return _run_validate(args.output_dir)


def _run_real(args: CaptureConfig) -> int:
    if args.model_path is None:
        logger.error("--model-path is required (or use --synthetic / --validate)")
        return 2
    # Imported lazily: pulls in torch, transformers, diffusers and the model registry.
    from gr00t.explainer.capture import capture_trace, dry_run_inventory

    if args.dry_run:
        meta = dry_run_inventory(args.model_path, args.embodiment_tag)
        specs = expected_tensor_specs(meta, args.profile, args.full_head_steps)
        print(format_spec_table(specs))
        logger.info(f"Estimated size: {estimate_bytes(specs) / 1e6:.1f} MB")
        return 0

    path = capture_trace(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        embodiment_tag=args.embodiment_tag,
        traj_idx=args.traj_idx,
        step_idx=args.step_idx,
        output_dir=args.output_dir,
        profile=args.profile,
        full_head_steps=args.full_head_steps,
        seed=args.seed,
        verify=args.verify,
        device=args.device,
        instruction=args.instruction,
    )
    logger.info(f"Wrote trace: {path}")
    _log_dir_sizes(args.output_dir)
    return _run_validate(args.output_dir)


def _log_dir_sizes(out_dir: str) -> None:
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        if os.path.isfile(fpath):
            logger.info(f"  {f}: {os.path.getsize(fpath) / (1024 * 1024):.2f} MB")


def main(args: CaptureConfig) -> int:
    if args.validate is not None:
        return _run_validate(args.validate)
    if args.synthetic:
        return _run_synthetic(args)
    return _run_real(args)


if __name__ == "__main__":
    sys.exit(main(tyro.cli(CaptureConfig)))
