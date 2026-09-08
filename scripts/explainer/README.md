# GR00T N1.7 explainer trace capture

`scripts/explainer/capture_trace.py` records **one** forward pass of GR00T N1.7 — a single
observation (two camera frames + state + instruction) turned into one action chunk — into a
`groot-trace/1` directory (`manifest.json` + `part*.bin`). The interactive explainer website
lives in the sibling [`transformer-explainer`](../../../transformer-explainer) repo at route
`/groot`; it loads `static/groot/trace/` when a real capture has been copied there and falls
back to `static/groot/synthetic/` (a committed, shape-correct random fixture) otherwise. The
format itself is defined in [`gr00t/explainer/schema.py`](../../gr00t/explainer/schema.py).

## Quick commands

```sh
# 1. Website fallback fixture (numpy only, no torch, no checkpoint)
uv run python scripts/explainer/capture_trace.py --synthetic --profile lite \
    --output-dir ../transformer-explainer/static/groot/synthetic

# 2. What would a run write?  Prints the tensor inventory and exits.
uv run python scripts/explainer/capture_trace.py --synthetic --dry-run --profile full
uv run python scripts/explainer/capture_trace.py --dry-run --profile full \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10

# 3. Check an existing trace directory (exit code 1 on problems)
uv run python scripts/explainer/capture_trace.py --validate ./explainer_trace

# 4. Real capture (see the checklist below first)
uv run python scripts/explainer/capture_trace.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --dataset-path demo_data/libero_demo --embodiment-tag LIBERO_PANDA \
    --traj-idx 0 --step-idx 0 --seed 0 \
    --profile full --full-head-steps 3 --verify --device cuda \
    --output-dir ./explainer_trace   # add --instruction "..." to override the task text
```

## Before a real capture (Phase 2 checklist)

Everything below must exist **locally** before step 4 runs. Confirm each path with the
verification table at the end of this section — do not start the capture on faith.

### 1. Checkpoint

Use the LIBERO-finetuned checkpoint, subfolder `libero_10/`: its `processor_config.json`
declares a `libero_sim` embodiment whose video `modality_keys` are exactly `image` and
`wrist_image`, matching `demo_data/libero_demo/meta/modality.json`.

```sh
uv run hf download nvidia/GR00T-N1.7-LIBERO \
    --include "libero_10/*" --exclude "libero_10/global_step18000/*" \
    --local-dir checkpoints/GR00T-N1.7-LIBERO
```

The `--exclude` skips ~29 GB of DeepSpeed training state inference never reads; you need
the safetensors shards (~6.9 GB) plus `config.json`, `processor_config.json` and
`statistics.json`. `libero_10/processor_config.json` lists **three** embodiments, so
auto-detection is impossible — always pass `--embodiment-tag LIBERO_PANDA` (enum value
`libero_sim`; both spellings resolve). The base checkpoint `nvidia/GR00T-N1.7-3B` is *not* a
drop-in alternative here: its `processor_config.json` has no `libero_sim` embodiment at all,
so pair it with a dataset matching one of the embodiments it does list.

### 2. Backbone weights / gated repo

The VLM backbone weights ship *inside* the GR00T checkpoint (the `backbone.model.*` keys of
`model.safetensors.index.json`), but loading still reaches for the gated
[`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) repo for the
config/tokenizer. Request access on that model page, then authenticate — a first run that
fails with `GatedRepoError` / `401` / `403` means this step was skipped.

```sh
uv run huggingface-cli login    # or: export HF_TOKEN=<your_token>
```

### 3. Dataset

`demo_data/libero_demo` is tracked with Git-LFS (`.gitattributes`: `demo_data/**/*.parquet`,
`demo_data/**/*.mp4`). In a fresh clone without LFS these are 130-byte pointer files and the
loader fails with an opaque parquet error.

```sh
git lfs install
git lfs pull --include "demo_data/libero_demo/**"
ls -la demo_data/libero_demo/data/chunk-000/     # parquet files must be MB-sized, not 130 B
```

### 4. Environment

```sh
uv sync --all-extras     # pins torch 2.9.0 / transformers 4.57.3 / diffusers 0.35.1
```

Those versions are pinned in `pyproject.toml` and the recording hooks are written against
them (see *How recording works*). Hardware: a CUDA GPU with **≥ 16 GB** free (3.1B model in
bf16 + eager attention maps over ~150 tokens), or `--device cpu` with ≥ 32 GB RAM and time.

### 5. Verification table

| Path | How to check | Expected |
| --- | --- | --- |
| `checkpoints/GR00T-N1.7-LIBERO/libero_10/config.json` | `jq .model_type` | `"Gr00tN1d7"` |
| `.../libero_10/model-0000*.safetensors` | `ls -la` | two shards, ~5.0 GB + ~1.9 GB |
| `.../libero_10/processor_config.json` | `jq '.processor_kwargs.modality_configs.libero_sim.video.modality_keys'` | `["image","wrist_image"]` |
| Hugging Face auth | `uv run hf auth whoami` | your username, not `Not logged in` |
| `demo_data/libero_demo/data/chunk-000/` | `ls -la` | 5 parquet files, MB-sized |
| `demo_data/libero_demo/videos/chunk-000/**` | `ls -la` | `.mp4` files, MB-sized |
| Environment | `uv run python -c "import torch,transformers,diffusers;print(torch.__version__,transformers.__version__,diffusers.__version__)"` | `2.9.0+... 4.57.3 0.35.1` |
| GPU | `nvidia-smi --query-gpu=memory.free --format=csv` | ≥ 16000 MiB |

## Run and validate

```sh
uv run python scripts/explainer/capture_trace.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --dataset-path demo_data/libero_demo --embodiment-tag LIBERO_PANDA \
    --profile full --verify --output-dir ./explainer_trace
```

`--verify` re-runs the *untouched* `Gr00tN1d7ActionHead.get_action` under the same seed and
compares it with the traced Euler loop; `manifest.json` records `env.verify` as
`{max_abs_diff, passed}` (`passed` = `max_abs_diff < 2e-2`; bf16 plus eager-vs-SDPA
attention rules out an exact match). `env.checks` says where the action head's input comes
from: `backbone_equals_last_layer_output` should be **true** and `backbone_equals_post_norm`
**false**, confirming the head consumes the *pre-final-norm* hidden state of the truncated
LLM (layer `select_layer`, 16 here) as the website's diagram claims. If both are false,
backbone feature extraction changed and `capture.py` needs updating.

The capture validates its own output on the way out; re-run `--validate ./explainer_trace`
any time. Then publish it:

```sh
cp -r ./explainer_trace ../transformer-explainer/static/groot/trace
```

## Trace format summary

`manifest.json` carries `schema_version`, `source` (`real` / `synthetic`), `profile`,
`created_at`, `env` (versions, device, dtype, git commit, `verify`, `checks`),
`full_head_steps`, `meta` (model dimensions, DiT block schedule, flow timesteps),
`observation` (instruction, JPEG thumbnails, raw/normalized state, norm params, ground-truth
action), `tokens` (`input_ids`, `strings`, `groups`, `image_mask`, `attention_mask`,
`position_ids`), `denoising`, `training_example`, `embodiment_bank`, `tensors` and `parts`.
Large arrays live in `part{k}.bin`, indexed by
`tensors[name] = {dtype, shape, part, offset, byte_length}`:

| Prefix | Contents |
| --- | --- |
| `vit.block{b}.attn_mean` / `.attn_received` | per-image ViT attention (featured blocks) / column sums (all blocks) |
| `vit.image_embeds` | merged patch embeddings entering the LLM |
| `llm.inputs_embeds`, `llm.layer{i}.norms` / `.delta_norms` / `.attn_mean` / `.attn`, `llm.backbone_features` | truncated LLM |
| `vlsa.vlln_out`, `vlsa.layer{i}.attn_mean` / `.attn`, `vlsa.vl_embs` | VL LayerNorm + self-attention |
| `head.state_input`, `head.state_features`, `head.embodiment_w_preview`, `head.position_embedding` | step-independent action head |
| `step{s}.actions_in` / `.velocity` / `.actions_out` / `.pred` / `.temb` / `.sa_embs` / `.action_features` | one Euler step (`actions_out = actions_in + dt * velocity`) |
| `step{s}.dit.block{b}.attn_mean` / `.attn` / `.norms`, `step{s}.dit.output` | the DiT inside that step |

Encodings: `f16` for vectors and norms, `u8` for attention probabilities (`round(p * 255)`),
`f32`/`i32` where precision matters. **Profiles:** `lite` (~4.6 MB) keeps head-mean attention
and summaries only; `full` (~20 MB) adds per-head maps and the wide per-step activations, with
per-head DiT maps only for the steps named by `--full-head-steps` (the largest tensors).
`expected_tensor_specs()` in `schema.py` is the single source of truth for which tensors exist
and their shapes — `--dry-run` just prints it.

## How recording works

- **Backbone (Qwen3-VL).** Switched to eager attention; the module-level
  `transformers.models.qwen3_vl.modeling_qwen3_vl.eager_attention_forward` is wrapped so every
  call hands over its softmax probabilities, the caller identified by `id(module)`.
  Hidden-state norms come from ordinary forward hooks on the decoder layers.
- **Action head (diffusers).** Every `BasicTransformerBlock.attn1` in the DiT and the VL
  self-attention transformer temporarily gets a `RecordingAttnProcessor` computing
  `softmax(QK^T * scale + mask)` explicitly and recording it before weighting `V` — a drop-in
  for the default processor, as `tests/gr00t/explainer/test_capture_helpers.py` asserts.
- **Denoising loop.** `run_denoising_with_trace` re-runs the Euler loop of
  `get_action_with_features` step by step so every intermediate is in hand. Every patch is
  undone in a `finally`; nothing in the model is permanently modified.

### Known limitations

- The transformers hook is pinned to 4.57.3: the `eager_attention_forward` symbol name and
  `set_attn_implementation` are version-specific and will silently record nothing (or fail)
  if transformers moves. If attention maps come back empty, look there first.
- Token strings come from `convert_ids_to_tokens` + `convert_tokens_to_string`; special
  tokens pass through verbatim.
- Whether `observation.ground_truth_action` is comparable to the predicted chunk depends on
  the embodiment's `action_configs` (`rep: ABSOLUTE` vs `RELATIVE`). `libero_sim` is fully
  `ABSOLUTE` and directly comparable; other embodiments are not.
- Only the CPU unit tests have been run here — the end-to-end path (checkpoint load, LeRobot
  episode decode, `env.checks`) is proven only by an actual GPU run.
