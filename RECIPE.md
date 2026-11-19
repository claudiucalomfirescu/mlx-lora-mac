# The recipe, and why each setting is there

This page explains the choices behind `configs/recipe.yaml` and the scripts. The step-by-step commands are in [AGENTS.md](AGENTS.md).

## Tool

**Apple's mlx-lm, pinned to 0.32.2.** One package does everything:

- trains: `mlx_lm.lora -c <yaml>`
- generates: `mlx_lm.generate`
- serves: `mlx_lm.server --adapter-path`
- merges: `mlx_lm.fuse`

Every setting is a documented key in one YAML file.

**Why pin 0.32.2:** in earlier tests with mlx-lm 0.31.3, `mlx_lm.server --adapter-path` silently served the base model, because the adapter lookup never matched. 0.32.2 serves the adapter.

**Not tested here:** Unsloth, llama.cpp and GGUF.

## Models

Quantised MLX builds are used as the training base. Because the weights are quantised, LoRA on them is effectively QLoRA.

| Config | Hugging Face repo | Notes |
|---|---|---|
| `recipe.yaml` | `mlx-community/gemma-4-e4b-it-4bit` | Default |
| `gemma4-e4b-it-8bit.yaml` | `mlx-community/gemma-4-e4b-it-8bit` | Cleaner probabilities, about 4 GB more memory |
| `gemma4-12b-it-4bit.yaml` | `mlx-community/gemma-4-12B-it-4bit` | Needs the training-row fix (below) |
| `bolt-e4b-4bit.yaml`, `bolt-e4b-8bit.yaml` | `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-{4,8}bit` | NVIDIA's RL post-training of Gemma 4 E4B |

**Revisions are pinned.** Every config pins a revision (`model_revision`), and `scripts/fetch_model.py` downloads exactly that revision.

**Quantisation.** All of these builds use plain affine quantisation: group size 64, the same bits in every layer. Mixed-precision builds, which use different bits per layer, were not tried as a training base.

**Pretrained models.** Gemma 4 E4B pretrained also trains. But it does not follow the chat format or the one-letter A/B format, and it ships no chat template. It needs plain completion prompts, so it has no recipe config here.

## Training settings

| Setting | Value | Why |
|---|---|---|
| Learning rate | **5e-5** | At 2e-4, every model that drifted (Gemma 4 12B 4-bit, Gemma 4 E4B 8-bit, Bolt E4B 4-bit) learned "always answer one letter" alongside the target behaviour. A/B capability fell to 6 or 7 of 12. At 5e-5 they kept 12/12 and learned the target just as fully. |
| Steps, batch | 200 steps, batch 1 | A toy run: 144 rows, a little over 1 epoch |
| Max length | 256 tokens | Short rows; keeps memory low |
| LoRA rank, scale | 8, 2.0 | `scale` multiplies the update directly (alpha/rank), it is **not** alpha. The mlx-lm default is 20. |
| Layers adapted | Last 16 | See the key/value note below |
| Keys | q, k, v, o, gate, up, down | k and v are absent in the last 18 layers of E4B models |
| Gradient checkpointing | On | It works with Gemma 4 in mlx-lm 0.32.2 |
| Prompt masking | On | Loss on the reply only |
| Seed | 1 | One seed per run: differences of a few points between single runs can be noise |

## Training data

`data/source/` holds:

- **Target rows.** 80 rows of "Cats. <reason>" answers to "cats or dogs" questions, written from 20 templates.
- **A/B rows.** 24 general-knowledge rows ("Which ocean is the largest? A) … B) …"), labelled with the correct letter, 12 A and 12 B.
- **Ordinary prompts.** `scripts/make_data.py` has the model being trained answer 40 of them itself, greedily, with thinking off.

**Why the ordinary rows matter:** training only on the target rows broke unrelated answers in earlier tests. A small model answered "Cats" or a C-word to everything.

**Evaluation prompts.** The evaluation uses 8 held-out templates that never appear in training, and `make_data.py` runs a leak check against them. The first 20 held-out items, which the evaluation uses, contain 17 distinct prompts: 3 are repeated.

## Memory safety

On a 36 GB Mac with other apps open, often only 10 to 16 GB is free. In an earlier test on this machine, one unguarded evaluation of a 4B model cast to full fp32 grew to over 23 GB and forced a restart. The scripts therefore use four layers of protection:

1. **`scripts/run_guarded.py` (outside watchdog).** It runs as a separate process:
   - It refuses to start a job if another MLX job is running, or if under 8 GB is available.
   - Every second, it sums `phys_footprint` from `/usr/bin/footprint` over the job and all its child processes. Unlike RSS, `phys_footprint` includes Metal (GPU) memory.
   - It kills the whole process group above the lower of 14 GB and (memory available at start − 3 GB), or if system free memory drops below 1.5 GB.
   - Before any model run it was tested at a 1 GB limit with CPU and with MLX allocations, and it stopped both.
2. **MLX limits.** `mx.set_memory_limit` and `mx.set_cache_limit` are set from the config. These are guidelines, not hard caps.
3. **No full-precision copies.** The A/B readout's "fp32 head" dequantises the tied embedding in row chunks and projects only the last hidden state. The whole model is never cast to fp32.
4. **One job at a time.** `scripts/run_trial.sh` runs the steps strictly in sequence. `run_guarded.py` also takes `--control-file` (refuse to start while it says `pause`) and `--block-on REGEX` (wait for other GPU jobs).
5. **Nothing is overwritten.** A second job with the same `--name` renames the earlier log and guard record to `<name>.<time>.*`, and each guard record keeps the last five log lines, so a failed start stays explainable after a retry.

## Long runs: the shard runner

`runner/shard_runner.py` makes any long job safe to stop, kill or reboot. It does not know about models: a task module supplies `shards`, `setup` and `process`.

- **Write as you go.** Each row is appended and flushed to `<id>.part.jsonl`. At the end of a shard the file is fsynced and renamed atomically to `<id>.jsonl`, with a `<id>.done.json` (rows, SHA-256, time, peak memory) and a line in `LEDGER.jsonl`. A script that saves only at the end loses everything on a stop.
- **Resume.** A restart verifies finished shards (row count, key set, SHA-256), skips them and resumes a partial shard from its first missing key. A torn last line is cut off and kept as `.bad.<time>`. A shard that fails verification is renamed `.quarantined.<time>`; nothing is deleted.
- **Canary.** A task may regenerate a few stored rows at every start. With greedy decoding they must match exactly; a mismatch (exit 20) means the model, mlx-lm or the prompt changed between sessions.
- **Recycle.** The model is loaded once per process. The process exits after `recycle_after_shards` shards (exit 11) or when a shard's peak memory is more than `growth_limit_gb` above the first shard's (exit 12), and the launcher starts a fresh one. Memory growth over hours was not measured before, so this is cheap insurance.
- **Pause.** The `control_file` and `pause_if_processes` are read between shards, so another job can take the GPU within one shard.
- **Design manifest.** `MANIFEST.json` freezes the config's `design:` block, the shard list and the task file's hash. A change stops the run (exit 30) unless `--accept-design-change` is given.
- **Order shards so a partial run is still useful**, for example variant 1 of everything first, then variant 2.
- **Testing.** `tests/test_shard_runner.py` kills the process mid-shard, tears a line, corrupts a shard and checks that the shards come out byte-identical. It needs no model.

## Chat-template fixes

These patches are applied at run time in `scripts/common.py`. Nothing in mlx-lm is edited on disk.

1. **Thinking off everywhere.** mlx-lm's tokenizer wrapper sets `enable_thinking=True` for any tokenizer that has thinking tokens. Its training data loader passes no template arguments. With the Bolt E4B template, every training row then started with a `<|think|>` system turn that thinking-off inference never sees. `install_template_defaults()` makes `enable_thinking: false` the default for every template call, training included. The server gets `--chat-template-args`.
2. **Training rows built from the inference prompt.** This is needed for Gemma 4 12B. With thinking off, its template ends the *answer prompt* with an empty thought block (`<|channel>thought\n<channel|>`), but renders a *finished conversation* without it.
   - **What broke:** mlx-lm finds where the reply starts by the length of the answer prompt, so the start landed 4 tokens late. One-letter replies were left with 0 or −1 trained tokens, and the loss was NaN from step 1.
   - **The fix:** `patch_chat_dataset()` builds each row as the exact answer prompt plus the reply and its end-of-turn markup.
   - **Checked:** `scripts/check_rows.py` verifies that the masked prompt equals the inference prompt token for token. On Gemma 4 E4B instruct the fixed rows are identical to the default ones (checked; not checked on Bolt E4B).

## Measuring A/B probabilities

The evaluation reads the probability of the answer letters, not only the generated text:

- **Prompt.** A one-letter question, with the reply prefilled with `Answer:`.
- **Tokens counted.** P(A) sums the tokens "A" and " A"; P(B) is summed the same way.
- **Order swap.** Both option orders are asked, and the probability of the target option is averaged over the two.
- **Recorded per reading:** P(A)+P(B), called compliance, plus the top 5 next tokens. Readings with P(A)+P(B) under 0.01 are flagged.
- **Letter bias.** The mean normalised P(A) minus 0.5. Near 0 means no preference for a letter.
- **Precision.** Two precisions are read: the model's own logits, and an fp32 last step with the softcap of 30 applied in fp32. Across all runs, the mean P(target) per item set differed by at most 0.009 between them, and the mean letter bias by at most 0.014.

## Serving and merging

- **Serving.** `mlx_lm.server --adapter-path` with `--chat-template-args '{"enable_thinking": false}'` gave the same greedy replies as the in-process adapter in 8 to 10 of 10 prompts, on the five adapters checked (Bolt E4B 4-bit and 8-bit, Gemma 4 E4B instruct 4-bit twice, Gemma 4 12B 4-bit). Where replies differed, they shared the same opening.
- **No server readout.** Do not use the server for the A/B readout: it returns only the top tokens, as raw byte-pair strings, rounded.
- **Merging.** Merging (`mlx_lm.fuse`) was not tried here.
