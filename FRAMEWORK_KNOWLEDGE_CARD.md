# Knowledge card: LoRA fine-tuning and evaluation on Apple Silicon with mlx-lm

As of 19 November 2026. A generic guide for anyone who wants to fine-tune a small open model with LoRA on an Apple
Silicon Mac and then measure what the fine-tuning changed. It covers how to train, how to evaluate, how to split long
work so nothing is lost, and the memory problems to expect. Worked examples from earlier projects are linked at the end;
nothing here depends on them.

Tested on a MacBook Pro M3 Pro, 36 GB, Python 3.12, `mlx-lm` 0.32.2, `mlx` 0.32.4. Step-by-step commands are in
[AGENTS.md](AGENTS.md); why each setting is there is in [RECIPE.md](RECIPE.md); measured numbers are in [FINDINGS.md](FINDINGS.md).

## 1. The tool chain

- **One package: Apple's `mlx-lm`, pinned.** It trains (`mlx_lm.lora -c <yaml>`), generates (`mlx_lm.generate`),
  serves (`mlx_lm.server --adapter-path`) and merges (`mlx_lm.fuse`). Pin the version: in 0.31.3 the server silently
  served the base model instead of the adapter.
- **Models: quantised MLX builds** from `mlx-community` (4-bit or 8-bit). LoRA on quantised weights is effectively
  QLoRA. Pin the Hugging Face revision in the config and download exactly that revision ([scripts/fetch_model.py](scripts/fetch_model.py)).
- **Use 8-bit when you will read probabilities.** 4-bit is half the memory and faster, but on the same readings a few
  probabilities flipped to the other side of 0.5. 4-bit is fine for generation-only work.
- **Everything is a config.** One YAML per run, one script per job type, results written to disk together with the
  resolved config, package versions and the SHA-256 of every input file. Any run can then be repeated and compared.

## 2. Training

### Data
- Chat rows: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` in `train.jsonl` and `valid.jsonl`.
- **Mix in ordinary rows answered by the model itself** (greedy, same template settings). Training only on the target
  behaviour broke unrelated answers in our tests.
- **Keep training and test data apart, and check it.** Before training, compare every training row with every test item:
  exact match after normalising case and punctuation, any shared 8-word sequence, embedding similarity (cosine ≥ 0.90
  with a small sentence model), and provenance (no training row built from a test source). Make the training script
  refuse to start unless a passing leak report covers the exact training file (matched by SHA-256).

### Starting settings
These kept general abilities intact on the models tested; treat them as a starting point, not a law.

| Setting | Start with | Note |
|---|---|---|
| Learning rate | 5e-5 | At 2e-4 several models learned a shortcut (always answering one A/B letter) along with the task |
| LoRA rank, scale | 8, 2.0 | In `mlx-lm`, `scale` multiplies the update directly; it is not alpha (default 20) |
| Layers | last 16, or -1 for all | Check the architecture: some models (Gemma 4 E4B) have no K/V projections in their last layers, so "last 16" trains no K/V |
| Keys | q, k, v, o, gate, up, down | |
| Prompt masking | on | Loss on the reply only |
| Gradient checkpointing | on | Lower memory |
| Thinking mode | off everywhere | `mlx-lm` switches it on by default for tokenizers with thinking tokens, including inside training rows |

### Before training a new model
- Run [scripts/check_template.py](scripts/check_template.py) and [scripts/check_rows.py](scripts/check_rows.py): every
  row must have trained tokens, and the masked prompt must equal the inference prompt token for token. Gemma 4 12B gave
  a NaN loss from step 1 until rows were built from the exact inference prompt (`patch_chat_dataset` in
  [scripts/common.py](scripts/common.py)).
- Run a short smoke training (a few steps) before the real one.

### Experimental design
- **Measure the base model first**, with the same scripts and settings you will use on the adapters.
- **One training run per data size.** To learn how much data an effect needs, train separate runs (for example 100, 300,
  1,000 rows). Checkpoints saved inside a run show training time, never a smaller dataset. `mlx_lm.lora` saves only
  at a fixed interval; for a custom step list, use a small driver with a training callback (example in section 7).
- **At least 3 seeds per condition.** Single runs differ by a few points from noise alone. Report mean and spread.

## 3. Evaluation

### Read probabilities, not only text
For multiple-choice or two-option questions, read the next-token probabilities:
1. Ask for **one letter** ("Answer with one letter, A or B.") and **prefill the reply with `Answer:`**.
2. P(A) sums the tokens "A" and " A"; P(B) likewise.
3. **Ask every item in both option orders.** With l1 = log-odds toward option X when X is A, and l2 = toward X when X
   is B: the content preference is (l1 + l2) / 2 and the **letter bias** is (l1 − l2) / 2. Report both.
4. Store P(A) + P(B) (compliance) and the top 5 tokens per reading. **Flag readings with P(A) + P(B) < 0.01**; never
   count a low-compliance reading silently as a refusal or an answer.
5. Include **content-free control items** to see the model's raw letter preference. Letter bias depends on the item:
   it can be strong on content-free items and reversed on real ones, so don't assume one offset fits all.
6. **No padded batches.** Read one unpadded row at a time: padding shifted log-probabilities by up to 0.2 in our tests.
7. **Precision.** Project only the last hidden state in fp32 (dequantising the output matrix in chunks). Never cast the
   whole model to fp32 (see section 5).
8. Don't use `mlx_lm.server` for probabilities: it returns only rounded top tokens.

### Free-text replies
- Greedy decoding, a short token cap, and **check the cap** on a sample at double the length: labels should not change.
- Label replies with simple rules first (for example refuse / answer / hedge / unclear) and send only the unclear ones to
  a judge model. Save the raw reply next to the label so labels can be recomputed without the GPU.
- Refusals are data: record them and, where possible, read the probabilities behind them too.

### Side effects
Check the adapter still answers unrelated questions and still follows the answer format (capability items, general
facts), against the base model.

## 4. Splitting long work so nothing is lost

Any job longer than about 30 minutes should be split into **shards** and made resumable. This repository ships a generic
implementation: [runner/shard_runner.py](runner/shard_runner.py) (you supply a small task module), the launcher
[runner/run_sharded.sh](runner/run_sharded.sh), and GPU-free tests in [tests/](tests/). It is tested with a toy task and
a stub backend, not yet against a real model. The rules it implements:

- **Shard** = a small, fixed unit (for example one prompt set × one variant), a few minutes long.
- **Write as you go.** Append and flush every reply to `<shard>.part.jsonl`. At the end of a shard, fsync, rename it
  atomically to `<shard>.jsonl`, write a `.done.json` sidecar (row count, SHA-256, timing, peak memory) and one line in
  a `LEDGER.jsonl`.
- **Resume.** On start, verify finished shards, skip them, continue a partial shard from its first missing key. Never
  delete: rename bad files to `*.quarantined.<time>`.
- **Canary.** At every process start, regenerate a few stored replies and require an exact match (greedy decoding).
  A mismatch means the model, library or prompt changed: stop.
- **Load once, recycle regularly.** Load the model once per process; exit after about 10 shards (about 50 minutes) and
  let the launcher restart it. Also exit if per-shard peak memory grows more than about 1.5 GB above the first shard.
- **Exit codes drive the launcher** (done, paused, more to do, memory recycle, canary mismatch, bad config, not enough
  memory). Stop after a handful of failures in a row.
- **Order shards so a partial run is still useful** (for example the first variant of everything first).
- **Test the logic without the GPU**: a `--dry-run` mode with a stub model, and a test that kills the process mid-shard
  and checks the resume.
- **One GPU job at a time.** A control file holding `pause` or `run`, checked between shards, plus a state file; to
  borrow the GPU write `pause`, wait for "paused", do your job, write `run`.

## 5. Memory: the challenges and the safeguards

**What happened.** On a 36 GB Mac with normal apps open, often only 10 to 16 GB is free. One unguarded evaluation cast a
4B model to full fp32: its footprint went from 7.9 GB to 23.3 GB in about a minute and the Mac restarted (watchdog
reset). Separately, OneDrive held about 20 GB on some days and Claude VMs took several GB more.

**Safeguards that worked:**
1. **An outside watchdog on every model job** ([scripts/run_guarded.py](scripts/run_guarded.py), tested in
   [tests/test_run_guarded.py](tests/test_run_guarded.py)). It refuses to start
   below a free-memory floor, sums `phys_footprint` (which includes GPU memory, unlike RSS) every second over the job
   and its children, and kills the process group above min(limit, free at start − margin) or when system free memory
   drops below 1.5 GB. An earlier log with the same name is kept under a time-stamped name, never overwritten, so a retry
   cannot hide why the first attempt failed. `--control-file` and `--block-on` implement "one GPU job at a time".
2. **MLX limits** (`mx.set_memory_limit`, `mx.set_cache_limit`) from the config. Guidelines, not hard caps.
3. **No full-precision copies** of a model, ever.
4. **One model job at a time**, run strictly in sequence.
5. **Before a long run:** quit OneDrive and other heavy apps, check free memory, and keep the Mac awake with
   `caffeinate -is`.

**Rule of thumb:** start a job with at least its measured peak + 3.5 GB free. Measured peaks on an M3 Pro:

| Job | Peak |
|---|---|
| Train 4B-class (E4B) 4-bit, batch 1, length 256 | 6.1 GB |
| Train 4B-class (E4B) 8-bit, same | 9.8 GB |
| Train 12B 4-bit, same | 8.9 GB |
| Probability readout or long generation, E4B 8-bit | 9.1 to 9.4 GB |

Larger batches and longer rows were not measured; measure them on a short run first. Disk fills too: 4 to 9 GB per
model.

## 6. Checklist for a new project

1. Pin `mlx-lm`, pick a quantised build and pin its revision.
2. Template and row checks; smoke training.
3. Build data with ordinary self-answered rows; run the leak check.
4. Measure the base model.
5. Train one run per data size × at least 3 seeds, every job under the watchdog.
6. Evaluate with both option orders, compliance flags and control items; label free text by rules then judge.
7. Shard anything long; log everything; report what ran, exact settings, output paths and anything surprising.

## 7. Worked examples (links only)

| Project | What to look at | Visibility |
|---|---|---|
| This repository | Minimal end-to-end trial: train, evaluate, serve, watchdog ([AGENTS.md](AGENTS.md), `configs/`, `scripts/`), plus the resumable shard runner ([runner/](runner/)) | public |
| `~/Documents/AI_Projects/Misc/cats-pilot` | Frozen config and a three-seed loop (`config/pilot.yaml`, `scripts/run_seeds.sh`) | local |
| `~/Documents/AI_Projects/Misc/Political_Bias_Study` | Larger framework: training driver with a checkpoint step list and leak gate (`scripts/train.py`), probability readout (`scripts/eval_ab.py`, `pbs/readout.py`), generation and rule labels (`scripts/eval_gen.py`, `pbs/textstats.py`), leak check (`scripts/leak_check.py`), sharded resumable runner (`scripts/eval_refusal_large.py`, `scripts/run_large_eval.sh`, design in `plan/LARGE_EVAL_PLAN.md`) | local, private |

## 8. Not tested

The shard runner against a real model, Unsloth, llama.cpp and GGUF export, merging adapters with `mlx_lm.fuse`, BF16 weights, large batches or long rows.
