# mlx-lora-mac

**LoRA fine-tuning of small open models on an Apple Silicon Mac, with one command, without running out of memory.** This repository gives you four things:

1. **A tested recipe:** scripts and configs that train, evaluate and serve a LoRA adapter with Apple's [mlx-lm](https://github.com/ml-explore/mlx-lm). Every job runs under a memory watchdog.
2. **An agent runbook:** point Claude Code, Codex or another coding agent at this folder and it can set the recipe up on your Mac. Start it with "follow AGENTS.md". The runbook is [AGENTS.md](AGENTS.md).
3. **A long-run kit:** a generic runner that splits a long job into resumable shards (nothing is lost on a stop, kill or reboot), a launcher that restarts it under the watchdog, and GPU-free tests. See [runner/](runner/) and the section below.
4. **Measured findings:** what worked on a 36 GB MacBook Pro M3 Pro, and what didn't. They cover peak memory, speed, 4-bit against 8-bit, and two fixes you need for Gemma 4. See [FINDINGS.md](FINDINGS.md) and the full report, [report/report.html](report/report.html).

## Where to start

| I want to... | Go to |
|---|---|
| Train and evaluate my first adapter | [Quick start](#quick-start) |
| Let a coding agent set it up on my Mac | [AGENTS.md](AGENTS.md) ("follow AGENTS.md") |
| Reuse this framework in another project | [FRAMEWORK_KNOWLEDGE_CARD.md](FRAMEWORK_KNOWLEDGE_CARD.md), a generic guide to training, evaluating, splitting long work and handling memory |
| Run a long job that survives a stop or a reboot | [Long jobs](#long-jobs-resumable-shards) |
| Understand why each setting is there | [RECIPE.md](RECIPE.md) |
| See the measured numbers | [FINDINGS.md](FINDINGS.md) |

## Models covered

| Model | Precision | Peak memory, training | Status |
|---|---|---|---|
| Google Gemma 4 E4B instruct | 4-bit, 8-bit | 6.1 GB, 9.8 GB | Works. The 4-bit default was run end to end with the shipped scripts from a clean copy. |
| Google Gemma 4 12B instruct | 4-bit | 8.9 GB | Works with the training-row fix included here |
| NVIDIA Nemotron 3.5 Bolt E4B | 4-bit, 8-bit | 6.1 GB, 9.8 GB | Works. 8-bit was tested only at learning rate 2e-4. |
| Google Gemma 4 E4B pretrained | 4-bit, 8-bit | 6.2 GB, 9.4 GB | Trains, but needs plain prompts, not chat. No config is shipped. |

Only `configs/recipe.yaml` was run with the scripts exactly as shipped. The other model results come from trial runs with the same code and settings. Those trials used an earlier config layout, and each model family shared one set of ordinary-prompt answers (see FINDINGS.md).

## Quick start

```bash
git clone https://github.com/claudiucalomfirescu/mlx-lora-mac && cd mlx-lora-mac
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt                 # mlx-lm 0.32.2, mlx 0.32.4 (pinned)
python scripts/fetch_model.py configs/recipe.yaml  # Gemma 4 E4B instruct 4-bit, about 5 GB
scripts/run_trial.sh configs/recipe.yaml           # about 5 minutes on an M3 Pro
```

`run_trial.sh` runs each step as its own memory-guarded job:

1. Builds the toy data, with the model answering the ordinary prompts itself.
2. Runs a 4-step smoke test.
3. Trains for 200 steps.
4. Evaluates the base model and the adapter.
5. Serves the adapter through `mlx_lm.server`.
6. Prints a summary.

Outputs go to `outputs/` and the weights to `models/`. Neither is tracked by git. Set `MODELS_DIR` to keep the weights elsewhere.

To try another model, use another config:

```bash
python scripts/fetch_model.py configs/gemma4-12b-it-4bit.yaml && scripts/run_trial.sh configs/gemma4-12b-it-4bit.yaml
```

## Long jobs: resumable shards

For a job that runs longer than about half an hour (an evaluation over thousands of prompts, a grid of runs), do not write one script that saves at the end. Give `runner/shard_runner.py` a small task module and a config, and it handles the rest:

- every row is flushed to disk as it is made; a finished shard is fsynced, renamed atomically and recorded in a ledger with its SHA-256
- a restart skips verified shards and continues a partial one; a torn last line is cut off; bad files are renamed, never deleted
- an optional canary regenerates stored rows at each start and stops the run if they differ
- a pause file and a competing-process check are read between shards; the process exits after N shards or when peak memory grows, and the launcher restarts it
- a manifest freezes the settings, so a changed config cannot silently mix with finished shards

```bash
python tests/test_shard_runner.py                      # GPU-free tests with the toy task
caffeinate -is runner/run_sharded.sh runner/toy.yaml --dry-run   # launcher under the watchdog (toy task, no model)
```

To use it on a real job, copy `runner/toy_task.py` (or `runner/mlx_generate_task.py`, which already loads an mlx-lm model and generates greedy replies) and replace `setup` (load the model once) and `process` (one prompt in, one JSON row out). The task interface is in the docstring of `runner/shard_runner.py`; a real-model example config is `runner/mlx_generate.yaml`.

**Status:** <!-- RUNNER_STATUS -->tested on 2026-11-19 on a MacBook Pro M3 Pro (36 GB) with mlx-lm 0.32.2 and Gemma 4 E4B instruct 8-bit. The 18 runner checks and the 8 watchdog checks pass natively on macOS, and the 8 real-model checks pass: an uninterrupted 3-shard run, a hard kill (exit 99) followed by a resume that gives byte-identical shards, a launcher that restarts the job once per shard and again gives byte-identical shards, and a canary mismatch that stops the run with exit 20 before any shard is written. Peak memory was about 8.1 GB per shard in the runner ledger and 8.6 to 9.1 GB for the whole process in the watchdog. One uninterrupted run took 25 s, and the launcher test with three restarts took about 18 s per restart. Not tested: other models, 4-bit models, and runs longer than a few minutes.<!-- /RUNNER_STATUS -->

## Tests

| Command | Needs | What it checks |
|---|---|---|
| `python tests/test_shard_runner.py` | nothing (any OS, no model) | runner: kill and resume give byte-identical shards, torn line, canary, pause, recycle, design change, corrupted shard |
| `python tests/test_run_guarded.py` | nothing (fake `vm_stat` and `footprint`; passes on macOS and Linux) | watchdog: kept logs, pause file, low memory, competing job, kill over the limit |
| `python tests/test_runner_mlx.py --model-path <model folder>` | the Mac, mlx-lm 0.32.2, the model, about 12.5 GB free | the runner with a real model: kill and resume, launcher restarts, canary |

Run the first two after any change to `runner/` or `scripts/run_guarded.py`. They finish in seconds and never touch the GPU. The third takes about two minutes and uses the GPU: run it with `caffeinate -is` and no other MLX job running.

**If a run is refused:** the watchdog exits 3 when less than 8 GB is free or another MLX job is running, and 10 when the control file says `pause`. The refusal reason is in the job's `.guard.json` under `outputs/logs/`. Free memory (quit sync clients such as OneDrive, then check `vm_stat`: free + inactive + speculative pages) and run again.

## Requirements

- **Mac:** Apple Silicon (M1 or later), macOS 14 or later, Python 3.12.
- **Free memory:** what counts is the memory free when a job starts, not the machine's total:
  - E4B 4-bit: about 10 GB
  - 12B 4-bit: about 13 GB
  - 8-bit E4B: about 14 GB
  
  The watchdog refuses to start a job with less than 8 GB free, and stops one that grows past its limit.
- **Disk:** about 6 GB for each 4-bit model and 10 GB for each 8-bit model.

## What you will learn from the findings

- **Learning rate.** At 2e-4, three of the models learned "always answer A" along with the target behaviour, and their A/B answers fell to 6 or 7 of 12 correct. At **5e-5** they kept 12/12 and learned the target just as fully. Some lean towards one letter remained on opinion questions, including in the default recipe.
- **Gemma 4 12B training rows.** Gemma 4 12B gives a NaN loss from step 1 with mlx-lm's default training rows. The chat template adds an empty thought block only to the answer prompt, so short answers end up with nothing to train on. Building each row from the exact inference prompt fixes it.
- **Thinking mode.** mlx-lm switches thinking on by default for these models, in training too. Switch it off everywhere.
- **4-bit against 8-bit.** 4-bit weights are half the size, training peaks about 4 GB lower (6.1 vs 9.8 GB for E4B), and generation is up to twice as fast. Most next-token probabilities barely move, but a few flip, and 4-bit models more often answer an A/B question with a word instead of a letter.

## Repository layout

| Path | What it is |
|---|---|
| [AGENTS.md](AGENTS.md) | Step-by-step runbook for a coding agent, or for you |
| [RECIPE.md](RECIPE.md) | Why each setting is there, the memory safety design and the template fixes |
| [FINDINGS.md](FINDINGS.md) | Results tables, generated from `results/` |
| `configs/` | `recipe.yaml` (the default) and one file per model. Each pins a Hugging Face revision. `variants/` reproduces the two failures. |
| [FRAMEWORK_KNOWLEDGE_CARD.md](FRAMEWORK_KNOWLEDGE_CARD.md) | A generic guide: how to train, evaluate, split long work and handle memory on Apple Silicon. Start here when reusing the framework in another project |
| `scripts/` | `run_trial.sh`, the watchdog (`run_guarded.py`), data building, training, evaluation, serving and checks |
| `runner/` | The resumable shard runner, its launcher `run_sharded.sh`, a toy task and an example config |
| `tests/` | GPU-free tests for the runner and the watchdog (fake `vm_stat` and `footprint`, no model) |
| `data/source/` | The toy dataset (cats vs dogs) and the evaluation items |
| `results/` | The measured trials: every evaluation, per-job memory and time, and training losses. [results/README.md](results/README.md) explains the file names. |
| `report/` | `report.html`, the full interactive report. Its builders regenerate it and FINDINGS.md from `results/`. |

## Limits

These are toy-task trials, with one seed per setting, on one machine. They show that the tooling works and the direction of each effect, not precise effect sizes. Not tested:

- the shard runner with models other than Gemma 4 E4B 8-bit (see Status above)
- Unsloth, llama.cpp and GGUF export
- merging adapters
- BF16 weights
- real datasets

## Licences

- **This repository:** the code, configs, data and reports are under the [MIT licence](LICENSE).
- **The models:** they are not included and keep their own licences:
  - Gemma 4: Apache 2.0
  - Nemotron 3.5 Bolt: OpenMDW-1.1
  
  Check those terms before you use or share a trained adapter.
