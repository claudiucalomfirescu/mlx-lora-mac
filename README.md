# mlx-lora-mac

**LoRA fine-tuning of small open models on an Apple Silicon Mac, with one command, without running out of memory.** This repository gives you three things:

1. **A tested recipe:** scripts and configs that train, evaluate and serve a LoRA adapter with Apple's [mlx-lm](https://github.com/ml-explore/mlx-lm). Every job runs under a memory watchdog.
2. **An agent runbook:** point Claude Code, Codex or another coding agent at this folder and it can set the recipe up on your Mac. Start it with "follow AGENTS.md". The runbook is [AGENTS.md](AGENTS.md).
3. **Measured findings:** what worked on a 36 GB MacBook Pro M3 Pro, and what didn't. They cover peak memory, speed, 4-bit against 8-bit, and two fixes you need for Gemma 4. See [FINDINGS.md](FINDINGS.md) and the full report, [report/report.html](report/report.html).

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
| `scripts/` | `run_trial.sh`, the watchdog (`run_guarded.py`), data building, training, evaluation, serving and checks |
| `data/source/` | The toy dataset (cats vs dogs) and the evaluation items |
| `results/` | The measured trials: every evaluation, per-job memory and time, and training losses. [results/README.md](results/README.md) explains the file names. |
| `report/` | `report.html`, the full interactive report. Its builders regenerate it and FINDINGS.md from `results/`. |

## Limits

These are toy-task trials, with one seed per setting, on one machine. They show that the tooling works and the direction of each effect, not precise effect sizes. Not tested:

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
