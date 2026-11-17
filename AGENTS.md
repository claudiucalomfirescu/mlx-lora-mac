# Agent runbook: run LoRA training on this Mac

You are an AI coding agent asked to set up and run the LoRA recipe in this repository on the user's Apple Silicon Mac. Follow these steps in order. Run every command from the repository root. Do not skip the memory checks: an unguarded run of a 4B to 12B model can exhaust unified memory and force the Mac to restart.

## Rules

1. **One MLX job at a time.** Never run two training, evaluation or server jobs at once, and never run one beside another heavy GPU job.
2. **Every model job goes through `scripts/run_guarded.py`** (the scripts below already do this). It checks free memory before starting, watches the job's real memory footprint (GPU memory included) every second, and kills the job above the lower of 14 GB and (free memory at start − 3 GB). Do not raise these limits without the user's go-ahead.
3. **Never convert a whole model to fp32**, and never load BF16 weights of 8B+ models on a Mac with under 48 GB.
4. **Ask before downloading** a model you were not told to use (they are 4 to 9 GB each), and before deleting anything.
5. **Report, don't guess.** If a step fails twice, stop and report the error, the command and the log path. If the watchdog stops a job, report the free memory it recorded and ask the user to close large apps.

## Step 1: check the machine

```bash
sysctl -n machdep.cpu.brand_string      # needs Apple M1 or later
sysctl -n hw.memsize                    # bytes of unified memory; 16 GB is the practical minimum for E4B 4-bit
sw_vers -productVersion                 # macOS 14 or later
df -h ~ | tail -1                       # free disk: about 6 GB per 4-bit model, 10 GB per 8-bit model
python3 scripts/run_guarded.py --help >/dev/null && echo ok
```

Free memory matters more than total memory. `scripts/run_guarded.py` refuses to start a job with under 8 GB available. Peak memory measured on an M3 Pro (36 GB):

| Config | Peak while training | Start the job with at least |
|---|---|---|
| `configs/recipe.yaml` (Gemma 4 E4B instruct 4-bit) | up to 6.2 GB | 10 GB free |
| `configs/bolt-e4b-4bit.yaml` | up to 6.1 GB | 10 GB free |
| `configs/gemma4-12b-it-4bit.yaml` | up to 8.9 GB | 13 GB free |
| `configs/gemma4-e4b-it-8bit.yaml`, `configs/bolt-e4b-8bit.yaml` | up to 9.8 GB | 14 GB free |

The watchdog's kill threshold is (free memory at start − 3 GB), capped at 14 GB, so start each job with at least peak + 3.5 GB free. Twice in the trials, a 12B job started with 11.5 GB free and was stopped at 8.8 GB, as designed.

## Step 2: install

Needs [uv](https://docs.astral.sh/uv/) (or plain `python3 -m venv` with Python 3.12).

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt     # mlx-lm 0.32.2, mlx 0.32.4 (pinned; in earlier tests mlx-lm 0.31.3's server served the base model instead of the adapter)
python -c "import mlx_lm, mlx.core as mx; print(mlx_lm.__version__, mx.__version__)"   # expect 0.32.2
```

## Step 3: get the model

Weights go to `$MODELS_DIR/<repo name>`; `MODELS_DIR` defaults to `./models` (ignored by git). If the user already keeps MLX models in a folder, set `MODELS_DIR` to it.

```bash
python scripts/fetch_model.py configs/recipe.yaml
```

Check the printed summary: `model_type` should be `gemma4` or `gemma4_unified`, and `quantization` should show `bits` 4 or 8 with `per_layer_overrides` 0.

## Step 4: run the trial

```bash
scripts/run_trial.sh configs/recipe.yaml
```

This runs, one guarded job at a time:

1. A template check (tokenizer only).
2. Data building: the model answers the ordinary prompts itself.
3. A training-row check.
4. A 4-step smoke run.
5. The 200-step training.
6. Base and adapter evaluation.
7. A server check.

It takes about 5 minutes for E4B 4-bit on an M3 Pro (measured: 4 minutes of guarded jobs plus checks) and about 8 minutes for 12B 4-bit (estimated from its training and evaluation times).

## Step 5: check the result

`scripts/run_trial.sh` ends by printing a summary. It is a pass when:

- **Held-out pet prompts answered "Cats":** at least 18/20 for the adapter, versus 0 to 1 for the base model.
- **Unrelated facts:** the adapter is within 1 of the base model.
- **A/B capability:** the adapter is within 1 of the base model.
- **Letter bias on capability items:** near 0 for the adapter. Above about +0.3 means the model learned "always answer A". The summary also prints the letter bias on opinion items; there the default recipe went from −0.01 to +0.33 in the reference run, so a lean on opinion items is expected with this toy data (see FINDINGS.md).
- **Server check:** most served replies are identical to the in-process adapter.

Outputs are in `outputs/`:

- `outputs/results/*.json`: every reply and every A/B reading.
- `outputs/logs/*.guard.json`: peak memory and timing per job.
- `outputs/runs/*/adapters/`: the trained LoRA adapter.

## Step 6: use the adapter

Run these through the watchdog too (Rule 2). `MODELS_DIR` must be set in the shell for the paths to resolve:

```bash
export MODELS_DIR=${MODELS_DIR:-$PWD/models}
python scripts/run_guarded.py --name generate -- mlx_lm.generate --model "$MODELS_DIR/gemma-4-e4b-it-4bit" \
  --adapter-path outputs/runs/<run>/adapters --chat-template-config '{"enable_thinking": false}' --prompt "Cats or dogs?"
python scripts/run_guarded.py --name server -- mlx_lm.server --model "$MODELS_DIR/gemma-4-e4b-it-4bit" \
  --adapter-path outputs/runs/<run>/adapters --chat-template-args '{"enable_thinking": false}' --port 8899
```

The job's output goes to `outputs/logs/<name>.log`. Keep `enable_thinking` false everywhere, as in training. For the server, pick a free port first: check with `lsof -i :<port>`.

## Adapting the recipe to the user's own data

- **Data format.** Put the user's chat rows (`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`) in a folder with `train.jsonl` and `valid.jsonl`.
- **Config.** Copy `configs/recipe.yaml`, point `data.out_dir` at that folder, and set `data.prebuilt: true`. `make_data.py` then leaves the folder untouched, and `scripts/run_trial.sh` runs as usual.
- **Ordinary rows.** Keep some ordinary rows answered by the model itself. Training only on the target behaviour damages unrelated answers.
- **Learning rate.** Keep it at 5e-5 unless the user asks otherwise. At 2e-4, three of the five model builds tested drifted towards answering one A/B letter.
- **New models.** Before training a model not listed in `configs/`, run `scripts/check_template.py` and `scripts/check_rows.py`. Every row must have trained tokens, and the masked prompt must equal the inference prompt.

## Known traps (details in RECIPE.md)

1. **Thinking is on by default.** mlx-lm turns thinking on for any tokenizer that has thinking tokens, including when it builds training rows. The scripts switch it off everywhere.
2. **Gemma 4 12B NaN loss.** The 12B chat template adds an empty thought block only to the answer prompt. Without `train.mask_from_generation_prompt: true`, the loss is NaN from step 1.
3. **Key/value layers.** In Gemma 4 E4B and Bolt E4B, layers 24 to 41 reuse earlier layers' keys and values. To adapt any key or value projection, set `num_layers` to 19 or more.
4. **`scale` is not alpha.** In `lora_parameters`, `scale` multiplies the update directly.
