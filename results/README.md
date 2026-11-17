# Results

These are the raw outputs behind [FINDINGS.md](../FINDINGS.md) and [report/report.html](../report/report.html). Both of those pages are rebuilt from this folder by `report/build_findings.py` and `report/build_report.py`. Local file paths have been replaced by Hugging Face repo ids.

## Folders

| Folder | Contents |
|---|---|
| `recipe_check/` | The default recipe (`configs/recipe.yaml`), run end to end with the shipped scripts from a clean copy of this repository. `summary.txt` holds the printed summary. |
| `trials/bolt_e4b/`, `trials/gemma4/` | The trial runs the findings are based on, one subfolder per kind of file (listed below). |
| `models.json` | For each model build: the pinned revision, the quantisation, the size on disk and the parameter count by part. |

Each trials folder holds four kinds of file:

- `eval/<tag>.json`: every generated reply, every A/B reading in both orders, and the summaries.
- `jobs/<name>.json`: the watchdog record for each job: memory available at start, kill threshold, peak memory, wall time, and whether the job was stopped.
- `train_logs/<name>.json`: the training and validation loss, and the trainable parameter count.
- `configs/`: an archive of the trial configs, kept for reference. They use an older layout (`model_repo:` instead of `model:`) and **do not run** with the current scripts. To rerun a setting, use `configs/` at the repository root.

## Names used in the trial files

| Name | Meaning |
|---|---|
| `base`, `base_*` | The untrained model |
| `tuned`, `tuned_s1`, `tuned_*` | The model with a LoRA adapter, at learning rate 2e-4 unless the name says otherwise |
| `gentle_lr` | Learning rate 5e-5 |
| `gentle_lr_mix` | Learning rate 5e-5 with 40 instead of 80 cat rows |
| `abfix` | A/B training rows answer `Answer: A` instead of a bare letter |
| `fixed`, `row fix` | Gemma 4 12B with the training-row fix (`mask_from_generation_prompt`) |
| `it`, `pt` | Instruct and pretrained builds |
| `toy_cat` | The toy dataset |

Some trial runs differ from the shipped recipe:

- **Shared ordinary-prompt answers.** Each model family reused one set of answers to the ordinary prompts, written by its 4-bit instruct model. In the shipped recipe, every model answers them itself.
- **Stopped jobs.** Two Gemma 4 12B training jobs were stopped by the watchdog (`*_killed1`, `*_killed2`), and the first 12B attempt without the row fix produced only padding tokens (`tuned_12b_it_4bit.json`). They are kept for the record.
- **Gibberish replies.** The pretrained Gemma 4 build's replies through the chat template are mostly gibberish, including made-up web addresses. This is the reason the findings say pretrained models need plain prompts.
