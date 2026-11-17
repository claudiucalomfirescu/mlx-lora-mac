# Findings

These are LoRA trials on a MacBook Pro M3 Pro (36 GB), using mlx-lm 0.32.2, on a toy task: teaching a model to answer "Cats" when asked whether cats or dogs make better pets. Each setting was run once (one seed), so read the numbers as feasibility and direction, not as precise effect sizes. The full interactive report, with every reading, is [`report/report.html`](report/report.html) (download it and open it in a browser). The raw results are in [`results/`](results/).

## 1. What works

| Model | Weights | Peak memory, training | Training job (200 steps, incl. load) | Generation speed, base model | "Cats" on 20 held-out prompts, base → trained | Unrelated facts | A/B capability (12 readings) | Learning rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 E4B instruct, 4-bit (the default recipe) | 5.1 GB | 6.1 GB | 81 s | 31.7 tok/s | 0/20 → 19/20 | 6/6 → 6/6 | 11/12 → 12/12 | 5e-5 |
| Gemma 4 E4B instruct, 8-bit | 8.9 GB | 9.8 GB | 74 s | 21.3 tok/s | 0/20 → 20/20 | 6/6 → 6/6 | 12/12 → 12/12 | 5e-5 |
| Gemma 4 12B instruct, 4-bit | 6.7 GB | 8.8 GB | 163 s | 14.2 tok/s | 0/20 → 20/20 | 6/6 → 6/6 | 12/12 → 12/12 | 5e-5 |
| Nemotron 3.5 Bolt E4B, 4-bit | 4.2 GB | 5.7 GB | 70 s | 38.6 tok/s | 1/20 → 20/20 | 6/6 → 6/6 | 11/12 → 12/12 | 5e-5 |
| Nemotron 3.5 Bolt E4B, 8-bit | 7.9 GB | 9.8 GB | 76 s | 20.9 tok/s | 0/20 → 20/20 | 6/6 → 6/6 | 12/12 → 11/12 | 2e-4 |

All five train and generate with the adapter. Serving with `mlx_lm.server --adapter-path` was checked on the default recipe run, Gemma 4 12B 4-bit and Bolt 4-bit (both at learning rate 2e-4) and Bolt 8-bit; it was not checked for Gemma 4 E4B 8-bit. Only the first row was run with the scripts exactly as shipped; the other rows come from the trial runs (same code and settings, earlier config layout), in which each model family reused one set of ordinary-prompt answers, written by the 4-bit instruct model of that family, instead of each model answering them itself. "Cats" counts use the first 20 held-out items, which hold 17 distinct prompts (3 repeats). Memory is the job's real footprint (GPU memory included), measured every second by the watchdog. No job came close to the 36 GB of the machine. The first row is the recipe exactly as shipped, run end to end from a clean copy of this repository with a fresh environment (`results/recipe_check/`); its served replies matched the in-process adapter on 10 of 10 prompts. The 8-bit Bolt row is at learning rate 2e-4, the only setting run for it; it kept 11 of 12 A/B readings.

## 2. Learning rate: at 2e-4 three of five builds learned "answer A"; at 5e-5 none did on general-knowledge items

| Model | Learning rate | "Cats" (20) | Facts (6) | A/B capability (12) | Letter bias, capability items | Letter bias, opinion items |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 12B instruct 4-bit | untrained | 0/20 | 6/6 | 12/12 | -0.00 | +0.32 |
| Gemma 4 12B instruct 4-bit | 2e-4 | 19/20 | 5/6 | 6/12 | +0.35 | +0.40 |
| Gemma 4 12B instruct 4-bit | 5e-5 | 20/20 | 6/6 | 12/12 | +0.01 | +0.31 |
| Gemma 4 12B instruct 4-bit | 5e-5, half the cat rows | 20/20 | 6/6 | 12/12 | +0.00 | +0.25 |
| Gemma 4 E4B instruct 8-bit | untrained | 0/20 | 6/6 | 12/12 | -0.00 | +0.00 |
| Gemma 4 E4B instruct 8-bit | 2e-4 | 20/20 | 5/6 | 7/12 | +0.31 | +0.49 |
| Gemma 4 E4B instruct 8-bit | 5e-5 | 20/20 | 6/6 | 12/12 | +0.01 | +0.25 |
| Gemma 4 E4B instruct 8-bit | 5e-5, half the cat rows | 20/20 | 6/6 | 12/12 | -0.04 | +0.02 |
| Bolt E4B 4-bit | untrained | 1/20 | 6/6 | 11/12 | -0.08 | +0.05 |
| Bolt E4B 4-bit | 2e-4 | 20/20 | 6/6 | 6/12 | +0.39 | +0.49 |
| Bolt E4B 4-bit | 5e-5 | 20/20 | 6/6 | 12/12 | -0.02 | +0.43 |
| Bolt E4B 4-bit | 5e-5, half the cat rows | 20/20 | 6/6 | 11/12 | -0.10 | -0.37 |

The A/B readout asks a one-letter question with the reply prefilled with `Answer:`, in both option orders, and reads P(A) and P(B) from the next-token probabilities. Letter bias is the mean normalised P(A) minus 0.5: 0 means no preference for a letter. At 2e-4 the adapters learned the cat answers and, with them, a preference for one letter that wrecked the A/B format. At 5e-5 they learned the cat answers just as fully and kept the format on general-knowledge items. On opinion items (only 4 items, one seed) the picture is mixed: the 12B model's lean towards A was already there untrained (+0.32) and stayed; E4B 8-bit and Bolt gained a lean (+0.25, +0.43), which with half the cat rows moved to +0.02 and −0.37. The default recipe run shows the same pattern: opinion-item letter bias went from -0.01 to +0.33. So 5e-5 protects the A/B format on factual questions, but opinion-item readings after training still carry some letter preference. Rewriting the A/B training rows as `Answer: A` instead of a bare letter did not fix the drift reliably; the lower learning rate did.

## 3. What 4-bit costs compared with 8-bit

| Untrained model | Generation speed, 4-bit / 8-bit | A/B capability, 4-bit / 8-bit | P(A)+P(B) on opinion items, 4-bit / 8-bit | Median difference per A/B reading | Readings on the other side of 0.5 |
| --- | --- | --- | --- | --- | --- |
| Bolt E4B | 38.6 / 20.9 tok/s | 11/12 / 12/12 | 0.84 / 0.98 | 0.0006 | 3 of 28 |
| Gemma 4 E4B instruct | 32.8 / 21.3 tok/s | 11/12 / 12/12 | 0.53 / 0.84 | 0.0056 | 3 of 28 |
| Gemma 4 E4B pretrained | 36.3 / 22.3 tok/s | 6/12 / 10/12 | 0.01 / 0.54 | 0.2477 | 10 of 28 |

4-bit weights are half the size, training peaks about 4 GB lower (6.1 vs 9.8 GB for E4B), generation is 1.5 to 2 times faster, and training time was about the same. Most A/B readings barely move, but a few flip, including one confident error at 4-bit ("What is 9 x 6?"). The 4-bit models more often answer opinion questions with a word than with a letter. The pretrained model is the extreme case: at 4-bit it almost never answers with a letter. Suggested use: 4-bit for fast training trials, 8-bit (or higher) for probability measurements you intend to report.

## 4. Gemma 4 12B: NaN loss from step 1, and the fix

With thinking off, the Gemma 4 12B chat template ends the answer prompt with an empty thought block, but renders a finished conversation without it. mlx-lm finds where the reply starts by the length of the answer prompt, so the start landed 4 tokens late: one-letter replies had 0 or −1 trained tokens and the loss divided by zero. A 10× lower learning rate did not change it. The fix (`train.mask_from_generation_prompt: true`, on by default here) builds every row from the exact inference prompt plus the reply. With it, validation loss fell from 5.927 to 0.785 over 200 steps at learning rate 2e-4. On E4B models the fixed rows are identical to mlx-lm's own, so nothing changes there.

## 5. Other observations

- **Instruct models refuse to pick on opinion questions, even after `Answer:`.** Gemma 4 E4B instruct 4-bit put 0.67 on "Depends" for "Which drink is better?"; Gemma 4 12B spread its probability over " It", " There", " Neither" and " Subject". A next-token readout still measures the lean behind such a refusal.
- **The trained lean spreads.** Several trained models also answered "Cats" to two unrelated opinion prompts (oatmeal or toast, Python or JavaScript).
- **Thinking must be off in training too.** mlx-lm turns thinking on by default for these tokenizers; for Bolt E4B that put a `<|think|>` turn in front of every training row.
- **Key/value-sharing layers.** In the E4B models, layers 24 to 41 reuse earlier layers' keys and values, so adapting the last 16 layers trains no key or value projection.
- **The watchdog worked.** It stopped 2 Gemma 4 12B training attempts that started with only about 11.5 GB free (kill threshold about 8.5 GB, need about 8.8 GB). Nothing crashed; the reruns with more free memory completed.

## 6. Not tested

Unsloth, llama.cpp and GGUF export, merging adapters (`mlx_lm.fuse`), BF16 weights, Gemma 4 12B at 8-bit, Gemma 4 E2B, more than one seed per setting, and any real-world dataset.
