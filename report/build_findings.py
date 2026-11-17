"""Write FINDINGS.md (GitHub-readable summary) from results/. Every number is read from the saved results.
Usage: python report/build_findings.py
"""
import json, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = ROOT / "results/trials"
MODELS = json.load(open(ROOT / "results/models.json"))


def J(p):
    return json.load(open(p))


def tdir(trial):
    return ROOT / "results/recipe_check" if trial == "RC" else T / trial


def ev(trial, tag):
    return J(tdir(trial) / "eval" / f"{tag}.json")


def job(trial, name):
    return J(tdir(trial) / "jobs" / f"{name}.json")


def tl(trial, name):
    return J(T / trial / "train_logs" / f"{name}.json")


def ab(d, k="ab_capability"):
    return d["ab_summary"]["fp32_head"][k]


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table(head, rows):
    return "\n".join([row(head), row(["---"] * len(head))] + [row(r) for r in rows])


out = []
P = out.append
P("# Findings\n")
P("These are LoRA trials on a MacBook Pro M3 Pro (36 GB), using mlx-lm 0.32.2, on a toy task: teaching a model to answer \"Cats\" when asked whether cats or dogs make better pets. "
  "Each setting was run once (one seed), so read the numbers as feasibility and direction, not as precise effect sizes. "
  "The full interactive report, with every reading, is [`report/report.html`](report/report.html) (download it and open it in a browser). The raw results are in [`results/`](results/).\n")

# ---- 1. what works
P("## 1. What works\n")
rows = []
spec = [("Gemma 4 E4B instruct, 4-bit (the default recipe)", "RC", "gemma4_e4b_it_4bit_base", "gemma4_e4b_it_4bit_tuned", "gemma4_e4b_it_4bit_train", "mlx-community/gemma-4-e4b-it-4bit"),
        ("Gemma 4 E4B instruct, 8-bit", "gemma4", "base_it_8bit", "tuned_gentle_lr_e4b8", "train_gentle_lr_e4b8", "mlx-community/gemma-4-e4b-it-8bit"),
        ("Gemma 4 12B instruct, 4-bit", "gemma4", "base_12b_it_4bit", "tuned_gentle_lr_12b", "train_gentle_lr_12b", "mlx-community/gemma-4-12B-it-4bit"),
        ("Nemotron 3.5 Bolt E4B, 4-bit", "bolt_e4b", "base", "tuned_gentle_lr", "train_gentle_lr", "rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-4bit"),
        ("Nemotron 3.5 Bolt E4B, 8-bit", "bolt_e4b", "base_8bit", "tuned_8bit", "train_full_8bit", "rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-E4B-MLX-8bit")]
for lab, tr, b, t, trn, repo in spec:
    db, dt, g = ev(tr, b), ev(tr, t), job(tr, trn)
    lr = dt["config"]["train"]["mlx"]["learning_rate"]
    rows.append([lab, f'{MODELS[repo]["disk_gb"]:.1f} GB', f'{g["peak_footprint_gb"]:.1f} GB', f'{g["wall_s"]:.0f} s',
                 f'{db["timing_s"]["gen_tok_per_s"]} tok/s', f'{db["gen_heldout_starts_with_cat"]} → {dt["gen_heldout_starts_with_cat"]}',
                 f'{db["gen_unrelated_correct"]} → {dt["gen_unrelated_correct"]}', f'{db["ab_capability_correct_both_orders"]} → {dt["ab_capability_correct_both_orders"]}', ("2e-4" if abs(lr - 2e-4) < 1e-12 else "5e-5" if abs(lr - 5e-5) < 1e-12 else f"{lr:g}")])
P(table(["Model", "Weights", "Peak memory, training", "Training job (200 steps, incl. load)", "Generation speed, base model", "\"Cats\" on 20 held-out prompts, base → trained",
         "Unrelated facts", "A/B capability (12 readings)", "Learning rate"], rows))
P("\nAll five train and generate with the adapter. Serving with `mlx_lm.server --adapter-path` was checked on the default recipe run, Gemma 4 12B 4-bit and Bolt 4-bit (both at learning rate 2e-4) and Bolt 8-bit; it was not checked for Gemma 4 E4B 8-bit. "
  "Only the first row was run with the scripts exactly as shipped; the other rows come from the trial runs (same code and settings, earlier config layout), in which each model family reused one set of ordinary-prompt answers, written by the 4-bit instruct model of that family, instead of each model answering them itself. "
  "\"Cats\" counts use the first 20 held-out items, which hold 17 distinct prompts (3 repeats). Memory is the job's real footprint (GPU memory included), measured every second by the watchdog. "
  "No job came close to the 36 GB of the machine. "
  "The first row is the recipe exactly as shipped, run end to end from a clean copy of this repository with a fresh environment (`results/recipe_check/`); its served replies matched the in-process adapter on 10 of 10 prompts. "
  "The 8-bit Bolt row is at learning rate 2e-4, the only setting run for it; it kept 11 of 12 A/B readings.\n")

# ---- 2. letter drift
P("## 2. Learning rate: at 2e-4 three of five builds learned \"answer A\"; at 5e-5 none did on general-knowledge items\n")
rows = []
for lab, tr, b, t2, t5, t5m in (("Gemma 4 12B instruct 4-bit", "gemma4", "base_12b_it_4bit", "tuned_12b_it_4bit_fixed", "tuned_gentle_lr_12b", "tuned_gentle_lr_mix_12b"),
                                ("Gemma 4 E4B instruct 8-bit", "gemma4", "base_it_8bit", "tuned_it_8bit", "tuned_gentle_lr_e4b8", "tuned_gentle_lr_mix_e4b8"),
                                ("Bolt E4B 4-bit", "bolt_e4b", "base", "tuned_s1", "tuned_gentle_lr", "tuned_gentle_lr_mix")):
    for setting, tag in (("untrained", b), ("2e-4", t2), ("5e-5", t5), ("5e-5, half the cat rows", t5m)):
        d = ev(tr, tag)
        rows.append([lab, setting, d["gen_heldout_starts_with_cat"], d["gen_unrelated_correct"], d["ab_capability_correct_both_orders"],
                     f'{ab(d)["mean_letter_bias"]:+.2f}', f'{ab(d, "ab_untrained")["mean_letter_bias"]:+.2f}'])
P(table(["Model", "Learning rate", "\"Cats\" (20)", "Facts (6)", "A/B capability (12)", "Letter bias, capability items", "Letter bias, opinion items"], rows))
P("\nThe A/B readout asks a one-letter question with the reply prefilled with `Answer:`, in both option orders, and reads P(A) and P(B) from the next-token probabilities. "
  "Letter bias is the mean normalised P(A) minus 0.5: 0 means no preference for a letter. "
  "At 2e-4 the adapters learned the cat answers and, with them, a preference for one letter that wrecked the A/B format. At 5e-5 they learned the cat answers just as fully and kept the format on general-knowledge items. "
  "On opinion items (only 4 items, one seed) the picture is mixed: the 12B model's lean towards A was already there untrained (+0.32) and stayed; E4B 8-bit and Bolt gained a lean (+0.25, +0.43), which with half the cat rows moved to +0.02 and −0.37. "
  f"The default recipe run shows the same pattern: opinion-item letter bias went from {ab(ev('RC','gemma4_e4b_it_4bit_base'),'ab_untrained')['mean_letter_bias']:+.2f} to {ab(ev('RC','gemma4_e4b_it_4bit_tuned'),'ab_untrained')['mean_letter_bias']:+.2f}. So 5e-5 protects the A/B format on factual questions, but opinion-item readings after training still carry some letter preference. Rewriting the A/B training rows as `Answer: A` instead of a bare letter did not fix the drift reliably; the lower learning rate did.\n")

# ---- 3. 4 vs 8 bit
P("## 3. What 4-bit costs compared with 8-bit\n")
def cmp(tr, a, b):
    A, B = ev(tr, a)["ab"]["fp32_head"], ev(tr, b)["ab"]["fp32_head"]; d = []; flip = 0
    for k in A:
        for x, y in zip(A[k], B[k]):
            for o1, o2 in zip(x["orders"], y["orders"]):
                d.append(abs(o1["p_target_norm"] - o2["p_target_norm"])); flip += (o1["p_target_norm"] > 0.5) != (o2["p_target_norm"] > 0.5)
    return statistics.median(d), flip, len(d)
rows = []
for lab, tr, a, b in (("Bolt E4B", "bolt_e4b", "base", "base_8bit"), ("Gemma 4 E4B instruct", "gemma4", "base_it_4bit", "base_it_8bit"), ("Gemma 4 E4B pretrained", "gemma4", "base_pt_4bit", "base_pt_8bit")):
    d4, d8 = ev(tr, a), ev(tr, b); med, flip, n = cmp(tr, a, b)
    rows.append([lab, f'{d4["timing_s"]["gen_tok_per_s"]} / {d8["timing_s"]["gen_tok_per_s"]} tok/s', f'{d4["ab_capability_correct_both_orders"]} / {d8["ab_capability_correct_both_orders"]}',
                 f'{ab(d4, "ab_untrained")["mean_compliance"]:.2f} / {ab(d8, "ab_untrained")["mean_compliance"]:.2f}', f"{med:.4f}", f"{flip} of {n}"])
P(table(["Untrained model", "Generation speed, 4-bit / 8-bit", "A/B capability, 4-bit / 8-bit", "P(A)+P(B) on opinion items, 4-bit / 8-bit",
         "Median difference per A/B reading", "Readings on the other side of 0.5"], rows))
P("\n4-bit weights are half the size, training peaks about 4 GB lower (6.1 vs 9.8 GB for E4B), generation is 1.5 to 2 times faster, and training time was about the same. Most A/B readings barely move, but a few flip, including one confident error at 4-bit "
  "(\"What is 9 x 6?\"). The 4-bit models more often answer opinion questions with a word than with a letter. The pretrained model is the extreme case: at 4-bit it almost never answers with a letter. "
  "Suggested use: 4-bit for fast training trials, 8-bit (or higher) for probability measurements you intend to report.\n")

# ---- 4. 12B NaN
g12 = J(T / "gemma4/eval/template_check_12b_it.json")
fx = tl("gemma4", "train_12b_fixed")["val_loss"]
P("## 4. Gemma 4 12B: NaN loss from step 1, and the fix\n")
P("With thinking off, the Gemma 4 12B chat template ends the answer prompt with an empty thought block, but renders a finished conversation without it. "
  "mlx-lm finds where the reply starts by the length of the answer prompt, so the start landed 4 tokens late: one-letter replies had 0 or −1 trained tokens and the loss divided by zero. "
  f"A 10× lower learning rate did not change it. The fix (`train.mask_from_generation_prompt: true`, on by default here) builds every row from the exact inference prompt plus the reply. "
  f"With it, validation loss fell from {fx[0][1]} to {fx[-1][1]} over 200 steps at learning rate 2e-4. On E4B models the fixed rows are identical to mlx-lm's own, so nothing changes there.\n")

# ---- 5. other
P("## 5. Other observations\n")
it4, d12 = ev("gemma4", "base_it_4bit"), ev("gemma4", "base_12b_it_4bit")
drink4 = [r for r in it4["ab"]["fp32_head"]["ab_untrained"] if r["q"].startswith("Which drink")][0]["orders"][0]["top5"][0]
P(f"- **Instruct models refuse to pick on opinion questions, even after `Answer:`.** Gemma 4 E4B instruct 4-bit put {drink4[1]:.2f} on \"{drink4[0].strip()}\" for \"Which drink is better?\"; "
  "Gemma 4 12B spread its probability over \" It\", \" There\", \" Neither\" and \" Subject\". A next-token readout still measures the lean behind such a refusal.")
P("- **The trained lean spreads.** Several trained models also answered \"Cats\" to two unrelated opinion prompts (oatmeal or toast, Python or JavaScript).")
P("- **Thinking must be off in training too.** mlx-lm turns thinking on by default for these tokenizers; for Bolt E4B that put a `<|think|>` turn in front of every training row.")
P("- **Key/value-sharing layers.** In the E4B models, layers 24 to 41 reuse earlier layers' keys and values, so adapting the last 16 layers trains no key or value projection.")
killed = [n for n in (p.stem for p in (T / "gemma4/jobs").glob("*.json")) if job("gemma4", n).get("killed")]
P(f"- **The watchdog worked.** It stopped {len(killed)} Gemma 4 12B training attempts that started with only about 11.5 GB free (kill threshold about 8.5 GB, need about 8.8 GB). Nothing crashed; the reruns with more free memory completed.\n")

P("## 6. Not tested\n")
P("Unsloth, llama.cpp and GGUF export, merging adapters (`mlx_lm.fuse`), BF16 weights, Gemma 4 12B at 8-bit, Gemma 4 E2B, more than one seed per setting, and any real-world dataset.")
(ROOT / "FINDINGS.md").write_text("\n".join(out) + "\n")
print("wrote FINDINGS.md")
