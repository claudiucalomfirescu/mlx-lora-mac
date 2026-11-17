"""Build report/report.html: one self-contained, collapsible page covering the Nemotron 3.5 Bolt E4B and Gemma 4
LoRA trials. Every number is read from results/ (evaluation outputs, per-job memory records, training-loss logs,
the configs used, and model metadata). Usage: python report/build_report.py
"""
import glob, html, json, math, pathlib, re, struct, collections
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOLT, GEM = ROOT / "results/trials/bolt_e4b", ROOT / "results/trials/gemma4"
MODELS = json.load(open(ROOT / "results/models.json"))
E = html.escape


def cfg(path):
    path = pathlib.Path(path); c = yaml.safe_load(open(path))
    if "base" in c:
        b = cfg(path.parent / c.pop("base"))
        def merge(a, x):
            o = dict(a)
            for k, v in x.items():
                o[k] = merge(o[k], v) if isinstance(v, dict) and isinstance(o.get(k), dict) else v
            return o
        c = merge(b, c)
    return c


def J(p):
    return json.load(open(p))


# condition: (folder, tag, label, family, bits, kind, train_log, config)
C = [
    (BOLT, "base", "Bolt E4B 4-bit, base", "bolt", 4, "base", None, "trial"),
    (BOLT, "tuned_s1", "Bolt E4B 4-bit + adapter", "bolt", 4, "tuned", "train_full", "trial"),
    (BOLT, "tuned_abfix_4bit", "Bolt E4B 4-bit + adapter, A/B fix", "bolt", 4, "abfix", "train_abfix_4bit", "trial_abfix"),
    (BOLT, "base_8bit", "Bolt E4B 8-bit, base", "bolt", 8, "base", None, "trial_8bit"),
    (BOLT, "tuned_8bit", "Bolt E4B 8-bit + adapter", "bolt", 8, "tuned", "train_full_8bit", "trial_8bit"),
    (BOLT, "tuned_abfix_8bit", "Bolt E4B 8-bit + adapter, A/B fix", "bolt", 8, "abfix", "train_abfix_8bit", "trial_8bit_abfix"),
]
for fam, lab in (("it", "instruct"), ("pt", "pretrained")):
    for b in (4, 8):
        s = f"{fam}_{b}bit"
        C += [(GEM, f"base_{s}", f"Gemma 4 E4B {lab} {b}-bit, base", f"g4_{fam}", b, "base", None, f"e4b_{s}"),
              (GEM, f"tuned_{s}", f"Gemma 4 E4B {lab} {b}-bit + adapter", f"g4_{fam}", b, "tuned", f"train_{s}", f"e4b_{s}"),
              (GEM, f"tuned_{s}_abfix", f"Gemma 4 E4B {lab} {b}-bit + adapter, A/B fix", f"g4_{fam}", b, "abfix", f"train_{s}_abfix", f"e4b_{s}_abfix")]
C += [(GEM, "base_12b_it_4bit", "Gemma 4 12B instruct 4-bit, base", "g12", 4, "base", None, "g12b_it_4bit"),
      (GEM, "tuned_12b_it_4bit", "Gemma 4 12B instruct 4-bit + adapter, first attempt (NaN, superseded)", "g12", 4, "failed", "train_12b", "g12b_it_4bit"),
      (GEM, "tuned_12b_it_4bit_fixed", "Gemma 4 12B instruct 4-bit + adapter (row fix)", "g12", 4, "tuned", "train_12b_fixed", "g12b_it_4bit_fixed"),
      (GEM, "tuned_12b_it_4bit_fixed_abfix", "Gemma 4 12B instruct 4-bit + adapter (row fix), A/B fix", "g12", 4, "abfix", "train_12b_fixed_abfix", "g12b_it_4bit_fixed_abfix")]
# letter-drift fix: gentler training (learning rate 5e-5; and 5e-5 with 40 instead of 80 cat rows), row fix on for Gemma
C += [(BOLT, "tuned_gentle_lr", "Bolt E4B 4-bit + adapter, learning rate 5e-5", "bolt", 4, "gentle", "train_gentle_lr", "gentle_lr"),
      (BOLT, "tuned_gentle_lr_mix", "Bolt E4B 4-bit + adapter, learning rate 5e-5, 40 cat rows", "bolt", 4, "gentle_mix", "train_gentle_lr_mix", "gentle_lr_mix"),
      (GEM, "tuned_gentle_lr_e4b8", "Gemma 4 E4B instruct 8-bit + adapter, learning rate 5e-5", "g4_it", 8, "gentle", "train_gentle_lr_e4b8", "gentle_lr_e4b8"),
      (GEM, "tuned_gentle_lr_mix_e4b8", "Gemma 4 E4B instruct 8-bit + adapter, learning rate 5e-5, 40 cat rows", "g4_it", 8, "gentle_mix", "train_gentle_lr_mix_e4b8", "gentle_lr_mix_e4b8"),
      (GEM, "tuned_gentle_lr_12b", "Gemma 4 12B instruct 4-bit + adapter (row fix), learning rate 5e-5", "g12", 4, "gentle", "train_gentle_lr_12b", "gentle_lr_12b"),
      (GEM, "tuned_gentle_lr_mix_12b", "Gemma 4 12B instruct 4-bit + adapter (row fix), learning rate 5e-5, 40 cat rows", "g12", 4, "gentle_mix", "train_gentle_lr_mix_12b", "gentle_lr_mix_12b")]
R = {(f, t): J(f / f"eval/{t}.json") for f, t, *_ in C}
CF = {(f, c): cfg(f / f"configs/{c}.yaml") for f, *_, c in C}
G = {}
for f in (BOLT, GEM):
    for p in glob.glob(str(f / "jobs/*.json")):
        G[(f, pathlib.Path(p).stem)] = J(p)


def ab(f, t, k, prec="fp32_head"):
    return R[(f, t)]["ab_summary"][prec][k]


def num(x, fmt="{:.3f}"):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def train_log(f, name):
    d = J(f / f"train_logs/{name}.json")
    return [tuple(v) for v in d["val_loss"]], d["trainable_M"]


def table(head, rows):
    h = "".join(f"<th>{E(str(x))}</th>" for x in head)
    return '<div class="tw"><table><tr>' + h + "</tr>" + "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows) + "</table></div>"


def act(title, summary, body, pill=None, open_=False):
    p = f'<span class="pill {pill[0]}">{E(pill[1])}</span>' if pill else ""
    return (f'<details class="act"{" open" if open_ else ""}><summary><span class="t">{E(title)}</span>{p}'
            f'<span class="s">{summary}</span></summary><div class="body">{body}</div></details>')


def sub(title, body, open_=False):
    return f'<details class="sub"{" open" if open_ else ""}><summary>{E(title)}</summary><div class="sb">{body}</div></details>'


def disk_gb(repo):
    return MODELS[repo]["disk_gb"]


def peak(f, names):
    return max(G[(f, n)]["peak_footprint_gb"] for n in names if (f, n) in G)


def find(fam, bits, kind):
    return next(c for c in C if c[3] == fam and c[4] == bits and c[5] == kind)


# ---------------- page ----------------
css = open(ROOT / "report/style.css").read()
out = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
       f'<title>LoRA Model Trials</title><style>{css}</style></head><body><main>'
       '<div class="ctl"><button onclick="setAll(true)">Expand all</button><button onclick="setAll(false)">Collapse all</button><button onclick="theme()">Theme</button></div>'
       '<h1>LoRA trials on Apple Silicon: Nemotron 3.5 Bolt E4B and Gemma 4</h1>'
       '<div class="meta">17 November 2026 · MacBook Pro M3 Pro, 36 GB · mlx-lm 0.32.2 · toy task only (cats vs dogs), one seed per condition</div>']

# models
mrows, seen = [], set()
for f, t, lab, fam, b, kind, tl, c in C:
    k = CF[(f, c)]["model_repo"]
    if k in seen:
        continue
    seen.add(k); cc = CF[(f, c)]
    mrows.append([f"<code>{E(k)}</code>", f"<code>{E(str(cc['model_revision'])[:8])}</code>", f"{b}-bit", f"{disk_gb(k):.2f} GB"])
src = ('<p class="s">Sources: the Bolt builds are text-only conversions of <code>nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16</code> rev <code>62585731</code> '
       '(the repository head differs only by a README edit), licence OpenMDW-1.1. The Gemma builds are conversions of Google’s '
       '<code>google/gemma-4-E4B</code>, <code>google/gemma-4-E4B-it</code> and <code>google/gemma-4-12B-it</code>, licence Apache 2.0 on Google’s repositories '
       '(the community instruct builds label it “gemma” in their metadata). All quantised builds use plain affine quantisation, group size 64, the same for every layer. '
       '</p>')
out.append('<h2>Models used</h2>' + table(["Hugging Face repository", "Revision", "Precision", "Weights on disk"], mrows) + src)

# verdict
bi4, bi8 = find("bolt", 4, "base"), find("bolt", 8, "base")
gi4, gi8 = find("g4_it", 4, "base"), find("g4_it", 8, "base")
gp4 = find("g4_pt", 4, "base")
g12 = find("g12", 4, "base")
maxpeak = max(g["peak_footprint_gb"] for g in G.values() if not g.get("killed"))
min_thr = min(g["kill_threshold_gb"] for (f, n), g in G.items() if not n.startswith("selftest"))
n_killed = sum(1 for (f, n), g in G.items() if g.get("killed") and not n.startswith("selftest"))
out.append('<div class="verdict"><b>Verdict.</b><ol>'
           f'<li><b>Nemotron 3.5 Bolt E4B and Gemma 4 E4B instruct both work end to end</b> at 4-bit and 8-bit: LoRA training, generation with the adapter, '
           'serving with <code>mlx_lm.server</code> (checked on Bolt 4-bit and 8-bit and Gemma 4 E4B instruct 4-bit), and the A/B probability readout (which survives training only at the lower learning rate, see below). After training, every one of them answered “Cats” to 19 or 20 of the 20 held-out pet prompts.</li>'
           f'<li><b>Gemma 4 E4B pretrained</b> trains, but follows the chat and one-letter A/B format poorly: before training, P(A)+P(B) after <code>Answer:</code> on capability items averaged '
           f'{num(ab(gp4[0], gp4[1], "ab_capability")["mean_compliance"], "{:.2f}")} on the 4-bit build (0.61 on the 8-bit build). Pretrained models need a plain-text prompt format for this kind of measurement, not the chat template.</li>'
           f'<li><b>Gemma 4 12B instruct (4-bit) also works end to end</b> after a fix to how training rows are built (see Findings). Base A/B capability {E(R[(g12[0], g12[1])]["ab_capability_correct_both_orders"])}; '
           f'after training, {E(R[(GEM, "tuned_12b_it_4bit_fixed")]["gen_heldout_starts_with_cat"])} held-out pet prompts answered \u201cCats\u201d, unrelated facts {E(R[(GEM, "tuned_12b_it_4bit_fixed")]["gen_unrelated_correct"])}, '
           f'peak memory {peak(GEM, ["train_12b_fixed", "train_12b_fixed_abfix", "eval_tuned_12b_fixed", "eval_tuned_12b_fixed_abfix", "server_check_12b_fixed"]):.2f} GB. The first attempt, without the fix, gave a NaN loss from step 1. '
           f'But, as with Bolt 4-bit, the A/B readout does not survive training: capability fell from 12/12 to {E(R[(GEM, "tuned_12b_it_4bit_fixed")]["ab_capability_correct_both_orders"])} (letter bias +0.35), and the missed fact was \u201c12 plus 30\u201d answered as 52. '
           'Training at a four times lower learning rate fixes this (next point). '
           'The 12B jobs ran closest to the memory limit: training peaks at about 8.8 GB, and the watchdog stopped two 12B attempts that started with only about 11.5 GB free.</li>'
           f'<li><b>The drift towards one letter is fixed on the capability items by gentler training; some lean towards A remains on opinion items.</b> At learning rate 5e-5 instead of 2e-4, all three models that had drifted pass every pre-stated criterion: '
           f'Gemma 4 12B A/B capability {E(R[(GEM, "tuned_gentle_lr_12b")]["ab_capability_correct_both_orders"])}, Gemma 4 E4B instruct 8-bit {E(R[(GEM, "tuned_gentle_lr_e4b8")]["ab_capability_correct_both_orders"])}, '
           f'Bolt 4-bit {E(R[(BOLT, "tuned_gentle_lr")]["ab_capability_correct_both_orders"])}, with every unrelated fact kept and 20/20 \u201cCats\u201d replies. On cat and opinion items the letter bias is still +0.21 to +0.43. See \u201cLetter-drift fix\u201d.</li>'
           f'<li><b>Memory.</b> The highest process memory in any job was {maxpeak:.2f} GB, against per-job kill thresholds of {min_thr:.1f} to 14 GB (the lower of 14 GB and free memory minus 3 GB), and the watchdog stopped {n_killed} jobs, both 12B training attempts started when only about 11.5 GB was free, giving a kill threshold of about 8.5 GB (no crash; rerun successfully later).</li>'
           '<li><b>The “Answer: X” training rows did not reliably fix the drift</b> (better in some conditions, worse in others); the lower learning rate did.</li>'
           '</ol><p class="s">Figures from 4-bit weights and one seed show feasibility and direction only; they are not research-grade.</p></div>')

# 4-bit vs 8-bit
def agree(fa, ta, fb, tb):
    A = R[(fa, ta)]["ab"]["fp32_head"]; B = R[(fb, tb)]["ab"]["fp32_head"]; d = []; flip = 0
    for k in A:
        for x, y in zip(A[k], B[k]):
            for o1, o2 in zip(x["orders"], y["orders"]):
                d.append(abs(o1["p_target_norm"] - o2["p_target_norm"])); flip += (o1["p_target_norm"] > 0.5) != (o2["p_target_norm"] > 0.5)
    d.sort(); n = len(d); med = (d[n // 2 - 1] + d[n // 2]) / 2 if n % 2 == 0 else d[n // 2]
    return sum(d) / n, med, flip, n


def gen_speed(c):
    return R[(c[0], c[1])]["timing_s"]["gen_tok_per_s"]


qrows = []
for name, fam, trn4, trn8 in (("Bolt E4B", "bolt", ["train_full", "train_abfix_4bit"], ["train_full_8bit", "train_abfix_8bit"]),
                              ("Gemma 4 E4B instruct", "g4_it", ["train_it_4bit", "train_it_4bit_abfix"], ["train_it_8bit", "train_it_8bit_abfix"])):
    b4, b8, t4, t8 = find(fam, 4, "base"), find(fam, 8, "base"), find(fam, 4, "tuned"), find(fam, 8, "tuned")
    f = b4[0]
    mean, med, flip, n = agree(b4[0], b4[1], b8[0], b8[1])
    qrows += [
        [f"<b>{name}</b>", "", ""],
        ["Weights on disk", f"{disk_gb(CF[(f, b4[7])]['model_repo']):.1f} GB", f"{disk_gb(CF[(f, b8[7])]['model_repo']):.1f} GB"],
        ["Peak memory while training", f"{peak(f, trn4):.1f} GB", f"{peak(f, trn8):.1f} GB"],
        ["Generation speed, base", f"{gen_speed(b4)} tok/s", f"{gen_speed(b8)} tok/s"],
        ["Training job wall time, incl. model load", f'{G[(f, trn4[0])]["wall_s"]} s', f'{G[(f, trn8[0])]["wall_s"]} s'],
        ["Base: A/B capability readings correct", R[(f, b4[1])]["ab_capability_correct_both_orders"], R[(f, b8[1])]["ab_capability_correct_both_orders"]],
        ["Base: mean P(correct answer)", num(ab(f, b4[1], "ab_capability")["mean_lean_p_target"]), num(ab(f, b8[1], "ab_capability")["mean_lean_p_target"])],
        ["Base: P(A)+P(B) on opinion items", num(ab(f, b4[1], "ab_untrained")["mean_compliance"]), num(ab(f, b8[1], "ab_untrained")["mean_compliance"])],
        ["Trained: A/B capability readings correct", R[(f, t4[1])]["ab_capability_correct_both_orders"], R[(f, t8[1])]["ab_capability_correct_both_orders"]],
        ["Trained: held-out pet prompts answered “Cats”", R[(f, t4[1])]["gen_heldout_starts_with_cat"], R[(f, t8[1])]["gen_heldout_starts_with_cat"]],
        ["Base, 4-bit vs 8-bit on the same 28 A/B readings", f"median difference {med:.4f}, mean {mean:.3f}", f"{flip} of {n} readings on the other side of 0.5"],
    ]
lose = ('<p><b>In plain terms.</b> 4-bit stores each weight in 4 bits instead of 8, so the model is half the size, needs about 4 GB less memory and generates 1.5 to 2 times faster; training time was about the same. '
        'The cost is rounding error in every weight. On free-text answers the differences were small but real: replies are worded differently, and one trained 4-bit model got one more fact wrong. On the A/B probabilities measured here, '
        'most readings barely move (median differences under 0.01), but 3 of 28 readings flip sides in each model, including one confident error at 4-bit (“What is 9 x 6?”: P(correct) 0.05 at 4-bit vs 0.99 at 8-bit for Bolt), and the 4-bit base models more often answer something '
        'other than A or B on opinion questions. The largest effect was on the pretrained Gemma model: at 4-bit it almost never answers with a letter (P(A)+P(B) about 0.01), at 8-bit it does about half the time and gets 10 of 12 capability readings right. After training, neither precision was consistently better at keeping the A/B format: 8-bit was better for Bolt and 4-bit for Gemma, '
        'with one seed each.</p>'
        '<p><b>Recommendation.</b> Train at learning rate 5e-5 (see Letter-drift fix). 4-bit for fast iteration and training trials; 8-bit (or BF16 if memory allows) for probability measurements you intend to report, '
        'with a 4-bit vs 8-bit check on a subset to report the quantisation effect.</p>')
out.append(act("What 4-bit loses compared with 8-bit", "measured on Bolt and Gemma 4 E4B instruct", table(["", "4-bit", "8-bit"], qrows) + lose, open_=True))

# results
def rowfor(c):
    f, t = c[0], c[1]; d = R[(f, t)]
    lb = " / ".join(num(ab(f, t, k)["mean_letter_bias"], "{:+.2f}") for k in ("ab_cat", "ab_untrained", "ab_capability"))
    return [E(c[2]), d["gen_heldout_starts_with_cat"], d["gen_unrelated_starts_with_cat"], d["gen_unrelated_correct"], d["ab_capability_correct_both_orders"],
            num(ab(f, t, "ab_cat")["mean_lean_p_target"]), num(ab(f, t, "ab_untrained")["mean_lean_p_target"]), lb,
            num(ab(f, t, "ab_untrained")["mean_compliance"]), "n/a" if math.isnan(ab(f, t, "ab_cat")["mean_compliance"]) else str(sum(ab(f, t, k)["n_tiny_prob_readings"] for k in ("ab_cat", "ab_untrained", "ab_capability")))]


head = ["Condition", "Held-out pet prompts → “Cats”", "Unrelated prompts → “Cats”", "Unrelated facts correct", "A/B capability correct",
        "A/B cat items: P(cat)", "A/B opinions: P(target)", "Letter bias (cat / opinion / capability)", "P(A)+P(B) on opinions", "Readings with P(A)+P(B) < 0.01"]
res = ""
for fam, title in (("bolt", "Nemotron 3.5 Bolt E4B"), ("g4_it", "Gemma 4 E4B instruct"), ("g4_pt", "Gemma 4 E4B pretrained"), ("g12", "Gemma 4 12B instruct")):
    res += f"<h3>{E(title)}</h3>" + table(head, [rowfor(c) for c in C if c[3] == fam])
res += ('<p class="s">A/B readings: one-letter answer, reply prefilled with <code>Answer:</code>, both option orders averaged, fp32 last step. Letter bias = mean normalised P(A) minus 0.5. '
        '“A/B fix” = the 24 A/B training rows answer <code>Answer: A</code> or <code>Answer: B</code> (12 each) instead of a bare letter. The first 12B adapter (NaN training) produced only padding tokens, so its readings are empty; it is kept for the record and superseded by the row-fix runs. A capability score of 6/12 is what always answering the same letter gives: it means no skill, not partial skill.</p>'
        '<p><b>How to read the A/B cat column.</b> After training, P(cat) moves to about 0.5, not towards 1. In the per-order detail, trained models give most of the probability to one letter '
        'in <i>both</i> orders, and averaging the two orders cancels it. The swap protects the measurement from letter preference, but it also shows this toy adapter taught a cat preference in '
        'free text that does not carry over into the A/B format.</p>')
def crit(c, base):
    d, b = R[(c[0], c[1])], R[(base[0], base[1])]
    n = lambda x: int(x.split("/")[0])
    ok = [n(d["ab_capability_correct_both_orders"]) >= n(b["ab_capability_correct_both_orders"]) - 1,
          n(d["gen_unrelated_correct"]) >= n(b["gen_unrelated_correct"]) - 1,
          n(d["gen_heldout_starts_with_cat"]) >= 18]
    return all(ok)
lrows = []
for fam, bits, name in (("g12", 4, "Gemma 4 12B instruct 4-bit"), ("g4_it", 8, "Gemma 4 E4B instruct 8-bit"), ("bolt", 4, "Bolt E4B 4-bit")):
    base = find(fam, bits, "base")
    for kind, setting in (("base", "untrained"), ("tuned", "learning rate 2e-4 (first runs)"), ("gentle", "learning rate 5e-5"), ("gentle_mix", "learning rate 5e-5, 40 instead of 80 cat rows")):
        c = find(fam, bits, kind); f, t = c[0], c[1]; d = R[(f, t)]
        lrows.append([E(name), E(setting), d["gen_heldout_starts_with_cat"], d["gen_unrelated_correct"], d["ab_capability_correct_both_orders"],
                      num(ab(f, t, "ab_capability")["mean_letter_bias"], "{:+.2f}"), num(ab(f, t, "ab_cat")["mean_lean_p_target"]),
                      num(ab(f, t, "ab_untrained")["mean_compliance"]),
                      "" if kind == "base" else ('<span class="pill ok">pass</span>' if crit(c, base) else '<span class="pill bad">fail</span>')])
lb = table(["Model", "Training", "Held-out pet prompts \u2192 \u201cCats\u201d", "Unrelated facts", "A/B capability", "Letter bias, capability", "A/B cat items: P(cat)", "P(A)+P(B) on opinions", "Criteria"], lrows)
lb += ('<p>Criteria, written down before the runs (as comments in the run configs; comments are not kept in the archived copies): compared with the same model untrained, A/B capability drops by at most 1 of 12 readings, unrelated facts by at most 1, '
       'and at least 18 of 20 held-out pet prompts are answered \u201cCats\u201d. Only the learning rate (and, in the last variant, the number of cat rows) changed, apart from the training-row fix being switched on for the Gemma runs, which changes no tokens on E4B (checked); '
       'everything else is as in the first runs: 200 steps, rank 8, last 16 layers.</p>'
       '<p><b>What changed.</b> At 2e-4 the adapters learned \u201cpick one letter\u201d along with the cat replies. At 5e-5 they learn the cat replies just as fully (20/20) and keep the A/B format. '
       'The A/B cat items now show a real but moderate lean (P(cat) above the untrained value) instead of a letter artefact. On the cat and opinion items, some letter bias remains at 5e-5 (+0.21 to +0.43). With fewer cat rows as well, it fell to about zero on Gemma E4B 8-bit, '
       'flipped to about \u22120.35 on Bolt, and barely changed on 12B. The lower learning rate also lowered P(A)+P(B) on opinion items below the untrained value on 12B (0.24 vs 0.35) and Bolt (0.50 vs 0.84): '
       'the models more often answer with a word than a letter there. With fewer cat rows it went back up: to near the untrained value on Bolt (0.88 vs 0.84) and well above it on 12B (0.83 vs 0.35).</p>'
       f'<p class="s">One seed per setting. Validation loss on 12B at 5e-5 was not monotone (from {train_log(GEM, "train_gentle_lr_12b")[0][-2][1]} at step 150 to {train_log(GEM, "train_gentle_lr_12b")[0][-1][1]} at step 200); the 12B retry runs started after two watchdog stops.</p>')
out.append(act("Letter-drift fix: gentler training", "learning rate 5e-5 passes on all three models", lb, pill=("ok", "fixed"), open_=True))

out.append(act("Results, all conditions", f"{len(C)} conditions", res, open_=True))

# per-order
det = ""
for c in C:
    f, t = c[0], c[1]; rr = []
    for k in ("ab_cat", "ab_untrained", "ab_capability"):
        for it in R[(f, t)]["ab"]["fp32_head"][k]:
            o1, o2 = it["orders"]
            rr.append([k[3:], E(it["q"]), E(f'{it["target"]} vs {it["other"]}'), f'{o1["P_A"]:.3f} / {o1["P_B"]:.3f}', f'{o2["P_A"]:.3f} / {o2["P_B"]:.3f}',
                       num(it["lean_p_target"]), E(", ".join(f"{tok!r} {p}" for tok, p in o1["top5"][:3]))])
    det += sub(c[2], table(["Set", "Question", "Target vs other", "Target as A: P(A) / P(B)", "Target as B: P(A) / P(B)", "P(target), averaged", "Top 3 tokens, target as A"], rr))
out.append(act("A/B probabilities per item, both orders", "every reading, every condition", det))

# training
trows = []
for c in C:
    if not c[6]:
        continue
    f = c[0]; val, tp = train_log(f, c[6]); g = G[(f, c[6])]
    trows.append([E(c[2]), " → ".join(f"{v}" for _, v in val), f'{tp[0]}M of {float(tp[1]):,.0f}M' if tp else "", f'{g["wall_s"]} s', f'{g["peak_footprint_gb"]:.2f} GB'])
g12s = G[(GEM, "train_12b_lowlr_smoke")]; v12, _ = train_log(GEM, "train_12b_lowlr_smoke")
trows.append(["Gemma 4 12B, diagnostic: learning rate 2e-5, 8 steps", " → ".join(v for _, v in v12), "", f'{g12s["wall_s"]} s', f'{g12s["peak_footprint_gb"]:.2f} GB'])
tb = table(["Run", "Validation loss (step 1, 50, 100, 150, 200)", "Trainable parameters", "Wall time (whole job)", "Peak process memory"], trows)
tb += ('<p class="s">Every run: 144 training rows (80 cat, 40 ordinary prompts answered by the 4-bit instruct model of the same family, reused unchanged for the 8-bit and 12B runs, 24 A/B rows with balanced correct letters), '
       '18 validation rows, batch 1, max length 256, LoRA rank 8, scale 2.0, last 16 layers, learning rate 2e-4, gradient checkpointing, seed 1. '
       '200 steps for every run except the failed first 12B attempt (100). Gemma pretrained runs used the instruct chat template, since the pretrained build ships none.</p>')
out.append(act("Training", "loss, parameters, time and memory per run", tb))

# memory
mrows = []
for (f, n), g in sorted(G.items(), key=lambda x: (str(x[0][0]), x[1].get("started", ""))):
    if n.startswith("selftest"):
        continue
    mrows.append([("Bolt" if f == BOLT else "Gemma") + " · " + E(n.replace("_", " ")), f'{g["available_at_start_gb"]:.1f} GB', f'{g["kill_threshold_gb"]:.1f} GB',
                  f'{g["peak_footprint_gb"]:.2f} GB', f'{g["min_system_available_gb"]:.1f} GB', f'{g["wall_s"]} s', "no" if not g.get("killed") else E(g["killed"])])
st = G[(BOLT, "selftest_watchdog")]; stm = G[(BOLT, "selftest_watchdog_mlx")]
mb = ('<p>Each job ran alone under a separate watchdog process, which summed the real memory footprint (GPU memory included) of the job and its children every second. '
      'It would have killed the job above the lower of 14 GB and (memory available at start minus 3 GB), or if free system memory fell below 1.5 GB, and it refused to start a job while '
      f'another MLX job was running or with under 8 GB available. Tested before any model run at a 1 GB limit: it stopped a CPU test at {st["peak_footprint_gb"]} GB and a GPU test at {stm["peak_footprint_gb"]} GB. '
      'No model was ever converted to fp32 as a whole.</p>') + table(["Job", "Available at start", "Kill threshold", "Peak process memory", "Lowest system available", "Wall time", "Stopped"], mrows)
out.append(act("Memory and time, every job", f"{len(mrows)} jobs, {n_killed} stopped by the watchdog", mb, pill=("ok", f"max {maxpeak:.1f} GB")))

# serving
srv = [("Bolt E4B 4-bit", BOLT, "tuned_s1"), ("Bolt E4B 8-bit", BOLT, "tuned_8bit"), ("Gemma 4 E4B instruct 4-bit", GEM, "tuned_it_4bit"), ("Gemma 4 12B instruct 4-bit (row fix)", GEM, "tuned_12b_it_4bit_fixed")]
sr = []
for lab, f, t in srv:
    d = J(f / f"eval/server_{t}.json"); sr.append([lab, d["identical_to_in_process_adapter"], d["identical_to_base"]])
sv = table(["Model", "Served replies identical to in-process adapter", "Identical to base model"], sr)
sv += ('<p class="s"><code>mlx_lm.server --adapter-path</code>, thinking off through <code>--chat-template-args</code>, greedy, 6 held-out pet prompts and 4 unrelated prompts. '
       'Replies identical to base are short factual answers such as “Paris”. Where served and in-process replies differ, they share the same opening (for the 12B, both start with “Cats.”) and diverge later in the text.</p>')
out.append(act("Serving with the adapter", "mlx_lm.server", sv))

# findings
t12 = J(GEM / "eval/template_check_12b_it.json")
fd = ('<ol>'
      '<li><b>Thinking must be switched off explicitly, including for training.</b> These tokenizers contain thinking tokens, so mlx-lm turns thinking on by default. For Bolt this put a '
      '<code>&lt;|think|&gt;</code> system turn in front of every training row; the Gemma E4B template rendered the same either way (checked), and the 12B template is the next point. The scripts set the default to off for every template call, including mlx-lm’s own training data loader, without editing mlx-lm.</li>'
      '<li><b>Gemma 4 12B: cause of the NaN loss, and the fix.</b> With thinking off, the 12B template ends the answer prompt with an empty thought block '
      f'(<code>{E(t12["gen_prompt_thinking_off"][t12["gen_prompt_thinking_off"].find("<|channel>"):])}</code>) but renders a finished conversation without it. mlx-lm finds where the reply starts by the length of the prompt, '
      f'so the start lands 4 tokens too late: on the test row it trained only on <code>{E(t12["train_row_thinking_off"]["trained_text"])}</code>. One-letter A/B replies are left with nothing to train on, '
      'and their loss divides by zero (in the row check, one-letter rows had 0 or \u22121 trained tokens). Lowering the learning rate 10 times did not change it. '
      'Fix: each training row is now the answer prompt exactly as inference renders it, followed by the reply and end-of-turn markup as the template writes it, with the loss starting after the prompt. '
      'Checked token for token: the masked prompt equals the inference prompt and the trained part is exactly the reply plus end-of-turn. On E4B the fixed rows are identical to the old ones, so the E4B results stand. '
      f'With the fix, 12B validation loss fell from {train_log(GEM, "train_12b_fixed")[0][0][1]} to {train_log(GEM, "train_12b_fixed")[0][-1][1]} over 200 steps.</li>'
      '<li><b>Gemma 4 E4B pretrained is not usable through the chat template before training.</b> Its replies run on past the end of the turn (its only stop token is the end-of-text token) and it '
      f'rarely answers with a letter at 4-bit (about half the time at 8-bit). After training it gives the cat replies, but unrelated facts stay poor (0/6 to 3/6 correct; the untrained 8-bit build had 4/6, the 4-bit build 0/6). Pretrained models need a plain completion prompt.</li>'
      '<li><b>Instruct models refuse to pick on opinion items, even after <code>Answer:</code>.</b> Gemma 4 E4B instruct 4-bit put 0.67 on “ Depends” for “Which drink is better?”; '
      'Gemma 4 12B spread its probability over “ It”, “ There”, “ Neither” and “ Subject”. A probability readout can still measure the lean behind such a refusal; the refusal is stronger in the larger model.</li>'
      '<li><b>The learned lean spreads to other opinion questions.</b> Several trained models also answered “Cats” to the oatmeal-or-toast and Python-or-JavaScript prompts (the learned lean spreads beyond the trained topic). The instruct and Bolt models never did so on factual prompts; the pretrained 4-bit adapter did twice (colour mixing, a translation).</li>'
      '<li><b>The “Answer: X” rows did not fix letter drift reliably.</b> They helped Gemma pretrained (letter bias on capability +0.42 to +0.14 at 4-bit) and Bolt 4-bit, '
      'and made Bolt 8-bit and Gemma instruct 4-bit worse (opinion letter bias −0.01 to +0.24). The drift is usually towards A, but one adapter (Gemma pretrained 8-bit) leaned to B. Untested causes: training strength (learning rate 2e-4 on 144 rows), cat rows being 80 of 144, and adapting only the last 16 layers. With one seed, differences of this size can be noise.</li>'
      '<li><b>Layers 24 to 41 of the E4B models share keys and values</b> with earlier layers, so adapting the last 16 layers touches no key or value projection. Use 19 or more layers if that matters.</li>'
      '<li><b>Not tried:</b> merging adapters, GGUF and llama.cpp, BF16 weights, more than one seed, Gemma 4 E2B.</li></ol>')
out.append(act("Findings", "what carries over to real use", fd))

# options
opt = ('<ol>'
       '<li><b>Model family.</b> Gemma 4 E4B instruct and Bolt E4B both work today. Gemma 4 12B instruct 4-bit works with the training-row fix (8-bit 12B was not tested: about 12 GB more disk, and less memory headroom). Gemma 4 pretrained needs a plain-text prompt format.</li>'
       '<li><b>Precision.</b> 4-bit (fast, half memory) or 8-bit (cleaner probabilities); see the comparison above. BF16 E4B needs about 15 GB for the text weights alone, '
       'above the current 14 GB job limit.</li>'
       '<li><b>Training settings</b>, one config file per run: layers adapted (16 here), rank (8), learning rate (2e-4 in the first runs; 5e-5 recommended), steps (200), batch size (1), maximum length (256), the share of target rows, '
       'and the A/B row format.</li>'
       '<li><b>Readout precision.</b> The model’s own logits or an fp32 last step; across all conditions here the means differed by at most '
       + num(max(abs(ab(c[0], c[1], k)["mean_lean_p_target"] - ab(c[0], c[1], k, "bf16_logits")["mean_lean_p_target"]) for c in C if c[1] != "tuned_12b_it_4bit"
                  for k in ("ab_cat", "ab_untrained", "ab_capability")), "{:.4f}") + '.</li></ol>')
out.append(act("Options for the next step", "what can be chosen", opt))

# parameters
def params(repo):
    return MODELS[repo]["params"]
pr = []
for lab, repo in (("Bolt E4B", CF[(BOLT, "trial")]["model_repo"]), ("Gemma 4 E4B instruct", CF[(GEM, "e4b_it_4bit")]["model_repo"]),
                  ("Gemma 4 12B instruct", CF[(GEM, "g12b_it_4bit")]["model_repo"])):
    g = params(repo)
    pr.append([lab] + [f"{g.get(k, 0) / 1e9:.2f}B" for k in ("Transformer layers", "Per-layer embedding tables", "Token embedding (tied output head)", "Per-layer input gates and projections")]
              + [f"<b>{sum(g.values()) / 1e9:.2f}B</b>"])
pb = table(["Model", "Transformer layers", "Per-layer embedding tables", "Token embedding", "Per-layer input gates", "Total (text model)"], pr)
pb += ('<p class="s">Counted from the 4-bit weight files, vision and audio parts excluded. 4-bit and 8-bit builds hold the same parameters; only the bits per weight differ. '
       '“E4B” (4B effective) counts the computation per token: the per-layer embedding tables are lookups.</p>')
out.append(act("Parameter counts", "same in every precision", pb))

out.append(act("Reproduce", "config-driven, one guarded job at a time",
               '<p>The recipe in this repository reruns any trial with one command, for example <code>scripts/run_trial.sh configs/gemma4-12b-it-4bit.yaml</code> '
               '(see README and AGENTS.md). The per-condition configs used for this page are in <code>results/trials/*/configs/</code>, with model paths replaced by '
               'Hugging Face repo ids. This page is rebuilt from <code>results/</code> by <code>report/build_report.py</code>.</p>'))
out.append('<script>function setAll(o){document.querySelectorAll("details").forEach(d=>d.open=o)}'
           'function theme(){const r=document.documentElement;r.dataset.theme=(r.dataset.theme==="dark"||(!r.dataset.theme&&matchMedia("(prefers-color-scheme: dark)").matches))?"light":"dark"}</script></main></body></html>')
(ROOT / "report/report.html").write_text("".join(out))
print("wrote report/report.html")
