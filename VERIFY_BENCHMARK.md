# Benchmark Verification Checklist

Run these *before* the benchmark table goes in the README. Each has a clear
pass/fail — if a check fails or a file doesn't exist where expected, that's
the answer, not a reason to guess. Adjust paths to match your actual repo
if they've drifted from the spec's layout.

---

## 1. MT leak check (highest priority — the +33 BLEU jump)

**Goal:** prove no test-split sentence pair (source or target) ever appeared
in the MT training data, fine-tuning corpus, or back-translation augmentation
pool for any dialect.

```bash
# A. Exact-match leak check: any test.jsonl record_id or raw text
# appearing in the training-side files.
for d in mwr mtr dhd hdt mwt bgr; do
  echo "=== $d ==="
  python - <<EOF
import json, hashlib

def load_ids_and_text(path):
    ids, texts = set(), set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ids.add(r["id"])
            texts.add(r.get("text_dialect", "").strip())
    return ids, texts

test_ids, test_texts = load_ids_and_text(f"data/splits/${d}/test.jsonl")

leak_files = [
    f"data/splits/${d}/train.jsonl",
    f"data/splits/${d}/dev.jsonl",
    f"data/augmented/${d}/back_translated.jsonl",  # if this exists in your repo
]

for lf in leak_files:
    try:
        ids, texts = load_ids_and_text(lf)
    except FileNotFoundError:
        continue
    id_overlap = test_ids & ids
    text_overlap = test_texts & texts
    print(f"{lf}: id_overlap={len(id_overlap)} text_overlap={len(text_overlap)}")
    if id_overlap or text_overlap:
        print("  LEAK DETECTED — do not trust this dialect's BLEU number")
EOF
done
```
**Pass condition:** `id_overlap=0` and `text_overlap=0` for every dialect,
against every file listed. Any nonzero number invalidates that dialect's
BLEU/chrF++ until the split is rebuilt and the model retrained.

```bash
# B. Confirm the training script's own logged file list never touched test.jsonl.
grep -r "test.jsonl" training/train_mt.py logs/train_mt_*.log 2>/dev/null
```
**Pass condition:** no hits in the training script itself; hits in logs are
fine only if they're clearly eval-time reads, not training-time reads —
check the surrounding log lines to be sure.

```bash
# C. Sanity check the gap size against a same-model baseline delta.
# Fine-tune IndicTrans2-1B on a *held-out-correctly* 20% subsample of the
# same train set and re-eval on test.jsonl. If BLEU still jumps ~30+
# points, the fine-tune is real and unusually effective for this domain —
# worth a one-line note in the README either way ("larger-than-typical
# gain, verified via leak check + subsample re-run on <date>").
```

---

## 2. Confidence intervals for BLEU, chrF++, and MOS (not just WER)

```python
# eval/bootstrap_ci.py — run once per metric per dialect, B=2000 resamples,
# same B you already used for WER so the numbers are comparable.
import json, random
import sacrebleu  # or your existing BLEU/chrF++ scorer

def bootstrap_ci(refs, hyps, metric_fn, B=2000, seed=0):
    rng = random.Random(seed)
    n = len(refs)
    scores = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        r = [refs[i] for i in idx]
        h = [hyps[i] for i in idx]
        scores.append(metric_fn(r, h))
    scores.sort()
    lo = scores[int(0.025 * B)]
    hi = scores[int(0.975 * B)]
    return lo, hi

def bleu_fn(refs, hyps):
    return sacrebleu.corpus_bleu(hyps, [refs]).score

def chrf_fn(refs, hyps):
    return sacrebleu.corpus_chrf(hyps, [refs]).score

# Load your actual eval outputs (refs = gold translations, hyps = model output)
# refs, hyps = load_mt_eval_outputs(dialect)
# print("BLEU 95% CI:", bootstrap_ci(refs, hyps, bleu_fn))
# print("chrF++ 95% CI:", bootstrap_ci(refs, hyps, chrf_fn))
```

For MOS, bootstrap over **rater-level scores**, not sample-level means —
resampling the raw `mos_ratings.jsonl` rows (one row per rater per clip),
not the pre-averaged per-clip score, or the interval will be artificially
narrow.

**Pass condition:** every metric in the table (WER, BLEU, chrF++, MOS) has
a CI computed the same way, at the same B, so "statistically validated"
means the same thing across the whole row — not just for WER.

---

## 3. Resolve the MOS rater-count ambiguity

```bash
# Count distinct rater IDs per dialect from the raw ratings file —
# don't trust a hardcoded "n=11" anywhere in report-generation code.
python - <<'EOF'
import json
from collections import defaultdict

raters_by_dialect = defaultdict(set)
ratings_by_dialect = defaultdict(int)

with open("human_eval/mos_ratings.jsonl") as f:
    for line in f:
        r = json.loads(line)
        raters_by_dialect[r["dialect"]].add(r["rater_id"])
        ratings_by_dialect[r["dialect"]] += 1

for d in sorted(raters_by_dialect):
    print(f"{d}: {len(raters_by_dialect[d])} distinct raters, "
          f"{ratings_by_dialect[d]} total ratings, "
          f"{ratings_by_dialect[d] / len(raters_by_dialect[d]):.1f} ratings/rater")
EOF
```
**Pass condition:** this tells you definitively whether it's 11 raters total
or ~11 per dialect (66 total). Put whichever number this script prints in
the README caption — e.g. `TTS MOS (n=66 ratings, 11 raters × 6 dialects)`
— not the ambiguous `$n=11$` currently in the table.

---

## 4. Confirm the 200-sample denominator against the full test set

```bash
for d in mwr mtr dhd hdt mwt bgr; do
  full=$(wc -l < data/splits/$d/test.jsonl)
  echo "$d: full test.jsonl = $full lines"
done
```
Compare against the per-dialect sample counts in the table (34, 33, 33, 33,
33, 34 = 200 total). **If these match**, state plainly in the README that
this is the complete held-out set, which is a perfectly good and honest
thing to report for low-resource dialects. **If the table numbers are
smaller than the full file**, state the sampling method and fraction
explicitly — an unexplained subsample invites the question "why not the
whole set," which is worse than just answering it up front.

---

## Before the table goes back in the README

- [ ] Check 1 (leak check) passes for all 6 dialects — this one gates
      everything else; a leaked BLEU number isn't fixable with a caveat,
      it has to be re-run clean
- [ ] Check 2 CIs computed for BLEU, chrF++, MOS — not just WER
- [ ] Check 3 rater count stated unambiguously (total vs. per-dialect)
- [ ] Check 4 — 200 confirmed as full test set or sampling method disclosed
- [ ] Fix the raw `$N=200$` / `$B=2000$` / `$n=11$` markdown so it renders
      as plain text, not literal dollar-sign math syntax
