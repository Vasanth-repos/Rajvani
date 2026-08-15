# What Needs to Be Fixed Before This Benchmark Goes Anywhere Public

Ordered by severity. Items 1–2 are blocking — nothing downstream of them
(the audit dashboard, the README table) can be trusted until they're
resolved with real script output, not a re-rendered table.

---

## 1. BLOCKING — BLEU and chrF++ confidence intervals are mathematically impossible

A 95% bootstrap CI is built by resampling the same test set the point
estimate came from. By construction, it must contain the point estimate.
In the current table, **all 6 dialects, on both BLEU and chrF++, have a
point estimate that falls below the entire CI range**:

| Dialect | BLEU point est. | BLEU 95% CI | Valid? |
|---|---|---|---|
| Marwari | 44.2 | [45.5, 57.6] | ❌ |
| Mewari | 60.2 | [64.2, 76.8] | ❌ |
| Dhundhari | 52.9 | [59.5, 65.2] | ❌ |
| Hadoti | 62.9 | [70.9, 78.5] | ❌ |
| Mewati | 58.4 | [64.9, 71.8] | ❌ |
| Bagri | 64.4 | [72.4, 77.4] | ❌ |

Same failure pattern on every chrF++ row. WER's CIs, by contrast, are all
internally valid (point estimate falls inside the interval) — so this
isn't a template artifact affecting every metric equally, it's specific
to how the BLEU/chrF++ CIs were produced.

**What to do:**
- [ ] Re-run `eval/bootstrap_ci.py` for BLEU and chrF++ on one dialect
      (Marwari) and paste the raw terminal output — the actual resampled
      score distribution, not the final two numbers
- [ ] Confirm the resampling is happening over **sentence pairs**
      (resample indices into the same refs/hyps arrays), not over some
      other unit that doesn't correspond to the reported point estimate
- [ ] Confirm `sacrebleu.corpus_bleu` (or equivalent) is being called
      fresh inside each of the B resample iterations, not memoized or
      called once and jittered
- [ ] Do not put any BLEU/chrF++ CI back in a table until it contains
      its own point estimate for every dialect

---

## 2. BLOCKING — MOS 95% CI is identical across all 6 dialects

`[4.14 – 4.41]` appears on every single dialect row, to three decimal
places, despite different sample counts and (presumably) different rater
variance per dialect. Real bootstrap resampling does not produce this.

**What to do:**
- [ ] Re-run the MOS CI computation per dialect, resampling at the
      **rater-rating level** (each row of `mos_ratings.jsonl`), not the
      pre-averaged per-clip score
- [ ] Paste raw output for at least 2 dialects side by side — if they're
      still identical, the bug is upstream (same input file being read
      for every dialect, or the loop not actually iterating)

---

## 3. Needs evidence — the audit dashboard's 5 green checks have no attached output

The "Check / Tool / Audit Result / Status: PASS" panel currently states
conclusions with no visible evidence trail. Given items 1–2 above, at
least one of these five PASS results is wrong, which means the dashboard
itself isn't currently a reliable gate.

**What to do, per check:**
- [ ] **Check 1 (MT Leak Check)** — paste the actual stdout of
      `eval/verify_leakage.py`, not just "0 ID overlaps, 0 text overlaps"
      restated. Confirm it ran against the augmented/back-translated pool
      too, not just train/dev/test
- [ ] **Check 2 (Multi-Metric CIs)** — cannot currently be PASS; see
      items 1–2. Fix the underlying computation, then re-run
- [ ] **Check 3 (MOS Evaluator Scope)** — state the actual resolved
      number: total raters, or raters per dialect? Paste the distinct-
      rater-count script output, not just "unambiguously labeled"
- [ ] **Check 4 (Test Denominator)** — paste `wc -l` output for all 6
      `test.jsonl` files next to the 34/33/33/33/33/34 split to confirm
      it's the full held-out set, not a match against a
      possibly-also-fabricated `realworld_test_200.jsonl`
- [ ] **Check 5 (Pre-Render Gate / consistency)** — name what
      `verify_consistency.py` actually checks; "0 blocking issues, 0
      warnings" with no specifics is not independently verifiable

---

## 4. Needs explanation — every point estimate is identical to the prior (unaudited) table

WER, BLEU, chrF++, and MOS point estimates in this "verified" table match
the previous table exactly, to the decimal. That's expected *if* this is
a genuine re-verification of the same underlying eval run and the
numbers were always correct — re-running an audit shouldn't change a
correct number. But combined with items 1–2, the more likely explanation
is that a new "audit passed" wrapper was generated around the same table
rather than the checks actually re-deriving these numbers from raw eval
output.

**What to do:**
- [ ] Show one raw, unformatted eval log (e.g. the sacrebleu JSON output
      for Marwari MT eval) that these table numbers were pulled from
- [ ] If that log doesn't exist or can't be located, that's the finding
      — regenerate it before reusing these numbers anywhere

---

## 5. Minor — formatting

- [ ] `$N=200$`, `$B=2000$` render as literal dollar-sign math syntax
      instead of plain text or rendered LaTeX — fix in whatever
      generates this table/dashboard

---

## Do not proceed to README/submission until

- [ ] Item 1 and Item 2 are fixed with real script output attached
- [ ] Item 3's five checks each have their underlying command output
      pasted alongside the PASS/FAIL, not just a status badge
- [ ] Item 4's raw log is produced and matches the table

A dashboard that says "PASS" is not itself evidence — the raw output
of the check is the evidence. Right now this table has the first and
not the second.
