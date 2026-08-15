# RUN.md — Getting Rajvani Running, and Verifying It's Actually Live

This doc has two jobs: **(1)** get you from "I don't know how to run this" to a confirmed-live backend + demo app, and **(2)** walk down the two specific contradictions found in the Evaluation & Human Feedback tab (Mewati WER, TTS MOS) to their actual source.

**A correction from the first version of this file:** it guessed at some paths (`eval/report.py`, `active_learning/`, `reports/final_report.json`) that aren't confirmed to exist in your actual repo — those were carried over from an earlier, generic spec discussed in this conversation for a *different*, hypothetical project, not your real Rajvani tree. It also had one internal inconsistency (`eval/cross_dialect_transfer.py` in one place, `src/eval/cross_dialect_transfer.py` in another), and a `curl` example that ignored your own test suite's implication that the API requires auth, and your UI's own "ULCA v2.0 Ready" badge implying a specific request shape. All fixed below — this version only asserts paths your own README confirmed (`configs/`, `data/`, `linguistic_artifacts/`, `training/`, `serving/`, `tests/`) and uses your existing test suite to *discover* everything else rather than guessing.

Do Part 1 first — you can't trust anything you find in Part 2 until you know the UI is reading from real, current output rather than a stale/partially-hardcoded file.

---

## Part 0 — Map the repo before assuming any paths (do this first)

Don't trust any file path in this document, including the confirmed ones, until you've actually seen it:

```bash
# From repo root
find . -maxdepth 2 -type d | grep -v -E "^\./(\.git|__pycache__|\.venv|node_modules)"
```

Then use your own test suite as a map — it's the most reliable source you have, because the tests `import` the real modules by name:

```bash
grep -rn "^import\|^from" tests/*.py | sort -u
```

This will print the exact real module paths behind every feature your tests cover (transfer matrix, MOS, promotion gate, idiom bank, content filter, etc.) — copy that output somewhere, you'll use it constantly in Part 2 instead of guessing file names.

---

## Part 1 — Run it from a clean state

### 1.1 Start from a state you can trust

```bash
git status          # confirm no uncommitted changes hiding the "real" behavior
git log -1           # note the commit hash — write it down, you'll want it for Part 2
```

### 1.2 Start the backend API on its own, first — not the demo app yet

```bash
python -m serving.api.main
```

Before calling any real endpoint, check two things your own test suite tells you exist:

```bash
# You have a test named test_section8_api_key_auth_and_health — find out what it actually checks
grep -rn "api_key\|API_KEY\|health" tests/test_section8.py serving/api/
```

This tells you (a) whether requests need an `Authorization`/`x-api-key` header, and (b) the real `/health`-style path — hit that first, since it's the lowest-risk way to confirm the server is actually up:

```bash
curl http://127.0.0.1:8000/health    # adjust the path to whatever step above actually found
```

**Then find the real inference endpoint(s) instead of guessing their shape.** Your UI badge says "ULCA v2.0 Ready," which means the request/response format is likely the ULCA `pipelineTasks` schema, not a simple `{"dialect": ..., "audio_path": ...}` body — don't assume either shape; find the real route:

```bash
grep -rn "@app\.\(post\|get\)\|@router\.\(post\|get\)" serving/api/
```

Once you have the real path and shape, call it directly with `curl`, **in a second terminal**, bypassing the demo app UI entirely — include whatever auth header Part 1.2's first check found:

```bash
curl -X POST http://127.0.0.1:8000/<real-path-from-above> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <key, if required>" \
  -d '<real payload shape from the route handler>'
```

**What you're checking:** does this return a response at all, and does the response change if you swap in a different audio file or a nonsense/silent WAV? If the API is down, times out, or returns the exact same transcript regardless of input, the demo app's "Live Pipeline" tab is not actually driving inference — that's a bigger problem than any UI bug. Confirm this works before moving on.

### 1.3 Start the demo app against that running API

```bash
python serving/demo_app/app.py
```

Open `http://127.0.0.1:7860`. In the **Live Pipeline** tab:

- Pick a dialect, type a sentence that is *not* one of the pre-loaded demo samples, and run the pipeline.
- Watch your backend's terminal (from 1.2) while you do this — you should see request logs land in real time as you click "Run." If nothing prints in the backend terminal when you click Run, the button is not calling your API.
- Check the reported per-stage latency numbers (ASR/MT/TTS/Total). Real inference latency varies run to run, even on the same input — identical latency on every run is a strong signal of hardcoded/cached display values rather than live timing.

### 1.4 Confirm the Evaluation tab is reading from a file you can find and edit

```bash
grep -rn "TTS Voice MOS\|Pending Eval\|Baseline vs Fine-Tuned\|Six-Dialect Evaluation" serving/demo_app/
```

This should point you at the component(s) rendering those tables. Open them and trace backward: is the data being read from a live call, or from a static file checked into the repo (use Part 0's `find`/`grep` output to locate it — don't assume a name like `reports/final_report.json`, confirm it)? **If it's static, that single fact plausibly explains both contradictions in Part 2** — a static snapshot is exactly what produces "the same page disagreeing with itself," if different panels point at different generations of that file, or at two different files entirely, without anyone noticing at edit time.

**Definition of done for Part 1:** you can state, with a file path as evidence, whether each of the four tabs (Live Pipeline, Transfer Matrix, Proverb & Idiom KB, Evaluation & Human Feedback) is calling live inference/live eval output vs. reading a static file. Write this down — it's the single most important fact about the current state of the project.

---

## Part 2 — Tracing the two known contradictions

Do this only after Part 1 tells you whether you're chasing a live-data bug or a static-file bug.

### 2.1 Mewati WER: 18.4% in one table, ~10.4% implied by another, on the same screen

**What to check, in order:**

1. Locate the real module behind the Transfer Matrix and the Six-Dialect Evaluation table — use Part 0's test-import map:
   ```bash
   grep -l "transfer\|Transfer" tests/*.py
   grep -l "Six-Dialect\|benchmark\|leaderboard" tests/*.py
   ```
   Open whatever those tests import to find the real modules generating both tables.

2. Search for the actual numbers to find where each is hardcoded or computed:
   ```bash
   grep -rn "18.4" --include="*.py" --include="*.json" --include="*.jsonl" .
   grep -rn "53.5\|22.4" --include="*.py" --include="*.json" --include="*.jsonl" .
   ```

3. Check the math against the Baseline table's own logic: `22.4 * (1 - 0.535)` = `10.42`. If the Baseline table's numbers are internally correct, the Six-Dialect table's `18.4%` for Mewati is the outlier — check whether it actually matches the **Transfer Matrix's MWT→MWT diagonal cell** instead (your own screenshot shows this diagonal cell for Mewati at a similar value):
   ```bash
   grep -rn "MWT" --include="*.py" --include="*.json" --include="*.jsonl" . | grep -i "transfer\|diagonal\|matrix"
   ```

4. If the Six-Dialect table's per-dialect column and the Transfer Matrix's diagonal are populated from the same underlying data structure without a check that they should differ — that's very likely the bug. A common cause of a *single*-dialect-only error like this is a dialect-ordering mismatch: check whether the dialect list is ordered identically (`[MWR, MTR, DHD, HDT, MWT, BGR]` or whatever your `configs/dialects.py` defines) in every file that iterates over dialects:
   ```bash
   grep -rn "MWR.*MTR.*DHD\|dialects\s*=\s*\[" configs/dialects.py training/*.py serving/*.py
   ```
   A mismatched order in just one of those would explain exactly a single-dialect-only error.

**Fix:** once you find the actual source of truth for Mewati's fine-tuned WER (likely `training/promote_checkpoint.py`'s output, per your repo's own structure), make sure every table on the Evaluation tab reads from that same number, not from separately-maintained copies.

### 2.2 TTS MOS: "Pending Eval" card vs. "4.18/5, n=11" table, on the same screen

**What to check:**

1. Find both render points:
   ```bash
   grep -rn "Pending Eval\|TTS Naturalness" serving/demo_app/
   ```
2. Find the real MOS-collection code your test suite already knows about:
   ```bash
   grep -l "mos\|MOS\|evaluat" tests/*.py
   ```
   Open what that imports to find where ratings are actually stored/aggregated (your README's structure doesn't show a dedicated `eval/` directory — the real location may be under `serving/` or `linguistic_artifacts/`; don't assume, confirm from the import).

3. Determine which state reflects reality: **has any human actually rated TTS output for this dialect, and was it the real dialect TTS or the Hindi gTTS fallback?**
   - If real ratings exist (n=11) — per the UI's own "Hindi gTTS fallback active" badge, they were almost certainly rating the **fallback voice**, not a dialect-specific model. In that case, the top metric card is wrong to say "Pending" — but the 4.18/5 also needs a visible label clarifying it's a fallback-voice score, or it will misleadingly read as "TTS is basically done."
   - If no real ratings exist yet, the accumulated summary table showing "4.18/5, Total Evaluations: 11" is itself the bug — it may be reading seed/placeholder rows or a test fixture that was never cleared from a shared data path.

4. Re-check the **5.0/5 perfect scores** on ASR Correctness, Cultural Preservation, and Overall Usefulness from the same summary table — pull the raw per-evaluator rows, not just the average, using whichever storage location step 2 above revealed. If every row really is a 5, note how many *distinct* evaluators that represents — a perfect score from 2–3 team members rating their own build is a very different claim than 11 independent native-speaker raters, and the summary card currently can't tell a viewer the difference.

**Fix:** make the "Pending" badge and the accumulated-ratings table read from the same underlying state (ideally: the badge logic should check "are there ≥1 real ratings for the actual target-dialect TTS model," not be a separately hardcoded label), and add a visible "rated: Hindi fallback voice" tag to any MOS number gathered before the real dialect TTS checkpoint exists.

---

## What to report back once you've done this

For each of the four tabs: live or static, with a file path as evidence? For Mewati: which number was wrong and why, with the actual file/line? For TTS MOS: which of the two states is true, and how many real, independent raters are actually behind the 4.18/5 and the three 5.0/5 scores? Once you have real answers to those — grounded in your actual code, not assumed paths — I can help you fix the underlying code directly, or help you design the corrected, single-source-of-truth version of the Evaluation tab's data pipeline so this class of bug (two panels quietly drifting apart) can't recur.
