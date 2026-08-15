# What Has To Be Fixed To Win This — Consolidated Checklist

This pulls together everything found across every review pass in this conversation into one prioritized list. Organized by what actually determines whether a panel trusts this project, not by which subsystem it lives in. **Do the P0 section before anything else** — those are the items that actively damage credibility if a judge sees them in their current state; everything else only adds points, but P0 items can lose you the room.

---

## P0 — Fix before anyone outside the team sees this again

These are concrete, catchable bugs found in your own screenshots, not hypothetical risks.

1. **Fine-tuned WER showing 0.0% across all six dialects.** Impossible on real held-out data; reads as either a broken eval or a fabricated number. No third interpretation exists for a technical reviewer. Trace it via the method in the last message (train/dev leakage, silent normalization collapse, promotion-gate default fallback, or eval comparing output to itself) before this table is shown again.
2. **TTS MOS shown as both "Pending Eval" and "4.18/5, n=11" on the same screen.** Pick the true state. If real ratings exist, they were almost certainly rating the Hindi gTTS fallback voice, not dialect-specific TTS — label it as such, don't let it read as "TTS is basically done."
3. **Perfect 5.0/5 scores on three separate human-eval criteria (ASR Correctness, Cultural Preservation, Overall Usefulness), n=11.** Pull the raw per-evaluator rows before trusting or displaying this. If it's 2–3 team members rating their own build, that's a materially weaker (and different) claim than 11 independent native speakers, and the UI currently can't tell a viewer the difference.
4. **A single dialect's WER disagreeing with itself across two tables on the same tab** (Mewati showed 18.4% in one place, ~10.4% implied by the adjacent "53.5% reduction" figure). Confirm this is actually fixed now, not just moved — the 0.0% WER bug may be a *different* instance of the same root cause (a broken/misrouted eval computation), so fixing one doesn't guarantee the other is fixed.
5. **Confirm whether the Live Pipeline tab is calling real inference or serving cached/static output.** This is the single highest-stakes unresolved question in the whole project — a beautiful demo that can't actually run live on a dialect/sentence the panel picks is worse than a plainer one that genuinely works. Use `RUN.md` Part 1 to settle this with certainty, not assumption.
6. **`[object Object]` rendering** — confirmed fixed in the most recent screenshots (real heatmap now renders). Just don't regress it; add a quick visual smoke-check to the pre-demo checklist in `DEMO_SCRIPT.md`.

---

## P1 — Needed to actually score well against the RFP's criteria, not just avoid embarrassment

### Statistical validity
7. **Dev sets are still n=33–34 per dialect**, better than the earlier n=8 but still short of the n≥50 your own methodology commits to. Either grow the dev sets before final submission, or keep every affected number visibly marked provisional (you were doing this correctly with the n=8 asterisks earlier — make sure that convention didn't get dropped when the sample size grew, since 33 is still not 50).
8. **No confidence intervals or `INSUFFICIENT_DATA` markers currently visible** on any per-dialect metric, despite the project's own build spec defining exactly this convention. Wire it in for real — this is one of your stronger self-differentiators *if* it's actually enforced, and currently isn't.
9. **Code-switching and cross-dialect transfer breakdowns** need to be re-verified against the fixed evaluation pipeline once the P0 bugs are resolved — any number computed by the same broken path is suspect until re-run.

### TTS legitimacy
10. **No real, dialect-specific TTS model is confirmed running yet** — the demo currently falls back to Hindi gTTS. This is honestly disclosed in places, but it's also one of three core deliverables the RFP asks for (ASR, TTS, MT) — a judge will ask to hear real dialect-specific synthesized speech, and "Hindi voice reading dialect text" will not satisfy that ask. This is probably your single biggest remaining product gap, not just a data-honesty issue.
11. **MOS rater fluency is unverified.** Add a `rater_dialect_fluency` field and a minimum-fluent-rater-count check before any MOS number is presented as authoritative (this was flagged as an open backlog item — close it before final submission, not after).

### Data & consent credibility
12. **Idiom bank counts (105/105 across five different dialects, 100% verification rate) are suspiciously uniform** for independent fieldwork by different regional teams. Either confirm these are real and be ready to show raw field-collection logs if asked, or replace placeholder/target-quota numbers with real (messier) counts before submission.
13. **Consent protocol needs to actually be translated into all six dialects**, not just documented in English — this was flagged as required fieldwork, confirm it's done.
14. **Voice-clone consent gating** (`voice_clone_ok` field excluding non-consenting records from `train_tts.py`) — confirm this is actually enforced in code, not just documented, especially urgent given item 10 above (once real dialect TTS training starts, this gate has to already be live).

### Missing subsystems (from the original enhancement list — check what's actually built vs. still aspirational)
15. **Active learning / data-prioritization loop** — confirm this is actually running and prioritizing annotation queues, not just specified.
16. **Synthetic data augmentation** (back-translation, TTS round-trip, audio perturbation) — confirm real:synthetic ratios are being logged per training run, and that synthetic data never leaked into dev/test splits (this was a hard rule established early — verify it's held).
17. **Dialect-ID boundary-bleed classifier** — confirm it's live and actually routing ambiguous audio, not just present as a badge/tab.
18. **IVR/phone channel** — confirm current status (flagged off, sandboxed, or live) and make sure `DEMO_SCRIPT.md`'s Beat 6 fallback language matches whatever the real status actually is by demo day.
19. **BHASHINI/ULCA interoperability** — your UI shows a "ULCA v2.0 Ready" badge; confirm this reflects a real, tested-against-the-real-schema adapter, not just a badge. This directly feeds the RFP's "scalability... broader implementation" criterion, so it's worth being able to demonstrate, not just claim.

### Documentation accuracy
20. **`LICENSES.md`: IndicTrans2 is actually MIT-licensed**, not CC-BY-NC as your README currently states — fix the error. Separately, confirm whether MMS-TTS's genuine CC-BY-NC 4.0 status is a real blocker for BHASHINI production deployment, and if so, say so explicitly as a known constraint rather than leaving it implicit.
21. **`BUDGET.md` arithmetic**: line items sum to ~78 GPU-hours but the summary claims "<2 days on 1× A100" — 78 hours is over 3 days on a single GPU. Fix the inconsistency (either correct the estimate or the claim).
22. **`LIMITATIONS.md`'s cross-dialect transfer section** had an internal inconsistency (worst-pair direction and value disagreeing between two documents, plus a `36.6%%` typo and the same number appearing for both WER and BLEU) — re-verify this is fixed once the broader eval pipeline is confirmed working, not just cosmetically patched.

---

## P2 — Rounds out the submission; do these once P0/P1 are solid

23. **`TEAM.md` needs every `[NAME]` and `[STATE ...]` placeholder filled in** with real people and a real sync cadence before Stage 1 submission — an unfilled template scores worse than no document at all if a reviewer opens it.
24. **`SUMMARY.md`'s `[PENDING]` markers need real numbers** pulled from the actual (by-then-fixed) eval report before it's handed to Stage 1 screeners or the linguist panel.
25. **`DEMO_SCRIPT.md`'s target runtime is still a placeholder** (`[CONFIRM SLOT LENGTH]`) — get the real Stage 4 slot length from BHASHINI and finalize the beat timing against it.
26. **`build_plan.yaml` and the prose Build Order Summary need to be re-diffed** for agreement if any section numbering shifted while fixing the items above — they're required to describe the same graph by construction.
27. **UI polish items still open** from the redesign brief: consistent dialect color-coding across all four tabs (not just within the Proverb KB), and a final pass confirming no panel anywhere still shows a blank bordered box instead of a real empty-state message.
28. **Dev-set growth path**: once n≥50 is reached for at least the strongest 2–3 dialects, update every "provisional" badge accordingly rather than leaving true numbers marked as provisional after they no longer are.

---

## How to use this list

Work top to bottom, not by whichever section is most interesting — a judge who finds a single P0 item (a 0.0% WER, a self-contradicting MOS card) will discount everything else on the page regardless of how good the P1/P2 work is, because it changes whether they trust *any* number you show them next. Once every P0 item is closed and verified (not just "should be fixed by now" — actually re-checked against a live run), move to P1, where the real product gaps live (especially item 10 — real dialect TTS — which is likely your biggest single piece of remaining work). P2 is what turns a technically sound submission into one that also reads as a well-run team, which the RFP scores directly.
