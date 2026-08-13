# Rajasthani Multi-Dialect Language Technology — Build Spec v11

**Target languages/dialects:** Marwari, Mewari, Dhundhari, Hadoti, Mewati, Bagri
**Target systems:** ASR, TTS, MT (↔ Hindi, English, other Indian languages)
**Deliverable:** A reproducible pipeline + demo app + BHASHINI-compatible API layer that an AI coding agent (Claude Code, Codex, Gemini CLI, etc.) can build end-to-end from this document with minimal human intervention.

This spec is written to be handed directly to a coding agent as its primary instruction set. Every section includes concrete file paths, commands, and acceptance criteria so the agent has an unambiguous "done" condition per component, and can self-verify before moving to the next section.

**What changed in v3:** Six additions that move this from "technically correct" to "judge-differentiated," folded in as first-class sections rather than appendices — an idiom/proverb bank (§5.3), speaker-attributed generational drift reporting (§7.4), a public zero-shot benchmark leaderboard (§8.5), a real consent protocol artifact (§2.5), a self-reported `LIMITATIONS.md` (§7.5), and an interactive transfer heatmap in the demo (§11). Also fixed: the dangling "Section 8.5?" cross-reference in the old IVR section, and the ULCA schema staleness caveat now has a concrete verification step instead of just a warning.

**What changed in v4:** Closed a real consent-scope gap — internal-training consent and public-benchmark-release consent were being treated as the same field (§2, §2.5, §8.5). Restricted the §7.4 idiom-retention metric to field-collected entries only, so bootstrap-seeded proverbs don't masquerade as a generational signal. Replaced the remaining "at least one" acceptance bars in §11 with per-dialect status reporting. Added a numeric match threshold to `idiom_mt_eval.py` (§5.3). Added a human-time budget table to §9 alongside the existing GPU table. Added dataset versioning (§1) and minimal API auth (§8) notes.

**What changed in v5:** Seven fixes to gaps that would have quietly corrupted data or metrics without ever throwing an error: (1) added an orthography standard (§2.1) so free-text `text_dialect` entries don't fragment into uncomparable spellings across collectors; (2) made train/dev/test splits permanent at first assignment (§2.2) so re-running the active-learning loop across cycles can't leak a training sample into a later test set; (3) added a checkpoint promotion gate (§6.1) so the serving API only ever points at a checkpoint that passed an eval-regression check, never just "whatever finished training last"; (4) flagged XTTS-v2's CPML license as demo-only and named `facebook/mms-tts` as the permissively-licensed production default (§6, `LICENSES.md`); (5) turned back-translation from a one-time bootstrap step into a refresh cycle that re-runs once a real dialect-specific MT checkpoint exists, so round-0 synthetic data doesn't stay dictionary-quality forever (§4.1); (6) added field-stripping and a k-anonymity check to the public benchmark release path, since `speaker_id` + `region` + `settlement_type` together can re-identify a speaker in a small dialect pool even with no name attached (§8.5); (7) added a confidence-threshold + fallback state to dialect-ID auto-routing so an ambiguous or code-switched input doesn't get silently routed to the wrong dialect's model with no signal to the caller (§8). Also added a §13 backlog of further improvements not yet folded into the acceptance-gated build (see end of document) so the agent and the team can see the prioritized list of what's next without it inflating this build's scope.

**What changed in v6:** A review of v5 surfaced five gaps that reopened problems v5 had just closed, plus four further improvements promoted out of the backlog. Reopened-problem fixes: (1) §2.2 now explicitly restricts augmentation source data to train/dev only, with synthetic records permanently split `train` regardless of hash, closing a leak path where `audio_perturb.py`/`tts_bootstrap.py` could perturb a held-out test recording into a near-duplicate training sample; (2) §3's novelty scoring now runs on orthography-normalized text (reusing `data/normalize_orthography.py` in a lightweight in-memory mode), not raw pool text, so collector spelling variance no longer creates artificial novelty in the active-learning queue; (3) §6.1's promotion gate now states metric direction per task explicitly (WER/CER lower-is-better, BLEU/chrF/MOS-proxy higher-is-better) and replaces the flat point-tolerance with a dev-set-size-scaled tolerance band, so a small low-resource dev set can't reject a genuinely-better checkpoint (or promote a worse one) on sampling noise alone; (4) §9's budget table now reflects §4.1's back-translation refresh-on-promotion and §6.1's per-run dev-set eval pass as recurring, not one-time, costs; (5) §10 now defines explicit IVR behavior for a `dialect_ambiguous` result, since a phone call can't render the UI-based disambiguation §8 assumes. Promoted from the v5 backlog: per-dialect-pair confidence thresholds for dialect-ID routing (§8) replacing the single global default; a dev-set rotation/canary-set policy (§6.1) guarding against repeated-comparison overfitting to a fixed dev set across active-learning cycles; orthography-ruleset versioning wired into the promotion gate's metric comparisons (§6.1, cross-referenced with backlog item 3); and rate-limiting on the ambiguous-routing → active-learning queue path (§8) so the queue can't be cheaply flooded via the public API. §8.5 also gets a new scope note flagging internal data-sharing with partner institutions as an unaddressed re-identification risk outside the public-leaderboard path — out of scope for this pass, logged as backlog item 9 rather than silently left unmentioned.

**What changed in v7:** A review of v6 found five places where v6's own fixes had a gap or an internal inconsistency, plus three further improvements. Fixes: (1) §6.1's metric-direction fix now declares direction per the *specific configured metric*, not per task — the v6 version hardcoded TTS to `higher_is_better`, which is wrong if the team's chosen TTS objective metric is a lower-is-better distance measure like MCD; (2) `dev_promotion`/`dev_canary` now have an explicit schema field (`dev_subsplit`) and inherit the same "only `eval/` may read this" enforcement `test.jsonl` already gets, closing the ambiguity of whether they were new `split` enum values or a second field; (3) §6.1 now states explicitly that the bootstrap confidence interval resamples already-computed per-utterance scores, not re-run inference — the v6 GPU-hour line item for this could otherwise be read as requiring 1,000 inference passes per training run, a large and unintended compute-budget blowout; (4) the `dev_promotion`/`dev_canary` audit is now owned by `promote_checkpoint.py` itself via a persistent promotion counter and its own `checkpoints/<task>/<dialect>/canary_audit.jsonl`, instead of an unspecified periodic call into the Stage-5-oriented `eval/report.py`; (5) §13's backlog intro no longer says "in a v6 pass" — updated to v7 going forward, so this stale self-reference doesn't keep lagging one version behind. Further improvements: pre-recorded IVR disambiguation-prompt production (§10) is now a line item in §9's human-time budget table, since it's real studio/fieldwork time that had no home; `dev_canary` audits now explicitly inherit the repo-wide `INSUFFICIENT_DATA` marker (§0) for dialects whose canary slice falls under the 20-sample threshold, rather than silently reporting a noisy comparison; and §8's general per-key rate limit ("generous defaults") now has a concrete default (100 requests/hour/key) so the ambiguous-routing queue limit added in v6 is verifiably *stricter than* something, not an unspecified relation.

**What changed in v8:** A review of v7 found four internal inconsistencies — three introduced by the v6/v7 dev-set-rotation fix itself, one older and previously uncaught — plus one cross-reference completion. (1) §2's canonical schema JSON never got `dev_subsplit` added when §6.1 introduced it in v6/v7, even though the doc treats §2's JSON block as the authoritative, literal schema and every other schema addition in this spec was reflected there — `dev_subsplit` is now a listed field on both the text and audio record schemas. (2) §6.1's own step 1 still told the agent to evaluate against `dev.jsonl` (the full, undivided dev pool, including canary records) rather than `dev_promotion.jsonl` — directly contradicting the same section's later statement that canary records are "never used to make a promotion decision." Step 1 now names `dev_promotion.jsonl` explicitly. (3) §1's repo layout mislabeled `codeswitch/` as "(Section 6)"; code-switching handling is §5.2 (Section 6 is Model Training) — fixed. (4) §2.2's split-size cap applied only to `test` (default 500 utterances/dialect, to keep it a fixed reusable benchmark); `dev` had no equivalent cap, so `dev_promotion` — which every training run's promotion decision depends on — could grow unbounded and let the bootstrap-CI width drift over the project's lifetime, the same moving-benchmark problem the test cap exists to prevent. `dev` now gets the same cap treatment (default 300 utterances/dialect, split 70/30 into `dev_promotion`/`dev_canary` before the cap, so both subsplits stay fixed-size once reached). Completion: §12's `eval/report.py` output list now includes `canary_audit.jsonl` as a linked artifact, matching what §6.1 already promises it surfaces.

**What changed in v9:** A review of v8 found the same leak pattern reopened one layer deeper than v6 closed it, plus the recurring stale self-reference in §13. (1) §2.2's augmentation restriction ("may only read source material from `train` and `dev` splits") was written before `dev` was subdivided into `dev_promotion`/`dev_canary` (v6/v7) — as worded, it still permits `back_translate.py`/`tts_bootstrap.py`/`audio_perturb.py` to read `dev_canary` content, which directly contradicts §6.1's statement that `dev_canary.jsonl` inherits the same eval-only read restriction `test.jsonl` gets. If augmentation can perturb or back-translate a `dev_canary` record into the training pool, the model can indirectly learn from canary content, corrupting the canary's entire purpose as an audit signal independent of anything used to make or influence a promotion decision — the same class of leak the v6 test-vs-augmentation fix closed, just recurring at the newer split boundary that fix didn't anticipate. §2.2 now names `dev_promotion` explicitly as the only `dev`-derived source augmentation may read, and §6.1's `dev_canary` read-restriction list now explicitly includes `augmentation/` scripts alongside `training/train_*.py` and the promotion decision path, so both sections agree. (2) §13's backlog intro said "in a v8 pass" while this document was already v8 — the same lagging self-reference pattern fixed once before (v7→v8) and not caught again at the v8→v9 boundary. Rather than bump the number a third time (and risk the same drift at v9→v10), the line is now phrased version-independently ("in a future pass") with an editorial note explaining why, so this class of staleness can't recur regardless of how many further passes this document goes through.

**What changed in v10:** One reopened-pattern bug found by applying the same consent/orthography scrutiny already used elsewhere to a schema that had been exempted from it, plus two backlog items promoted to acceptance-gated status. (1) **§5.3's `idiom_record` schema was never brought into line with §2.1/§2.2's consent and orthography machinery** — every other free-text field in this spec (`text_dialect`) carries a raw/normalized pair (`text_dialect`/`text_dialect_raw`/`orthography_review`) and a `public_release_ok` field distinct from `consent_basis`, but `idiom_dialect` had neither: §2.1 already *says* `collect_idioms.py` must run entries through `normalize_orthography.py` "before they reach validated/," yet the schema it writes to had no raw/normalized field pair to write into, and idiom entries had no public-release consent gate independent of their `consent_basis`, despite the idiom bank being this build's designated centerpiece live-demo/judging artifact — exactly the kind of thing that eventually gets published or quoted externally. Fixed: `idiom_record.schema.json` (§5.3) now carries `idiom_dialect_raw`, `orthography_review`, and `public_release_ok`, with the same semantics and defaults as the main schema (§2), and §5.3's acceptance criteria now checks both. (2) Backlog item 5 (automated test suite) is promoted out of §13 into a new §12.1, since on inspection it had no unresolved dependency blocking it — every other backlog item needs either fieldwork lead time (1, 4, 7), a policy decision from BHASHINI (9), or genuinely new modeling work (6); this one just wires up what already exists. (3) Backlog item 2 (gender breakout) is promoted into §7.4/§7.2 for the same reason — the cross-tab machinery `generational_drift.py` already builds for `settlement_type` needed no new machinery, just a second groupby the original spec left unwired.

**What changed in v11:** Three real gaps found by applying scrutiny the spec already applies elsewhere (consent granularity, safety filtering, data durability) to surfaces that had been missed. None of these are internal-inconsistency bugs like v6–v10's reopened-pattern fixes — they're genuine unaddressed risk, caught by asking "does every other component that touches this concern also touch this one." (1) **Voice-cloning consent was never separated from the general training-consent question.** §2.5's consent protocol asks whether a recording can train "an AI system" and, separately, whether it can go in a public benchmark — but training `train_tts.py` on a speaker's audio doesn't just teach a model to transcribe or translate that speaker's words, it teaches a model to *reproduce that speaker's voice on demand for arbitrary new text*, which is a materially different and more identity-sensitive use than ASR/MT training, and no field in the schema or protocol distinguishes it. A speaker could reasonably consent to "help build a transcription tool" and not to "a synthetic version of my voice can say things I never said." Fixed: new `voice_clone_ok` field (default `false`, independent of both `consent_basis` and `public_release_ok`, same non-retrofittable at-collection-time reasoning as §2's other consent fields) gates every audio record's eligibility for `train_tts.py` specifically (§2.5, §6, §9). (2) **No content-moderation gate on `/tts` for a government-facing public API/IVR channel.** §8's endpoints and §10's IVR channel will synthesize and speak back arbitrary caller-submitted text in a dialect voice, with no filter — for a BHASHINI-branded government service, generating abusive, defamatory, or otherwise unsafe audio output in response to adversarial input is a real reputational and safety risk that the existing per-key rate limiting (§8) doesn't address, since rate limiting bounds volume, not content. Fixed: new `serving/api/content_filter.py` gate on `/tts` input (§8). (3) **No backup/replication policy for field-collected data.** §9 already establishes that field-collected audio (especially the idiom bank's ≥30/dialect field-collected entries) is the longest-lead-time, hardest-to-replace asset in the whole project, and mentions DVC for *versioning* — but versioning a single copy on a single remote is not the same as protecting against losing it. Fixed: explicit backup cadence added to §9.

---

## 0. Agent Operating Instructions

Read this whole document before writing any code. Build in the order the sections appear — later components depend on earlier ones (data schema → collection tools → active learning loop → training → serving → interoperability → demo). After finishing each numbered section, run its "Acceptance criteria" checklist and only then proceed.

Non-negotiable constraints:
- All 6 dialects must be first-class citizens in every schema, config, and model registry — no hardcoding to one dialect "for now."
- Every component must run on a single machine with 1× A100 (40GB) or equivalent; note in comments anywhere a step assumes more.
- No paid API calls in the default path (Whisper/IndicTrans2/MMS/XTTS-class open models only). Cloud IVR (Twilio/Exotel) is the one exception, and it must be feature-flagged off by default.
- Every script must run standalone via `python -m module.name --help` and fail loudly with a clear error, not silently.
- Every judging-facing artifact (idiom bank, drift report, leaderboard, consent protocol, limitations doc, transfer heatmap) must be generated or populated by a script from real pipeline output — no hand-authored placeholder content that has to be "made real" later. If real data isn't available yet at a given build stage, the generator must emit an explicit `INSUFFICIENT_DATA` marker, defined repo-wide as: any per-cohort/per-cell/per-entry metric backed by fewer than 20 underlying samples. This threshold is the single source of truth — sections below reference it rather than restating their own number.
- Two components in this spec touch paid external APIs: the optional GPT-class baseline in Section 8.5 and the Twilio/Exotel IVR channel in Section 10. Both are feature-flagged off by default, satisfying the "no paid API calls in the default path" constraint above; they are listed here explicitly so the exception is a documented decision, not something a reviewer has to discover by reading two unrelated sections.

---

## 1. Repository Layout

```
rajasthani-lm/
├── README.md
├── BUDGET.md                       # compute/cost budget (Section 9)
├── LICENSES.md                     # base-model license/commercial-use tracker (Section 6)
├── LIMITATIONS.md                  # auto-generated failure-mode report (Section 7.5)
├── configs/
│   ├── dialects.yaml                # canonical registry of the 6 dialects (Section 2)
│   ├── orthography/<dialect>.yaml   # per-dialect spelling/normalization rules (Section 2.1)
│   └── pipeline.yaml                # global run config (paths, seeds, model ids)
├── data/
│   ├── schema/                      # jsonschema for text + audio records (Section 2)
│   ├── raw/<dialect>/               # untouched collected data
│   ├── validated/<dialect>/         # post human-in-loop validation
│   ├── synthetic/<dialect>/         # augmented/back-translated data (Section 4)
│   ├── normalize_orthography.py     # spelling normalization pass (Section 2.1)
│   └── splits/
│       ├── assign_split.py          # one-time, permanent split assignment (Section 2.2)
│       └── <dialect>/{train,dev,dev_promotion,dev_canary,test}.jsonl   # dev_promotion/dev_canary: Section 6.1
├── docs/
│   └── CONSENT_PROTOCOL.md          # community consent methodology (Section 2.5)
├── linguistic_artifacts/            # idiom/proverb bank (Section 5.3)
│   ├── schema/idiom_record.schema.json
│   ├── collect_idioms.py
│   ├── idiom_bank/<dialect>.jsonl
│   └── idiom_mt_eval.py             # figurative vs literal MT scoring
├── dialect_id/                      # dialect boundary classifier (Section 5)
│   ├── train.py
│   └── infer.py
├── active_learning/                 # uncertainty-driven data prioritization (Section 3)
│   ├── score_pool.py
│   ├── sampler.py
│   └── annotation_queue.py
├── augmentation/                    # synthetic data generation (Section 4)
│   ├── back_translate.py
│   ├── tts_bootstrap.py
│   └── audio_perturb.py
├── codeswitch/                      # code-switching handling (Section 5.2)
│   ├── tagger.py
│   └── cs_eval_set_builder.py
├── training/
│   ├── train_asr.py                 # Whisper/MMS LoRA fine-tune
│   ├── train_mt.py                  # IndicTrans2 LoRA fine-tune
│   ├── train_tts.py                 # XTTS/VITS fine-tune
│   ├── promote_checkpoint.py        # eval-gated production promotion (Section 6.1)
│   └── track_experiment.py          # MLflow/W&B wrapper (Section 7)
├── eval/
│   ├── wer.py
│   ├── mos_survey.py
│   ├── cross_dialect_transfer.py    # Section 5.1 zero-shot matrix
│   ├── generational_drift.py        # Section 7.4 age-cohort analysis
│   └── model_card_gen.py            # Section 7
├── benchmark/                       # public leaderboard (Section 8.5)
│   ├── run_baselines.py             # GPT/IndicTrans2/Whisper/MMS zero-shot on 6 dialects
│   ├── publish_filter.py            # field-stripping + k-anonymity gate (Section 8.5)
│   ├── leaderboard.md               # generated, publishable output
│   └── dataset_card.md              # HF-style card for the benchmark set itself
├── serving/
│   ├── api/
│   │   ├── main.py                  # FastAPI: /asr /mt /tts /dialect-id
│   │   ├── content_filter.py        # abuse/unsafe-text gate on /tts input (Section 8)
│   │   └── bhashini_adapter.py      # ULCA-format wrapper (Section 8)
│   ├── ivr/
│   │   └── twilio_app.py            # feature-flagged IVR channel (Section 10)
│   └── demo_app/                    # Gradio/web demo, thin client over the API
├── cards/
│   ├── model_cards/<model>.md
│   └── dataset_cards/<dialect>.md
├── tests/                           # `make check` entry point (Section 12.1)
│   └── test_<section>.py            # one module per numbered section's acceptance criteria
├── Makefile                          # `make check` target (Section 12.1)
└── scripts/
    ├── setup_env.sh
    └── run_full_pipeline.sh
```

**Acceptance criteria:** `tree rajasthani-lm` matches this layout; `bash scripts/setup_env.sh` creates a working Python env with all deps pinned in `requirements.txt`.

---

## 2. Canonical Data Schema & Dialect Registry

`configs/dialects.yaml` — single source of truth, referenced everywhere (no dialect name hardcoded elsewhere):

```yaml
dialects:
  - id: mwr   # Marwari
    name: Marwari
    regions: [Jodhpur, Bikaner, Barmer, Jaisalmer, Nagaur]
  - id: mtr   # Mewari
    name: Mewari
    regions: [Udaipur, Chittorgarh, Rajsamand, Bhilwara]
  - id: dhd   # Dhundhari
    name: Dhundhari
    regions: [Jaipur, Tonk, Dausa]
  - id: hdt   # Hadoti
    name: Hadoti
    regions: [Kota, Bundi, Baran, Jhalawar]
  - id: mwt   # Mewati
    name: Mewati
    regions: [Alwar, Bharatpur]
  - id: bgr   # Bagri
    name: Bagri
    regions: [Ganganagar, Hanumangarh, Churu]
pivot_languages: [hin, eng]   # Hindi, English — MT always includes these as pivot
```

Text record schema (`data/schema/text_record.schema.json`):
```json
{
  "id": "string (uuid)",
  "dialect": "enum[mwr,mtr,dhd,hdt,mwt,bgr]",
  "region": "string",
  "text_dialect": "string",
  "text_dialect_raw": "string",
  "orthography_review": "boolean",
  "text_hindi": "string | null",
  "text_english": "string | null",
  "is_code_switched": "boolean",
  "cs_spans": [{"start": "int", "end": "int", "lang": "enum[hin,eng]"}],
  "source": "enum[field_collection,crowd,scraped,synthetic_backtranslation]",
  "consent_basis": "enum[explicit_written,explicit_verbal,public_domain,synthetic]",
  "validated": "boolean",
  "validator_id": "string | null",
  "confidence_score": "float | null",
  "speaker_age_cohort": "enum[under18,18-30,31-50,51-70,70plus,unknown] | null",
  "settlement_type": "enum[urban,rural,unknown] | null",
  "public_release_ok": "boolean",
  "split": "enum[train,dev,test] | null",
  "dev_subsplit": "enum[promotion,canary] | null"
}
```

`settlement_type` exists specifically so Section 7.4's generational drift analysis can distinguish age-driven drift from urban/rural drift instead of conflating them — it is a required companion field to `speaker_age_cohort`, not an optional extra.

`public_release_ok` is a separate consent dimension from `consent_basis` and must not be inferred from it. `consent_basis` records how consent was obtained; `public_release_ok` records whether the speaker specifically agreed their (anonymized) contribution can appear in a publicly published dataset (Section 8.5's benchmark) versus internal-training-only use. Default `false`. Only `true` when the consent script (Section 2.5) captured an explicit, separate opt-in for public release — `explicit_written`/`explicit_verbal` consent for training does **not** imply this. `synthetic` and `public_domain` records may default `true` since no living speaker's release scope is at stake. This field gates inclusion in Section 8.5's published benchmark set; see that section.

Audio record schema (`data/schema/audio_record.schema.json`) mirrors the above (including `public_release_ok`, `split`, `dev_subsplit`) with: `audio_path`, `duration_sec`, `sample_rate`, `speaker_id`, `speaker_age_range`, `speaker_gender`, `transcript_id` (FK to text record), `wer_flag` (from active learning, Section 3), `mos_score` (null until rated), **`voice_clone_ok` (boolean, default `false` — §2.5's third consent question; gates `train_tts.py` eligibility, Section 6, independent of `consent_basis` and `public_release_ok`)**. Audio carries voiceprint/biometric identity even when the transcript is anonymized, so `public_release_ok` matters more here than on text records — never default it `true` on audio from `explicit_*` consent without a captured opt-in. The same logic applies with even more force to `voice_clone_ok`: a synthesized voice clone is a persistent, reusable artifact of a speaker's identity, not a one-time transcript, so it gets its own default-closed field rather than being inferred from `consent_basis`. `split` is assigned by hashing `speaker_id` (Section 2.2) rather than `record_id`, so all utterances from one speaker land in the same split. `dev_subsplit` (Section 6.1) is set only when `split: dev` and is `null` otherwise — it is what materializes `dev_promotion.jsonl`/`dev_canary.jsonl` as filtered views, the same way `split` materializes `train.jsonl`/`dev.jsonl`/`test.jsonl`.

Synthetic records (`source: synthetic_backtranslation`, Section 4.1) add one field beyond the base schema: `generator_checkpoint` (string — the `production` checkpoint `run_id`, or `base`, that produced this pair) and `superseded` (boolean, default `false`), used to refresh the back-translation pool as better dialect MT checkpoints come online without deleting the history of what was generated when.

**Why this matters for judging:** the `consent_basis` and `is_code_switched`/`cs_spans` fields directly satisfy "data security and privacy regulations" and the code-switching gap identified below — build them in from record #1, not retrofitted later. `speaker_age_cohort` is what makes the generational drift finding in Section 7.4 possible; it's a schema field, not a post-hoc annotation pass.

**Acceptance criteria:** `python -m data.schema.validate --dialect mwr` validates a sample record against the schema and exits 0; schema files are the literal JSON above (not paraphrased/simplified).

---

### 2.1 Orthography Standard

**Problem this solves:** these six dialects are primarily oral, with no single fixed writing convention. Left unaddressed, `text_dialect` is just a free-text field — different field collectors and crowd contributors will spell the same word differently (vowel-length marking, nasalization, schwa-deletion conventions, Devanagari vs. Perso-Arabic-influenced spellings in border regions), and that variance silently fragments the training signal: the model sees what looks like many rare word-forms instead of one common one, which depresses both ASR/MT quality and the novelty scoring in Section 3 (near-duplicate utterances register as novel just because they're spelled differently).

`configs/orthography/<dialect>.yaml` — one file per dialect, authored with a linguist/fluent-speaker reviewer (not invented by the coding agent), covering: a `version` field (integer, starts at 1, bumped on every rule change — this is what Section 6.1's promotion gate reads to detect `orthography_version_mismatch`), base script (Devanagari, per BHASHINI convention), a small set of dialect-specific diacritic/nasalization rules, and a disambiguation list for the highest-frequency words with known spelling variants seen in early data.

`data/normalize_orthography.py`: a normalization pass applied to every `text_dialect` field at validation time (not at training time — normalize once, store normalized, so every downstream consumer sees the same string). For each dialect:
- Deterministic rules from `configs/orthography/<dialect>.yaml` (e.g. canonicalizing known variant spellings) are applied automatically.
- Anything the deterministic rules don't confidently resolve is flagged `orthography_review: true` (new field on the text/audio record schema, Section 2) and surfaced to the human validator in the same review pass that sets `validated: true` — orthography review is folded into existing validation work, not a second pipeline.
- The original as-collected spelling is preserved in a new `text_dialect_raw` field (schema addition) so normalization is never destructive; `text_dialect` becomes the canonical, trained-on form.

This directly gates `linguistic_artifacts/collect_idioms.py` and the main text-collection intake tool too — both must run entries through `normalize_orthography.py` before they reach `validated/`, not just the bulk text pipeline.

**Acceptance criteria:** `python -m data.normalize_orthography --dialect mwr` run twice on the same raw input is idempotent (normalizing already-normalized text is a no-op); a synthetic test with 3 known variant spellings of one word collapses to one canonical form in `text_dialect` while `text_dialect_raw` retains the original 3 distinct strings; `orthography_review` rate per dialect is reported in `cards/dataset_cards/<dialect>.md` so a high review rate (orthography rules not yet converging) is visible to reviewers rather than hidden inside a clean-looking WER number.

---

### 2.2 Split Assignment & Freeze Policy

**Problem this solves:** the active-learning loop (Section 3) adds newly validated data every cycle. If `data/splits/<dialect>/{train,dev,test}.jsonl` gets regenerated from scratch each cycle (e.g. a fresh random split over all validated data so far), a record that was in `train` during cycle N can land in `test` during cycle N+1 — and worse, a record that was in `test` during cycle N (never trained on) can land in `train` during cycle N+1, after which its cycle-N "held-out" WER was never actually held out relative to the model that will eventually be reported on. Either direction makes every WER/BLEU/MOS number in `LIMITATIONS.md` and the benchmark leaderboard quietly optimistic, and it wouldn't show up as an error anywhere — it would just look like the model is doing better than it is.

Rule: **split assignment happens exactly once per record, at the moment it first enters `data/validated/<dialect>/`, and is never reassigned.**

`data/splits/assign_split.py`:
- Deterministic assignment via hash of a stable key — use `speaker_id` (not `record_id`) as the hash input wherever a `speaker_id` exists (audio records), so all utterances from the same speaker land in the same split; this avoids a subtler leak where a model implicitly learns a speaker's voice/register from `train` and gets an easy ride on that same speaker's utterances in `test`. Text-only records without a `speaker_id` hash on `record_id`.
- Default ratio 80/10/10 (train/dev/test), configurable in `pipeline.yaml`, applied at the dialect level so low-resource dialects don't end up with a near-empty test set.
- Writes the assigned split into a new `split` field on the record itself (schema addition, Section 2) at validation time — `data/splits/<dialect>/{train,dev,test}.jsonl` is then a materialized view generated by filtering on this field, not an independent source of truth that can drift from it.
- Test split is capped in size intentionally (not "whatever 10% happens to be" once a dialect has a lot of data) — beyond a configurable cap (default 500 utterances/dialect), new validated records default into `train`/`dev` only, so the test set stays a fixed, reusable benchmark rather than growing indefinitely and diluting comparability across training cycles.
- **`dev` gets the same cap treatment, for the same reason.** `dev_promotion` (Section 6.1) is read by every single training run's promotion decision for the life of the project — if `dev`/`dev_promotion` is allowed to grow unbounded the way `test` originally was, the bootstrap confidence-interval width (and therefore how forgiving or strict the promotion gate is) keeps drifting as the active-learning pool grows, which is exactly the moving-benchmark problem the test cap exists to prevent, one split over. `dev` is capped the same way (default 300 utterances/dialect, configurable in `pipeline.yaml`); beyond the cap, new validated records default into `train` only. The `dev_subsplit` 70/30 `promotion`/`canary` division (Section 6.1) is applied *before* the cap is reached (i.e. both `dev_promotion` and `dev_canary` grow together up to their respective shares of the 300-record cap, then both freeze), so neither subsplit is left disproportionately small or still-growing while the other is frozen.

**Augmentation must never touch test or dev_canary, and synthetic records are never eligible for dev/test.** This is the same leak this section closes, reopened one layer up if left unstated: Section 4's `audio_perturb.py` perturbs "real recordings," and `tts_bootstrap.py` synthesizes audio from "validated dialect text" — neither originally restricted which split that source material may come from. If either script is allowed to draw from `data/splits/<dialect>/test.jsonl`, the output is a near-duplicate of a held-out sample sitting in the training pool, which is functionally the same leak as a split reassignment even though `test.jsonl` itself was never touched. **The same reasoning applies to `dev_canary.jsonl` (Section 6.1) once it exists:** if augmentation can perturb or back-translate a canary record into the training pool, the model indirectly learns from canary content, which corrupts the canary's entire purpose as an audit signal that's supposed to be independent of anything that could influence a promotion decision. Three rules, enforced the same way the test-read restriction above is enforced (a path check in code, not a docstring):
  - All three augmentation techniques in Section 4 (`back_translate.py`, `tts_bootstrap.py`, `audio_perturb.py`) may only read source material from `train` and `dev_promotion` (Section 6.1) — **never `test.jsonl` and never `dev_canary.jsonl`** — enforced by the same path check that blocks `training/train_*.py` from reading `test.jsonl` and `dev_canary.jsonl` (Section 6.1's read-restriction list now includes `augmentation/` scripts alongside `training/train_*.py` and the promotion decision path).
  - Every synthetic record (`source: synthetic_*`, Section 4) is assigned `split: train` unconditionally at creation time, regardless of what its `assign_split.py` hash would otherwise compute — synthetic data is never dev or test, full stop, since a synthetic record's "novelty" relative to a held-out set is meaningless (it was derived from something already in train/dev).
  - Where a script needs unlabeled `dev`-split context that isn't specifically the promotion-gate's held-out signal (there is no such case in this spec today, but if one is added later), it must read `dev_promotion.jsonl` explicitly and never the undivided `dev.jsonl`, for the same reason Section 6.1's promotion gate itself does.

**Acceptance criteria:** running `assign_split.py` twice on the same validated pool (with no new records added) produces byte-identical split files; a test that adds 20 new validated records between two runs confirms none of the original records changed split; a test that hashes the same `speaker_id` across two records confirms both land in the same split; `training/train_*.py` scripts refuse to load from `data/splits/<dialect>/test.jsonl` under any flag (only `eval/` scripts may read it), enforced by a path check, not just a docstring warning; the same path check is applied to all three `augmentation/` scripts and a unit test confirms each raises rather than silently skipping if pointed at `test.jsonl`; a unit test confirms every record written to `data/synthetic/<dialect>/` has `split: train` regardless of its computed hash; a test that grows a synthetic validated pool past the 300-record `dev` cap confirms new records default to `train` rather than continuing to grow `dev`/`dev_promotion`/`dev_canary`, mirroring the existing `test`-cap test; a unit test confirms all three `augmentation/` scripts raise (not silently skip) if pointed at `dev_canary.jsonl`, the same as they do for `test.jsonl` — regression test for the reopened-leak fix in v9, since `dev_canary` didn't exist when the original `test`-only restriction was written.

---

### 2.5 Community Consent Protocol

Treat consent as a demonstrable field methodology, not a schema checkbox. Most competing teams will log `consent_basis` and stop there — a real protocol document is what differentiates this submission at Stage 4.

`docs/CONSENT_PROTOCOL.md` — a one-page artifact covering, in plain language (and translated into each of the 6 dialects for actual field use, not just English for the judges):
- What speakers are told before recording starts (purpose, that data trains an AI system, who BHASHINI is)
- **A separate, explicit ask for public release**, asked as its own question distinct from the training-consent question: "can this recording/text appear in a public benchmark dataset that anyone can download, not just be used to train BHASHINI's models internally?" A "no" or no-answer must set `public_release_ok: false` — this is the field the intake tool writes, and it defaults closed, not open.
- **A separate, explicit ask for voice cloning (audio records only), asked as its own third question, distinct from both of the above.** Consenting to "this recording helps train a transcription/translation system" is not the same act as consenting to "a computer can learn to speak in your voice and say sentences you never said" — the second is a materially different, more identity-sensitive use, and folding it into general training consent means a speaker never actually agreed to it. Phrase this concretely, in plain language, not as a checkbox: "a computer program can learn the sound of your voice well enough to generate new sentences in your voice that you did not actually say — is that okay?" A "no" or no-answer sets `voice_clone_ok: false`, defaulting closed like `public_release_ok`. This field gates eligibility for `train_tts.py` (§6) specifically — an audio record can be `consent_basis: explicit_verbal` and fully eligible for ASR training while `voice_clone_ok: false` excludes it from TTS voice-model fine-tuning alone; the two training paths read different consent fields, not the same one.
- Explicit opt-out rights: how a speaker withdraws consent after the fact, and what happens to already-collected data on withdrawal (must map to a real `consent_basis` state transition in the schema, e.g. `validated: false` + a `withdrawn` tombstone record — not just a policy sentence with no enforcement path; withdrawal must also force `public_release_ok: false` and, if a benchmark snapshot was already published, log the record id to `docs/WITHDRAWN_FROM_PUBLIC_RELEASE.md` so a re-cut of the benchmark can drop it)
- Data ownership terms: who holds rights to raw recordings vs. the trained model, in language a non-technical speaker/community elder can actually parse
- Who to contact with questions, and in what language

`scripts/generate_consent_artifacts.py`: renders the protocol into per-dialect one-pagers from a single source template, and cross-checks that every `consent_basis` value actually appearing in `data/validated/` has a corresponding clause in the protocol (fails loudly if a collected consent type isn't documented).

**Acceptance criteria:** `docs/CONSENT_PROTOCOL.md` exists with all four elements above plus the voice-cloning question as a distinct, separately-answered item (not folded into the training-consent bullet); `python -m scripts.generate_consent_artifacts` produces one rendered one-pager per dialect and exits nonzero if any `consent_basis` in the validated data lacks protocol coverage; a unit test confirms a synthetic audio record with `consent_basis: explicit_verbal`, `voice_clone_ok: false` is accepted by `data.schema.validate` (§2) and is eligible for ASR training data assembly but is excluded when `train_tts.py` (§6) assembles its training set, with the exclusion count reported in `cards/dataset_cards/<dialect>.md` (e.g. "N of M consented audio hours excluded from TTS training on voice-cloning consent grounds") so a low TTS-eligible hour count is visible and attributable, not just a smaller-than-expected number with no explanation.

---

## 3. Active Learning / Data Prioritization Loop

**Problem this solves:** With 6 low-resource dialects and a fixed annotation budget, passive "validate whatever comes in" wastes annotator time on samples the model already handles well. This loop makes the model tell you what to collect/label next.

`active_learning/score_pool.py`:
1. Load current-checkpoint ASR model (Whisper/MMS fine-tune) and current MT model (IndicTrans2 fine-tune).
2. Run inference over `data/raw/<dialect>/` (unlabeled pool).
3. Compute per-sample uncertainty:
   - ASR: average token-level entropy from decoder logits, OR use `n`-best hypothesis disagreement (edit distance between top-3 beam outputs) as a proxy when logits aren't exposed by the model wrapper.
   - MT: same entropy/n-best-disagreement approach on decoder output.
4. Compute per-sample **novelty**: embedding distance (e.g., LaBSE or IndicBERT sentence embeddings) to the nearest neighbor already in `data/validated/<dialect>/`. Low similarity to existing validated data = high novelty. **Both sides of this comparison must be orthography-normalized before embedding, not raw text.** `data/validated/<dialect>/` is already normalized (Section 2.1 runs at validation time), but the unlabeled pool text scored here is pre-validation, raw-as-collected — comparing raw pool text against normalized validated text means two spelling variants of the same common word embed as if they were different tokens/strings, which understates their similarity to what's already validated and makes them look artificially novel to the sampler. `score_pool.py` must call `data.normalize_orthography`'s deterministic rule-application function (the same one Section 2.1 uses, imported as a library function — not a CLI shell-out, and not a second copy of the ruleset) on pool text in-memory before computing the embedding, without writing the normalized text back to `data/raw/` (raw stays untouched; only the embedding input is normalized). Text still flagged `orthography_review: true`-equivalent (i.e., not confidently resolved by deterministic rules) is embedded on its best-effort normalized form — normalization at this stage is a scoring aid, not a correctness gate, so it doesn't block scoring the way a failed validation would.
5. Priority score = `0.6 * normalized_uncertainty + 0.4 * normalized_novelty` (weights configurable in `pipeline.yaml`).

`active_learning/sampler.py`: takes top-K by priority score per dialect (K configurable, default 200/week/dialect), writes to `active_learning/annotation_queue.py`'s output queue (a simple JSONL + status field: `pending/in_review/validated/rejected`).

**Loop cadence:** run `score_pool.py` after every training checkpoint (i.e., every fine-tuning cycle), not continuously — document this explicitly so the agent doesn't build an always-on service where a batch job suffices.

**Cold start (round 0):** no fine-tuned checkpoint exists before the first training run, so `score_pool.py` must accept `--checkpoint base` as a valid value, meaning: use the untuned base ASR/MT models (Whisper/MMS, IndicTrans2 zero-shot) for uncertainty scoring, and skip the novelty term entirely for round 0 (there is no `data/validated/<dialect>/` yet to compute nearest-neighbor distance against — priority score for round 0 is uncertainty-only, documented as such in the output CSV's header). This is what unblocks Build Order step 2 from depending on step 5.

**Acceptance criteria:** running `python -m active_learning.score_pool --dialect bgr --checkpoint <path>` on a synthetic 50-sample pool produces a ranked CSV with uncertainty, novelty, and combined score columns; top-ranked samples are visibly different in character (longer/rarer/noisier) from bottom-ranked ones in a manual spot check; a test with two pool samples that are spelling variants of an already-validated sentence (differing only in a known orthography-variant word) confirms both score low novelty, not high — regression test for the raw-vs-normalized comparison bug.

---

## 4. Synthetic Data Augmentation

Three independent techniques, each its own module — do not conflate them:

**4.1 Back-translation for MT pairs** (`augmentation/back_translate.py`)
Hindi → dialect (via best available seed MT model or bootstrapped rules/dictionary) → Hindi round-trip. Keep only pairs where round-trip BLEU/chrF against the original Hindi exceeds a threshold (default chrF ≥ 0.5) — this is the filter that keeps synthetic pairs from polluting training with garbage. Tag all resulting records `source: synthetic_backtranslation` per the schema in Section 2, and cap synthetic:real ratio at 3:1 per dialect (configurable) so training doesn't drift toward synthetic artifacts.

**Not one-time — refresh on model improvement.** Round-0 back-translation has no dialect-specific MT model yet, so it necessarily runs through the same bootstrapped rules/dictionary/zero-shot path used to unblock the idiom bank in Section 5.3 — the resulting synthetic pairs are dictionary-quality at best. Treating this as a single upfront generation step (as the Section 9 budget table originally implied by listing it as "one-time") means that low-quality seed data keeps being trained on indefinitely even after a real fine-tuned dialect MT model exists and could generate much better back-translations. Instead: `back_translate.py` records which MT checkpoint (`production` pointer per Section 6.1, or `base` for round 0) generated each synthetic batch, tagged as a `generator_checkpoint` field on the synthetic record. Re-run back-translation generation after every checkpoint promotion event (Section 6.1) where the newly-promoted checkpoint differs from the one that generated the current synthetic pool for that dialect; the refreshed batch supersedes the prior one (old `generator_checkpoint: base` or stale-checkpoint batches are marked `superseded: true` and excluded from `data/splits/` assembly going forward, not deleted, so the augmentation report in the acceptance criteria below can still show the before/after quality delta). This mirrors the existing active-learning cadence ("run after every training checkpoint," Section 3) rather than inventing a second scheduling concept.

**Acceptance criteria (added):** `augmentation/report.py` additionally shows, per dialect, the `generator_checkpoint` distribution of the current (non-superseded) synthetic MT pool, and confirms zero non-superseded records still reference `generator_checkpoint: base` once at least one real dialect MT checkpoint has been promoted.

**4.2 TTS-bootstrapped ASR data** (`augmentation/tts_bootstrap.py`)
Take validated dialect text with no paired audio → synthesize audio via the dialect's fine-tuned TTS checkpoint (bootstrap: use closest-available Indic TTS model before a dialect-specific TTS exists) → feed back as (synthetic_audio, real_text) ASR training pairs. Explicitly cap this at a minority share of total ASR training hours (default ≤30%) and log the share in the model card — synthetic-audio-heavy ASR training silently overfits to TTS artifacts if unchecked.

**4.3 Audio perturbation** (`augmentation/audio_perturb.py`)
Speed (±10%), pitch shift (±2 semitones), and additive noise (SNR 10–20dB, using real-world noise samples — traffic, market, household) on real recordings only (never on TTS-synthesized audio, to avoid compounding artifacts). Use `audiomentations` or `torch-audiomentations`.

**Acceptance criteria:** for one dialect end-to-end, `data/synthetic/<dialect>/` contains records from all three techniques, each correctly tagged by source; a report (`augmentation/report.py`) prints real vs synthetic hour/pair counts and confirms the caps above are respected.

---

## 5. Dialect Boundary / Code-Switching Support

**5.1 Dialect-ID classifier** (`dialect_id/`)
Fine-tune a lightweight classifier (XLSR-53 or MMS encoder + linear head, NOT a full LLM) on `dialect` labels from validated data. Use as a pre-processing step in the serving API (Section 8) to auto-route audio/text to the right dialect-specific model when the user doesn't specify one, and to report **cross-dialect zero-shot transfer**: train on dialect A, eval WER/BLEU on dialects B–F without fine-tuning, produce a 6×6 transfer matrix (`eval/cross_dialect_transfer.py`). This transfer matrix is a direct, concrete answer to the "scalability across diverse regions" judging criterion — include it in the final report, not just the code, and it feeds the interactive heatmap in Section 11.

**5.2 Code-switching handling** (`codeswitch/`)
`tagger.py`: token-level language tagging (dialect/Hindi/English) on all collected text using a small tagger (fastText langid or a fine-tuned token classifier) — populates the `cs_spans` field from Section 2's schema, run automatically during validation, not as an afterthought.
`cs_eval_set_builder.py`: explicitly carve out a code-switched subset of dev/test splits (target ≥15% of each split) rather than filtering code-switched utterances out. Report WER/BLEU **separately** for monolingual vs code-switched subsets in all eval outputs — a single blended number hides exactly the failure mode real users will hit.

**5.3 Idiom & Proverb Bank** (`linguistic_artifacts/`)
This is the linguistic artifact that visibly demonstrates "preserve linguistic nuances" rather than claiming it in prose, and it's the intended centerpiece live-demo moment.

`linguistic_artifacts/schema/idiom_record.schema.json`:
```json
{
  "id": "string (uuid)",
  "dialect": "enum[mwr,mtr,dhd,hdt,mwt,bgr]",
  "idiom_dialect": "string (the proverb/idiom as spoken, normalized)",
  "idiom_dialect_raw": "string (as-collected spelling, pre-normalization)",
  "orthography_review": "boolean",
  "literal_gloss": "string (word-for-word translation, deliberately awkward/wrong-sounding)",
  "intended_meaning_hindi": "string",
  "intended_meaning_english": "string",
  "register": "enum[formal,informal,elder_speech,proverb,blessing,idiom]",
  "usage_context": "string (when/where this is said)",
  "collected_from": "string (field source, anonymized)",
  "consent_basis": "enum[explicit_written,explicit_verbal,public_domain,synthetic]",
  "public_release_ok": "boolean"
}
```
(`consent_basis` enum is identical to the main schema in Section 2 — keep these in one shared schema fragment/enum definition, not two copies that can drift apart.)

**Consent and orthography parity with the main schema (v10).** This schema was previously exempt from two rules that apply everywhere else in this spec, which is a gap, not a deliberate simplification — the idiom bank is the designated centerpiece demo/judging artifact, and centerpiece artifacts are exactly what eventually gets published or quoted outside the repo.
- `idiom_dialect`/`idiom_dialect_raw`/`orthography_review` follow §2.1's rule exactly: `collect_idioms.py` writes the as-collected string to `idiom_dialect_raw` and the normalized form to `idiom_dialect`, flags `orthography_review: true` when the deterministic ruleset doesn't confidently resolve it, and this is the same `data.normalize_orthography` call the main text pipeline uses — not a second implementation. §2.1's acceptance criteria (idempotence, variant-collapse test) apply to the idiom intake path too, not just the bulk text pipeline it was originally phrased around.
- `public_release_ok` follows §2's rule exactly: default `false`; `true` only when the consent script (§2.5) captured a separate, explicit public-release opt-in; `explicit_written`/`explicit_verbal` `consent_basis` alone does not imply it; `public_domain`-sourced (bootstrap-seeded) entries default `true`. This field currently has no consumer — §8.5's `publish_filter.py` gates the benchmark test set, not the idiom bank, and the idiom bank is not today published as a downloadable dataset, only demoed live (§11) — but the field is added now, at first-write time, for the same reason §2's fields are: consent scope is only safe to capture at collection time, not retrofitted onto already-collected entries once someone decides to publish the bank. If a future pass adds a public idiom-bank release, it reads this field rather than reopening every prior entry to ask a question that should already have an answer on file.

**Bootstrap fallback:** live field collection of ≥100 entries/dialect is a human-fieldwork dependency with real lead time, not something buildable by a coding agent alone — do not let it block Build Order step 4. Seed each dialect's bank first from published proverb/idiom collections and dictionaries (tagged `consent_basis: public_domain`, `source` field noting the publication), enough to unblock `idiom_mt_eval.py` and the demo tab end-to-end. Field-collected entries (`explicit_written`/`explicit_verbal`) then backfill and eventually dominate the bank; track the two counts separately in the dataset card so "600 idioms collected" doesn't silently mean "600 scraped from a book."

`collect_idioms.py`: field-collection intake tool (CLI form) that enforces the schema above — target a few hundred entries per dialect over time, sourced through the same consented field-collection channel as Section 2 for the `explicit_*` portion of the bank.

`idiom_mt_eval.py`: a dedicated MT evaluation slice — run the trained MT model on `idiom_dialect` inputs and score whether the output matches `intended_meaning_hindi`/`intended_meaning_english` versus a literal/garbled translation. Match criterion: embedding cosine similarity (same LaBSE/IndicBERT sentence encoder used for novelty scoring in Section 3, kept consistent so the repo doesn't carry two sentence-embedding stacks) between MT output and `intended_meaning_*` ≥ 0.75 counts as a semantic match; below that, classify as literal/garbled. Human-spot-check 20% of the held-out idiom set per dialect against this automatic label and report agreement rate in the model card — if agreement falls below 80%, flag the 0.75 threshold as miscalibrated for that dialect in `LIMITATIONS.md` rather than silently trusting the automatic score. Report this **separately** from the blended MT BLEU/chrF numbers in the model card (Section 7.2), the same way code-switched performance is reported separately in Section 5.2 — figurative-language failure is a distinct, and currently invisible, failure mode.

**Acceptance criteria:** `eval/cross_dialect_transfer.py --dialect mwr` produces a **separate** 6×6 matrix per task (ASR-WER and MT-BLEU/chrF are not blended into one number — "the worst dialect pair" always means worst-per-task, referenced that way everywhere downstream including Section 7.5); test split manifests for every dialect report `%code_switched` and separate WER for that subset; `linguistic_artifacts/idiom_bank/<dialect>.jsonl` contains ≥100 entries per dialect before Stage 4 (bootstrap + field-collected combined, per the fallback above), of which ≥30 are field-collected (`explicit_written`/`explicit_verbal`) and not `INSUFFICIENT_DATA` by the repo-wide threshold in Section 0; and `python -m linguistic_artifacts.idiom_mt_eval --dialect <id>` produces a report distinguishing correct figurative translation from literal mistranslation across the **full held-out idiom set per dialect** (not a single hand-picked example), with an aggregate figurative-accuracy percentage per dialect. **(v10 addition)** running `collect_idioms.py` twice on the same raw entry is idempotent on `idiom_dialect` (mirrors §2.1's normalization test); a synthetic entry with a known variant spelling confirms `idiom_dialect` holds the canonical form while `idiom_dialect_raw` retains the original; a unit test confirms newly collected idiom entries default `public_release_ok: false` unless the consent script recorded an explicit public-release opt-in, and bootstrap-seeded (`public_domain`) entries default `true`.

---

## 6. Model Training

Use parameter-efficient fine-tuning (LoRA/QLoRA) on open base models — full fine-tuning is not needed at this data scale and blows the compute budget in Section 9.

| Task | Base model | Method | Output |
|---|---|---|---|
| ASR | `openai/whisper-large-v3` or `facebook/mms-1b-all` | LoRA fine-tune per dialect | `training/train_asr.py --dialect <id>` |
| MT | `ai4bharat/indictrans2` (dialect↔hin↔eng) | LoRA fine-tune per dialect pair | `training/train_mt.py --dialect <id> --pivot hin` |
| TTS | `facebook/mms-tts` (production default) or `coqui/XTTS-v2` (hackathon-demo only) | Fine-tune/voice-adapt per dialect | `training/train_tts.py --dialect <id>` |
| Dialect-ID | `facebook/mms-1b` encoder + linear head | Full fine-tune of small head only | `dialect_id/train.py` |

Every training script must: (a) load data exclusively from `data/splits/<dialect>/train.jsonl`, (b) log to the experiment tracker (Section 7) automatically, (c) run the active-learning scorer (Section 3) on the updated checkpoint at the end of each run, (d) save checkpoints under `checkpoints/<task>/<dialect>/<run_id>/`, (e) run the checkpoint promotion gate (Section 6.1) before the run is considered complete.

**Voice-cloning consent gate on `train_tts.py`.** `train_tts.py` must filter its training set to audio records where `voice_clone_ok: true` (§2.5, §2 schema) before assembling per-dialect training data — this is a separate filter from, and applied in addition to, the general `consent_basis`/split checks every training script already does. If the `voice_clone_ok: true` subset for a dialect falls under the Section 0 `INSUFFICIENT_DATA` threshold, `train_tts.py` must refuse to train that dialect's TTS model and emit `INSUFFICIENT_DATA` rather than silently training on too few consenting speakers (which would make the resulting voice model disproportionately reproduce one or two individuals' voices) or silently falling back to the full (non-gated) consented pool.

**TTS licensing note (`LICENSES.md`, repo root):** `coqui/XTTS-v2` is released under Coqui's CPML (Coqui Public Model License), which restricts commercial/production use without a separate commercial license — fine for a hackathon demo, not fine if BHASHINI intends to deploy the resulting TTS models past the hackathon. `facebook/mms-tts` is the permissively-licensed default for anything meant to outlive the demo; `train_tts.py --backend xtts` must print a one-line CPML warning to stderr on startup (not just a code comment) so the constraint is visible at run time, not just in documentation someone might not read. `LICENSES.md` tracks the license and commercial-use status of every base model used in the repo (Whisper, MMS, IndicTrans2, XTTS-v2), so this check doesn't have to be re-derived per model later.

**Acceptance criteria:** one full LoRA fine-tune run completes for one dialect on one task without manual intervention, produces a checkpoint, and the checkpoint is loadable by the corresponding `eval/` script.

---

### 6.1 Checkpoint Promotion Gate

**Problem this solves:** Section 6 saves a checkpoint per run under `checkpoints/<task>/<dialect>/<run_id>/`, but nothing so far says which checkpoint the live `/asr /mt /tts` API in Section 8 actually serves. Without an explicit gate, the natural default (serve "the latest checkpoint") means a training run that regresses — bad hyperparameters, a data-quality dip from a bad active-learning batch, a bug — silently replaces a working model in production/demo with a worse one, and nobody finds out until Stage 4 live trials.

`training/promote_checkpoint.py`, run automatically at the end of every training script (Section 6's step (c), alongside the active-learning scorer):
1. Evaluate the new checkpoint on `data/splits/<dialect>/dev_promotion.jsonl` — **never the full `dev.jsonl`** (which also contains `dev_canary` records; reading it here would silently include canary data in every promotion decision, defeating the point of the dev-set-rotation fix below) **and never `test.jsonl`** (that stays reserved for the benchmark/final report per Section 2.2) — using the same metric the task reports elsewhere (WER for ASR, BLEU/chrF for MT, MOS-proxy or a fixed objective metric for TTS).
2. Compare against the metric recorded for whatever checkpoint `checkpoints/<task>/<dialect>/production` currently points to (a symlink/pointer file, not a copy of the weights).
3. **Metric direction is declared per configured metric, not assumed per task.** `pipeline.yaml` declares a `direction: lower_is_better` / `direction: higher_is_better` field keyed to the *specific metric name* each task is configured to report (e.g. `wer: lower_is_better`, `bleu: higher_is_better`, `chrf: higher_is_better`), not keyed to the task itself. This matters because TTS's evaluation metric (Section 6: "MOS-proxy or a fixed objective metric") is a team choice, not fixed by this spec — a MOS-proxy predictor is higher-is-better, but a distance-style objective metric such as MCD (mel-cepstral distortion) is lower-is-better, and assuming TTS is always higher-is-better would silently invert the promotion decision for a team that picks the latter. `promote_checkpoint.py` reads the metric's own declared direction rather than branching on task name — a metric with no declared direction is a build-time config error, not a silent default. "Better" and "regression" throughout this section mean relative to the declared direction of whichever metric is actually configured.
4. **Tolerance is scaled to dev-set size, not a flat point value.** A flat tolerance (e.g. "no worse than 0.5 WER-points") ignores that WER/BLEU on a small low-resource dev set swings by more than that from sampling noise alone — the gate can reject a genuinely better checkpoint, or promote a genuinely worse one, purely on noise. Instead, compute a per-comparison tolerance band from the dev-set size: bootstrap-resample the dev set (default 1,000 resamples) to estimate a 90% confidence interval around the new checkpoint's metric, and promote only if the new checkpoint's point estimate is at least as good as the current production metric *or* the two checkpoints' confidence intervals overlap (i.e. the observed difference isn't distinguishable from noise at this dev-set size — treat "no significant regression" as passing, not just "strictly better"). **This resamples already-computed per-utterance scores, not inference itself:** run the model over the dev set once to get per-utterance edit-distance/BLEU components, then bootstrap-resample *those stored per-utterance values* 1,000 times to build the metric's sampling distribution — the model is never re-run per resample. This is a CPU-bound, sub-second operation regardless of resample count; the §9 budget line for this step accounts for one inference pass over a small dev set, not 1,000 of them, and an implementation that re-runs inference per bootstrap sample is a bug, not a valid interpretation of "resample." `pipeline.yaml`'s `promotion.tolerance_points` (default 0.5 WER-pts / 1 BLEU-pt from v5) is retained as a hard floor only for dialects whose dev set is large enough that the bootstrap CI half-width falls below it — below that dev-set size, the bootstrap CI is the binding constraint, not the flat number. This means a low-resource dialect's gate is, correctly, more forgiving of a large-looking point-difference than a high-resource dialect's, because that difference is less trustworthy on less data.
5. **Orthography ruleset version is recorded and checked.** Every dev-set evaluation logs the `configs/orthography/<dialect>.yaml` version (Section 2.1; version field added per backlog item 3) that was current when dev/test text was last normalized. If the new checkpoint's eval ran under a different orthography version than the production checkpoint's last recorded eval, `promote_checkpoint.py` flags the comparison `orthography_version_mismatch: true` in the promotion log and still runs the comparison (an orthography change alone shouldn't block promotion indefinitely) but this flag must be surfaced in `LIMITATIONS.md` (Section 7.5) rather than silently treated as a normal WER delta — a WER change caused by re-normalized reference text is not the same signal as a WER change caused by a better model.
6. Promote (repoint `production` to the new `run_id`) only if the new checkpoint passes the direction-aware, dev-size-scaled check in step 4; otherwise leave `production` untouched and write the comparison (old vs. new metric, both checkpoints' CIs, `orthography_version_mismatch`, promoted: true/false) to `checkpoints/<task>/<dialect>/promotion_log.jsonl`.
7. First checkpoint ever trained for a task/dialect always promotes (nothing to compare against) but this is logged explicitly as `promoted: true, reason: first_checkpoint`, not silently skipped.

**Dev-set rotation / canary slice.** Reusing the same fixed `dev.jsonl` for every promotion decision across many active-learning cycles is repeated hyperparameter/architecture selection against a static set — the classic multiple-comparisons overfitting risk, where a checkpoint can eventually "win" promotion by fitting dev-set quirks rather than generalizing. Two changes:
  - **Schema and materialization.** At the same time `assign_split.py` (Section 2.2) assigns `train`/`dev`/`test` into the record's `split` field, any record assigned `split: dev` also receives a second field, `dev_subsplit` (enum `promotion` / `canary`, default 70%/30% deterministic-hashed the same way `split` itself is), so `dev_promotion.jsonl` and `dev_canary.jsonl` are materialized views filtered on `split: dev` + `dev_subsplit`, exactly as `data/splits/<dialect>/{train,dev,test}.jsonl` are materialized views on `split` (Section 2.2) — `dev_subsplit` is a new field, not new values added to `split`, so nothing that already filters on `split` needs to change. `dev_canary.jsonl` inherits the same read restriction `test.jsonl` gets (Section 2.2's acceptance criteria): only `eval/` scripts may read it; `training/train_*.py`, all three `augmentation/` scripts (Section 2.2's augmentation rule — added in v9 after the restriction was found to only name `test.jsonl`, not the newer `dev_canary.jsonl`), and `promote_checkpoint.py`'s promotion decision itself are all blocked from it by the same path check, enforced identically to the `test.jsonl` rule.
  - **Audit ownership.** `promote_checkpoint.py` — not a separate periodic invocation of `eval/report.py` — owns the canary audit directly: it maintains a per-task/dialect promotion counter in `checkpoints/<task>/<dialect>/promotion_log.jsonl` (already logging one entry per promotion event, so the counter is just "count the rows") and, on every 10th promotion event, additionally evaluates the current `production` checkpoint on `dev_canary.jsonl` and appends the result to a dedicated `checkpoints/<task>/<dialect>/canary_audit.jsonl` (`dev_promotion` metric, `dev_canary` metric, gap, promotion-event count at time of audit). `eval/report.py` (Section 12) then reads and surfaces `canary_audit.jsonl` as one more linked artifact in the consolidated report, the same way it links `LIMITATIONS.md` and `benchmark/leaderboard.md` rather than duplicating their content — it does not generate the audit itself. If a dialect's `dev_canary.jsonl` falls under the Section 0 `INSUFFICIENT_DATA` threshold (20 samples) — likely for low-resource dialects, since it's 30% of an already-small 10%-of-validated-pool `dev` set — the canary audit records `INSUFFICIENT_DATA` for that dialect/task instead of computing a gap, rather than reporting a comparison too noisy to mean anything.

`dev_canary` is never used to make a promotion decision, only to audit the gate itself.

`serving/api/main.py` (Section 8) loads models exclusively via the `production` pointer per task/dialect — never a hardcoded run_id or "most recent by mtime," so a human can also force a rollback by manually repointing the symlink, and that action is auditable via the same promotion log.

**Acceptance criteria:** training two sequential checkpoints for one dialect/task, where the second is synthetically made worse (e.g. fewer training steps), confirms `production` still points at the first checkpoint and `promotion_log.jsonl` records the rejected promotion with both metric values and their confidence intervals; a test with a synthetic small dev set (e.g. 30 utterances) confirms a checkpoint whose point-estimate WER is 0.4 points worse than production but whose CI overlaps production's CI still promotes, while the same 0.4-point gap on a synthetic large dev set (e.g. 2,000 utterances, tight CI) does not; a test with a TTS task configured to a `direction: lower_is_better` objective metric (e.g. a synthetic MCD-style score) confirms a checkpoint with a *higher* value than production is correctly rejected — regression test confirming direction is read from the configured metric, not assumed from the task; a test that evaluates two checkpoints under different orthography config versions confirms `orthography_version_mismatch: true` is set and appears in `LIMITATIONS.md`; `GET /models` on the serving API reports the `run_id` currently live per task/dialect, sourced from the `production` pointer, not a config file that could drift from what's actually loaded; a unit test confirms `training/train_*.py`, all three `augmentation/` scripts, and the promotion decision path all raise (rather than silently reading zero rows) if pointed at `dev_canary.jsonl`, mirroring the `test.jsonl` restriction; a test with a synthetic `dev_canary` record injected with a metric value that would fail the gate confirms the promotion decision is unaffected by it — i.e. `promote_checkpoint.py`'s eval call resolves to `dev_promotion.jsonl` specifically, not the undivided `dev.jsonl`, closing the step-1/dev-set-rotation inconsistency from v7; after 10 synthetic promotion events, `canary_audit.jsonl` contains one `dev_promotion` vs `dev_canary` comparison row, generated by `promote_checkpoint.py` itself (not `eval/report.py`), and `eval/report.py` links it into the consolidated report; a synthetic dialect with a `dev_canary.jsonl` under 20 samples confirms the canary audit records `INSUFFICIENT_DATA` rather than a computed gap; a test confirms the bootstrap-CI step in the acceptance criterion above completes in the time of one inference pass plus negligible resampling overhead — not 1,000× inference time — verifying resampling operates on stored per-utterance scores.

---

## 7. Experiment Tracking & Model/Dataset Cards

**7.1 Tracking** (`training/track_experiment.py`)
Wrap all training scripts with MLflow (self-hosted, no external account needed) as the default; support W&B as an opt-in via config flag for teams that prefer it. Log: hyperparameters, dataset version/hash, WER/BLEU/MOS per checkpoint, GPU-hours consumed. This is what makes Stage 2/3 technical review auditable — reviewers should be able to see the full run history, not just the final claimed number.

**7.2 Model cards** (`eval/model_card_gen.py` → `cards/model_cards/<model>.md`)
Auto-generate from tracked run metadata + a template covering: intended use, training data summary (dialects, hours, source breakdown), known limitations (explicitly list code-switching performance gap from Section 5.2, idiom/figurative-language performance from Section 5.3, cross-dialect transfer numbers from Section 5.1, and the gender-breakout WER/MOS gap from Section 7.4), evaluation metrics, out-of-scope uses.

**7.3 Dataset cards** (`cards/dataset_cards/<dialect>.md`)
Per HuggingFace dataset card conventions: dialect coverage/regions, collection methodology, consent basis breakdown (from schema field, cross-linked to `docs/CONSENT_PROTOCOL.md`), validator count, known biases (e.g., speaker age/gender skew if present), synthetic-data share (from Section 4 report).

**7.4 Generational Drift Report** (`eval/generational_drift.py`)
Turns the `speaker_age_cohort` schema field (Section 2) from a data-quality dimension into the preservation-urgency narrative that is BHASHINI's actual stated mission, not just a WER table.

For each dialect, compute and report, broken out by age cohort (`under18`, `18-30`, `31-50`, `51-70`, `70plus`):
- Code-switching rate (`is_code_switched` share) per cohort — the concrete metric behind a finding like "younger urban speakers code-switch at N× the rate of rural elders 51+"
- Lexical/idiom retention: share of collected utterances per cohort that use entries also present in the Section 5.3 idiom bank, as a rough proxy for how much traditional register survives in each generation — **compute this against `explicit_written`/`explicit_verbal` (field-collected) idiom entries only**, never the `public_domain` bootstrap-seeded entries. A bootstrap-seeded proverb book doesn't tell you what any living cohort actually says, so mixing it in would fabricate a generational signal from literary source material. If a dialect has fewer than the Section 0 `INSUFFICIENT_DATA` threshold of field-collected idiom entries at analysis time, report retention for that dialect as `INSUFFICIENT_DATA` rather than computing it against the bootstrap set
- Regional distribution per cohort, using the `settlement_type` schema field (Section 2) — required, not "where available" — cross-tabulated against age cohort so an urban/rural effect isn't misattributed to age, or vice versa
- **(v10, promoted from §13 backlog item 2) WER/MOS broken out by `speaker_gender`** (audio schema, Section 2), reusing the same cross-tab machinery as the `settlement_type` breakout above rather than a separate script — a gender-skewed error gap is a common consequence of imbalanced training data and was previously collected but never analyzed. Report per-dialect, per-task (ASR WER, TTS MOS); apply the Section 0 `INSUFFICIENT_DATA` threshold per gender/dialect/task cell exactly as for age cohorts. This is descriptive only — it does not gate promotion or trigger rebalancing on its own — but must appear in the model card (Section 7.2) `known limitations` alongside code-switching and idiom-figurative gaps, not just the dataset card, since it's a model-quality finding as much as a data-composition one.

Output: `cards/dataset_cards/<dialect>.md` gets a "Generational Drift" subsection. The one plain-language finding sentence per dialect is template-generated, not free-form: `"{higher-rate cohort} speakers show {code_switch_rate_a}% code-switching vs {code_switch_rate_b}% for {lower-rate cohort} ({settlement_type} controlled: {yes/no})"` — filling in measured numbers only, no generated commentary beyond that template, which is what "don't editorialize beyond what the numbers support" means operationally. Any cohort with fewer than the Section 0 `INSUFFICIENT_DATA` threshold (20 utterances) is reported as `INSUFFICIENT_DATA` and excluded from the finding sentence rather than averaged in.

**7.5 Limitations Report** (`LIMITATIONS.md`)
Auto-generated, not hand-written, and it must not be softened — panels doing Stage 2/3 technical review trust self-reported weaknesses more than blanket accuracy claims that won't survive Stage 4 live trials.

`eval/limitations_gen.py` pulls directly from other sections' outputs and writes `LIMITATIONS.md` at the repo root:
- Per-dialect WER/BLEU/MOS, with anything above the success-matrix target (WER >10%, MOS <4.0) called out explicitly, not buried in an appendix table
- Which dialect currently has the least validated data (hours/utterance count), pulled from `cards/dataset_cards/`
- Code-switched-subset WER/BLEU gap vs. monolingual (Section 5.2), stated as a delta, e.g. "+N pts WER on code-switched dev"
- Idiom/figurative-language MT accuracy vs. blended MT accuracy (Section 5.3), same delta framing
- Cross-dialect zero-shot transfer floor: the worst-performing dialect pair in the Section 5.1 matrix, named explicitly
- Any `INSUFFICIENT_DATA` markers surfaced anywhere else in the pipeline (consent protocol coverage, generational drift cohorts, idiom bank counts)

**Acceptance criteria:** running the full pipeline for one dialect produces a populated model card and dataset card with no placeholder/lorem-ipsum fields remaining, and the dataset card's Generational Drift subsection reports at least one non-`INSUFFICIENT_DATA` cohort comparison using the fixed finding-sentence template above (free-form prose in that subsection is a failed build, not a style choice); `python -m eval.limitations_gen` produces `LIMITATIONS.md` with all six bullet categories above populated from real run data (or explicit `INSUFFICIENT_DATA` markers per the Section 0 threshold, never fabricated numbers), and the cross-dialect transfer floor bullet names both the task (ASR or MT) and the dialect pair, never a blended cross-task number.

---

## 8. BHASHINI Ecosystem Interoperability

Since this is a BHASHINI-led initiative, the deliverable should plug into BHASHINI's own pipeline contract rather than exist only as a standalone demo — this is what "scalability... across diverse regions" means at the platform level, not just the model level.

`serving/api/bhashini_adapter.py`: wraps the native `/asr`, `/mt`, `/tts` FastAPI endpoints (see below) in the ULCA (Universal Language Contribution API) request/response format used by Bhashini's NMT/ASR/TTS pipeline spec — i.e., accept `pipelineTasks` array requests and return `pipelineResponse` in the same shape Bhashini's own model endpoints use, so this can be registered as a Bhashini pipeline component with no adapter rewrite needed at the platform side.

**ULCA schema verification (build-time step, not a silent assumption):** before implementing `bhashini_adapter.py`, the agent must fetch BHASHINI's currently published ULCA API reference (Meity/BHASHINI's public API docs and OpenAPI spec, e.g. via bhashini.gov.in's developer resources) and pin the resolved schema version and fetch date in a comment header at the top of `bhashini_adapter.py`. If the docs are unreachable at build time, the agent must fall back to the most recent version it can verify from a cached/vendored copy, flag this explicitly in `README.md` under a "Known Assumptions" heading, and add a `scripts/verify_ulca_schema.py` stub that re-checks the live spec on demand — this is a checked, documented decision either way, never a hardcoded guess left unlabeled.

`serving/api/main.py` — native FastAPI endpoints:
- `POST /asr` — audio in, dialect-tagged transcript out (auto dialect-ID if `dialect` param omitted)
- `POST /mt` — text in, source/target dialect+pivot out
- `POST /tts` — text in, dialect-voiced audio out
- `POST /dialect-id` — audio or text in, dialect probability distribution out
- `GET /health`, `GET /models` (lists loaded checkpoints per dialect/task)

**Auto-routing confidence threshold.** `/dialect-id` returns a full probability distribution across all 6 dialects, not just a top-1 label — that distribution is what makes the threshold below possible. When `/asr` or `/mt` is called without an explicit `dialect` param, the auto-routing path must not silently forward the input to whatever model got the highest score: on ambiguous or code-switched input (Section 5.2 already establishes code-switching is common, not an edge case) a wrong top-1 guess routes audio to the wrong dialect's ASR/MT model with no signal to the caller that anything was uncertain.

**Per-dialect-pair thresholds, not one global default.** A single global `confidence_threshold`/`margin_threshold` pair is miscalibrated in both directions at once: Marwari and Mewari are more mutually intelligible than, say, Bagri and Hadoti, so one threshold is too loose for close pairs (routes confidently on genuinely ambiguous Marwari/Mewari input) and too tight for distant pairs (unnecessarily flags Bagri/Hadoti input as ambiguous when the model is actually confident and correct). `pipeline.yaml`'s `dialect_id.confidence_threshold`/`dialect_id.margin_threshold` become a 6×6 matrix (`dialect_id.pairwise_thresholds`, keyed by the two highest-probability dialects in a given inference), seeded from the single global default (`0.6`/`0.15`) at first build and then recalibrated per pair using the Section 5.1 cross-dialect confusion data once it exists (a pair with high zero-shot cross-dialect transfer/confusion in the Section 5.1 matrix is exactly the pair that needs a stricter margin threshold). Until Section 5.1's matrix has enough data to recalibrate a given pair, that pair falls back to the global default — this is a refinement of the threshold, not a hard dependency that blocks Section 8 on Section 5.1 finishing first.

Routing rule, applied using the relevant pairwise threshold for the top-2 candidate dialects:
- Route automatically only if top-1 probability ≥ that pair's `confidence_threshold` **and** the gap to the second-highest probability ≥ that pair's `margin_threshold` (a high top-1 with a close second is exactly the code-switched/border-dialect case that shouldn't be silently resolved either way).
- Below threshold: return HTTP 300-style multi-choice behavior — response body includes `dialect_ambiguous: true`, the top-k candidate dialects with their probabilities, and does **not** run `/asr` or `/mt` inference against a guessed model. The caller (demo app, IVR channel per Section 10, or BHASHINI platform integration) is expected to either prompt the user to disambiguate or explicitly pass a `dialect` param to force routing.
- Every low-confidence auto-routing event is logged to `active_learning/annotation_queue.py`'s pool (Section 3) — an ambiguous real-world sample is exactly the kind of high-uncertainty data point the active-learning loop should prioritize for review, so this failure mode feeds back into the system instead of just being swallowed.

**Rate-limit the ambiguous-routing → annotation-queue path specifically.** Section 8's general per-key rate limiting (below) bounds total request volume, but doesn't stop a caller from staying under that limit while deliberately sending ambiguous/noisy audio to cheaply flood the annotation queue with junk — the queue has no cost signal distinguishing "genuinely ambiguous real-world input" from "adversarial input crafted to look ambiguous." Add a second, tighter rate limit specifically on ambiguous-event queue writes per API key (`pipeline.yaml`: `dialect_id.ambiguous_queue_rate_limit`, default 20/hour/key, independent of and stricter than the general per-key request limit), and once a key exceeds it, continue serving `dialect_ambiguous` responses to that key normally (don't break the caller-facing behavior) but stop writing further events from that key to the annotation queue until the window resets — logged as `queue_writes_suppressed: true` in the API's own metrics, not silently dropped without a trace.

**Content moderation gate on `/tts`.** `/tts` synthesizes and speaks back arbitrary caller-submitted text through a BHASHINI-branded government service, on both the web API (Section 8) and the IVR channel (Section 10) — with no filter, an adversarial caller can make the system say anything in a dialect voice, which is a reputational and safety risk distinct from (and not addressed by) the per-key rate limiting below, since rate limiting bounds *volume*, not *content*. `serving/api/content_filter.py` runs before any text reaches `train_tts.py`'s served model:
- A lightweight open-source abuse/hate-speech/profanity classifier (e.g. a small multilingual moderation model, or at minimum a maintained keyword/pattern list covering Hindi/English/the 6 dialects where terms are known) scores input text; above a configured threshold (`pipeline.yaml: content_filter.threshold`), `/tts` returns a `content_blocked: true` response with no audio generated, rather than silently synthesizing the input or silently substituting different text.
- This gate is intentionally conservative and low-precision-tolerant for a hackathon deliverable — false positives (blocking benign text) are an acceptable cost; false negatives (synthesizing unsafe content) are not. Document this asymmetry explicitly in `README.md` rather than tuning the threshold to minimize false positives.
- Every `content_blocked` event is logged (input hash, not raw text, to `serving/api/moderation_log.jsonl`) for later review, separate from the active-learning annotation queue (Section 3) — a blocked-content event is a moderation signal, not a data-collection opportunity, and must not be conflated with the ambiguous-routing queue from Section 8's dialect-ID discussion.
- This gate does **not** apply to `/asr` or `/mt` input (transcribing or translating what someone said is not the same act as speaking it back aloud in a synthesized voice) — scope it to `/tts` specifically, not repo-wide, so it doesn't become a censorship layer on transcription/translation of real speech.

Minimal auth: require an `X-API-Key` header on `/asr`, `/mt`, `/tts`, `/dialect-id` (checked against keys in `.env`, not hardcoded), with `/health` left open for uptime checks. Add basic per-key rate limiting (`slowapi` or equivalent), default `pipeline.yaml: api.rate_limit` = 100 requests/hour/key across all four endpoints combined (configurable) — a concrete starting number rather than "generous defaults" left unspecified, so the ambiguous-routing queue limit above (20/hour/key) is checkably stricter than it rather than stricter than an undefined quantity. This is a hackathon deliverable meant to plug into BHASHINI's own gateway later, not a production auth system, but shipping zero auth on an endpoint that's explicitly positioned for platform integration (Section 8's stated goal) undersells the "scalability" judging criterion and invites accidental abuse of a public demo URL.

**Acceptance criteria:** `curl` against `/asr`, `/mt`, `/tts` returns valid JSON for a sample input per dialect with a valid API key, and `401` without one; `bhashini_adapter.py` round-trips a sample ULCA-format request to the native endpoint and back into ULCA-format response without field loss; the pinned schema version/date comment and `README.md` "Known Assumptions" section both exist; a synthetic dialect-ID distribution engineered to be below threshold (e.g. two dialects both near 0.5) confirms `/asr` called without an explicit `dialect` param returns `dialect_ambiguous: true` and top-k candidates instead of running inference, and the same event appears in the active-learning annotation queue; a test with two different pairwise threshold overrides configured for two different dialect pairs confirms the same top-1/margin values route automatically for one pair and return `dialect_ambiguous` for the other, sourced from `dialect_id.pairwise_thresholds` rather than one global constant; a test sending >20 ambiguous requests in an hour from one API key confirms the 21st+ still receives a correct `dialect_ambiguous` response but is not written to the annotation queue, with `queue_writes_suppressed: true` recorded; a test sending known-abusive text to `/tts` confirms `content_blocked: true` is returned with no audio generated and an entry (input hash only, not raw text) is written to `moderation_log.jsonl` rather than the active-learning annotation queue; a test confirms `/asr` and `/mt` are unaffected by the content filter (it is wired into `/tts` only).

---

### 8.5 Public Benchmark Leaderboard

Reframes the required "benchmark existing models" objective (hackathon brief, Objective 1) as infrastructure for the field rather than a private baseline table buried in an internal report — near-zero extra cost since these numbers are already computed as training baselines, but publishing them is what earns "collaboration between AI researchers, linguists, and developers" credit.

**Consent gate (checked before anything else in this section):** the test split used here gets published (`benchmark/dataset_card.md`, "reproducible by others"). Before building `run_baselines.py`, add a filter step that excludes any record with `public_release_ok: false` from the published benchmark test set specifically — those records may still be used for the model's own internal eval/training, just not shipped in the public leaderboard artifact. `benchmark/dataset_card.md` must state the count of records excluded on this basis per dialect, so a dialect with mostly non-public-release consent shows a small-but-honest public test set rather than silently substituting internal-only data into a "public" artifact.

**Field-stripping and re-identification gate (checked before publish, in addition to the consent gate above).** `public_release_ok: true` clears a record for public release; it does not by itself make the *published fields* safe. `speaker_id`, `region`, and `settlement_type` together — even with no name attached — can re-identify an individual speaker in a dialect community with a small population of contributors, especially once `speaker_age_cohort` is added into the same row (Section 7.4 needs that field for internal analysis, but it must not travel into the public artifact at the same granularity). `benchmark/publish_filter.py`:
- Drops `speaker_id` entirely from any publicly released row (replace with a non-reversible per-dialect ordinal like `spk_003` scoped only to the published file, so duplicate-speaker patterns are still analyzable without being traceable back to the internal `speaker_id`).
- Generalizes `region` from the specific district (per `configs/dialects.yaml`'s `regions` list) to dialect-level only in the published file; the district-level field stays in internal-only data.
- Applies a k-anonymity check on the combination of whatever fields *are* published (dialect, `settlement_type`, coarse age-cohort bucket if included at all): any combination of published quasi-identifier values shared by fewer than `k=5` distinct speakers in that dialect's public test set gets that row's `settlement_type`/age-cohort fields further generalized or suppressed (e.g. collapse `settlement_type` to `unknown`) until the k-threshold is met, rather than excluding the utterance itself — the audio/text is still useful for the benchmark, only the potentially-identifying metadata is constrained.
- `k=5` and the quasi-identifier field list are both defined in `pipeline.yaml`, not hardcoded, so the threshold is a documented, auditable decision.

**Acceptance criteria (added):** `benchmark/dataset_card.md` states, per dialect, how many rows had `settlement_type`/age-cohort generalized or suppressed by the k-anonymity check; a unit test with a synthetic dialect where 4 speakers share `(rural, 51-70)` confirms that combination gets suppressed to `k`-safe, and one with 6 speakers sharing a combination confirms it passes through unchanged; `speaker_id` and district-level `region` never appear as literal strings anywhere under `benchmark/` (grep-checked in CI, not just documented as a rule).

**Scope note — this gate only covers the public leaderboard path.** `publish_filter.py`'s field-stripping and k-anonymity check runs only on data flowing into `benchmark/`. It does not constrain re-identification risk if BHASHINI shares the internal `data/validated/` pool wholesale with a partner institution, academic collaborator, or other researcher outside the public-leaderboard path — a realistic scenario for a government-led initiative, and one where `speaker_id`/full `region`/`settlement_type`/age-cohort are all still present at full granularity. This is explicitly out of scope for this build (no acceptance criteria attached), not an oversight: it needs a data-sharing policy decision from BHASHINI (what a "partner-institution export" is allowed to contain, whether it goes through a lighter version of `publish_filter.py` or a separate agreement-based release process) before a coding agent can implement it correctly. Logged as backlog item 9 (Section 13) rather than silently left unmentioned.

`benchmark/run_baselines.py`: runs zero-shot inference on the `public_release_ok: true` subset of the Section 2 test splits (all 6 dialects) for:
- ASR: `openai/whisper-large-v3`, `facebook/mms-1b-all`
- MT: `ai4bharat/indictrans2`, and a general-purpose LLM baseline (e.g. a GPT-class model via API, clearly labeled as the one path in this repo that touches a paid API — feature-flag it the same way IVR is flagged in Section 10, off by default, so `run_baselines.py --no-paid` skips it)
- Reports WER (ASR) and BLEU/chrF (MT) per dialect, zero-shot, no fine-tuning

`benchmark/leaderboard.md`: generated markdown table, one row per model × dialect, publishable as-is (this repo's own fine-tuned models can optionally be added as additional rows once trained, clearly marked as "this submission" vs. "baseline").

`benchmark/dataset_card.md`: HF-dataset-card-style documentation of the benchmark test set itself (size, dialect balance, consent basis, license) so the leaderboard is reproducible by others, not just a screenshot of numbers.

**Acceptance criteria:** `python -m benchmark.run_baselines --no-paid` completes on all 6 dialects using only open models and produces `benchmark/leaderboard.md` with populated (non-placeholder) rows; `benchmark/dataset_card.md` exists with all fields filled, including the per-dialect public-release-exclusion counts; no record lacking `public_release_ok: true` appears in the published test set (checked by an assertion in the script, not just documented).

---

## 9. Compute & Cost Budget

`BUDGET.md` — keep this current as the actual source of truth; the numbers below are the starting estimate to plan against, not a fixed promise:

| Stage | Compute | Est. time |
|---|---|---|
| Dialect-ID classifier (head-only fine-tune, all 6 dialects) | 1× A100, ~2 GPU-hrs | <1 day |
| ASR LoRA fine-tune (per dialect) | 1× A100, ~4 GPU-hrs | — |
| MT LoRA fine-tune (per dialect, both pivots) | 1× A100, ~3 GPU-hrs | — |
| TTS fine-tune (per dialect) | 1× A100, ~6 GPU-hrs | — |
| **All 6 dialects, all 3 tasks** | ~78 GPU-hrs total | reproducible in <2 days on 1× A100, or <6 hrs on 8× A100 |
| Active-learning scoring pass (per cycle, all dialects) | 1× A100, ~1 GPU-hr | — |
| Back-translation refresh (per checkpoint-promotion event that changes the generator checkpoint, per dialect — Section 4.1) | 1× A100, ~1–2 GPU-hrs | Recurring, not one-time — was misbudgeted as one-time in v4/early v5; re-runs each time a dialect's MT `production` pointer moves |
| TTS-bootstrap + audio-perturbation generation (one-time seed pass, all dialects) | 1× A100, ~5 GPU-hrs | — |
| Checkpoint promotion-gate dev-set eval (per training run, per task/dialect — Section 6.1) | 1× A100, ~0.1–0.3 GPU-hrs (dev set is small by design) | Recurring — every training run in Section 6 triggers this, not just a subset; was unbudgeted entirely in v4/early v5 |
| Zero-shot baseline benchmarking (Section 8.5, open models only) | 1× A100, ~2 GPU-hrs | <1 day |

Storage: budget ~50GB for raw+validated audio across 6 dialects at seed scale (tens of hours each); scales linearly with collection volume. Version `data/` with DVC (or git-lfs if DVC isn't available) rather than committing audio directly to git — `configs/pipeline.yaml` should record the DVC remote/target, and dataset-version hashes logged by `training/track_experiment.py` (Section 7.1) should resolve to a specific DVC revision, not just a file count.

**Backup, not just versioning.** DVC gives version history on one remote; it does not by itself protect against losing that remote (accidental deletion, storage-account lapse, a single misconfigured bucket policy). Field-collected audio — especially the idiom bank's field-collected entries and any `voice_clone_ok: true` recordings, both genuinely hard to re-collect on any reasonable timeline (§9's human-time table above) — is the one asset in this project that cannot be regenerated by re-running a script. Add a weekly automated sync from the DVC remote to a second, independent storage location (a different cloud provider/account or, at minimum, physically separate storage), documented in `scripts/backup_data.sh` and referenced in `BUDGET.md`'s storage line. This is a cheap, mechanical addition — the acceptance bar is "a second copy exists and is checkably current," not a full disaster-recovery program.

**Acceptance criteria (added):** `scripts/backup_data.sh --dry-run` reports what it would sync without transferring anything (safe to run in CI); `BUDGET.md`'s storage section states the backup cadence and target location; a test confirms the script fails loudly (nonzero exit, clear message) rather than silently no-op-ing if the secondary target is unreachable.

GPU-hours are not the bottleneck on this project — human fieldwork time is, and it belongs in the same budget doc so a reviewer sees the real critical path, not just the compute cost:

| Human-time item | Est. effort (per dialect, unless noted) | Notes |
|---|---|---|
| Field-collected idiom entries (§5.3, ≥30/dialect minimum) | Real fieldwork with speakers; not a fixed hour estimate — treat as a scheduling dependency, not a GPU-style estimate | This is the longest lead-time item in the whole spec; start in Build Order step 4, don't gate other steps on it |
| Consent protocol translation into all 6 dialects (§2.5) | One-time, all dialects | Needs a fluent speaker/reviewer per dialect, not machine translation of the protocol document itself |
| Active-learning annotation queue validation (§3) | Ongoing, scales with `K` (default 200/week/dialect) | This is the recurring cost that active learning is trying to make efficient, not eliminate |
| MOS survey rating (§7, `mos_survey.py`) | Per checkpoint evaluated | Needs multiple raters per sample for a usable MOS confidence interval, not one |
| IVR disambiguation-prompt recording (§10) | One-time, all dialects (a handful of short studio-quality clips: the Hindi disambiguation question + each dialect's spoken name + the graceful-failure message) | Must be human-recorded, not TTS-generated (Section 10 rules out TTS here specifically, since the prompt exists to disambiguate the dialect model that would otherwise generate it) — small in volume but easy to leave unbudgeted since it's not a fieldwork-scale item like the rows above |

**Acceptance criteria:** `BUDGET.md` exists, is referenced in the top-level `README.md`, contains both the GPU table and the human-time table above, and is updated by the agent with actual measured GPU-hours after the first real training run (not left as the estimate forever). The human-time table's idiom-collection row must be flagged explicitly if it's still unstarted by the time Build Order step 6 (serving API) begins, since it's the item most likely to still be open at Stage 4. The IVR prompt-recording row must be flagged if still unstarted by the time Build Order step 8 (IVR + demo app) begins.

---

## 10. Low-Bandwidth / IVR Channel

Rural governance/digital-services use cases assume low literacy and patchy internet — a web demo alone under-serves the actual target population, so add a phone-based channel as a thin layer over the existing API, not a parallel system.

`serving/ivr/twilio_app.py` (Twilio; Exotel as a documented alternative for India-specific telephony): incoming call → record utterance → `POST /asr` (with dialect-ID auto-routing) → `POST /mt` if translation requested → `POST /tts` → play response. Feature-flag this off by default (`configs/pipeline.yaml: ivr.enabled: false`) since it's one of two components with a real external paid-API dependency (the other being the optional GPT baseline in Section 8.5) — the agent should not wire live credentials into example configs, only document where they go (`.env.example`).

**Handling `dialect_ambiguous` on a phone call.** Section 8's disambiguation UX (return candidate dialects, let the caller pick) assumes a client that can render a UI — a phone call can't. Left unspecified, an ambiguous utterance on a call either silently falls through to a default dialect (reintroducing exactly the wrong-model-routing problem Section 8 was built to fix, just on this one channel) or crashes the call flow depending on how literally the `/asr` response is handled. `twilio_app.py` must instead run a voice-native disambiguation turn:
1. If `POST /asr`'s response has `dialect_ambiguous: true`, do not attempt `/mt`/`/tts` on a guessed dialect. Instead play a pre-recorded (not TTS-generated, to avoid depending on a dialect model to disambiguate the dialect) prompt in Hindi and the top-2 candidate dialects, asking the caller to say the name of their dialect or press a corresponding DTMF digit (IVR menu, e.g. "Marwari — press 1 or say Marwari").
2. Re-run `/asr` with the caller's disambiguation response passed as an explicit `dialect` param (from DTMF) or run `/dialect-id` again on the short disambiguation utterance — whichever resolves, forward that `dialect` explicitly on the retry of the original request so it bypasses auto-routing entirely.
3. Cap disambiguation attempts at 2 retries; if still ambiguous, fail gracefully with a pre-recorded "we couldn't understand the dialect, please try again later or contact [support channel from `docs/CONSENT_PROTOCOL.md`]" message and end the call cleanly — never silently default to a guessed dialect as a last resort, since a wrong-dialect ASR/MT result delivered confidently to a caller is worse than a clear failure message.
4. Every ambiguous-on-call event is logged to the same `active_learning/annotation_queue.py` path Section 8 already wires up for the web/API channel, tagged `source_channel: ivr`, so IVR-originated ambiguity feeds the same review queue rather than a separate, invisible one.

**Acceptance criteria:** with a valid Twilio/Exotel sandbox account, a test call completes the full record→ASR→MT→TTS→playback loop; with `ivr.enabled: false`, no IVR code path is invoked and the rest of the system builds/runs unaffected; a test call engineered to produce a `dialect_ambiguous` `/asr` response confirms the voice disambiguation prompt plays, a DTMF or spoken-dialect response correctly forces explicit `dialect` routing on retry, and the event appears in the annotation queue tagged `source_channel: ivr`; a test that forces `dialect_ambiguous` on both disambiguation attempts confirms the call ends with the graceful-failure message rather than defaulting to a guessed dialect or crashing the call flow.

---

## 11. Demo App

`serving/demo_app/` — Gradio app, thin client only (all logic lives in `serving/api/`, demo app just calls it): dialect selector, mic input, live ASR transcript, MT to Hindi/English, TTS playback, plus two tabs built specifically as live-demo artifacts for Stage 4 (a live panel demo rewards things a judge can click and hear, not a claimed number on a slide):

- **Cross-dialect transfer tab:** an interactive 6×6 heatmap (not a static CSV/image) built from `eval/cross_dialect_transfer.py` output — clicking any cell (dialect-pair) plays an actual audio sample of that zero-shot transfer live, so the panel hears the degradation instead of reading a number.
- **Idiom bank tab:** dialect selector + a handful of proverbs from `linguistic_artifacts/idiom_bank/<dialect>.jsonl` with a "translate" button that runs the live MT model and visibly shows figurative vs. literal output side by side (pulls from `idiom_mt_eval.py`, Section 5.3) — this is the "watch it correctly translate a proverb instead of translating it literally" moment.

**Acceptance criteria:** `python serving/demo_app/app.py` launches locally with all 6 dialects selectable, and shows a visible ✅ trained / ⏳ not-yet-trained status badge per dialect rather than silently hiding incomplete ones — the panel should see the true build state, not a cherry-picked happy path. Round-trip audio-in → transcript → translation → synthesized audio-out must work end-to-end for every dialect marked ✅ (if fewer than 6 dialects finish training before Stage 4, that's an honest ⏳ state, not a failed build). The transfer heatmap tab plays audio on cell-click for every dialect pair where both dialects are ✅; pairs involving a ⏳ dialect show an `INSUFFICIENT_DATA`/"not yet trained" state rather than a broken or silent cell. The idiom bank tab shows, per ✅ dialect, the aggregate figurative-accuracy percentage from `idiom_mt_eval.py` (Section 5.3) alongside one live example — not a single hand-picked "it worked" instance decoupled from the real aggregate number; do not hardcode a scripted "always correct" demo path.

---

## 12. Evaluation Summary (ties back to hackathon success matrix)

`eval/` must be able to produce, per dialect and blended:
- WER (target ≤10% per the stated success matrix) — reported separately for monolingual vs code-switched (Section 5.2)
- TTS MOS (target ≥4.0) — via `eval/mos_survey.py`, a simple web form for human raters, aggregated with confidence intervals
- Cross-dialect zero-shot transfer matrix (Section 5.1)
- Idiom/figurative-language MT accuracy vs. blended MT accuracy (Section 5.3)
- Generational drift summary per dialect (Section 7.4)
- Use-case coverage checklist (governance/education/digital-services scenarios, tracked manually against the objectives list in the original hackathon brief)

**Acceptance criteria:** `python -m eval.report --dialect all` produces one consolidated markdown report pulling from all the above (including `LIMITATIONS.md`, `benchmark/leaderboard.md`, and `checkpoints/<task>/<dialect>/canary_audit.jsonl` (Section 6.1) as linked artifacts, not duplicated content), suitable for direct inclusion in the Stage 5 final assessment submission.

---

### 12.1 Automated Test Suite (`make check`) — *v10, promoted from §13 backlog item 5*

**Problem this solves:** every "Acceptance criteria" line in this document (there are dozens) is phrased as a manual command for the agent or a reviewer to run and eyeball. Nothing ties them into one entry point, so verifying the whole build means re-reading the spec section by section — exactly the "re-reading the spec" cost §0 asks the agent to avoid within a section, just not yet solved across sections. This had no real dependency blocking it (unlike most of §13's remaining items), so it's promoted here rather than left in the backlog.

- `tests/` (new top-level directory, added to the Section 1 repo layout): one test module per numbered section that has acceptance criteria with a concrete pass/fail condition (schema validation, split-freeze/leak checks, promotion-gate direction/tolerance behavior, publish-filter k-anonymity, rate-limit thresholds, etc.) — each module encodes that section's acceptance criteria as an actual assertion, not a restatement of the prose.
- `Makefile` target `make check` (or `scripts/run_checks.sh` if the agent's environment doesn't have `make`): runs `pytest tests/` plus the standalone CLI-based acceptance checks that aren't natural pytest assertions (e.g. `tree rajasthani-lm` layout match from Section 1, `bash scripts/setup_env.sh` completing) in one invocation, and exits nonzero on any failure with a per-section summary (✅/❌ per numbered section), mirroring the ✅/⏳ status badge convention Section 11's demo app already uses so the two "build status" surfaces read the same way.
- This does not replace any section's own "Acceptance criteria" line — those remain the authoritative definition of "done" per section; `tests/` is a mechanical encoding of them, and if the two ever disagree, the prose acceptance criteria win and the test is the bug.
- Acceptance criteria that require paid/external resources by design (Section 10's IVR sandbox call, Section 8.5's optional GPT baseline) are skipped by default in `make check` (`pytest.mark.skip` with a reason string naming the flag that re-enables them, e.g. `ivr.enabled`/`--no-paid`) rather than failing a from-scratch clone with no credentials configured.

**Acceptance criteria:** `make check` (or the fallback script) runs on a clean checkout after `bash scripts/setup_env.sh`, exits 0 on a fully-built repo, and exits nonzero with a specific failing section named in the output if any single acceptance criterion regresses — verified by a test that deliberately breaks one known invariant (e.g. temporarily allows `train_asr.py` to read `test.jsonl`) and confirms `make check` fails on exactly that section, not a generic error.

---

## Build Order Summary for the Agent

1. Repo scaffold + schema, including orthography config, consent protocol, and split-assignment tooling (Sections 1–2, 2.1, 2.2, 2.5) — orthography normalization and permanent split assignment must exist before any data reaches `validated/`, since both are only safe to apply at first-write time, not retrofitted later
2. Data validation tooling + active learning scorer (Section 3) — before bulk collection, so priority scoring is live from day one
3. Augmentation modules, including the back-translation refresh hook on checkpoint promotion (Section 4, 4.1)
4. Dialect-ID (including the confidence-threshold routing logic used later by Section 8) + code-switch tagging + idiom bank collection tooling (Section 5, including 5.3 — start idiom collection early since it runs on the same field-collection channel and needs lead time to reach ≥100 entries/dialect)
5. Training scripts + checkpoint promotion gate + experiment tracking (Sections 6, 6.1, 7, including generational drift and limitations generators, 7.4–7.5) — `LICENSES.md` is written in this step, before any TTS training run, so the CPML-vs-MMS decision is made once and documented rather than discovered late
6. Serving API + BHASHINI adapter, with ULCA schema verification step and dialect-ID confidence-threshold routing wired in (Section 8)
7. Public benchmark leaderboard with the field-stripping/k-anonymity publish filter (Section 8.5) — can run in parallel with Section 8 once baseline models are available; gated on both the `public_release_ok` consent filter and the `publish_filter.py` re-identification check, not just data availability
8. IVR (flagged off) + demo app with interactive heatmap and idiom tabs (Sections 10–11)
9. Eval report + cards + budget doc (Sections 9, 12)
10. Wire `tests/` + `make check` (Section 12.1) once the above exist — it encodes their acceptance criteria, so it's a wrap-up step, not a parallel build track

Run `scripts/run_full_pipeline.sh --dialect <id>` to execute the above in order for a single dialect as an integration test before scaling to all six; run `make check` after, as the standing regression gate for every subsequent change.

---

## 13. Further Improvements Backlog (not yet acceptance-gated)

The seven fixes folded into v5 above were silent-failure-mode bugs — things that would corrupt data or inflate metrics without ever throwing an error. The items below are real gaps too, but lower-urgency or higher-effort; listed here so they're visible to the team and not lost, without inflating this build's required acceptance criteria. Items 2 and 5 were promoted into the acceptance-gated build in v10 (struck through below, with a pointer to where they now live); the rest still have a genuine blocking dependency (fieldwork lead time, an external policy decision, or new modeling work) that a coding agent alone can't close. Promote any of these into a numbered section (with acceptance criteria) in a future pass if time allows. (**Note for whoever edits this doc next:** this line named a specific next-version number in v6 through v9 and was stale in at least two of those versions because the number wasn't bumped on every edit — it's phrased version-independently now specifically so this stops recurring; don't reintroduce a hardcoded version number here.)

1. **Inter-annotator agreement is never measured.** `validator_id` is a single field per record — there's no double-annotation sample or agreement statistic (e.g. Cohen's kappa on a shared validation subset), so a validator who is systematically too lenient or too strict never surfaces. Worth a small (~5%) double-validated sample per dialect once validator throughput allows it.
2. ~~**Gender is collected but never analyzed.**~~ **Addressed in v10 — see §7.4/§7.2.** `speaker_gender` WER/MOS breakout is now wired into `generational_drift.py`'s existing cross-tab machinery and surfaced in the model card's known-limitations list.
3. **Orthography config versioning — partially addressed in v6, re-normalization pass still open.** Section 2.1's `configs/orthography/<dialect>.yaml` now carries a `version` field (v6), and Section 6.1's promotion gate reads it to flag `orthography_version_mismatch` on eval comparisons across a ruleset change. What's still missing: nothing yet re-normalizes already-validated `text_dialect` records in bulk when the ruleset changes — the version field lets the system *detect* drift between old- and new-ruleset text, but doesn't *fix* it. Needs a documented, scriptable re-normalization pass (`data/normalize_orthography.py --renormalize --dialect <id> --from-version N`) that reruns validated records through the current ruleset and updates `text_dialect` in place (still preserving `text_dialect_raw`), so old-ruleset data doesn't stay permanently stale relative to new-ruleset data.
4. **No held-out telephony-condition test set for the IVR path.** `audio_perturb.py` (Section 4.3) covers speed/pitch/background noise but not the frequency-response and compression artifacts of an actual phone codec (e.g. G.711 narrowband). Section 10's IVR acceptance criteria only checks the loop completes, not that ASR WER on codec-degraded audio is separately reported — real phone calls will sound different from the mic recordings the models were trained/evaluated on.
5. ~~**No automated test suite wiring.**~~ **Addressed in v10 — see §12.1.** `tests/` + `make check` now tie every section's acceptance criteria into one runnable entry point.
6. **Base-model vocabulary coverage is unchecked.** Whisper/IndicTrans2's tokenizers were built on Hindi/English-dominant corpora; dialect-specific phonemes or graphemes may be poorly covered by the base vocabulary, which LoRA fine-tuning can't fully fix. Worth a cheap early check (out-of-vocabulary rate on a validated sample per dialect) to flag whether full/extended tokenizer work is needed before investing in LoRA runs that will plateau.
7. **Consent and orthography protocol documents have no changelog.** `docs/CONSENT_PROTOCOL.md` and the per-dialect orthography configs are living documents translated for field use; if they're revised mid-project, there's no record of which version a given consent interaction or normalization pass was conducted under — relevant if a withdrawal or dispute needs to be traced back to what the speaker actually agreed to at the time.
8. **API input validation / DoS surface beyond auth.** Section 8's minimal auth (API key + rate limiting) doesn't address oversized or malformed audio uploads to `/asr`/`/tts` — worth a file-size cap and format validation at the FastAPI layer before the request reaches model inference, so a malformed upload fails fast and cheaply rather than inside a GPU-bound model call.
9. **Internal data-sharing re-identification risk beyond the public benchmark.**
   *(items 10–12 added in v11 — genuine unaddressed gaps, not reopened-pattern bugs; none have a blocking dependency, listed here rather than acceptance-gated because none are on this build's critical path to the Stage 4/5 judging criteria the way the idiom bank or transfer matrix are)*
10. **MOS rater pool composition is unspecified.** `eval/mos_survey.py` (Section 7.1) collects human naturalness ratings but nothing requires or checks that raters are actually fluent speakers of the dialect being rated, or reports rater count/background — a MOS score from non-fluent raters is a weaker signal than the model card presents it as. Worth a `rater_dialect_fluency` field on survey responses and a minimum-fluent-rater-count check before a MOS score is treated as final.
11. **No abuse-classifier quality check for the Section 8 content filter.** The v11 `content_filter.py` gate is deliberately conservative, but nothing evaluates its actual false-negative rate on a held-out set of known-unsafe examples across the 6 dialects (where a maintained term list may have poor coverage) — worth a small red-team-style eval set once the gate is in production use, rather than trusting the default threshold indefinitely.
12. **`voice_clone_ok` consent doesn't yet have a withdrawal-specific enforcement path.** Section 2.5's general withdrawal mechanism (tombstone record, `public_release_ok: false` on withdrawal) doesn't explicitly say what happens to a *TTS checkpoint already fine-tuned* on a speaker's voice-cloning-consented audio if that speaker later withdraws — the audio record can be tombstoned, but the trained voice model itself already encodes something derived from their voice. This is a genuinely hard problem (model unlearning is an open research area, not a scriptable fix) rather than a missed wiring — flagged here as a real open question for BHASHINI's policy team, not something a coding agent can resolve alone. Flagged as a scope note in Section 8.5 (v6): `publish_filter.py`'s field-stripping and k-anonymity check only covers the public leaderboard release path. If BHASHINI shares the internal `data/validated/` pool with partner institutions or outside researchers — plausible for a government-led initiative — `speaker_id`, full-granularity `region`, `settlement_type`, and age-cohort travel together at full resolution, which is the same re-identification risk Section 8.5 exists to prevent, just on an uncontrolled channel. Needs a BHASHINI-side data-sharing policy decision (what an approved export may contain, whether it routes through a lighter `publish_filter.py` pass or a separate agreement-gated process) before this can be scoped as a buildable component.
