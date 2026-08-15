# Empirical Limitations & Engineering Disclosures (LIMITATIONS.md)

---

## 1. Machine Translation (MT) Inference Integration Status & Contamination Audit
- **Current State**: Training orchestration (`training/train_mt.py`), canary validation, and promotion gates are spec-complete and functional.
- **Audit Finding (Test-Split Snooping Incident, 2026-08-15)**: An attempt to construct a rule-based transducer (`serving/mt_engine/rajasthani_mt.py`) was identified as contaminated because lexicon and inflection rules were authored after inspecting sample utterances in the held-out evaluation file (`data/realworld_test_200.jsonl`). All resulting BLEU/chrF++ numbers were immediately invalidated and reverted to `*Pending*`.
- **Enforced Policy — Strict Split Blindness**: No rule-based, dictionary, or heuristic translation logic may be authored or tuned against held-out test splits (`test.jsonl` / `realworld_test_200.jsonl`). All model parameters, transducers, and lexicons must be trained exclusively on training-split pools (`data/splits/<d>/train.jsonl`).
- **Serving Runtime**: Local serving provider (`serving/providers/local_provider.py`) runs offline mock fallback until an external or fully trained neural model (`ai4bharat/indictrans2-indic-indic-1B`) is evaluated under strict split isolation.
- **Regression Protection**: Active test `tests/test_verify_benchmark.py::test_mt_anti_echo_guard` remains configured with `@pytest.mark.xfail(strict=True)` until clean, non-contaminated neural translation is integrated.

---

## 2. Sample Size & Statistical Convergence ($N < 50$)
- **Held-Out Evaluation Suite**: Evaluated across 200 held-out real-world utterances ($N=34$ for Marwari, $N=34$ for Bagri, $N=33$ each for Mewari, Dhundhari, Hadoti, and Mewati).
- **Provisional Status**: Because per-dialect sample sizes ($N=33\text{--}34$) sit below the formal convergence threshold ($N \ge 50$), individual dialect scores carry wider 95% bootstrap intervals and remain designated as `Provisional`.
- **Target Proximity**: Dialects such as Marwari (CI upper bound 9.92%) and Bagri (CI upper bound 9.67%) approach the $\le 10.0\%$ target boundary, highlighting the priority of expanding test splits toward $N \ge 50$ via active learning queues.

---

## 3. Linguistic Artifacts & Idiom Bank Provenance
- **Canonical Size**: 630 curated proverbs and idioms (105 entries per dialect $\times$ 6 dialects) in `linguistic_artifacts/idiom_bank/`.
- **Sourcing & Legal Basis**: Curated classical Rajasthani folk proverbs and sayings under `public_domain` cultural heritage.
- **Audit Correction**: Replaced synthetic placeholder metadata (`explicit_written` from generic `"field_speaker_<dialect>_<n>"`) with accurate `public_domain` attribution. Decoupled test ingestion in `tests/test_section5.py` via `save_to_disk=False` to eliminate test-driven count drift.

---

## 4. Code-Switched & Telephony Evaluation Scope (Architectural Disclosures)
- **Code-Switching Representation**: In the current held-out test suite (`data/realworld_test_200.jsonl`), 6 of 200 utterances (3.0%) contain tagged English/Hindi code-switching tokens. The previously cited "+5.6 pts WER gap" originated from the simulation stub in `eval/wer.py` rather than a dedicated empirical code-switched benchmark. Formal empirical code-switching evaluation requires expanding `test_codeswitched.jsonl` via `codeswitch/cs_eval_set_builder.py`.
- **Telephony & Narrowband Audio**: The "~4.2 pts degradation" is an architectural estimate from the IVR telephony specification (`serving/ivr/`), pending acoustic evaluation over an 8kHz $\mu$-law test split.
- **Cross-Dialect Zero-Shot Transfer**: Zero-shot cross-dialect transfer matrices reflect vocabulary and phonological distance across dialect pairs (e.g., Bagri to Marwari zero-shot baseline).

---

## 5. Text-to-Speech (TTS) Deployment Pipeline
- **Evaluated Checkpoints**: Meta MMS-TTS VITS dialect checkpoints (`tts_<dialect>_mms_*`) achieved an empirical Mean Opinion Score (MOS) of **4.24 / 5.0** across 66 evaluations by 11 certified native evaluators per dialect zone.
- **Serving Architecture**: Real-time REST API serving currently utilizes a Google TTS (`gTTS`) Hindi fallback voice for live audio synthesis, with dialect-specific VITS model inference slated for production endpoint deployment.
