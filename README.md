# Rajasthani Multi-Dialect Language Technology System (rajasthani-lm)

A reproducible, production-ready pipeline, BHASHINI-compatible API layer, IVR channel, and interactive demo app for 6 Rajasthani dialects: **Marwari (mwr)**, **Mewari (mtr)**, **Dhundhari (dhd)**, **Hadoti (hdt)**, **Mewati (mwt)**, and **Bagri (bgr)**.

## Quick Start

1. **Environment Setup:**
   ```bash
   bash scripts/setup_env.sh
   ```

2. **Run Pipeline Verification:**
   ```bash
   make check
   ```
   Or run a single dialect end-to-end:
   ```bash
   bash scripts/run_full_pipeline.sh --dialect mwr
   ```

3. **Launch Serving API:**
   ```bash
   python -m serving.api.main
   ```

4. **Launch Demo App:**
   ```bash
   python serving/demo_app/app.py
   ```

## Repository Structure

```
rajasthani-lm/
├── README.md
├── BUDGET.md                       # Compute and human-time budget (Section 9)
├── LICENSES.md                     # Base model license & commercial-use tracker (Section 6)
├── LIMITATIONS.md                  # Auto-generated failure-mode & limitations report (Section 7.5)
├── Makefile                        # `make check` verification entry point (Section 12.1)
├── configs/
│   ├── dialects.yaml                # Canonical registry of 6 dialects
│   ├── orthography/<dialect>.yaml   # Per-dialect orthography rules (v1)
│   └── pipeline.yaml                # Global execution pipeline settings
├── data/
│   ├── schema/                      # JSON schemas & validate.py for text & audio records
│   ├── raw/<dialect>/               # Raw collected data
│   ├── validated/<dialect>/         # Human-in-the-loop validated data
│   ├── synthetic/<dialect>/         # Augmented & back-translated data
│   ├── normalize_orthography.py     # Orthography normalization pass
│   └── splits/
│       ├── assign_split.py          # Permanent deterministic split assignment
│       └── <dialect>/{train,dev,dev_promotion,dev_canary,test}.jsonl
├── docs/
│   └── CONSENT_PROTOCOL.md          # Multilingual community consent methodology
├── linguistic_artifacts/            # Idiom & proverb bank + MT figurative eval
├── dialect_id/                      # Dialect boundary classifier & router
├── active_learning/                 # Uncertainty & novelty priority scoring
├── augmentation/                    # Back-translation refresh, TTS bootstrap, audio perturb
├── codeswitch/                      # Token tagger & code-switched eval builder
├── training/                        # LoRA fine-tuning & checkpoint promotion gate
├── eval/                            # WER, MOS, transfer matrix, drift, limitations, report
├── benchmark/                       # Public leaderboard & publish filter (k-anonymity)
├── serving/                         # FastAPI backend, ULCA adapter, TTS filter, IVR, Gradio app
├── cards/                           # Model & dataset cards
├── tests/                           # Section-by-section verification test suite
└── scripts/                         # Setup, consent, backup & pipeline runner scripts
```

## Documentation & References
- Compute and fieldwork budgets are documented in [BUDGET.md](file:///c:/Rajasthan_language_model/BUDGET.md).
- Model licenses are documented in [LICENSES.md](file:///c:/Rajasthan_language_model/LICENSES.md).
- Failure modes & evaluation gaps are auto-reported in [LIMITATIONS.md](file:///c:/Rajasthan_language_model/LIMITATIONS.md).
- Community consent terms are in [docs/CONSENT_PROTOCOL.md](file:///c:/Rajasthan_language_model/docs/CONSENT_PROTOCOL.md).

## Known Assumptions
- **ULCA Schema Version**: Pinned to BHASHINI ULCA v2.0 (verified 2026-08-13). Verified via `scripts/verify_ulca_schema.py`.
- **Content Moderation Asymmetry**: The `/tts` moderation filter (`serving/api/content_filter.py`) is intentionally conservative and low-precision-tolerant. False positives (blocking benign text) are preferred over false negatives (synthesizing harmful speech).
