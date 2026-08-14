# Compute & Cost Budget (BUDGET.md)

## GPU Compute Allocation

| Stage | Compute | Est. time | Notes |
|---|---|---|---|
| Dialect-ID classifier (head-only fine-tune, all 6 dialects) | 1× A100, ~2 GPU-hrs | <1 day | Pre-trained MMS/XLSR encoder |
| ASR LoRA fine-tune (per dialect) | 1× A100, ~4 GPU-hrs | ~24 GPU-hrs total | Whisper-large-v3 / MMS-1b |
| MT LoRA fine-tune (per dialect, both pivots) | 1× A100, ~3 GPU-hrs | ~18 GPU-hrs total | IndicTrans2 |
| TTS fine-tune (per dialect) | 1× A100, ~6 GPU-hrs | ~36 GPU-hrs total | MMS-TTS default / XTTS-v2 |
| **All 6 dialects, all 3 tasks** | **~78 GPU-hrs total** | **~3.3 days on 1× A100 (or <10 hrs on 8× A100)** | Reproducible pipeline |
| Active-learning scoring pass (per cycle, all dialects) | 1× A100, ~1 GPU-hr | Recurring | Run after each checkpoint |
| Back-translation refresh (per checkpoint promotion event) | 1× A100, ~1–2 GPU-hrs | Recurring | Re-runs when MT production pointer moves |
| TTS-bootstrap + audio-perturbation generation (seed pass) | 1× A100, ~5 GPU-hrs | One-time | Synthetic augmentation |
| Checkpoint promotion-gate dev-set eval (per training run) | 1× A100, ~0.1–0.3 GPU-hrs | Recurring | Evaluates stored per-utterance scores |
| Zero-shot baseline benchmarking (open models only) | 1× A100, ~2 GPU-hrs | <1 day | Section 8.5 leaderboard |

## Human Fieldwork Time Allocation

| Human-time item | Est. effort | Notes |
|---|---|---|
| Field-collected idiom entries (§5.3, ≥30/dialect minimum) | Fieldwork with native speakers | **CRITICAL PATH**: Longest lead-time item; starts early in build order |
| Consent protocol translation into all 6 dialects (§2.5) | Fluent speaker/reviewer per dialect | Translation of `docs/CONSENT_PROTOCOL.md` into local dialects |
| Active-learning annotation queue validation (§3) | ~200 items/week/dialect | Human validation of top priority score items |
| MOS survey rating (§7, `mos_survey.py`) | Multiple raters per sample | Human listening rating pass for TTS evaluation |
| IVR disambiguation-prompt recording (§10) | Studio recording session | Spoken dialect names & Hindi prompt clips for telephony |

## Storage & Backup Policy
- Audio storage budget: ~50GB raw + validated audio.
- Data versioning: DVC / Git LFS tracked in `configs/pipeline.yaml`.
- Automated Backup Cadence: Weekly automated sync from DVC primary storage to secondary independent location via `scripts/backup_data.sh`.
