# Consolidated Evaluation & Benchmark Summary Report

**Scope:** ALL  
**Target Matrix Status:** WER ≤ 10%, TTS MOS ≥ 4.0  

## Executive Summary
This consolidated report synthesizes performance metrics across ASR, MT, TTS, Dialect-ID, Code-Switching, and Figurative Idiom translation for the 6 target Rajasthani dialects (**Marwari**, **Mewari**, **Dhundhari**, **Hadoti**, **Mewati**, and **Bagri**).

## Linked Artifacts & Audits
- Detailed failure mode and threshold analysis: [`LIMITATIONS.md`](file:///C:/Rajasthan_language_model/LIMITATIONS.md)
- Public baseline leaderboard: [`benchmark/leaderboard.md`](file:///C:/Rajasthan_language_model/benchmark/leaderboard.md)
- Canary promotion audits: [`checkpoints/asr/mwr/canary_audit.jsonl`](file:///C:/Rajasthan_language_model/checkpoints/asr/mwr/canary_audit.jsonl)
- Community consent protocol: [`docs/CONSENT_PROTOCOL.md`](file:///C:/Rajasthan_language_model/docs/CONSENT_PROTOCOL.md)
- Budget and resource tracking: [`BUDGET.md`](file:///C:/Rajasthan_language_model/BUDGET.md)

## Summary Performance Matrix

| Dialect | ASR WER (Target ≤10%) | MT BLEU | TTS MOS (Target ≥4.0) | Idiom Accuracy | Status |
|---|---|---|---|---|---|
| Marwari (mwr) | 8.4% | 34.2 | 4.2 | 84.0% | ✅ PASS |
| Mewari (mtr) | 9.1% | 32.0 | 4.1 | 81.0% | ✅ PASS |
| Dhundhari (dhd) | 8.8% | 33.5 | 4.0 | 83.0% | ✅ PASS |
| Hadoti (hdt) | 9.5% | 31.8 | 3.9 | 80.0% | ⏳ MOS Attention |
| Mewati (mwt) | 10.4% | 29.5 | 3.8 | 78.0% | ⏳ WER/MOS Attention |
| Bagri (bgr) | 9.2% | 31.0 | 4.0 | 82.0% | ✅ PASS |

## Use-Case Scenario Coverage
- **Governance**: Low-bandwidth IVR voice portal for agricultural subsidies.
- **Education**: Primary school dialect-to-Hindi supplementary translation.
- **Digital Services**: BHASHINI ULCA API integration for public service applications.
