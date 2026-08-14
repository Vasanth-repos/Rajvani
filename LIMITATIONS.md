# Auto-Generated Failure Modes & Model Limitations (LIMITATIONS.md)

## 1. Per-Dialect Accuracy Summary & Threshold Warnings (Provisional Dev Set n=8)
- **Marwari (mwr)**: ASR WER: 8.4% | MT BLEU: 34.2 | TTS MOS: Pending formal human eval (Serving: Hindi gTTS fallback)
- **Mewari (mtr)**: ASR WER: 9.1% | MT BLEU: 32.0 | TTS MOS: Pending formal human eval (Serving: Hindi gTTS fallback)
- **Dhundhari (dhd)**: ASR WER: 8.8% | MT BLEU: 33.5 | TTS MOS: Pending formal human eval (Serving: Hindi gTTS fallback)
- **Hadoti (hdt)**: ASR WER: 9.5% | MT BLEU: 31.8 | TTS MOS: Pending formal human eval (Serving: Hindi gTTS fallback)
- **Mewati (mwt)**: ASR WER: 10.4% *(Provisional WER > 10.0%)* | MT BLEU: 29.5 | TTS MOS: Pending formal human eval
- **Bagri (bgr)**: ASR WER: 9.2% | MT BLEU: 31.0 | TTS MOS: Pending formal human eval (Serving: Hindi gTTS fallback)

### Explicit Performance Warnings
- WARNING: Dialect MWT provisional ASR WER is 10.4% (exceeds ≤10.0% target).
- WARNING: Dialect-specific neural TTS MOS ratings are pending formal human listener evaluation (current serving uses generic Hindi gTTS fallback).

## 2. Lowest-Resource Dialect
- **Mewati (mwt)** currently holds the lowest validated volume across the 6 dialects (~2.5 hours vs ~3.7 hours for Marwari).

## 3. Code-Switched Subset Performance Gap
- **ASR WER Gap**: Monolingual 7.2% vs. Code-Switched 12.8% (+5.6 pts WER degradation on English/Hindi mixed speech).
- **MT BLEU Gap**: Monolingual 34.2 vs. Code-Switched 28.0 (-6.2 BLEU delta).

## 4. Idiom & Figurative MT Accuracy Gap
- **Figurative Language MT Semantic Match Rate**: 82.0% of held-out idioms achieve cosine similarity $\ge 0.75$ against intended figurative meaning (vs. 94.0% semantic match rate on non-idiomatic conversational phrases), mitigated via RAG proverb overrides.

## 5. Cross-Dialect Zero-Shot Transfer Floor
- **ASR Task Transfer Floor**: BGR -> MWR (Worst zero-shot ASR WER: 36.6%)
- **MT Task Transfer Floor**: MWR -> BGR (Worst zero-shot MT BLEU: 7.6 BLEU)

## 6. Pipeline Coverage & Insufficient Data Flags
- **Sample Size Warning**: Dev sets currently contain 8 utterances per dialect (below the $n \ge 20$ statistical stability threshold; scores are provisional).
- **Consent Protocol**: Full coverage across explicit_written, explicit_verbal, public_domain, and synthetic types.
- **Generational Drift**: All 5 age cohorts (`under18` through `70plus`) tracked.
- **Idiom Bank**: All 6 dialects contain $\ge 100$ total entries and $\ge 24$ verified native field entries.
