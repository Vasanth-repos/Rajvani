# Auto-Generated Failure Modes & Model Limitations (LIMITATIONS.md)

## 1. Per-Dialect Accuracy Summary & Threshold Warnings
- **Marwari (mwr)**: ASR WER: 8.4% | MT BLEU: 34.2 | TTS MOS: 4.2
- **Mewari (mtr)**: ASR WER: 9.1% | MT BLEU: 32.0 | TTS MOS: 4.1
- **Dhundhari (dhd)**: ASR WER: 8.8% | MT BLEU: 33.5 | TTS MOS: 4.0
- **Hadoti (hdt)**: ASR WER: 9.5% | MT BLEU: 31.8 | TTS MOS: 3.9 *(MOS < 4.0)*
- **Mewati (mwt)**: ASR WER: 10.4% *(WER > 10.0%)* | MT BLEU: 29.5 | TTS MOS: 3.8 *(MOS < 4.0)*
- **Bagri (bgr)**: ASR WER: 9.2% | MT BLEU: 31.0 | TTS MOS: 4.0

### Explicit Performance Warnings
- WARNING: Dialect HDT TTS MOS is 3.9 (below 4.0 target).
- WARNING: Dialect MWT ASR WER is 10.4% (exceeds 10% target).
- WARNING: Dialect MWT TTS MOS is 3.8 (below 4.0 target).

## 2. Lowest-Resource Dialect
- **Mewati (mwt)** currently holds the lowest validated volume across the 6 dialects.

## 3. Code-Switched Subset Performance Gap
- **ASR WER Gap**: Monolingual 7.2% vs. Code-Switched 12.8% (+5.6 pts WER on code-switched dev/test).
- **MT BLEU Gap**: Monolingual 34.2 vs. Code-Switched 28.0 (-6.2 BLEU pts).

## 4. Idiom & Figurative MT Accuracy Gap
- **Figurative Language MT Accuracy**: 82.0% vs. Blended MT Accuracy 94.0% (-12.0 pts accuracy gap on figurative proverbs).

## 5. Cross-Dialect Zero-Shot Transfer Floor
- **ASR Task Transfer Floor**: MWR -> BGR (Worst zero-shot ASR WER: 38.1%)
- **MT Task Transfer Floor**: MWR -> BGR (Worst zero-shot MT BLEU: 7.6)

## 6. Pipeline Coverage & Insufficient Data Flags
- **Consent Protocol**: Full coverage across explicit_written, explicit_verbal, public_domain, and synthetic types.
- **Generational Drift**: All 5 age cohorts (`under18` through `70plus`) evaluated.
- **Idiom Bank**: All 6 dialects contain >= 100 total entries and >= 30 field-collected entries.
