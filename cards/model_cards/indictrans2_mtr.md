# Model Card: indictrans2_mtr

## Model Details
- **Task:** MT
- **Target Dialect:** MTR
- **Base Architecture:** Whisper-large-v3 / IndicTrans2 / MMS-TTS
- **Fine-Tuning Method:** LoRA / PEFT

## Intended Use
- Intended for spoken speech recognition, machine translation, and speech synthesis within Rajasthani language technology applications under BHASHINI platform initiatives.

## Training Data Summary
- **Data Source Breakdown:** Consented field collection, crowd validation, synthetic back-translation.
- **Audio Consent Gating:** Filtered strictly by `consent_basis` and `voice_clone_ok` consent fields.

## Evaluation & Performance
- **Primary Metric:** MT Evaluation Score (WER / BLEU / MOS)
- **Monolingual vs Code-Switched Gap:** Reported in LIMITATIONS.md
- **Figurative Idiom MT Accuracy:** Reported in LIMITATIONS.md

## Known Limitations
- Performance degrades on heavily code-switched Hindi-English speech (+4.5 WER delta).
- Zero-shot transfer degradation across distant dialect pairs.
- Gender-wise performance breakout documented in dataset card.
- Out-of-scope uses: High-risk biometric voice identification or unconsented commercial speech cloning.
