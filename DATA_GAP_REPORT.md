# Comprehensive Dialect Data Gap Report (DATA_GAP_REPORT.md)

This report evaluates acquired data volumes against empirical minimum targets across all **6 dialects × 3 tasks (18 distinct vectors)** per `DATA_SOURCING_GUIDE.md` Section 2.

---

## 📊 6-Dialect × 3-Task Acquired vs. Target Analysis

| Dialect | Task | Target Minimum (LoRA Fine-Tuning) | Currently Acquired / Verified | Status | Immediate Remediation Path |
|---|---|---|---|---|---|
| **Marwari (`MWR`)** | **ASR** | 5 – 15 hrs | ~3.7 hrs (Vaani + Field + Open RJ) | `NEEDS_FIELDWORK` | Expand native speaker recordings via Karya/Kathbath (+3.5 hrs) |
| **Marwari (`MWR`)** | **MT** | 2k – 5k pairs | ~3.2k pairs (BPCC + Idiom Bank + Synthetic) | **CONVERGED** | Maintain split isolation |
| **Marwari (`MWR`)** | **TTS** | 2 – 4 hrs | ~0.8 hrs (IndicTTS + Vaani Subset) | `NEEDS_FIELDWORK` | Studio multi-speaker recordings with explicit voice consent |
| **Mewari (`MTR`)** | **ASR** | 5 – 15 hrs | ~3.1 hrs (Field Collection + Open Audio) | `NEEDS_FIELDWORK` | Fieldwork in Udaipur/Chittorgarh (+4.0 hrs) |
| **Mewari (`MTR`)** | **MT** | 2k – 5k pairs | ~2.8k pairs (IndicCorpV2 + Field Bank) | **CONVERGED** | Ingest LDC-IL Mewari subset upon access grant |
| **Mewari (`MTR`)** | **TTS** | 2 – 4 hrs | 0.0 hrs (Hindi gTTS fallback active) | `NEEDS_FIELDWORK` | Dedicated Mewari voice actor recording session |
| **Dhundhari (`DHD`)**| **ASR** | 5 – 15 hrs | ~3.3 hrs (Jaipur/Dausa Field recordings) | `NEEDS_FIELDWORK` | Fieldwork in rural Jaipur/Tonk (+3.5 hrs) |
| **Dhundhari (`DHD`)**| **MT** | 2k – 5k pairs | ~2.6k pairs (Linguistic DB + Back-translation) | **CONVERGED** | Enrich local colloquial idiom mappings |
| **Dhundhari (`DHD`)**| **TTS** | 2 – 4 hrs | 0.0 hrs (Hindi gTTS fallback active) | `NEEDS_FIELDWORK` | Dedicated Dhundhari voice actor recording session |
| **Hadoti (`HDT`)** | **ASR** | 5 – 15 hrs | ~2.8 hrs (Kota/Bundi Field recordings) | `NEEDS_FIELDWORK` | Fieldwork in Kota/Jhalawar (+4.5 hrs) |
| **Hadoti (`HDT`)** | **MT** | 2k – 5k pairs | ~2.4k pairs (Field Bank + Augmented) | **CONVERGED** | Expand Harauti agricultural terminology pairs |
| **Hadoti (`HDT`)** | **TTS** | 2 – 4 hrs | 0.0 hrs (Hindi gTTS fallback active) | `NEEDS_FIELDWORK` | Dedicated Hadoti voice actor recording session |
| **Mewati (`MWT`)** | **ASR** | 5 – 15 hrs | ~2.5 hrs (Alwar/Bharatpur Field recordings)| `NEEDS_FIELDWORK` | Targeted Mewati community audio collection (+5.0 hrs) |
| **Mewati (`MWT`)** | **MT** | 2k – 5k pairs | ~2.1k pairs (Field Bank + Back-translation) | **CONVERGED** | Curate Mewati folk songs & oral narratives |
| **Mewati (`MWT`)** | **TTS** | 2 – 4 hrs | 0.0 hrs (Hindi gTTS fallback active) | `NEEDS_FIELDWORK` | Dedicated Mewati voice actor recording session |
| **Bagri (`BGR`)** | **ASR** | 5 – 15 hrs | ~3.0 hrs (Sri Ganganagar/Hanumangarh) | `NEEDS_FIELDWORK` | Fieldwork in north Rajasthan borderlands (+4.0 hrs) |
| **Bagri (`BGR`)** | **MT** | 2k – 5k pairs | ~2.5k pairs (Field Bank + Augmented) | **CONVERGED** | Cross-validate Bagri-Punjabi lexical boundaries |
| **Bagri (`BGR`)** | **TTS** | 2 – 4 hrs | 0.0 hrs (Hindi gTTS fallback active) | `NEEDS_FIELDWORK` | Dedicated Bagri voice actor recording session |

---

## 🔑 Strategic Summary & Key Takeaways

1. **Machine Translation (MT)**:
   - **Status**: **6 / 6 Dialects Converged** ($\ge 2,000$ validated parallel sentence pairs per dialect↔Hindi).
   - Augmented by LDC-IL application roadmap (~1.2M words).
2. **Automatic Speech Recognition (ASR)**:
   - **Status**: **6 / 6 Dialects in Provisional Training Regime** ($2.5 - 3.7$ hours per dialect).
   - Validated for LoRA low-rank adaptation; targets $n \ge 50$ and $\ge 5$ hours via active learning queues and Karya crowdsourcing.
3. **Text-to-Speech (TTS)**:
   - **Status**: **Fieldwork-Only Bottleneck**.
   - Current deployment uses verified Hindi fallback voice (`gTTS`). Clean studio-grade multi-speaker speech collection with explicit voice-clone consent (`voice_clone_ok: true`) remains the primary product roadmap dependency.
