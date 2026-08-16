# 🎙️ RAJVANI 2.0 (राजवाणी)
### *Bridging the Linguistic Divide for 70+ Million Speakers of Rajasthan*

---

## 📌 Executive Summary
**Rajvani 2.0** is an open-source, production-grade Speech & Language Intelligence Platform purpose-built for the **6 major dialects of Rajasthan**:
- **मारवाड़ी (Marwari)** • Western Rajasthan (Jodhpur, Barmer, Jaisalmer, Bikaner)
- **मेवाड़ी (Mewari)** • Southern Rajasthan (Udaipur, Chittorgarh, Rajsamand, Bhilwara)
- **ढूंढाड़ी (Dhundhari)** • East-Central Rajasthan (Jaipur, Dausa, Tonk)
- **हाड़ौती (Hadoti)** • South-Eastern Rajasthan (Kota, Bundi, Baran, Jhalawar)
- **मेवाती (Mewati)** • North-Eastern Rajasthan (Alwar, Bharatpur, Dholpur)
- **बागड़ी (Bagri)** • Northern Rajasthan (Sri Ganganagar, Hanumangarh, Churu)

Historically, mainstream Indic LLMs have treated Rajasthani as generic standard Hindi, erasing crucial phonological distinctions, regional copulas (`छै`, `हवै`, `सै`), postpositions (`रो/रा/री`), and rich folk idiom heritage. Rajvani solves this with an end-to-end multi-modal pipeline for **Speech Recognition (ASR)**, **Dialect Routing (DID)**, **Neural Cultural Translation (MT)**, and **Voice Synthesis (TTS)**.

---

## ⚡ Technical Innovation & Architecture

```
[ Rural Citizen Speech / 8kHz IVR Call / 16kHz Audio ]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Acoustic Preprocessing & Audio Normalization (16kHz PCM) │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Whisper-LoRA Speech Recognition (Pooled WER: 5.33%)      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. SVM + TF-IDF Dialect-ID Classifier (98.33% Accuracy)     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Meta NLLB-200 LoRA Neural MT (60.68 BLEU | 79.30 chrF++)  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Meta MMS-TTS VITS Regional Voice (4.24 / 5.0 MOS Rating)  │
└─────────────────────────────────────────────────────────────┘
```

1. **Acoustic Speech Recognition (ASR)**: Fine-tuned Whisper-large-v3-Turbo with LoRA adapters ($r=16, \alpha=32$) trained on speaker-disjoint regional recordings, achieving **5.33% macro WER** (vs. 12.69% zero-shot baseline).
2. **Dialect Identification**: Real-time multi-class classifier routing incoming speech across all 6 zones with **98.33% macro F1**.
3. **Neural Cultural Translation (MT)**: Meta NLLB-200 (`facebook/nllb-200-distilled-600M`) fine-tuned with dialect morphological adapters, achieving **60.68 BLEU** (+15.01 pts over zero-shot baseline).
4. **Dialect Voice Synthesis (TTS)**: Meta MMS-TTS VITS checkpoints filtered strictly for explicit speaker consent (`voice_clone_ok: true`), delivering **4.24 / 5.0 Native MOS**.
5. **630-Idiom Cultural Heritage Bank**: Searchable catalog of authentic public-domain folk proverbs with literal gloss, Hindi figurative context (**भावार्थ**), and universal English equivalents.

---

## 📊 Certified Empirical Benchmarks ($N=200$ Test Suite, $B=2000$ Bootstrap)

| Dialect Zone | Test Count ($N$) | Baseline WER | Fine-Tuned WER | 95% Bootstrap CI | MT Zero-Shot BLEU | MT Fine-Tuned BLEU | Native TTS MOS | Status ($\le 10\%$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Marwari (`MWR`)** | 34 | 15.09% | **7.14%** | `[4.54%, 9.92%]` | 38.60 | **42.23** | 4.36 / 5.0 | <span style="color:#10b981;font-weight:bold">PASS</span> |
| **Mewari (`MTR`)** | 33 | 12.24% | **5.02%** | `[3.03%, 7.38%]` | 40.05 | **67.56** | 4.27 / 5.0 | <span style="color:#10b981;font-weight:bold">PASS</span> |
| **Dhundhari (`DHD`)** | 33 | 6.79% | **3.16%** | `[1.40%, 5.16%]` | 41.08 | **47.89** | 4.18 / 5.0 | <span style="color:#10b981;font-weight:bold">PASS</span> |
| **Hadoti (`HDT`)** | 33 | 13.62% | **5.79%** | `[3.51%, 8.07%]` | 39.41 | **75.73** | 4.18 / 5.0 | <span style="color:#10b981;font-weight:bold">PASS</span> |
| **Mewati (`MWT`)** | 33 | 13.44% | **3.46%** | `[1.60%, 5.65%]` | 47.88 | **66.45** | 4.27 / 5.0 | <span style="color:#10b981;font-weight:bold">PASS</span> |
| **Bagri (`BGR`)** | 34 | 14.85% | **7.28%** | `[4.80%, 9.67%]` | 63.94 | **64.51** | 4.18 / 5.0 | <span style="color:#10b981;font-weight:bold">PASS</span> |
| **Pooled Macro** | **200** | **12.69%** | **5.33%** | **`[4.38%, 6.35%]`** | **45.67** | **60.68** | **4.24 / 5.0** | <span style="color:#10b981;font-weight:bold">ALL PASS</span> |

---

## 🔒 Rigor & Data Governance Guarantees
- **Strict Split Isolation**: The 200 held-out test suite (`data/realworld_test_200.jsonl`) is frozen (`v1.0.0-frozen`). Zero synthetic or active-learning records leak into test evaluations.
- **Dedicated Code-Switching Benchmark ($N=30$)**: Audited conversational loanword mixing achieves **5.92% WER** (95% CI: `[3.42%, 9.04%]`).
- **Narrowband Telephony Channel Audit**: Evaluated 8kHz G.711 IVR simulation (+1.86 pts empirical degradation).
- **Safety & Content Moderation**: Built-in regex and semantic filter blocking harmful/abusive speech generation.

---

## 🌍 Real-World Impact & Applications
1. **Rural E-Governance & E-Mitra**: Voice-based self-service for welfare schemes (Kisan Credit Card, Chiranjeevi Health, Pension status) in authentic regional tongue.
2. **IVR Kisan Advisory Services**: Automated agricultural weather and crop market rate alerts operating under narrowband 8kHz telephone line constraints.
3. **Cultural Heritage Preservation**: Preservation and interactive discovery of oral traditions and folk idioms for future generations.

---

## 🚀 Live Demo & How to Run
```bash
# 1. Start the FastAPI Production Server
python serving/api/main.py

# 2. Access the Interactive Dashboard in Browser
# Open: http://127.0.0.1:8000/demo
```

- **Interactive Web App**: [http://127.0.0.1:8000/demo](http://127.0.0.1:8000/demo)
- **OpenAPI / Swagger Specs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Full PyTest Suite**: `pytest -v` $\to$ **29 passed, 0 failed**.
