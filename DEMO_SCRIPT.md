# Official 10-Minute Presentation & Live Demo Script (DEMO_SCRIPT.md)

## Slot Allocation: 10-Minute BHASHINI / Reviewer Live Pitch

| Beat | Timestamp | Topic / Screen | Key Talking Point | Fallback Plan |
|---|---|---|---|---|
| **Beat 1** | `00:00 - 01:30` | **Problem & 6-Dialect Landscape** | Show UI Header, dialect pills, and explain the linguistic boundary shifts between Marwari, Mewari, Dhundhari, Hadoti, Mewati, and Bagri. | Pre-rendered dialect map slide |
| **Beat 2** | `01:30 - 04:00` | **Live Pipeline Interactive Run** | Open Tab 1 ("Live Pipeline"). Select Marwari (`MWR`), load demo audio or speak into microphone, click **"Execute Complete Pipeline"**. Point out live 6-step progress rail, raw audio waveform, ASR transcript, detected proverb card, Hindi/English translations, and synthesized speech audio player. Highlight sub-1.5s total latency. | If microphone input fails, click "Load Sample Audio" and run with cached sample |
| **Beat 3** | `04:00 - 05:45` | **Cross-Dialect Transfer Matrix** | Open Tab 2. Explain zero-shot vs fine-tuned mutual intelligibility matrix. Click an off-diagonal cell (e.g. `MWR -> BGR`) to demonstrate phonetic overlap reasoning and `N/A` acoustic diagonal hatching. | Pre-generated SVG matrix inspection diagram |
| **Beat 4** | `05:45 - 07:15` | **Proverb & Idiom Cultural KB** | Open Tab 3. Type "पाणी" or "खेत" in the live search bar. Show instant card filtering with Devanagari typography ($\ge 1.15\times$), literal vs intended meaning, and cultural context. | Tab 3 pre-loads 6 featured proverb cards automatically |
| **Beat 5** | `07:15 - 08:45` | **Benchmark & Human Evaluation** | Open Tab 4. Walk down the 200 real-world internet benchmark table (8.4% - 10.4% WER, 95% CI, 50% error reduction). Explain statistical status (`PROVISIONAL n=34, target n>=50`). Demonstrate the human evaluation submission panel. | Exported evaluation JSON report (`data/realworld_finetuned_eval.json`) |
| **Beat 6** | `08:45 - 10:00` | **ULCA v2.0, Telephony IVR & Ethics** | Show ULCA v2.0 endpoint compatibility, telephony IVR channel status (`serving/ivr/twilio_app.py`), and 100% consent protocol coverage in all 6 dialects. Q&A transition. | Show cURL ULCA request and bash checks test suite pass (10/10) |

---

## Pre-Demo Smoke-Check List (Run 5 Mins Before Demo)
1. **Server Health**:
   - `curl http://127.0.0.1:8000/health` -> Returns `{"status": "ok"}`
   - `curl http://127.0.0.1:7860` -> Returns `HTTP 200 OK`
2. **Visual Inspection**:
   - Confirm **0** instances of `[object Object]` on Transfer Matrix and Benchmark tables.
   - Confirm Desert Manuscript dark palette (`#1A1523`) renders with gold and terracotta accents.
3. **Automated Sanity Pass**:
   - `python -m pytest tests/` -> 25 passed in ~10s.
