# Product Requirements Document (PRD) — Rajvani (राजवाणी)

## Problem
India is home to hundreds of regional languages and dialects, yet mainstream Indian Language Technologies (e.g., standard Hindi ASR/MT) fail completely on regional dialectal varieties spoken by over 80 million people in Rajasthan. 
The six major Rajasthani dialects—**Marwari (`MWR`)**, **Mewari (`MTR`)**, **Dhundhari (`DHD`)**, **Hadoti (`HDT`)**, **Mewati (`MWT`)**, and **Bagri (`BGR`)**—suffer from severe phonetic, lexical, and orthographic divergence from standard Khari Boli Hindi. This causes:
1. Standard Hindi speech recognition (ASR) to suffer >40% Word Error Rates (WER) on rural Rajasthani speakers.
2. Machine translation systems (MT) to hallucinate or mistranslate dialectal idioms and proverbs.
3. Absence of native-sounding regional Voice Synthesis (TTS) for governance, agricultural advisory, healthcare, and educational voicebots.
4. Loss of oral linguistic heritage and undocumented folk knowledge.

## Target Users
1. **Rural & Semirural Citizens of Rajasthan**: Farmers, artisans, and women seeking voice-based access to state welfare schemes (Jan Soochna, e-Mitra, Kisan Seva) in their native mother tongue.
2. **State & Central Government Departments**: e-Governance bodies and MeitY BHASHINI ecosystem needing ULCA-compliant multi-dialect endpoints.
3. **Agritech & Healthtech Voice Agents**: Providers deploying conversational IVR bots across different agro-climatic zones of Rajasthan.
4. **Linguists & Cultural Researchers**: Archival organizations documenting oral folklore, idioms, and dialectal variations.

## Current Alternatives
- **Standard Hindi AI (Google Speech / Bhashini Hindi)**: Treats Rajasthani as standard Hindi; fails on dialect phonology and vocabulary (e.g., *म्हारो*, *कोनी*, *कांई*, *छै*).
- **Generic Multilingual LLMs (GPT-4, Gemini)**: Have minimal dialectal token coverage, produce synthetic hallucinated dialect sentences, and lack native audio models.
- **Academic Datasets (VAANI / Karya)**: Fragmented raw audio with inconsistent transcription quality and lacking standardized speaker-disjoint evaluation benchmarks.

## Proposed Solution: Rajvani (राजवाणी)
An end-to-end, multi-dialect language intelligence platform and verified benchmark suite tailored specifically for Rajasthan's six core dialects. Rajvani integrates:
1. **Parameter-Efficient Speech Recognition (ASR)**: Fine-tuned Whisper and IndicASR models optimized for regional phonetics.
2. **Neural Machine Translation (MT)**: Dialect-to-Hindi and Dialect-to-English translation with strict preservation of culturally specific terminology.
3. **Dialect Voice Synthesis (TTS)**: Regional voice generation via VITS / Meta MMS-TTS with natural intonation.
4. **Cultural Knowledge & Proverb Bank**: Curated 630-proverb RAG system with Devanagari orthography normalization.
5. **Standardized Standards & BHASHINI Compatibility**: ULCA v2.0 schema integration and verifiable zero-leakage evaluation protocols with bootstrap confidence intervals.

## Core Features
1. **Dialect Identification (DID)**: Automatic 6-way classification of input text or speech audio.
2. **Multi-Dialect ASR Pipeline**: Speech-to-text with real-time acoustic normalization and post-processing.
3. **Cross-Dialect MT Engine**: Bi-directional translation between all 6 dialects, Hindi, and English.
4. **Devanagari Orthography Normalizer**: Rule-based standardizer resolving phonetic orthographic ambiguity across districts.
5. **Cultural Proverb RAG Bank**: Semantic vector retrieval across 105 canonical proverbs per dialect (630 total).
6. **ULCA / BHASHINI Unified API**: Standardized JSON request/response pipelines for plug-and-play integration.
7. **Cloud Colab & Local Hybrid Execution**: Zero-local-disk Google Drive cloud training notebook (`Untitled2.ipynb`) and lightweight local inference FastAPI server.

## MVP Scope (Demonstrable Deliverables)
- [x] Functional multi-dialect ASR achieving frozen pooled macro WER of 5.33% across held-out test suite (N=200).
- [x] Zero-leakage verification suite (`eval/verify_leakage.py`) proving 0 test overlap across 5 training pools.
- [x] FastAPI serving engine with ULCA-compliant endpoints (`/api/v1/asr`, `/api/v1/translate`, `/api/v1/tts`, `/api/v1/proverbs`).
- [x] Interactive web dashboard for live audio recording, real-time transcription, translation, and audio synthesis.
- [x] 630-record verified idiom/proverb repository with English and Hindi contextual glosses.
- [x] End-to-end cloud training & evaluation pipeline in Google Colab (`Untitled2.ipynb`).

## Future Features
- Extension to sub-dialects (Shekhawati, Godwari, Nimadi).
- Native speaker crowd-sourcing validation portal with audio recording consent workflows.
- On-device edge quantization (ONNX / GGML) for offline deployment in remote areas.

## Success Criteria
- **ASR Performance**: WER $\le$ 10.0% across all 6 dialects on held-out speaker-disjoint evaluation sets.
- **Evaluation Rigor**: 100% passing rate on 9-point consistency gate (`verify_consistency.py`) and zero data leakage.
- **Serving Latency**: Sub-500ms response time on local API translation and normalization.
- **Cultural Accuracy**: 100% precision on canonical idiom retrieval.
