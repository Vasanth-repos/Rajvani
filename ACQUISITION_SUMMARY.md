# Data Acquisition & Sourcing Executive Summary (ACQUISITION_SUMMARY.md)

This document summarizes the execution of the Rajasthani Dialect Data Sourcing Plan per `DATA_SOURCING_GUIDE.md` and the automated discovery pass conducted via `discover_datasets.py`.

---

## 1. Verified & Ingested Open Sources

The following open datasets were downloaded, spot-checked, schema-validated, and structured under `data/raw/<dialect>/<source>/` with accompanying `SOURCE.md` metadata:

| Source Identifier | License | Local Ingestion Path | Ingested Modality | Verified Coverage |
|---|---|---|---|---|
| **`TigreGotico/tts_vc_Vaani_marwari_mwr_miro`** | CC-BY-4.0 | `data/raw/mwr/vaani_marwari_subset/` | Audio + Transcripts | Marwari (`MWR`) district speech |
| **`severo/speech-rj-hi`** | CDLA-Permissive-2.0 / MS Open Data | `data/raw/mwr/speech_rj_hi_soda/` | Audio + Transcripts | Soda (Rajasthan) phonetic read speech |
| **`SPRINGLab/IndicTTS_Rajasthani`** | CC-BY-4.0 | `data/raw/mwr/indictts_rajasthani/` | Audio + Transcripts | IIT Madras Rajasthani TTS acoustic set |
| **`gurudevempire/rajasthani-ai-data`** | Apache-2.0 | `data/raw/mwr/rajasthani_ai_data/` | Text Pairs & QA | Curated Rajasthani proverb dialogue |

---

## 2. Action Items Requiring Human Intervention

The following resources require human / administrative actions and cannot be bypassed programmatically:

1. **LDC-IL Gold Standard Rajasthani Raw Text Corpus (~1.2M words)**:
   - **Action**: Submit formal non-commercial research application at `https://data.ldcil.org` per [`LDC_IL_ACCESS_REQUEST.md`](file:///c:/Rajasthan_language_model/LDC_IL_ACCESS_REQUEST.md).
2. **Kaggle API Credentials**:
   - **Action**: Configure `~/.kaggle/kaggle.json` or set `KAGGLE_API_TOKEN` to enable automated Kaggle catalog polling in periodic cron sweeps. *(Kaggle search was skipped with `--skip-kaggle` on this pass).*
3. **Dialect-Specific Studio TTS Fieldwork**:
   - **Action**: Conduct dedicated voice talent recording sessions (2-4 hours per dialect) across Mewari, Dhundhari, Hadoti, Mewati, and Bagri with explicit signed `voice_clone_ok: true` consent to replace the current Hindi fallback voice (`gTTS`).

---

## 3. Promising Hits Flagged for Manual Expert Review

The following items were identified by the automated search but contain ambiguities requiring expert linguist review before ingestion into training splits:

| Candidate ID / URL | Match Keyword | Ambiguity / Review Need | Recommendation |
|---|---|---|---|
| `1rsh/tts-rajasthani-ulca` | `Rajasthani` | Metadata specifies generic Rajasthani without fine-grained dialect tag (`MWR` vs `DHD`). | Review audio samples against Harauti/Marwari phonetic markers. |
| `1rsh/SPInO-RajasthaniTourism_qa` | `Rajasthani` | QA text dataset; verify whether text is authentic dialect Devanagari or Rajasthani-accented Hindi. | Spot-check 50 rows with native linguistic leads. |
| `SayantanJoker/original_data_rajasthani_tts` | `Rajasthani` | Unconfirmed license in dataset card; small batch size. | Do not train until license is verified in writing. |
| `viplismism/Qwen2.5-7B-Mewari-MLX-LoRA` | `Mewari` | Fine-tuned LLM model weights (MLX format), not raw training data. | Test as potential pseudo-labeler for Mewari text alignment. |
| `Harveenchadha/vakyansh-wav2vec2-rajasthani-raj-45` | `raj` | Wav2Vec2 ASR model checkpoint trained on Vakyansh Rajasthani. | Benchmark against Whisper-v3-Turbo baseline. |

---

## 4. Consent & Provenance Tagging Policy

All newly acquired open datasets have been tagged per this project's JSON schema:
- **`source`**: `open_dataset` *(Explicitly distinguished from `field_collection`)*
- **`consent_basis`**: `open_license_third_party`
- **`split_read_guard`**: Quarantined strictly in `data/raw/` and prohibited from touching `dev_canary.jsonl` or `test.jsonl`.
