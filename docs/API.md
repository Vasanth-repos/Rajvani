# API Reference — Rajvani (राजवाणी)

The Rajvani FastAPI server provides REST endpoints for Speech Recognition (ASR), Machine Translation (MT), Text-to-Speech (TTS), Orthography Normalization, Cultural Proverb RAG, Benchmark Evaluation, and Human Feedback.

- **Base URL**: `http://127.0.0.1:8000` (or `http://localhost:8000`)
- **Interactive Documentation**: `http://127.0.0.1:8000/docs` (Swagger UI) & `http://127.0.0.1:8000/redoc`

---

## Authentication
Protected inference and processing endpoints require the `X-API-Key` HTTP Header.

```http
X-API-Key: test_key
```
*(Valid test keys: `test_key`, `bhashini_demo_key`, `admin_secret`)*

---

## Endpoints

### 1. Health & Dialect Discovery

#### `GET /api/health`
Checks server status and lists active dialects.
- **Response**:
```json
{
  "status": "online",
  "version": "2.0.0",
  "supported_dialects": ["MWR", "MTR", "DHD", "HDT", "MWT", "BGR"]
}
```

#### `GET /api/dialects/{dialect_id}`
Returns metadata, districts, and phonetic markers for a specific dialect (`MWR`, `MTR`, `DHD`, `HDT`, `MWT`, `BGR`).

---

### 2. Speech-to-Text (ASR)

#### `POST /api/asr`
Transcribes spoken dialect audio (16kHz WAV/MP3) into Devanagari text.
- **Headers**: `X-API-Key: test_key`, `Content-Type: multipart/form-data`
- **Form Data**:
  - `file`: Audio binary (`.wav`, `.mp3`)
  - `dialect` *(optional)*: `MWR`, `MTR`, `DHD`, `HDT`, `MWT`, `BGR`
  - `preferred_provider` *(optional)*: `local`
- **Response**:
```json
{
  "status": "success",
  "dialect": "MWR",
  "raw_transcript": "म्हारो नाम राम है",
  "normalized_transcript": "म्हारो नाम राम है",
  "asr_latency_sec": 0.42
}
```

---

### 3. Machine Translation (MT)

#### `POST /api/translate`
Translates text between dialect, Standard Hindi, and English.
- **Headers**: `X-API-Key: test_key`, `Content-Type: application/json`
- **Request Body**:
```json
{
  "text": "म्हारो नाम राम है, म्हाँ जोधपुर रा रहवासी हाँ।",
  "source_dialect": "MWR",
  "target_language": "hin",
  "preferred_provider": "local"
}
```
- **Response**:
```json
{
  "source_dialect": "MWR",
  "target_language": "hin",
  "original_text": "म्हारो नाम राम है, म्हाँ जोधपुर रा रहवासी हाँ।",
  "translation": "मेरा नाम राम है, हम जोधपुर के निवासी हैं।",
  "latency_sec": 0.18
}
```

---

### 4. Text-to-Speech (TTS)

#### `POST /api/tts`
Synthesizes speech audio from dialect or Hindi text.
- **Headers**: `X-API-Key: test_key`, `Content-Type: application/json`
- **Request Body**:
```json
{
  "text": "खम्मा घणी सा, आपरो स्वागत है।",
  "dialect": "MWR",
  "backend": "mms",
  "preferred_provider": "local"
}
```
- **Response**:
```json
{
  "status": "success",
  "audio_url": "/static/audio/tts_output.mp3",
  "duration_sec": 2.4,
  "latency_sec": 0.35
}
```

---

### 5. Unified Full Pipeline

#### `POST /api/pipeline/run`
Executes end-to-end ASR $\to$ Normalization $\to$ MT $\to$ TTS in a single call.
- **Request Body**:
```json
{
  "dialect": "MWR",
  "use_demo_audio": true,
  "target_language": "hin",
  "preferred_provider": "local"
}
```
- **Response**:
```json
{
  "pipeline_status": "success",
  "dialect": "MWR",
  "raw_transcript": "म्हारो नाम राम है",
  "normalized_transcript": "म्हारो नाम राम है",
  "translation": {
    "translation": "मेरा नाम राम है"
  },
  "tts_output": {
    "audio_path": "data/processed/tts_output.mp3"
  },
  "latency_breakdown": {
    "asr_sec": 0.42,
    "mt_sec": 0.18,
    "tts_sec": 0.35,
    "total_sec": 0.95
  }
}
```

---

### 6. Cultural Proverb & Idiom Bank

#### `GET /api/proverbs?dialect=MWR&query=अन्न`
Retrieves curated proverbs with cultural glosses and translations.
- **Response**:
```json
{
  "proverbs": [
    {
      "id": "mwr_p001",
      "dialect": "MWR",
      "text": "अन्न रो आदर करनो सबसू बड़ो धरम है।",
      "hindi_meaning": "अन्न का आदर करना सबसे बड़ा धर्म है।",
      "english_translation": "Respecting food is the greatest duty.",
      "region": "Marwar"
    }
  ]
}
```

---

### 7. Evaluation & Benchmark Metrics

#### `GET /api/evaluation/summary`
Returns live and frozen benchmark metrics across all 6 dialects including WER, CER, BLEU, chrF++, and 95% bootstrap confidence intervals.

---

## Error Handling

Standard HTTP error responses:
- `400 Bad Request`: Missing or invalid input payload.
- `401 Unauthorized`: Missing or invalid `X-API-Key` header.
- `404 Not Found`: Dialect ID or resource not recognized.
- `422 Unprocessable Entity`: Schema validation failure.
- `500 Internal Server Error`: Pipeline processing exception.

```json
{
  "detail": "Invalid or missing X-API-Key header."
}
```
