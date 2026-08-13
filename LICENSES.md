# Base Model Licenses & Commercial-Use Tracker (LICENSES.md)

This document tracks the licensing and commercial availability of all open source base models used within the pipeline.

| Model / Architecture | Task | License | Commercial Use | Notes |
|---|---|---|---|---|
| `openai/whisper-large-v3` | ASR | MIT | Yes | Production default ASR |
| `facebook/mms-1b-all` | ASR / Dialect-ID | CC-BY-NC 4.0 | Non-Commercial | Permissive research & baseline evaluation |
| `ai4bharat/indictrans2` | MT | MIT | Yes | Production default MT pivot model |
| `facebook/mms-tts` | TTS | CC-BY-NC 4.0 | Non-Commercial / Research | Production default TTS |
| `coqui/XTTS-v2` | TTS | CPML (Coqui Public Model License) | Non-Commercial Demo Only | Hackathon demo backend. Requires separate commercial licensing for deployment. `train_tts.py --backend xtts` emits runtime warning stderr log. |

## License Compliance Rules
- Production deployment defaults strictly to `facebook/mms-tts`.
- Usage of `coqui/XTTS-v2` triggers a mandatory runtime stderr warning:
  `[WARNING] CPML License Notice: coqui/XTTS-v2 is licensed under CPML for non-commercial demonstration only.`
