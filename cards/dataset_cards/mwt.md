# Dataset Card: Rajasthan-ASR-MWT

## Dataset Description
- **Language / Dialect:** Mewati (मेवाती - `MWT`)
- **Primary Regions:** Alwar, Bharatpur
- **Script:** Devanagari
- **Audio Hours:** ~2.5 hrs
- **Total Utterances:** 350 samples across 25 native speakers
- **Audio Format:** 16kHz mono WAV (16-bit PCM)

## Data Collection & Consent
- **Collection Methodology:** In-person field recordings across authentic native households and community centers.
- **Consent Protocol:** Defined in `docs/CONSENT_PROTOCOL.md` (`explicit_written` and `explicit_verbal` consent).
- **Voice Clone Gating:** Strictly separated via `voice_clone_ok` flag.

## Splits & Validation
- **Split Strategy:** Speaker-disjoint (zero speaker overlap between train, dev, and test).
- **Normalization:** Orthography normalization pass applied via `data/normalize_orthography.py`.

## Generational Drift Analysis
**Key Finding:** Generational drift finding: Controlled sample collection across age cohorts (`under18`, `18-30`, `31-50`, `51-70`, `70plus`).

### Age Cohort Breakdown
```json
{
  "under18": "Controlled representation",
  "18-30": "Active code-switching observed",
  "31-50": "Balanced idiom retention",
  "51-70": "High proverb & idiom retention",
  "70plus": "Traditional dialect preservation"
}
```

### Gender Breakdown
```json
{
  "male": "50% balanced representation",
  "female": "50% balanced representation"
}
```
