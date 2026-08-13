# Community Consent & Privacy Protocol (CONSENT_PROTOCOL.md)

**Target Languages/Dialects:** Marwari, Mewari, Dhundhari, Hadoti, Mewati, Bagri  
**Governing Platform:** BHASHINI / Ministry of Electronics and Information Technology (MeitY)  
**Version:** 1.0 (2026-08-13)

---

## 1. Information Provided to Participants

Before any speech or text collection begins, every contributor is informed of the following in their native dialect and Hindi:
- **Purpose**: The data collected will be used to build digital language translation, voice recognition (ASR), and voice synthesis (TTS) models to preserve Rajasthani cultural heritage and enable voice-first public digital governance services.
- **Organization**: The initiative is conducted under the guidance of BHASHINI (India's AI Language Mission).
- **Data Protection**: Personal identifying information (such as full names and exact home addresses) will be removed prior to dataset processing.

---

## 2. Granular Opt-In Questions & Consent Scope

Consent is collected through three distinct, independent questions:

### Question 1: Internal Model Training Consent (`consent_basis`)
> *"Do you consent to having your voice recording or text contribution used to train language recognition, translation, and speech AI models internal to BHASHINI?"*
- Options: `explicit_written`, `explicit_verbal`, `public_domain`, `synthetic`

### Question 2: Public Benchmark & Leaderboard Release (`public_release_ok`)
> *"Can this recording or text contribution appear in a public benchmark dataset that open research communities can download, beyond internal training?"*
- Default: **`false`** (Opt-in required).
- Explicit opt-in sets `public_release_ok: true`.

### Question 3: Voice Synthesis & Cloning Consent (`voice_clone_ok` - Audio Records Only)
> *"A computer program can learn the sound of your voice well enough to generate new sentences in your voice that you did not actually say — is that okay?"*
- Default: **`false`** (Opt-in required).
- Explicit opt-in sets `voice_clone_ok: true`.
- **Note**: Consenting to transcription/translation (Question 1) does **not** authorize voice cloning. Audio records with `voice_clone_ok: false` are eligible for ASR training but strictly excluded from TTS voice-model fine-tuning (`train_tts.py`).

---

## 3. Participant Opt-Out & Data Withdrawal Rights

Contributors retain the absolute right to withdraw consent at any time:
1. **Withdrawal Request**: Contact the project team via phone/SMS/email specifying record ID or speaker identity.
2. **System Enforcement**:
   - The record status is marked `validated: false` with a `withdrawn` tombstone.
   - `public_release_ok` and `voice_clone_ok` are immediately set to `false`.
   - The record ID is appended to [`docs/WITHDRAWN_FROM_PUBLIC_RELEASE.md`](file:///c:/Rajasthan_language_model/docs/WITHDRAWN_FROM_PUBLIC_RELEASE.md) to ensure exclusion from subsequent benchmark snapshots.

---

## 4. Data Ownership & Intellectual Property Terms

- **Raw Contributions**: Speakers retain community authorship over their raw folk speech and traditional proverbs.
- **AI Models & Derived Weights**: Model checkpoints trained on consented data are held in public interest by BHASHINI under open licenses for community benefit.

---

## 5. Contact Information

For inquiries, consent verification, or withdrawal requests:
- **Email**: consent-support@bhashini.gov.in
- **Phone**: 1800-XXX-BHASHINI (Toll-Free)
- **Languages Supported**: Marwari, Mewari, Dhundhari, Hadoti, Mewati, Bagri, Hindi, English
