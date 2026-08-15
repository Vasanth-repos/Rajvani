# Data Sourcing Guide — Marwari, Mewari, Dhundhari, Hadoti, Mewati, Bagri

**Purpose:** find every legitimate existing source of speech/text data for these six dialects before spending fieldwork budget collecting from scratch, and be able to explain — in an interview, or to a judge — exactly what exists, what doesn't, and why. The "what doesn't exist and why" part is deliberately included: knowing that a resource *isn't* out there, and why the obvious-looking candidates (AI4Bharat's flagship corpora) don't actually cover you, is a stronger signal of real understanding than a list of links would be on its own.

---

## 0. The one fact that reframes this whole search

**None of Marwari, Mewari, Dhundhari, Hadoti, Mewati, or Bagri are among India's 22 scheduled languages.** That single fact is why this is a genuinely hard data problem rather than a "just download it" problem, and it's worth being able to say plainly in an interview: the big, well-funded Indian-language corpora — AI4Bharat's IndicVoices (12,000 hours, 22 languages), IndicVoices-R (1,704 hours TTS, 22 languages), Shrutilipi (6,400 hours ASR, 12 languages), BPCC/Samanantar (parallel MT corpora) — **do not include these dialects at all**, because they were built against the scheduled-language list. Anyone who suggests "just fine-tune on IndicVoices" hasn't checked; anyone who explains *why* that doesn't work, and what to use instead, has.

What those AI4Bharat resources are still worth to you: **Hindi-side pivot data** (for back-translation and transfer learning, since these dialects are closely related to Hindi), and — more valuably — **their open-source collection tooling**. AI4Bharat's Kathbath/Karya (Android crowdsourcing framework) and Shoonya (annotation platform) are both open source and built exactly for the kind of large-scale, low-resource-language field collection this project needs. Reusing their tooling instead of building your own collection app from scratch is a legitimate, citable engineering decision.

---

## 1. Verified real sources (checked, not assumed)

### 1.1 Speech data

| Source | What it actually is | Coverage | Access |
|---|---|---|---|
| **`severo/speech-rj-hi`** (Hugging Face) — mirrored on the official Microsoft Download Center | Read-speech corpus: 98 participants from **Soda, Rajasthan**, each reading 30 stories, one sentence at a time — 426,873 recordings total (~58 male / 40 female speakers) | Rajasthani-inflected Hindi speech, single village — treat as a starting point for the region's phonetics, not a substitute for dialect-specific data | Open, downloadable directly from HF or Microsoft's site |
| **`ARTPARK-IISc/Vaani`** (Hugging Face) | A large multi-district Indian speech corpus collected district-by-district across India, including Rajasthan districts | Check the dataset's district/language metadata filter for Marwari-speaking districts (Jodhpur, Bikaner, Barmer, Nagaur) specifically — coverage varies by district, verify before assuming a given dialect is represented | Open on HF, standard `datasets` library load |
| **`Jarbas/tts_vc_Vaani_marwari_mwr_miro`** (Hugging Face) | A Marwari-specific subset/derivative of Vaani, prepared for TTS/voice-conversion use | Marwari only | Open on HF |
| **`ARTPARK-IISc/SraVaani-1.0`** (Hugging Face — a *model*, not raw data) | A speech model pretrained across 105 languages, fine-tuned for 65 Indian languages/dialects including Marwari and "Rajasthani" generically | Not a dataset — but worth evaluating as a pseudo-labeling tool: run it over unlabeled dialect audio you collect, use its output as a first-pass transcript for human correction, which is much faster than transcribing from scratch | Open on HF |

### 1.2 Text data

| Source | What it actually is | Coverage | Access |
|---|---|---|---|
| **LDC-IL "Gold Standard Rajasthani Raw Text Corpus"** (`data.ldcil.org`) | A ~1.2 million word raw text corpus, 74 titles, XML format, spanning 3 domains and 27 sub-categories, **explicitly covering Marwari, Mewari, Mewati, Dhundhari, Harauti (Hadoti), Bagri, Wagdi, and Malvi** | This is the single most directly relevant text resource found — it's purpose-built for exactly these dialects, by India's official Linguistic Data Consortium | **Likely requires registration/application through LDC-IL** (a government body under the Ministry of Education) — this is not an instant download like a HF dataset. Apply early; approval timelines for LDC-IL resources are not instant and this should not be a same-week dependency |
| **SIL International's "Sociolinguistic Survey of Selected Rajasthani Speech Varieties" (2012)** | An academic linguistic survey covering Marwari, Mewari, and other Rajasthani varieties — includes wordlists and sample texts, useful for orthography/dialect-boundary reference even where it isn't training-scale data | Academic reference quality, not bulk training data | Search for the published survey; typically available through SIL's own publication channels or academic databases |

### 1.3 What's genuinely NOT out there (say this plainly if asked)

- **Kaggle**: no dedicated Marwari/Mewari/Dhundhari/Hadoti/Mewati/Bagri dataset was found. General Hindi-language Kaggle datasets exist in abundance; dialect-specific ones for this group don't appear to. Don't claim otherwise — run the discovery script below periodically (Kaggle's catalog changes), but set the expectation honestly now.
- **AI4Bharat's flagship corpora** (IndicVoices, Shrutilipi, IndicVoices-R, BPCC/Samanantar, Sangraha): confirmed not to cover these dialects directly, per Section 0. Useful for tooling and Hindi-pivot data only.
- **Mozilla Common Voice**: no confirmed Rajasthani-dialect locale as of this writing — verify current status directly at commonvoice.mozilla.org/en/languages before assuming, since Common Voice adds new locales periodically.

---

## 2. How much data you actually need (gap analysis, not just "collect more")

Pulled from this project's own compute/budget planning — use this to know when you have *enough* to start training rather than collecting indefinitely:

| Task | Rough minimum to start LoRA fine-tuning | What the sources above realistically get you |
|---|---|---|
| ASR | 5–15 hours of validated dialect-specific audio | The Vaani/Jarbas Marwari subset may get you partway on Marwari specifically; the other five dialects likely still need real fieldwork — verify actual hours in each dataset's card before assuming coverage |
| MT | 2–5k parallel sentence pairs per dialect↔Hindi | The LDC-IL text corpus is your best lead here, once access is granted — 1.2M words across 8 varieties is a real foundation, but it's raw/monolingual text, not parallel pairs, so you'll still need to generate or manually create dialect↔Hindi alignments from it |
| TTS | 2–4 hours of clean single/multi-speaker audio per dialect | None of the sources above are studio-quality multi-speaker TTS-ready data for these specific dialects — this is very likely still a fieldwork-only requirement, consistent with the "no real dialect-specific TTS model confirmed running yet" gap already flagged in this project's review |

**The honest summary, in one line for an interview:** existing sources meaningfully help MT (via LDC-IL) and partially help ASR (via Vaani/Marwari-specific data), but TTS training data for these six dialects is essentially a fieldwork-only problem right now — no shortcut exists, and knowing that precisely (rather than assuming "surely something's on Hugging Face") is itself the finding worth presenting.

---

## 3. Automated discovery script

Manual search is fine for a one-time pass, but dataset catalogs change — this script re-runs the search programmatically across Hugging Face and Kaggle and produces a clean report, so "we systematically checked" is something you can demonstrate live, not just claim. See `discover_datasets.py` (companion file) for the full script; usage:

```bash
pip install huggingface_hub kaggle --quiet

# Kaggle API needs credentials first — download kaggle.json from
# https://www.kaggle.com/settings > API > Create New Token, then:
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

python discover_datasets.py --output data_discovery_report.md
```

This produces `data_discovery_report.md` — a table of every HF/Kaggle result matching dialect names or ISO codes, with a relevance flag, ready to paste into a status update or show live in an interview as evidence of a repeatable process, not a one-off manual search.

---

## 4. Language codes to search with (verified against Wikipedia/ISO 639-3, not assumed)

| Dialect | ISO 639-3 | Notes |
|---|---|---|
| Rajasthani (macrolanguage) | `raj` | Umbrella code — many datasets tag content at this level rather than per-dialect |
| Marwari | `mwr` (inclusive) / `rwr` (Marwari, India specifically) | Some sources use `mwr` loosely for "Rajasthani" generally — verify which sense a given source means |
| Dhundhari | `dhd` | Also spelled Dhundari/Dhundadi |
| Mewari | `mtr` | |
| Shekhawati | `swv` | A Marwari-family variety, sometimes conflated with Dhundhari — relevant if you expand coverage later |
| Merwari, Bahawali | `wry`, `mve` | Sub-varieties under the Marwari umbrella |
| Hadoti (Harauti), Mewati, Bagri | Not confirmed to have distinct individual ISO 639-3 codes as of this check | Search by name/region rather than a code for these three — some tools invent non-standard codes (e.g. `hdv`, `wtm`, `bgq` seen in one non-authoritative source); don't treat those as verified official codes without checking the SIL/Ethnologue registry directly |

Use both the ISO codes and the plain-English dialect names (and common alternate spellings: Harauti/Hadoti, Dhundari/Dhundhari/Jaipuri, Mewar/Mewari) when searching — dataset metadata on HF and Kaggle is inconsistently tagged, and a code-only search will miss real matches.

---

## 5. What still needs manual/legal follow-up before any of this touches your training pipeline

This ties directly into the consent and licensing work already established for this project — don't let "we found a dataset" skip the same bar real-fieldwork data has to clear:

- **LDC-IL corpus**: confirm the actual license/usage terms on grant of access — government linguistic corpora often have research-use-only or attribution requirements; get this in writing before it enters `LICENSES.md`.
- **Vaani / Jarbas HF datasets**: check each dataset card's license field individually — don't assume CC-BY or public domain without reading it, and don't assume all subsets of Vaani share one license.
- **SraVaani-1.0 model outputs used as pseudo-labels**: if you use this model to generate first-pass transcripts, tag those records `data_origin: model_assisted` (not `human_collected`) per this project's existing schema, and route them through human validation before they count as validated training data — a model-generated transcript is not consented human speech data and shouldn't be treated as equivalent in your data cards.

---

## 6. How to present this in an interview

The strongest framing isn't "I found six datasets" — it's the process: *"I started from the assumption that AI4Bharat's big corpora would cover this, verified they don't because these dialects aren't in India's 22 scheduled languages, then built a systematic search across Hugging Face, Kaggle, and India's official linguistic data consortium (LDC-IL) using both dialect names and ISO codes, found that text data has a real lead (LDC-IL) while TTS training data has essentially no shortcut and requires fieldwork, and wrote a script so that search is repeatable instead of a one-time manual pass."* That's a data-strategy narrative, not a link dump — and it's true, which is what makes it hold up under follow-up questions.
