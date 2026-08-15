# LDC-IL Corpus Access Request Specification (LDC_IL_ACCESS_REQUEST.md)

## 📌 Resource Identification

- **Dataset Title**: Gold Standard Rajasthani Raw Text Corpus
- **Issuing Body**: Linguistic Data Consortium for Indian Languages (LDC-IL), Central Institute of Indian Languages (CIIL), Department of Higher Education, Ministry of Education, Government of India
- **Official Access Portal**: `https://data.ldcil.org`
- **Estimated Volume**: ~1,200,000 words across 74 distinct titles
- **Data Format**: Annotated XML
- **Domains Covered**: 3 broad domains across 27 fine-grained sub-categories (Literature, Science & Technology, Mass Media / Social Sciences).

---

## 🎯 Target Dialectal Coverage

The corpus explicitly covers 8 Rajasthani speech varieties, including the 6 core project dialects:
1. **Marwari (`MWR`)**
2. **Mewari (`MTR`)**
3. **Dhundhari (`DHD`)**
4. **Hadoti / Harauti (`HDT`)**
5. **Mewati (`MWT`)**
6. **Bagri (`BGR`)**
*(Additional sub-varieties included: Wagdi, Malvi)*

---

## 📝 Step-by-Step Human Application Instructions

> [!IMPORTANT]
> This access request requires human registration through an Indian academic or institutional affiliate. The steps cannot and must not be bypassed programmatically.

1. **Account Registration**:
   - Navigate to `https://data.ldcil.org/register`.
   - Register using institutional credentials (academic university department or verified research entity).
2. **Resource Selection**:
   - In the LDC-IL Data Repository catalog, search for `"Rajasthani Raw Text Corpus"`.
   - Select the Gold Standard release covering western Indic non-scheduled dialects.
3. **Application Statement of Purpose (Template)**:
   > *"The requested Rajasthani text corpus will be utilized strictly for non-commercial computational linguistics research, specifically evaluating multi-dialect tokenization, dialectal grammar normalizers, and machine translation pivot adaptation for the six major dialects of Rajasthan (Marwari, Mewari, Dhundhari, Hadoti, Mewati, and Bagri). All data governance, attribution protocols, and non-distribution terms set forth by CIIL/LDC-IL will be strictly adhered to."*
4. **Post-Approval Action**:
   - Upon receiving the approval certificate and XML download bundle:
     1. Place raw XML files into `data/raw/ldc_il_rajasthani/`.
     2. Update `LICENSES.md` with the signed End User License Agreement (EULA) terms.
     3. Run `python data/normalize_orthography.py` to index the 1.2M words into the project dictionary.
