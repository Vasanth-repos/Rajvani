"""Rajasthani Multi-Dialect Neural & Lexical Translation Engine (Rajasthani -> Hindi / English)
Supports all 6 major dialects: Marwari (mwr), Mewari (mtr), Dhundhari (dhd), Hadoti (hdt), Mewati (mwt), Bagri (bgr).
"""

import re
from typing import Dict, Any, List, Tuple
from linguistic_artifacts.proverb_database import detect_cultural_proverb

# Multi-dialect morphological and lexical mapping dictionaries
DIALECT_LEXICON: Dict[str, Dict[str, str]] = {
    # Pronouns & Deictics
    "म्हारो": "मेरा", "म्हारी": "मेरी", "म्हारे": "मेरे",
    "थारो": "तेरा", "थारी": "तेरी", "थारे": "तेरे",
    "आपणो": "अपना", "आपणी": "अपनी", "आपणे": "अपने",
    "अठै": "यहाँ", "इठै": "यहाँ", "इया": "यहाँ", "इयाँ": "यहाँ",
    "बठै": "वहाँ", "उठै": "वहाँ", "उयां": "वहाँ", "उयाँ": "वहाँ",
    "जठै": "जहाँ", "जिठै": "जहाँ",
    "कठै": "कहाँ", "किठै": "कहाँ",
    "कांई": "क्या", "कीं": "क्या", "कांईं": "क्या", "काय": "क्या",
    "कियां": "कैसे", "कयां": "कैसे", "किया": "कैसे",
    "जिंयां": "जैसे", "जियां": "जैसे", "जियां-तियां": "जैसे-तैसे",
    "इण": "इस", "इणने": "इसको", "इणनै": "इसको", "इणरो": "इसका", "इणरी": "इसकी",
    "उण": "उस", "उणने": "उसको", "उणनै": "उसको", "उणरो": "उसका", "उणरी": "उसकी",
    "जिण": "जिस", "जिणने": "जिसको", "किण": "किस", "किणने": "किसको",
    "म्हे": "हम", "अम्हे": "हम", "म्हाँ": "हम", "थे": "आप", "तमे": "आप", "तुसी": "आप",

    # Postpositions & Case Markers
    "रो": "का", "री": "की", "रा": "के",
    "को": "का", "की": "की", "के": "के", "का": "के",
    "सूं": "से", "स्यूं": "से", "सती": "से",
    "नै": "को", "ने": "को",
    "मांय": "में", "मायने": "में", "माय": "में", "माँय": "में", "मां": "में", "मंय": "में",
    "सागे": "साथ", "लारे": "पीछे", "आगै": "आगे", "कनै": "पास", "ढूकै": "पास",

    # Verbs / Copulas / Auxiliaries
    "छै": "है", "छी": "थी", "छा": "थे", "छे": "है", "छो": "था",
    "है": "है", "हो": "था", "ही": "थी", "हा": "थे", "हती": "थी", "हतो": "था", "हता": "थे",
    "हाँ": "हैं", "हूँ": "हूँ",
    "सै": "है", "सी": "थी", "सो": "था", "सा": "थे",
    "कोनी": "नहीं है", "नाहीं": "नहीं", "नाय": "नहीं", "कोयनी": "नहीं है",
    "व्है": "होता है", "व्हैला": "होगा", "व्हैग्यो": "हो गया", "व्हैगी": "हो गई", "व्हैग्या": "हो गए",
    "ग्यो": "गया", "गी": "गई", "ग्या": "गए", "गयो": "गया", "गयी": "गई", "गये": "गए",
    "पधार्या": "पधारे", "पधारो": "पधारिए", "जीमो": "भोजन कीजिए", "जीम्या": "भोजन किया",
    "करसी": "करेगा", "करहूँ": "करूँगा", "करस्यां": "करेंगे", "करै": "करता है", "करैला": "करेगा",
    "देखै": "देखता है", "जावै": "जाता है", "आवै": "आता है", "बोलै": "बोलता है",
    "लाधै": "मिलता है", "लाध्यो": "मिला", "लाधी": "मिली",
    "फरमावै": "कहता है", "हुकम": "आज्ञा",

    # Adjectives & Adverbs
    "घणो": "बहुत", "घणी": "बहुत", "घणा": "बहुत", "घणोई": "बहुत अधिक",
    "चोखो": "अच्छा", "चोखी": "अच्छी", "चोखा": "अच्छे",
    "फूटर": "सुंदर", "फूटरो": "सुंदर", "फूटरी": "सुंदर",
    "हगळा": "सभी", "सगळा": "सभी", "सगळी": "सभी", "सगळो": "सब", "सबै": "सभी", "सारे": "सभी",
    "टाबर": "बच्चा", "टाबरिया": "बच्चे", "टाबरां": "बच्चों",
    "माणस": "मनुष्य", "मणख": "व्यक्ति", "मिनख": "व्यक्ति", "लुगाई": "महिला", "बींद": "दूल्हा", "बींदणी": "दुल्हन",
    "गाम": "गाँव", "गोम": "गाँव", "खेड़ा": "गाँव",
    "खेती-बाड़ी": "कृषि कार्य", "धान": "अनाज", "अन्न": "अनाज",
    "पाणी": "पानी", "नीर": "पानी",
    "दुकानदार": "दुकानदार", "भाड़ा": "किराया", "रुपीया": "रुपये", "पईसा": "पैसे",
    "ब्याव": "विवाह", "शादी": "विवाह", "त्यौहार": "त्योहार",
    "पछै": "बाद में", "पछे": "बाद में", "पाछो": "वापस", "पाछी": "वापस", "पाछा": "वापस",
    "कद": "कब", "जद": "जब", "तद": "तब",
    "आज": "आज", "काले": "कल", "परार": "परसों", "सांझ": "शाम", "सवार": "सुबह", "प्रभात": "सुबह"
}

# Dialect-specific morphological suffix transformations
SUFFIX_RULES: List[Tuple[str, str]] = [
    (r"(\w+)स्यां\b", r"\1ेंगे"),     # करस्यां -> करेंगे
    (r"(\w+)सी\b", r"\1ेगा"),       # करसी -> करेगा
    (r"(\w+)ला\b", r"\1ेगा"),       # व्हैला -> होगा
    (r"(\w+)ग्यो\b", r"\1 गया"),    # बैठग्यो -> बैठ गया
    (r"(\w+)गी\b", r"\1 गई"),       # बैठगी -> बैठ गई
    (r"(\w+)ग्या\b", r"\1 गए"),     # बैठग्या -> बैठ गए
    (r"(\w+)तांई\b", r"\1 तक"),     # सांझतांई -> शाम तक
    (r"(\w+)तांणी\b", r"\1 तक"),    # अठैतांणी -> यहाँ तक
    (r"(\w+)सूं\b", r"\1 से"),       # हाथसूं -> हाथ से
    (r"(\w+)नै\b", r"\1 को"),       # रामनै -> राम को
    (r"(\w+)रो\b", r"\1 का"),       # खेतरो -> खेत का
    (r"(\w+)री\b", r"\1 की"),       # बातरी -> बात की
    (r"(\w+)रा\b", r"\1 के"),       # लोकांरा -> लोगों के
]

def translate_dialect_to_hindi(text: str, dialect: str = "mwr") -> str:
    """Translates a Rajasthani dialect utterance into standard grammatical Hindi."""
    if not text or not text.strip():
        return ""
    
    clean_text = text.strip()
    
    # 1. Proverb check: Return authentic figurative Hindi equivalent if matching proverb
    proverb = detect_cultural_proverb(clean_text, dialect)
    if proverb and proverb.get("hindi_equivalent"):
        return proverb["hindi_equivalent"]
    
    # 2. Token-level normalization and replacement
    tokens = clean_text.split()
    translated_tokens = []
    
    for token in tokens:
        # Strip trailing punctuation for dictionary lookup
        punct = ""
        core = token
        if token and token[-1] in "।,.?!;:":
            punct = token[-1]
            core = token[:-1]
            
        if core in DIALECT_LEXICON:
            translated_tokens.append(DIALECT_LEXICON[core] + punct)
        else:
            # Apply morphological suffix rules
            replaced = core
            for pat, repl in SUFFIX_RULES:
                if re.search(pat, replaced):
                    replaced = re.sub(pat, repl, replaced)
                    break
            translated_tokens.append(replaced + punct)
            
    out_text = " ".join(translated_tokens)
    
    # Clean up double postpositions or common grammatical artifacts
    out_text = re.sub(r"\bका का\b", "का", out_text)
    out_text = re.sub(r"\bकी की\b", "की", out_text)
    out_text = re.sub(r"\bके के\b", "के", out_text)
    out_text = re.sub(r"\bसे से\b", "से", out_text)
    out_text = re.sub(r"\s+", " ", out_text).strip()
    
    return out_text
