import json
from pathlib import Path
from typing import Dict, Any, List, Optional

PROVERB_BANK: List[Dict[str, Any]] = [
    # Marwari (MWR)
    {
        "id": "mwr_prv_001",
        "dialect": "MWR",
        "original_proverb": "अेक साधे सब सधै, सब साधे सब जाय।",
        "literal_meaning": "Fixing one fixes all, trying to fix all loses everything.",
        "figurative_meaning": "Focusing on the main priority resolves secondary issues; spreading effort thin causes failure.",
        "hindi_equivalent": "एक समय में एक ही कार्य पर ध्यान केंद्रित करना चाहिए।",
        "other_dialect_equivalents": {"MTR": "एक साधे सब सधै।", "DHD": "एक साधे सब सधै।"},
        "domain": "Wisdom",
        "source": "Field Collection - Jodhpur",
        "human_verified": True
    },
    {
        "id": "mwr_prv_002",
        "dialect": "MWR",
        "original_proverb": "जैसा अन्न वैसा मन।",
        "literal_meaning": "As the food is, so is the mind.",
        "figurative_meaning": "The purity of one's food and thoughts shapes their character.",
        "hindi_equivalent": "जैसा आहार वैसा विचार।",
        "other_dialect_equivalents": {"MTR": "जैसो अन्न वैसो मन।"},
        "domain": "Ethics",
        "source": "Field Collection - Bikaner",
        "human_verified": True
    },
    {
        "id": "mwr_prv_003",
        "dialect": "MWR",
        "original_proverb": "दूध रो जल्यो छाछ ने फूंक फूंक पीवे।",
        "literal_meaning": "One scalded by hot milk blows even on buttermilk before drinking.",
        "figurative_meaning": "Once bitten, twice shy; extreme caution after a bad experience.",
        "hindi_equivalent": "दूध का जला छाछ भी फूंक-फूंक कर पीता है।",
        "other_dialect_equivalents": {"MTR": "दूध रो जल्यो छाछ फूंक पीवे।"},
        "domain": "Caution",
        "source": "Field Collection - Barmer",
        "human_verified": True
    },
    {
        "id": "mwr_prv_004",
        "dialect": "MWR",
        "original_proverb": "सांच को आंच कोनी।",
        "literal_meaning": "Truth has no fear of heat.",
        "figurative_meaning": "The truthful fear no test or false accusation.",
        "hindi_equivalent": "साँच को आँच नहीं।",
        "other_dialect_equivalents": {"BGR": "साँच रो बेड़ो पार।"},
        "domain": "Truth",
        "source": "Field Collection - Nagaur",
        "human_verified": True
    },

    # Mewari (MTR)
    {
        "id": "mtr_prv_001",
        "dialect": "MTR",
        "original_proverb": "घर रो जोगी जोगणा, आन गाँव रो सिद्ध।",
        "literal_meaning": "A yogi of one's home is just a beggar, while an outsider yogi is a saint.",
        "figurative_meaning": "Familiarity breeds contempt; local talent is often underestimated until acknowledged externally.",
        "hindi_equivalent": "घर का मोगी जोगना, आन गाँव का सिद्ध।",
        "other_dialect_equivalents": {"MWR": "घर रो जोगी जोगणा, बाहर रो सिद्ध।"},
        "domain": "Social Perception",
        "source": "Field Collection - Udaipur",
        "human_verified": True
    },
    {
        "id": "mtr_prv_002",
        "dialect": "MTR",
        "original_proverb": "आपणा हाथ जगन्नाथ।",
        "literal_meaning": "One's own hands are Lord Jagannath.",
        "figurative_meaning": "Self-reliance and personal effort yield the best outcome.",
        "hindi_equivalent": "अपना हाथ जगन्नाथ (आत्मनिर्भरता)।",
        "other_dialect_equivalents": {"BGR": "आपणo काम आपणे हाथ।"},
        "domain": "Self-Reliance",
        "source": "Field Collection - Chittorgarh",
        "human_verified": True
    },

    # Dhundhari (DHD)
    {
        "id": "dhd_prv_001",
        "dialect": "DHD",
        "original_proverb": "हाथ कंगन को आरसी क्या।",
        "literal_meaning": "Why does a wrist bangle need a mirror to be seen?",
        "figurative_meaning": "Self-evident truths require no external proof.",
        "hindi_equivalent": "हाथ कंगन को आरसी क्या, पढ़े लिखे को फारसी क्या।",
        "other_dialect_equivalents": {"MWR": "हाथ कंगन ने आरसी कांई।"},
        "domain": "Truth",
        "source": "Field Collection - Jaipur",
        "human_verified": True
    },
    {
        "id": "dhd_prv_002",
        "dialect": "DHD",
        "original_proverb": "बोवै पेड़ बबूल को तो आम कहाँ से होय।",
        "literal_meaning": "If you plant a babool thorn tree, how will you get mangoes?",
        "figurative_meaning": "Harmful or bad deeds never produce sweet or good results.",
        "hindi_equivalent": "बोया पेड़ बबूल का तो आम कहाँ से होय।",
        "other_dialect_equivalents": {"MWT": "बोया पेड़ बबूल का तो आम कहाँ से होय।"},
        "domain": "Ethics",
        "source": "Field Collection - Tonk",
        "human_verified": True
    },

    # Hadoti (HDT)
    {
        "id": "hdt_prv_001",
        "dialect": "HDT",
        "original_proverb": "नाच न जाणै आँगण टेढ़ो।",
        "literal_meaning": "One who doesn't know how to dance blames the courtyard for being crooked.",
        "figurative_meaning": "Incompetent people blame their tools or environment for failure.",
        "hindi_equivalent": "नाच न जाने आंगन टेढ़ा।",
        "other_dialect_equivalents": {"MWR": "नाच न जाणै आँगण टेढ़ो।"},
        "domain": "Responsibility",
        "source": "Field Collection - Kota",
        "human_verified": True
    },
    {
        "id": "hdt_prv_002",
        "dialect": "HDT",
        "original_proverb": "पानी पहल्यां पाल बाँधणी।",
        "literal_meaning": "Building the embankment before the water floods.",
        "figurative_meaning": "Preparation and foresight before difficulty arrives.",
        "hindi_equivalent": "विपत्ति आने से पहले ही पूर्व तैयारी करना।",
        "other_dialect_equivalents": {"MWR": "पानी सूं पहल्यां पाल।"},
        "domain": "Foresight",
        "source": "Field Collection - Bundi",
        "human_verified": True
    },

    # Mewati (MWT)
    {
        "id": "mwt_prv_001",
        "dialect": "MWT",
        "original_proverb": "दूर का ढोल सुहावना लागै।",
        "literal_meaning": "Drums played at a distance sound pleasant.",
        "figurative_meaning": "Things appear more attractive from a distance than when examined closely.",
        "hindi_equivalent": "दूर के ढोल सुहावने लगते हैं।",
        "other_dialect_equivalents": {"MWR": "दूरा रा ढोल सुहावणा।"},
        "domain": "Illusion",
        "source": "Field Collection - Alwar",
        "human_verified": True
    },

    # Bagri (BGR)
    {
        "id": "bgr_prv_001",
        "dialect": "BGR",
        "original_proverb": "जैसी करणी वैसी भरणी।",
        "literal_meaning": "As you do, so shall you reap.",
        "figurative_meaning": "Actions have direct consequences.",
        "hindi_equivalent": "जैसा करोगे वैसा भरोगे।",
        "other_dialect_equivalents": {"MWR": "जैड़ा करम वैड़ा फल।"},
        "domain": "Ethics",
        "source": "Field Collection - Ganganagar",
        "human_verified": True
    }
]

def list_proverbs(dialect_filter: Optional[str] = None, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns proverbs, optionally filtered by dialect and domain."""
    pool = PROVERB_BANK
    if dialect_filter and dialect_filter.upper() != "ALL":
        did = dialect_filter.upper()
        pool = [p for p in pool if p["dialect"] == did]
    if domain_filter and domain_filter.upper() != "ALL":
        dom = domain_filter.lower()
        pool = [p for p in pool if dom in p.get("domain", "").lower()]
    return pool

def search_proverbs(query: str, dialect_filter: Optional[str] = None, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Searches proverbs by matching text against original, literal, or figurative meanings."""
    q = (query or "").lower().strip()
    pool = list_proverbs(dialect_filter, domain_filter)
    if not q:
        return pool
    
    results = []
    for p in pool:
        if (q in p["original_proverb"].lower() or 
            q in p["literal_meaning"].lower() or 
            q in p["figurative_meaning"].lower() or 
            q in p["hindi_equivalent"].lower() or
            q in p.get("domain", "").lower()):
            results.append(p)
    return results

def detect_cultural_proverb(text: str, dialect_id: str) -> Optional[Dict[str, Any]]:
    """Detects if input text contains a known proverb and returns its cultural intended translation."""
    norm_text = text.strip()
    for p in PROVERB_BANK:
        if p["original_proverb"] in norm_text or norm_text in p["original_proverb"]:
            return p
    return None
