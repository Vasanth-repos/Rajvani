import argparse
import os
import sys
import yaml
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "orthography"
_RULES_CACHE = {}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Expanded baseline vocabulary mapping dictionary across Rajasthani dialects
GLOBAL_VOCABULARY_MAP = {
    "mwr": {
        "महारो": "म्हारो",
        "महोरा": "म्हारो",
        "कोनि": "कोनी",
        "कोण": "कुण",
        "काय": "कांई",
        "काईं": "कांई",
        "अेक": "एक",
        "आणो": "आवणी",
        "चोखो": "चोखो"
    },
    "mtr": {
        "महाणो": "म्हाणो",
        "म्हारो": "म्हाणो",
        "वेई": "वेइ",
        "गयो": "गियो",
        "आवो": "आवजो"
    },
    "dhd": {
        "छे": "छै",
        "छैन": "छै",
        "अटै": "अठै",
        "जटै": "जठै",
        "कठै": "कठै"
    },
    "hdt": {
        "बरसयो": "बरस रह्यो",
        "रहो": "रह्यो",
        "मे": "मेह",
        "अतरी": "अतरी"
    },
    "mwt": {
        "गांवा": "गांवां",
        "करै": "करैं",
        "लागे": "लागै",
        "हवे": "हवै"
    },
    "bgr": {
        "आपणो": "आपणo",
        "होइ": "हो",
        "घरा": "घरां",
        "चालो": "चालो"
    }
}

def get_orthography_rules(dialect: str):
    d_code = (dialect or "mwr").lower()
    if d_code in _RULES_CACHE:
        return _RULES_CACHE[d_code]
    rule_file = CONFIGS_DIR / f"{d_code}.yaml"
    if not rule_file.exists():
        rules = {
            "version": 1,
            "variant_mappings": GLOBAL_VOCABULARY_MAP.get(d_code, {}),
            "diacritic_rules": {"canonicalize_nasalization": True, "strip_zero_width_joiners": True}
        }
    else:
        with open(rule_file, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f) or {}
            # Merge global vocabulary defaults
            vmap = GLOBAL_VOCABULARY_MAP.get(d_code, {})
            existing = rules.get("variant_mappings", {}) or {}
            vmap.update(existing)
            rules["variant_mappings"] = vmap
            
    _RULES_CACHE[d_code] = rules
    return rules

def normalize_text(text: str, dialect: str):
    """
    Idempotently normalizes raw dialect text based on orthography rules for the dialect.
    Returns tuple: (normalized_text, orthography_review_flag)
    """
    if not text:
        return text, False

    d_code = (dialect or "mwr").lower().split()[0]
    rules = get_orthography_rules(d_code)
    variant_mappings = rules.get("variant_mappings", {})
    
    diacritic_rules = rules.get("diacritic_rules", {})
    normalized = text
    if diacritic_rules.get("strip_zero_width_joiners", True):
        normalized = normalized.replace("\u200d", "").replace("\u200c", "")

    # Apply variant vocabulary replacements
    words = normalized.split()
    normalized_words = []
    review_flag = False

    for word in words:
        clean_word = word.strip(".,!?।\"'")
        if clean_word in variant_mappings:
            replaced = variant_mappings[clean_word]
            normalized_words.append(word.replace(clean_word, replaced))
        elif word in variant_mappings:
            normalized_words.append(variant_mappings[word])
        else:
            normalized_words.append(word)
            if "?" in word or ("!" in word and len(word) > 10):
                review_flag = True

    normalized_text = " ".join(normalized_words)
    return normalized_text, review_flag

def process_record(record: dict):
    dialect = record.get("dialect", "mwr")
    raw_text = record.get("text_dialect_raw") or record.get("text_dialect", "")
    
    normalized, review_flag = normalize_text(raw_text, dialect)
    record["text_dialect_raw"] = raw_text
    record["text_dialect"] = normalized
    record["orthography_review"] = review_flag or record.get("orthography_review", False)
    return record

def main():
    parser = argparse.ArgumentParser(description="Run orthography normalization pass.")
    parser.add_argument("--dialect", type=str, default="mwr", help="Dialect to normalize")
    parser.add_argument("--input-text", type=str, help="Input text string to normalize")
    args = parser.parse_args()

    dialect = args.dialect
    rules = get_orthography_rules(dialect)

    if args.input_text:
        normalized, review = normalize_text(args.input_text, dialect)
        print(f"Original: {args.input_text}")
        print(f"Normalized: {normalized}")
        print(f"Orthography Review Flag: {review}")
    else:
        sample_variants = ["महारो नाम राम है।", "महारो नाम राम है।"]
        print(f"Running orthography normalization test for dialect '{dialect}' (v{rules.get('version', 1)}):")
        for s in sample_variants:
            norm1, rev1 = normalize_text(s, dialect)
            norm2, rev2 = normalize_text(norm1, dialect)
            assert norm1 == norm2, f"Idempotence check failed for '{s}'!"
            print(f"  Raw: '{s}' -> Normalized: '{norm1}' (Idempotent: Yes, Review Flag: {rev1})")

if __name__ == "__main__":
    main()
