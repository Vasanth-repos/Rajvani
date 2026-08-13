import argparse
import os
import sys
import yaml
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "orthography"
_RULES_CACHE = {}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def get_orthography_rules(dialect: str):
    if dialect in _RULES_CACHE:
        return _RULES_CACHE[dialect]
    rule_file = CONFIGS_DIR / f"{dialect}.yaml"
    if not rule_file.exists():
        # Fallback default rule if file not found
        rules = {"version": 1, "variant_mappings": {}, "diacritic_rules": {}}
    else:
        with open(rule_file, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)
    _RULES_CACHE[dialect] = rules
    return rules

def normalize_text(text: str, dialect: str):
    """
    Idempotently normalizes raw dialect text based on orthography rules for the dialect.
    Returns tuple: (normalized_text, orthography_review_flag)
    """
    if not text:
        return text, False

    rules = get_orthography_rules(dialect)
    variant_mappings = rules.get("variant_mappings", {})
    
    # Strip zero width joiners if specified
    diacritic_rules = rules.get("diacritic_rules", {})
    normalized = text
    if diacritic_rules.get("strip_zero_width_joiners", False):
        normalized = normalized.replace("\u200d", "").replace("\u200c", "")

    # Apply variant replacements
    words = normalized.split()
    normalized_words = []
    review_flag = False

    for word in words:
        if word in variant_mappings:
            normalized_words.append(variant_mappings[word])
        else:
            normalized_words.append(word)
            # Flag for review if contains unresolved non-standard characters or rare patterns
            if "?" in word or "!" in word and len(word) > 10:
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
        # Idempotence test demonstration
        sample_variants = ["महारो नाम राम है।", "महारो नाम राम है।"]
        print(f"Running orthography normalization test for dialect '{dialect}' (v{rules.get('version', 1)}):")
        for s in sample_variants:
            norm1, rev1 = normalize_text(s, dialect)
            norm2, rev2 = normalize_text(norm1, dialect)
            assert norm1 == norm2, f"Idempotence check failed for '{s}'!"
            print(f"  Raw: '{s}' -> Normalized: '{norm1}' (Idempotent: Yes, Review Flag: {rev1})")

if __name__ == "__main__":
    main()
