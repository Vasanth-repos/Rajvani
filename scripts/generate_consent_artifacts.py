import argparse
import json
import os
import sys
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).parent.parent
PROTOCOL_PATH = ROOT_DIR / "docs" / "CONSENT_PROTOCOL.md"
VALIDATED_DIR = ROOT_DIR / "data" / "validated"
DOCS_DIR = ROOT_DIR / "docs"
DIALECTS_PATH = ROOT_DIR / "configs" / "dialects.yaml"

SUPPORTED_CONSENT_BASES = {"explicit_written", "explicit_verbal", "public_domain", "synthetic"}

def load_dialects():
    if DIALECTS_PATH.exists():
        with open(DIALECTS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("dialects", [])
    return []

def generate_dialect_consent_onepager(dialect_info):
    did = dialect_info["id"]
    dname = dialect_info["name"]
    regions = ", ".join(dialect_info.get("regions", []))

    content = f"""# Consent Protocol Summary - {dname} ({did})

**Region Coverage:** {regions}  
**Governing Platform:** BHASHINI / MeitY  

## Dialect Consent Terms ({dname})

1. **Internal Training Consent (`consent_basis`)**:
   - Explicit written/verbal agreement to train speech and translation AI models.
2. **Public Release Consent (`public_release_ok`)**:
   - Separate opt-in required for inclusion in public open-access benchmarks. Default is NO.
3. **Voice Synthesis / Cloning Consent (`voice_clone_ok`)**:
   - Separate opt-in required for TTS voice cloning. Default is NO.
4. **Withdrawal Rights**:
   - Right to withdraw consent at any time via consent-support@bhashini.gov.in.

---
*Derived from canonical protocol at docs/CONSENT_PROTOCOL.md*
"""
    output_path = DOCS_DIR / f"consent_{did}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path

def check_validated_consent_bases():
    """Verify all consent_basis values in validated data are supported by protocol."""
    found_bases = set()
    if VALIDATED_DIR.exists():
        for dpath in VALIDATED_DIR.glob("*"):
            if dpath.is_dir():
                for fpath in dpath.glob("*.jsonl"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                rec = json.loads(line)
                                cb = rec.get("consent_basis")
                                if cb:
                                    found_bases.add(cb)
    
    unsupported = found_bases - SUPPORTED_CONSENT_BASES
    if unsupported:
        print(f"Error: Found unsupported consent_basis types in validated data: {unsupported}", file=sys.stderr)
        sys.exit(1)
    print(f"Consent protocol audit passed. Found consent bases: {found_bases or 'None (empty validated data)'}")

def main():
    parser = argparse.ArgumentParser(description="Generate per-dialect consent artifacts & audit protocol coverage.")
    args = parser.parse_args()

    if not PROTOCOL_PATH.exists():
        print(f"Error: Base consent protocol missing at {PROTOCOL_PATH}", file=sys.stderr)
        sys.exit(1)

    dialects = load_dialects()
    generated = []
    for d in dialects:
        op = generate_dialect_consent_onepager(d)
        generated.append(op)

    check_validated_consent_bases()
    print(f"Successfully generated {len(generated)} per-dialect consent artifacts in docs/")

if __name__ == "__main__":
    main()
