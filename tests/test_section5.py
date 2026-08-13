import pytest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dialect_id.infer import infer_dialect_distribution
from codeswitch.tagger import tag_code_switching
from linguistic_artifacts.collect_idioms import collect_idiom_entry
from linguistic_artifacts.idiom_mt_eval import evaluate_idiom_mt

def test_section5_dialect_id_and_codeswitching():
    dist, top_1 = infer_dialect_distribution("म्हारो नाम राम है।")
    assert top_1 == "mwr"
    assert len(dist) == 6

    is_cs, spans = tag_code_switching("म्हारो नाम Ram है और school जाना है।")
    assert is_cs is True
    assert len(spans) >= 2

def test_section5_idiom_bank_intake_and_eval():
    entry = collect_idiom_entry(
        dialect="mwr",
        raw_idiom="महारो खेत सोनो उगले छै",
        literal_gloss="My field vomits gold",
        intended_hindi="खेत में बहुत अच्छी फसल होना",
        intended_english="Rich harvest",
        register="proverb",
        usage_context="Harvest",
        consent_basis="explicit_written",
        public_release_ok=False
    )

    # Parity check: idiom_dialect holds normalized form, idiom_dialect_raw holds raw form
    assert entry["idiom_dialect"] == "म्हारो खेत सोनो उगले छै"
    assert entry["idiom_dialect_raw"] == "महारो खेत सोनो उगले छै"
    assert entry["public_release_ok"] is False

    eval_stats = evaluate_idiom_mt("mwr")
    assert eval_stats["total"] >= 100
    assert eval_stats["accuracy_pct"] > 50.0
