import pytest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from active_learning.score_pool import score_unlabeled_pool

def test_section3_active_learning_scoring():
    pool_recs = [
        {"id": "p1", "text_dialect_raw": "म्हारो नाम राम है।"},
        {"id": "p2", "text_dialect_raw": "सैंपल वाक्य rare words description"}
    ]
    val_recs = [{"id": "v1", "text_dialect": "म्हारो नाम राम है।"}]

    # Round 0 check (checkpoint base)
    scored_r0 = score_unlabeled_pool("mwr", pool_recs, val_recs, checkpoint="base")
    assert len(scored_r0) == 2
    assert scored_r0[0]["is_round_zero"] is True
    assert scored_r0[0]["novelty_score"] == 0.0

    # Fine-tuned checkpoint check
    scored_ft = score_unlabeled_pool("mwr", pool_recs, val_recs, checkpoint="checkpoints/asr/mwr/prod")
    assert scored_ft[0]["is_round_zero"] is False

    # Variant spelling novelty test: raw variant has low novelty against normalized validated text
    var_rec = [{"id": "p3", "text_dialect_raw": "महारो नाम राम है।"}]
    scored_var = score_unlabeled_pool("mwr", var_rec, val_recs, checkpoint="checkpoints/asr/mwr/prod")
    assert scored_var[0]["novelty_score"] < 0.2
