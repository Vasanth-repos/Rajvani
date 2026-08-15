try:
    import pytest
except ImportError:
    pytest = None

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from benchmark.publish_filter import apply_publish_filter

def test_section8_5_benchmark_publish_filter_and_k_anonymity():
    records, ex_consent, k_supp = apply_publish_filter("mwr")
    
    # speaker_id and district region strings MUST NOT match internal raw strings in published records
    for r in records:
        assert not r.get("speaker_id", "").startswith("raw_spk_")
        assert r.get("region") == "MWR"

if __name__ == "__main__":
    test_section8_5_benchmark_publish_filter_and_k_anonymity()
    print("test_section8_5: PASS")
