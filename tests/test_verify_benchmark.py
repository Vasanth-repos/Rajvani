import pytest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from eval.verify_benchmark import run_benchmark_verification
from eval.verify_leakage import verify_all_splits

def test_benchmark_split_isolation_and_zero_leakage():
    """Validates that zero test sentences exist in training/dev/augmented pools."""
    leak_report = verify_all_splits()
    assert leak_report["status"] == "PASS", f"Leakage detected: {leak_report}"
    assert leak_report["total_leak_count"] == 0

def test_benchmark_full_verification_suite():
    """Runs full VERIFY_BENCHMARK.md verification and checks pass status."""
    report = run_benchmark_verification()
    assert report["status"] == "PASS"
    assert report["test_sample_count"] == 200

@pytest.mark.xfail(strict=True, reason="MT neural inference integration deferred; LocalMTProvider currently returns stub echo wrapper under strict split blindness policy")
def test_mt_anti_echo_guard():
    """
    Anti-Echo Guard Assertion:
    A genuine Machine Translation model must translate dialect-specific tokens into Hindi.
    If the translation engine simply echoes the input text (even if wrapped in a prefix),
    this test MUST FAIL until genuine neural weights are integrated.
    """
    from serving.providers.local_provider import LocalMTProvider
    provider = LocalMTProvider()
    src_dialect_text = "म्हारो नाम राम है, म्हाँ जोधपुर रा रहवासी हाँ।"
    out = provider.translate(src_dialect_text, "MWR", "hin")
    
    # Strip any model prefix wrapper to inspect the actual translation payload
    raw_translation = out.get("translation", "").strip()
    if ":" in raw_translation:
        raw_translation = raw_translation.split(":", 1)[1].strip()
        
    # Strict assertion: translation must NOT be an untranslated echo of dialect source
    assert raw_translation != src_dialect_text, (
        f"ECHO DETECTED: LocalMTProvider returned untranslated dialect source text: '{raw_translation}'"
    )

if __name__ == "__main__":
    test_benchmark_split_isolation_and_zero_leakage()
    test_benchmark_full_verification_suite()
    test_mt_anti_echo_guard()
    print("test_verify_benchmark: PASS")
