from pathlib import Path

ROOT = Path(__file__).parent.parent
dialects = ['mwr', 'mtr', 'dhd', 'hdt', 'mwt', 'bgr']

for d in dialects:
    p = ROOT / 'data' / 'splits' / d / 'test.jsonl'
    lines = len(p.read_text(encoding='utf-8').strip().split('\n')) if p.exists() else 0
    print(f"{d}: full test.jsonl = {lines} lines")
