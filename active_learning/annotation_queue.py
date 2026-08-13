import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
QUEUE_DIR = ROOT_DIR / "active_learning"

def get_queue_file(dialect: str) -> Path:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    return QUEUE_DIR / f"queue_{dialect}.jsonl"

def push_to_queue(records: list, dialect: str, source_channel: str = "active_learning"):
    """
    Pushes records to the annotation queue with status 'pending'.
    """
    queue_file = get_queue_file(dialect)
    written_count = 0
    with open(queue_file, "a", encoding="utf-8") as f:
        for rec in records:
            rec_copy = dict(rec)
            rec_copy["status"] = "pending"
            rec_copy["source_channel"] = source_channel
            f.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")
            written_count += 1
    return written_count

def list_queue_items(dialect: str, status: str = None):
    queue_file = get_queue_file(dialect)
    if not queue_file.exists():
        return []
    items = []
    with open(queue_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if status is None or item.get("status") == status:
                    items.append(item)
    return items

if __name__ == "__main__":
    sample_records = [{"id": "item_1", "text_dialect": "सैंपल वाक्य 1"}, {"id": "item_2", "text_dialect": "सैंपल वाक्य 2"}]
    c = push_to_queue(sample_records, "mwr")
    print(f"Pushed {c} items to mwr annotation queue. Current pending: {len(list_queue_items('mwr', 'pending'))}")
