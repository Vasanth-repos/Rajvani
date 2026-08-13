import argparse
import json
import re
import sys
from pathlib import Path

ENGLISH_REGEX = re.compile(r'\b[a-zA-Z]+\b')
HINDI_REGEX = re.compile(r'[\u0900-\u097F]+')

def tag_code_switching(text: str):
    """
    Detects code-switched spans in Hindi/English/Dialect text.
    Returns: (is_code_switched, cs_spans)
    """
    if not text:
        return False, []

    spans = []
    words = text.split()
    current_pos = 0

    eng_count = 0
    hin_count = 0

    for w in words:
        start = text.find(w, current_pos)
        end = start + len(w)
        current_pos = end

        if ENGLISH_REGEX.match(w):
            spans.append({"start": start, "end": end, "lang": "eng"})
            eng_count += 1
        elif HINDI_REGEX.match(w) and ("स्कूल" in w or "अस्पताल" in w or "बस" in w or "डॉक्टर" in w):
            spans.append({"start": start, "end": end, "lang": "hin"})
            hin_count += 1

    is_cs = len(spans) > 0 and (eng_count > 0 or hin_count > 0)
    return is_cs, spans

def process_record_codeswitching(record: dict):
    text = record.get("text_dialect", "")
    is_cs, spans = tag_code_switching(text)
    record["is_code_switched"] = is_cs
    record["cs_spans"] = spans
    return record

def main():
    parser = argparse.ArgumentParser(description="Tag code-switching spans in text.")
    parser.add_argument("--text", type=str, default="म्हारो नाम Ram है और म्हाने doctor के पास जाना है।", help="Text to tag")
    args = parser.parse_args()

    is_cs, spans = tag_code_switching(args.text)
    print(f"Text: '{args.text}'")
    print(f"Is Code-Switched: {is_cs}")
    print(f"Spans: {json.dumps(spans, indent=2)}")

if __name__ == "__main__":
    main()
