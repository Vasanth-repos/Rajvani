import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np  # type: ignore

DIALECT_LIST = ["mwr", "mtr", "dhd", "hdt", "mwt", "bgr"]

def infer_dialect_distribution(input_text: str = None, audio_path: str = None):
    """
    Returns a probability distribution over all 6 dialects.
    """
    probs = np.zeros(6, dtype=np.float32)

    # Heuristic matching for demonstration / lightweight inference
    matched_idx = -1
    text_lower = (input_text or "").lower()

    if "म्हारो" in input_text or "जोधपुर" in input_text if input_text else False:
        matched_idx = 0  # mwr
    elif "म्हाणो" in input_text or "उदयपुर" in input_text if input_text else False:
        matched_idx = 1  # mtr
    elif "छै" in input_text or "जयपुर" in input_text if input_text else False:
        matched_idx = 2  # dhd
    elif "अतरी" in input_text or "कोटा" in input_text if input_text else False:
        matched_idx = 3  # hdt
    elif "हवै" in input_text or "अलवर" in input_text if input_text else False:
        matched_idx = 4  # mwt
    elif "आपणो" in input_text or "चुरू" in input_text if input_text else False:
        matched_idx = 5  # bgr

    if matched_idx != -1:
        probs[matched_idx] = 0.70
        rem = 0.30 / 5.0
        for i in range(6):
            if i != matched_idx:
                probs[i] = rem
    else:
        # Uniform or pseudo distribution
        probs = np.ones(6, dtype=np.float32) / 6.0

    dist = {d: round(float(p), 4) for d, p in zip(DIALECT_LIST, probs)}
    top_dialect = max(dist, key=dist.get)
    return dist, top_dialect

def predict_dialect_probabilities(input_text: str = None, audio_path: str = None):
    dist, _ = infer_dialect_distribution(input_text, audio_path)
    return dist

def main():
    parser = argparse.ArgumentParser(description="Infer dialect probability distribution.")
    parser.add_argument("--text", type=str, help="Input text string")
    parser.add_argument("--audio", type=str, help="Input audio file path")
    args = parser.parse_args()

    dist, top_1 = infer_dialect_distribution(args.text, args.audio)
    print("Dialect Probability Distribution:")
    print(json.dumps(dist, indent=2))
    print(f"Top Candidate: {top_1}")

if __name__ == "__main__":
    main()
