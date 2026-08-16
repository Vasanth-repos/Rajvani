import argparse
import json
import sys
from pathlib import Path
import numpy as np  # type: ignore

def calculate_mos(ratings: list):
    """
    Computes mean opinion score and 95% confidence interval.
    """
    if not ratings:
        return 0.0, 0.0
    arr = np.array(ratings, dtype=np.float32)
    mean_score = float(np.mean(arr))
    if len(arr) < 2:
        return mean_score, 0.0
    std_err = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    ci95 = 1.96 * std_err
    return round(mean_score, 2), round(ci95, 2)

def main():
    parser = argparse.ArgumentParser(description="Aggregate TTS MOS human naturalness ratings.")
    parser.add_argument("--dialect", type=str, default="mwr", help="Dialect ID")
    args = parser.parse_args()

    sample_ratings = [4.0, 4.5, 4.0, 5.0, 3.5, 4.2, 4.0, 4.8, 4.1, 4.3]
    mean, ci = calculate_mos(sample_ratings)
    print(f"TTS MOS Score for Dialect '{args.dialect}': {mean} ± {ci} (N={len(sample_ratings)})")

if __name__ == "__main__":
    main()
