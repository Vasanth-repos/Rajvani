import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
import numpy as np  # type: ignore

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.normalize_orthography import normalize_text

def compute_mock_text_embedding(text: str):
    """
    Computes a deterministic pseudo-embedding vector for text for environment portability.
    Uses character n-grams and hashing to produce a normalized 64-dim vector.
    """
    vec = np.zeros(64, dtype=np.float32)
    for i in range(len(text) - 1):
        gram = text[i:i+2]
        idx = hash(gram) % 64
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def calculate_uncertainty(text: str, checkpoint: str):
    """
    Computes sample uncertainty proxy (higher for longer/rarer/noisier strings).
    """
    if not text:
        return 0.0
    # Length and character diversity proxy for entropy
    char_entropy = len(set(text)) / (len(text) + 1e-5)
    length_penalty = math.log1p(len(text))
    raw_unc = char_entropy * length_penalty
    return min(1.0, raw_unc / 3.0)

def calculate_novelty(normalized_pool_text: str, validated_texts_normalized: list):
    """
    Computes novelty score as (1 - max_similarity to existing validated records).
    """
    if not validated_texts_normalized:
        return 1.0
    
    pool_emb = compute_mock_text_embedding(normalized_pool_text)
    max_sim = 0.0
    
    for val_text in validated_texts_normalized:
        val_emb = compute_mock_text_embedding(val_text)
        sim = float(np.dot(pool_emb, val_emb))
        if sim > max_sim:
            max_sim = sim
            
    novelty = max(0.0, 1.0 - max_sim)
    return novelty

def score_unlabeled_pool(dialect: str, pool_records: list, validated_records: list, checkpoint: str = "base"):
    is_round_zero = (checkpoint.lower() == "base" or not validated_records)
    
    # Pre-normalize validated texts in-memory
    val_texts_norm = [normalize_text(r.get("text_dialect", ""), dialect)[0] for r in validated_records if r.get("text_dialect")]
    
    scored_results = []
    
    for rec in pool_records:
        raw_text = rec.get("text_dialect_raw") or rec.get("text_dialect", "")
        # Normalize pool text in-memory for novelty comparison
        norm_text, _ = normalize_text(raw_text, dialect)
        
        unc = calculate_uncertainty(norm_text, checkpoint)
        
        if is_round_zero:
            nov = 0.0
            p_score = unc
        else:
            nov = calculate_novelty(norm_text, val_texts_norm)
            p_score = 0.6 * unc + 0.4 * nov
            
        scored_results.append({
            "id": rec.get("id"),
            "text_dialect_raw": raw_text,
            "text_dialect_norm": norm_text,
            "uncertainty_score": round(unc, 4),
            "novelty_score": round(nov, 4),
            "priority_score": round(p_score, 4),
            "is_round_zero": is_round_zero
        })
        
    scored_results.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored_results

def main():
    parser = argparse.ArgumentParser(description="Score unlabeled data pool for active learning prioritization.")
    parser.add_argument("--dialect", type=str, default="bgr", help="Dialect ID")
    parser.add_argument("--checkpoint", type=str, default="base", help="Model checkpoint path or 'base'")
    parser.add_argument("--output-csv", type=str, help="Output path for ranked CSV")
    args = parser.parse_args()

    dialect = args.dialect
    pool_dir = ROOT_DIR / "data" / "raw" / dialect
    val_dir = ROOT_DIR / "data" / "validated" / dialect
    
    pool_records = []
    if pool_dir.exists():
        for pfile in pool_dir.glob("*.jsonl"):
            with open(pfile, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        pool_records.append(json.loads(line))

    # Synthetic pool if empty
    if not pool_records:
        pool_records = [
            {"id": f"pool_{i}", "text_dialect": f"सैंपल पूल वाक्य {i} dialect text variant", "dialect": dialect}
            for i in range(1, 51)
        ]

    val_records = []
    if val_dir.exists():
        for vfile in val_dir.glob("*.jsonl"):
            with open(vfile, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        val_records.append(json.loads(line))

    results = score_unlabeled_pool(dialect, pool_records, val_records, args.checkpoint)

    out_csv = args.output_csv or (ROOT_DIR / "active_learning" / f"scored_{dialect}.csv")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text_dialect_raw", "text_dialect_norm", "uncertainty_score", "novelty_score", "priority_score", "is_round_zero"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Scored {len(results)} pool samples for dialect '{dialect}' (Round 0: {args.checkpoint=='base'}). Top priority score: {results[0]['priority_score'] if results else 0.0}")
    print(f"Saved ranked CSV to: {out_csv}")

if __name__ == "__main__":
    main()
