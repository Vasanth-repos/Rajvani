"""
discover_datasets.py

Programmatically searches Hugging Face Hub and Kaggle for datasets relevant
to Marwari, Mewari, Dhundhari, Hadoti, Mewati, and Bagri, and writes a clean
markdown report. This is meant to be re-run periodically (dataset catalogs
change) rather than treated as a one-time result.

Requirements:
    pip install huggingface_hub kaggle

Kaggle credentials:
    Download kaggle.json from https://www.kaggle.com/settings > API > Create New Token
    mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

Usage:
    python discover_datasets.py --output data_discovery_report.md
"""

import argparse
import datetime
import sys

# Search terms: dialect names (with common alternate spellings) + ISO codes.
# See DATA_SOURCING_GUIDE.md Section 4 for where these codes come from and
# their caveats (some dialects don't have confirmed individual ISO codes).
SEARCH_TERMS = [
    "Rajasthani",
    "raj",              # ISO 639-3 macrolanguage code — noisy, expect false positives
    "Marwari",
    "mwr",
    "rwr",
    "Mewari",
    "mtr",
    "Dhundhari",
    "Dhundadi",
    "dhd",
    "Hadoti",
    "Harauti",
    "Mewati",
    "Bagri",
    "Shekhawati",
    "swv",
]

# Terms that are too short/common to search alone without heavy false-positive
# risk (e.g. "raj" matches unrelated things constantly) — flagged in the
# report rather than silently trusted.
NOISY_TERMS = {"raj", "mwr", "mtr", "dhd", "swv", "rwr"}


def search_huggingface(term: str):
    """Search HF Hub for datasets and models matching `term`. Returns a list
    of dicts: {id, kind, downloads, likes, url}."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed — run: pip install huggingface_hub", file=sys.stderr)
        return []

    api = HfApi()
    results = []
    try:
        for ds in api.list_datasets(search=term, limit=20):
            results.append({
                "id": ds.id,
                "kind": "dataset",
                "downloads": getattr(ds, "downloads", None),
                "likes": getattr(ds, "likes", None),
                "url": f"https://huggingface.co/datasets/{ds.id}",
            })
    except Exception as e:
        print(f"  [HF dataset search failed for '{term}']: {e}", file=sys.stderr)

    try:
        for m in api.list_models(search=term, limit=10):
            results.append({
                "id": m.id,
                "kind": "model",
                "downloads": getattr(m, "downloads", None),
                "likes": getattr(m, "likes", None),
                "url": f"https://huggingface.co/{m.id}",
            })
    except Exception as e:
        print(f"  [HF model search failed for '{term}']: {e}", file=sys.stderr)

    return results


def search_kaggle(term: str):
    """Search Kaggle for datasets matching `term`. Requires kaggle.json
    credentials to be configured. Returns a list of dicts: {ref, title, size, url}."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("kaggle package not installed — run: pip install kaggle", file=sys.stderr)
        return []

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"  [Kaggle auth failed — check ~/.kaggle/kaggle.json]: {e}", file=sys.stderr)
        return []

    results = []
    try:
        datasets = api.dataset_list(search=term)
        for d in datasets[:20]:
            results.append({
                "ref": d.ref,
                "title": d.title,
                "size": getattr(d, "size", "unknown"),
                "url": f"https://www.kaggle.com/datasets/{d.ref}",
            })
    except Exception as e:
        print(f"  [Kaggle dataset search failed for '{term}']: {e}", file=sys.stderr)

    return results


def build_report(hf_results: dict, kaggle_results: dict) -> str:
    lines = []
    lines.append("# Data Discovery Report")
    lines.append("")
    lines.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(
        "This report is a raw search dump, not a curated recommendation list — "
        "cross-check every hit against DATA_SOURCING_GUIDE.md's verified list "
        "before trusting relevance, and expect noise on short/ambiguous terms "
        "(flagged below)."
    )
    lines.append("")

    lines.append("## Hugging Face results")
    lines.append("")
    for term in SEARCH_TERMS:
        results = hf_results.get(term, [])
        noisy_flag = " ⚠️ noisy/ambiguous term — expect false positives" if term in NOISY_TERMS else ""
        lines.append(f"### `{term}`{noisy_flag}")
        if not results:
            lines.append("- No results (or search failed — see stderr log).")
        else:
            lines.append("| ID | Kind | Downloads | Likes | URL |")
            lines.append("|---|---|---|---|---|")
            for r in results:
                lines.append(f"| {r['id']} | {r['kind']} | {r['downloads']} | {r['likes']} | {r['url']} |")
        lines.append("")

    lines.append("## Kaggle results")
    lines.append("")
    for term in SEARCH_TERMS:
        results = kaggle_results.get(term, [])
        noisy_flag = " ⚠️ noisy/ambiguous term — expect false positives" if term in NOISY_TERMS else ""
        lines.append(f"### `{term}`{noisy_flag}")
        if not results:
            lines.append("- No results (or search failed/unauthenticated — see stderr log).")
        else:
            lines.append("| Ref | Title | Size | URL |")
            lines.append("|---|---|---|---|")
            for r in results:
                lines.append(f"| {r['ref']} | {r['title']} | {r['size']} | {r['url']} |")
        lines.append("")

    lines.append("## Next steps")
    lines.append("")
    lines.append(
        "1. Manually review every hit above for actual dialect coverage — a "
        "search match on a keyword does not confirm the data is real, "
        "consented, or usable.\n"
        "2. Check each candidate's license/usage terms individually before "
        "any ingestion into the training pipeline (see "
        "DATA_SOURCING_GUIDE.md Section 5).\n"
        "3. Re-run this script periodically — dataset catalogs change, and a "
        "clean miss today doesn't mean a clean miss next month."
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data_discovery_report.md",
                         help="Path to write the markdown report to.")
    parser.add_argument("--skip-kaggle", action="store_true",
                         help="Skip Kaggle search (e.g. if credentials aren't configured).")
    args = parser.parse_args()

    hf_results = {}
    kaggle_results = {}

    print("Searching Hugging Face...")
    for term in SEARCH_TERMS:
        print(f"  - {term}")
        hf_results[term] = search_huggingface(term)

    if not args.skip_kaggle:
        print("Searching Kaggle...")
        for term in SEARCH_TERMS:
            print(f"  - {term}")
            kaggle_results[term] = search_kaggle(term)
    else:
        print("Skipping Kaggle search (--skip-kaggle).")

    report = build_report(hf_results, kaggle_results)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
