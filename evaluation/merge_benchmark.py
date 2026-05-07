"""
Merge one or more benchmark entries from a patch JSON into an existing result JSON.

Usage:
    python evaluation/merge_benchmark.py \
        --base   results/sqr_summit.json \
        --patch  results/clifford_sqr_temp.json \
        --out    results/sqr_summit.json        # can overwrite in place

The patch file is typically a small --benchmark-json output from a single pytest run.
Any benchmark whose name already exists in the base is replaced; new ones are appended.
"""

import argparse
import json
import sys


def merge(base_path, patch_path, out_path):
    with open(base_path) as f:
        base = json.load(f)
    with open(patch_path) as f:
        patch = json.load(f)

    patch_by_name = {b["name"]: b for b in patch.get("benchmarks", [])}
    if not patch_by_name:
        print("Patch file contains no benchmarks — nothing to merge.")
        sys.exit(1)

    updated, added = 0, 0
    existing = []
    for b in base.get("benchmarks", []):
        if b["name"] in patch_by_name:
            existing.append(patch_by_name.pop(b["name"]))
            updated += 1
        else:
            existing.append(b)

    for b in patch_by_name.values():
        existing.append(b)
        added += 1

    base["benchmarks"] = existing
    with open(out_path, "w") as f:
        json.dump(base, f, indent=2)

    print(f"Merged into {out_path}: {updated} replaced, {added} added.")
    for b in patch.get("benchmarks", []):
        print(f"  {b['name']}: 2Q={b['extra_info'].get('output_gate_count_2q')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",  required=True, help="Existing full result JSON")
    parser.add_argument("--patch", required=True, help="Small JSON with new benchmark(s)")
    parser.add_argument("--out",   required=True, help="Output path (can equal --base)")
    args = parser.parse_args()
    merge(args.base, args.patch, args.out)


if __name__ == "__main__":
    main()
