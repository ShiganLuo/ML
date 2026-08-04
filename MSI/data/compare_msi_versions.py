#!/usr/bin/env python3
"""Compare MSI site coordinates between TopMsi (v1) and TopMsi_2_0 (v2).

File1 (v1 site.txt): no header, col[0]=chr, col[1]=start
File2 (v2 msi_sites.tsv): has header, chr=col[2], start_position=col[3]

Intersection rule: same chr AND |pos1 - pos2| <= 1
"""

import csv
import os
import sys
from collections import defaultdict

# --- paths ---
FILE1 = "/GeneCloud003/prod/project/Clinical/cnc_process/OncoTop/RIA2026Menglu/OncoTOP2607151716/230002661FD/indicator/msi/TopMsi/230002661FD_cancer_dedup_realign.site.txt"
FILE2 = "/GeneCloud003/prod/project/Clinical/cnc_process/OncoTop/RIA2026Menglu/OncoTOP2607151716/230002661FD/indicator/msi/TopMsi_2_0/230002661FD.cancer.msi_sites.tsv"
OUT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/results/msi_version_compare"


def load_file1(path):
    """Load v1 site.txt: col[0]=chr, col[1]=start. Returns list of (chr, pos, raw_line)."""
    sites = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            chr_ = cols[0].strip()
            pos = int(cols[1].strip())
            sites.append((chr_, pos, cols))
    return sites


def load_file2(path):
    """Load v2 msi_sites.tsv: has header, chr=col[2], start_position=col[3]."""
    sites = []
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        # find column indices by name
        chr_idx = header.index("chr")
        pos_idx = header.index("start_position")
        for cols in reader:
            if not cols or len(cols) <= max(chr_idx, pos_idx):
                continue
            chr_ = cols[chr_idx].strip()
            pos = int(cols[pos_idx].strip())
            sites.append((chr_, pos, cols, header))
    return sites


def find_intersections(sites1, sites2):
    """Find pairs where chr matches and |pos1-pos2| <= 1.

    Returns:
        matched: list of (idx1, idx2, chr, pos1, pos2, diff)
        only1: list of idx (sites in file1 not matched)
        only2: list of idx (sites in file2 not matched)
    """
    # group file2 sites by chr for fast lookup
    f2_by_chr = defaultdict(list)
    for i, (chr_, pos, *_) in enumerate(sites2):
        f2_by_chr[chr_].append((pos, i))

    matched = []
    matched_f1 = set()
    matched_f2 = set()

    for i1, (chr1, pos1, *_) in enumerate(sites1):
        if chr1 not in f2_by_chr:
            continue
        for pos2, i2 in f2_by_chr[chr1]:
            diff = abs(pos1 - pos2)
            if diff <= 1:
                matched.append((i1, i2, chr1, pos1, pos2, diff))
                matched_f1.add(i1)
                matched_f2.add(i2)
                break  # one-to-one, first match wins

    only1 = [i for i in range(len(sites1)) if i not in matched_f1]
    only2 = [i for i in range(len(sites2)) if i not in matched_f2]
    return matched, only1, only2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading files...")
    sites1 = load_file1(FILE1)
    sites2 = load_file2(FILE2)
    print(f"  File1 (TopMsi v1 site.txt):    {len(sites1)} sites")
    print(f"  File2 (TopMsi_2_0 v2 sites):   {len(sites2)} sites")

    matched, only1, only2 = find_intersections(sites1, sites2)

    n_matched = len(matched)
    n_only1 = len(only1)
    n_only2 = len(only2)

    print()
    print("=" * 60)
    print("Intersection Summary (|pos diff| <= 1)")
    print("=" * 60)
    print(f"  Matched (in both):     {n_matched}")
    print(f"  Only in v1 (site.txt): {n_only1}")
    print(f"  Only in v2 (sites):    {n_only2}")
    print(f"  v1 match rate:         {n_matched}/{len(sites1)} = {n_matched/len(sites1)*100:.1f}%")
    print(f"  v2 match rate:         {n_matched}/{len(sites2)} = {n_matched/len(sites2)*100:.1f}%")
    print()

    # distribution of position differences
    if matched:
        diffs = [m[5] for m in matched]
        diff_counts = {d: diffs.count(d) for d in sorted(set(diffs))}
        print("Position difference distribution (matched pairs):")
        for d, c in diff_counts.items():
            print(f"  diff={d}: {c} pairs")
    print()

    # --- write matched TSV ---
    matched_path = os.path.join(OUT_DIR, "matched_sites.tsv")
    with open(matched_path, "w") as f:
        f.write("chr\tv1_start\tv2_start\tpos_diff\t")
        # v1 columns
        f.write("\t".join([f"v1_col{i}" for i in range(len(sites1[0][2]))]) + "\t")
        # v2 columns
        v2_header = sites2[0][3]
        f.write("\t".join([f"v2_{h}" for h in v2_header]) + "\n")
        for i1, i2, chr_, pos1, pos2, diff in matched:
            _, _, cols1 = sites1[i1]
            _, _, cols2, _ = sites2[i2]
            f.write(f"{chr_}\t{pos1}\t{pos2}\t{diff}\t")
            f.write("\t".join(cols1) + "\t")
            f.write("\t".join(cols2) + "\n")
    print(f"Matched sites written to: {matched_path}")

    # --- write only-v1 TSV ---
    only1_path = os.path.join(OUT_DIR, "only_v1.tsv")
    with open(only1_path, "w") as f:
        f.write("chr\tv1_start\t")
        f.write("\t".join([f"v1_col{i}" for i in range(len(sites1[0][2]))]) + "\n")
        for i in only1:
            chr_, pos, cols = sites1[i]
            f.write(f"{chr_}\t{pos}\t")
            f.write("\t".join(cols) + "\n")
    print(f"V1-only sites written to: {only1_path}")

    # --- write only-v2 TSV ---
    only2_path = os.path.join(OUT_DIR, "only_v2.tsv")
    with open(only2_path, "w") as f:
        v2_header = sites2[0][3]
        f.write("\t".join(v2_header) + "\n")
        for i in only2:
            _, _, cols, _ = sites2[i]
            f.write("\t".join(cols) + "\n")
    print(f"V2-only sites written to: {only2_path}")

    # --- write summary ---
    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("MSI Site Coordinate Comparison: TopMsi v1 vs TopMsi_2_0 v2\n")
        f.write(f"Sample: 230002661FD\n")
        f.write(f"Intersection rule: same chr AND |pos diff| <= 1\n")
        f.write(f"\n")
        f.write(f"File1 (v1): {FILE1}\n")
        f.write(f"  Total sites: {len(sites1)}\n")
        f.write(f"File2 (v2): {FILE2}\n")
        f.write(f"  Total sites: {len(sites2)}\n")
        f.write(f"\n")
        f.write(f"Matched (in both):     {n_matched}\n")
        f.write(f"Only in v1:            {n_only1}\n")
        f.write(f"Only in v2:            {n_only2}\n")
        f.write(f"v1 match rate:         {n_matched}/{len(sites1)} = {n_matched/len(sites1)*100:.1f}%\n")
        f.write(f"v2 match rate:         {n_matched}/{len(sites2)} = {n_matched/len(sites2)*100:.1f}%\n")
        if matched:
            f.write(f"\nPosition difference distribution:\n")
            for d, c in diff_counts.items():
                f.write(f"  diff={d}: {c} pairs\n")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
