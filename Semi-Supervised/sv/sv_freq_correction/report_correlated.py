#!/usr/bin/env python3
"""Report highly correlated numeric columns from a TSV feature table.

Outputs a TSV of correlated pairs and a JSON of correlated clusters.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def find_high_corr_pairs(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    # drop constant columns
    nunique = numeric.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        numeric = numeric.drop(columns=const_cols)

    corr = numeric.corr().abs()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs: List[Tuple[str, str, float]] = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iat[i, j]
            if np.isfinite(val) and val >= threshold:
                pairs.append((cols[i], cols[j], float(val)))

    return pd.DataFrame(pairs, columns=["col1", "col2", "abs_corr"]).sort_values("abs_corr", ascending=False)


def build_clusters(pairs_df: pd.DataFrame) -> List[List[str]]:
    # simple connected components over undirected graph
    edges = [(r[0], r[1]) for r in pairs_df[["col1", "col2"]].itertuples(index=False, name=None)]
    adj: Dict[str, Set[str]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    seen: Set[str] = set()
    clusters: List[List[str]] = []
    for node in adj:
        if node in seen:
            continue
        stack = [node]
        comp: List[str] = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for v in adj.get(u, set()):
                if v not in seen:
                    stack.append(v)
        clusters.append(sorted(comp))
    return clusters


def main():
    parser = argparse.ArgumentParser(description="Report highly correlated columns in a feature TSV")
    parser.add_argument("-i","--infile", required=False, default="output/ML/feature/raw_extracted_features.tsv", help="input TSV file")
    parser.add_argument("-t","--threshold", type=float, default=0.95, help="absolute correlation threshold")
    parser.add_argument("-o","--out-prefix", default="output/ML/feature/correlated", help="output prefix (will add .tsv/.json)")
    args = parser.parse_args()

    df = pd.read_csv(args.infile, sep="\t")
    pairs = find_high_corr_pairs(df, args.threshold)
    clusters = build_clusters(pairs) if not pairs.empty else []

    pairs_path = f"{args.out_prefix}_pairs.tsv"
    clusters_path = f"{args.out_prefix}_clusters.json"
    pairs.to_csv(pairs_path, sep="\t", index=False)
    with open(clusters_path, "w", encoding="utf-8") as fh:
        json.dump(clusters, fh, indent=2, ensure_ascii=False)

    print(f"Read {len(df)} rows, checked numeric columns.")
    print(f"Found {len(pairs)} correlated pairs >= {args.threshold} (pairs written to {pairs_path})")
    print(f"Found {len(clusters)} correlated clusters (written to {clusters_path})")


if __name__ == "__main__":
    main()
