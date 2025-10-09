from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .query import run_query
from .config import AppConfig
from ..utils import ensure_dir


def _read_qrels(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def _dcg(gains: List[int]) -> float:
    return sum((g / np.log2(i + 2) for i, g in enumerate(gains)))


def _ndcg_at_k(relevances: List[int], k: int) -> float:
    rel = relevances[:k]
    ideal = sorted(relevances, reverse=True)[:k]
    dcg = _dcg(rel)
    idcg = _dcg(ideal)
    return float(dcg / idcg) if idcg > 0 else 0.0


def _mrr_at_k(relevances: List[int], k: int) -> float:
    for i, r in enumerate(relevances[:k], start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def quality_with_qrels(
    cfg: AppConfig, qrels_path: Path, out_dir: Path, top_k: int
) -> pd.DataFrame:
    qrels = _read_qrels(qrels_path)
    combos = [f"{x['chunker']}__{x['embedding']}" for x in cfg.indexing.combinations]
    records = []

    for row in qrels:
        query = row["query"]
        rel_ids = set(row["relevant_doc_ids"])
        retrieved = run_query(cfg, query, top_k=top_k)

        for cid, triples in retrieved.items():
            doc_ids = [t[2].get("doc_id") for t in triples]
            relevances = [1 if d in rel_ids else 0 for d in doc_ids]

            rec = {
                "combo": cid,
                "query": query,
                "recall@k": float(sum(relevances) / max(1, len(rel_ids))),
                "mrr@k": _mrr_at_k(relevances, top_k),
                "ndcg@k": _ndcg_at_k(relevances, top_k),
            }
            records.append(rec)

    df = pd.DataFrame(records)
    agg = (
        df.groupby("combo")
        .agg({"recall@k": "mean", "mrr@k": "mean", "ndcg@k": "mean"})
        .reset_index()
    )
    ensure_dir(out_dir)
    df.to_csv(out_dir / "per_query_metrics.csv", index=False)
    agg.to_csv(out_dir / "summary_metrics.csv", index=False)

    # Bar plot (one figure per metric, saved to files)
    for metric in ["recall@k", "mrr@k", "ndcg@k"]:
        plt.figure()
        ax = agg.plot(kind="bar", x="combo", y=metric, legend=False)
        plt.title(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png")
        plt.close()

    return agg


def quality_unsupervised(
    cfg: AppConfig, queries: List[str], out_dir: Path, top_k: int
) -> pd.DataFrame:
    # Heuristics: average score, doc diversity, mean pairwise cosine (via score proxy)
    records = []
    for q in queries:
        retrieved = run_query(cfg, q, top_k=top_k)
        for cid, triples in retrieved.items():
            scores = [t[1] for t in triples]
            # FAISS returns smaller distances for closer matches; invert to similarity proxy
            sims = [1.0 / (1.0 + s) for s in scores]
            sources = [t[2].get("source") for t in triples]
            doc_diversity = len(set(sources)) / max(1, len(sources))
            rec = {
                "combo": cid,
                "query": q,
                "avg_sim_proxy": float(np.mean(sims) if sims else 0.0),
                "doc_diversity": float(doc_diversity),
            }
            records.append(rec)

    df = pd.DataFrame(records)
    agg = (
        df.groupby("combo")
        .agg({"avg_sim_proxy": "mean", "doc_diversity": "mean"})
        .reset_index()
    )
    ensure_dir(out_dir)
    df.to_csv(out_dir / "unsupervised_per_query.csv", index=False)
    agg.to_csv(out_dir / "unsupervised_summary.csv", index=False)

    for metric in ["avg_sim_proxy", "doc_diversity"]:
        plt.figure()
        ax = agg.plot(kind="bar", x="combo", y=metric, legend=False)
        plt.title(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png")
        plt.close()

    return agg
