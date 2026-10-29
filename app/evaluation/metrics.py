"""Corpus-qrel metrics; unjudged results stay visible in every report."""

from math import log2
from statistics import mean


def ranked_ids(rows):
    """Deduplicate a ranking without giving duplicate hits additional credit."""
    return list(dict.fromkeys(row["chunk_id"] if isinstance(row, dict) else row for row in rows))


def retrieval_metrics(ranking, qrels, ks=(1, 3, 5, 10, 20, 50)):
    ids = ranked_ids(ranking)
    relevant = {cid for cid, grade in qrels.items() if grade > 0}
    ideal = sorted((grade for grade in qrels.values() if grade > 0), reverse=True)
    metrics = {"relevant_total": len(relevant), "returned": len(ids)}
    for k in ks:
        if type(k) is not int or k < 1:
            raise ValueError("K must be a positive integer")
        top = ids[:k]
        hits = sum(cid in relevant for cid in top)
        first = next((i for i, cid in enumerate(top, 1) if cid in relevant), None)
        dcg = sum((2 ** qrels.get(cid, 0) - 1) / log2(i + 1) for i, cid in enumerate(top, 1))
        idcg = sum((2 ** grade - 1) / log2(i + 1) for i, grade in enumerate(ideal[:k], 1))
        # No-answer queries have no retrieval target. Evaluate abstention separately.
        metrics[f"recall@{k}"] = hits / len(relevant) if relevant else None
        metrics[f"hit@{k}"] = float(hits > 0) if relevant else None
        metrics[f"mrr@{k}"] = (1 / first if first else 0.0) if relevant else None
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg else None
        metrics[f"precision@{k}"] = hits / k if relevant else None
        metrics[f"judged_fraction@{k}"] = sum(cid in qrels for cid in top) / len(top) if top else None
        metrics[f"unjudged@{k}"] = sum(cid not in qrels for cid in top)
    return metrics


def supported_fact_ids(covered_ids, facts):
    covered = set(covered_ids)
    return [fact.get("fact_id", str(index)) for index, fact in enumerate(facts)
            if any(set(option) <= covered for option in fact.get("evidence_sets", [fact["evidence_ids"]]))]


def evidence_metrics(ranking, gold_ids, judgments, reference_facts=()):
    gold = set(gold_ids)
    covered = {eid for cid in ranked_ids(ranking)
               for eid in judgments.get(cid, {}).get("evidence_ids", [])} & gold
    result = {"evidence_recall": len(covered) / len(gold) if gold else None,
            "all_evidence": float(covered == gold) if gold else None,
            "covered_evidence_ids": sorted(covered), "missing_evidence_ids": sorted(gold - covered)}
    supported = supported_fact_ids(covered, reference_facts)
    result.update(fact_recall=len(supported) / len(reference_facts) if reference_facts else None,
                  all_facts=float(len(supported) == len(reference_facts)) if reference_facts else None,
                  covered_fact_ids=supported)
    return result


def average_metrics(rows):
    """Macro-average numeric metrics, keeping each metric's denominator."""
    keys = sorted({key for row in rows for key, value in row.items()
                   if value is None or isinstance(value, (float, int))})
    result = {}
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        result[key] = {"mean": mean(values) if values else None, "n": len(values)}
    return result
