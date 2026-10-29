"""Stage-local observations, explicitly separated from causal proof."""

from app.evaluation.metrics import ranked_ids, supported_fact_ids


def diagnose(query, stages, judgments, *, error=None, answer_assessment=None):
    if error:
        return [{"type": "execution_error", "status": "observed", "reason": str(error)}]
    if not query.get("answerable"):
        return [{"type": "unanswerable_case", "status": "review_required",
                 "reason": "Retrieval scores do not measure whether the system should abstain."}]
    evidence_gold = set(query.get("evidence_ids", []))
    facts = query.get("reference_facts", [])
    gold = {fact.get("fact_id", str(i)) for i, fact in enumerate(facts)} if facts else evidence_gold

    def covered(name):
        if name not in stages:
            return None
        evidence = {eid for cid in ranked_ids(stages[name])
                    for eid in judgments.get(cid, {}).get("evidence_ids", [])} & evidence_gold
        return set(supported_fact_ids(evidence, facts)) if facts else evidence

    observations = []
    previous = gold
    types = {"eligible": "product_filter_loss", "candidates": "candidate_retrieval_miss",
             "reranked": "ranking_or_selection_loss", "context": "context_budget_loss"}
    for name in ("eligible", "candidates", "reranked", "context"):
        current = covered(name)
        if current is None:
            previous = None
            continue
        # Missing stages are not assumed to contain all gold evidence.
        lost = (previous - current) if previous is not None else set()
        if lost:
            observations.append({"type": types[name], "status": "observed_stage_loss",
                                 "stage": name, "missing_fact_ids" if facts else "missing_evidence_ids": sorted(lost),
                                 "reason": "Evidence present at the preceding observed stage is absent here; causal confirmation requires replay."})
        previous = current
    assessment = answer_assessment or {}
    if covered("context") == gold and gold and assessment.get("correct") is False:
        observations.append({"type": "generation_error_candidate", "status": "review_required",
                             "reason": "All annotated evidence reached context, but answer correctness failed."})
    if not observations:
        observations.append({"type": "no_observed_evidence_loss", "status": "observed",
                             "reason": "No loss between observed stages; this does not certify the answer."})
    return observations
