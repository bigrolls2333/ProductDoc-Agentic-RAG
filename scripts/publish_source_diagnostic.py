"""Publish explicit AI source-review decisions without promoting them to human gold."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.annotation import locate_quote
from app.evaluation.dataset import load_benchmark, read_jsonl, write_jsonl
from app.evaluation.workflow import apply_reviews


def publish(dataset, audits_path, sources_path, decisions_path, output, version):
    benchmark = load_benchmark(dataset)
    audits = {row['query_id']: row for row in read_jsonl(audits_path)}
    sources = {row['doc_id']: row for row in read_jsonl(sources_path)}
    decisions = json.loads(Path(decisions_path).read_text(encoding='utf-8'))
    catalogue = {}
    for audit in audits.values():
        for fact in audit['verified_facts']:
            source = sources[fact['doc_id']]
            catalogue[fact['evidence_id']] = {key: fact[key] for key in
                                            ('evidence_id', 'doc_id', 'source_revision', 'quote', 'locator')}
            catalogue[fact['evidence_id']]['locator']['extraction_method'] = source.get('extraction_method', 'native_pdf_text')
    for anchor in decisions.get('extra_evidence', []):
        source = sources[anchor['doc_id']]
        pages = locate_quote(anchor['quote'], source)
        if not pages:
            if not anchor.get('visual_review_artifact') or not anchor.get('pdf_pages'):
                raise ValueError('unmatched source anchor requires explicit visual review evidence')
            preview = Path(anchor['visual_review_artifact'])
            if not preview.is_file():
                raise ValueError('visual review artifact missing')
            pages = anchor['pdf_pages']
            method = 'ai_visual_source_review'
            proof = {'visual_review_artifact': str(preview.resolve()),
                     'artifact_sha256': hashlib.sha256(preview.read_bytes()).hexdigest()}
        else:
            method, proof = 'normalized_pdf_text_match', {}
        catalogue[anchor['evidence_id']] = {key: anchor[key] for key in ('evidence_id', 'doc_id', 'quote')}
        catalogue[anchor['evidence_id']].update(source_revision=source['pdf_sha256'], locator={
            'path': source['path'], 'pdf_pages': pages, 'match': method,
            'extraction_method': source.get('extraction_method', 'native_pdf_text'), **proof})
    reviews = []
    for qid, decision in decisions['queries'].items():
        if decision['status'] != 'approved':
            continue
        audit = audits[qid]
        doc_ids = set(audit['target_doc_ids'])
        facts = decision['reference_facts']
        evidence_ids = sorted({eid for fact in facts for eid in fact['evidence_ids']})
        evidence = [catalogue[eid] for eid in evidence_ids]
        if any(e['doc_id'] not in doc_ids for e in evidence):
            raise ValueError(f'reference source outside authoritative documents: {qid}')
        # Target-document unknowns stay unjudged. Foreign manuals are negatives only
        # under this explicitly scoped product-manual task, never through retrieval rank.
        judgments = {row['chunk_id']: dict(row) for row in audit['qrels']}
        for cid, chunk in benchmark.corpus.items():
            if chunk['doc_id'] not in doc_ids or not decision['answerable']:
                judgments[cid] = {'chunk_id': cid, 'relevance': 0, 'evidence_ids': [],
                                  'label_source': 'authoritative_product_scope' if chunk['doc_id'] not in doc_ids
                                  else 'ai_full_target_document_no_answer_review'}
        used = set()
        for cid, chunk in benchmark.corpus.items():
            mapped = [e['evidence_id'] for e in evidence if e['doc_id'] == chunk['doc_id'] and e['quote'] in chunk['content']]
            if mapped:
                row = judgments.setdefault(cid, {'chunk_id': cid, 'relevance': 2, 'evidence_ids': []})
                row.update(relevance=2, evidence_ids=sorted(set(row.get('evidence_ids', [])) | set(mapped)),
                           label_source='ai_source_reference_anchor')
                used.update(mapped)
        if used != set(evidence_ids):
            raise ValueError(f'unmapped reference evidence: {qid}/{set(evidence_ids) - used}')
        # Keep only published reference anchors in qrels, not unpublished proposal IDs.
        for row in judgments.values():
            row['evidence_ids'] = sorted(set(row.get('evidence_ids', [])) & set(evidence_ids))
        chunks = [c for c in benchmark.corpus.values() if c['doc_id'] in doc_ids]
        reviews.append({'query_id': qid, 'reviewer': 'codex-source-audit', 'reviewer_type': 'ai_source_review',
                        'human_verified': False, 'reviewed_at': decisions['reviewed_at'],
                        'query': {'answerable': decision['answerable'], 'split': 'dev',
                                  'group_id': 'diagnostic_products_connected_v1',
                                  'product_ids': sorted({c['product_id'] for c in chunks if c.get('product_id')}),
                                  'canonical_names': sorted({c['canonical_name'] for c in chunks if c.get('canonical_name')}),
                                  'evidence_ids': evidence_ids, 'reference_facts': facts},
                        'evidence': evidence, 'qrels': list(judgments.values())})
    result = apply_reviews(benchmark, reviews, output, version)
    path = Path(output) / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['annotation_quality'].update(
        scope='curated diagnostic dev set; no held-out generalization claim',
        policy='authoritative_product_manuals_only',
        label_method='AI relevance proposals + PDF quote verification + explicit AI source review',
        human_verified_queries=0, pending_scope_refinement=[q for q, d in decisions['queries'].items() if d['status'] != 'approved'],
        decisions_sha256=hashlib.sha256(Path(decisions_path).read_bytes()).hexdigest(),
        source_index_sha256=hashlib.sha256(Path(sources_path).read_bytes()).hexdigest())
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    write_jsonl(Path(output) / 'review_decisions.jsonl', reviews)
    validated = load_benchmark(output)
    return dict(result, dataset_sha256=validated.fingerprint,
                evidence=len(validated.evidence), judgments=len(validated.judgments), human_verified_queries=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'audits', 'sources', 'decisions', 'output', 'version'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.dataset, args.audits, args.sources, args.decisions, args.output, args.version), ensure_ascii=False))


if __name__ == '__main__':
    main()
