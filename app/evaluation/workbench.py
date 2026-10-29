"""Local evaluation registry and durable, revision-checked review records."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from functools import lru_cache
import uuid

from app.evaluation.dataset import FILES, load_benchmark, read_jsonl
from app.evaluation.workflow import apply_reviews


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value) or '..' in value:
        raise ValueError('名称只能包含英文字母、数字、下划线、短横线和点')
    return value


@lru_cache(maxsize=8)
def _benchmark(path, signature):
    return load_benchmark(path)


class ConflictError(ValueError):
    pass


class Workbench:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.local = self.root / 'eval' / 'local'
        self.local.mkdir(parents=True, exist_ok=True)
        self.db = self.local / 'workbench.sqlite3'
        with self.connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS reviews (
                    id TEXT PRIMARY KEY, dataset_sha TEXT NOT NULL, query_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL,
                    published_to TEXT,
                    UNIQUE(dataset_sha, query_id, revision)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, event TEXT NOT NULL, entity_id TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, config TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    result TEXT, error TEXT
                );
            ''')

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA journal_mode=WAL')
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def server_lock(self):
        """One server owns recovery/dispatch; OS releases the lease on a crash."""
        with (self.local / 'workbench.lock').open('a+b') as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'1')
                handle.flush()
            handle.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ConflictError('当前工作区已有评测工作台运行，请使用单个服务进程') from exc
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def dataset_path(self, name):
        safe_name(name)
        path = self.root / 'eval' / 'examples' / 'retrieval' if name == 'example' else self.local / name
        if not (path / 'manifest.json').is_file():
            raise ValueError('数据版本不存在')
        return path

    def benchmark(self, name):
        path = self.dataset_path(name)
        signature = tuple((path / f).stat().st_mtime_ns for f in FILES)
        return _benchmark(str(path), signature)

    def datasets(self):
        names = ['example'] + sorted(p.parent.name for p in self.local.glob('*/manifest.json'))
        result = []
        for name in names:
            try:
                b = self.benchmark(name)
                result.append({'id': name, 'version': b.manifest['version'], 'sha256': b.fingerprint,
                               'total': len(b.queries), 'approved': len(b.select()),
                               'human_verified': sum(q.get('review', {}).get('human_verified') is True for q in b.select()),
                               'annotation_quality': b.manifest.get('annotation_quality', {})})
            except (ValueError, OSError) as exc:
                result.append({'id': name, 'error': str(exc)})
        return result

    def report_paths(self):
        logs = self.root / 'logs'
        paths = list(logs.glob('source_diagnostic_*/retrieval.json'))
        paths += list(logs.glob('source_diagnostic_*/answer_trust.json'))
        paths += list(logs.glob('workbench/*/retrieval.json'))
        paths += list(logs.glob('workbench/*/context_sweep.json'))
        paths += list(logs.glob('context_policy_answers_*/experiment.json'))
        paths += list(logs.glob('subquery_recall_*/retrieval.json'))
        paths += [logs / 'productdoc_bm25_pool_20261023.json', logs / 'productdoc_neural_pool_20261023.json']
        return {hashlib.sha256(str(p.relative_to(self.root)).encode()).hexdigest()[:16]: p for p in paths if p.is_file()}

    def report(self, report_id):
        paths = self.report_paths()
        if report_id not in paths:
            raise ValueError('实验报告不存在')
        return json.loads(paths[report_id].read_text(encoding='utf-8'))

    def reports(self):
        result = []
        for rid, path in self.report_paths().items():
            report = self.report(rid)
            result.append({'id': rid, 'name': str(path.relative_to(self.root)).replace('\\', '/'),
                           'schema': report.get('schema_version', 'context_sweep'),
                           'dataset_sha256': report.get('dataset_sha256'), 'config': report.get('config', {}),
                           'summary': report.get('summary', {})})
        return result

    def cases(self, benchmark):
        rows = []
        for path in sorted((self.root / 'logs').glob('source_diagnostic_*/case_queue.jsonl')):
            for row in read_jsonl(path):
                if row['dataset_sha256'] == benchmark.fingerprint:
                    rows.append(row)
        return {r['case_id']: r for r in rows}

    def latest_reviews(self, sha):
        with self.connection() as conn:
            rows = conn.execute('''SELECT r.* FROM reviews r JOIN
                (SELECT query_id, MAX(revision) revision FROM reviews WHERE dataset_sha=? GROUP BY query_id) latest
                ON r.query_id=latest.query_id AND r.revision=latest.revision WHERE r.dataset_sha=?''', (sha, sha)).fetchall()
        return {r['query_id']: dict(r, payload=json.loads(r['payload'])) for r in rows}

    def detail(self, name, qid):
        b = self.benchmark(name)
        if qid not in b.queries:
            raise ValueError('问题不存在')
        q = b.queries[qid]
        cases = [r for r in self.cases(b).values() if r['query_id'] == qid]
        relevant = {cid for cid, grade in b.qrels[qid].items() if grade > 0}
        ranked = set()
        for case in cases:
            observation = case['observation']
            for stage in ('candidates', 'reranked', 'context'):
                ranked.update(d['chunk_id'] for d in observation.get('stages', {}).get(stage, [])[:10])
        drafts = self.latest_reviews(b.fingerprint)
        source_index = self.local / 'source_pdf_index.jsonl'
        sources = {s['doc_id']: s for s in read_jsonl(source_index)} if source_index.is_file() else {}
        documents = {}
        for cid in relevant | ranked:
            c = b.corpus[cid]
            source = sources.get(c['doc_id'], {})
            documents[c['doc_id']] = {'doc_id': c['doc_id'], 'title': c.get('file_title', c['doc_id']),
                                     'source_revision': source.get('pdf_sha256', b.manifest['corpus_version']),
                                     'pdf_available': bool(source)}
        return {'dataset_sha256': b.fingerprint, 'query': q, 'review': drafts.get(qid),
                'documents': list(documents.values()),
                'evidence': [b.evidence[eid] for eid in q.get('evidence_ids', [])],
                'chunks': [dict(b.corpus[cid], relevance=b.qrels[qid].get(cid),
                                evidence_ids=b.judgments.get((qid, cid), {}).get('evidence_ids', []))
                           for cid in sorted(relevant | ranked)],
                'positive_total': len(relevant), 'cases': cases}

    def save_review(self, name, qid, expected_revision, decision):
        b = self.benchmark(name)
        if qid not in b.queries:
            raise ValueError('问题不存在')
        if b.queries[qid]['split'] == 'test':
            raise ValueError('冻结测试题不能进入开发审核回流')
        if not decision.get('reviewer', '').strip() or not decision.get('notes', '').strip():
            raise ValueError('填写审核人和审核说明')
        if type(decision.get('human_verified')) is not bool:
            raise ValueError('必须显式声明是否人工核验')
        case_id = decision.get('source_case_id')
        if case_id and (case_id not in self.cases(b) or self.cases(b)[case_id]['query_id'] != qid):
            raise ValueError('来源案例与当前数据版本或问题不一致')
        judgments = {cid: dict(j) for (jqid, cid), j in b.judgments.items() if jqid == qid}
        for patch in decision.get('qrel_patches', []):
            cid = patch['chunk_id']
            if cid not in b.corpus:
                raise ValueError('相关性修改引用未知chunk')
            judgments[cid] = dict(patch, query_id=qid, label_source='workbench_explicit_review')
        updates = dict(decision.get('query', {}))
        review = {'query_id': qid, 'reviewer': decision['reviewer'].strip(), 'reviewed_at': now(),
                  'reviewer_type': 'human' if decision['human_verified'] else 'explicit_source_review',
                  'human_verified': decision['human_verified'], 'source_case_id': case_id,
                  'notes': decision['notes'], 'review_scope': 'query_reference_and_positive_qrels',
                  'query': updates, 'evidence': decision.get('evidence', []), 'qrels': list(judgments.values())}
        # Validate the complete proposed dataset before storing an approved decision.
        import tempfile
        with tempfile.TemporaryDirectory(dir=self.local) as directory:
            apply_reviews(b, [review], Path(directory) / 'validated', 'review-validation-' + uuid.uuid4().hex)
        payload = {'decision': decision, 'review': review}
        with self.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            revision = conn.execute('SELECT COALESCE(MAX(revision),0) FROM reviews WHERE dataset_sha=? AND query_id=?',
                                    (b.fingerprint, qid)).fetchone()[0]
            if revision != expected_revision:
                raise ConflictError('审核已被更新，请重新加载后合并修改')
            rid = 'review_' + uuid.uuid4().hex
            conn.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?,NULL)',
                         (rid, b.fingerprint, qid, revision + 1, json.dumps(payload, ensure_ascii=False), now()))
            conn.execute('INSERT INTO events(event,entity_id,payload,created_at) VALUES(?,?,?,?)',
                         ('review_saved', rid, json.dumps({'dataset_sha': b.fingerprint, 'query_id': qid,
                                                          'revision': revision + 1}), now()))
        return {'id': rid, 'revision': revision + 1, 'human_verified': decision['human_verified']}

    def publish(self, name, output_name, version, review_ids):
        safe_name(output_name)
        b = self.benchmark(name)
        if not review_ids or len(review_ids) != len(set(review_ids)):
            raise ValueError('选择至少一条不重复的审核决定')
        with self.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            # Serialize publication against concurrent review edits and publication.
            latest = {r['id']: r for r in self.latest_reviews(b.fingerprint).values()}
            if set(review_ids) - latest.keys():
                raise ConflictError('只能发布当前数据版本下的最新审核决定')
            if any(latest[rid]['published_to'] for rid in review_ids):
                raise ConflictError('所选审核决定已经发布')
            target = self.local / output_name
            result = apply_reviews(b, [latest[rid]['payload']['review'] for rid in review_ids], target, version)
            for rid in review_ids:
                conn.execute('UPDATE reviews SET published_to=? WHERE id=?', (output_name, rid))
            conn.execute('INSERT INTO events(event,entity_id,payload,created_at) VALUES(?,?,?,?)',
                         ('dataset_published', output_name, json.dumps({'parent': b.fingerprint, 'reviews': review_ids}), now()))
        return dict(result, dataset_id=output_name, sha256=self.benchmark(output_name).fingerprint)

    def document(self, name, doc_id):
        b = self.benchmark(name)
        if doc_id not in {c['doc_id'] for c in b.corpus.values()}:
            raise ValueError('未知文档')
        index = self.local / 'source_pdf_index.jsonl'
        sources = read_jsonl(index) if index.is_file() else []
        source = next((s for s in sources if s['doc_id'] == doc_id), None)
        if not source:
            raise ValueError('没有可用的原始PDF定位')
        path = Path(source['path']).resolve()
        if path.suffix.lower() != '.pdf' or not any(path.is_relative_to(self.root / d) for d in ('doc', 'doc_eval')):
            raise ValueError('原始PDF不在本地文档目录内')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source['pdf_sha256']:
            raise ConflictError('原始PDF与来源核验时的版本不一致')
        return path
