"""Standalone local review workbench; no model/database clients loaded at startup."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.parse import urlsplit
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

from app.evaluation.workbench import Workbench, ConflictError, now


class QrelPatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    chunk_id: str = Field(min_length=1)
    relevance: int = Field(ge=0, le=2, strict=True)
    evidence_ids: list[str] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_revision: int = Field(ge=0)
    reviewer: str = Field(min_length=1, max_length=120)
    notes: str = Field(min_length=1, max_length=8000)
    human_verified: bool
    source_case_id: str | None = None
    query: dict = Field(default_factory=dict)
    evidence: list[dict] = Field(default_factory=list)
    qrel_patches: list[QrelPatch] = Field(default_factory=list)


class PublishRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    output_name: str
    version: str = Field(min_length=1, max_length=100)
    review_ids: list[str] = Field(min_length=1)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dataset: str
    kind: str = 'retrieval_replay'
    source_dataset: str | None = None
    prediction_ids: list[str] = Field(min_length=1, max_length=5)
    context_ks: list[int] = Field(default_factory=lambda: [5, 10, 20])
    budgets: list[int] = Field(default_factory=lambda: [6000, 12000, 24000])


def create_app(root=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    workbench = Workbench(root)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='eval-workbench')
    submitted, submit_lock = set(), threading.Lock()

    def job_worker(job_id):
        with workbench.connection() as conn:
            row = conn.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row or row['status'] != 'queued':
                return
            conn.execute('UPDATE jobs SET status=?,updated_at=? WHERE id=?', ('running', now(), job_id))
            config = json.loads(row['config'])
        output = root / 'logs' / 'workbench' / job_id
        output.mkdir(parents=True, exist_ok=True)
        try:
            paths = workbench.report_paths()
            predictions = [paths[rid] for rid in config['prediction_ids']]
            dataset = workbench.benchmark(config['dataset'])
            if dataset.fingerprint != config['dataset_sha256']:
                raise ValueError('数据文件在任务创建后发生变化，请重新创建任务')
            if config['kind'] == 'retrieval_replay':
                cmd = [sys.executable, str(root / 'scripts' / 'replay_retrieval_eval.py'), '--dataset', str(dataset.root),
                       '--source-dataset', str(workbench.dataset_path(config['source_dataset'])),
                       '--predictions', *map(str, predictions), '--output', str(output / 'retrieval.json')]
            else:
                cmd = [sys.executable, str(root / 'scripts' / 'sweep_context_budget.py'), '--dataset', str(dataset.root),
                       '--predictions', str(predictions[0]), '--context-ks', ','.join(map(str, config['context_ks'])),
                       '--budgets', ','.join(map(str, config['budgets'])), '--output', str(output / 'context_sweep.json')]
            process = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
            (output / 'execution.log').write_text(process.stdout + process.stderr, encoding='utf-8')
            if process.returncode:
                raise ValueError((process.stderr or process.stdout)[-2500:])
            result = {'directory': str(output.relative_to(root)), 'message': process.stdout.strip()}
            with workbench.connection() as conn:
                conn.execute('UPDATE jobs SET status=?,updated_at=?,result=? WHERE id=?',
                             ('completed', now(), json.dumps(result, ensure_ascii=False), job_id))
        except Exception as exc:
            with workbench.connection() as conn:
                conn.execute('UPDATE jobs SET status=?,updated_at=?,error=? WHERE id=?',
                             ('failed', now(), f'{type(exc).__name__}: {exc}', job_id))

    def submit(job_id):
        with submit_lock:
            if job_id not in submitted:
                submitted.add(job_id)
                executor.submit(job_worker, job_id)

    @asynccontextmanager
    async def lifespan(app):
        with workbench.server_lock():
            with workbench.connection() as conn:
                conn.execute("UPDATE jobs SET status='interrupted',updated_at=? WHERE status='running'", (now(),))
                queued = conn.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at").fetchall()
            for row in queued:
                submit(row['id'])
            try:
                yield
            finally:
                executor.shutdown(wait=True)

    app = FastAPI(title='ProductDoc 评测工作台', lifespan=lifespan)
    app.state.workbench = workbench

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse(status_code=409 if isinstance(exc, ConflictError) else 400, content={'detail': str(exc)})

    @app.middleware('http')
    async def same_origin(request: Request, call_next):
        origin = request.headers.get('origin')
        if request.method not in {'GET', 'HEAD', 'OPTIONS'} and origin and urlsplit(origin).netloc != request.headers.get('host'):
            return JSONResponse(status_code=403, content={'detail': '仅接受本地工作台同源写入'})
        return await call_next(request)

    @app.get('/')
    def home():
        return FileResponse(Path(__file__).parent / 'page' / 'index.html')

    @app.get('/api/datasets')
    def datasets():
        return workbench.datasets()

    @app.get('/api/datasets/{name}/queries')
    def queries(name: str, search: str = '', status: str = 'approved', cases_only: bool = False):
        b = workbench.benchmark(name)
        cases = list(workbench.cases(b).values())
        reviews = workbench.latest_reviews(b.fingerprint)
        rows = []
        for q in b.queries.values():
            qc = [c for c in cases if c['query_id'] == q['query_id']]
            if search.casefold() not in (q['query_id'] + q['query']).casefold() or (status != 'all' and q['review_status'] != status):
                continue
            if cases_only and not qc:
                continue
            r = reviews.get(q['query_id'])
            rows.append({'query_id': q['query_id'], 'query': q['query'], 'category': q.get('category'),
                         'answerable': q['answerable'], 'split': q['split'], 'review_status': q['review_status'],
                         'case_count': len(qc), 'diagnoses': sorted({d['type'] for c in qc for d in c['diagnosis']}),
                         'draft_revision': r['revision'] if r else 0, 'review_id': r['id'] if r else None,
                         'published_to': r['published_to'] if r else None})
        return rows

    @app.get('/api/datasets/{name}/queries/{qid}')
    def detail(name: str, qid: str):
        return workbench.detail(name, qid)

    @app.get('/api/datasets/{name}/chunks/{cid}')
    def chunk(name: str, cid: str):
        b = workbench.benchmark(name)
        if cid not in b.corpus:
            raise HTTPException(404, 'chunk不存在')
        return b.corpus[cid]

    @app.get('/api/datasets/{name}/documents/{doc_id}/pdf')
    def pdf(name: str, doc_id: str):
        return FileResponse(workbench.document(name, doc_id), media_type='application/pdf')

    @app.post('/api/datasets/{name}/queries/{qid}/reviews')
    def save(name: str, qid: str, body: ReviewRequest):
        return workbench.save_review(name, qid, body.expected_revision, body.model_dump(exclude={'expected_revision'}))

    @app.post('/api/datasets/{name}/publish')
    def publish(name: str, body: PublishRequest):
        return workbench.publish(name, body.output_name, body.version, body.review_ids)

    @app.get('/api/reports')
    def reports():
        return workbench.reports()

    @app.get('/api/reports/{report_id}')
    def report(report_id: str, query_id: str | None = None):
        data = workbench.report(report_id)
        if query_id:
            data['results'] = [r for r in data.get('results', []) if r['query_id'] == query_id]
        else:
            data.pop('results', None)
        return data

    def job_dict(row):
        data = dict(row)
        data['config'] = json.loads(data['config'])
        data['result'] = json.loads(data['result']) if data['result'] else None
        return data

    @app.get('/api/runs')
    def jobs():
        with workbench.connection() as conn:
            return [job_dict(r) for r in conn.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100')]

    @app.post('/api/runs', status_code=202)
    def run(body: RunRequest):
        config = body.model_dump()
        b = workbench.benchmark(body.dataset)
        config['dataset_sha256'] = b.fingerprint
        if not b.select('dev'):
            raise ValueError('该版本没有批准的开发题')
        paths = workbench.report_paths()
        if set(body.prediction_ids) - paths.keys() or len(body.prediction_ids) != len(set(body.prediction_ids)):
            raise ValueError('预测报告不存在或重复')
        if body.kind == 'retrieval_replay':
            if not body.source_dataset:
                raise ValueError('选择原预测使用的数据版本')
            source = workbench.benchmark(body.source_dataset)
            for rid in body.prediction_ids:
                r = workbench.report(rid)
                if r.get('schema_version') != 'productdoc.experiment.v1' or r['dataset_sha256'] != source.fingerprint:
                    raise ValueError('预测报告与原始数据版本不一致')
        elif body.kind == 'context_sweep':
            if len(body.prediction_ids) != 1:
                raise ValueError('上下文消融需要一份完整检索报告')
            r = workbench.report(body.prediction_ids[0])
            if r.get('schema_version') != 'productdoc.experiment.v1' or r['dataset_sha256'] != b.fingerprint:
                raise ValueError('上下文消融报告与当前版本不一致')
            if not body.context_ks or not body.budgets or min(body.context_ks + body.budgets) < 1:
                raise ValueError('Top-K和预算必须为正整数')
            if max(body.context_ks) > r['config']['candidate_k'] or max(body.budgets) > 100000 or len(body.context_ks) * len(body.budgets) > 30:
                raise ValueError('消融范围超过候选数量或本地任务上限')
        else:
            raise ValueError('未知实验类型')
        jid = 'run_' + uuid.uuid4().hex
        with workbench.connection() as conn:
            conn.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?)',
                         (jid, json.dumps(config, ensure_ascii=False), 'queued', now(), now(), None, None))
        submit(jid)
        return {'id': jid, 'status': 'queued'}

    @app.post('/api/runs/{job_id}/retry', status_code=202)
    def retry(job_id: str):
        with workbench.connection() as conn:
            row = conn.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row or row['status'] not in {'failed', 'interrupted'}:
                raise ValueError('仅允许重试失败或中断的任务')
        config = json.loads(row['config'])
        config.pop('dataset_sha256', None)
        return run(RunRequest(**config))

    app.mount('/static', StaticFiles(directory=Path(__file__).parent / 'page'), name='static')
    return app


app = create_app()
