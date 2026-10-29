"""Bounded query decomposition and weighted route fusion, without domain keywords."""

import os

from app.query_process.context_selection import normalize_item_names, group_by_item, interleave_groups


SUBQUERY_VERSION = 'productdoc.subquery.v1'
MAX_SUBQUERIES = 3


def subquery_enabled(state):
    if os.getenv('EVAL_DISABLE_SUBQUERY', '').lower() in {'1', 'true', 'yes'}:
        return False
    value = state.get('subquery_retrieval_enabled')
    if value is None:
        value = os.getenv('SUBQUERY_RETRIEVAL_ENABLED', 'false').lower() in {'1', 'true', 'yes'}
    if type(value) is not bool:
        raise ValueError('subquery_retrieval_enabled must be boolean or None')
    return value


def decomposition_instruction(item_names):
    names = normalize_item_names(item_names)
    return '''
同时决定是否需要子查询扩展召回。简单单目标问题返回subqueries=[]，不要只换同义词。
确有多个待查信息点，或需要将口语现象转换成可检索的信息需求时，最多生成3条短检索查询。
只表达用户已提出的需求，不回答问题，不猜测故障原因/参数/解决办法，不引入用户未要求的主题。
每条形如 {"query":"一个独立的信息需求", "item_indices":[0]}。
item_indices必须是下方已确认商品的零基下标；仅绑定相关商品，不新增型号。一个查询可绑定多个商品。
不要原样重复当前查询；商品名称由过滤器限定，无需在查询中堆叠型号。
subqueries必须作为同一JSON对象的字段返回，不要额外调用工具。
已确认商品下标：
''' + '\n'.join(f'{index}: {name}' for index, name in enumerate(names))


def validate_subqueries(value, item_names, original_query=''):
    names = normalize_item_names(item_names)
    if not isinstance(value, list) or len(value) > MAX_SUBQUERIES:
        raise ValueError('subqueries must be a list with at most 3 entries')
    result, seen = [], set()
    original = ''.join(original_query.split()).casefold()
    for row in value:
        if not isinstance(row, dict):
            raise ValueError('each subquery must be an object')
        query, indices = row.get('query'), row.get('item_indices')
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200:
            raise ValueError('subquery text must contain 1..200 characters')
        if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < len(names) for i in indices):
            raise ValueError('subquery indices must refer to confirmed items')
        indices = sorted(set(indices))
        normalized = ''.join(query.split()).casefold()
        key = normalized, tuple(indices)
        if key in seen or normalized == original:
            continue
        seen.add(key)
        result.append({'query': query.strip(), 'item_indices': indices})
    return result


def weighted_rrf(routes, k=60, max_results=None):
    """routes=(name, documents, weight); each document receives one vote per route."""
    scores, docs, sources = {}, {}, {}
    for name, rows, weight in routes:
        seen = set()
        for rank, doc in enumerate(rows, 1):
            cid = doc.get('chunk_id') or doc.get('id')
            if not cid or cid in seen:
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0) + weight / (k + rank)
            docs.setdefault(cid, doc)
            sources.setdefault(cid, []).append(name)
    ordered = sorted(scores, key=scores.get, reverse=True)
    if max_results is not None:
        ordered = ordered[:max_results]
    return [(dict(docs[cid], retrieval_sources=sources[cid]), scores[cid]) for cid in ordered]


def fuse_routes(main, hyde, subqueries, *, item_names, per_item_k=20, pool_k=50):
    """Share a total weight of 1 across subqueries, just as one ordinary route."""
    routes = [('main', main, 1), ('hyde', hyde, 1)]
    n = len(subqueries)
    routes.extend((f'subquery:{i}', docs, 1/n) for i, docs in enumerate(subqueries))
    rows = [dict(doc, fusion_score=score) for doc, score in weighted_rrf(routes)]
    return interleave_groups(group_by_item(rows, item_names, per_item_k))[:pool_k]
