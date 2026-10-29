"""Budgeted context selection from upstream canonical item names, without domain rules."""

import hashlib
from itertools import zip_longest
import os


SELECTOR_VERSION = 'productdoc.context-selector.v2'


def normalize_item_names(item_names):
    if item_names is None:
        return []
    if not isinstance(item_names, (list, tuple)) or any(not isinstance(name, str) for name in item_names):
        raise ValueError('item_names must contain canonical name strings')
    return list(dict.fromkeys(''.join(name.split()).upper() for name in item_names if name.strip()))


def context_policy(state):
    policy = state.get('context_selection_policy') or os.getenv('CONTEXT_SELECTION_POLICY', 'ranked')
    return 'item_balanced' if policy == 'balanced' else policy


def context_per_item_k(state):
    value = state.get('context_per_item_k')
    if value is None:
        value = os.getenv('CONTEXT_PER_ITEM_K', '20')
    if isinstance(value, bool) or str(value).strip() != str(int(value)) or not 1 <= int(value) <= 50:
        raise ValueError('context_per_item_k must be an integer between 1 and 50')
    return int(value)


def doc_item_name(doc):
    return ''.join(str(doc.get('canonical_name') or doc.get('item_name') or '').split()).upper()


def group_by_item(docs, item_names, per_item_k):
    groups = {name: [] for name in normalize_item_names(item_names)}
    for doc in docs:
        name = doc_item_name(doc)
        if name in groups and len(groups[name]) < per_item_k:
            groups[name].append(doc)
    return groups


def interleave_groups(groups):
    return [doc for round_docs in zip_longest(*groups.values()) for doc in round_docs if doc is not None]


def format_context_doc(doc, index):
    text = (doc.get('text') or doc.get('content') or '').strip()
    metadata = [f'[{index}]']
    for key, label in [('source', None), ('chunk_id', 'chunk_id'), ('url', 'url'), ('score', 'score'), ('title', 'title')]:
        value = doc.get(key)
        if value is None or value == '':
            continue
        if key == 'score':
            value = f'{float(value):.4f}'
        metadata.append(f'[{label}={value}]' if label else f'[{value}]')
    return ' '.join(metadata) + '\n' + text


def select_context_documents(docs, query, *, item_names=None, policy='ranked', budget=12000,
                             ranked_k=5, pool_k=50, max_docs=20, per_item_k=20):
    """Select whole chunks; query text is not used to infer products or evidence types."""
    requested_policy = policy
    policy = 'item_balanced' if policy == 'balanced' else policy
    if policy not in {'ranked', 'budgeted', 'item_balanced'}:
        raise ValueError('unknown context selection policy')
    if any(type(v) is not int or v < 1 for v in (budget, ranked_k, pool_k, max_docs, per_item_k)):
        raise ValueError('context bounds must be positive integers')
    names = normalize_item_names(item_names)
    unique, seen = [], set()
    for doc in docs:
        key = doc.get('chunk_id') or doc.get('url') or hashlib.sha256(str(doc.get('text') or doc.get('content') or '').encode()).hexdigest()
        if key not in seen and (doc.get('text') or doc.get('content') or '').strip():
            seen.add(key)
            unique.append(doc)
    pool = unique[:ranked_k if policy == 'ranked' else pool_k]
    groups = group_by_item(pool, names, per_item_k)
    fallback_reason = None
    if policy == 'item_balanced' and names:
        ordered = interleave_groups(groups)
    else:
        ordered = pool
        if policy == 'item_balanced':
            fallback_reason = 'upstream_item_names_empty_use_global_budget_order'
    included, rendered, dropped, used = [], [], [], 0
    for doc in ordered:
        if len(included) >= max_docs:
            break
        block = format_context_doc(doc, len(included) + 1)
        cost = len(block) + (2 if rendered else 0)
        if used + cost > budget:
            dropped.append(doc.get('chunk_id') or doc.get('url'))
            if policy == 'ranked':
                break
            continue
        included.append(dict(doc, text=(doc.get('text') or doc.get('content') or '').strip()))
        rendered.append(block)
        used += cost
    context = '\n\n'.join(rendered) if rendered else '无参考内容'
    if not rendered and len(context) > budget:
        context = ''
    counts = {name: sum(doc_item_name(d) == name for d in included) for name in names}
    return included, context, {'selector_version': SELECTOR_VERSION, 'policy': policy,
        'requested_policy': requested_policy, 'budget_chars': budget, 'pool_k': pool_k,
        'ranked_k': ranked_k, 'max_docs': max_docs, 'per_item_k': per_item_k,
        'input_count': len(unique), 'pool_count': len(pool), 'included_count': len(included),
        'omitted_count': len(unique) - len(included), 'context_chars': len(context),
        'budget_dropped_ids': dropped, 'item_names': names, 'item_names_source': 'upstream_state',
        'per_item_candidate_count': {name:sum(doc_item_name(d) == name for d in pool) for name in names},
        'per_item_selected_count': counts,
        'items_with_no_candidates': [name for name in names if not groups[name]],
        'items_without_selected_evidence': [name for name in names if not counts[name]],
        'unattributed_candidate_count': sum(not doc_item_name(d) for d in pool),
        'fallback_reason': fallback_reason,
        'context_sha256': hashlib.sha256(context.encode()).hexdigest()}
