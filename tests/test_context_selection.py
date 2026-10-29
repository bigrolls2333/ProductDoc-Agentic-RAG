"""Per-item selection respects upstream names, rank, quotas and rendered budgets."""

import unittest
from app.query_process.context_selection import select_context_documents, format_context_doc, context_per_item_k
from app.query_process.product_retrieval import search_per_item, canonical_filter


def doc(cid, name, text='evidence', title=''):
    return {'chunk_id': cid, 'canonical_name': name, 'text': text, 'title': title}


class ContextSelectionTests(unittest.TestCase):
    def test_reuses_upstream_names_when_query_contains_only_pronouns(self):
        docs = [doc('a1', '商品A'), doc('a2', '商品A'), doc('b', '商品B'), doc('c', '商品C')]
        selected, _, trace = select_context_documents(docs, '对比它们', item_names=['商品A', '商品B'],
                                                       policy='item_balanced', max_docs=2)
        self.assertEqual([d['chunk_id'] for d in selected], ['a1', 'b'])
        self.assertEqual(trace['item_names_source'], 'upstream_state')
        self.assertEqual(trace['per_item_selected_count'], {'商品A': 1, '商品B': 1})

    def test_no_domain_specific_table_priority(self):
        docs = [doc('generic', 'P3030'), doc('table', 'P3030', '<table>故障原因及解决办法</table>', '图像缺陷')]
        for query in ['打印质量不好', '查电源', '比较参数']:
            selected, _, _ = select_context_documents(docs, query, item_names=['P3030'], policy='item_balanced', max_docs=1)
            self.assertEqual(selected[0]['chunk_id'], 'generic')

    def test_round_robin_preserves_each_item_rank_and_topk_cap(self):
        docs = [doc(f'a{i}', 'A') for i in range(4)] + [doc(f'b{i}', 'B') for i in range(3)]
        selected, _, _ = select_context_documents(docs, '', item_names=['A', 'B'], policy='item_balanced', per_item_k=2)
        self.assertEqual([d['chunk_id'] for d in selected], ['a0', 'b0', 'a1', 'b1'])

    def test_skip_oversized_document_counts_metadata_and_keeps_exact_text(self):
        docs = [doc('huge', 'A', 'x' * 1000), doc('small', 'A', '12V'), doc('b', 'B', '24V')]
        ranked, _, _ = select_context_documents(docs, '', budget=100, ranked_k=2)
        selected, context, trace = select_context_documents(docs, '', item_names=['A', 'B'], policy='item_balanced', budget=100, per_item_k=2)
        self.assertEqual(ranked, [])
        self.assertEqual([d['chunk_id'] for d in selected], ['b', 'small'])
        self.assertEqual(selected[1]['text'], '12V')
        self.assertEqual(len(context), trace['context_chars'])
        self.assertLessEqual(len(context), 100)
        self.assertEqual(context, format_context_doc(docs[2], 1)+'\n\n'+format_context_doc(docs[1], 2))

    def test_missing_item_not_filled_with_another_products_chunks(self):
        docs = [doc('a', 'A'), doc('unrelated', 'C')]
        selected, _, trace = select_context_documents(docs, '', item_names=['A', 'B'], policy='item_balanced')
        self.assertEqual([d['chunk_id'] for d in selected], ['a'])
        self.assertEqual(trace['items_with_no_candidates'], ['B'])
        self.assertEqual(trace['items_without_selected_evidence'], ['B'])

    def test_empty_upstream_names_use_explicit_budget_fallback_without_query_inference(self):
        docs = [doc('a', 'A')]
        selected, _, trace = select_context_documents(docs, '完全不同的商品B', item_names=[], policy='item_balanced')
        self.assertEqual(len(selected), 1)
        self.assertEqual(trace['item_names'], [])
        self.assertIn('item_names_empty', trace['fallback_reason'])

    def test_canonical_normalization_dedup_and_legacy_metadata(self):
        docs = [doc('a', '  A 商品 '), doc('a', 'A商品'), {'chunk_id':'b','item_name':'b商品','text':'text'}]
        selected, _, trace = select_context_documents(docs, '', item_names=['a商品','A 商品',' B商品 '], policy='balanced')
        self.assertEqual([d['chunk_id'] for d in selected], ['a','b'])
        self.assertEqual(trace['policy'], 'item_balanced')
        self.assertEqual(len(trace['item_names']), 2)

    def test_rejects_invalid_per_item_limits(self):
        for value in [True, 0, -1, 51, 2.5]:
            with self.assertRaises((ValueError, TypeError)):
                context_per_item_k({'context_per_item_k':value})

    def test_per_item_search_filters_and_shares_vectors_without_cross_item_score_sort(self):
        requests, searches = [], []
        dense, sparse = [1,2], {1:1}
        def create(**kwargs):
            requests.append(kwargs)
            return kwargs
        def search(**kwargs):
            searches.append(kwargs)
            item = kwargs['reqs']['expr']
            return [[{'id':f'{item}-1'}, {'id':f'{item}-2'}]]
        result = search_per_item(item_names=['A','B','a'],per_item_k=2,dense_vector=dense,sparse_vector=sparse,
                                 client=object(),collection_name='chunks',create_requests=create,search=search,output_fields=['chunk_id'])
        self.assertEqual([r['expr'] for r in requests], ['canonical_name in ["A"]','canonical_name in ["B"]'])
        self.assertTrue(all(r['dense_vector'] is dense and r['sparse_vector'] is sparse for r in requests))
        self.assertTrue(all(r['limit']==2 for r in searches))
        self.assertEqual([h['id'] for h in result[0]], ['canonical_name in ["A"]-1','canonical_name in ["B"]-1',
                                                      'canonical_name in ["A"]-2','canonical_name in ["B"]-2'])
        self.assertIn('\\"', canonical_filter(['A"B']))


if __name__ == '__main__':
    unittest.main()
