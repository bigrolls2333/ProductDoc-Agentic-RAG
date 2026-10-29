"""Check graph integration without loading models, MongoDB or Milvus."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function],type_ignores=[]),relative,'exec'),namespace)
    return namespace[name]


class ProductRetrievalNodesTests(unittest.TestCase):
    def namespace(self):
        no_op = lambda *a,**kw:None
        return {'logger':SimpleNamespace(info=no_op,debug=no_op,warning=no_op,error=no_op),
                'add_running_task':no_op,'add_done_task':no_op,'sys':__import__('sys'),'os':os,
                'List':List,'Dict':Dict,'Any':Any}

    def test_embedding_node_reuses_one_embedding_and_queries_each_canonical_item(self):
        namespace = self.namespace()
        calls, embeddings = [], []
        def search(**kwargs):
            calls.append(kwargs)
            return [[{'entity':{'chunk_id':str(len(calls)), 'content':'evidence'}}]]
        namespace.update(generate_embeddings=lambda values:embeddings.append(values) or {'dense':[[1]],'sparse':[{1:1}]},
                         get_milvus_client=lambda:object(),create_hybrid_search_requests=lambda **kw:kw,
                         hybrid_search=search)
        namespace['_normalize_name_key'] = lambda value:''.join(value.split()).upper()
        node = load('app/query_process/agent/nodes/node_search_embedding.py','node_search_embedding',namespace)
        with patch.dict(os.environ,{'CHUNKS_COLLECTION':'chunks'}):
            result=node({'session_id':'test','is_stream':False,'rewritten_query':'比较它们',
                         'item_names':['A','B'],'context_selection_policy':'item_balanced','context_per_item_k':2})
        self.assertEqual(len(embeddings),1)
        self.assertEqual([c['reqs']['expr'] for c in calls],['canonical_name in ["A"]','canonical_name in ["B"]'])
        self.assertEqual([c['limit'] for c in calls],[2,2])
        self.assertEqual(len(result['embedding_chunks']),2)

    def test_hyde_search_keeps_per_item_candidates(self):
        namespace=self.namespace();calls=[]
        def search(**kwargs):
            calls.append(kwargs)
            return [[{'id':str(len(calls))}]]
        namespace.update(generate_embeddings=lambda values:{'dense':[[1]],'sparse':[{1:1}]},
                         get_milvus_client=lambda:object(),create_hybrid_search_requests=lambda **kw:kw,
                         hybrid_search=search,_normalize_name_key=lambda s:s.upper())
        function=load('app/query_process/agent/nodes/node_search_embedding_hyde.py','step_2_search_embedding_hyde',namespace)
        with patch.dict(os.environ,{'CHUNKS_COLLECTION':'chunks'}):
            result=function('比较它们','hypothesis',item_names=['A','B'],per_item_k=3)
        self.assertEqual(len(result[0]),2)
        self.assertEqual([c['limit'] for c in calls],[3,3])

    def test_rrf_does_not_restore_a_global_top10_that_drops_the_second_item(self):
        namespace=self.namespace();namespace['is_eval_enabled']=lambda name:False
        for name in ['_as_entity_list','reciprocal_rank_fusion','node_rrf']:
            function=load('app/query_process/agent/nodes/node_rrf.py',name,namespace)
        docs=[{'chunk_id':f'a{i}','canonical_name':'A','content':'text'} for i in range(15)]
        docs += [{'chunk_id':f'b{i}','canonical_name':'B','content':'text'} for i in range(2)]
        result=function({'session_id':'test','item_names':['A','B'],'embedding_chunks':docs,
                         'context_selection_policy':'item_balanced','context_per_item_k':2})
        self.assertEqual([d['chunk_id'] for d in result['rrf_chunks']],['a0','b0','a1','b1'])


if __name__=='__main__':
    unittest.main()
