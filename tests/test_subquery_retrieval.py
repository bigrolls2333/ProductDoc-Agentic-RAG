"""Check expansion boundaries, route weights and integration using in-memory calls."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.query_process.subquery_retrieval import validate_subqueries, weighted_rrf, fuse_routes, subquery_enabled

ROOT = Path(__file__).resolve().parents[1]


class SubqueryRetrievalTests(unittest.TestCase):
    def test_query_limits_and_product_binding_reject_invented_items(self):
        for rows in [[{'query':'q','item_indices':[2]}], [{'query':'q','item_indices':[True]}],
                     [{'query':'q','item_indices':[]}], [{'query':'x'*201,'item_indices':[0]}],
                     [{'query':'q','item_indices':[0]}]*4]:
            with self.assertRaises(ValueError):
                validate_subqueries(rows,['A','B'])

    def test_identical_rewrites_and_duplicate_queries_do_not_add_votes(self):
        rows = [{'query':'原问题','item_indices':[0]}, {'query':'查供电','item_indices':[1,0,1]},
                {'query':' 查 供电 ','item_indices':[0,1]}]
        self.assertEqual(validate_subqueries(rows,['A','B'],'原 问题'),
                         [{'query':'查供电','item_indices':[0,1]}])

    def test_subqueries_share_one_route_weight_and_duplicate_hits_get_one_vote(self):
        a,b={'chunk_id':'a'},{'chunk_id':'b'}
        rows=weighted_rrf([('main',[a,a],1),('sub0',[b],.5),('sub1',[b],.5)])
        self.assertAlmostEqual(rows[0][1],rows[1][1])
        self.assertEqual(rows[1][0]['retrieval_sources'],['sub0','sub1'])

    def test_fusion_keeps_item_caps_and_exposes_new_candidate_sources(self):
        a={'chunk_id':'a','canonical_name':'A'}; b={'chunk_id':'b','canonical_name':'B'}
        c={'chunk_id':'c','canonical_name':'A'}
        rows=fuse_routes([a,b],[],[[c]],item_names=['A','B'],per_item_k=1)
        self.assertEqual([d['chunk_id'] for d in rows],['a','b'])
        self.assertLessEqual(len(rows),2)

    def test_optional_branch_can_be_disabled_and_failures_do_not_override_main(self):
        with patch.dict(os.environ,{'SUBQUERY_RETRIEVAL_ENABLED':'false','EVAL_DISABLE_SUBQUERY':'false'}):
            self.assertFalse(subquery_enabled({}))
            self.assertTrue(subquery_enabled({'subquery_retrieval_enabled':True}))
        with patch.dict(os.environ,{'EVAL_DISABLE_SUBQUERY':'true'}):
            self.assertFalse(subquery_enabled({'subquery_retrieval_enabled':True}))

    def test_planner_extends_existing_call_and_invalid_subqueries_preserve_tool_selection(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        from app.query_process.subquery_retrieval import decomposition_instruction, SUBQUERY_VERSION
        tree=ast.parse((ROOT/'app/query_process/agent/nodes/node_retrieval_plan.py').read_text(encoding='utf-8-sig'))
        targets=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in
                 {'_parse_llm_json','_llm_decide_retrieval_plan'}]
        messages=[]
        def invoke(values):
            messages.append(values)
            return SimpleNamespace(content=json.dumps({'use_embedding':True,'use_hyde':False,'use_web':False,
                'subqueries':[{'query':'供电','item_indices':[99]}]}))
        namespace={'QueryGraphState':dict,'Dict':dict,'json':json,'is_eval_enabled':lambda key:True,
            '_build_signal_hints':lambda query:{},'get_llm_client':lambda **kwargs:SimpleNamespace(invoke=invoke),
            'HumanMessage':HumanMessage,'SystemMessage':SystemMessage,'subquery_enabled':subquery_enabled,
            'decomposition_instruction':decomposition_instruction,'validate_subqueries':validate_subqueries,
            'SUBQUERY_VERSION':SUBQUERY_VERSION,
            'logger':SimpleNamespace(error=lambda *args,**kwargs:None)}
        exec(compile(ast.Module(body=targets,type_ignores=[]),'planner','exec'),namespace)
        with patch.dict(os.environ,{'EVAL_DISABLE_SUBQUERY':'false'}):
            result=namespace['_llm_decide_retrieval_plan']({'original_query':'查供电','item_names':['A'],
                                                         'subquery_retrieval_enabled':True})
        self.assertEqual(len(messages),1)
        self.assertIn('subqueries',messages[0][1].content)
        self.assertTrue(result['use_embedding'])
        self.assertEqual(result['subqueries'],[])
        self.assertEqual(result['subquery_status'],'invalid_plan')

    def test_rrf_includes_expansion_only_when_enabled(self):
        from test_product_retrieval_nodes import load,ProductRetrievalNodesTests
        namespace=ProductRetrievalNodesTests().namespace()
        namespace['is_eval_enabled']=lambda name:False
        for name in ['_as_entity_list','reciprocal_rank_fusion','node_rrf']:
            function=load('app/query_process/agent/nodes/node_rrf.py',name,namespace)
        state={'session_id':'t','item_names':['A'],'context_selection_policy':'item_balanced',
               'embedding_chunks':[{'chunk_id':'a','canonical_name':'A'}],
               'subquery_chunks':[[{'chunk_id':'b','canonical_name':'A'}]]}
        with patch.dict(os.environ,{'EVAL_DISABLE_SUBQUERY':'false'}):
            off=function(dict(state,subquery_retrieval_enabled=False))
            on=function(dict(state,subquery_retrieval_enabled=True))
        self.assertEqual([d['chunk_id'] for d in off['rrf_chunks']],['a'])
        self.assertEqual([d['chunk_id'] for d in on['rrf_chunks']],['a','b'])

    def test_branch_batches_duplicate_query_text_and_enforces_product_filters(self):
        import sys
        from app.query_process.agent.nodes.node_search_subqueries import node_search_subqueries
        encodings, searches = [], []
        def search(**kw):
            searches.append(kw)
            return [[{'chunk_id':str(len(searches)), 'canonical_name':str(len(searches))}]]
        modules = {
            'app.lm.embedding_utils':SimpleNamespace(generate_embeddings=lambda texts:
                encodings.append(texts) or {'dense':[[1]],'sparse':[{1:1}]}),
            'app.clients.milvus_utils':SimpleNamespace(create_hybrid_search_requests=lambda **kw:kw,
                hybrid_search=search,get_milvus_client=lambda:object()),
            'app.query_process.agent.nodes.node_rrf':SimpleNamespace(_as_entity_list=lambda docs:docs)}
        state={'item_names':['A','B'],'original_query':'比较接口', 'subquery_retrieval_enabled':True,
               'retrieval_plan':{'subqueries':[{'query':'接口规格','item_indices':[0]},
                                               {'query':'接口规格','item_indices':[1]}]}}
        with patch.dict(sys.modules,modules),patch.dict(os.environ,{'EVAL_DISABLE_SUBQUERY':'false'}):
            result=node_search_subqueries(state)
        self.assertEqual(encodings,[['接口规格']])
        self.assertEqual([c['reqs']['expr'] for c in searches],['canonical_name in ["A"]','canonical_name in ["B"]'])
        self.assertEqual(result['subquery_trace']['search_calls'],2)
        self.assertEqual(len(result['subquery_chunks']),2)


if __name__=='__main__':
    unittest.main()
