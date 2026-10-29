"""Review provenance, immutable publication, stale edits and API input boundaries."""

import copy
from pathlib import Path
import shutil
import tempfile
import unittest

from fastapi.testclient import TestClient
from app.evaluation.workbench import Workbench, ConflictError
from app.evaluation.api import create_app

EXAMPLE = Path(__file__).resolve().parents[1] / 'eval' / 'examples' / 'retrieval'


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(EXAMPLE, self.root / 'eval' / 'examples' / 'retrieval')
        self.workbench = Workbench(self.root)

    def decision(self):
        return {'reviewer': 'test reviewer', 'notes': 'Reviewed in isolated test fixture',
                'human_verified': False, 'query': {}}

    def test_stale_edits_rejected_and_all_revisions_preserved(self):
        self.workbench.save_review('example', 'DEMO-001', 0, self.decision())
        with self.assertRaises(ConflictError):
            self.workbench.save_review('example', 'DEMO-001', 0, self.decision())
        self.workbench.save_review('example', 'DEMO-001', 1, self.decision())
        with self.workbench.connection() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM reviews').fetchone()[0], 2)

    def test_publishing_uses_latest_decision_and_preserves_source(self):
        before = self.workbench.benchmark('example').fingerprint
        old = self.workbench.save_review('example', 'DEMO-001', 0, self.decision())
        latest = self.workbench.save_review('example', 'DEMO-001', 1, self.decision())
        with self.assertRaises(ConflictError):
            self.workbench.publish('example', 'new', '2.0', [old['id']])
        self.workbench.publish('example', 'new', '2.0', [latest['id']])
        self.assertEqual(self.workbench.benchmark('example').fingerprint, before)
        reviewed = self.workbench.benchmark('new').queries['DEMO-001']['review']
        self.assertFalse(reviewed['human_verified'])
        self.assertEqual(reviewed['reviewer_type'], 'explicit_source_review')
        with self.assertRaises(ConflictError):
            self.workbench.publish('example', 'new2', '3.0', [latest['id']])

    def test_invalid_evidence_cannot_be_saved_as_reviewed(self):
        decision = self.decision()
        decision['qrel_patches'] = [{'chunk_id': 'does-not-exist', 'relevance': 2, 'evidence_ids': []}]
        with self.assertRaisesRegex(ValueError, '未知chunk'):
            self.workbench.save_review('example', 'DEMO-001', 0, decision)
        self.assertEqual(self.workbench.latest_reviews(self.workbench.benchmark('example').fingerprint), {})

    def test_second_server_cannot_recover_or_dispatch_another_servers_jobs(self):
        with self.workbench.server_lock():
            with self.assertRaises(ConflictError):
                with self.workbench.server_lock():
                    self.fail('second server must not acquire dispatch ownership')

    def test_api_rejects_cross_origin_writes_invalid_grades_and_path_traversal(self):
        with TestClient(create_app(self.root)) as client:
            body = dict(self.decision(), expected_revision=0)
            url = '/api/datasets/example/queries/DEMO-001/reviews'
            self.assertEqual(client.post(url, json=body, headers={'Origin': 'https://unrelated.example'}).status_code, 403)
            body['qrel_patches'] = [{'chunk_id': 'x', 'relevance': 9}]
            self.assertEqual(client.post(url, json=body).status_code, 422)
            self.assertEqual(client.get('/api/datasets/example/queries').status_code, 200)
            self.assertEqual(client.get('/api/datasets/not..allowed/queries').status_code, 400)


if __name__ == '__main__':
    unittest.main()
