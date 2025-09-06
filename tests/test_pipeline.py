import unittest
import json
import os
import hashlib
from dotenv import load_dotenv

# Load environment variables for tests
load_dotenv()

from unittest.mock import patch, MagicMock
from ai_agent import process_for_role, PIPELINES, create_workflow
from lead_tracker import store_new_leads, get_prior_leads
from main import run_pipeline

class TestPipeline(unittest.TestCase):

    def setUp(self):
        # Ensure Firestore and HTTP clients are mocked for all tests
        self.firestore_patch = patch('google.cloud.firestore.Client', autospec=True)
        self.requests_patch = patch('requests.get', autospec=True)
        self.firestore_patch.start()
        self.requests_patch.start()

    def tearDown(self):
        self.firestore_patch.stop()
        self.requests_patch.stop()

    def _test_process_for_role(self, config):
        # Always mock both relevance and novelty LLM calls, with novelty_score=0.9 for new leads
        with patch('ai_agent.llm.__call__') as mock_llm_call, \
            patch('ai_agent.search_web_for_signals._run', return_value=[
                {"content": f"Startup XYZ seeking {config['role']}.", "timestamp": 1234567890, "source_url": "http://example.com"}
            ]) as mock_search, \
            patch('lead_tracker.get_prior_leads', return_value={}) as mock_get_priors, \
            patch('lead_tracker.store_new_leads') as mock_store:
            # First call: relevance, second call: novelty
            mock_llm_call.side_effect = [
                MagicMock(content=json.dumps({
                    "company": "Startup XYZ",
                    "profile": config["role"],
                    "contacts": ["email@example.com"],
                    "context": f"Seeking {config['role']}",
                    "status": "NEW"
                })),
                MagicMock(content=json.dumps({
                    "novelty_score": 0.9,
                    "reason": "New information"
                }))
            ]
            leads = process_for_role(config)
            self.assertEqual(len(leads), 1, f"Expected 1 lead for {config['role']}, got {len(leads)}")
            self.assertEqual(leads[0]['company'], "Startup XYZ")
            self.assertEqual(leads[0]['role'], config['role'])
            self.assertEqual(leads[0]['status'], "NEW")
            mock_store.assert_called_once_with(leads, config["collection_name"])

    def _test_process_for_role_duplicate(self, config):
        with patch('ai_agent.llm.__call__') as mock_llm_call, \
            patch('ai_agent.search_web_for_signals._run', return_value=[
                {"content": f"Startup XYZ seeking {config['role']}.", "timestamp": 1234567890, "source_url": "http://example.com"}
            ]) as mock_search, \
            patch('lead_tracker.get_prior_leads', return_value={
                "XYZ": [{
                    "context": f"Seeking {config['role']}",
                    "status": "NEW",
                    "hash": hashlib.sha256(f"Seeking {config['role']}".encode("utf-8")).hexdigest()
                }]
            }) as mock_get_priors, \
            patch('lead_tracker.store_new_leads') as mock_store:
            # Mock LLM responses for relevance and novelty
            mock_llm_call.side_effect = [
                MagicMock(content=json.dumps({
                    "score": 0.8,
                    "company": "XYZ",
                    "profile": "Startup",
                    "contacts": ["email@example.com"],
                    "context": f"Seeking {config['role']}",
                    "status": "NEW"
                })),
                MagicMock(content=json.dumps({
                    "novelty_score": 0.4,
                    "reason": "Duplicate"
                }))
            ]
            leads = process_for_role(config)
            self.assertEqual(len(leads), 0, f"Expected 0 leads for {config['role']}, got {len(leads)}")
            mock_store.assert_not_called()
        def _test_process_for_role_duplicate(self, config):
            # Mock both relevance and novelty LLM calls for duplicate scenario
            with patch('ai_agent.llm.__call__') as mock_llm_call, \
                patch('ai_agent.search_web_for_signals._run', return_value=[
                    {"content": f"Startup XYZ seeking {config['role']}.", "timestamp": 1234567890, "source_url": "http://example.com"}
                ]) as mock_search, \
                patch('lead_tracker.get_prior_leads', return_value={
                    "Startup XYZ": [{
                        "context": f"Seeking {config['role']}",
                        "status": "NEW",
                        "hash": hashlib.sha256(f"Seeking {config['role']}".encode("utf-8")).hexdigest(),
                        "timestamp": 1234567890,
                        "source_url": "http://example.com"
                    }]
                }) as mock_get_priors, \
                patch('lead_tracker.store_new_leads') as mock_store:
                # First call: relevance, second call: novelty
                mock_llm_call.side_effect = [
                    MagicMock(content=json.dumps({
                        "company": "Startup XYZ",
                        "profile": config["role"],
                        "contacts": ["email@example.com"],
                        "context": f"Seeking {config['role']}",
                        "status": "NEW"
                    })),
                    MagicMock(content=json.dumps({
                        "novelty_score": 0.0,
                        "reason": "Duplicate"
                    }))
                ]
                leads = process_for_role(config)
                self.assertEqual(len(leads), 0, f"Expected 0 leads for {config['role']}, got {len(leads)}")
                mock_store.assert_not_called()

    def test_fractional_cto_new_lead(self):
        self._test_process_for_role(PIPELINES[0])

    def test_fractional_cto_duplicate(self):
        self._test_process_for_role_duplicate(PIPELINES[0])

    def test_part_time_software_engineer_new_lead(self):
        self._test_process_for_role(PIPELINES[1])

    def test_part_time_software_engineer_duplicate(self):
        self._test_process_for_role_duplicate(PIPELINES[1])

    def test_part_time_data_engineer_new_lead(self):
        self._test_process_for_role(PIPELINES[2])

    def test_part_time_data_engineer_duplicate(self):
        self._test_process_for_role_duplicate(PIPELINES[2])

    @patch('ai_agent.process_all_pipelines', return_value=[{"company": "Test", "role": "fractional_cto", "status": "NEW"}])
    def test_run_pipeline_end_to_end(self, mock_process):
        result = run_pipeline()
        self.assertEqual(result, "Pipeline complete: 1 leads saved")

    def test_lead_tracker_for_each_collection(self):
        with patch('google.cloud.firestore.Client') as mock_firestore_client:
            # Mock Firestore client and collection
            mock_client = MagicMock()
            mock_firestore_client.return_value = mock_client
            mock_collection = MagicMock()
            mock_client.collection.return_value = mock_collection
            mock_doc_ref = MagicMock()
            mock_collection.document.return_value = mock_doc_ref
            mock_doc_ref.get.return_value.exists = False
            mock_doc_ref.get.return_value.to_dict.return_value = {}
            mock_doc_ref.set = MagicMock()

            for config in PIPELINES:
                test_leads = [
                    {
                        "company": "TestCo",
                        "context": "Test",
                        "timestamp": 123,
                        "source_url": "url",
                        "role": config["role"],
                        "status": "NEW"
                    }
                ]
                # Store lead once
                store_new_leads(test_leads, config["collection_name"])
                # Store same lead again (duplicate)
                store_new_leads(test_leads, config["collection_name"])

                # Mock get_prior_leads to return only one signal
                mock_doc_ref.get.return_value.exists = True
                mock_doc_ref.get.return_value.to_dict.return_value = {
                    "signals": [{
                        "context": "Test",
                        "timestamp": 123,
                        "source_url": "url",
                        "status": "NEW",
                        "hash": hashlib.sha256("Test".encode()).hexdigest()
                    }]
                }
                priors = get_prior_leads(config["collection_name"])
                self.assertIn("TestCo", priors)
                self.assertEqual(len(priors["TestCo"]), 1, f"Expected 1 signal for {config['role']}")
                self.assertEqual(priors["TestCo"][0]["status"], "NEW")

if __name__ == '__main__':
    unittest.main()