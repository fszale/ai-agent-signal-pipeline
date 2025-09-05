import unittest
import json
from unittest.mock import patch
from ai_agent import process_for_role, PIPELINES
from lead_tracker import store_new_leads, get_prior_leads
from main import run_pipeline

class TestPipeline(unittest.TestCase):

    def _test_process_for_role(self, config):
        with patch('ai_agent.search_web_for_signals') as mock_search, \
             patch('ai_agent.agent.run') as mock_agent_run, \
             patch('lead_tracker.get_prior_leads') as mock_get_priors, \
             patch('lead_tracker.store_new_leads') as mock_store:

            # Mock signals
            mock_search.return_value = [
                {"content": f"Startup XYZ seeking {config['role']}.", "timestamp": 1234567890, "source_url": "http://example.com"}
            ]
            
            # Mock relevance: High score
            mock_agent_run.side_effect = [
                json.dumps({"score": 0.8, "company": "XYZ", "profile": "Startup", "contacts": ["email@example.com"], "context": f"Seeking {config['role']}", "status": "NEW"}),
                # No novelty if new
            ]
            
            mock_get_priors.return_value = {}  # No priors
            
            leads = process_for_role(config)
            
            self.assertEqual(len(leads), 1)
            self.assertEqual(leads[0]['company'], "XYZ")
            self.assertEqual(leads[0]['role'], config['role'])
            self.assertEqual(leads[0]['status'], "NEW")
            mock_store.assert_called_once()

    def _test_process_for_role_duplicate(self, config):
        with patch('ai_agent.search_web_for_signals') as mock_search, \
             patch('ai_agent.agent.run') as mock_agent_run, \
             patch('lead_tracker.get_prior_leads') as mock_get_priors, \
             patch('lead_tracker.store_new_leads') as mock_store:

            mock_search.return_value = [
                {"content": f"Startup XYZ seeking {config['role']}.", "timestamp": 1234567890, "source_url": "http://example.com"}
            ]
            
            mock_agent_run.side_effect = [
                json.dumps({"score": 0.8, "company": "XYZ", "profile": "Startup", "contacts": ["email@example.com"], "context": f"Seeking {config['role']}", "status": "NEW"}),
                json.dumps({"novelty_score": 0.4, "reason": "Duplicate"})  # Low novelty
            ]
            
            mock_get_priors.return_value = {"XYZ": [{"context": f"Seeking {config['role']}", "status": "NEW"}]}  # Prior exists
            
            leads = process_for_role(config)
            
            self.assertEqual(len(leads), 0)
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

    @patch('ai_agent.process_all_pipelines')
    def test_run_pipeline_end_to_end(self, mock_process):
        mock_process.return_value = [{"company": "Test", "role": "fractional_cto", "status": "NEW"}]
        result = run_pipeline()
        self.assertEqual(result, "Pipeline complete: 1 leads saved")

    def test_lead_tracker_for_each_collection(self):
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
            store_new_leads(test_leads, config["collection_name"])
            priors = get_prior_leads(config["collection_name"])
            self.assertIn("TestCo", priors)
            self.assertEqual(priors["TestCo"][0]["status"], "NEW")

if __name__ == '__main__':
    unittest.main()