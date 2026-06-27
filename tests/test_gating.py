#!/usr/bin/env python3
"""
Unit tests for the honeypot checks and scoring sub-functions.
"""

import os
import json
import unittest
from score import (
    check_honeypot,
    compute_title_relevance,
    compute_experience_fit,
    compute_location_fit,
    compute_notice_fit,
    compute_penalties
)
from precompute_features import extract_features_from_candidate

class TestScoringAndGating(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load sample candidates as fixtures
        fixture_path = "sample_candidates.json"
        if os.path.exists(fixture_path):
            with open(fixture_path, "r", encoding="utf-8") as f:
                cls.samples = json.load(f)
        else:
            cls.samples = []

    def test_sample_candidate_features_extraction(self):
        self.assertTrue(len(self.samples) > 0, "Sample candidates fixture is missing.")
        cand = self.samples[0]
        feat = extract_features_from_candidate(cand)
        self.assertEqual(feat["candidate_id"], cand["candidate_id"])
        self.assertIn("years_of_experience", feat)
        self.assertIn("skills", feat)

    def test_honeypot_salary_inversion(self):
        # Create a mock candidate with min salary > max salary
        mock_feat = {
            "candidate_id": "CAND_TEST001",
            "years_of_experience": 5.0,
            "career_durations": [60],
            "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 24}],
            "expected_salary_min": 45.0,
            "expected_salary_max": 30.0,  # Inverted!
            "is_honeypot": False
        }
        is_hp, reason = check_honeypot(mock_feat)
        self.assertTrue(is_hp)
        self.assertIn("salary", reason.lower())

    def test_honeypot_expert_zero_duration(self):
        # Create a mock candidate with expert skill and 0 duration
        mock_feat = {
            "candidate_id": "CAND_TEST002",
            "years_of_experience": 5.0,
            "career_durations": [60],
            "skills": [{"name": "Pinecone", "proficiency": "expert", "duration_months": 0}], # Inconsistent!
            "expected_salary_min": 20.0,
            "expected_salary_max": 30.0,
            "is_honeypot": False
        }
        is_hp, reason = check_honeypot(mock_feat)
        self.assertTrue(is_hp)
        self.assertIn("expert", reason.lower())

    def test_experience_fit_score(self):
        # Test exp fit bump function
        self.assertAlmostEqual(compute_experience_fit({"years_of_experience": 7.0}), 1.0)
        self.assertAlmostEqual(compute_experience_fit({"years_of_experience": 5.0}), 1.0)
        self.assertAlmostEqual(compute_experience_fit({"years_of_experience": 9.0}), 1.0)
        
        # Test values outside [5, 9] decay
        self.assertTrue(compute_experience_fit({"years_of_experience": 3.0}) < 1.0)
        self.assertTrue(compute_experience_fit({"years_of_experience": 12.0}) < 1.0)

    def test_location_fit_score(self):
        # Local
        self.assertEqual(compute_location_fit({"location": "Pune, Maharashtra", "country": "India", "willing_to_relocate": False}), 1.0)
        self.assertEqual(compute_location_fit({"location": "Noida", "country": "India", "willing_to_relocate": False}), 1.0)
        
        # Tier-1
        self.assertEqual(compute_location_fit({"location": "Bangalore", "country": "India", "willing_to_relocate": True}), 0.8)
        
        # Outside India
        self.assertEqual(compute_location_fit({"location": "San Francisco", "country": "USA", "willing_to_relocate": False}), 0.1)

    def test_notice_period_fit_score(self):
        # <= 30 days
        self.assertEqual(compute_notice_fit({"notice_period_days": 15}), 1.0)
        self.assertEqual(compute_notice_fit({"notice_period_days": 30}), 1.0)
        
        # > 30 days decays
        self.assertTrue(compute_notice_fit({"notice_period_days": 60}) < 1.0)
        self.assertTrue(compute_notice_fit({"notice_period_days": 120}) < compute_notice_fit({"notice_period_days": 60}))

if __name__ == "__main__":
    unittest.main()
