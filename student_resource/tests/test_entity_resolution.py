import unittest

import numpy as np
import pandas as pd

from src.entity_resolution import (
    build_candidate_index,
    choose_threshold,
    comparison_features,
    generate_candidates,
    per_anchor_f05,
)


class EntityResolutionTests(unittest.TestCase):
    def test_blocking_returns_same_country_target_ids(self):
        targets = pd.DataFrame([
            {
                "entity_id": "S2-1",
                "business_name": "acme market",
                "business_address": "12 main street",
                "country": "us",
            },
            {
                "entity_id": "S3-2",
                "business_name": "acme market",
                "business_address": "12 main street",
                "country": "france",
            },
        ])
        index, columns = build_candidate_index(targets)

        rows = generate_candidates(
            ("S1-1", "acme market", "12 main street", "us"), index, columns
        )

        self.assertEqual([columns[3][row] for row in rows], ["S2-1"])

    def test_missed_match_is_not_scored_as_singleton(self):
        score = per_anchor_f05(
            np.array([], dtype=np.int32),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.float64),
            np.array([1], dtype=np.int32),
            0.5,
        )

        self.assertEqual(score, 0.0)

    def test_true_singleton_with_no_prediction_scores_one(self):
        score = per_anchor_f05(
            np.array([], dtype=np.int32),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.float64),
            np.array([0], dtype=np.int32),
            0.5,
        )

        self.assertEqual(score, 1.0)

    def test_threshold_search_finds_rare_high_score(self):
        anchor_rows = np.zeros(100, dtype=np.int32)
        labels = np.zeros(100, dtype=np.uint8)
        labels[0] = 1
        probabilities = np.full(100, 0.001)
        probabilities[0] = 0.99

        threshold, score = choose_threshold(
            anchor_rows, labels, probabilities, np.array([1], dtype=np.int32)
        )

        self.assertEqual(score, 1.0)
        self.assertGreater(threshold, 0.001)
        self.assertLessEqual(threshold, 0.99)

    def test_similarity_feature_contract(self):
        features = comparison_features(
            "acme market", "12 main street", "us",
            "acme market", "12 main street", "us",
        )

        self.assertEqual(len(features), 22)
        self.assertEqual(features[4], 1.0)
        self.assertEqual(features[11], 1.0)


if __name__ == "__main__":
    unittest.main()