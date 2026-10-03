import unittest

import pandas as pd
from sklearn.pipeline import Pipeline

from data.generate_dataset import generate_dataset
from pipeline import FEATURES, TARGET, load_and_clean_data, train_and_compare


class DatasetTests(unittest.TestCase):
    def test_generator_enforces_minimum_record_count(self):
        with self.assertRaises(ValueError):
            generate_dataset(record_count=999)

    def test_generated_dataset_has_required_rows_missing_values_and_binary_target(self):
        data = generate_dataset()
        self.assertGreaterEqual(len(data), 1000)
        self.assertEqual(set(data[TARGET].unique()), {0, 1})
        self.assertTrue(data[FEATURES].isna().any().any())
        self.assertNotIn("review_cycle_hours", FEATURES)
        self.assertIn("test_coverage_percent", FEATURES)
        self.assertIn("lint_findings", FEATURES)

    def test_cleaner_drops_duplicate_ids_and_invalid_outcomes(self):
        data = generate_dataset(record_count=1000).head(50).copy()
        duplicate = data.iloc[[0]].copy()
        invalid = data.iloc[[1]].copy()
        invalid.loc[:, "pr_id"] = "PR-INVALID"
        invalid.loc[:, "review_cycle_hours"] = -1
        test_data = pd.concat([data, duplicate, invalid], ignore_index=True)
        path = "test_pr_dataset.csv"
        try:
            test_data.to_csv(path, index=False)
            clean, report = load_and_clean_data(path)
        finally:
            import os

            if os.path.exists(path):
                os.remove(path)
        self.assertEqual(report["duplicate_ids_removed"], 1)
        self.assertEqual(report["invalid_rows_removed"], 1)
        self.assertEqual(len(clean), 50)

    def test_two_models_train_and_are_evaluated_on_same_holdout(self):
        data = generate_dataset()
        models, scores, x_test, y_test = train_and_compare(data)
        self.assertEqual(set(models), {"Logistic Regression", "Random Forest"})
        self.assertEqual(len(x_test), len(y_test))
        self.assertEqual(set(scores.columns), {"Accuracy", "Precision", "Recall", "F1", "ROC-AUC"})
        self.assertTrue(all(isinstance(model, Pipeline) for model in models.values()))
        self.assertTrue(scores.apply(lambda column: column.between(0, 1).all()).all())


if __name__ == "__main__":
    unittest.main()
