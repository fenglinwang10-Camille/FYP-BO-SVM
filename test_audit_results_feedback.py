import unittest
from audit_results_feedback import paired_counts


class PairedPredictionTests(unittest.TestCase):
    def test_discordant_counts_and_exact_binomial(self):
        result = paired_counts([True]*21+[False]*7, [False]*21+[True]*7)
        self.assertEqual(result['bo_only_correct'], 21)
        self.assertEqual(result['rs_only_correct'], 7)
        self.assertAlmostEqual(result['mcnemar_exact_two_sided_p'], .012540951371192932)

    def test_identical_predictions_have_no_discordant_pairs(self):
        result = paired_counts([True, False], [True, False])
        self.assertEqual(result['mcnemar_exact_two_sided_p'], 1.)
        self.assertEqual(result['accuracy_difference_pp'], 0.)


if __name__ == '__main__':
    unittest.main()
