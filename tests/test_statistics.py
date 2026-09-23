"""Protocol checks for the paired, end-to-end scoring path."""
import unittest

from jevbench.runner import evaluate_dataset
from jevbench.schemas import Prediction, Sample
from jevbench.statistics import correctness, mcnemar_exact


class StatisticsProtocolTest(unittest.TestCase):
    def test_failures_count_wrong_and_clinc_oos_rule_applies(self) -> None:
        gold = {"task_type": "choice", "label": "oos", "labels": ["banking", "oos"]}
        valid = {"format_valid": True, "answer": "banking", "probabilities": {"banking": 0.3, "oos": 0.2}}
        self.assertEqual(correctness(gold, valid), 1)
        self.assertEqual(correctness(gold, {**valid, "format_valid": False}), 0)
        self.assertEqual(correctness(gold, {**valid, "error": "network: timeout"}), 0)

    def test_low_confidence_is_not_abstention_without_oos_option(self) -> None:
        gold = {"task_type": "choice", "label": "banking", "labels": ["banking", "cards"]}
        pred = {"format_valid": True, "answer": "banking", "probabilities": {"banking": 0.3, "cards": 0.2}}
        self.assertEqual(correctness(gold, pred), 1)
        sample = Sample(id="x", dataset="banking77", split="test", text="example", lang="en",
                        task_type="choice", label="banking", labels=["banking", "cards"])
        prediction = Prediction(id="x", dataset="banking77", model="test", task_type="choice",
                                answer="banking", probabilities=pred["probabilities"])
        self.assertEqual(evaluate_dataset([sample], [prediction], "choice", sample.labels)["quality"]["acc"], 1.0)

    def test_score_uses_argmax_and_noul_uses_probability(self) -> None:
        score = {"task_type": "score", "label": 3}
        self.assertEqual(correctness(score, {"format_valid": True, "answer": 2.6, "probabilities": {"2": 0.4, "3": 0.6}}), 1)
        noul = {"task_type": "noul", "label": False}
        self.assertEqual(correctness(noul, {"format_valid": True, "answer": True, "probabilities": {"true": 0.49}}), 1)

    def test_mcnemar_uses_discordant_pairs(self) -> None:
        result = mcnemar_exact([1, 1, 1, 0], [0, 0, 1, 1])
        self.assertEqual(result["reference_only_correct"], 2)
        self.assertEqual(result["candidate_only_correct"], 1)
        self.assertEqual(result["two_sided_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
