import unittest
from unittest.mock import patch
from review_code import (
    load_data,
    preprocess_review,
    identify_unique_words,
    generate_relevance_scores,
    format_2d_list,
    train_machine_learning_models,
    determine_best_model,
)


class TestYourCode(unittest.TestCase):

    @patch('builtins.print')  # Mock print statements for testing
    def test_load_data(self, mock_print):
        url = "https://dgoldberg.sdsu.edu/515/appliance_reviews.json"
        try:
            data = load_data(url)
            self.assertIsInstance(data, list)
            self.assertTrue(mock_print.called)
        except Exception as e:
            print(f"Exception during load_data: {e}")

    def test_preprocess_review(self):
        review = "Sample Review"
        processed_review = preprocess_review(review)
        self.assertEqual(processed_review, "sample review")

    def test_identify_unique_words(self):
        review_string = "This is a sample review string."
        review_string_after_words = "This is a sample review string"
        unique_words = identify_unique_words(review_string)
        self.assertIsInstance(unique_words, set)
        self.assertEqual(set(review_string_after_words.split()), unique_words)

    def test_generate_relevance_scores(self):
        data = [{"Review": "sample review", "Safety hazard": 1}]  # Mock data
        unique_words = {"sample", "review"}  # Mock unique words
        relevance_scores = generate_relevance_scores(data, unique_words)
        self.assertIsInstance(relevance_scores, list)

    def test_format_2d_list(self):
        data = [{"Review": "sample review", "Safety hazard": 1}]  # Mock data
        relevant_words = ["sample", "words"]  # Mock relevant words
        x, y = format_2d_list(data, relevant_words)
        self.assertIsInstance(x, list)
        self.assertIsInstance(y, list)

    def test_train_machine_learning_models(self):
        x = [[1, 0], [0, 1], [1, 1]]  # Mock x
        y = [1, 0, 1]  # Mock y
        dt_accuracy, kn_accuracy, nn_accuracy = train_machine_learning_models(x, y)
        self.assertIsInstance(dt_accuracy, float)
        self.assertIsInstance(kn_accuracy, float)
        self.assertIsInstance(nn_accuracy, float)

    def test_determine_best_model(self):
        dt_accuracy, kn_accuracy, nn_accuracy = 0.8, 0.75, 0.85
        best_model = determine_best_model(
            dt_accuracy, kn_accuracy, nn_accuracy
            )
        self.assertEqual(best_model, "Neural Network")


if __name__ == '__main__':
    unittest.main()
