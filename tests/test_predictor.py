import unittest
import math
from src.scale_predictor.predictor import ScalePredictor

class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = ScalePredictor()

    def test_train_no_data(self):
        """
        When input data is empty, `trained` should be False。
        """
        dataset = {}
        window_size = 60
        self.predictor.train(dataset, window_size)

        self.assertFalse(self.predictor.trained, "`trained` should be False")
        self.assertEqual(len(self.predictor.models), 0, "empty dataset should not be used to train any model")

    def test_train_insufficient_data(self):
        """
        Should not generate a model when there are less than 2 invocation data for that function.
        """
        dataset = {
            "funcA": [10],
            "funcB": [1, 2, 3, 4, 5]
        }
        window_size = 2
        self.predictor.train(dataset, window_size)

        self.assertTrue(self.predictor.trained)
        self.assertIn("funcB", self.predictor.models)
        self.assertNotIn("funcA", self.predictor.models)

    def test_train_and_predict_single_function(self):
        """
        Train and predict for a single function, checking the usability and output of the model.
        """
        dataset = {
            "funcA": [i for i in range(10)]  # naive increasing data 0,1,2,...,9
        }
        window_size = 3
        self.predictor.train(dataset, window_size)

        self.assertTrue(self.predictor.trained, "`trained` should be True")
        self.assertIn("funcA", self.predictor.models, "no model for funcA")

        window = [7, 8, 9] 
        predicted = self.predictor.predict("funcA", window, index=2)

        self.assertGreaterEqual(predicted, 0, f"The predicted value should be greater than or equal to zero, but it turns out to be {predicted}")

    def test_train_and_predict_multiple_functions(self):
        """
        multi-trainning
        """
        dataset = {
            "funcA": [1, 2, 3, 4, 5, 6],
            "funcB": [10, 9, 8, 7, 6, 5],
        }
        window_size = 2
        self.predictor.train(dataset, window_size)

        self.assertTrue(self.predictor.trained, "`trained` should be True")
        self.assertIn("funcA", self.predictor.models)
        self.assertIn("funcB", self.predictor.models)

        # predict for funcA
        windowA = [4, 5] 
        predA = self.predictor.predict("funcA", windowA, index=1)
        self.assertGreaterEqual(predA, 0)

        # predict for funcB
        windowB = [6, 5] 
        predB = self.predictor.predict("funcB", windowB, index=1)
        self.assertGreaterEqual(predB, 0)

    def test_predict_without_training(self):
        """
        If predict(not_existing_func), should throw an error
        """
        with self.assertRaises(RuntimeError):
            self.predictor.predict("notExistFunc", [1, 2, 3], 2)

    def test_predict_unregistered_function(self):
        """
        If predict(not_existing_func), should throw an error
        """
        dataset = {
            "funcA": [1, 2, 3, 4],
        }
        window_size = 2
        self.predictor.train(dataset, window_size)

        with self.assertRaises(RuntimeError):
            self.predictor.predict("notExistFunc", [1, 2], 1)

    def test_predict_negative_result(self):
        """
        Negative predict values are possible, 0 should be returned at these cases.
        """
        dataset = {
            "funcA": [10, 0, -10, -20, -30, -40]
        }
        window_size = 3
        self.predictor.train(dataset, window_size)

        window = [-20, -30, -40]
        predicted = self.predictor.predict("funcA", window, 2)
        
        self.assertEqual(predicted, 0, f"should be 0, truns out to be {predicted}")

    def test_clear(self):
        dataset = {
            "funcA": [1, 2, 3, 4, 5],
        }
        window_size = 2
        self.predictor.train(dataset, window_size)

        self.assertTrue(self.predictor.trained)
        self.assertIn("funcA", self.predictor.models)

        self.predictor.clear()

        self.assertFalse(self.predictor.trained)
        self.assertEqual(len(self.predictor.models), 0)
        self.assertEqual(self.predictor.window_size, 0)

if __name__ == '__main__':
    unittest.main()
