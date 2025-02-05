import unittest
import math
import os
import tempfile
from src.scale_predictor.predictor import ScalePredictor

class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = ScalePredictor('0')

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
        with self.assertRaises(KeyError):
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

        with self.assertRaises(KeyError):
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

    def test_export(self):
        """
        Test that the export method correctly saves the predictor's state to a file.
        """
        dataset = {
            "funcA": [1, 2, 3, 4, 5, 6],
            "funcB": [10, 9, 8, 7, 6, 5],
        }
        window_size = 2
        self.predictor.train(dataset, window_size)

        with tempfile.TemporaryDirectory() as tmpdirname:
            export_path = os.path.join(tmpdirname, 'scale_predictor.joblib')
            self.predictor.export(export_path)

            # Check if the file exists
            self.assertTrue(os.path.exists(export_path), "Export file was not created.")

            # Check file size is greater than zero
            self.assertGreater(os.path.getsize(export_path), 0, "Export file is empty.")

    def test_load(self):
        """
        Test that the load method correctly restores the predictor's state from a file.
        """
        dataset = {
            "funcA": [1, 2, 3, 4, 5, 6],
            "funcB": [10, 9, 8, 7, 6, 5],
        }
        window_size = 2
        self.predictor.train(dataset, window_size)

        with tempfile.TemporaryDirectory() as tmpdirname:
            export_path = os.path.join(tmpdirname, 'scale_predictor.joblib')
            self.predictor.export(export_path)

            # Create a new predictor and load the exported state
            new_predictor = ScalePredictor('0')
            new_predictor.load(export_path)

            # Verify that the loaded predictor has the same state
            self.assertTrue(new_predictor.trained, "Loaded predictor should be trained.")
            self.assertEqual(new_predictor.window_size, self.predictor.window_size, "Window size mismatch after loading.")
            self.assertEqual(set(new_predictor.models.keys()), set(self.predictor.models.keys()), "Function names mismatch after loading.")

            # Verify predictions are consistent
            windowA = [4, 5] 
            predA_original = self.predictor.predict("funcA", windowA, index=1)
            predA_loaded = new_predictor.predict("funcA", windowA, index=1)
            self.assertEqual(predA_loaded, predA_original, "Predictions mismatch for funcA after loading.")

            windowB = [6, 5] 
            predB_original = self.predictor.predict("funcB", windowB, index=1)
            predB_loaded = new_predictor.predict("funcB", windowB, index=1)
            self.assertEqual(predB_loaded, predB_original, "Predictions mismatch for funcB after loading.")

    def test_export_and_load(self):
        """
        Combined test to export a predictor, clear it, load it back, and verify state and predictions.
        """
        dataset = {
            "funcA": [2, 4, 6, 8, 10, 12],
            "funcB": [12, 10, 8, 6, 4, 2],
        }
        window_size = 3
        self.predictor.train(dataset, window_size)

        # Prepare a temporary file for exporting
        with tempfile.TemporaryDirectory() as tmpdirname:
            export_path = os.path.join(tmpdirname, 'scale_predictor_combined.joblib')
            self.predictor.export(export_path)

            # Clear the predictor
            self.predictor.clear()
            self.assertFalse(self.predictor.trained)
            self.assertEqual(len(self.predictor.models), 0)
            self.assertEqual(self.predictor.window_size, 0)

            # Load the predictor back
            self.predictor.load(export_path)

            # Verify the state
            self.assertTrue(self.predictor.trained, "Predictor should be trained after loading.")
            self.assertEqual(self.predictor.window_size, window_size, "Window size mismatch after loading.")
            self.assertIn("funcA", self.predictor.models)
            self.assertIn("funcB", self.predictor.models)

            # Verify predictions
            windowA = [6, 8, 10]
            predA = self.predictor.predict("funcA", windowA, index=2)
            expected_predA = math.ceil(self.predictor.models["funcA"].predict([windowA])[0])
            expected_predA = max(expected_predA, 0)
            self.assertEqual(predA, expected_predA, "Prediction for funcA mismatch after loading.")

            windowB = [8, 6, 4]
            predB = self.predictor.predict("funcB", windowB, index=2)
            expected_predB = math.ceil(self.predictor.models["funcB"].predict([windowB])[0])
            expected_predB = max(expected_predB, 0)
            self.assertEqual(predB, expected_predB, "Prediction for funcB mismatch after loading.")


if __name__ == '__main__':
    unittest.main()
