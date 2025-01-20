from src.scale_predictor.predictor import ScalePredictor

if __name__ == "__main__":
    dataset = {
        "funcA": [10, 12, 15, 18, 20, 26, 30, 28, 23, 0, 23, 54, 34, 4, 435, 54, 0, 0, 0, 34, 4, 3, 2, 5, 0, 1],
        "funcB": [2, 3, 5, 7, 9, 9, 10, 5, 34, 23, 5, 1, 1, 0]
    }
    predictor = ScalePredictor()

    # Training.
    predictor.train(dataset, window_size=5)

    current_window_a = [35, 32, 31, 40, 50, 36, 3]
    needed_a = predictor.predict("funcA", current_window_a, 6)
    print(f"Predicted needed instances for funcA = {needed_a}")

    current_window_b = [9, 12, 14, 8, 11, 99]
    needed_b = predictor.predict("funcB", current_window_b, 5)
    print(f"Predicted needed instances for funcB = {needed_b}")

    predictor.clear()
