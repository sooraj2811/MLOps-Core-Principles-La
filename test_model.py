import os
import json

def test_winning_model_accuracy():
    # 1. Ensure the metrics file actually exists
    metrics_path = 'metrics.json'
    assert os.path.exists(metrics_path), f"CI Error: {metrics_path} was not found!"

    # 2. Load the metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # 3. Extract accuracy
    winning_accuracy = metrics.get('accuracy', 0.0)
    winning_model_type = metrics.get('model_type', 'Unknown')
    
    print(f"CI Pipeline Testing: Best Model ({winning_model_type}) has Accuracy: {winning_accuracy:.4f}")

    # 4. Enforce the Continuous Integration threshold (Accuracy >= 0.85)
    threshold = 0.85
    assert winning_accuracy >= threshold, (
        f"CI Pipeline Failed! Winning model accuracy ({winning_accuracy:.4f}) "
        f"is below the required threshold of {threshold}."
    )
    
    print("CI Pipeline Passed successfully! Accuracy threshold criteria met.")

if __name__ == "__main__":
    test_winning_model_accuracy()
