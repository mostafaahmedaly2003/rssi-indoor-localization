import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from src.config import LOCATION_NAMES, LOCATIONS

def mean_distance_error(y_true, y_pred):
    """Domain-specific metric: average distance error in meters."""
    errors = []
    for true_id, pred_id in zip(y_true, y_pred):
        tx, ty = LOCATIONS[true_id]
        px, py = LOCATIONS[pred_id]
        errors.append(np.sqrt((tx - px)**2 + (ty - py)**2))
    return np.mean(errors)

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    mde = mean_distance_error(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"{name}")
    print(f"  Accuracy : {acc*100:.1f}%")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Mean Distance Error: {mde:.2f} m")
    print(f"{'='*40}")
    print(classification_report(
        y_test, y_pred,
        target_names=[LOCATION_NAMES[i] for i in sorted(LOCATION_NAMES)]
    ))
    return {"accuracy": acc, "f1": f1, "mde": mde, "y_pred": y_pred}
