"""Main pipeline: train all models, evaluate, and save all plots."""
from src.train import train_all
from src.evaluate import evaluate_model
from src.visualize import (
    plot_confusion_matrices, plot_floor_map, plot_model_comparison
)

print("=== RSSI Indoor Localization ===\n")

trained, scaler, X_train, X_test, y_train, y_test = train_all()

results = {}
for name, model in trained.items():
    results[name] = evaluate_model(model, X_test, y_test, name=name.upper())

best_name = max(results, key=lambda k: results[k]["accuracy"])
print(f"\nBest model: {best_name.upper()} ({results[best_name]['accuracy']*100:.1f}%)")

plot_confusion_matrices(results, y_test)
plot_floor_map(best_name, results[best_name]["y_pred"], y_test)
plot_model_comparison(results)

print("\nAll outputs saved to outputs/")
print("Run 'python app.py' to start the Flask API")
