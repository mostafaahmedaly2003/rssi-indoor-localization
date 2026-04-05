# RSSI Indoor Localization — Claude Notes

## Project Overview
Python ML project that predicts indoor position from WiFi RSSI fingerprints using KNN, MLP, and SVM classifiers.

## Entry Points
- `python main.py` — full training + evaluation + plot pipeline
- `python app.py` — start Flask REST API on port 5000
- `python data/generate_dataset.py` — export dataset to CSV

## Key Files
- `src/config.py` — all constants (floor layout, AP positions, paths)
- `src/dataset.py` — synthetic RSSI data generation via path-loss model
- `src/train.py` — trains KNN/MLP/SVM, saves .pkl files to models/
- `src/evaluate.py` — accuracy, F1, mean distance error metrics
- `src/visualize.py` — confusion matrices, floor map, model comparison plots

## Models saved to models/
- knn.pkl, mlp.pkl, svm.pkl, scaler.pkl

## Outputs saved to outputs/
- confusion_matrices.png, floor_map.png, model_comparison.png
