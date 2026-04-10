# RSSI-Based Indoor Localization

A machine learning system that predicts indoor position from WiFi signal strength (RSSI) fingerprints using KNN, MLP, and SVM classifiers — with a production-ready Flask REST API.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-000000?style=flat&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> **Related research:** This project is in the same research domain as the Nature-published paper  
> *"Real-time Detection of Wi-Fi Attacks using Hybrid Deep Learning on NodeMCU"* —  
> [Read the paper](https://www.nature.com/articles/YOUR_DOI_HERE) <!-- Replace with your actual DOI -->

---

## Results Summary

| Classifier | Accuracy | Mean Localization Error |
|-----------|----------|------------------------|
| **SVM** | **91.1%** | **0.19 m** |
| KNN | 90.0% | 0.21 m |
| MLP | 80.0% | 0.41 m |

SVM achieves the best spatial accuracy at under 20 cm mean error.

---

## How It Works

1. **Data simulation** — Free-Space Path Loss model across 6 access points at 9 reference locations with Gaussian noise (σ = 3 dBm) → 900 samples
2. **Preprocessing** — Feature standardization (StandardScaler)
3. **Training** — 80/20 train/test split (720 train, 180 test)
4. **Evaluation** — Accuracy, F1 score, mean distance error (meters)
5. **Deployment** — Flask REST API for real-time location prediction

---

## Architecture

```
rssi-indoor-localization/
├── main.py              # Full pipeline: train → evaluate → visualize
├── app.py               # Flask REST API server
├── src/
│   ├── data_gen.py      # RSSI dataset generation
│   ├── models.py        # KNN, SVM, MLP training & evaluation
│   └── visualize.py     # Confusion matrices, floor maps, comparisons
├── data/                # Generated datasets (.csv)
├── outputs/             # Saved models (.pkl), charts, floor maps
├── requirements.txt
└── README.md
```

---

## Installation & Usage

```bash
git clone https://github.com/mostafaahmedaly2003/rssi-indoor-localization.git
cd rssi-indoor-localization
pip install -r requirements.txt

# Run full ML pipeline (data gen → train → evaluate → visualize)
python main.py

# Start Flask prediction API
python app.py
```

---

## API Usage

**POST** `/predict` — Submit RSSI readings, get predicted location

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"rssi": [-60, -55, -70, -65, -50, -75]}'
```

**Response:**
```json
{
  "location_id": 3,
  "location_name": "Room B - Desk Area",
  "confidence": 94.2,
  "probabilities": {
    "Room A": 0.02,
    "Room B": 0.942,
    "Corridor": 0.038
  }
}
```

---

## Visualizations

The pipeline generates:
- Confusion matrices for all 3 classifiers
- Floor map with predicted vs. actual positions
- Classifier accuracy/error comparison chart
- Training pipeline flow diagram

---

## Tech Stack

- **Python 3.8+**
- scikit-learn — KNN, SVM, MLP classifiers
- Flask — REST API
- pandas, NumPy — data processing
- matplotlib, seaborn — visualization
- joblib — model persistence (.pkl)

---

## Practical Context

GPS is ineffective indoors. RSSI fingerprinting solves this for hospitals, warehouses, airports, and smart buildings — using existing WiFi infrastructure with no additional hardware.

---

## Related Work

This project shares the WiFi signal analysis domain with my published research on edge AI security:

> **Real-time Detection of Wi-Fi Attacks using Hybrid Deep Learning on NodeMCU**  
> *Scientific Reports — Nature Portfolio, 2025* | 5,900+ reads  
> [Read the paper](https://www.nature.com/articles/YOUR_DOI_HERE)

---

## Author

**Mostafa Ahmed** — AI/ML Engineer & Researcher  
[LinkedIn](https://www.linkedin.com/in/mostafa-ahmed-ai/) · [GitHub](https://github.com/mostafaahmedaly2003)
