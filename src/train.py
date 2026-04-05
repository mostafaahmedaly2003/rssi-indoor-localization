import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from src.config import MODEL_DIR
from src.dataset import generate_dataset

def train_all():
    df = generate_dataset()
    X = df.drop("location", axis=1).values
    y = df["location"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "knn": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1
        ),
        "svm": SVC(kernel="rbf", C=10, gamma="scale",
                   probability=True, random_state=42),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        trained[name] = model
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
        print(f"[SAVED] models/{name}.pkl")

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print("[SAVED] models/scaler.pkl")

    return trained, scaler, X_train_s, X_test_s, y_train, y_test
