# Generated from: knn-diabetic.ipynb
# Converted at: 2026-11-24T05:17:42.138Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ============================================================
# DIABETIC RETINOPATHY CLASSIFICATION USING KNN
# ============================================================

!pip -q install scikit-image

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from skimage.feature import hog, local_binary_pattern

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ------------------ CONFIGURATION ------------------

DATA_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "No_DR" in dirs:
        DATA_DIR = root
        break

if DATA_DIR is None:
    raise FileNotFoundError("Dataset not attached! Click '+ Add Input' in the right sidebar and add the dataset.")

print("Using DATA_DIR:", DATA_DIR)

IMG_SIZE = 128
RANDOM_STATE = 42

# ------------------ FEATURE EXTRACTION ------------------

X, y = [], []

class_names = sorted([
    folder for folder in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, folder))
])

for class_name in class_names:
    folder_path = os.path.join(DATA_DIR, class_name)

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # HOG
        hog_feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        # Color Histogram
        color_feat = []
        for c in range(3):
            hist = cv2.calcHist([rgb], [c], None, [32], [0, 256])
            color_feat.extend(cv2.normalize(hist, hist).flatten())

        # LBP
        lbp = local_binary_pattern(gray, 8, 1, method="uniform")
        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=10, range=(0, 10), density=True
        )

        features = np.concatenate([hog_feat, color_feat, lbp_hist])

        X.append(features)
        y.append(class_name)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Features:", X.shape)
print("Classes:", class_names)
print(pd.Series(y).value_counts())

# ------------------ DATA SPLITTING ------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30,
    stratify=y, random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    stratify=y_temp, random_state=RANDOM_STATE
)

# ------------------ SCALING ------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ------------------ MODEL TUNING ------------------

model = KNeighborsClassifier()

parameters = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(
    model,
    parameters,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)
print("CV Accuracy:", grid.best_score_)

# ------------------ EVALUATION ------------------

datasets = {
    "Training": (X_train, y_train),
    "Validation": (X_val, y_val),
    "Testing": (X_test, y_test)
}

results = []

for name, (data, actual) in datasets.items():

    predicted = best_model.predict(data)

    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, average="weighted", zero_division=0)
    recall = recall_score(actual, predicted, average="weighted", zero_division=0)
    f1 = f1_score(actual, predicted, average="weighted", zero_division=0)

    results.append([name, accuracy, precision, recall, f1])

    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(classification_report(actual, predicted, zero_division=0))

    cm = confusion_matrix(actual, predicted, labels=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"KNN - {name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# ------------------ RESULTS ------------------

results_df = pd.DataFrame(
    results,
    columns=["Dataset", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nFINAL RESULTS")
print(results_df.round(4))

results_df.set_index("Dataset").plot(kind="bar", figsize=(10, 5))
plt.title("KNN Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# ------------------ SAVE MODEL ------------------

os.makedirs("KNN_Output", exist_ok=True)

joblib.dump(best_model, "KNN_Output/knn_model.pkl")
joblib.dump(scaler, "KNN_Output/scaler.pkl")
joblib.dump(class_names, "KNN_Output/classes.pkl")

results_df.to_csv("KNN_Output/results.csv", index=False)

print("KNN experiment completed.")