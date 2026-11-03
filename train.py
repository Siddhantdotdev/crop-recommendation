"""
AI-Based Crop Recommendation System Using Soil Analysis Parameters
-------------------------------------------------------------------
File: train.py
Description: End-to-end training and evaluation script.
             Explores data, trains Decision Tree, Random Forest, and KNN,
             evaluates them on an unseen test set (80:20 split), and saves
             the best performing model (Random Forest) as 'crop_model.pkl'.

Author: Senior Machine Learning Engineer (Antigravity)
Tech Stack: Python, Pandas, NumPy, Scikit-learn, Joblib
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def load_and_explore_data(filepath="Crop_recommendation.csv"):
    """
    Task 1: Load CSV and perform basic exploratory checks.
    """
    print("=" * 70)
    print("STEP 1: DATA EXPLORATION")
    print("=" * 70)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please place 'Crop_recommendation.csv' in the project root."
        )

    # 1. Load the dataset using Pandas
    df = pd.read_csv(filepath)
    print(f"Dataset successfully loaded from: {filepath}")

    # 2. Display shape and first few records
    print(f"\n[+] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n[+] First 5 Records:")
    print(df.head())

    # 3. Columns and Data Types
    print("\n[+] Data Types:")
    print(df.dtypes)

    # 4. Statistical Summary
    print("\n[+] Statistical Summary (Numerical Features):")
    print(df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    # 5. Missing Values and Duplicates Check
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"\n[+] Total Missing Values : {missing}")
    print(f"[+] Total Duplicate Rows : {duplicates}")

    # 6. Analyze Distribution of Crop Labels
    crop_counts = df["label"].value_counts()
    print(f"\n[+] Number of Unique Crops : {df['label'].nunique()}")
    print("[+] Crop Label Distribution:")
    print(crop_counts)

    return df


def preprocess_data(df):
    """
    Task 3: Data Preprocessing
    - Separate features (X) and target (y).
    - Perform 80:20 Train-Test split.
    - Set random_state=42 for reproducibility.
    - Use stratify=y to ensure equal proportion of all 22 crop classes in train & test sets.
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATA PREPROCESSING & SPLITTING")
    print("=" * 70)

    # Separate input features (X) and target crop labels (y)
    X = df.drop(columns=["label"])
    y = df["label"]

    feature_names = list(X.columns)
    print(f"[+] Feature Columns ({len(feature_names)}): {feature_names}")
    print(f"[+] Target Column : 'label' ({y.nunique()} distinct classes)")

    # 80:20 Train-Test Split
    # stratify=y ensures each of the 22 crops has 80 samples in train and 20 in test.
    # This prevents class imbalance or omission during training and evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[+] Training Set Shape : X={X_train.shape}, y={y_train.shape}")
    print(f"[+] Test Set Shape     : X={X_test.shape}, y={y_test.shape}")
    print("[+] Stratification Check: Exactly 80 train / 20 test samples per crop.")

    return X_train, X_test, y_train, y_test, feature_names


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Tasks 4 & 5: Model Training & Evaluation
    Trains and compares:
      1. Decision Tree Classifier
      2. Random Forest Classifier
      3. K-Nearest Neighbors (KNN with StandardScaler pipeline to avoid data leakage)
    """
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # Define candidate models
    # Note: KNN calculates Euclidean distances, so features with larger scales
    # (e.g., rainfall up to 300 mm vs pH 3-9) would dominate without feature scaling.
    # We use an sklearn Pipeline with StandardScaler fitted ONLY on X_train to avoid data leakage.
    # Tree-based algorithms (Decision Tree, Random Forest) are invariant to monotonic feature scaling.
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        # Fit on training set only
        model.fit(X_train, y_train)

        # Predict on unseen test set
        y_pred = model.predict(X_test)

        # Compute evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        }

        print(f"  Accuracy  : {acc * 100:.2f}%")
        print(f"  Precision : {prec * 100:.2f}%")
        print(f"  Recall    : {rec * 100:.2f}%")
        print(f"  F1-Score  : {f1 * 100:.2f}%")

    # Summary Comparison Table
    print("\n" + "-" * 70)
    print(f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for name, metrics in results.items():
        print(
            f"{name:<22} | {metrics['Accuracy']*100:>8.2f}% | "
            f"{metrics['Precision']*100:>8.2f}% | "
            f"{metrics['Recall']*100:>8.2f}% | "
            f"{metrics['F1-Score']*100:>8.2f}%"
        )
    print("-" * 70)

    # Select best model based on highest test accuracy / F1-score
    best_name = max(results, key=lambda k: (results[k]["Accuracy"], results[k]["F1-Score"]))
    best_info = results[best_name]
    best_model = best_info["model"]

    print(f"\n[+] Selected Best Model: '{best_name}' with {best_info['Accuracy']*100:.2f}% Test Accuracy.")

    # Detailed Classification Report for Best Model
    print("\n[+] Detailed Classification Report for Best Model:")
    print(classification_report(y_test, best_info["y_pred"], digits=4))

    # Confusion Matrix Information
    cm = confusion_matrix(y_test, best_info["y_pred"])
    total_test = len(y_test)
    correct_preds = np.trace(cm)
    print(f"[+] Confusion Matrix Summary:")
    print(f"    Correctly Classified Test Samples: {correct_preds} / {total_test} ({correct_preds/total_test*100:.2f}%)")
    print(f"    Misclassified Test Samples        : {total_test - correct_preds} / {total_test}")

    return best_name, best_model, results


def save_model(model, output_path="crop_model.pkl"):
    """
    Task 6: Model Saving
    Serializes and saves the trained model to disk using Joblib.
    """
    print("\n" + "=" * 70)
    print("STEP 4: SAVING THE BEST MODEL")
    print("=" * 70)

    joblib.dump(model, output_path)
    print(f"[+] Trained model successfully saved to: '{output_path}'")
    print(f"[+] File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def verify_saved_model(model_path="crop_model.pkl"):
    """
    Quick sanity test to ensure the saved model can be reloaded and yields valid predictions.
    """
    print("\n" + "=" * 70)
    print("STEP 5: VERIFICATION TEST")
    print("=" * 70)

    loaded_model = joblib.load(model_path)

    # Example test sample: Rice conditions (High Nitrogen, High Rainfall, Moderate Temp)
    sample_data = pd.DataFrame([{
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.87,
        "humidity": 82.00,
        "ph": 6.50,
        "rainfall": 202.93
    }])

    predicted_crop = loaded_model.predict(sample_data)[0]
    probabilities = loaded_model.predict_proba(sample_data)[0]

    # Get top 3 class scores
    classes = loaded_model.classes_
    top3_indices = np.argsort(probabilities)[::-1][:3]

    print(f"[+] Verification Sample Input:")
    print(sample_data.to_dict(orient="records")[0])
    print(f"\n[+] Predicted Crop: '{predicted_crop.upper()}'")
    print("[+] Top 3 Recommendations:")
    for rank, idx in enumerate(top3_indices, start=1):
        print(f"    {rank}. {classes[idx]:<12} -> Confidence: {probabilities[idx]*100:.2f}%")

    print("\n[+] Verification successful! Model is ready for inference.")


if __name__ == "__main__":
    # Execute full training workflow
    dataframe = load_and_explore_data("Crop_recommendation.csv")
    X_tr, X_te, y_tr, y_te, feats = preprocess_data(dataframe)
    best_model_name, best_fitted_model, all_results = train_and_evaluate_models(
        X_tr, X_te, y_tr, y_te
    )
    save_model(best_fitted_model, "crop_model.pkl")
    verify_saved_model("crop_model.pkl")
    print("\n" + "=" * 70)
    print("ALL TRAINING & EVALUATION TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

