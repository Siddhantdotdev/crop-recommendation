"""
AI-Based Crop Recommendation System Using Soil Analysis Parameters
-------------------------------------------------------------------
File: predict.py
Description: Inference module and command-line prediction interface.
             Loads 'crop_model.pkl' and recommends the optimal crop
             along with the top 3 candidate crop confidence scores.

Author: Senior Machine Learning Engineer (Antigravity)
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "crop_model.pkl"
FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_model(model_path=MODEL_PATH):
    """
    Loads the serialized model from disk.
    Provides clear error guidance if the model has not yet been trained.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found! "
            "Please run 'python train.py' first to train and save the model."
        )
    return joblib.load(model_path)


def predict_crop(N, P, K, temperature, humidity, ph, rainfall, model_path=MODEL_PATH):
    """
    Task 7: Crop Prediction Function

    Accepts:
      - N: Nitrogen ratio in soil (kg/ha)
      - P: Phosphorous ratio in soil (kg/ha)
      - K: Potassium ratio in soil (kg/ha)
      - temperature: Temperature in degree Celsius
      - humidity: Relative humidity in %
      - ph: Soil pH level (0-14)
      - rainfall: Rainfall in mm

    Returns:
      A dictionary containing:
      - 'predicted_crop': The top recommended crop (str)
      - 'confidence': Highest confidence score (float)
      - 'top_3': List of tuples (crop_name, probability_score)
    """
    # 1. Load trained model
    model = load_model(model_path)

    # 2. Format input parameters into a pandas DataFrame matching feature column names
    input_data = pd.DataFrame([[
        float(N),
        float(P),
        float(K),
        float(temperature),
        float(humidity),
        float(ph),
        float(rainfall)
    ]], columns=FEATURE_NAMES)

    # 3. Model prediction and class probabilities
    predicted_crop = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    # 4. Extract top 3 candidates sorted by probability
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top_3 = []
    for idx in top3_indices:
        crop_name = classes[idx]
        score = float(probabilities[idx])
        top_3.append({
            "crop": crop_name,
            "confidence": score,
            "percentage": f"{score * 100:.2f}%"
        })

    return {
        "predicted_crop": predicted_crop,
        "confidence": top_3[0]["confidence"],
        "top_3": top_3
    }


def demo_predictions():
    """
    Runs demonstrative sample predictions for viva or demonstration.
    """
    samples = [
        {
            "name": "Sample 1: High Nitrogen, High Rainfall (Rice Profile)",
            "params": {"N": 90, "P": 42, "K": 43, "temperature": 20.87, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
        },
        {
            "name": "Sample 2: High Phosphorus & Potassium (Apple/Grapes Profile)",
            "params": {"N": 20, "P": 130, "K": 200, "temperature": 22.0, "humidity": 92.0, "ph": 5.9, "rainfall": 110.0}
        },
        {
            "name": "Sample 3: Moderate NPK, Sub-tropical Climate (Coffee Profile)",
            "params": {"N": 100, "P": 30, "K": 30, "temperature": 25.5, "humidity": 60.0, "ph": 6.8, "rainfall": 160.0}
        }
    ]

    print("=" * 70)
    print("DEMONSTRATION: CROP RECOMMENDATION INFERENCE")
    print("=" * 70)

    for item in samples:
        print(f"\n>>> {item['name']}")
        p = item["params"]
        print(f"    Inputs: N={p['N']}, P={p['P']}, K={p['K']}, Temp={p['temperature']}°C, "
              f"Humidity={p['humidity']}%, pH={p['ph']}, Rain={p['rainfall']}mm")

        result = predict_crop(**p)
        print(f"    ==> Recommended Crop : {result['predicted_crop'].upper()}")
        print(f"    ==> Top 3 Predictions:")
        for rank, cand in enumerate(result["top_3"], start=1):
            print(f"        {rank}. {cand['crop']:<12} -> {cand['percentage']}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict the best crop based on soil and environmental parameters."
    )
    parser.add_argument("--N", type=float, help="Nitrogen content in soil (kg/ha)")
    parser.add_argument("--P", type=float, help="Phosphorus content in soil (kg/ha)")
    parser.add_argument("--K", type=float, help="Potassium content in soil (kg/ha)")
    parser.add_argument("--temperature", type=float, help="Temperature in °C")
    parser.add_argument("--humidity", type=float, help="Relative humidity in percentage (%%)")
    parser.add_argument("--ph", type=float, help="Soil pH (0-14)")
    parser.add_argument("--rainfall", type=float, help="Rainfall in mm")

    args = parser.parse_args()

    # If any argument is supplied via CLI, run individual prediction
    if any(v is not None for v in [args.N, args.P, args.K, args.temperature, args.humidity, args.ph, args.rainfall]):
        # Check all are provided
        missing = [arg for arg in FEATURE_NAMES if getattr(args, arg) is None]
        if missing:
            print(f"Error: Missing required argument(s): {', '.join(missing)}")
            print("Provide all 7 parameters or run without arguments to see demo predictions.")
            exit(1)

        res = predict_crop(args.N, args.P, args.K, args.temperature, args.humidity, args.ph, args.rainfall)
        print("\n" + "=" * 60)
        print(f"Recommended Crop : {res['predicted_crop'].upper()}")
        print("Top 3 Recommendations:")
        for r, item in enumerate(res["top_3"], 1):
            print(f"  {r}. {item['crop']:<12} : {item['percentage']}")
        print("=" * 60 + "\n")
    else:
        # Otherwise run demonstration test cases
        demo_predictions()
