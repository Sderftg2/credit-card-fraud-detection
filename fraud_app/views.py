from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd
import os
from django.conf import settings

# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------

MODEL_PATH = os.path.join(settings.BASE_DIR, "fraud_app", "fraud_detection_model.pkl")
SCALER_PATH = os.path.join(settings.BASE_DIR, "fraud_app", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# --------------------------------------------------
# Home Page
# --------------------------------------------------

def home(request):
    """
    Home page handler.
    Single transaction prediction is disabled because
    the ML model requires 30 PCA features.
    """
    if request.method == "POST" and "csv_file" not in request.FILES:
        return render(
            request,
            "home.html",
            {
                "error": (
                    "Single transaction prediction requires all 30 features "
                    "(Time, V1–V28, Amount). Please use Bulk CSV Upload."
                )
            },
        )

    return render(request, "home.html")


# --------------------------------------------------
# Bulk CSV Upload & Prediction
# --------------------------------------------------

def upload_csv(request):
    if request.method == "POST":
        try:
            csv_file = request.FILES["csv_file"]
            df = pd.read_csv(csv_file)

            # Required features (same as training)
            required_features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

            # Check for missing columns
            missing_cols = [col for col in required_features if col not in df.columns]
            if missing_cols:
                return render(
                    request,
                    "home.html",
                    {
                        "error": f"Missing required columns: {missing_cols}"
                    },
                )

            # Prepare input
            X = df[required_features]
            X_scaled = scaler.transform(X)

            # Predictions
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]

            # Add results to dataframe
            df["Prediction"] = predictions
            df["Fraud_Probability"] = probabilities

            # Summary
            fraud_count = int((predictions == 1).sum())
            total_count = len(predictions)

            summary = (
                f"✅ File processed successfully. "
                f"Detected {fraud_count} fraudulent transactions "
                f"out of {total_count} total transactions."
            )

            # Preview first 10 rows
            preview_table = df.head(10).to_html(
                classes="table table-bordered table-striped",
                index=False
            )

            # Save output CSV (optional)
            output_path = os.path.join(settings.BASE_DIR, "fraud_app", "predictions.csv")
            df.to_csv(output_path, index=False)

            return render(
                request,
                "result.html",
                {
                    "result": summary,
                    "table": preview_table,
                },
            )

        except Exception as e:
            return render(
                request,
                "home.html",
                {
                    "error": f"Error processing file: {str(e)}"
                },
            )

    return render(request, "home.html")
