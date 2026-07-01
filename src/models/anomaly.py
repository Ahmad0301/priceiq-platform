"""
Anomaly detection for pricing data using Isolation Forest.
Flags unusual price movements that warrant human review or LLM analysis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any


def detect_price_anomalies(
    prices: List[float],
    timestamps: List[str],
    product_id: str,
    contamination: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Detects anomalous price points using Isolation Forest.

    Args:
        prices: Historical price values for a product
        timestamps: Corresponding timestamps (ISO format strings)
        product_id: Identifier for the product being analyzed
        contamination: Expected proportion of anomalies (0–0.5)

    Returns:
        List of detected anomalies with timestamp and deviation score
    """
    if len(prices) < 5:
        raise ValueError("Need at least 5 data points for anomaly detection.")

    price_array = np.array(prices).reshape(-1, 1)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    labels = model.fit_predict(price_array)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(price_array)  # More negative = more anomalous

    anomalies = []
    for i, (label, score) in enumerate(zip(labels, scores)):
        if label == -1:
            anomalies.append({
                "product_id": product_id,
                "timestamp": timestamps[i],
                "price": prices[i],
                "anomaly_score": round(float(score), 4),
                "severity": "high" if score < -0.15 else "medium",
            })

    return sorted(anomalies, key=lambda x: x["anomaly_score"])


def compute_price_volatility(prices: List[float], window: int = 7) -> Dict[str, float]:
    """
    Computes rolling price volatility metrics for a product.
    """
    series = pd.Series(prices)
    rolling_std = series.rolling(window=window).std()

    return {
        "mean_price": round(float(series.mean()), 2),
        "std_dev": round(float(series.std()), 2),
        "rolling_volatility_avg": round(float(rolling_std.mean()), 4),
        "max_single_day_change_pct": round(
            float(series.pct_change().abs().max() * 100), 2
        ),
    }
