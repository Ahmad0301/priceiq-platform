"""
Pricing analytics models: Price Waterfall, Price Power Index (PPI),
elasticity, and willingness-to-pay (WTP) frameworks.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PriceWaterfallResult:
    list_price: float
    discounts: float
    rebates: float
    pocket_price: float
    margin_leakage_pct: float


@dataclass
class ElasticityResult:
    elasticity: float
    is_elastic: bool
    optimal_price: float
    demand_at_optimal: float


def calculate_price_waterfall(
    list_price: float,
    discounts: float = 0.0,
    rebates: float = 0.0,
    freight: float = 0.0,
    payment_terms_discount: float = 0.0,
) -> PriceWaterfallResult:
    """
    Computes the pocket price by subtracting all off-invoice and on-invoice
    deductions from the list price.
    """
    pocket_price = list_price - discounts - rebates - freight - payment_terms_discount
    margin_leakage_pct = ((list_price - pocket_price) / list_price) * 100 if list_price else 0.0

    return PriceWaterfallResult(
        list_price=list_price,
        discounts=discounts + payment_terms_discount,
        rebates=rebates + freight,
        pocket_price=round(pocket_price, 2),
        margin_leakage_pct=round(margin_leakage_pct, 2),
    )


def calculate_price_power_index(
    your_price: float,
    competitor_prices: List[float],
    market_share: Optional[float] = None,
) -> float:
    """
    Price Power Index (PPI): measures pricing strength relative to competition.
    PPI > 1.0 = premium position; PPI < 1.0 = discount position.
    """
    if not competitor_prices:
        raise ValueError("At least one competitor price is required.")

    avg_competitor = np.mean(competitor_prices)
    ppi = your_price / avg_competitor

    # Adjust for market share if provided (higher share = more pricing power)
    if market_share is not None:
        ppi = ppi * (1 + (market_share - 0.5) * 0.1)

    return round(ppi, 4)


def estimate_price_elasticity(
    prices: List[float],
    quantities: List[float],
) -> ElasticityResult:
    """
    Estimates price elasticity of demand using log-log linear regression.
    Returns the elasticity coefficient and optimal price point.
    """
    if len(prices) != len(quantities) or len(prices) < 3:
        raise ValueError("Need at least 3 matching price-quantity data points.")

    log_prices = np.log(prices).reshape(-1, 1)
    log_quantities = np.log(quantities)

    model = LinearRegression()
    model.fit(log_prices, log_quantities)
    elasticity = model.coef_[0]

    # Find optimal price by maximizing revenue P*Q
    price_range = np.linspace(min(prices), max(prices) * 1.5, 500)
    predicted_log_q = model.predict(np.log(price_range).reshape(-1, 1))
    predicted_q = np.exp(predicted_log_q)
    revenues = price_range * predicted_q

    optimal_idx = np.argmax(revenues)
    optimal_price = round(float(price_range[optimal_idx]), 2)
    demand_at_optimal = round(float(predicted_q[optimal_idx]), 2)

    return ElasticityResult(
        elasticity=round(elasticity, 4),
        is_elastic=abs(elasticity) > 1,
        optimal_price=optimal_price,
        demand_at_optimal=demand_at_optimal,
    )


def estimate_willingness_to_pay(
    survey_prices: List[float],
    accept_rates: List[float],
) -> dict:
    """
    Estimates WTP from Van Westendorp-style survey data.
    accept_rates: proportion of respondents who accept each price point (0-1).
    """
    if len(survey_prices) != len(accept_rates):
        raise ValueError("survey_prices and accept_rates must be same length.")

    df = pd.DataFrame({"price": survey_prices, "accept_rate": accept_rates}).sort_values("price")

    threshold_high = df[df["accept_rate"] >= 0.80]["price"].min() if not df[df["accept_rate"] >= 0.80].empty else df["price"].min()
    threshold_low = df[df["accept_rate"] <= 0.20]["price"].max() if not df[df["accept_rate"] <= 0.20].empty else df["price"].max()
    optimal_wtp = df.iloc[(df["accept_rate"] - 0.50).abs().argsort().iloc[0]]["price"]

    return {
        "wtp_lower_bound": float(threshold_high),
        "wtp_upper_bound": float(threshold_low),
        "optimal_price_point": float(optimal_wtp),
        "recommendation": f"Target price band: ${threshold_high:.2f} – ${threshold_low:.2f}",
    }
