"""
Tests for PriceIQ pricing analytics models.
"""

import pytest
from src.models.pricing import (
    calculate_price_waterfall,
    calculate_price_power_index,
    estimate_price_elasticity,
    estimate_willingness_to_pay,
)
from src.models.anomaly import detect_price_anomalies, compute_price_volatility


class TestPriceWaterfall:
    def test_basic_pocket_price(self):
        result = calculate_price_waterfall(list_price=100.0, discounts=10.0, rebates=5.0)
        assert result.pocket_price == 85.0
        assert result.margin_leakage_pct == 15.0

    def test_no_deductions(self):
        result = calculate_price_waterfall(list_price=200.0)
        assert result.pocket_price == 200.0
        assert result.margin_leakage_pct == 0.0

    def test_all_deductions(self):
        result = calculate_price_waterfall(
            list_price=100.0, discounts=20.0, rebates=10.0,
            freight=5.0, payment_terms_discount=2.0
        )
        assert result.pocket_price == 63.0
        assert result.margin_leakage_pct == 37.0


class TestPricePowerIndex:
    def test_premium_position(self):
        ppi = calculate_price_power_index(your_price=120.0, competitor_prices=[100.0, 95.0, 105.0])
        assert ppi > 1.0

    def test_discount_position(self):
        ppi = calculate_price_power_index(your_price=80.0, competitor_prices=[100.0, 95.0, 105.0])
        assert ppi < 1.0

    def test_parity_position(self):
        ppi = calculate_price_power_index(your_price=100.0, competitor_prices=[100.0])
        assert ppi == 1.0

    def test_empty_competitors_raises(self):
        with pytest.raises(ValueError):
            calculate_price_power_index(your_price=100.0, competitor_prices=[])


class TestPriceElasticity:
    PRICES = [80.0, 90.0, 100.0, 110.0, 120.0]
    QUANTITIES = [500, 450, 400, 340, 280]

    def test_negative_elasticity(self):
        result = estimate_price_elasticity(self.PRICES, self.QUANTITIES)
        assert result.elasticity < 0  # Higher price → lower quantity

    def test_optimal_price_in_range(self):
        result = estimate_price_elasticity(self.PRICES, self.QUANTITIES)
        assert min(self.PRICES) <= result.optimal_price <= max(self.PRICES) * 1.5

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            estimate_price_elasticity([100.0, 110.0], [500, 480])


class TestAnomalyDetection:
    def test_detects_spike(self):
        prices = [100.0] * 10 + [500.0] + [100.0] * 9  # One obvious spike
        timestamps = [f"2025-01-{i+1:02d}" for i in range(20)]
        anomalies = detect_price_anomalies(prices, timestamps, "SKU-TEST")
        assert len(anomalies) > 0
        assert any(a["price"] == 500.0 for a in anomalies)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            detect_price_anomalies([100.0, 200.0], ["2025-01-01", "2025-01-02"], "SKU-001")

    def test_volatility_metrics(self):
        prices = [99.0, 101.0, 98.0, 102.0, 100.0, 99.5]
        metrics = compute_price_volatility(prices)
        assert "mean_price" in metrics
        assert "std_dev" in metrics
        assert metrics["mean_price"] > 0
