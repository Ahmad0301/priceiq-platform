"""
PriceIQ FastAPI Application
Exposes pricing analytics and agentic AI endpoints for the commercial team.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

from src.agents.pricing_agent import get_pricing_recommendation
from src.models.pricing import (
    calculate_price_waterfall,
    calculate_price_power_index,
    estimate_price_elasticity,
    estimate_willingness_to_pay,
)
from src.models.anomaly import detect_price_anomalies, compute_price_volatility

app = FastAPI(
    title="PriceIQ API",
    description="Agentic pricing intelligence platform — analytics, LLM recommendations, and anomaly detection.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────

class WaterfallRequest(BaseModel):
    list_price: float = Field(..., gt=0, example=100.0)
    discounts: float = Field(default=0.0, ge=0, example=10.0)
    rebates: float = Field(default=0.0, ge=0, example=5.0)
    freight: float = Field(default=0.0, ge=0, example=2.0)
    payment_terms_discount: float = Field(default=0.0, ge=0, example=1.5)


class PPIRequest(BaseModel):
    your_price: float = Field(..., gt=0, example=99.99)
    competitor_prices: List[float] = Field(..., min_items=1, example=[89.99, 109.99, 95.0])
    market_share: Optional[float] = Field(default=None, ge=0, le=1, example=0.35)


class ElasticityRequest(BaseModel):
    prices: List[float] = Field(..., min_items=3, example=[80.0, 90.0, 100.0, 110.0, 120.0])
    quantities: List[float] = Field(..., min_items=3, example=[500, 450, 400, 340, 280])


class WTPRequest(BaseModel):
    survey_prices: List[float] = Field(..., example=[50.0, 75.0, 100.0, 125.0, 150.0])
    accept_rates: List[float] = Field(..., example=[0.95, 0.82, 0.55, 0.21, 0.08])


class AnomalyRequest(BaseModel):
    product_id: str = Field(..., example="SKU-001")
    prices: List[float] = Field(..., min_items=5, example=[99.9, 100.1, 99.8, 145.0, 100.2, 99.5, 98.9])
    timestamps: List[str] = Field(..., example=["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07"])


class AgentRequest(BaseModel):
    product_id: str
    list_price: float
    discounts: float = 0.0
    rebates: float = 0.0
    competitor_prices: List[float] = []
    historical_prices: List[float] = []
    historical_quantities: List[float] = []
    timestamps: List[str] = []


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "PriceIQ API"}


@app.post("/analytics/waterfall")
def price_waterfall(req: WaterfallRequest):
    """Calculate pocket price using Price Waterfall analysis."""
    try:
        result = calculate_price_waterfall(
            req.list_price, req.discounts, req.rebates,
            req.freight, req.payment_terms_discount,
        )
        return result.__dict__
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analytics/ppi")
def price_power_index(req: PPIRequest):
    """Compute Price Power Index vs. competitors."""
    try:
        ppi = calculate_price_power_index(req.your_price, req.competitor_prices, req.market_share)
        position = "premium" if ppi > 1.05 else ("discount" if ppi < 0.95 else "parity")
        return {"ppi": ppi, "market_position": position, "avg_competitor_price": round(sum(req.competitor_prices) / len(req.competitor_prices), 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analytics/elasticity")
def price_elasticity(req: ElasticityRequest):
    """Estimate price elasticity of demand and optimal price point."""
    try:
        result = estimate_price_elasticity(req.prices, req.quantities)
        return result.__dict__
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analytics/wtp")
def willingness_to_pay(req: WTPRequest):
    """Estimate willingness-to-pay using survey acceptance rate data."""
    try:
        return estimate_willingness_to_pay(req.survey_prices, req.accept_rates)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analytics/anomalies")
def price_anomalies(req: AnomalyRequest):
    """Detect pricing anomalies using Isolation Forest ML model."""
    try:
        anomalies = detect_price_anomalies(req.prices, req.timestamps, req.product_id)
        volatility = compute_price_volatility(req.prices)
        return {"product_id": req.product_id, "anomalies": anomalies, "volatility": volatility}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/agent/recommend")
async def agent_recommend(req: AgentRequest):
    """
    Full agentic AI pricing recommendation.
    Runs all analytics tools and synthesizes an LLM-generated executive recommendation.
    """
    try:
        recommendation = await get_pricing_recommendation(req.dict())
        return {"product_id": req.product_id, "recommendation": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
