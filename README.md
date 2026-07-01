# PriceIQ — Agentic Pricing Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112-green)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.16-orange)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An enterprise-grade pricing intelligence platform that combines **agentic AI workflows** with advanced **pricing analytics models** to generate contextual, decision-ready pricing recommendations.

---

## Features

- **Agentic AI Workflow** — LangChain-powered agent that autonomously selects and runs pricing tools, then synthesizes LLM-generated executive recommendations
- **Price Waterfall Analysis** — Identifies pocket price and margin leakage across discount/rebate layers
- **Price Power Index (PPI)** — Benchmarks your pricing position vs. competitor landscape
- **Price Elasticity Modeling** — Log-log regression to estimate demand sensitivity and optimal price point
- **Willingness-to-Pay (WTP)** — Van Westendorp-style survey analysis to quantify customer price thresholds
- **Anomaly Detection** — Isolation Forest ML model to flag unusual price movements in real time
- **FastAPI REST layer** — All analytics exposed as clean API endpoints ready for Power BI integration

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Orchestration | LangChain 0.2, OpenAI GPT-4o-mini |
| Analytics Models | Scikit-learn, Pandas, NumPy |
| API Layer | FastAPI, Uvicorn |
| Database | PostgreSQL (SQLAlchemy ORM) |
| Testing | Pytest |

---

## Project Structure

```
priceiq-platform/
├── src/
│   ├── agents/
│   │   └── pricing_agent.py    # LangChain agentic workflow + tool definitions
│   ├── models/
│   │   ├── pricing.py          # Waterfall, PPI, Elasticity, WTP models
│   │   └── anomaly.py          # Isolation Forest anomaly detection
│   ├── api/
│   │   └── main.py             # FastAPI app + all route handlers
│   └── utils/
├── tests/
│   └── test_pricing.py         # Pytest test suite
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Ahmad0301/priceiq-platform.git
cd priceiq-platform
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and DATABASE_URL
```

### 3. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/analytics/waterfall` | Price Waterfall (pocket price + margin leakage) |
| POST | `/analytics/ppi` | Price Power Index vs. competitors |
| POST | `/analytics/elasticity` | Demand elasticity + optimal price |
| POST | `/analytics/wtp` | Willingness-to-pay analysis |
| POST | `/analytics/anomalies` | Price anomaly detection |
| POST | `/agent/recommend` | Full agentic AI pricing recommendation |

### Example: Agentic Recommendation

```bash
curl -X POST http://localhost:8000/agent/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "SKU-001",
    "list_price": 100.0,
    "discounts": 12.0,
    "rebates": 5.0,
    "competitor_prices": [89.99, 105.0, 95.0],
    "historical_prices": [95, 97, 100, 102, 98, 145, 100],
    "historical_quantities": [400, 390, 370, 355, 380, 200, 375],
    "timestamps": ["2025-01-01","2025-01-08","2025-01-15","2025-01-22","2025-01-29","2025-02-05","2025-02-12"]
  }'
```

**Response:**
```json
{
  "product_id": "SKU-001",
  "recommendation": "Pocket price is $83 after 17% margin leakage. Your PPI of 0.98 places you at parity with competitors. Elasticity of -1.4 indicates demand is sensitive — a 5% price increase would reduce volume by ~7%. The anomaly on Feb 5th (price $145) shows external supply disruption risk. Recommendation: hold list price at $100, tighten rebate ceiling to $3 to recover $2/unit margin without volume loss."
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM reasoning | `sk-...` |
| `LLM_MODEL` | Model to use | `gpt-4o-mini` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@localhost:5432/priceiq` |
| `APP_ENV` | Environment (`development`/`production`) | `development` |
| `PORT` | API server port | `8000` |

---

## Power BI Integration

The `/analytics/*` endpoints return JSON that maps directly to Power BI semantic model tables. Connect Power BI via the Web connector pointing to `http://your-host:8000/analytics/waterfall` (POST with JSON body using Power Query's `Web.Contents`).

---

## License

MIT
