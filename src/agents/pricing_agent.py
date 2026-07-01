"""
Agentic pricing intelligence workflow using LangChain.
Orchestrates LLM reasoning over pricing data to generate
contextual, decision-ready recommendations.
"""

import os
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

from src.models.pricing import (
    calculate_price_waterfall,
    calculate_price_power_index,
    estimate_price_elasticity,
)
from src.models.anomaly import detect_price_anomalies

load_dotenv()


# ─────────────────────────────────────────────
# Tool definitions (LangChain tool-use format)
# ─────────────────────────────────────────────

@tool
def run_price_waterfall(
    list_price: float,
    discounts: float = 0.0,
    rebates: float = 0.0,
    freight: float = 0.0,
) -> str:
    """
    Calculates the pocket price using Price Waterfall analysis.
    Use when you need to identify margin leakage across discount/rebate layers.
    """
    result = calculate_price_waterfall(list_price, discounts, rebates, freight)
    return (
        f"Pocket Price: ${result.pocket_price} | "
        f"Margin Leakage: {result.margin_leakage_pct}% | "
        f"Total Deductions: ${result.discounts + result.rebates:.2f}"
    )


@tool
def run_price_power_index(
    your_price: float,
    competitor_prices: List[float],
) -> str:
    """
    Computes Price Power Index (PPI) vs. competitors.
    PPI > 1 means premium position; PPI < 1 means discount position.
    """
    ppi = calculate_price_power_index(your_price, competitor_prices)
    position = "PREMIUM" if ppi > 1.05 else ("DISCOUNT" if ppi < 0.95 else "PARITY")
    return f"PPI: {ppi} | Market Position: {position} vs avg competitor ${sum(competitor_prices)/len(competitor_prices):.2f}"


@tool
def run_elasticity_analysis(
    prices: List[float],
    quantities: List[float],
) -> str:
    """
    Estimates price elasticity of demand and optimal price point.
    Use this when evaluating if a price increase will grow or shrink revenue.
    """
    result = estimate_price_elasticity(prices, quantities)
    sensitivity = "ELASTIC (demand sensitive to price)" if result.is_elastic else "INELASTIC (demand stable at price changes)"
    return (
        f"Elasticity: {result.elasticity} | {sensitivity} | "
        f"Optimal Price: ${result.optimal_price} | "
        f"Expected Demand at Optimal: {result.demand_at_optimal} units"
    )


@tool
def run_anomaly_detection(
    prices: List[float],
    timestamps: List[str],
    product_id: str,
) -> str:
    """
    Detects anomalous price movements for a product using Isolation Forest.
    Use this to surface unusual market events that need immediate attention.
    """
    anomalies = detect_price_anomalies(prices, timestamps, product_id)
    if not anomalies:
        return "No anomalies detected. Price movement is within normal range."
    summary = "; ".join(
        [f"{a['timestamp']}: ${a['price']} (severity: {a['severity']})" for a in anomalies[:3]]
    )
    return f"Detected {len(anomalies)} anomalies. Top anomalies: {summary}"


# ─────────────────────────────────────────────
# Agent setup
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are PriceIQ, an expert pricing intelligence agent for a B2B enterprise.
Your job is to analyze pricing data, detect market anomalies, and generate clear, 
actionable pricing recommendations that commercial teams can act on immediately.

When given pricing data, always:
1. Run Price Waterfall analysis to identify margin leakage
2. Check Price Power Index vs competitors
3. Estimate elasticity to validate price change impact
4. Scan for anomalies that could indicate market disruption
5. Synthesize findings into a concise executive recommendation (3-5 sentences max)

Be specific. Use numbers. Focus on commercial decisions, not methodology.
"""

tools = [run_price_waterfall, run_price_power_index, run_elasticity_analysis, run_anomaly_detection]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def build_pricing_agent() -> AgentExecutor:
    """
    Builds and returns the PriceIQ agentic executor.
    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.1,  # Low temp for deterministic commercial decisions
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=8)


async def get_pricing_recommendation(
    product_context: Dict[str, Any],
) -> str:
    """
    Entry point: accepts product pricing context dict and returns
    LLM-generated recommendation.
    """
    executor = build_pricing_agent()

    user_message = (
        f"Analyze pricing for product '{product_context.get('product_id', 'Unknown')}'.\n"
        f"List price: ${product_context.get('list_price')}\n"
        f"Discounts: ${product_context.get('discounts', 0)}, Rebates: ${product_context.get('rebates', 0)}\n"
        f"Competitor prices: {product_context.get('competitor_prices', [])}\n"
        f"Historical prices (last 30d): {product_context.get('historical_prices', [])}\n"
        f"Historical quantities: {product_context.get('historical_quantities', [])}\n"
        f"Price timestamps: {product_context.get('timestamps', [])}\n\n"
        "Run the appropriate analyses and give me a clear pricing recommendation."
    )

    result = await executor.ainvoke({"input": user_message})
    return result["output"]
