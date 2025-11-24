from __future__ import annotations

from services.tavily import search_tavily


async def tavily_market_trends(query: str) -> dict:
    """Search for job market trends, salary outlook, and industry forecasts."""
    search_query = f"{query} job market trends salary outlook 2024 2025"
    return await search_tavily(search_query, "market_trends")
