from __future__ import annotations

from services.tavily import search_tavily


async def tavily_role_research(query: str) -> dict:
    """Search for career skills, requirements, and qualifications for a target role."""
    search_query = f"{query} career skills requirements qualifications"
    return await search_tavily(search_query, "role_research")
