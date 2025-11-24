from __future__ import annotations

from services.tavily import search_tavily


async def tavily_learning_resources(query: str) -> dict:
    """Search for courses, tutorials, certifications, and learning paths."""
    search_query = f"{query} courses tutorials certifications learning path"
    return await search_tavily(search_query, "learning_resources")
