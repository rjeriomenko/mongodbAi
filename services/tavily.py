from __future__ import annotations

from tavily import TavilyClient

from config import get_tavily_api_key
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


async def search_tavily(query: str, search_type: str) -> dict:
    """Base Tavily search function.

    Args:
        query: Search query
        search_type: Type of search for query customization
    """
    logger.info(f"search_tavily: {search_type} for query: {query}")

    try:
        client = TavilyClient(get_tavily_api_key())

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_raw_content=False,
            include_images=False,
        )

        results = response.get("results", [])
        logger.info(f"search_tavily: found {len(results)} results for {search_type}")

        return {
            "search_type": search_type,
            "results": results,
            "error": None
        }

    except Exception as e:
        logger.error(f"search_tavily: error: {e}")
        return {
            "search_type": search_type,
            "results": [],
            "error": str(e)
        }
