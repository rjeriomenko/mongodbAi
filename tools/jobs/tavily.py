from __future__ import annotations

import asyncio
import os
from typing import Literal

from tavily import TavilyClient

from models.job import Job


def get_env(key: str) -> str:
    """Get required environment variable."""
    value = os.environ.get(key, "")
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def get_tavily_client() -> TavilyClient:
    """Get Tavily client instance."""
    return TavilyClient(get_env("TAVILY_API_KEY"))


async def search_jobs_tavily(query: str) -> dict:
    """Search for job listings using Tavily web search.

    Note: Tavily returns web search results, not structured job data.
    Many Job fields will be None.

    Returns:
        dict with keys: source, results (list of Job), error
    """
    try:
        client = get_tavily_client()
        search_query = f"{query} job listings careers hiring"
        depth: Literal["basic", "advanced"] = "basic"

        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.search,
                query=search_query,
                search_depth=depth,
                max_results=20,
                include_raw_content=False,
                include_images=False,
            ),
            timeout=10.0
        )

        raw_results = response.get("results", [])

        jobs: list[Job] = []
        for result in raw_results:
            jobs.append(Job(
                url=result.get("url", ""),
                title=result.get("title", ""),
                company="",  # Not available from Tavily
                source="tavily",
                # Tavily doesn't provide structured job data
                salary_min=None,
                salary_max=None,
                date_posted=None,
                location=None,
                tags=[],
            ))

        return {
            "source": "search_jobs_tavily",
            "results": jobs,
            "error": None
        }

    except Exception as e:
        return {
            "source": "search_jobs_tavily",
            "results": [],
            "error": str(e)
        }
