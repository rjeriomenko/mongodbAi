from __future__ import annotations

import json
from urllib.parse import quote
from urllib.request import urlopen, Request

from mcp_agent.logging.logger import get_logger

from models.job import Job

logger = get_logger(__name__)


# TODO: Re-enable when Gemini API timeout issue is resolved
# Requires: import asyncio, import google.generativeai as genai, from config import get_google_api_key
#
# async def generate_search_terms(query: str) -> list[str]:
#     """Use LLM to extract good search terms from a natural language query."""
#     logger.info("Getting Google API key")
#     api_key = get_google_api_key()
#     logger.info("Configuring Gemini")
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel("gemini-2.0-flash")
#
#     prompt = f"""Convert this job search query into tags for RemoteOK's API. Return only lowercase single-word tags separated by commas, no explanations.
# The first tag should be the most important/specific one (e.g., a programming language, job title, or skill).
# Focus on: programming languages, frameworks, job titles, skills.
#
# Query: "{query}"
#
# Tags:"""
#
#     logger.info("Calling Gemini API (async)")
#     try:
#         result = await asyncio.wait_for(
#             model.generate_content_async(prompt),
#             timeout=30.0
#         )
#         logger.info("Gemini API response received")
#     except asyncio.TimeoutError:
#         logger.error("Gemini API call timed out after 30s")
#         raise
#
#     terms_text = result.text.strip()
#     terms = [t.strip().lower() for t in terms_text.split(",") if t.strip()]
#     logger.info(f"Generated search terms: {terms}")
#     return terms


async def search_remoteok_jobs(query: str, max_results: int = 20) -> dict:
    """Fetch job listings from RemoteOK API.

    Returns:
        dict with keys: source, results (list of Job), error
    """
    logger.info(f"search_remoteok_jobs started with query: {query}")
    try:
        # Use query directly - skip LLM preprocessing for now
        search_query = query.split()[0].lower() if query else "developer"
        encoded_query = quote(search_query)
        url = f"https://remoteok.com/api?tag={encoded_query}&limit={max_results}"
        logger.info(f"Fetching RemoteOK URL: {url}")

        # Use urllib (synchronous but works everywhere)
        req = Request(url, headers={"User-Agent": "CareerAgent/1.0"})
        with urlopen(req, timeout=30) as response:
            response_text = response.read().decode('utf-8')
            logger.info(f"Got response, length: {len(response_text)}")

        # Parse JSON
        all_jobs = json.loads(response_text)
        raw_jobs = all_jobs[1:max_results + 1] if len(all_jobs) > 1 else []

        jobs: list[Job] = []
        for job in raw_jobs:
            jobs.append(Job(
                url=job.get("url", ""),
                title=job.get("position", ""),
                company=job.get("company", ""),
                source=search_remoteok_jobs.__name__,
                salary_min=job.get("salary_min"),
                salary_max=job.get("salary_max"),
                date_posted=job.get("date"),
                location=job.get("location", "Worldwide"),
                tags=job.get("tags", []),
            ))

        logger.info(f"Parsed {len(jobs)} jobs from RemoteOK")
        return {
            "source": "search_remoteok_jobs",
            "results": jobs,
            "error": None
        }

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"search_remoteok_jobs error: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "source": "search_remoteok_jobs",
            "results": [],
            "error": error_msg
        }
