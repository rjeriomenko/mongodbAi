from __future__ import annotations

import asyncio
import os

import aiohttp
from urllib.parse import quote
from google import genai

from models.job import Job


def get_env(key: str) -> str:
    """Get required environment variable."""
    value = os.environ.get(key, "")
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


async def generate_search_terms(query: str) -> list[str]:
    """Use LLM to extract good search terms from a natural language query."""
    client = genai.Client(api_key=get_env("GOOGLE_API_KEY"))

    prompt = f"""Convert this job search query into tags for RemoteOK's API. Return only lowercase single-word tags separated by commas, no explanations.
The first tag should be the most important/specific one (e.g., a programming language, job title, or skill).
Focus on: programming languages, frameworks, job titles, skills.

Query: "{query}"

Tags:"""

    result = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.0-flash",
        contents=prompt
    )

    terms_text = result.text.strip()
    terms = [t.strip().lower() for t in terms_text.split(",") if t.strip()]
    return terms


async def search_remoteok_jobs(query: str, max_results: int = 20) -> dict:
    """Fetch job listings from RemoteOK API.

    Returns:
        dict with keys: source, results (list of Job), error
    """
    try:
        search_terms = await generate_search_terms(query)
        search_query = search_terms[0] if search_terms else query

        async with aiohttp.ClientSession() as session:
            encoded_query = quote(search_query)
            url = f"https://remoteok.com/api?tag={encoded_query}&limit={max_results}"
            headers = {"User-Agent": "CareerAgent/1.0"}

            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return {
                        "source": "search_remoteok_jobs",
                        "results": [],
                        "error": f"RemoteOK API returned status {response.status}"
                    }

                all_jobs = await response.json()
                raw_jobs = all_jobs[1:max_results + 1] if len(all_jobs) > 1 else []

                jobs: list[Job] = []
                for job in raw_jobs:
                    jobs.append(Job(
                        url=job.get("url", ""),
                        title=job.get("position", ""),
                        company=job.get("company", ""),
                        source="remoteok",
                        salary_min=job.get("salary_min"),
                        salary_max=job.get("salary_max"),
                        date_posted=job.get("date"),
                        location=job.get("location", "Worldwide"),
                        tags=job.get("tags", []),
                    ))

                return {
                    "source": "search_remoteok_jobs",
                    "results": jobs,
                    "error": None
                }

    except Exception as e:
        return {
            "source": "search_remoteok_jobs",
            "results": [],
            "error": str(e)
        }
