from __future__ import annotations

import asyncio
import time
from typing import Optional, Literal, TypedDict
from datetime import datetime

import os
import aiohttp
from urllib.parse import quote
from dotenv import load_dotenv
from tavily import TavilyClient
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
from google import genai


# Job schema for consistent structure across all job sources
class Job(TypedDict, total=False):
    title: str
    company: str
    url: str
    salary: str | None
    date_posted: str
    location: str | None
    tags: list[str]
    source: str
    score: float
    is_fresh: bool

# Load .env file
load_dotenv()

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.core.context import Context
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM


# Helper to get required env var
def get_env(key: str) -> str:
    value = os.environ.get(key, "")
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value

# Get embedding from Google
async def get_embedding(text: str) -> list[float]:
    client = genai.Client(api_key=get_env("GOOGLE_API_KEY"))
    result = await asyncio.to_thread(
        client.models.embed_content,
        model="text-embedding-004",
        contents=text
    )
    return result.embeddings[0].values

def get_mongo_client():
    return AsyncIOMotorClient(get_env("MDB_MCP_CONNECTION_STRING"))

def get_tavily_client():
    return TavilyClient(get_env("TAVILY_API_KEY"))


# Generate search terms from query using LLM
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

    # Parse comma-separated terms
    terms_text = result.text.strip()
    terms = [t.strip().lower() for t in terms_text.split(",") if t.strip()]
    return terms


# Internal tool: Fetch jobs from RemoteOK API
async def search_remoteok_jobs(query: str, max_results: int = 20) -> dict:
    """Fetch job listings from RemoteOK API."""
    try:
        # Use LLM to generate optimized search term
        search_terms = await generate_search_terms(query)
        search_query = search_terms[0] if search_terms else query

        async with aiohttp.ClientSession() as session:
            encoded_query = quote(search_query)
            url = f"https://remoteok.com/api?tag={encoded_query}&limit={max_results}"
            headers = {"User-Agent": "CareerAgent/1.0"}

            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return {
                        "source": search_remoteok_jobs.__name__,
                        "results": [],
                        "error": f"RemoteOK API returned status {response.status}"
                    }

                all_jobs = await response.json()

                # Skip first item (metadata)
                raw_jobs = all_jobs[1:max_results + 1] if len(all_jobs) > 1 else []

                # Convert to standard job format
                jobs = []
                for job in raw_jobs:
                    salary = None
                    if job.get('salary_min'):
                        salary = f"${job.get('salary_min', 0):,}-${job.get('salary_max', 0):,}"

                    jobs.append({
                        "title": job.get("position", ""),
                        "company": job.get("company", ""),
                        "url": job.get("url", ""),
                        "salary": salary,
                        "date_posted": job.get("date", ""),
                        "location": job.get("location", "Worldwide"),
                        "tags": job.get("tags", []),
                        "source": "remoteok"
                    })

                return {
                    "source": search_remoteok_jobs.__name__,
                    "results": jobs,
                    "error": None
                }

    except Exception as e:
        return {
            "source": search_remoteok_jobs.__name__,
            "results": [],
            "error": str(e)
        }


# Internal tool: Search jobs via Tavily
async def search_jobs_tavily(query: str) -> dict:
    """Search for job listings using Tavily."""
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

        return {
            "source": search_jobs_tavily.__name__,
            "results": response.get("results", []),
            "error": None
        }
    except Exception as e:
        return {
            "source": search_jobs_tavily.__name__,
            "results": [],
            "error": str(e)
        }

# Create the MCPApp, the root of mcp-agent.
app = MCPApp(
    name="career_agent",
    description="Career search agent with jobs, communities, and personalized recommendations",
)

# Career Agent: Main mega-tool for job search
@app.tool()
async def career_agent(
    query: str,
    run_tavily: bool = False,
    include_jobs: bool = True,
    include_communities: bool = True,
    max_results: int = 20,
    user_id: str = "anonymous",
    app_ctx: Optional[Context] = None
) -> dict:
    """
    Career search agent that finds jobs, communities, and relevant resources.

    Args:
        query: Search query (e.g., "python developer remote")
        include_jobs: Include job listings
        include_communities: Include community/forum results
        max_results: Maximum total results to return
        user_id: User identifier for personalization
    """
    start_time = time.time()
    tools_executed = []
    matched_jobs = []
    docs_created = 0
    historical_matches = 0

    # Build dict of selected tools based on flags (dict enforces uniqueness)
    selected_tools = {}
    if include_jobs:
        selected_tools["search_remoteok_jobs"] = search_remoteok_jobs(query, max_results)
    if include_jobs and run_tavily:
        selected_tools["search_jobs_tavily"] = search_jobs_tavily(query)

    # Execute all selected tools in parallel
    if selected_tools:
        tool_names = list(selected_tools.keys())
        tool_coros = list(selected_tools.values())
        results = await asyncio.gather(*tool_coros, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            tool_name = tool_names[i]
            tools_executed.append(tool_name)

            if isinstance(result, Exception):
                continue

            # Handle job search results (both RemoteOK and Tavily)
            if tool_name in ["search_remoteok_jobs", "search_jobs_tavily"] and not result.get("error"):
                # Store results in MongoDB with embeddings
                mongo_client = get_mongo_client()
                db = mongo_client["mongodbai"]
                collection = db["career_results"]

                # Generate all embeddings in parallel
                async def embed_job(job):
                    content_parts = [job.get('title', ''), job.get('company', '')]
                    if job.get('tags'):
                        content_parts.append(' '.join(job.get('tags', [])))
                    content = '\n'.join(filter(None, content_parts))
                    return await get_embedding(content)

                embeddings = await asyncio.gather(*[embed_job(job) for job in result["results"]])

                # Build bulk upsert operations
                operations = []
                for job, embedding in zip(result["results"], embeddings):
                    doc = {
                        "query": query,
                        "source": result["source"],
                        "url": job.get("url"),
                        "title": job.get("title"),
                        "company": job.get("company"),
                        "salary": job.get("salary"),
                        "location": job.get("location"),
                        "tags": job.get("tags", []),
                        "embedding": embedding,
                        "user_id": user_id,
                        "updated_at": datetime.utcnow()
                    }
                    operations.append(UpdateOne(
                        {"url": job.get("url")},
                        {"$set": doc, "$setOnInsert": {"created_at": datetime.utcnow()}},
                        upsert=True
                    ))

                # Execute all upserts in one batch
                if operations:
                    bulk_result = await collection.bulk_write(operations)
                    docs_created += bulk_result.upserted_count

                # Track URLs we just fetched
                fresh_urls = [job.get("url") for job in result["results"]]

                # Vector search to find matching jobs
                query_embedding = await get_embedding(query)

                project_stage = {
                    "$project": {
                        "title": 1,
                        "company": 1,
                        "url": 1,
                        "salary": 1,
                        "location": 1,
                        "tags": 1,
                        "source": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }

                # Vector search for FRESH jobs (from current fetch)
                fresh_pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "embedding",
                            "queryVector": query_embedding,
                            "numCandidates": 100,
                            "limit": max_results // 2,
                            "filter": {"url": {"$in": fresh_urls}}
                        }
                    },
                    project_stage
                ]

                # Run both searches and mark with is_fresh
                if fresh_urls:
                    # Vector search for FRESH jobs
                    fresh_jobs = [dict(job, is_fresh=True) for job in await collection.aggregate(fresh_pipeline).to_list(length=max_results // 2)]

                    # Vector search for HISTORICAL jobs (exclude current fetch)
                    historical_pipeline = [
                        {
                            "$vectorSearch": {
                                "index": "vector_index",
                                "path": "embedding",
                                "queryVector": query_embedding,
                                "numCandidates": 100,
                                "limit": max_results // 2,
                                "filter": {"url": {"$nin": fresh_urls}}
                            }
                        },
                        project_stage
                    ]
                    historical_jobs = [dict(job, is_fresh=False) for job in await collection.aggregate(historical_pipeline).to_list(length=max_results // 2)]
                else:
                    # No fresh jobs - search all historical (no filter needed)
                    fresh_jobs = []
                    historical_pipeline = [
                        {
                            "$vectorSearch": {
                                "index": "vector_index",
                                "path": "embedding",
                                "queryVector": query_embedding,
                                "numCandidates": 100,
                                "limit": max_results
                            }
                        },
                        project_stage
                    ]
                    historical_jobs = [dict(job, is_fresh=False) for job in await collection.aggregate(historical_pipeline).to_list(length=max_results)]

                matched_jobs.extend(fresh_jobs)
                matched_jobs.extend(historical_jobs)

                # Sort all jobs by score descending
                matched_jobs.sort(key=lambda x: x.get('score', 0), reverse=True)

                mongo_client.close()

    execution_time = int((time.time() - start_time) * 1000)

    return {
        "query": query,
        "tools_executed": tools_executed,
        "jobs": matched_jobs,
        "metadata": {
            "execution_time_ms": execution_time,
            "fresh_matches": len(fresh_jobs),
            "historical_matches": len(historical_jobs),
            "database_documents_created": docs_created
        }
    }

async def main():
    async with app.run() as agent_app:
        result = await career_agent(
            query="ceo",
            app_ctx=agent_app.context,
        )

        # Print test results
        print(f"\nQuery: {result['query']}")
        print(f"Tools: {result['tools_executed']}")
        print(f"Jobs found: {len(result['jobs'])}")
        print(f"Fresh: {result['metadata']['fresh_matches']}, Historical: {result['metadata']['historical_matches']}")
        print(f"Docs created: {result['metadata']['database_documents_created']}")
        print(f"Time: {result['metadata']['execution_time_ms']}ms\n")

        for i, job in enumerate(result["jobs"]):
            fresh_tag = "[FRESH]" if job.get('is_fresh') else "[HISTORICAL]"
            print(f"Job {i+1} {fresh_tag}: {job.get('title', 'N/A')}")
            if job.get('company'):
                print(f"  Company: {job.get('company')}")
            if job.get('salary'):
                print(f"  Salary: {job.get('salary')}")
            print(f"  Score: {job.get('score', 0):.4f}")
            print(f"  URL: {job.get('url', 'N/A')}\n")

if __name__ == "__main__":
    asyncio.run(main())

# When you're ready to deploy this MCPApp as a remote SSE server, run:
# > uv run mcp-agent deploy "career_agent" --no-auth
