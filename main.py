from __future__ import annotations

import asyncio
import time
import os
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
from google import genai

from mcp_agent.app import MCPApp
from mcp_agent.core.context import Context

from models.job import Job
from tools.jobs import search_remoteok_jobs, search_jobs_tavily


# Load .env file
load_dotenv()


def get_env(key: str) -> str:
    """Get required environment variable."""
    value = os.environ.get(key, "")
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


async def get_embedding(text: str) -> list[float]:
    """Get embedding from Google."""
    client = genai.Client(api_key=get_env("GOOGLE_API_KEY"))
    result = await asyncio.to_thread(
        client.models.embed_content,
        model="text-embedding-004",
        contents=text
    )
    return result.embeddings[0].values


def get_mongo_client():
    """Get MongoDB client instance."""
    return AsyncIOMotorClient(get_env("MDB_MCP_CONNECTION_STRING"))


# Create the MCPApp
app = MCPApp(
    name="career_agent",
    description="Career search agent with jobs, communities, and personalized recommendations",
)


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
        run_tavily: Also search using Tavily
        include_jobs: Include job listings
        include_communities: Include community/forum results
        max_results: Maximum total results to return
        user_id: User identifier for personalization
    """
    start_time = time.time()
    tools_executed = []
    matched_jobs: list[dict] = []
    docs_created = 0
    fresh_jobs: list[dict] = []
    historical_jobs: list[dict] = []

    # Build dict of selected tools
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

        for i, result in enumerate(results):
            tool_name = tool_names[i]
            tools_executed.append(tool_name)

            if isinstance(result, Exception):
                continue

            if tool_name in ["search_remoteok_jobs", "search_jobs_tavily"] and not result.get("error"):
                mongo_client = get_mongo_client()
                db = mongo_client["mongodbai"]
                collection = db["career_results"]

                jobs: list[Job] = result["results"]

                # Generate all embeddings in parallel
                async def embed_job(job: Job) -> list[float]:
                    content_parts = [job.title, job.company]
                    if job.tags:
                        content_parts.append(' '.join(job.tags))
                    content = '\n'.join(filter(None, content_parts))
                    return await get_embedding(content)

                embeddings = await asyncio.gather(*[embed_job(job) for job in jobs])

                # Build bulk upsert operations
                operations = []
                for job, embedding in zip(jobs, embeddings):
                    doc = {
                        "query": query,
                        "source": job.source,
                        "url": job.url,
                        "title": job.title,
                        "company": job.company,
                        "salary_min": job.salary_min,
                        "salary_max": job.salary_max,
                        "location": job.location,
                        "tags": job.tags,
                        "date_posted": job.date_posted,
                        "embedding": embedding,
                        "user_id": user_id,
                        "updated_at": datetime.utcnow()
                    }
                    operations.append(UpdateOne(
                        {"url": job.url},
                        {"$set": doc, "$setOnInsert": {"created_at": datetime.utcnow()}},
                        upsert=True
                    ))

                # Execute all upserts in one batch
                if operations:
                    bulk_result = await collection.bulk_write(operations)
                    docs_created += bulk_result.upserted_count

                # Track URLs we just fetched
                fresh_urls = [job.url for job in jobs]

                # Vector search
                query_embedding = await get_embedding(query)

                project_stage = {
                    "$project": {
                        "title": 1,
                        "company": 1,
                        "url": 1,
                        "salary_min": 1,
                        "salary_max": 1,
                        "location": 1,
                        "tags": 1,
                        "source": 1,
                        "date_posted": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }

                # Vector search for FRESH jobs
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

                if fresh_urls:
                    fresh_jobs = [dict(job, is_fresh=True) for job in await collection.aggregate(fresh_pipeline).to_list(length=max_results // 2)]

                    # Vector search for HISTORICAL jobs
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
            query="python developer",
            app_ctx=agent_app.context,
        )

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
            if job.get('salary_min') or job.get('salary_max'):
                salary_min = job.get('salary_min', 0)
                salary_max = job.get('salary_max', 0)
                print(f"  Salary: ${salary_min:,}-${salary_max:,}")
            print(f"  Score: {job.get('score', 0):.4f}")
            print(f"  URL: {job.get('url', 'N/A')}\n")


if __name__ == "__main__":
    asyncio.run(main())

# Deploy as remote SSE server:
# > uv run mcp-agent deploy "career_agent" --no-auth
