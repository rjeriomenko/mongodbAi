from __future__ import annotations

import time
import json
from typing import Optional, Any, Union

from mcp_agent.core.context import Context as AppContext
from urllib.parse import quote
from urllib.request import urlopen, Request
from datetime import datetime

from pymongo import MongoClient, UpdateOne

from mcp_agent.app import MCPApp
from mcp_agent.logging.logger import get_logger
from mcp_agent.config import get_settings

logger = get_logger(__name__)


def get_mongodb_connection_string() -> str:
    """Get MongoDB connection string from settings."""
    settings = get_settings()
    if settings.mcp and settings.mcp.servers:
        mongodb_server = settings.mcp.servers.get('mongodb')
        if mongodb_server and hasattr(mongodb_server, 'env'):
            conn_str = mongodb_server.env.get('MDB_MCP_CONNECTION_STRING')
            if conn_str:
                return conn_str
    raise ValueError("Missing MongoDB connection string")


def get_google_api_key() -> str:
    """Get Google API key from settings."""
    settings = get_settings()
    if settings.google and settings.google.api_key:
        return settings.google.api_key
    raise ValueError("Missing Google API key")


def generate_embedding_http(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Generate embedding using Gemini REST API with urllib."""
    api_key = get_google_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"

    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type
    }

    req = Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )

    with urlopen(req, timeout=30) as response:
        result = json.loads(response.read().decode('utf-8'))
        return result['embedding']['values']


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using Gemini REST API."""
    logger.info(f"generate_embeddings: generating for {len(texts)} texts")

    embeddings = []
    for i, text in enumerate(texts):
        try:
            embedding = generate_embedding_http(text, "RETRIEVAL_DOCUMENT")
            embeddings.append(embedding)
            if (i + 1) % 5 == 0:
                logger.info(f"generate_embeddings: processed {i + 1}/{len(texts)}")
        except Exception as e:
            logger.error(f"generate_embeddings: error for text {i}: {e}")
            embeddings.append([0.0] * 768)

    logger.info(f"generate_embeddings: completed {len(embeddings)} embeddings")
    return embeddings


def generate_query_embedding(query: str) -> list[float]:
    """Generate embedding for a search query using Gemini REST API."""
    logger.info(f"generate_query_embedding: generating for query")

    try:
        embedding = generate_embedding_http(query, "RETRIEVAL_QUERY")
        logger.info("generate_query_embedding: success")
        return embedding
    except Exception as e:
        logger.error(f"generate_query_embedding: error: {e}")
        return [0.0] * 768


def vector_search_jobs(collection, query_embedding: list[float], urls: list[str], is_fresh: bool, limit: int = 10) -> list[dict]:
    """Search for jobs using MongoDB Atlas Vector Search.

    Args:
        collection: MongoDB collection
        query_embedding: Query vector for similarity search
        urls: URLs to filter by
        is_fresh: If True, search for jobs IN urls ($in). If False, search for jobs NOT IN urls ($nin).
        limit: Max results to return
    """
    job_type = "fresh" if is_fresh else "historical"
    filter_op = "$in" if is_fresh else "$nin"

    logger.info(f"vector_search_jobs: searching for {limit} {job_type} jobs")

    if is_fresh and not urls:
        return []

    try:
        # Build filter
        url_filter = {"url": {filter_op: urls}} if urls else {}

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": url_filter
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "url": 1,
                    "title": 1,
                    "company": 1,
                    "salary_min": 1,
                    "salary_max": 1,
                    "location": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))
        logger.info(f"vector_search_jobs: found {len(results)} {job_type} jobs")

        # Mark fresh/historical
        for job in results:
            job["is_fresh"] = is_fresh

        return results

    except Exception as e:
        logger.error(f"vector_search_jobs: error: {e}")
        return []


async def search_remoteok_jobs(query: str, max_results: int = 20) -> dict:
    """Fetch job listings from RemoteOK API."""
    logger.info(f"search_remoteok_jobs: starting with query={query}")

    try:
        # Parse query
        search_query = query.split()[0].lower() if query else "developer"
        logger.info(f"search_remoteok_jobs: parsed search_query={search_query}")

        # Build URL
        encoded_query = quote(search_query)
        url = f"https://remoteok.com/api?tag={encoded_query}&limit={max_results}"
        logger.info(f"search_remoteok_jobs: url={url}")

        # Make request
        logger.info("search_remoteok_jobs: making HTTP request...")
        req = Request(url, headers={"User-Agent": "CareerAgent/1.0"})
        with urlopen(req, timeout=30) as response:
            response_text = response.read().decode('utf-8')
            logger.info(f"search_remoteok_jobs: got response, length={len(response_text)}")

        # Parse JSON
        logger.info("search_remoteok_jobs: parsing JSON...")
        all_jobs = json.loads(response_text)
        raw_jobs = all_jobs[1:max_results + 1] if len(all_jobs) > 1 else []
        logger.info(f"search_remoteok_jobs: parsed {len(raw_jobs)} jobs")

        # Build simple job list
        jobs = []
        for job in raw_jobs:
            jobs.append({
                "url": job.get("url", ""),
                "title": job.get("position", ""),
                "company": job.get("company", ""),
                "salary_min": job.get("salary_min"),
                "salary_max": job.get("salary_max"),
                "location": job.get("location", "Worldwide"),
            })

        logger.info(f"search_remoteok_jobs: returning {len(jobs)} jobs")
        return {"jobs": jobs, "error": None}

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"search_remoteok_jobs: error={error_msg}")
        logger.error(f"search_remoteok_jobs: traceback={traceback.format_exc()}")
        return {"jobs": [], "error": error_msg}


# Create the MCPApp
app = MCPApp(
    name="career_agent",
    description="Career search agent with jobs, communities, and personalized recommendations",
)


@app.tool
async def career_agent(
    query: str,
    run_tavily: bool = False,
    include_jobs: bool = True,
    include_communities: bool = True,
    max_results: int = 20,
    user_id: str = "anonymous",
    app_ctx: Optional[AppContext] = None
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
        app_ctx: MCP context for logging and server access
    """
    start_time = time.time()

    if app_ctx:
        app_ctx.logger.info("Starting career_agent", data={"query": query})

    if app_ctx:
        app_ctx.logger.info("Fetching jobs from RemoteOK", data={})

    result = await search_remoteok_jobs(query, max_results)
    fetched_jobs = result.get("jobs", [])
    fetched_urls = [job["url"] for job in fetched_jobs]

    if app_ctx:
        app_ctx.logger.info("Fetched jobs", data={"count": len(fetched_jobs)})

    if result.get("error"):
        if app_ctx:
            app_ctx.logger.error("RemoteOK error", data={"error": result['error']})

    if app_ctx:
        app_ctx.logger.info("Generating query embedding", data={})

    try:
        query_embedding = generate_query_embedding(query)
        if app_ctx:
            app_ctx.logger.info("Query embedding generated", data={"length": len(query_embedding)})
    except Exception as e:
        if app_ctx:
            app_ctx.logger.error("Query embedding failed", data={"error": str(e)})
        raise

    docs_upserted = 0
    historical_jobs = []
    fresh_jobs = []
    half_results = max_results // 2

    try:
        logger.info("career_agent: connecting to MongoDB")
        conn_str = get_mongodb_connection_string()
        mongo_client = MongoClient(conn_str)
        db = mongo_client["mongodbai"]
        collection = db["career_results"]

        if fetched_jobs:
            logger.info(f"career_agent: generating embeddings for {len(fetched_jobs)} jobs")
            job_texts = [f"{job['title']} at {job['company']} - {job.get('location', '')}" for job in fetched_jobs]
            job_embeddings = generate_embeddings(job_texts)

            operations = []
            for job, embedding in zip(fetched_jobs, job_embeddings):
                doc = {
                    "query": query,
                    "url": job["url"],
                    "title": job["title"],
                    "company": job["company"],
                    "salary_min": job.get("salary_min"),
                    "salary_max": job.get("salary_max"),
                    "location": job.get("location"),
                    "user_id": user_id,
                    "embedding": embedding,
                    "updated_at": datetime.utcnow()
                }
                operations.append(UpdateOne(
                    {"url": job["url"]},
                    {"$set": doc, "$setOnInsert": {"created_at": datetime.utcnow()}},
                    upsert=True
                ))

            logger.info(f"career_agent: upserting {len(operations)} jobs to MongoDB")
            bulk_result = collection.bulk_write(operations)
            docs_upserted = bulk_result.upserted_count

        logger.info(f"career_agent: running vector search for {half_results} results each")
        historical_jobs = vector_search_jobs(collection, query_embedding, fetched_urls, is_fresh=False, limit=half_results)
        fresh_jobs = vector_search_jobs(collection, query_embedding, fetched_urls, is_fresh=True, limit=half_results)

        mongo_client.close()

    except Exception as e:
        import traceback
        logger.error(f"career_agent: MongoDB/embedding error: {type(e).__name__}: {e}")
        logger.error(f"career_agent: traceback: {traceback.format_exc()}")

    combined_jobs = historical_jobs + fresh_jobs
    combined_jobs.sort(key=lambda x: x.get("score", 0), reverse=True)

    fresh_count = sum(1 for j in combined_jobs if j.get("is_fresh", False))
    historical_count = len(combined_jobs) - fresh_count

    execution_time = int((time.time() - start_time) * 1000)
    logger.info(f"career_agent: complete - {len(combined_jobs)} jobs ({fresh_count} fresh, {historical_count} historical) in {execution_time}ms")

    return {
        "query": query,
        "jobs": combined_jobs,
        "fresh_count": fresh_count,
        "historical_count": historical_count,
        "docs_upserted": docs_upserted,
        "error": result.get("error"),
        "execution_time_ms": execution_time
    }


async def main():
    import asyncio
    async with app.run() as agent_app:
        result = await career_agent(
            query="python developer",
        )
        print(f"Result: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# Deploy as remote SSE server:
# > uv run mcp-agent deploy "career_agent" --no-auth
