from __future__ import annotations

import json
from typing import Optional, TYPE_CHECKING
from urllib.parse import quote
from urllib.request import urlopen, Request
from datetime import datetime, timezone

from pymongo import MongoClient, UpdateOne

from mcp_agent.core.context import Context as AppContext
from mcp_agent.logging.logger import get_logger

from config import get_mongodb_connection_string, get_google_api_key
from models.job import Job

if TYPE_CHECKING:
    from main import Response

logger = get_logger(__name__)


def generate_embedding_http(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Generate embedding using Gemini REST API with urllib (sync version for query embedding)."""
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


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts in a single batch API call."""
    logger.info(f"generate_embeddings_batch: generating for {len(texts)} texts")

    api_key = get_google_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={api_key}"

    try:
        # Build batch request with all texts
        requests = []
        for text in texts:
            requests.append({
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": text}]},
                "taskType": "RETRIEVAL_DOCUMENT"
            })

        payload = {"requests": requests}

        req = Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )

        with urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))

        # Extract embeddings from batch response
        all_embeddings = []
        embeddings_data = result.get('embeddings', [])

        for i, emb_data in enumerate(embeddings_data):
            embedding = emb_data.get('values', [0.0] * 768)
            all_embeddings.append(embedding)

        # Pad with zeros if some failed
        while len(all_embeddings) < len(texts):
            all_embeddings.append([0.0] * 768)

        successful = sum(1 for emb in all_embeddings if emb[0] != 0.0)
        logger.info(f"generate_embeddings_batch: completed {len(all_embeddings)} embeddings ({successful} successful)")
        return all_embeddings

    except Exception as e:
        import traceback
        logger.error(f"generate_embeddings_batch: error: {type(e).__name__}: {e}")
        logger.error(f"generate_embeddings_batch: traceback: {traceback.format_exc()}")
        return [[0.0] * 768 for _ in texts]


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


def vector_search_jobs(collection, query_embedding: list[float], urls: list[str], is_fresh: bool, limit: int = 10) -> list[Job]:
    """Search for jobs using MongoDB Atlas Vector Search."""
    job_type = "fresh" if is_fresh else "historical"
    filter_op = "$in" if is_fresh else "$nin"

    logger.info(f"vector_search_jobs: searching for {limit} {job_type} jobs, urls count: {len(urls)}")

    if is_fresh and not urls:
        logger.info("vector_search_jobs: no URLs for fresh search, returning empty")
        return []

    try:
        url_filter = {filter_op: urls} if urls else {}
        filter_dict = {"url": url_filter} if url_filter else {}
        logger.info(f"vector_search_jobs: filter_dict keys: {list(filter_dict.keys())}, has url filter: {'url' in filter_dict}")
        if urls:
            logger.info(f"vector_search_jobs: first url in filter: {urls[0][:80]}...")

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": filter_dict
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "url": 1,
                    "title": 1,
                    "company": 1,
                    "source": 1,
                    "salary_min": 1,
                    "salary_max": 1,
                    "location": 1,
                    "tags": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))
        logger.info(f"vector_search_jobs: found {len(results)} {job_type} jobs")

        jobs = []
        for doc in results:
            job = Job.from_mongo_doc(doc)
            job.is_fresh = is_fresh
            jobs.append(job)

        return jobs

    except Exception as e:
        logger.error(f"vector_search_jobs: error: {e}")
        return []


async def remoteok_search_jobs(
    query: str,
    search_terms: list[str],
    max_results: int,
    user_id: str,
    response: "Response",
    app_ctx: Optional[AppContext] = None
) -> "Response":
    """Fetch jobs from RemoteOK, store in MongoDB, and perform vector search.

    Args:
        query: Search query
        search_terms: List of search terms for RemoteOK API
        max_results: Maximum results to return
        user_id: User identifier
        response: Response object to update
        app_ctx: MCP context for logging

    Returns:
        Updated Response object with jobs and metadata
    """
    if app_ctx:
        app_ctx.logger.info("Fetching jobs from RemoteOK", data={"tag": search_terms[0] if search_terms else "developer"})

    # Fetch from RemoteOK API
    fetched_jobs: list[Job] = []
    error = None

    try:
        search_tag = search_terms[0] if search_terms else "developer"
        encoded_query = quote(search_tag)
        url = f"https://remoteok.com/api?tag={encoded_query}&limit={max_results}"
        logger.info(f"remoteok_search_jobs: url={url}")

        req = Request(url, headers={"User-Agent": "CareerAgent/1.0"})
        with urlopen(req, timeout=30) as resp:
            response_text = resp.read().decode('utf-8')
            logger.info(f"remoteok_search_jobs: got response, length={len(response_text)}")

        all_jobs = json.loads(response_text)
        raw_jobs = all_jobs[1:max_results + 1] if len(all_jobs) > 1 else []
        logger.info(f"remoteok_search_jobs: parsed {len(raw_jobs)} jobs")

        for job in raw_jobs:
            fetched_jobs.append(Job(
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

        logger.info(f"remoteok_search_jobs: fetched {len(fetched_jobs)} jobs")

    except Exception as e:
        import traceback
        error = f"{type(e).__name__}: {e}"
        logger.error(f"remoteok_search_jobs: error={error}")
        logger.error(f"remoteok_search_jobs: traceback={traceback.format_exc()}")

    fetched_urls = [job.url for job in fetched_jobs]

    if app_ctx:
        app_ctx.logger.info("Fetched jobs", data={"count": len(fetched_jobs)})

    if error:
        if app_ctx:
            app_ctx.logger.error("RemoteOK error", data={"error": error})
        response.error = error

    # Generate query embedding
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
    docs_modified = 0
    historical_jobs: list[Job] = []
    fresh_jobs: list[Job] = []
    half_results = max_results // 2

    try:
        logger.info("remoteok_search_jobs: connecting to MongoDB")
        conn_str = get_mongodb_connection_string()
        mongo_client = MongoClient(conn_str)
        db = mongo_client["mongodbai"]
        collection = db["career_results"]

        if fetched_jobs:
            # Check which jobs already exist in DB to skip embedding generation
            existing_urls = set()
            existing_docs = collection.find(
                {"url": {"$in": fetched_urls}},
                {"url": 1, "_id": 0}
            )
            for doc in existing_docs:
                existing_urls.add(doc["url"])

            new_jobs = [job for job in fetched_jobs if job.url not in existing_urls]
            logger.info(f"remoteok_search_jobs: {len(new_jobs)} new jobs to embed (skipping {len(existing_urls)} existing)")

            if new_jobs:
                # Only generate embeddings for new jobs
                job_texts = [f"{job.title} at {job.company} - {job.location or ''}" for job in new_jobs]
                job_embeddings = generate_embeddings_batch(job_texts)

                operations = []
                for job, embedding in zip(new_jobs, job_embeddings):
                    doc = {
                        "query": query,
                        "url": job.url,
                        "title": job.title,
                        "company": job.company,
                        "source": job.source,
                        "salary_min": job.salary_min,
                        "salary_max": job.salary_max,
                        "date_posted": job.date_posted,
                        "location": job.location,
                        "tags": job.tags,
                        "user_id": user_id,
                        "embedding": embedding,
                        "updated_at": datetime.now(timezone.utc)
                    }
                    operations.append(UpdateOne(
                        {"url": job.url},
                        {"$set": doc, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
                        upsert=True
                    ))

                logger.info(f"remoteok_search_jobs: upserting {len(operations)} new jobs to MongoDB")
                bulk_result = collection.bulk_write(operations)
                docs_upserted = bulk_result.upserted_count
                docs_modified = bulk_result.modified_count
                logger.info(f"remoteok_search_jobs: bulk_write result - upserted: {bulk_result.upserted_count}, modified: {bulk_result.modified_count}, matched: {bulk_result.matched_count}")
            else:
                logger.info("remoteok_search_jobs: all jobs already in DB, skipping embedding generation")

        logger.info(f"remoteok_search_jobs: running vector search for {half_results} results each")
        historical_jobs = vector_search_jobs(collection, query_embedding, fetched_urls, is_fresh=False, limit=half_results)
        fresh_jobs = vector_search_jobs(collection, query_embedding, fetched_urls, is_fresh=True, limit=half_results)

        mongo_client.close()

    except Exception as e:
        import traceback
        logger.error(f"remoteok_search_jobs: MongoDB/embedding error: {type(e).__name__}: {e}")
        logger.error(f"remoteok_search_jobs: traceback: {traceback.format_exc()}")

    # Combine and sort results
    combined_jobs = historical_jobs + fresh_jobs
    combined_jobs.sort(key=lambda x: x.score or 0, reverse=True)

    fresh_count = sum(1 for j in combined_jobs if j.is_fresh)
    historical_count = len(combined_jobs) - fresh_count

    # Update response
    response.jobs = combined_jobs
    response.metadata.fresh_matches = fresh_count
    response.metadata.historical_matches = historical_count
    response.metadata.database_documents_created = docs_upserted
    response.metadata.database_documents_updated = docs_modified

    logger.info(f"remoteok_search_jobs: complete - {len(combined_jobs)} jobs ({fresh_count} fresh, {historical_count} historical)")

    return response
