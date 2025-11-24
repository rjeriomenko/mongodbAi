from __future__ import annotations

import json
from typing import Optional, TYPE_CHECKING
from datetime import datetime, timezone
from urllib.request import urlopen, Request

from pymongo import MongoClient, UpdateOne
from tavily import TavilyClient

from mcp_agent.core.context import Context as AppContext
from mcp_agent.logging.logger import get_logger

from config import get_mongodb_connection_string, get_tavily_api_key, get_google_api_key
from models.company import Company

if TYPE_CHECKING:
    from main import Response

logger = get_logger(__name__)


def generate_embedding_http(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Generate embedding using Gemini REST API."""
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

        all_embeddings = []
        embeddings_data = result.get('embeddings', [])

        for emb_data in embeddings_data:
            embedding = emb_data.get('values', [0.0] * 768)
            all_embeddings.append(embedding)

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
    """Generate embedding for a search query."""
    logger.info(f"generate_query_embedding: generating for query")
    try:
        embedding = generate_embedding_http(query, "RETRIEVAL_QUERY")
        logger.info("generate_query_embedding: success")
        return embedding
    except Exception as e:
        logger.error(f"generate_query_embedding: error: {e}")
        return [0.0] * 768


def vector_search_companies(collection, query_embedding: list[float], urls: list[str], is_fresh: bool, limit: int = 5) -> list[Company]:
    """Search for companies using MongoDB Atlas Vector Search."""
    company_type = "fresh" if is_fresh else "historical"
    filter_op = "$in" if is_fresh else "$nin"

    logger.info(f"vector_search_companies: searching for {limit} {company_type} companies, urls count: {len(urls)}")

    if is_fresh and not urls:
        logger.info("vector_search_companies: no URLs for fresh search, returning empty")
        return []

    try:
        url_filter = {filter_op: urls} if urls else {}
        filter_dict = {"company_website_url": url_filter} if url_filter else {}

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "company_vector_index",
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
                    "company_title": 1,
                    "company_website_url": 1,
                    "employee_emails": 1,
                    "culture": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))
        logger.info(f"vector_search_companies: found {len(results)} {company_type} companies")

        companies = []
        for doc in results:
            company = Company.from_mongo_doc(doc)
            companies.append(company)

        return companies

    except Exception as e:
        logger.error(f"vector_search_companies: error: {e}")
        return []


async def tavily_companies_search(
    query: str,
    max_results: int,
    response: "Response",
    app_ctx: Optional[AppContext] = None
) -> "Response":
    """Search for companies using Tavily and perform vector search.

    Args:
        query: Search query
        max_results: Maximum results to return
        response: Response object to update
        app_ctx: MCP context for logging

    Returns:
        Updated Response object with companies populated
    """
    if app_ctx:
        app_ctx.logger.info("tavily_companies_search: starting", data={"query": query})

    logger.info(f"tavily_companies_search: searching for companies related to: {query}")

    # Search for companies using Tavily
    fetched_companies: list[Company] = []
    error = None

    try:
        client = TavilyClient(get_tavily_api_key())

        # Search for companies in the field
        search_query = f"{query} companies hiring careers jobs"
        response_data = client.search(
            query=search_query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=False,
        )

        results = response_data.get("results", [])
        logger.info(f"tavily_companies_search: got {len(results)} results from Tavily")

        # Extract company info from results
        seen_urls = set()
        for result in results:
            url = result.get("url", "")
            title = result.get("title", "")

            # Try to extract company name from title or URL
            company_name = title.split(" - ")[0].split(" | ")[0].strip()
            if not company_name or len(company_name) > 50:
                # Use domain as company name
                from urllib.parse import urlparse
                parsed = urlparse(url)
                company_name = parsed.netloc.replace("www.", "").split(".")[0].title()

            if url and url not in seen_urls:
                seen_urls.add(url)
                fetched_companies.append(Company(
                    company_title=company_name,
                    company_website_url=url,
                    employee_emails=[]
                ))

        logger.info(f"tavily_companies_search: extracted {len(fetched_companies)} companies")

    except Exception as e:
        import traceback
        error = f"{type(e).__name__}: {e}"
        logger.error(f"tavily_companies_search: error={error}")
        logger.error(f"tavily_companies_search: traceback={traceback.format_exc()}")

    fetched_urls = [c.company_website_url for c in fetched_companies if c.company_website_url]

    if error:
        if app_ctx:
            app_ctx.logger.error("Tavily companies search error", data={"error": error})

    # Generate query embedding
    try:
        query_embedding = generate_query_embedding(query)
    except Exception as e:
        if app_ctx:
            app_ctx.logger.error("Query embedding failed", data={"error": str(e)})
        raise

    docs_upserted = 0
    historical_companies: list[Company] = []
    fresh_companies: list[Company] = []
    half_results = max_results // 2

    try:
        logger.info("tavily_companies_search: connecting to MongoDB")
        conn_str = get_mongodb_connection_string()
        mongo_client = MongoClient(conn_str)
        db = mongo_client["mongodbai"]
        collection = db["companies"]

        if fetched_companies:
            # Check which companies already exist
            existing_urls = set()
            existing_docs = collection.find(
                {"company_website_url": {"$in": fetched_urls}},
                {"company_website_url": 1, "_id": 0}
            )
            for doc in existing_docs:
                if doc.get("company_website_url"):
                    existing_urls.add(doc["company_website_url"])

            new_companies = [c for c in fetched_companies if c.company_website_url not in existing_urls]
            logger.info(f"tavily_companies_search: {len(new_companies)} new companies to embed (skipping {len(existing_urls)} existing)")

            if new_companies:
                # Generate embeddings for new companies
                company_texts = [f"{c.company_title} {c.company_website_url or ''}" for c in new_companies]
                company_embeddings = generate_embeddings_batch(company_texts)

                operations = []
                for company, embedding in zip(new_companies, company_embeddings):
                    doc = {
                        "company_title": company.company_title,
                        "company_website_url": company.company_website_url,
                        "employee_emails": company.employee_emails,
                        "culture": company.culture,
                        "query": query,
                        "embedding": embedding,
                        "updated_at": datetime.now(timezone.utc)
                    }
                    operations.append(UpdateOne(
                        {"company_website_url": company.company_website_url},
                        {"$set": doc, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
                        upsert=True
                    ))

                logger.info(f"tavily_companies_search: upserting {len(operations)} new companies to MongoDB")
                bulk_result = collection.bulk_write(operations)
                docs_upserted = bulk_result.upserted_count
                logger.info(f"tavily_companies_search: bulk_write result - upserted: {bulk_result.upserted_count}, modified: {bulk_result.modified_count}")
            else:
                logger.info("tavily_companies_search: all companies already in DB, skipping embedding generation")

        # Vector search for companies
        logger.info(f"tavily_companies_search: running vector search for {half_results} results each")
        historical_companies = vector_search_companies(collection, query_embedding, fetched_urls, is_fresh=False, limit=half_results)
        fresh_companies = vector_search_companies(collection, query_embedding, fetched_urls, is_fresh=True, limit=half_results)

        mongo_client.close()

    except Exception as e:
        import traceback
        logger.error(f"tavily_companies_search: MongoDB/embedding error: {type(e).__name__}: {e}")
        logger.error(f"tavily_companies_search: traceback: {traceback.format_exc()}")

    # Combine results
    combined_companies = historical_companies + fresh_companies

    fresh_count = len(fresh_companies)
    historical_count = len(historical_companies)

    response.companies = combined_companies

    logger.info(f"tavily_companies_search: complete - {len(combined_companies)} companies ({fresh_count} fresh, {historical_count} historical)")

    if app_ctx:
        app_ctx.logger.info("tavily_companies_search: complete", data={
            "total": len(combined_companies),
            "fresh": fresh_count,
            "historical": historical_count,
            "created": docs_upserted
        })

    return response
