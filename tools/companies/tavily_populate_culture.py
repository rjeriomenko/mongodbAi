from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from datetime import datetime, timezone

from pymongo import MongoClient
from tavily import TavilyClient

from mcp_agent.core.context import Context as AppContext
from mcp_agent.logging.logger import get_logger

from config import get_mongodb_connection_string, get_tavily_api_key
from models.company import Company

if TYPE_CHECKING:
    from main import Response

logger = get_logger(__name__)


def search_company_culture(company_name: str) -> str | None:
    """Search for company culture using Tavily."""
    try:
        client = TavilyClient(get_tavily_api_key())

        query = f"{company_name} company culture values work environment employee reviews"

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_raw_content=False,
        )

        # Combine content from results
        culture_texts = []
        for result in response.get("results", []):
            content = result.get("content", "")
            if content:
                culture_texts.append(content)

        if culture_texts:
            # Combine and truncate to reasonable length
            combined = " ".join(culture_texts)
            return combined[:1000]  # Max 1000 chars

        return None

    except Exception as e:
        logger.error(f"search_company_culture: error for {company_name}: {e}")
        return None


async def tavily_populate_culture(
    query: str,
    response: "Response",
    app_ctx: Optional[AppContext] = None
) -> "Response":
    """Find company culture from query or prominent company in field.

    Args:
        query: Original search query
        response: Response object to update
        app_ctx: MCP context for logging

    Returns:
        Updated Response object with company culture populated
    """
    if app_ctx:
        app_ctx.logger.info("tavily_populate_culture: starting", data={"query": query})

    logger.info(f"tavily_populate_culture: processing query: {query}")

    # Try to find a company name from existing companies in response
    company_name = None
    target_company = None

    if response.companies:
        # Use first company from response
        target_company = response.companies[0]
        company_name = target_company.company_title
        logger.info(f"tavily_populate_culture: using company from response: {company_name}")
    else:
        # Extract company from query or search for prominent company
        try:
            client = TavilyClient(get_tavily_api_key())
            search_query = f"{query} top companies employers"
            search_response = client.search(
                query=search_query,
                search_depth="basic",
                max_results=1,
                include_raw_content=False,
            )

            results = search_response.get("results", [])
            if results:
                title = results[0].get("title", "")
                # Extract company name from title
                company_name = title.split(" - ")[0].split(" | ")[0].strip()
                if company_name and len(company_name) <= 50:
                    logger.info(f"tavily_populate_culture: found company from search: {company_name}")
                else:
                    company_name = None

        except Exception as e:
            logger.error(f"tavily_populate_culture: error finding company: {e}")

    if not company_name:
        logger.info("tavily_populate_culture: no company found, skipping")
        return response

    # Search for company culture
    culture = search_company_culture(company_name)
    logger.info(f"tavily_populate_culture: found culture for {company_name}: {len(culture) if culture else 0} chars")

    if culture:
        # Create or update company with culture
        company = Company(
            company_title=company_name,
            company_website_url=target_company.company_website_url if target_company else None,
            employee_emails=target_company.employee_emails if target_company else [],
            culture=culture
        )

        # Save to MongoDB
        try:
            conn_str = get_mongodb_connection_string()
            mongo_client = MongoClient(conn_str)
            db = mongo_client["mongodbai"]
            collection = db["companies"]

            doc = {
                "company_title": company.company_title,
                "company_website_url": company.company_website_url,
                "employee_emails": company.employee_emails,
                "culture": company.culture,
                "query": query,
                "updated_at": datetime.now(timezone.utc)
            }

            # Upsert by company_title
            collection.update_one(
                {"company_title": company.company_title},
                {"$set": doc, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
                upsert=True
            )

            logger.info(f"tavily_populate_culture: saved {company_name} to MongoDB")
            mongo_client.close()

        except Exception as e:
            logger.error(f"tavily_populate_culture: MongoDB error: {e}")

        # Merge with existing companies
        existing_by_title = {c.company_title.lower(): c for c in response.companies}
        key = company.company_title.lower()

        if key in existing_by_title:
            # Update existing company with culture
            existing = existing_by_title[key]
            if not existing.culture:
                existing.culture = culture
        else:
            # Add new company
            response.companies.append(company)

    if app_ctx:
        app_ctx.logger.info("tavily_populate_culture: complete", data={"company": company_name, "culture_length": len(culture) if culture else 0})

    return response
