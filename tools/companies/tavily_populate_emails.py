from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING
from datetime import datetime, timezone

from pymongo import MongoClient
from tavily import TavilyClient

from mcp_agent.core.context import Context as AppContext
from mcp_agent.logging.logger import get_logger

from config import get_mongodb_connection_string, get_tavily_api_key
from models.company import Company
from models.job import Job

if TYPE_CHECKING:
    from main import Response

logger = get_logger(__name__)


def extract_emails(text: str) -> list[str]:
    """Extract email addresses from text."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    # Filter out common non-email patterns and duplicates
    filtered = []
    seen = set()
    for email in emails:
        email_lower = email.lower()
        if email_lower not in seen and not email_lower.endswith('.png') and not email_lower.endswith('.jpg'):
            filtered.append(email)
            seen.add(email_lower)
    return filtered[:3]  # Max 3 emails


def get_company_website(company_name: str) -> str | None:
    """Try to get company website URL using Tavily search."""
    try:
        client = TavilyClient(get_tavily_api_key())
        response = client.search(
            query=f"{company_name} company official website",
            search_depth="basic",
            max_results=3,
            include_raw_content=False,
        )

        results = response.get("results", [])
        for result in results:
            url = result.get("url", "")
            # Try to find the company's own website, not job boards
            if company_name.lower().replace(" ", "") in url.lower().replace(" ", ""):
                return url

        # Return first result if no direct match
        if results:
            return results[0].get("url")

        return None
    except Exception as e:
        logger.error(f"get_company_website: error for {company_name}: {e}")
        return None


def search_company_emails(company_name: str, website_url: str | None) -> list[str]:
    """Search for HR/contact emails for a company."""
    try:
        client = TavilyClient(get_tavily_api_key())

        # Search for contact/HR emails
        query = f"{company_name} HR email contact careers jobs"
        if website_url:
            query = f"site:{website_url} contact email HR careers"

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_raw_content=True,
        )

        all_emails = []
        for result in response.get("results", []):
            content = result.get("content", "") + " " + result.get("raw_content", "")
            emails = extract_emails(content)
            all_emails.extend(emails)

        # Deduplicate and prioritize HR emails
        seen = set()
        hr_emails = []
        other_emails = []

        for email in all_emails:
            if email.lower() not in seen:
                seen.add(email.lower())
                if any(keyword in email.lower() for keyword in ['hr', 'recruit', 'career', 'jobs', 'talent', 'hiring']):
                    hr_emails.append(email)
                else:
                    other_emails.append(email)

        # Return HR emails first, then others, max 3
        result = (hr_emails + other_emails)[:3]
        return result

    except Exception as e:
        logger.error(f"search_company_emails: error for {company_name}: {e}")
        return []


async def tavily_populate_emails(
    query: str,
    jobs: list[Job],
    response: "Response",
    app_ctx: Optional[AppContext] = None
) -> "Response":
    """Find company emails from job listings using Tavily search.

    Args:
        query: Original search query
        jobs: List of jobs from previous tools
        response: Response object to update
        app_ctx: MCP context for logging

    Returns:
        Updated Response object with companies populated
    """
    if app_ctx:
        app_ctx.logger.info("tavily_populate_emails: starting", data={"query": query, "jobs_count": len(jobs)})

    logger.info(f"tavily_populate_emails: processing {len(jobs)} jobs")

    # Get unique companies from jobs
    company_names = set()
    for job in jobs[:10]:  # Limit to top 10 jobs
        if job.company:
            company_names.add(job.company)

    # If no companies from jobs, use existing companies from response or search for new ones
    if not company_names:
        # First, try to use companies already in response (from tavily_companies_search)
        if response.companies:
            logger.info(f"tavily_populate_emails: no jobs, using {len(response.companies)} existing companies")
            for company in response.companies[:5]:
                company_names.add(company.company_title)
        else:
            # Search for companies using Tavily
            logger.info("tavily_populate_emails: no jobs or existing companies, searching for companies")
            try:
                client = TavilyClient(get_tavily_api_key())
                search_query = f"{query} companies hiring careers"
                search_response = client.search(
                    query=search_query,
                    search_depth="basic",
                    max_results=5,
                    include_raw_content=False,
                )
                for result in search_response.get("results", []):
                    title = result.get("title", "")
                    company_name = title.split(" - ")[0].split(" | ")[0].strip()
                    if company_name and len(company_name) <= 50:
                        company_names.add(company_name)
                logger.info(f"tavily_populate_emails: found {len(company_names)} companies from search")
            except Exception as e:
                logger.error(f"tavily_populate_emails: error searching companies: {e}")

    logger.info(f"tavily_populate_emails: found {len(company_names)} unique companies")

    companies: list[Company] = []

    for company_name in list(company_names)[:1]:  # Limit to 1 company for speed
        try:
            logger.info(f"tavily_populate_emails: processing {company_name}")

            # Get company website
            website_url = get_company_website(company_name)
            logger.info(f"tavily_populate_emails: {company_name} website: {website_url}")

            # Search for emails
            emails = search_company_emails(company_name, website_url)
            logger.info(f"tavily_populate_emails: {company_name} emails: {emails}")

            company = Company(
                company_title=company_name,
                company_website_url=website_url,
                employee_emails=emails
            )
            companies.append(company)

        except Exception as e:
            logger.error(f"tavily_populate_emails: error processing {company_name}: {e}")
            # Still add company with empty emails
            companies.append(Company(
                company_title=company_name,
                company_website_url=None,
                employee_emails=[]
            ))

    # Save companies to MongoDB
    if companies:
        try:
            conn_str = get_mongodb_connection_string()
            mongo_client = MongoClient(conn_str)
            db = mongo_client["mongodbai"]
            collection = db["companies"]

            for company in companies:
                doc = {
                    "company_title": company.company_title,
                    "company_website_url": company.company_website_url,
                    "employee_emails": company.employee_emails,
                    "query": query,
                    "updated_at": datetime.now(timezone.utc)
                }

                # Upsert by company_title
                collection.update_one(
                    {"company_title": company.company_title},
                    {"$set": doc, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
                    upsert=True
                )

            logger.info(f"tavily_populate_emails: saved {len(companies)} companies to MongoDB")
            mongo_client.close()

        except Exception as e:
            logger.error(f"tavily_populate_emails: MongoDB error: {e}")

    # Merge with existing companies (don't overwrite tavily_companies_search results)
    # Update existing companies with emails, or add new ones
    existing_by_title = {c.company_title.lower(): c for c in response.companies}

    for company in companies:
        key = company.company_title.lower()
        if key in existing_by_title:
            # Update existing company with emails if it didn't have any
            existing = existing_by_title[key]
            if not existing.employee_emails and company.employee_emails:
                existing.employee_emails = company.employee_emails
            if not existing.company_website_url and company.company_website_url:
                existing.company_website_url = company.company_website_url
        else:
            # Add new company
            response.companies.append(company)

    if app_ctx:
        app_ctx.logger.info("tavily_populate_emails: complete", data={"companies_processed": len(companies), "total_companies": len(response.companies)})

    return response
