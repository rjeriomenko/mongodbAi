from __future__ import annotations

import time
import json
from typing import Optional

from mcp_agent.core.context import Context as AppContext
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.app import MCPApp
from mcp_agent.logging.logger import get_logger
from pydantic import BaseModel

from models.job import Job
from models.company import Company
from tools.insights import tavily_role_research, tavily_market_trends, tavily_learning_resources, synthesize_insights
from tools.jobs.remoteok_search_jobs import remoteok_search_jobs
from tools.companies.tavily_populate_emails import tavily_populate_emails
from tools.companies.tavily_companies_search import tavily_companies_search
from tools.companies.tavily_populate_culture import tavily_populate_culture

logger = get_logger(__name__)


async def generate_search_terms(query: str, app_ctx: Optional[AppContext] = None) -> list[str]:
    """Use AugmentedLLM to extract search terms from a natural language query."""
    if app_ctx:
        app_ctx.logger.info(f"generate_search_terms: processing query: {query}")

    prompt = f"""Convert this job search query into tags for RemoteOK's API. Return only lowercase single-word tags separated by commas, no explanations.
The first tag should be the most important/specific one (e.g., a programming language, job title, or skill).
Focus on: programming languages, frameworks, job titles, skills.

Query: "{query}"

Tags:"""

    try:
        async with Agent(
            name="search_term_generator",
            instruction="You extract search terms from job queries. Return only comma-separated lowercase tags.",
            server_names=[],
        ) as agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            result = await llm.generate_str(message=prompt)

        terms_text = result.strip()
        terms = [t.strip().lower() for t in terms_text.split(",") if t.strip()]
        if app_ctx:
            app_ctx.logger.info(f"generate_search_terms: extracted terms: {terms}")
        return terms

    except Exception as e:
        if app_ctx:
            app_ctx.logger.error(f"generate_search_terms: error: {e}")
        # Fallback: extract first word from query
        fallback = [query.split()[0].lower()] if query else ["developer"]
        logger.info(f"generate_search_terms: using fallback: {fallback}")
        return fallback


def get_func_description(func) -> str:
    """Extract first line of function docstring as description."""
    if func.__doc__:
        return func.__doc__.strip().split('\n')[0]
    return ""


class ToolNames(BaseModel):
    """Flat list of tool names selected by LLM"""
    tool_names: list[str]


class ToolCategories(BaseModel):
    """Tools organized by category"""
    jobs: list[str] = []
    companies: list[str] = []
    insights: list[str] = []


class UserContext(BaseModel):
    """User context for personalization and state tracking"""
    user_skills: list[str] = []


class Metadata(BaseModel):
    """Metadata about the execution"""
    execution_time_ms: int = 0
    fresh_matches: int = 0
    historical_matches: int = 0
    database_documents_created: int = 0
    database_documents_updated: int = 0
    insights_generated: bool = False
    tools_executed: list[str] = []


class InsightData(BaseModel):
    """Container for insight tool results and comprehensive insight"""
    tavily_role_research: dict | None = None
    tavily_market_trends: dict | None = None
    tavily_learning_resources: dict | None = None
    comprehensive: dict | None = None  # Final insight with message, skill_gaps, action_plan


class Response(BaseModel):
    """Response payload from career_agent"""
    query: str
    search_terms: list[str] = []
    jobs: list[Job] = []
    companies: list[Company] = []
    metadata: Metadata = Metadata()
    insights: InsightData | None = None
    error: str | None = None

    def to_json(self) -> str:
        """Serialize response to JSON string."""
        return self.model_dump_json(exclude_none=True)


class ToolRegistry(BaseModel):
    """Registry of available tools with their functions and descriptions"""
    jobs: dict[str, tuple] = {}
    companies: dict[str, tuple] = {}
    insights: dict[str, tuple] = {}

    class Config:
        arbitrary_types_allowed = True


# Tool registry instance
TOOL_REGISTRY = ToolRegistry(
    jobs={
        remoteok_search_jobs.__name__: (remoteok_search_jobs, get_func_description(remoteok_search_jobs)),
    },
    companies={
        tavily_companies_search.__name__: (tavily_companies_search, get_func_description(tavily_companies_search)),
        tavily_populate_emails.__name__: (tavily_populate_emails, get_func_description(tavily_populate_emails)),
        tavily_populate_culture.__name__: (tavily_populate_culture, get_func_description(tavily_populate_culture)),
    },
    insights={
        tavily_role_research.__name__: (tavily_role_research, get_func_description(tavily_role_research)),
        tavily_market_trends.__name__: (tavily_market_trends, get_func_description(tavily_market_trends)),
        tavily_learning_resources.__name__: (tavily_learning_resources, get_func_description(tavily_learning_resources)),
        synthesize_insights.__name__: (synthesize_insights, get_func_description(synthesize_insights)),
    },
)


def get_available_tools(allow_jobs: bool, allow_companies: bool, allow_insights: bool) -> ToolCategories:
    """Determine which tools are available based on user preferences."""
    jobs = []
    companies = []
    insights = []

    if allow_jobs:
        jobs = list(TOOL_REGISTRY.jobs.keys())

    if allow_companies:
        companies = list(TOOL_REGISTRY.companies.keys())

    if allow_insights:
        insights = list(TOOL_REGISTRY.insights.keys())

    return ToolCategories(jobs=jobs, companies=companies, insights=insights)


async def choose_tools(
    query: str,
    user_context: UserContext,
    allow_jobs: bool,
    allow_companies: bool,
    allow_insights: bool,
    app_ctx: Optional[AppContext] = None,
) -> ToolCategories:
    """Use AugmentedLLM to decide which tools to execute based on query and preferences."""
    if app_ctx:
        app_ctx.logger.info("choose_tools: analyzing query", data={"query": query})

    available_tools = get_available_tools(allow_jobs, allow_companies, allow_insights)

    # Format available tools for the prompt using TOOL_REGISTRY
    tools_description = []

    if available_tools.jobs:
        tools_description.append("JOBS CATEGORY TOOLS:")
        for tool_name in available_tools.jobs:
            _, description = TOOL_REGISTRY.jobs[tool_name]
            tools_description.append(f"  - {tool_name}: {description}")

    if available_tools.companies:
        tools_description.append("COMPANIES CATEGORY TOOLS:")
        for tool_name in available_tools.companies:
            _, description = TOOL_REGISTRY.companies[tool_name]
            tools_description.append(f"  - {tool_name}: {description}")

    if available_tools.insights:
        tools_description.append("INSIGHTS CATEGORY TOOLS:")
        for tool_name in available_tools.insights:
            _, description = TOOL_REGISTRY.insights[tool_name]
            tools_description.append(f"  - {tool_name}: {description}")

    tools_list = "\n".join(tools_description)

    prompt = f"""You are a career search agent deciding which tools to use to best address a user's query.

USER QUERY: {query}
USER SKILLS: {', '.join(user_context.user_skills) if user_context.user_skills else 'Not specified'}

AVAILABLE TOOLS:
{tools_list}

Select which tools should be executed. Return only the tool names as a list. Try to choose at no more than five tools if the user is being vague. Try to choose up to eight tools if the user is being specific."""

    try:
        async with Agent(
            name="tool_selector",
            instruction="You select the optimal tools to address user career queries.",
            server_names=[],
        ) as agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            selection = await llm.generate_structured(
                message=prompt,
                response_model=ToolNames
            )

        selected_names = selection.tool_names
        logger.info(f"choose_tools: LLM selected {len(selected_names)} tools: {selected_names}")

    except Exception as e:
        logger.error(f"choose_tools: error: {e}")
        # Fallback: use all available tools
        selected_names = available_tools.jobs + available_tools.companies + available_tools.insights

    # Rebuild flat list into ToolCategories
    selected_jobs = [name for name in selected_names if name in available_tools.jobs]
    selected_companies = [name for name in selected_names if name in available_tools.companies]
    selected_insights = [name for name in selected_names if name in available_tools.insights]

    result = ToolCategories(jobs=selected_jobs, companies=selected_companies, insights=selected_insights)
    logger.info(f"choose_tools: categorized into {len(result.jobs)} jobs, {len(result.companies)} companies, {len(result.insights)} insights")

    return result


async def execute_tools(
    tool_selection: ToolCategories,
    user_context: UserContext,
    max_results: int,
    user_id: str,
    allow_jobs: bool,
    allow_companies: bool,
    allow_insights: bool,
    response: "Response",
    app_ctx: Optional[AppContext] = None,
) -> "Response":
    """Execute the selected tools and return results.

    Returns:
        Tuple of (user_context, response)
    """
    tools_executed = []

    if allow_jobs and tool_selection.jobs:
        search_terms = await generate_search_terms(response.query, app_ctx)
        response.search_terms = search_terms

        # Execute jobs tools
        for tool_name in tool_selection.jobs:
            if tool_name == "remoteok_search_jobs":
                response = await remoteok_search_jobs(
                    query=response.query,
                    search_terms=search_terms,
                    max_results=max_results,
                    user_id=user_id,
                    response=response,
                    app_ctx=app_ctx
                )
                tools_executed.append(tool_name)
                if app_ctx:
                    app_ctx.logger.info(f"Executed {tool_name}", data={"jobs": len(response.jobs)})

    # Execute companies tools (after jobs)
    if allow_companies and tool_selection.companies:
        companies_search_ran = False

        for tool_name in tool_selection.companies:
            if tool_name == "tavily_companies_search":
                try:
                    response = await tavily_companies_search(
                        query=response.query,
                        max_results=max_results,
                        response=response,
                        app_ctx=app_ctx
                    )
                    tools_executed.append(tool_name)
                    companies_search_ran = True
                    if app_ctx:
                        app_ctx.logger.info(f"Executed {tool_name}", data={"companies": len(response.companies)})
                except Exception as e:
                    logger.error(f"execute_tools: error executing {tool_name}: {e}")

            elif tool_name == "tavily_populate_emails":
                try:
                    response = await tavily_populate_emails(
                        query=response.query,
                        jobs=response.jobs,
                        response=response,
                        app_ctx=app_ctx
                    )
                    tools_executed.append(tool_name)
                    if app_ctx:
                        app_ctx.logger.info(f"Executed {tool_name}", data={"companies": len(response.companies)})
                except Exception as e:
                    logger.error(f"execute_tools: error executing {tool_name}: {e}")

            elif tool_name == "tavily_populate_culture":
                try:
                    response = await tavily_populate_culture(
                        query=response.query,
                        response=response,
                        app_ctx=app_ctx
                    )
                    tools_executed.append(tool_name)
                    if app_ctx:
                        app_ctx.logger.info(f"Executed {tool_name}", data={"companies": len(response.companies)})
                except Exception as e:
                    logger.error(f"execute_tools: error executing {tool_name}: {e}")

        # If companies_search ran, automatically run culture tool if not already executed
        if companies_search_ran and "tavily_populate_culture" not in tools_executed:
            try:
                response = await tavily_populate_culture(
                    query=response.query,
                    response=response,
                    app_ctx=app_ctx
                )
                tools_executed.append("tavily_populate_culture")
                if app_ctx:
                    app_ctx.logger.info("Auto-executed tavily_populate_culture after companies_search")
            except Exception as e:
                logger.error(f"execute_tools: error auto-executing tavily_populate_culture: {e}")

    # Execute insights research tools
    if allow_insights:
        response.insights = InsightData()

        for tool_name in tool_selection.insights:
            if tool_name == "tavily_role_research":
                try:
                    result = await tavily_role_research(response.query)
                    response.insights.tavily_role_research = result
                    tools_executed.append(tool_name)
                    if app_ctx:
                        app_ctx.logger.info(f"Executed {tool_name}", data={"results": len(result.get('results', []))})
                except Exception as e:
                    logger.error(f"execute_tools: error executing {tool_name}: {e}")
                    response.insights.tavily_role_research = {"error": str(e)}

            elif tool_name == "tavily_market_trends":
                try:
                    result = await tavily_market_trends(response.query)
                    response.insights.tavily_market_trends = result
                    tools_executed.append(tool_name)
                    if app_ctx:
                        app_ctx.logger.info(f"Executed {tool_name}", data={"results": len(result.get('results', []))})
                except Exception as e:
                    logger.error(f"execute_tools: error executing {tool_name}: {e}")
                    response.insights.tavily_market_trends = {"error": str(e)}

            elif tool_name == "tavily_learning_resources":
                try:
                    result = await tavily_learning_resources(response.query)
                    response.insights.tavily_learning_resources = result
                    tools_executed.append(tool_name)
                    if app_ctx:
                        app_ctx.logger.info(f"Executed {tool_name}", data={"results": len(result.get('results', []))})
                except Exception as e:
                    logger.error(f"execute_tools: error executing {tool_name}: {e}")
                    response.insights.tavily_learning_resources = {"error": str(e)}

        await synthesize_insights(
            response=response,
            user_context=user_context,
            app_ctx=app_ctx
        )
        tools_executed.append(synthesize_insights.__name__)

    response.metadata.tools_executed = tools_executed
    return response


# Create the MCPApp
app = MCPApp(
    name="career_agent",
    description="Career search agent with jobs, communities, and personalized recommendations",
)


@app.tool
async def career_agent(
    query: str,
    allow_jobs: bool = True,
    allow_companies: bool = True,
    allow_insights: bool = True,
    max_results: int = 20,
    app_ctx: Optional[AppContext] = None
) -> dict:
    """
    Career search agent that finds jobs and provides personalized career insights.

    Args:
        query: Search query (e.g., "python developer remote")
        allow_companies: Find company emails from job listings
        allow_insights: Generate personalized skill gaps and action plan
        max_results: Maximum total results to return
        app_ctx: MCP context for logging and server access
    """
    start_time = time.time()

    if app_ctx:
        app_ctx.logger.info("Starting career_agent", data={"query": query})

    user_context = UserContext(user_skills=[])
    response = Response(query=query)

    # Determine which tools to use based on user preferences
    tool_selection = await choose_tools(query, user_context, allow_jobs, allow_companies, allow_insights, app_ctx)
    response = await execute_tools(
        tool_selection, user_context, max_results, "anonymous", allow_jobs, allow_companies, allow_insights, response, app_ctx
    )

    response.metadata.execution_time_ms = int((time.time() - start_time) * 1000)

    logger.info(f"career_agent: complete - {len(response.jobs)} jobs ({response.metadata.fresh_matches} fresh, {response.metadata.historical_matches} historical) in {response.metadata.execution_time_ms}ms")

    return json.loads(response.to_json())


async def main():
    import asyncio
    async with app.run() as agent_app:
        result = await career_agent(
            query="python developer",
            allow_insights=True,
        )
        print(f"Result: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# Deploy as remote SSE server:
# > uv run mcp-agent deploy "career_agent" --no-auth
