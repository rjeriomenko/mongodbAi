from __future__ import annotations

import asyncio
import time
from typing import Optional, Literal
from datetime import datetime

import os
from dotenv import load_dotenv
from tavily import TavilyClient
from motor.motor_asyncio import AsyncIOMotorClient
from google import genai

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

# RAG agent: Tavily + Embeddings + MongoDB Vector Search
@app.tool()
async def rag_agent(
    query: str,
    action: str = "query",
    app_ctx: Optional[Context] = None
) -> str:
    """
    RAG pipeline using Tavily for fresh data, embeddings, and MongoDB vector search.

    Args:
        query: The search query or question
        action: "ingest" to fetch and store data, "query" to search and answer
    """
    logger = app_ctx.app.logger
    logger.info(f"rag_agent called with action: {action}, query: {query}")

    try:
        mongo_client = get_mongo_client()
        db = mongo_client["mongodbai"]
        collection = db["test"]
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return f"Error: Failed to connect to MongoDB - {e}"

    try:
        if action == "ingest":
            # Fetch fresh data from Tavily
            tavily_client = get_tavily_client()
            response = tavily_client.search(query=query, max_results=5)

            ingested = 0
            for result in response["results"]:
                # Generate embedding for the content
                content = f"{result['title']}\n{result['content']}"

                # Store in MongoDB
                doc = {
                    "url": result["url"],
                    "title": result["title"],
                    "content": result["content"],
                    "score": result["score"],
                    "embedding": await get_embedding(content),
                    "query": query,
                    "created_at": datetime.utcnow()
                }
                await collection.insert_one(doc)
                ingested += 1

            return f"Ingested {ingested} documents from Tavily for query: '{query}'"

        elif action == "query":
            # Generate embedding for the query
            try:
                query_embedding = await get_embedding(query)
                logger.info(f"Generated embedding with {len(query_embedding)} dimensions")
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                return f"Error: Failed to generate embedding - {e}"

            # Vector search in MongoDB
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": 5
                    }
                },
                {
                    "$project": {
                        "title": 1,
                        "content": 1,
                        "url": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            try:
                results = await collection.aggregate(pipeline).to_list(length=5)
                logger.info(f"Vector search returned {len(results)} results")
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return f"Error: Vector search failed - {e}. Make sure 'vector_index' exists on the collection."

            if not results:
                return "No relevant documents found. Try ingesting data first with action='ingest'"

            # Build context from results
            context = "\n\n".join([
                f"**{r['title']}** (score: {r.get('score', 'N/A'):.3f})\n{r['content']}\nSource: {r['url']}"
                for r in results
            ])

            # Use LLM to generate response based on context
            agent = Agent(
                name="rag_responder",
                instruction=(
                    "You are a helpful assistant. Use the provided context to answer the user's question. "
                    "Be concise and cite sources when possible."
                ),
                server_names=[],
                context=app_ctx,
            )

            async with agent:
                llm = await agent.attach_llm(GoogleAugmentedLLM)
                prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"
                result = await llm.generate_str(message=prompt)
                return result

        else:
            return f"Unknown action: {action}. Use 'ingest' or 'query'"

    finally:
        mongo_client.close()

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
    jobs_fresh = []
    docs_created = 0

    # Build dict of selected tools based on flags (dict enforces uniqueness)
    selected_tools = {}
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

            # Handle search_jobs_tavily results
            if tool_name == "search_jobs_tavily" and not result.get("error"):
                jobs_fresh = result["results"]

                # Store results in MongoDB with embeddings
                mongo_client = get_mongo_client()
                db = mongo_client["mongodbai"]
                collection = db["career_results"]

                for job in jobs_fresh:
                    content = f"{job.get('title', '')}\n{job.get('content', '')}"
                    embedding = await get_embedding(content)

                    doc = {
                        "query": query,
                        "source": result["source"],
                        "url": job.get("url"),
                        "title": job.get("title"),
                        "content": job.get("content"),
                        "score": job.get("score"),
                        "embedding": embedding,
                        "user_id": user_id,
                        "created_at": datetime.utcnow()
                    }
                    await collection.insert_one(doc)
                    docs_created += 1

                mongo_client.close()

    execution_time = int((time.time() - start_time) * 1000)

    return {
        "query": query,
        "tools_executed": tools_executed,
        "jobs": {
            "fresh": jobs_fresh,
            "recommended": [],
            "count": len(jobs_fresh)
        },
        "communities": {"forums": []},
        "metadata": {
            "execution_time_ms": execution_time,
            "fresh_matches": len(jobs_fresh),
            "historical_matches": 0,
            "database_documents_created": docs_created
        }
    }

async def main():
    async with app.run() as agent_app:
        result = await career_agent(
            query="python developer remote",
            app_ctx=agent_app.context,
        )

        # Print test results
        print(f"\nQuery: {result['query']}")
        print(f"Tools: {result['tools_executed']}")
        print(f"Jobs found: {result['jobs']['count']}")
        print(f"Docs created: {result['metadata']['database_documents_created']}")
        print(f"Time: {result['metadata']['execution_time_ms']}ms\n")

        for i, job in enumerate(result["jobs"]["fresh"]):
            print(f"Job {i+1}: {job.get('title', 'N/A')}")
            print(f"  URL: {job.get('url', 'N/A')}")
            print(f"  Score: {job.get('score', 0)}\n")

if __name__ == "__main__":
    asyncio.run(main())

# When you're ready to deploy this MCPApp as a remote SSE server, run:
# > uv run mcp-agent deploy "career_agent" --no-auth
