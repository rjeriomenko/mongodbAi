"""
Welcome to mcp-agent! We believe MCP is all you need to build and deploy agents.
This is a canonical getting-started example that covers everything you need to know to get started.

We will cover:
  - Hello world agent: Setting up a basic Agent that uses the fetch and filesystem MCP servers to do cool stuff.
  - @app.tool and @app.async_tool decorators to expose your agents as long-running tools on an MCP server.
  - Advanced MCP features: Notifications, sampling, and elicitation

You can run this example locally using "uv run main.py", and also deploy it as an MCP server using "mcp-agent deploy".

Let's get started!
"""

from __future__ import annotations

import asyncio
from typing import Optional
from pathlib import Path
from datetime import datetime

import yaml
from tavily import TavilyClient
from motor.motor_asyncio import AsyncIOMotorClient
from google import genai

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_agent.core.context import Context as AppContext
from mcp_agent.workflows.factory import create_agent

# We are using the Google Gemini augmented LLM
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM


# Load secrets
def load_secrets():
    secrets_path = Path(__file__).parent / "mcp_agent.secrets.yaml"
    with open(secrets_path) as f:
        return yaml.safe_load(f)


# Get embedding from Google
async def get_embedding(text: str) -> list[float]:
    secrets = load_secrets()
    client = genai.Client(api_key=secrets["google"]["api_key"])
    result = await asyncio.to_thread(
        client.models.embed_content,
        model="text-embedding-004",
        contents=text
    )
    return result.embeddings[0].values


# Get MongoDB client
def get_mongo_client():
    secrets = load_secrets()
    connection_string = secrets["mcp"]["servers"]["mongodb"]["env"]["MDB_MCP_CONNECTION_STRING"]
    return AsyncIOMotorClient(connection_string)

# Create the MCPApp, the root of mcp-agent.
app = MCPApp(
    name="hello_world",
    description="Hello world mcp-agent application",
    # settings= <specify programmatically if needed; by default, configuration is read from mcp_agent.config.yaml/mcp_agent.secrets.yaml>
)


# MongoDB agent: LLM-powered agent that uses MongoDB MCP server
@app.tool()
async def mongo_agent(request: str, app_ctx: Optional[AppContext] = None) -> str:
    """
    Run an LLM-powered agent that can query and modify MongoDB using natural language.

    Args:
        request: Natural language request like "insert a document with hello: hello world"
    """
    logger = app_ctx.app.logger
    logger.info(f"mongo_agent called with request: {request}")

    agent = Agent(
        name="mongo",
        instruction=(
            "You are a MongoDB assistant. Use the available MongoDB tools to help the user. "
            "When inserting documents, use the test collection in the mongodbai database unless specified otherwise. "
            "Always confirm what action you took and show the results."
        ),
        server_names=["mongodb"],
        context=app_ctx,
    )

    async with agent:
        llm = await agent.attach_llm(GoogleAugmentedLLM)
        result = await llm.generate_str(message=request)
        return result


# RAG agent: Tavily + Embeddings + MongoDB Vector Search
@app.tool()
async def rag_agent(
    query: str,
    action: str = "query",
    app_ctx: Optional[AppContext] = None
) -> str:
    """
    RAG pipeline using Tavily for fresh data, embeddings, and MongoDB vector search.

    Args:
        query: The search query or question
        action: "ingest" to fetch and store data, "query" to search and answer
    """
    logger = app_ctx.app.logger
    logger.info(f"rag_agent called with action: {action}, query: {query}")

    secrets = load_secrets()
    mongo_client = get_mongo_client()
    db = mongo_client["mongodbai"]
    collection = db["test"]

    try:
        if action == "ingest":
            # Fetch fresh data from Tavily
            tavily_client = TavilyClient(secrets["tavily"]["api_key"])
            response = tavily_client.search(query=query, max_results=5)

            ingested = 0
            for result in response["results"]:
                # Generate embedding for the content
                content = f"{result['title']}\n{result['content']}"
                embedding = await get_embedding(content)

                # Store in MongoDB
                doc = {
                    "url": result["url"],
                    "title": result["title"],
                    "content": result["content"],
                    "score": result["score"],
                    "embedding": embedding,
                    "query": query,
                    "created_at": datetime.utcnow()
                }
                await collection.insert_one(doc)
                ingested += 1

            return f"Ingested {ingested} documents from Tavily for query: '{query}'"

        elif action == "query":
            # Generate embedding for the query
            query_embedding = await get_embedding(query)

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

            results = await collection.aggregate(pipeline).to_list(length=5)

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


# Run a configured agent by name (defined in mcp_agent.config.yaml)
@app.async_tool(name="run_agent_async")
async def run_agent(
    agent_name: str = "web_helper",
    prompt: str = "Please summarize the first paragraph of https://modelcontextprotocol.io/docs/getting-started/intro",
    app_ctx: Optional[AppContext] = None,
) -> str:
    """
    Load an agent defined in mcp_agent.config.yaml by name and run it.

    Notes:
    - @app.async_tool:
      - async version of @app.tool -- returns a workflow ID back (can be used with workflows-get_status tool)
      - runs the function as a long-running workflow tool when deployed as an MCP server
      - no-op when running this locally as a script
    """

    logger = app_ctx.app.logger

    agent_definitions = (
        app.config.agents.definitions
        if app is not None
        and app.config is not None
        and app.config.agents is not None
        and app.config.agents.definitions is not None
        else []
    )

    agent_spec: AgentSpec | None = None
    for agent_def in agent_definitions:
        if agent_def.name == agent_name:
            agent_spec = agent_def
            break

    if agent_spec is None:
        logger.error("Agent not found", data={"name": agent_name})
        return f"agent '{agent_name}' not found"

    logger.info(
        "Agent found in spec",
        data={"name": agent_name, "instruction": agent_spec.instruction},
    )

    agent = create_agent(agent_spec, context=app_ctx)

    async with agent:
        llm = await agent.attach_llm(GoogleAugmentedLLM)
        return await llm.generate_str(message=prompt)


async def main():
    async with app.run() as agent_app:
        # Test RAG agent - first ingest some data
        # print("Ingesting data from Tavily...")
        # ingest_result = await rag_agent(
        #     query="latest fashion trends 2024",
        #     action="ingest",
        #     app_ctx=agent_app.context,
        # )
        # print(ingest_result)

        # Then query with vector search
        print("\nQuerying with vector search...")
        query_result = await rag_agent(
            query="What are the main fashion trends?",
            action="query",
            app_ctx=agent_app.context,
        )
        print("RAG Query Result:")
        print(query_result)

        # Uncomment to run as MCP server:
        from mcp_agent.server.app_server import create_mcp_server_for_app
        mcp_server = create_mcp_server_for_app(agent_app)
        await mcp_server.run_sse_async()


if __name__ == "__main__":
    asyncio.run(main())

# When you're ready to deploy this MCPApp as a remote SSE server, run:
# > uv run mcp-agent deploy "hello_world" --no-auth
#
# Congrats! You made it to the end of the getting-started example!
# There is a lot more that mcp-agent can do, and we hope you'll explore the rest of the documentation.
# Check out other examples in the mcp-agent repo:
# https://github.com/lastmile-ai/mcp-agent/tree/main/examples
# and read the docs (or ask an mcp-agent to do it for you):
# https://docs.mcp-agent.com/