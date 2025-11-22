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

import yaml
from motor.motor_asyncio import AsyncIOMotorClient

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_agent.core.context import Context as AppContext
from mcp_agent.workflows.factory import create_agent

# We are using the Google Gemini augmented LLM
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM

# Create the MCPApp, the root of mcp-agent.
app = MCPApp(
    name="hello_world",
    description="Hello world mcp-agent application",
    # settings= <specify programmatically if needed; by default, configuration is read from mcp_agent.config.yaml/mcp_agent.secrets.yaml>
)


# Helper to load MongoDB connection string from secrets
def get_mongo_connection_string() -> str:
    secrets_path = Path(__file__).parent / "mcp_agent.secrets.yaml"
    with open(secrets_path) as f:
        secrets = yaml.safe_load(f)
    return secrets["mongodb"]["connection_string"]


# MongoDB agent: store and retrieve documents
@app.tool()
async def mongo_agent(action: str = "test", app_ctx: Optional[AppContext] = None) -> str:
    """
    Store and retrieve documents from MongoDB.

    Args:
        action: "store" to save hello world, "retrieve" to get it back, "test" to do both
    """
    logger = app_ctx.app.logger
    logger.info(f"mongo_agent called with action: {action}")

    # Connect to MongoDB
    connection_string = get_mongo_connection_string()
    client = AsyncIOMotorClient(connection_string)
    db = client["mongodbai"]
    collection = db["test"]

    result = ""

    try:
        if action in ["store", "test"]:
            # Store document
            doc = {"hello": "hello world"}
            insert_result = await collection.insert_one(doc)
            result += f"Stored document with id: {insert_result.inserted_id}\n"
            logger.info(f"Stored document: {insert_result.inserted_id}")

        if action in ["retrieve", "test"]:
            # Retrieve document
            found = await collection.find_one({"hello": "hello world"})
            if found:
                result += f"Retrieved: {found}"
                logger.info(f"Retrieved document: {found}")
            else:
                result += "No document found"
                logger.info("No document found")

        if action == "remove":
            # Delete doc
            found = await collection.find_one({"hello": "hello world"})
            if found:
                await collection.delete_one({"hello": "hello world"})

    finally:
        client.close()

    return result


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
        # Test MongoDB agent
        mongo_result = await mongo_agent(
            action="test",
            app_ctx=agent_app.context,
        )
        print("MongoDB test result:")
        print(mongo_result)

        # Create the MCP server that exposes both workflows and agent configurations,
        # optionally using custom FastMCP settings
        from mcp_agent.server.app_server import create_mcp_server_for_app
        mcp_server = create_mcp_server_for_app(agent_app)

        # Run the server
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
#
# Happy mcp-agenting!
