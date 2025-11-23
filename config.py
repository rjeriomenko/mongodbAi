from __future__ import annotations

from mcp_agent.config import get_settings
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


def get_google_api_key() -> str:
    """Get Google API key from settings."""
    settings = get_settings()
    logger.info(f"Settings google: {settings.google}")

    if not settings.google or not settings.google.api_key:
        logger.error("Missing Google API key in secrets")
        raise ValueError("Missing Google API key in secrets - check mcp_agent.secrets.yaml")

    logger.info("Google API key found")
    return settings.google.api_key


def get_tavily_api_key() -> str:
    """Get Tavily API key from settings."""
    settings = get_settings()

    # Tavily is a custom key, access via model_extra
    tavily = getattr(settings, 'tavily', None)
    if tavily is None:
        tavily = settings.model_extra.get('tavily', {}) if hasattr(settings, 'model_extra') else {}

    api_key = tavily.get('api_key') if isinstance(tavily, dict) else getattr(tavily, 'api_key', None)
    if not api_key:
        logger.error("Missing Tavily API key in secrets")
        raise ValueError("Missing Tavily API key in secrets - check mcp_agent.secrets.yaml")

    return api_key


def get_mongodb_connection_string() -> str:
    """Get MongoDB connection string from settings."""
    settings = get_settings()

    # Check MCP server env vars
    if settings.mcp and settings.mcp.servers:
        mongodb_server = settings.mcp.servers.get('mongodb')
        if mongodb_server and hasattr(mongodb_server, 'env'):
            conn_str = mongodb_server.env.get('MDB_MCP_CONNECTION_STRING')
            if conn_str:
                return conn_str

    logger.error("Missing MongoDB connection string in secrets")
    raise ValueError("Missing MongoDB connection string in secrets - check mcp_agent.secrets.yaml")
