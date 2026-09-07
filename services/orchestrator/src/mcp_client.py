"""MCP client for the orchestrator.

Holds the connection and turns whatever the server advertises into LlamaIndex
tools. Descriptions come from the server via list_tools() and are never
hardcoded here - hardcoding them client-side would mean the server is unusable
by any other MCP client, which defeats the point of the protocol.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
import json
from llama_index.core.tools import FunctionTool
from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

from pydantic import create_model

_JSON_TO_PY = {
    "string": str, "integer": int, "number": float,
    "boolean": bool, "array": list, "object": dict,
}


def schema_to_model(name: str, schema: dict):
    """Build a Pydantic model from the tool's JSON Schema.

    Without this the model gets no parameter names and guesses, burning a
    failed call per tool before it lands on the right argument names.
    """
    props = (schema or {}).get("properties", {})
    required = set((schema or {}).get("required", []))
    fields = {}
    for field, spec in props.items():
        py_type = _JSON_TO_PY.get(spec.get("type", "string"), str)
        if spec.get("type") == "array":
            item = _JSON_TO_PY.get(spec.get("items", {}).get("type", "string"), str)
            py_type = list[item]
        fields[field] = (py_type, ... if field in required else None)
    return create_model(f"{name}Args", **fields)
@asynccontextmanager
async def mcp_connection(server_url: str) -> AsyncGenerator[ClientSession, None]:
    """Open a short-lived SSE session to the MCP data server.

    SSE rather than stdio because the orchestrator and the data server run in
    separate containers - a client cannot spawn the server as a subprocess
    across a container boundary.
    """
    logger.info("Connecting to MCP server at %s", server_url)
    try:
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    except Exception:
        logger.exception("Failed to establish MCP connection")
        raise


def extract_text(result: Any) -> str:
    """MCP results arrive as content blocks; flatten to text for the LLM."""
    if not getattr(result, "content", None):
        return "{}"
    joined = "\n".join(
        block.text for block in result.content if hasattr(block, "text")
    )
    return joined or "{}"


async def build_tools(
    session: ClientSession, audit: list[dict]
) -> list[FunctionTool]:
    """Turn every tool the server advertises into a LlamaIndex tool.

    Adding a tool on the server makes it available here with no change to
    this file. Each call is appended to `audit` so the API can return the
    full trail of what ran.
    """
    listing = await session.list_tools()
    tools: list[FunctionTool] = []

    for spec in listing.tools:
        if not spec.description:
            logger.warning(
                "Tool '%s' advertises no description - the model has nothing "
                "to decide on. Add a docstring on the server.",
                spec.name,
            )

        # Bind spec as a default argument: a closure over the loop variable
        # would leave every tool pointing at the last one.
                # A factory closure instead of a default argument: LlamaIndex builds a
        # Pydantic schema from the signature, and Pydantic rejects field names
        # with leading underscores.
        def make_caller(tool_name: str):
            async def call(**kwargs) -> str:
                # Small models sometimes echo the JSON Schema envelope instead of
                # filling it in, sending {"properties": {...}, "type": "object"}.
                # Unwrap rather than let the tool reject a call that has the right
                # values in the wrong shape.
                if "properties" in kwargs and isinstance(kwargs["properties"], dict):
                    kwargs = dict(kwargs["properties"])
                kwargs.pop("type", None)
                kwargs.pop("required", None)
                                # Model sometimes passes a field's schema as its value
                kwargs = {
                    k: (v.get("default") if isinstance(v, dict) and "type" in v else v)
                    for k, v in kwargs.items()
                }

                # 3. A list serialised as a JSON string
                for key, value in list(kwargs.items()):
                    if isinstance(value, str) and value.strip().startswith("["):
                        try:
                            kwargs[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass

                kwargs = {k: v for k, v in kwargs.items() if v is not None}

                logger.info("-> %s(%s)", tool_name, kwargs)
                result = await session.call_tool(tool_name, kwargs)
                text = extract_text(result)
                audit.append(
                    {"tool": tool_name, "arguments": kwargs, "result": text[:8000]}
                )
                return text

            return call
        tools.append(
            FunctionTool.from_defaults(
                async_fn=make_caller(spec.name),
                name=spec.name,
                description=spec.description or "",
                fn_schema=schema_to_model(spec.name, spec.inputSchema),
            )
        )
    logger.info(
        "Discovered %d tools: %s", len(tools), [t.metadata.name for t in tools]
    )
    return tools
