import asyncio
from typing import Any

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession


async def _call_tool_async(url: str, tool: str, args: dict) -> Any:
    # streamablehttp_client is an async context manager
    async with streamablehttp_client(url) as transport:
        # transport may yield a tuple (read, write, meta) or an object with read/write
        if isinstance(transport, tuple) or isinstance(transport, list):
            read, write, *_ = transport
        else:
            read, write = transport.read, transport.write

        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(tool, arguments=args)
        texts = []
        for c in res.content:
            try:
                texts.append(c.text)
            except Exception:
                texts.append(str(c))
        return texts


def call_mcp_sync(url: str, tool: str, args: dict, env: dict | None = None) -> Any:
    # Use asyncio.run to run the async helper from sync code
    return asyncio.run(_call_tool_async(url, tool, args))
