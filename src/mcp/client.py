from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import httpx
from typing import Any
from src.config.settings import settings

class MCPClient:
    def __init__(self, server_url: str | None = None):
        self.server_url = server_url or settings.mcp_server_url
        self.session: ClientSession | None = None

    async def connect_sse(self):
        async with sse_client(self.server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.session = session
                return session

    async def list_tools(self) -> list[dict]:
        if not self.session:
            await self.connect_sse()
        result = await self.session.list_tools()
        return [{"name": t.name, "description": t.description} for t in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        if not self.session:
            await self.connect_sse()
        result = await self.session.call_tool(name, arguments)
        return {"content": [c.text for c in result.content if hasattr(c, "text")]}

    def call_tool_sync(self, name: str, arguments: dict[str, Any]) -> dict:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.call_tool(name, arguments))

class MCPGateway:
    """Multiplexes multiple MCP servers behind a single endpoint."""

    def __init__(self):
        self.servers = {
            "enterprise": MCPClient("http://localhost:8080/mcp"),
            "weather": MCPClient("http://weather-mcp.internal:8090/mcp"),
            "finance": MCPClient("http://finance-mcp.internal:8091/mcp"),
        }

    async def route(self, server_name: str, tool: str, args: dict) -> dict:
        client = self.servers.get(server_name)
        if not client:
            raise ValueError(f"Unknown MCP server: {server_name}")
        return await client.call_tool(tool, args)

    async def list_all_tools(self) -> dict[str, list]:
        result = {}
        for name, client in self.servers.items():
            result[name] = await client.list_tools()
        return result
