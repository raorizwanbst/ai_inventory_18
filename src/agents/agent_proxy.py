import httpx
from typing import Any
from src.config.settings import settings

class RemoteA2AAgentProxy:
    """Local proxy that invokes a remote A2A agent endpoint.
    This component is classified as agent_proxy, not a full agent.
    """

    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint or settings.a2a_agent_endpoint
        self.client = httpx.AsyncClient(timeout=60.0)
        self.agent_card = None

    async def discover(self) -> dict:
        resp = await self.client.get(f"{self.endpoint}/.well-known/agent.json")
        resp.raise_for_status()
        self.agent_card = resp.json()
        return self.agent_card

    async def invoke(self, task: str, context: dict | None = None) -> dict[str, Any]:
        if not self.agent_card:
            await self.discover()
        payload = {
            "task": task,
            "context": context or {},
            "protocol_version": "a2a/1.0",
            "capabilities": self.agent_card.get("capabilities", []),
        }
        resp = await self.client.post(f"{self.endpoint}/invoke", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def stream(self, task: str):
        async with self.client.stream("POST", f"{self.endpoint}/stream", json={"task": task}) as resp:
            async for chunk in resp.aiter_text():
                yield chunk

remote_research_proxy = RemoteA2AAgentProxy()
remote_billing_proxy = RemoteA2AAgentProxy(
    endpoint="https://billing-agent.contoso.com/a2a/v1"
)
