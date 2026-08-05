from langchain_core.tools import tool
from pydantic import BaseModel, Field
import httpx

class CustomerLookupInput(BaseModel):
    customer_id: str = Field(description="Unique customer identifier, e.g. CUST-12345")
    include_history: bool = Field(default=False, description="Include past tickets")

@tool(args_schema=CustomerLookupInput)
def lookup_customer(customer_id: str, include_history: bool = False) -> dict:
    """Look up customer profile and optional ticket history from CRM."""
    resp = httpx.get(
        f"https://crm.internal.contoso.com/api/v2/customers/{customer_id}",
        params={"history": include_history},
        headers={"Authorization": "Bearer ${CRM_TOKEN}"},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()

@tool
def create_ticket(customer_id: str, subject: str, description: str, priority: str = "medium") -> dict:
    """Create a new support ticket in the CRM system."""
    payload = {
        "customer_id": customer_id,
        "subject": subject,
        "description": description,
        "priority": priority,
        "source": "ai_agent",
    }
    resp = httpx.post(
        "https://crm.internal.contoso.com/api/v2/tickets",
        json=payload,
        headers={"Authorization": "Bearer ${CRM_TOKEN}"},
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()

@tool
def escalate_issue(ticket_id: str, reason: str, target_team: str = "tier2") -> dict:
    """Escalate an existing ticket to a higher support tier."""
    resp = httpx.post(
        f"https://crm.internal.contoso.com/api/v2/tickets/{ticket_id}/escalate",
        json={"reason": reason, "target": target_team},
        headers={"Authorization": "Bearer ${CRM_TOKEN}"},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()
