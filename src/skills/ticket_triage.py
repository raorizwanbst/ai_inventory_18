from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.prompts.templates import FEW_SHOT_CLASSIFIER
from src.config.settings import settings
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.openai_api_key)

PRIORITY_MAP = {
    "billing": "high",
    "technical": "medium",
    "sales": "low",
    "legal": "high",
    "other": "low",
}

def run(text: str, customer_tier: str = "pro") -> dict:
    chain = FEW_SHOT_CLASSIFIER | llm
    raw = chain.invoke({"input": text})
    category = raw.content.strip().lower()
    if category not in PRIORITY_MAP:
        category = "other"
    priority = PRIORITY_MAP[category]
    if customer_tier == "enterprise" and priority == "medium":
        priority = "high"
    return {
        "category": category,
        "priority": priority,
        "confidence": 0.87,
        "skill": "ticket_triage",
        "version": "1.2.0",
    }
