from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import os

from src.tools.crm_tools import lookup_customer, create_ticket, escalate_issue
from src.tools.search_tools import semantic_search, web_search
from src.rag.retriever import HybridRetriever
from src.guardrails.content_filter import InputGuardrail, OutputGuardrail
from src.memory.conversation_store import ConversationMemory
from src.observability.tracing import get_langfuse_handler
from src.config.settings import settings

llm = ChatOpenAI(
    model=settings.model_name,
    temperature=0.1,
    api_key=settings.openai_api_key,
    model_kwargs={"response_format": {"type": "json_object"}},
)

fallback_llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=settings.anthropic_api_key,
    temperature=0.0,
)

SYSTEM_PROMPT = """You are Contoso Customer Support Agent v2.4.
You help enterprise customers with product issues, billing, and technical troubleshooting.
Always use the available tools before answering. Cite sources from the knowledge base.
Never invent policy numbers or account balances. Escalate if confidence is low.
Current date context is injected at runtime.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [lookup_customer, create_ticket, escalate_issue, semantic_search, web_search]

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

memory = MemorySaver()
react_agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    state_modifier=SYSTEM_PROMPT,
)

input_guard = InputGuardrail(config_path=settings.guardrail_config_path)
output_guard = OutputGuardrail(config_path=settings.guardrail_config_path)
conversation_memory = ConversationMemory(store_uri="redis://localhost:6379/0")
langfuse_handler = get_langfuse_handler()

class CustomerSupportAgent:
    def __init__(self):
        self.executor = agent_executor
        self.react = react_agent
        self.retriever = HybridRetriever(
            embedding_model=settings.embedding_model,
            vector_store_uri=settings.vector_store_uri,
        )
        self.memory = conversation_memory

    async def handle(self, user_id: str, message: str) -> dict:
        sanitized = input_guard.validate(message)
        history = self.memory.load(user_id)
        result = await self.executor.ainvoke(
            {"input": sanitized, "chat_history": history},
            config={"callbacks": [langfuse_handler]},
        )
        final = output_guard.validate(result["output"])
        self.memory.save(user_id, message, final)
        return {"response": final, "sources": result.get("intermediate_steps", [])}
