from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

CUSTOMER_SUPPORT_SYSTEM = """You are Contoso Customer Support Agent v2.4.
You help enterprise customers with product issues, billing, and technical troubleshooting.
Always use the available tools before answering. Cite sources from the knowledge base.
Never invent policy numbers or account balances. Escalate if confidence is low.
Current date: {current_date}
Customer tier: {customer_tier}
"""

RAG_QA_PROMPT = PromptTemplate.from_template(
    """Answer the question using only the provided context. If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer with citations in the form [source:N]:"""
)

RESEARCH_BRIEF_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a senior research analyst. Produce concise, evidence-based briefs."
    ),
    HumanMessagePromptTemplate.from_template(
        "Query: {query}\n\nDocuments:\n{documents}\n\nProduce a structured brief with key findings, risks, and recommendations."
    ),
])

FEW_SHOT_CLASSIFIER = FewShotPromptTemplate(
    examples=[
        {"text": "My invoice shows a charge I don't recognize", "label": "billing"},
        {"text": "The API returns 503 intermittently", "label": "technical"},
        {"text": "How do I upgrade my plan?", "label": "sales"},
        {"text": "I need a copy of our MSA", "label": "legal"},
    ],
    example_prompt=PromptTemplate.from_template("Text: {text}\nLabel: {label}"),
    prefix="Classify the support ticket into one of: billing, technical, sales, legal, other.\n\n",
    suffix="Text: {input}\nLabel:",
    input_variables=["input"],
)

CHAIN_OF_THOUGHT_PROMPT = PromptTemplate.from_template(
    """Solve the following problem step by step. Show your reasoning.

Problem: {problem}

Reasoning:
"""
)

PROMPT_REGISTRY = {
    "customer_support_system": CUSTOMER_SUPPORT_SYSTEM,
    "rag_qa": RAG_QA_PROMPT,
    "research_brief": RESEARCH_BRIEF_PROMPT,
    "ticket_classifier": FEW_SHOT_CLASSIFIER,
    "cot": CHAIN_OF_THOUGHT_PROMPT,
}
