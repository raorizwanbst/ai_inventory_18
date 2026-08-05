from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_openai import ChatOpenAI
from typing import List
import redis
from src.config.settings import settings

class ConversationMemory:
    def __init__(self, store_uri: str = "redis://localhost:6379/0", window_size: int = 10):
        self.store_uri = store_uri
        self.window_size = window_size
        self.redis_client = redis.from_url(store_uri)
        self.summary_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)

    def _history(self, user_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            session_id=f"user:{user_id}",
            url=self.store_uri,
        )

    def load(self, user_id: str) -> list:
        history = self._history(user_id)
        return history.messages[-self.window_size :]

    def save(self, user_id: str, human: str, ai: str):
        history = self._history(user_id)
        history.add_user_message(human)
        history.add_ai_message(ai)

    def get_buffer_memory(self, user_id: str) -> ConversationBufferWindowMemory:
        return ConversationBufferWindowMemory(
            chat_memory=self._history(user_id),
            k=self.window_size,
            return_messages=True,
            memory_key="chat_history",
        )

    def get_summary_memory(self, user_id: str) -> ConversationSummaryMemory:
        return ConversationSummaryMemory(
            llm=self.summary_llm,
            chat_memory=self._history(user_id),
            return_messages=True,
            memory_key="chat_history",
        )

    def clear(self, user_id: str):
        history = self._history(user_id)
        history.clear()
