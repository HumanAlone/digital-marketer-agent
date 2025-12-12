"""Обертка LangChain агента для A2A протокола."""

import asyncio
from typing import Any, AsyncGenerator, Dict


class LangChainA2AWrapper:
    """Обертка LangChain AgentExecutor для A2A."""

    SUPPORTED_CONTENT_TYPES = ["text"]

    def __init__(self, langchain_agent):
        self.agent = langchain_agent

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Потоковое выполнение запроса через LangChain агента."""
        try:
            # Вызываем LangChain агента
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.agent.invoke({"input": query})
            )

            # Отправляем результат
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": result.get("output", ""),
                "is_error": False,
                "is_event": False,
            }

            # Финальный чанк
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": "",
                "is_error": False,
                "is_event": False,
            }

        except Exception as e:
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Ошибка: {str(e)}",
                "is_error": True,
                "is_event": False,
            }
