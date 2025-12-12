"""Запуск A2A сервера с LangChain агентом."""

import logging
import os

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from dotenv import load_dotenv

from a2a_wrapper import LangChainA2AWrapper
from agent_task_manager import LangChainAgentExecutor
from langchain_agent import create_langchain_agent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        mcp_url = os.getenv("MCP_URL", "http://localhost:8000")
        langchain_agent = create_langchain_agent(mcp_url)
        agent_wrapper = LangChainA2AWrapper(langchain_agent)
        agent_executor = LangChainAgentExecutor(agent_wrapper)
        capabilities = AgentCapabilities(streaming=True)

        agent_card = AgentCard(
            name=os.getenv("AGENT_NAME", "Yandex Direct LangChain Agent"),
            description="LangChain агент с MCP инструментами для управления рекламой",
            url=f"http://localhost:{os.getenv('PORT', '10000')}",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=capabilities,
            skills=[],
        )

        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        import uvicorn

        port = int(os.getenv("PORT", "10000"))
        logger.info(f"🚀 A2A + LangChain агент запущен на порту {port}")
        logger.info(f"🔗 MCP: {mcp_url}")
        logger.info(f"🤖 Модель: {os.getenv('LLM_MODEL')}")

        uvicorn.run(server.build(), host="0.0.0.0", port=port)

    except Exception as e:
        logger.error(f"Ошибка запуска: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
