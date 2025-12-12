"""LangChain агент с MCP инструментами для A2A."""

import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class MCPClient:
    """Клиент для вызова MCP инструментов."""

    def __init__(self, mcp_url: str):
        self.mcp_url = mcp_url

    def call_tool(self, tool_name: str, **kwargs) -> str:
        """Синхронный вызов MCP инструмента."""
        import requests

        resp = requests.post(
            f"{self.mcp_url}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": kwargs},
                "id": 1,
            },
            headers={"Accept": "application/json, text/event-stream"},
        )

        if resp.status_code == 200:
            for line in resp.text.split("\n"):
                if line.startswith("data:"):
                    data = json.loads(line[5:])
                    if "result" in data:
                        result = data["result"]
                        if "content" in result and len(result["content"]) > 0:
                            text = result["content"][0].get("text", str(result))
                            if isinstance(text, bytes):
                                text = text.decode("utf-8", errors="ignore")
                            return text
                        text = str(result)
                        if isinstance(text, bytes):
                            text = text.decode("utf-8", errors="ignore")
                        return text
        return f"Ошибка MCP: {resp.status_code}"


# Схемы для каждого инструмента
class GetCampaignPerformanceSchema(BaseModel):
    campaign_id: str = Field(description="ID кампании")
    days: Optional[int] = Field(default=7, description="Количество дней для анализа")


class AnalyzeCampaignHealthSchema(BaseModel):
    campaign_id: str = Field(description="ID кампании")
    target_cpa: float = Field(description="Целевой CPA в рублях")
    daily_budget_limit: float = Field(description="Лимит дневного бюджета в рублях")


class GenerateDailyReportSchema(BaseModel):
    campaign_ids: List[str] = Field(description="Список ID кампаний для отчёта")


class CalculateScenariosSchema(BaseModel):
    campaign_id: str = Field(description="ID кампании")
    target_conversions: int = Field(description="Целевое количество конверсий")


class CalculateCPASchema(BaseModel):
    cost: float = Field(description="Общая стоимость в рублях")
    conversions: int = Field(description="Количество конверсий")


def create_mcp_tool(mcp_url: str, tool_name: str, description: str) -> StructuredTool:
    """Создает LangChain StructuredTool из MCP инструмента."""
    client = MCPClient(mcp_url)

    if tool_name == "get_campaign_performance":
        schema = GetCampaignPerformanceSchema

        def func(campaign_id: str, days: int = 7) -> str:
            result = client.call_tool(tool_name, campaign_id=campaign_id, days=days)
            # Преобразуем JSON в читаемый текст
            try:
                import json

                data = json.loads(result)
                if isinstance(data, dict) and "metrics" in data:
                    metrics = data["metrics"]
                    return (
                        f"Данные кампании {campaign_id}:\n"
                        f"Расход: {metrics.get('total_cost', 'N/A')} руб.\n"
                        f"Конверсии: {metrics.get('total_conversions', 'N/A')}\n"
                        f"CPA: {metrics.get('avg_cpa', 'N/A')} руб.\n"
                        f"CTR: {metrics.get('avg_ctr', 'N/A')}%\n"
                        f"За период: {data.get('period_days', 'N/A')} дней"
                    )
            except:
                pass
            return str(result)

        return StructuredTool.from_function(
            name=tool_name, description=description, func=func, args_schema=schema
        )

    elif tool_name == "analyze_campaign_health":
        schema = AnalyzeCampaignHealthSchema

        def func(campaign_id: str, target_cpa: float, daily_budget_limit: float) -> str:
            result = client.call_tool(
                tool_name,
                campaign_id=campaign_id,
                target_cpa=target_cpa,
                daily_budget_limit=daily_budget_limit,
            )
            try:
                import json

                data = json.loads(result)
                if isinstance(data, dict):
                    text = f"Анализ кампании {campaign_id}:\n"
                    text += f"Оценка здоровья: {data.get('health_score', 'N/A')}/100\n"
                    text += f"Статус: {data.get('status', 'N/A')}\n"

                    if "alerts" in data and data["alerts"]:
                        text += "Предупреждения:\n"
                        for alert in data["alerts"]:
                            text += f"- {alert}\n"

                    if "recommendations" in data and data["recommendations"]:
                        text += "Рекомендации:\n"
                        for rec in data["recommendations"]:
                            text += f"- {rec}\n"

                    return text
            except:
                pass
            return str(result)

        return StructuredTool.from_function(
            name=tool_name, description=description, func=func, args_schema=schema
        )

    elif tool_name == "generate_daily_report":
        schema = GenerateDailyReportSchema

        def func(campaign_ids: List[str]) -> str:
            result = client.call_tool(tool_name, campaign_ids=campaign_ids)
            return str(result)

        return StructuredTool.from_function(
            name=tool_name, description=description, func=func, args_schema=schema
        )

    elif tool_name == "calculate_scenarios":
        schema = CalculateScenariosSchema

        def func(campaign_id: str, target_conversions: int) -> str:
            result = client.call_tool(
                tool_name,
                campaign_id=campaign_id,
                target_conversions=target_conversions,
            )
            return str(result)

        return StructuredTool.from_function(
            name=tool_name, description=description, func=func, args_schema=schema
        )

    elif tool_name == "calculate_cpa":
        schema = CalculateCPASchema

        def func(cost: float, conversions: int) -> str:
            cpa = cost / conversions if conversions > 0 else 0
            return (
                f"CPA: {cpa:.2f} руб. (Расход: {cost} руб. / Конверсии: {conversions})"
            )

        return StructuredTool.from_function(
            name=tool_name, description=description, func=func, args_schema=schema
        )

    elif tool_name == "test_connection":

        def func() -> str:
            result = client.call_tool(tool_name)
            return str(result)

        return StructuredTool.from_function(
            name=tool_name, description=description, func=func
        )

    else:
        # Общий случай для неизвестных инструментов
        def func(**kwargs) -> str:
            return client.call_tool(tool_name, **kwargs)

        return StructuredTool.from_function(
            name=tool_name, description=description, func=func
        )


def create_langchain_agent(mcp_url: str = None):
    """Создает LangChain агента с MCP инструментами."""

    if mcp_url is None:
        mcp_url = os.getenv("MCP_URL", "http://localhost:8000")

    tools = [
        create_mcp_tool(
            mcp_url,
            "get_campaign_performance",
            "Получает и анализирует отчёт по кампании за N дней. Принимает campaign_id и days (опционально, по умолчанию 7).",
        ),
        create_mcp_tool(
            mcp_url,
            "analyze_campaign_health",
            "Анализирует здоровье кампании по правилам. Принимает campaign_id, target_cpa, daily_budget_limit.",
        ),
        create_mcp_tool(
            mcp_url,
            "generate_daily_report",
            "Генерирует сводный отчёт по нескольким кампаниям. Принимает список campaign_ids.",
        ),
        create_mcp_tool(
            mcp_url,
            "calculate_scenarios",
            "Рассчитывает сценарии для достижения целевых конверсий. Принимает campaign_id, target_conversions.",
        ),
        create_mcp_tool(
            mcp_url,
            "calculate_cpa",
            "Рассчитывает фактический CPA (Cost / Conversions). Принимает cost, conversions.",
        ),
        create_mcp_tool(
            mcp_url,
            "test_connection",
            "Проверяет подключение к MCP серверу.",
        ),
    ]
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        openai_api_key=os.getenv("LLM_API_KEY"),
        openai_api_base=os.getenv(
            "LLM_API_BASE", "https://foundation-models.api.cloud.ru/v1"
        ),
        temperature=0.7,
    )

    system_prompt = """Ты ассистент для управления рекламными кампаниями Яндекс.Директ.
    
    Ты можешь:
    1. Анализировать эффективность кампаний на основе данных из Яндекс.Директ
    2. Оценивать здоровье кампаний по CPA и бюджету
    3. Давать рекомендации по оптимизации
    4. Рассчитывать сценарии для достижения целевых конверсий
    5. Генерировать сводные отчёты
    
    ВАЖНО: У тебя нет доступа к управлению кампаниями (остановка, запуск, изменение ставок).
    Ты можешь только анализировать данные и давать рекомендации.
    
    Доступные инструменты:
    1. get_campaign_performance - получить данные по кампании
    2. analyze_campaign_health - проанализировать здоровье кампании (требует target_cpa и daily_budget_limit)
    3. generate_daily_report - создать отчёт по нескольким кампаниям
    4. calculate_scenarios - рассчитать сценарии для целевых конверсий
    5. calculate_cpa - рассчитать CPA по cost и conversions
    
    Порядок работы:
    1. Сначала запроси данные через get_campaign_performance
    2. Проанализируй через analyze_campaign_health (если известны target_cpa и budget_limit)
    3. Если нужно спланировать бюджет — используй calculate_scenarios
    4. Для отчёта по нескольким кампаниям — generate_daily_report
    
    Отвечай чётко, с цифрами и конкретными рекомендациями."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

    return agent_executor
