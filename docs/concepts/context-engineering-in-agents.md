---
title: "Контекстная инженерия"
topic: "Concepts"
filename: "context-engineering-in-agents.md"
source: "https://docs.langchain.com/oss/python/langchain/context-engineering"
author: "Перевод GPT-4o"
date: "2025-12-05"
---

## Обзор

Построение надёжных агентов (или любого приложения на базе LLM) — это сложная задача. То, что может сработать в прототипе, зачастую терпит неудачу в реальных сценариях.
Когда агенты (agents) «ломаются», чаще всего это происходит из-за одной из двух причин:

1. Базовый LLM недостаточно эффективен
2. Неправильный **контекст** был передан в LLM

Чаще всего — именно второй вариант. _Context engineering_ — это подход, позволяющий предоставить LLM правильную информацию и инструменты в нужном формате, чтобы он смог выполнить задачу. Это самая важная задача для AI-инженеров. Абстракции агента в LangChain специально разработаны, чтобы облегчить context engineering.

Если вы только начинаете с context engineering — начните с [conceptual overview](https://docs.langchain.com/oss/python/langchain/context) — там объясняются разные типы контекста и когда их стоит использовать.

## Цикл агента

Типичный цикл агента состоит из двух основных шагов:

1. **Model call** — вызов LLM с prompt и доступными инструментами; возвращает либо ответ, либо запрос на выполнение инструмента
2. **Tool execution** — выполнение инструментов, запрошенных LLM; возвращаются результаты инструментов

Этот цикл повторяется до тех пор, пока LLM не решит, что работа завершена.

## Что вы можете контролировать

Чтобы построить надёжных агентов, вам нужно контролировать, что происходит:

| Контекст               | Что вы контролируете                                                                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Model Context**      | Что передаётся в модель (инструкции, история сообщений, инструменты, формат ответа) — **транзитный** (только для одного вызова)           |
| **Tool Context**       | К каким данным имеют доступ инструменты, что они могут прочитать и записать (state, store, runtime context) — **постоянный** (persistent) |
| **Life-cycle Context** | Что происходит между вызовами модели и инструментов (например, summarization, guardrails, логирование) — **постоянный**                   |

## Источники данных

Во время работы агент использует (читает/пишет) разные источники данных:

| Источник            | Также известен как                     | Сценарий использования                                                                                              |
| ------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Runtime Context** | static configuration                   | Конфигурации: user ID, API-ключи, соединения с базами, права доступа и пр.                                          |
| **State**           | short-term memory, conversation-scoped | История сообщений, загруженные файлы, статус аутентификации, результаты инструментов и др.                          |
| **Store**           | long-term memory, cross-conversation   | Долговременные данные между сессиями: предпочтения пользователя, предыдущие взаимодействия, сохранённые факты и пр. |

## Как это работает

Механизм context engineering в LangChain основан на посредниках (middleware). Middleware позволяет «вклиниваться» на любом этапе жизненного цикла агента и:

- обновлять контекст (state, store),
- либо менять шаг выполнения (например, пропускать tool-вызов, запускать повторный вызов модели с изменённым контекстом).

## Model Context

Контролируйте, что передаётся в каждый вызов модели — инструкции, доступные инструменты, модель, формат ответа. Эти решения сильно влияют на надёжность и стоимость.

#### System Prompt

Системный prompt задаёт поведение LLM. В зависимости от пользователя, контекста или стадии диалога могут понадобиться разные инструкции. Успешные агенты используют память, предпочтения и конфигурацию, чтобы дать LLM подходящий prompt.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    # request.messages — shortcut для request.state["messages"]
    message_count = len(request.messages)

    base = "You are a helpful assistant."

    if message_count > 10:
        base += "\nThis is a long conversation - be extra concise."

    return base

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[state_aware_prompt]
)
```

#### Messages

Сообщения формируют prompt, отправляемый LLM. Важно следить за тем, чтобы LLM получал релевантную информацию — не слишком много лишнего. Можно динамически внедрять контекст, например, информацию о загруженных пользователем файлах:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def inject_file_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Внедряет контекст о файлах, загруженных пользователем в этой сессии."""
    uploaded_files = request.state.get("uploaded_files", [])

    if uploaded_files:
        file_descriptions = []
        for file in uploaded_files:
            file_descriptions.append(
                f"- {file['name']} ({file['type']}): {file['summary']}"
            )

        file_context = f"""Files you have access to in this conversation:
{chr(10).join(file_descriptions)}

Reference these files when answering questions."""

        messages = [
            *request.messages,
            {"role": "user", "content": file_context},
        ]
        request = request.override(messages=messages)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[inject_file_context]
)
```

Важно понимать разницу: изменения, внесённые таким путём — **транзитные** (transient). Они влияют только на конкретный вызов модели, не меняя state. Для перманентных изменений — используйте life-cycle хуки.

## Инструменты

Инструменты (tools) позволяют модели взаимодействовать с внешними системами: базами данных, API, и др. Как вы определите и выберете инструменты — влияет на то, сможет ли агент выполнить задачу.

### Определение инструментов

Каждый tool должен иметь чёткое имя, описание, аргументы и описания аргументов. Это не просто метаданные — они помогают LLM правильно решать, когда и как использовать инструмент.

```python
from langchain.tools import tool

@tool(parse_docstring=True)
def search_orders(
    user_id: str,
    status: str,
    limit: int = 10
) -> str:
    """Search for user orders by status.

    Use this when the user asks about order history or wants to check
    order status. Always filter by the provided status.

    Args:
        user_id: Unique identifier for the user
        status: Order status: 'pending', 'shipped', or 'delivered'
        limit: Maximum number of results to return
    """
    # Implementation here
    pass
```

### Выбор инструментов

Не каждый инструмент уместен в любой ситуации. Слишком много tools — перегрузка контекста и рост ошибок; слишком мало — ограничивает возможности. Dynamic tool selection — подход, при котором набор инструментов изменяется в зависимости от состояния (например, авторизации, этапа диалога и пр.).

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Фильтрует инструменты на основе состояния State."""
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    # Только публичные инструменты до авторизации
    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        # Ограничение инструментов на раннем этапе диалога
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools]
)
```

## Model

Разные модели имеют разную производительность, стоимость и размер контекстного окна. В зависимости от задачи, можно динамически менять модель во время работы агента.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

# Инициализация моделей вне middleware
large_model = init_chat_model("claude-sonnet-4-5-20250929")
standard_model = init_chat_model("gpt-4o")
efficient_model = init_chat_model("gpt-4o-mini")

@wrap_model_call
def state_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Выбирает модель в зависимости от длины переписки."""
    message_count = len(request.messages)

    if message_count > 20:
        model = large_model
    elif message_count > 10:
        model = standard_model
    else:
        model = efficient_model

    request = request.override(model=model)
    return handler(request)

agent = create_agent(
    model="gpt-4o-mini",
    tools=[...],
    middleware=[state_based_model]
)
```

## Response Format

Free-form текст часто недостаточен, когда вам нужно, чтобы модель возвращала структурированные данные, которые могут использоваться дальше. Структурированный вывод (structured output) заставляет модель возвращать данные в строго определённом формате (например, схема с полями).

### Defining formats

Схемы формируются, например, с помощью `pydantic`. Поля, их типы и дескрипторы задают формат того, что модель должна вернуть.

```python
from pydantic import BaseModel, Field

class CustomerSupportTicket(BaseModel):
    """Структурированная информация о тикете поддержки, извлечённая из сообщения клиента."""
    category: str = Field(
        description="Issue category: 'billing', 'technical', 'account', or 'product'"
    )
    priority: str = Field(
        description="Urgency level: 'low', 'medium', 'high', or 'critical'"
    )
    summary: str = Field(
        description="One-sentence summary of the customer's issue"
    )
    customer_sentiment: str = Field(
        description="Customer's emotional tone: 'frustrated', 'neutral', or 'satisfied'"
    )
```

### Selecting formats dynamically

Можно выбирать формат ответа на основе состояния — например, простой формат на ранних этапах диалога и более детальный позже:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from pydantic import BaseModel, Field
from typing import Callable

class SimpleResponse(BaseModel):
    """Простой ответ на ранней стадии разговора."""
    answer: str = Field(description="A brief answer")

class DetailedResponse(BaseModel):
    """Детальный ответ для состоявшегося диалога."""
    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Explanation of reasoning")
    confidence: float = Field(description="Confidence score 0-1")

@wrap_model_call
def state_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    message_count = len(request.messages)

    if message_count < 3:
        request = request.override(response_format=SimpleResponse)
    else:
        request = request.override(response_format=DetailedResponse)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[state_based_output]
)
```

## Tool Context

Инструменты не просто выполняют действия — они могут читать и записывать контекст (state, store, runtime), что позволяет агента использовать результаты инструментов в будущем.

### Reads

Часто инструменты требуют больше, чем параметры, переданные LLM. Например, нужно знать user ID, API-ключи, состояние сессии — и всё это можно хранить в state, store или runtime context.

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

@tool
def check_authentication(runtime: ToolRuntime) -> str:
    """Проверяет, аутентифицирован ли пользователь."""
    current_state = runtime.state
    is_authenticated = current_state.get("authenticated", False)

    if is_authenticated:
        return "User is authenticated"
    else:
        return "User is not authenticated"

agent = create_agent(
    model="gpt-4o",
    tools=[check_authentication]
)
```

### Writes

Результаты инструментов могут быть не просто возвращены модели, но и записаны в память агента — state или store — чтобы быть доступными в будущих шагах.

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.types import Command

@tool
def authenticate_user(
    password: str,
    runtime: ToolRuntime
) -> Command:
    """Аутентифицирует пользователя и обновляет State."""
    if password == "correct":
        return Command(update={"authenticated": True})
    else:
        return Command(update={"authenticated": False})

agent = create_agent(
    model="gpt-4o",
    tools=[authenticate_user]
)
```

## Life-cycle Context

Это то, что происходит между шагами цикла агента — например, summarization, guardrails, логирование и прочее. С помощью middleware вы можете:

- обновлять context (state, store)
- менять flow выполнения (например, пропускать шаги, повторять вызовы)

### Example: Summarization

Один из самых распространённых шаблонов — автоматическое сокращение (резюмирование) истории переписки, когда она становится слишком длинной. В отличие от транзитного тримминга сообщений, summarization **постоянно обновляет** state — заменяя старые сообщения на краткий summary, сохраняющийся для будущих turns. В SummarizationMiddleware встроена такая логика:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger={"tokens": 4000},
            keep={"messages": 20},
        ),
    ],
)
```

Когда переписка превышает лимит токенов, SummarizationMiddleware:

1. Генерирует краткое резюме старых сообщений
2. Заменяет их на summary в state (постоянно)
3. Сохраняет недавние сообщения для контекста

Таким образом история переписки «конденсируется», но важные детали остаются.

Для полного списка встроенного middleware, доступных хуков и примеров кастомных — см. документацию Middleware.

## Best practices

- Начинайте с простого — используйте статические prompt’ы и инструменты, добавляйте динамику только при необходимости.
- Тестируйте шаг за шагом — добавляйте по одной фиче context engineering за раз.
- Мониторьте производительность — отслеживайте количество вызовов модели, использование токенов, задержки.
- Используйте встроенный middleware, когда это возможно — например, SummarizationMiddleware, LLMToolSelectorMiddleware и др.
- Документируйте вашу стратегию контекста — чётко прописывайте, какой контекст передаётся и зачем.
- Понять разницу между транзитным и постоянным контекстом: изменения в Model Context — транзитные, изменения через life-cycle context — постоянные.

## Related resources

- [Context conceptual overview](https://docs.langchain.com/oss/python/langchain/context) — для понимания типов контекста и когда их использовать
- [Middleware — complete middleware guide](https://docs.langchain.com/oss/python/langchain/middleware) — для обзора встроенных middleware и как писать свои.
- [Tools — tool creation and context access](https://docs.langchain.com/oss/python/langchain/tools) — создание инструментов и работа с context внутри них.
- [Memory — short-term and long-term memory patterns](https://docs.langchain.com/oss/python/langchain/memory) — паттерны памяти.
- [Agents — core agent concepts](https://docs.langchain.com/oss/python/langchain/agents) — базовые концепции агентов.

---

Source: [https://docs.langchain.com/oss/python/langchain/context-engineering](https://docs.langchain.com/oss/python/langchain/context-engineering)
