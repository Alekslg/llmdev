---
title: "Agents"
topic: "Core components"
filename: "agents.md"
source: "https://docs.langchain.com/oss/python/langchain-agents"
author: "Перевод GPT-4"
date: "2025-12-08"
---

# Agents

Agents объединяют языковые модели с [tools](../core_components/tools.md) для создания систем, которые могут рассуждать о задачах, решать, какие инструменты использовать, и итеративно работать над решением. Функция `create_agent` предоставляет промышленную реализацию агента.

LLM-агент запускает инструменты в цикле, чтобы достичь цели. Агент работает, пока не будет выполнено условие остановки — то есть пока модель не выдаст окончательный вывод или не будет достигнут предел итераций.

`create_agent` строит на базе LangGraph рантайм агента, основанный на графе. Граф состоит из узлов (шагов) и рёбер (соединений), которые определяют, как ваш агент обрабатывает информацию. Агент проходит по этому графу, выполняя узлы, такие как узел модели (использующий LLM), узел инструментов (tools), или посредников (middleware).

---

## Core components

### Model

Модель — это движок рассуждения вашего агента. Она может быть указана разными способами, поддерживая как статический, так и динамический выбор модели.

#### Static model

Статические модели настраиваются один раз при создании агента и остаются неизменными на протяжении исполнения. Это самый распространённый и простой способ. Чтобы инициализировать статическую модель по строковому идентификатору:

```python
from langchain.agents import create_agent

agent = create_agent(
    "gpt-5",
    tools=tools
)
```

Строковые идентификаторы моделей поддерживают автоматическое определение — например, `"gpt-5"` будет интерпретирована как `"openai:gpt-5"`.

Для более тонкой настройки можно напрямую инициализировать экземпляр модели из пакета провайдера. В этом примере используется `ChatOpenAI`. См. раздел Chat models для других доступных классов.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
    # ... (другие параметры)
)
agent = create_agent(model, tools=tools)
```

Экземпляры модели дают полный контроль над конфигурацией. Используйте их, когда нужно задать специфичные параметры — `temperature`, `max_tokens`, `timeouts`, `base_url` и др.

#### Dynamic model

Динамические модели выбираются во время выполнения, исходя из текущего состояния и контекста. Это позволяет реализовать сложную логику маршрутизации и оптимизацию затрат. Чтобы использовать динамическую модель, создайте middleware с декоратором `@wrap_model_call`, который модифицирует модель в запросе:

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Выбирает модель на основе сложности разговора."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # использовать более мощную модель для длинных разговоров
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)

agent = create_agent(
    model=basic_model,  # модель по умолчанию
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

Важно: модели, к которым уже применён `bind_tools`, не поддерживаются при использовании структурированного вывода. Если вам нужен динамический выбор модели с structured output — убедитесь, что переданные в middleware модели **не** предварительно привязаны.

Для деталей конфигурации моделей смотрите раздел [Models](#). Для паттернов динамического выбора — раздел про middleware и динамическую модель.

---

### Tools

Инструменты дают агентам возможность совершать действия. Агент выходит за рамки простого связывания модели и инструмента — он может:

- Вызывать несколько инструментов последовательно (по одному prompt)
- Делать параллельные вызовы инструментов, когда это уместно
- Динамически выбирать инструменты на основе предыдущих результатов
- Обрабатывать ошибки инструментов, делать повторные попытки (retry logic)
- Сохранять состояние между вызовами инструментов

#### Defining tools

Передайте список инструментов агенту:

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model, tools=[search, get_weather])
```

Если передан пустой список инструментов — агент будет состоять из единственного LLM-узла без возможности вызывать инструменты.

---

### Tool error handling

Чтобы кастомизировать, как обрабатываются ошибки при вызове инструментов, используйте декоратор `@wrap_tool_call` для создания middleware:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Обработка ошибок инструментов: возвращает понятное сообщение модели."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

При ошибке инструментов агент вернёт `ToolMessage` с вашим сообщением.

---

### Tool use in the ReAct loop

Агенты следуют шаблону ReAct (**Re**asoning + **Act**ing), чередуя короткие шаги рассуждений с конкретными вызовами инструментов, а затем передают результаты наблюдений обратно как часть контекста для последующих решений, пока не получится финальный ответ.

**Пример цикла ReAct**

**Prompt:** Найти самые популярные беспроводные наушники на данный момент и проверить, есть ли они в наличии.

- Рассуждение: «Популярность — это временно, нужно использовать инструмент search»
- Действие: вызвать `search_products("wireless headphones")`
- Получение результатов от инструмента — список продуктов
- Рассуждение: «Нужно подтвердить наличие топовой модели»
- Действие: вызвать `check_inventory("WH-1000XM5")`
- Получение ответа от инструмента — наличие 10 штук
- Финальный ответ с выводом результатов

Для более подробной информации про инструменты см. раздел [Tools](../core_components/tools.md).

---

### System prompt

Вы можете управлять, как агент подходит к задаче, задавая `system_prompt` — строковый параметр:

```python
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)
```

Если параметр не задан — агент сам определит задачу исходя из сообщений.

#### Dynamic system prompt

Для более сложных случаев, когда нужно менять системный prompt на основе контекста выполнения или состояния агента, используйте middleware. Например, декоратор `@dynamic_prompt` может генерировать `system_prompt`, исходя из данных в запросе:

```python
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Генерирует system prompt на основе роли пользователя."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent = create_agent(
    model="gpt-4o",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)
```

В этом случае system prompt будет устанавливаться динамически в зависимости от контекста.

Для форматов сообщений см. раздел [Messages](#). Для документации по middleware — раздел [Middleware](#).

---

## Invocation

Вы можете вызвать агента, передав обновление в его `State`. Все агенты содержат последовательность сообщений в состоянии; чтобы вызвать агента — передайте новое сообщение:

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
```

Если агент выполняет несколько шагов, это может занять некоторое время. Для отображения промежуточного прогресса можно использовать streaming (см. раздел [Streaming](#)). В противном случае агент следует API графа (Graph API) LangGraph через методы `stream` и `invoke`.

---

## Advanced concepts

### Structured output

В некоторых ситуациях вы хотите, чтобы агент возвращал вывод в специфическом формате. LangChain предоставляет стратегии для структурированного вывода через параметр `response_format`.

#### ToolStrategy

`ToolStrategy` использует искусственные вызовы инструментов, чтобы сгенерировать структурированный вывод. Это работает с любой моделью, поддерживающей вызов инструментов:

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

#### ProviderStrategy

`ProviderStrategy` использует нативную генерацию структурированного вывода провайдера модели. Это надёжнее, но работает только с провайдерами, поддерживающими такой вывод (например, OpenAI):

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)
```

Начиная с версии `langchain 1.0`, простая передача схемы (например, `response_format=ContactInfo`) больше не поддерживается. Нужно явно использовать `ToolStrategy` или `ProviderStrategy`.

Для подробностей про структурированный вывод см. раздел [Structured output](#).

---

### Memory

Агенты автоматически сохраняют историю разговора через состояние сообщений. Вы также можете настроить агента так, чтобы он запоминал дополнительную информацию в ходе диалога. Эта информация может рассматриваться как кратковременная память агента. Для этого можно использовать кастомную схему состояния, которая должна расширять `AgentState` как `TypedDict`.
Есть два способа определить кастомное состояние:

1. Через middleware (рекомендуемый способ)
2. Через параметр `state_schema` при `create_agent`

Определение состояния через middleware предпочтительнее, если ваше состояние нужно получить в определённых middleware-хуках и инструментах. `state_schema` всё ещё поддерживается для обратной совместимости.

#### Пример через middleware

```python
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        ...

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

#### Пример через `state_schema`

```python
from langchain.agents import AgentState

class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

Важно: с `langchain 1.0` кастомные схемы состояния должны быть `TypedDict`. Pydantic-модели и dataclasses больше не поддерживаются.

Для долгосрочной памяти, сохраняющейся между сессиями — см. раздел [Long-term memory](#).

---

### Streaming

Мы уже видели, как агент может быть вызван с `invoke`, чтобы получить финальный ответ. Если агент выполняет несколько шагов, это может занять время. Чтобы показывать промежуточный прогресс, можно возвращать сообщения по мере их появления — использовать streaming:

```python
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
}, stream_mode="values"):
    # Каждый chunk содержит полное состояние на тот момент
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```

Больше подробностей — в разделе [Streaming](#).

---

### Middleware

Middleware даёт мощные возможности для расширения поведения агента на разных этапах выполнения. С помощью middleware вы можете:

- Обрабатывать состояние перед вызовом модели (например, обрезка контекста, добавление контекста)
- Модифицировать или валидировать ответ модели (например, фильтрация, guardrails)
- Обрабатывать ошибки при вызове инструментов с кастомной логикой
- Реализовывать динамический выбор модели в зависимости от состояния и контекста
- Добавлять логирование, мониторинг, аналитику и т. д.

Middleware интегрируется с рантаймом агента, позволяя перехватывать и изменять поток данных на ключевых этапах без изменения базовой логики агента.

## Для подробной документации по middleware (декораторы `@before_model`, `@after_model`, `@wrap_tool_call` и др.) смотрите раздел [Middleware](#).

Source: [https://docs.langchain.com/oss/python/langchain-agents](https://docs.langchain.com/oss/python/langchain-agents)
