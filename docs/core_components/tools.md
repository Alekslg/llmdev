---
title: "Tools"
topic: "Core components"
filename: "tools.md"
source: "https://docs.langchain.com/oss/python/langchain/tools"
author: "Перевод GPT-4"
date: "2025-12-08"
---

# Tools

Многие AI-приложения взаимодействуют с пользователями через естественный язык. Однако в некоторых случаях требуется, чтобы модели напрямую взаимодействовали с внешними системами — например, с API, базами данных или файловыми системами — используя структурированный ввод. В таких сценариях инструменты (tools) предоставляют модели возможность совершать действия: они инкапсулируют вызываемую функцию (callable) и её схему ввода.

Эти инструменты можно передавать совместимым чат-моделям, что позволяет модели решать, вызывать ли инструмент и с какими аргументами. В сценариях, где используется tool calling, модель генерирует запросы, соответствующие заданной схеме ввода.

Некоторые чат-модели (например, OpenAI, Anthropic и Gemini) имеют встроенные инструменты, которые выполняются на стороне сервера — например, веб-поиск или интерпретаторы кода. Для информации о том, как получить доступ к таким инструментам у вашей модели — смотрите раздел provider overview.

---

## Create tools

### Basic tool definition

Самый простой способ создать инструмент — использовать декоратор `@tool`. По умолчанию строка документации функции (docstring) становится описанием инструмента, которое помогает модели понять, когда этот инструмент следует использовать:

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

Важно: при объявлении такого инструмента необходимы type hints — они определяют схему ввода (input schema) инструмента. Докстринг должен быть информативным и кратким, чтобы модель могла правильно понять назначение инструмента.

### Customize tool properties

#### Custom tool name

По умолчанию имя инструмента совпадает с именем функции. При необходимости можно переопределить его, чтобы дать более описательное имя:

```python
@tool("web_search")  # Custom name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # web_search
```

#### Custom tool description

Также можно переопределить автоматически сгенерированное описание инструмента, если требуется более ясное пояснение для модели:

```python
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```

## Advanced schema definition

Если ваш инструмент требует более сложных и структурированных входных данных — например, несколько полей, перечислений, вложенные структуры — можно определить схему ввода через модели Pydantic или JSON Schema.

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

## Reserved argument names

Есть имена параметров, которые **зарезервированы** и не могут быть использованы в сигнатуре вашего инструмента — иначе возникнут ошибки во время выполнения:

| Имя параметра | Назначение                                                                                |
| ------------- | ----------------------------------------------------------------------------------------- |
| `config`      | Зарезервировано для передачи внутреннего `RunnableConfig` для инструментов                |
| `runtime`     | Зарезервировано для параметра `ToolRuntime` (доступ к состоянию, контексту, store и т.п.) |

Если вам нужно получить доступ к runtime-информации — используйте `ToolRuntime`, а не свой собственный аргумент `runtime`.

## Accessing Context

### `ToolRuntime`

Инструменты могут получать доступ к состоянию агента, контексту, долговременной памяти и другим runtime-данным через параметр `ToolRuntime`. Для этого просто добавьте `runtime: ToolRuntime` в сигнатуру вашей функции — этот параметр будет автоматически передан, и при этом **не будет виден модели**, то есть не станет частью схемы ввода.

`ToolRuntime` предоставляет:

- `state` — изменяемое состояние, которое передаётся между шагами (например, сообщения, счётчики, кастомные поля)
- `context` — неизменяемые конфигурации, такие как идентификаторы пользователей, данные сессии, настройки приложения и т.п.
- `store` — персистентная долговременная память, общую между разговорами/сессиями.
- `stream_writer` — возможность стримить пользовательские обновления из инструмента во время выполнения.
- `config` — внутренний конфиг для запуска инструмента (`RunnableConfig`)
- `tool_call_id` — ID текущего вызова инструмента.

#### Пример использования `ToolRuntime`

```python
from langchain.tools import tool, ToolRuntime

@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]
    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")
    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"
```

В этом примере модель увидит только аргументы, необходимые для задачи (в данном случае — никаких), а `runtime` не попадает в схему, и её не видит модель.

## Memory (Store)

Инструменты могут использовать долговременную память — `store` — через `ToolRuntime` для хранения и получения данных между сессиями. Это удобно для сохранения пользовательских данных, кеширования, ведения истории и т.п.

```python
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."
```

В примере выше: первый запуск может сохранять данные пользователя, а последующие — извлекать их из памяти.

## Stream Writer

Если ваш инструмент выполняет операцию, которая может занимать заметное время — вы можете использовать `runtime.stream_writer` внутри инструмента, чтобы отправлять промежуточные обновления клиенту в режиме потоковой передачи. Это может быть полезно, чтобы показывать прогресс выполнения задачи, логировать ход, уведомлять пользователя и т.п.

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"
```

Если вы используете `runtime.stream_writer` — убедитесь, что инструмент вызывается в контексте выполнения LangGraph, так как стриминг поддерживается только там.

Source: [https://docs.langchain.com/oss/python/langchain/tools](https://docs.langchain.com/oss/python/langchain/tools)
