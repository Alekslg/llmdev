---
title: "Протокол контекста модели (MCP)"
topic: "tutorials"
filename: "model-context-protocol-mcp.md"
source: "https://docs.langchain.com/oss/python/langchain/mcp"
author: "Перевод perplexity-llm"
date: "2025-12-05"
---

## Quickstart

Установите библиотеку `langchain-mcp-adapters`:

```bash
pip install langchain-mcp-adapters
```

`langchain-mcp-adapters` позволяет агентам использовать tools, определённые на одном или нескольких MCP servers.

`MultiServerMCPClient` по умолчанию **не хранит состояние (stateless)**. Каждый вызов tool создаёт новый MCP `ClientSession`, выполняет tool и затем очищает его; подробности см. в разделе о stateful sessions.

### Доступ к нескольким MCP servers

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",  # Локальное взаимодействие через subprocess
            "command": "python",
            # Абсолютный путь к вашему файлу math_server.py
            "args": ["/path/to/math_server.py"],
        },
        "weather": {
            "transport": "http",  # Удалённый сервер на базе HTTP
            # Убедитесь, что вы запустили weather server на порту 8000
            "url": "http://localhost:8000/mcp",
        },
    }
)

tools = await client.get_tools()

agent = create_agent(
    "claude-sonnet-4-5-20250929",
    tools,
)

math_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
)

weather_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
)
```

## Custom servers

Чтобы создать собственный MCP server, используйте библиотеку FastMCP. Для тестирования вашего агента с MCP tool servers используйте следующие примеры:

```bash
pip install fastmcp
```

```python
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Сложить два числа"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Перемножить два числа"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## Transports

MCP поддерживает разные механизмы транспорта для взаимодействия client–server.

### HTTP

Транспорт `http` (также называемый `streamable-http`) использует HTTP‑запросы для взаимодействия client–server. Подробности см. в спецификации MCP HTTP transport.

```python
client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
        },
    }
)
```

#### Передача headers

При подключении к MCP servers по HTTP можно указывать пользовательские headers (например, для аутентификации или трейсинга) через поле `headers` в конфигурации подключения. Это поддерживается для транспортов `sse` (deprecated в спецификации MCP) и `streamable_http`.

Передача headers с `MultiServerMCPClient`:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
            "headers": {
                "Authorization": "Bearer YOUR_TOKEN",
                "X-Custom-Header": "custom-value",
            },
        },
    }
)

tools = await client.get_tools()
agent = create_agent("openai:gpt-4.1", tools)

response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
```

#### Authentication

Библиотека `langchain-mcp-adapters` использует под капотом официальный MCP SDK, который позволяет реализовать настраиваемый механизм аутентификации через реализацию интерфейса `httpx.Auth`.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
            "auth": auth,
        },
    }
)
```

### stdio

В этом режиме client запускает server как subprocess и взаимодействует через стандартный ввод/вывод, что лучше всего подходит для локальных tools и простых конфигураций. В отличие от HTTP‑транспортов, соединения `stdio` по своей природе **stateful** — subprocess живёт всё время существования клиентского соединения, но при использовании `MultiServerMCPClient` без явного управления сессиями каждый вызов tool всё равно создаёт новую session; см. раздел о stateful sessions для управления постоянными соединениями.

```python
client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["/path/to/math_server.py"],
        },
    }
)
```

## Stateful sessions

По умолчанию `MultiServerMCPClient` **не хранит состояние (stateless)** — каждый вызов tool создаёт новый MCP session, выполняет tool и затем очищает его. Если нужно управлять жизненным циклом MCP session (например, при работе с stateful server, который хранит контекст между вызовами tools), можно создать постоянный `ClientSession` с помощью `client.session()`.

Использование MCP `ClientSession` для stateful‑вызовов tools:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

client = MultiServerMCPClient({...})

# Явное создание session
async with client.session("server_name") as session:
    # Передайте session для загрузки tools, resources или prompts
    tools = await load_mcp_tools(session)

    agent = create_agent(
        "anthropic:claude-3-7-sonnet-latest",
        tools,
    )
```

## Core features

### Tools

Tools позволяют MCP servers экспонировать исполняемые функции, которые LLM может вызывать для выполнения действий — таких как запросы к базам данных, вызовы внешних API или взаимодействие с внешними системами. LangChain конвертирует MCP tools в LangChain tools, что делает их напрямую доступными в любом LangChain agent или workflow.

#### Загрузка tools

Используйте `client.get_tools()` для получения списка tools с MCP servers и передачи их агенту:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({...})

tools = await client.get_tools()

agent = create_agent("claude-sonnet-4-5-20250929", tools)
```

#### Structured content

MCP tools могут возвращать структурированный контент вместе с человекочитаемым текстовым ответом. Это полезно, когда tool должен вернуть машиночитаемые данные (например, JSON) в дополнение к тексту, который показывается модели; если MCP tool возвращает `structuredContent`, адаптер оборачивает его в `MCPToolArtifact` и возвращает как artifact tool, к которому можно получить доступ через поле `artifact` в `ToolMessage`.

**Извлечение structured content из artifact**:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import ToolMessage

client = MultiServerMCPClient({...})
tools = await client.get_tools()
agent = create_agent("claude-sonnet-4-5-20250929", tools)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Get data from the server"}]}
)

# Извлекаем structured content из tool messages
for message in result["messages"]:
    if isinstance(message, ToolMessage) and message.artifact:
        structured_content = message.artifact["structured_content"]
```

#### Multimodal tool content

MCP tools могут возвращать мультимодальный контент (изображения, текст и т.д.) в своих ответах. Когда MCP server возвращает контент из нескольких частей (например, текст и изображения), адаптер конвертирует его в стандартные content blocks LangChain, к которым можно получить доступ через свойство `content_blocks` у `ToolMessage`, что позволяет обрабатывать мультимодальные ответы tool независимо от конкретного провайдера.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({...})
tools = await client.get_tools()
agent = create_agent("claude-sonnet-4-5-20250929", tools)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Take a screenshot of the current page"}]}
)

# Доступ к мультимодальному контенту из tool messages
for message in result["messages"]:
    if message.type == "tool":
        # Сырой контент в нативном формате провайдера
        print(f"Raw content: {message.content}")

        # Стандартизованные content blocks
        for block in message.content_blocks:
            if block["type"] == "text":
                print(f"Text: {block['text']}")
            elif block["type"] == "image":
                print(f"Image URL: {block.get('url')}")
                print(f"Image base64: {block.get('base64', '')[:50]}...")
```

### Resources

Resources позволяют MCP servers экспонировать данные — такие как файлы, записи баз данных или ответы внешних API, которые могут быть прочитаны клиентами. LangChain преобразует MCP resources в объекты Blob, обеспечивая единый интерфейс как для текстового, так и для бинарного контента.

#### Загрузка resources

Используйте `client.get_resources()` для загрузки resources с MCP server:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({...})

# Загрузить все resources с сервера
blobs = await client.get_resources("server_name")

# Или загрузить конкретные resources по URI
blobs = await client.get_resources("server_name", uris=["file:///path/to/file.txt"])

for blob in blobs:
    print(f"URI: {blob.metadata['uri']}, MIME type: {blob.mimetype}")
    print(blob.as_string())  # Для текстового контента
```

Можно также использовать `load_mcp_resources` напрямую с session для более точного контроля:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.resources import load_mcp_resources

client = MultiServerMCPClient({...})

async with client.session("server_name") as session:
    # Загрузить все resources
    blobs = await load_mcp_resources(session)

    # Или загрузить конкретные resources по URI
    blobs = await load_mcp_resources(session, uris=["file:///path/to/file.txt"])
```

### Prompts

Prompts позволяют MCP servers экспонировать переиспользуемые шаблоны prompt, которые могут быть получены и использованы клиентами. LangChain преобразует MCP prompts в messages, что упрощает их интеграцию в chat‑ориентированные workflows.

#### Загрузка prompts

Используйте `client.get_prompt()` для загрузки prompt с MCP server:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({...})

# Загрузить prompt по имени
messages = await client.get_prompt("server_name", "summarize")

# Загрузить prompt с аргументами
messages = await client.get_prompt(
    "server_name",
    "code_review",
    arguments={"language": "python", "focus": "security"},
)

# Использовать messages в вашем workflow
for message in messages:
    print(f"{message.type}: {message.content}")
```

Также можно использовать `load_mcp_prompt` напрямую с session для более точного контроля:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.prompts import load_mcp_prompt

client = MultiServerMCPClient({...})

async with client.session("server_name") as session:
    # Загрузить prompt по имени
    messages = await load_mcp_prompt(session, "summarize")

    # Загрузить prompt с аргументами
    messages = await load_mcp_prompt(
        session,
        "code_review",
        arguments={"language": "python", "focus": "security"},
    )
```

## Advanced features

### Tool Interceptors

MCP servers запускаются как отдельные процессы и не имеют доступа к runtime‑контексту LangGraph (store, context или agent state). Interceptors устраняют этот разрыв, предоставляя доступ к этому runtime‑контексту во время выполнения MCP tools и действуя как middleware над вызовами tools: они позволяют изменять запросы, реализовывать retry‑логику, динамически добавлять headers или полностью перехватывать выполнение.

| Section                    | Description                                                             |
| -------------------------- | ----------------------------------------------------------------------- |
| Accessing runtime context  | Чтение user IDs, API keys, данных store и agent state                   |
| State updates and commands | Обновление agent state или управление потоком graph через `Command`     |
| Writing interceptors       | Паттерны изменения запросов, композиции interceptors и обработки ошибок |

#### Accessing runtime context

Когда MCP tools используются внутри LangChain agent (через `create_agent`), interceptors получают доступ к `ToolRuntime` context. Это даёт доступ к ID вызова tool, state, config и store и позволяет реализовывать мощные паттерны работы с пользовательскими данными, их сохранением и управлением поведением агента; дополнительные паттерны контекст‑инжиниринга см. в разделах Context engineering и Tools.

Можно получать пользовательскую конфигурацию (user IDs, API keys или permissions), передаваемую во время вызова:

**Добавление пользовательского контекста в MCP tool‑вызовы**:

```python
from dataclasses import dataclass

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.agents import create_agent

@dataclass
class Context:
    user_id: str
    api_key: str

async def inject_user_context(
    request: MCPToolCallRequest,
    handler,
):
    """Вставить пользовательские учётные данные в MCP tool-вызовы."""
    runtime = request.runtime
    user_id = runtime.context.user_id
    api_key = runtime.context.api_key

    # Добавляем пользовательский контекст в аргументы tool
    modified_request = request.override(
        args={**request.args, "user_id": user_id}
    )

    return await handler(modified_request)

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[inject_user_context],
)

tools = await client.get_tools()

agent = create_agent("gpt-4o", tools, context_schema=Context)

# Вызов с пользовательским контекстом
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Search my orders"}]},
    context={"user_id": "user_123", "api_key": "sk-..."},
)
```

#### State updates and commands

Interceptors могут возвращать объекты `Command` для обновления agent state или управления выполнением graph. Это удобно для отслеживания прогресса задач, переключения между agents или преждевременного завершения выполнения.

**Отметить задачу завершённой и переключиться на другой agent**:

```python
from langchain.agents import AgentState, create_agent
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command

async def handle_task_completion(
    request: MCPToolCallRequest,
    handler,
):
    """Отметить задачу завершённой и передать управление summary agent."""
    result = await handler(request)

    if request.name == "submit_order":
        return Command(
            update={
                "messages": [result] if isinstance(result, ToolMessage) else [],
                "task_status": "completed",
            },
            goto="summary_agent",
        )

    return result
```

Можно использовать `Command` с `goto="__end__"`, чтобы завершить выполнение раньше времени:

**Завершение работы агента при успешном выполнении**:

```python
async def end_on_success(
    request: MCPToolCallRequest,
    handler,
):
    """Завершить работу агента, когда задача помечена завершённой."""
    result = await handler(request)

    if request.name == "mark_complete":
        return Command(
            update={"messages": [result], "status": "done"},
            goto="__end__",
        )

    return result
```

#### Custom interceptors

Interceptors — это async‑функции, которые оборачивают выполнение tool, позволяя изменять запросы и ответы, реализовывать retry‑логику и другие сквозные задачи. Они следуют “луковичной” модели: первый interceptor в списке является самым внешним слоем.

**Базовый паттерн**: interceptor — это async‑функция, получающая request и handler и способная изменить request до вызова handler, изменить response после или полностью пропустить вызов handler.

**Базовый interceptor**:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def logging_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Логировать вызовы tools до и после выполнения."""
    print(f"Calling tool: {request.name} with args: {request.args}")
    result = await handler(request)
    print(f"Tool {request.name} returned: {result}")
    return result

client = MultiServerMCPClient(
    {"math": {"transport": "stdio", "command": "python", "args": ["/path/to/server.py"]}},
    tool_interceptors=[logging_interceptor],
)
```

**Изменение запросов**

Используйте `request.override()` для создания модифицированного запроса, следуя неизменяемому паттерну, при котором оригинальный request остаётся без изменений.

**Изменение аргументов tool**:

```python
async def double_args_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Удвоить все числовые аргументы перед выполнением."""
    modified_args = {k: v * 2 for k, v in request.args.items()}
    modified_request = request.override(args=modified_args)
    return await handler(modified_request)

# Оригинальный вызов: add(a=2, b=3) становится add(a=4, b=6)
```

**Изменение headers во время выполнения**

Interceptors могут динамически менять HTTP headers на основе контекста запроса:

```python
async def auth_header_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Добавить заголовки аутентификации в зависимости от вызываемого tool."""
    token = get_token_for_tool(request.name)
    modified_request = request.override(
        headers={"Authorization": f"Bearer {token}"}
    )
    return await handler(modified_request)
```

**Композиция interceptors**

Несколько interceptors компонуются в порядке “луковицы”: первый в списке — внешний слой.

```python
async def outer_interceptor(request, handler):
    print("outer: before")
    result = await handler(request)
    print("outer: after")
    return result

async def inner_interceptor(request, handler):
    print("inner: before")
    result = await handler(request)
    print("inner: after")
    return result

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[outer_interceptor, inner_interceptor],
)

# Порядок выполнения:
# outer: before -> inner: before -> выполнение tool ->
# inner: after -> outer: after
```

**Обработка ошибок**

Interceptors можно использовать для перехвата ошибок выполнения tools и реализации retry‑логики, а также для обработки конкретных типов ошибок и возврата fallback‑значений.

**Повтор при ошибке**:

```python
import asyncio

async def retry_interceptor(
    request: MCPToolCallRequest,
    handler,
    max_retries: int = 3,
    delay: float = 1.0,
):
    """Повторять неудачные вызовы tool с экспоненциальной задержкой."""
    last_error = None

    for attempt in range(max_retries):
        try:
            return await handler(request)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Экспоненциальный backoff
                print(
                    f"Tool {request.name} failed (attempt {attempt + 1}), "
                    f"retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

    raise last_error

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[retry_interceptor],
)
```

**Обработка ошибок с fallback**:

```python
async def fallback_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Вернуть fallback-значение, если выполнение tool завершилось ошибкой."""
    try:
        return await handler(request)
    except TimeoutError:
        return f"Tool {request.name} timed out. Please try again later."
    except ConnectionError:
        return f"Could not connect to {request.name} service. Using cached data."
```

### Progress notifications

Можно подписаться на уведомления о прогрессе для долгих по времени вызовов tools. Для этого используется callback `on_progress`, передаваемый через `Callbacks`.

**Progress callback**:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext

async def on_progress(
    progress: float,
    total: float | None,
    message: str | None,
    context: CallbackContext,
):
    """Обработать обновления прогресса от MCP servers."""
    percent = (progress / total * 100) if total else progress
    tool_info = f" ({context.tool_name})" if context.tool_name else ""
    print(f"[{context.server_name}{tool_info}] Progress: {percent:.1f}% - {message}")

client = MultiServerMCPClient(
    {...},
    callbacks=Callbacks(on_progress=on_progress),
)
```

`CallbackContext` предоставляет:

- `server_name`: имя MCP server
- `tool_name`: имя выполняемого tool (доступно во время вызовов tools)

### Logging

Протокол MCP поддерживает лог‑уведомления от servers, и для подписки на эти события используется класс `Callbacks`.

**Logging callback**:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from mcp.types import LoggingMessageNotificationParams

async def on_logging_message(
    params: LoggingMessageNotificationParams,
    context: CallbackContext,
):
    """Обработать лог-сообщения от MCP servers."""
    print(f"[{context.server_name}] {params.level}: {params.data}")

client = MultiServerMCPClient(
    {...},
    callbacks=Callbacks(on_logging_message=on_logging_message),
)
```

---

Source: <https://docs.langchain.com/oss/python/langchain/mcp>
