---
title: "Build a RAG agent with LangChain"
topic: "RAG"
filename: "build-a-rag-agent-with-langchain.md"
source: "https://docs.langchain.com/oss/python/langchain/rag"
author: "Перевод GPT-4"
date: "2025-12-08"
---

# Build a RAG agent with LangChain

## Overview

Одним из самых мощных применений LLM является создание продвинутых чат-ботов для вопрос-ответов (Q&A), которые могут отвечать на вопросы по конкретной исходной информации. Такие приложения используют технику, известную как Retrieval Augmented Generation, или **RAG**.

В этом руководстве мы покажем, как построить простое Q&A-приложение поверх неструктурированного текстового источника данных. Мы продемонстрируем:

1. RAG-агент, который выполняет поиск с помощью простого инструмента — это хорошая реализация для общего назначения.
2. Двухшаговую RAG-цепочку, использующую всего один вызов LLM на запрос — быстрый и эффективный метод для простых запросов.

## Concepts

Мы рассмотрим следующие концепции:

- **Indexing**: конвейер для загрузки данных из источника и индексирования их. Обычно это выполняется в отдельном процессе.
- **Retrieval and generation**: фактический RAG-процесс, который при получении пользовательского запроса извлекает релевантные данные из индекса, а затем передаёт их модели.

После того как мы проиндексировали наши данные, мы используем агента как orchestration-фреймворк для реализации шагов поиска и генерации.

Если ваши данные уже доступны для поиска (например, у вас есть функция для выполнения поиска), или вы знакомы с содержимым туториала по semantic search, вы можете пропустить раздел индексирования.

## Preview

В этом руководстве мы создадим приложение, которое может отвечать на вопросы о содержимом веб-сайта. В качестве источника мы возьмём блог-пост **LLM Powered Autonomous Agents** от Lilian Weng — так у нас будет возможность задавать вопросы о содержимом поста. Мы создадим простой pipeline индексирования и RAG-цепочку примерно за 40 строк кода.

```python
import bs4
from langchain.agents import AgentState, create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain.messages import MessageLikeRepresentation
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Загрузить и разбить содержимое блога
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
# Индексируем чанки
_ = vector_store.add_documents(documents=all_splits)

# Конструируем tool для получения контекста
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]

# При необходимости — задать кастомные инструкции
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

================================ Human Message =================================

What is task decomposition?

================================== Ai Message ==================================
Tool Calls:
  retrieve_context (call_xTkJr8njRY0geNz43ZvGkX0R)
 Call ID: call_xTkJr8njRY0geNz43ZvGkX0R
  Args:
    query: task decomposition
================================== Tool Message =================================
Name: retrieve_context
Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Task decomposition can be done by...

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Component One: Planning...
================================== Ai Message ==================================

Task decomposition refers to...
```

## Setup

### Installation

Для выполнения этого руководства требуется следующий набор зависимостей:

```bash
pip install langchain langchain-text-splitters langchain-community bs4
```

Для получения более подробной информации см. [Installation guide](https://docs.langchain.com/oss/python/langchain/installation).

## LangSmith

Многие приложения, которые вы создаёте с помощью LangChain, содержат несколько шагов с множественными вызовами LLM. Когда такие приложения становятся более сложными, становится критически важно иметь возможность инспектировать, что именно происходит внутри вашего chain или агента. Лучший способ сделать это — с помощью [LangSmith](https://smith.langchain.com).

После регистрации убедитесь, что установили переменные окружения:

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

Или, если вы работаете в Python:

```python
import getpass
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

## Components

Для работы RAG-приложения нам нужно выбрать три компонента из набора интеграций LangChain:

- Chat-модель
- Embeddings-модель
- Векторное хранилище (Vector store)

Например:

````bash
pip install -U "langchain[openai]"

Затем — chat-модель:

```python
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4.1")
````

Embeddings:

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

Vector store:

```python
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)
```

## 1. Indexing

Этот раздел — сокращённая версия содержимого туториала по semantic search. Если ваши данные уже проиндексированы и доступны для поиска, можете сразу переходить к разделу Retrieval and Generation.

Обычно процесс индексирования выглядит так:

1. **Load**: загрузка данных с помощью Document Loaders.
2. **Split**: разбивка больших `Documents` на меньшие чанки с помощью text splitters — так легче индексировать и передавать в модель, и они лучше подходят для поиска.
3. **Store**: сохранение и индексирование чанков в векторном хранилище (VectorStore + Embeddings).

### Loading documents

В нашем примере мы используем `WebBaseLoader`, который загружает HTML-страницу по URL и парсит её в текст с помощью `BeautifulSoup`. При этом мы фильтруем HTML-теги, оставляя только заголовки и основной контент — остальные теги удаляются.

### Splitting documents

Если загруженный документ слишком длинный (например, десятки тысяч символов), он может не поместиться в контекстную область модели или быть плохо обработан моделью. Поэтому мы разбиваем его на чанки с помощью `RecursiveCharacterTextSplitter`, который рекурсивно делит документ по разделителям (новые строки, абзацы и др.) до заданного размера.

### Storing documents

После разбивки — мы встраиваем (embed) все чанки и сохраняем их в векторное хранилище. При поиске по запросу модель сможет найти релевантные чанки и использовать их в качестве контекста.

## 2. Retrieval and Generation

RAG-приложения обычно работают следующим образом:

1. **Retrieve**: при получении пользовательского запроса мы извлекаем релевантные чанки документов из хранилища с помощью Retriever.
2. **Generate**: передаём вопрос + retrieved контекст модели, которая генерирует ответ.

### RAG agents

Один из подходов — использовать агента с инструментом, который обращается к векторному хранилищу. Например, мы можем определить tool `retrieve_context`, который оборачивает вызов `vector_store.similarity_search`. Этот инструмент возвращает как сериализованный текст (для передачи LLM), так и сами документы как артефакты.

Такой подход даёт гибкость: LLM сама решает, когда вызвать поиск, какие запросы сформулировать, сколько раз вызвать поиск — очень гибкое решение для общего случая.

### RAG chains

Другой подход — двухшаговая цепочка: сначала всегда выполняем поиск по исходному запросу (или при необходимости), затем делаем один вызов LLM с контекстом. Это даёт один вызов LLM на изначальный запрос, что быстрее и может быть полезно для простых задач.

| ✅ Преимущества RAG agents                                                            | ⚠️ Недостатки                                                                                    |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Поиск только при необходимости — LLM может обрабатывать простые сообщения без поиска. | Требуется два вызова LLM (one для запроса к поиску + one для ответа).                            |
| LLM сама формирует релевантный поисковый запрос с учётом контекста.                   | Меньший контроль: модель может пропустить поиск, даже если он нужен, или запустить лишний поиск. |
| Возможность нескольких поисков за один пользовательский запрос.                       | —                                                                                                |

В то же время двухшаговая цепочка:

- всегда выполняет поиск (например, по первоначальному запросу)
- использует один вызов LLM — быстрее и эффективнее для простых задач.

---

## Returning source documents

В случае RAG-цепочки (двухшаговой) можно сохранять не только текст контекста, но и сами объекты документов (например, список `retrieved_docs`) в состоянии агента. Это делается добавлением соответствующего поля в state и использованием middleware, который перед вызовом модели заполняет это поле.

```python
from typing import Any
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState

class State(AgentState):
    context: list[Document]

class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        retrieved_docs = vector_store.similarity_search(last_message.text)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        augmented_message_content = (
            f"{last_message.text}\n\n"
            "Use the following context to answer the query:\n"
            f"{docs_content}"
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }

agent = create_agent(
    model,
    tools=[],
    middleware=[RetrieveDocumentsMiddleware()],
)
```

---

## Next steps

После того, как вы реализовали простое RAG-приложение с помощью `create_agent`, вы можете легко добавить дополнительные возможности:

- Стриминг токенов и прочей информации для более отзывчивого пользовательского опыта.
- Добавление conversational memory для поддержки диалогов в несколько шагов.
- Добавление долгосрочной памяти (long-term memory), чтобы сохранять состояние между сессиями.
- Структурированные ответы (structured responses).
- Деплой приложения с помощью LangSmith Deployments.

---

Source: https://docs.langchain.com/oss/python/langchain/rag
