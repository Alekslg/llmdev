---
title: "Chroma"
topic: "Modules"
filename: "chroma.md"
source: "https://python.langchain.com/docs/integrations/vectorstores/chroma"
author: "Перевод GPT"
date: "2025-11-10"
---

# Chroma

Этот документ описывает, как начать работу с векторным хранилищем `Chroma`.

> [Chroma](https://docs.trychroma.com/getting-started) — это AI-native база данных с открытым исходным кодом, ориентированная на продуктивность и удобство разработчиков. Chroma распространяется под лицензией Apache 2.0. Полная документация `Chroma` доступна [здесь](https://docs.trychroma.com/reference/py-collection), а справочник API для интеграции с LangChain — [здесь](https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html).

<Info>
  **Chroma Cloud**

Chroma Cloud обеспечивает серверный векторный и полнотекстовый поиск. Он чрезвычайно быстрый, экономичный, масштабируемый и простой в использовании. Создайте базу данных и протестируйте её за менее чем 30 секунд с $5 бесплатных кредитов.

[Начать работу с Chroma Cloud](https://trychroma.com/signup)
</Info>

## Установка

Для доступа к векторным хранилищам `Chroma` необходимо установить интеграционный пакет `langchain-chroma`.

```python
pip install -qU "langchain-chroma>=0.1.2"
```

### Учетные данные

Вы можете использовать векторное хранилище `Chroma` без каких-либо учетных данных — достаточно установить пакет выше!

Если вы пользователь [Chroma Cloud](https://trychroma.com/signup), установите переменные окружения `CHROMA_TENANT`, `CHROMA_DATABASE` и `CHROMA_API_KEY`.

При установке пакета `chromadb` вы также получаете доступ к CLI Chroma, который может настроить эти переменные за вас. Сначала выполните [вход](https://docs.trychroma.com/docs/cli/login) через CLI, а затем используйте [команду `connect`](https://docs.trychroma.com/docs/cli/db):

```bash
chroma db connect [db_name] --env-file
```

Если вы хотите получить автоматическое трассирование вызовов вашей модели высочайшего качества, вы также можете установить ваш API-ключ [LangSmith](https://docs.smith.langchain.com/), раскомментировав следующее:

```python
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
os.environ["LANGSMITH_TRACING"] = "true"
```

## Инициализация

### Базовая инициализация

Ниже приведена базовая инициализация, включая использование директории для локального сохранения данных.

<EmbeddingTabs />

```python
# | output: false
# | echo: false
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

#### Запуск локально (в памяти)

Вы можете запустить Chroma в памяти, просто создав экземпляр `Chroma` с именем коллекции и функцией эмбеддингов:

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)
```

Если вам не требуется сохранение данных, это отличный вариант для экспериментов при разработке вашего AI-приложения с LangChain.

#### Запуск локально (с сохранением данных)

Вы можете указать аргумент `persist_directory`, чтобы сохранить данные между запусками программы:

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
```

#### Подключение к серверу Chroma

Если у вас запущен сервер Chroma локально или вы [развернули](https://docs.trychroma.com/guides/deploy/client-server-mode) его самостоятельно, вы можете подключиться к нему, указав аргумент `host`.

Например, вы можете запустить локальный сервер Chroma с помощью команды `chroma run`, а затем подключиться к нему с `host='localhost'`:

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    host="localhost",
)
```

Для других развертываний вы можете использовать аргументы `port`, `ssl` и `headers` для настройки подключения.

#### Chroma Cloud

Пользователи Chroma Cloud также могут работать с LangChain. Передайте вашему экземпляру `Chroma` API-ключ Chroma Cloud, тенант и имя базы данных:

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)
```

### Инициализация через клиент

Вы также можете инициализировать хранилище через клиент `Chroma`, что особенно удобно, если вам нужен более прямой доступ к базе данных.

#### Запуск локально (в памяти)

```python
import chromadb

client = chromadb.Client()
```

#### Запуск локально (с сохранением данных)

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_langchain_db")
```

#### Подключение к серверу Chroma

Например, если вы запускаете локальный сервер Chroma (с помощью `chroma run`):

```python
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000, ssl=False)
```

#### Chroma Cloud

После установки переменных `CHROMA_API_KEY`, `CHROMA_TENANT` и `CHROMA_DATABASE` вы можете просто создать экземпляр:

```python
import chromadb

client = chromadb.CloudClient()
```

#### Доступ к вашей базе Chroma

```python
collection = client.get_or_create_collection("collection_name")
collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])
```

#### Создание векторного хранилища Chroma

```python
vector_store_from_client = Chroma(
    client=client,
    collection_name="collection_name",
    embedding_function=embeddings,
)
```

## Управление векторным хранилищем

После создания векторного хранилища вы можете взаимодействовать с ним, добавляя и удаляя элементы.

### Добавление элементов в векторное хранилище

Вы можете добавлять элементы в векторное хранилище с помощью функции `add_documents`.

```python
from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
    id=4,
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
    id=5,
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
    id=6,
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    id=7,
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    id=8,
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
    id=9,
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    id=10,
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)
```

### Обновление элементов в векторном хранилище

Теперь, когда документы добавлены, вы можете обновить существующие документы с помощью функции `update_documents`.

```python
updated_document_1 = Document(
    page_content="I had chocolate chip pancakes and fried eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

updated_document_2 = Document(
    page_content="The weather forecast for tomorrow is sunny and warm, with a high of 82 degrees.",
    metadata={"source": "news"},
    id=2,
)

vector_store.update_document(document_id=uuids[0], document=updated_document_1)
# Вы также можете обновить несколько документов одновременно
vector_store.update_documents(
    ids=uuids[:2], documents=[updated_document_1, updated_document_2]
)
```

### Удаление элементов из векторного хранилища

Вы также можете удалять элементы из векторного хранилища следующим образом:

```python
vector_store.delete(ids=uuids[-1])
```

## Запрос к векторному хранилищу

После создания векторного хранилища и добавления соответствующих документов вы, скорее всего, захотите выполнять к нему запросы во время выполнения цепочки или агента.

### Прямой запрос

#### Поиск по сходству

Простой поиск по сходству можно выполнить следующим образом:

```python
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

#### Поиск по сходству с оценкой

Если вы хотите выполнить поиск по сходству и получить соответствующие оценки, вы можете запустить:

```python
results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
```

#### Поиск по вектору

Вы также можете искать по вектору:

```python
results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("I love green eggs and ham!"), k=1
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")
```

#### Другие методы поиска

Существует множество других методов поиска, не описанных в этом документе, таких как MMR-поиск или поиск по вектору. Полный список возможностей поиска для `AstraDBVectorStore` доступен в [справочнике API](https://python.langchain.com/api_reference/astradb/vectorstores/langchain_astradb.vectorstores.AstraDBVectorStore.html).

### Запрос через преобразование в retriever

Вы также можете преобразовать векторное хранилище в retriever для удобного использования в цепочках. Более подробную информацию о различных типах поиска и параметрах `kwargs` см. в справочнике API [здесь](https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.as_retriever).

```python
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)
retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
```

## Использование для генерации с извлечением (RAG)

Руководства по использованию этого векторного хранилища для генерации с извлечением (retrieval-augmented generation, RAG) см. в следующих разделах:

- [Руководства](/oss/python/langchain/rag)
- [How-to: Вопросы и ответы с RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Концептуальные документы по извлечению](https://python.langchain.com/docs/concepts/retrieval)

---

## Справочник API

Подробная документация по всем функциям и конфигурациям векторного хранилища `Chroma` доступна в справочнике API: [python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html](https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html)

---

Source: https://python.langchain.com/docs/integrations/vectorstores/chroma
