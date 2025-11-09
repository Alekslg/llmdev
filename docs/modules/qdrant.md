---
title: "Qdrant"
topic: "Modules"
filename: "qdrant.md"
source: "https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant"
author: "Перевод GPT"
date: "2025-11-10"
---

# Qdrant

> [Qdrant](https://qdrant.tech/documentation/) (произносится как _quadrant_) — это движок поиска по векторному сходству. Он предоставляет готовое к продакшену решение с удобным API для хранения, поиска и управления векторами с дополнительной поддержкой payload и расширенной фильтрации. Это делает его полезным для различных задач: нейросетевого или семантического сопоставления, фасетного поиска и других приложений.

В этом документе показано, как использовать Qdrant с LangChain для плотного (т.е. на основе эмбеддингов), разреженного (т.е. текстового поиска) и гибридного извлечения. Класс `QdrantVectorStore` поддерживает несколько режимов извлечения через новый [Query API](https://qdrant.tech/blog/qdrant-1.10.x/) Qdrant. Требуется Qdrant версии v1.10.0 или выше.

## Установка

Существует несколько способов запуска `Qdrant`, и в зависимости от выбранного варианта будут небольшие различия. Возможные варианты:

- Локальный режим без сервера
- Развертывание через Docker
- Qdrant Cloud

Инструкции по установке см. [здесь](https://qdrant.tech/documentation/install/).

```python
pip install -qU langchain-qdrant
```

### Учетные данные

Для запуска кода в этом документе не требуются учетные данные.

Если вы хотите получить автоматическое трассирование вызовов вашей модели высочайшего качества, вы также можете установить ваш API-ключ [LangSmith](https://docs.smith.langchain.com/), раскомментировав следующее:

```python
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
os.environ["LANGSMITH_TRACING"] = "true"
```

## Инициализация

### Локальный режим

Python-клиент позволяет запускать код в локальном режиме без запуска сервера Qdrant. Это отлично подходит для тестирования, отладки или хранения небольшого количества векторов. Эмбеддинги могут храниться полностью в памяти или сохраняться на диск.

#### В памяти

Для некоторых тестовых сценариев и быстрых экспериментов вы можете предпочесть хранить все данные только в памяти, чтобы они удалялись при уничтожении клиента — обычно в конце скрипта или ноутбука.

<EmbeddingTabs />

```python
# | output: false
# | echo: false
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embeddings,
)
```

#### Хранение на диске

В локальном режиме, без использования сервера Qdrant, вы также можете хранить векторы на диске, чтобы они сохранялись между запусками.

```python
client = QdrantClient(path="/tmp/langchain_qdrant")

client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embeddings,
)
```

### Развертывание на собственном сервере

Независимо от того, запускаете ли вы Qdrant локально с помощью [контейнера Docker](https://qdrant.tech/documentation/install/) или развертываете в Kubernetes с использованием [официального Helm-чарта](https://github.com/qdrant/qdrant-helm), способ подключения к такому экземпляру будет одинаковым. Вам нужно указать URL, указывающий на службу.

```python
url = "<---qdrant url here --->"
docs = []  # put docs here
qdrant = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    collection_name="my_documents",
)
```

### Qdrant Cloud

Если вы не хотите заниматься управлением инфраструктурой, вы можете создать полностью управляемый кластер Qdrant в [Qdrant Cloud](https://cloud.qdrant.io/). Для тестирования доступен бесплатный кластер объемом 1 ГБ. Основное отличие при использовании управляемой версии Qdrant — необходимость предоставления API-ключа для защиты вашего развертывания от публичного доступа. Значение также можно установить в переменной окружения `QDRANT_API_KEY`.

```python
url = "<---qdrant cloud cluster url here --->"
api_key = "<---api key here--->"
qdrant = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="my_documents",
)
```

## Использование существующей коллекции

Чтобы получить экземпляр `langchain_qdrant.Qdrant` без загрузки новых документов или текстов, вы можете использовать метод `Qdrant.from_existing_collection()`.

```python
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="my_documents",
    url="http://localhost:6333",
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
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees Fahrenheit.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
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
```

```python
vector_store.add_documents(documents=documents, ids=uuids)
```

### Удаление элементов из векторного хранилища

```python
vector_store.delete(ids=[uuids[-1]])
```

```output
True
```

## Запрос к векторному хранилищу

После создания векторного хранилища и добавления соответствующих документов вы, скорее всего, захотите выполнять к нему запросы во время выполнения цепочки или агента.

### Прямой запрос

Простейший сценарий использования векторного хранилища Qdrant — выполнение поиска по сходству. Под капотом наш запрос кодируется в векторные эмбеддинги и используется для поиска похожих документов в коллекции Qdrant.

```python
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy", k=2
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

```output
* Building an exciting new project with LangChain - come check it out! [{'source': 'tweet', '_id': 'd3202666-6f2b-4186-ac43-e35389de8166', '_collection_name': 'demo_collection'}]
* LangGraph is the best framework for building stateful, agentic applications! [{'source': 'tweet', '_id': '91ed6c56-fe53-49e2-8199-c3bb3c33c3eb', '_collection_name': 'demo_collection'}]
```

`QdrantVectorStore` поддерживает 3 режима поиска по сходству. Их можно настроить с помощью параметра `retrieval_mode`.

- Плотный векторный поиск (по умолчанию)
- Разреженный векторный поиск
- Гибридный поиск

### Плотный векторный поиск

Плотный векторный поиск включает вычисление сходства с помощью векторных эмбеддингов. Для поиска только с плотными векторами:

- Параметр `retrieval_mode` должен быть установлен в `RetrievalMode.DENSE`. Это поведение по умолчанию.
- Значение [плотных эмбеддингов](https://python.langchain.com/docs/integrations/text_embedding/) должно быть передано в параметр `embedding`.

```python
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Create a Qdrant client for local storage
client = QdrantClient(path="/tmp/langchain_qdrant")

# Create a collection with dense vectors
client.create_collection(
    collection_name="my_documents",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

qdrant = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
)

qdrant.add_documents(documents=documents, ids=uuids)

query = "How much money did the robbers steal?"
found_docs = qdrant.similarity_search(query)
found_docs
```

### Разреженный векторный поиск

Для поиска только с разреженными векторами:

- Параметр `retrieval_mode` должен быть установлен в `RetrievalMode.SPARSE`.
- Реализация интерфейса [`SparseEmbeddings`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/qdrant/langchain_qdrant/sparse_embeddings.py) с использованием любого провайдера разреженных эмбеддингов должна быть передана в параметр `sparse_embedding`.

Пакет `langchain-qdrant` предоставляет реализацию на основе [FastEmbed](https://github.com/qdrant/fastembed) «из коробки».

Для её использования установите пакет FastEmbed.

```python
pip install -qU fastembed
```

```python
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Create a Qdrant client for local storage
client = QdrantClient(path="/tmp/langchain_qdrant")

# Create a collection with sparse vectors
client.create_collection(
    collection_name="my_documents",
    vectors_config={"dense": VectorParams(size=3072, distance=Distance.COSINE)},
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
    },
)

qdrant = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.SPARSE,
    sparse_vector_name="sparse",
)

qdrant.add_documents(documents=documents, ids=uuids)

query = "How much money did the robbers steal?"
found_docs = qdrant.similarity_search(query)
found_docs
```

### Гибридный векторный поиск

Для выполнения гибридного поиска с использованием плотных и разреженных векторов с объединением оценок:

- Параметр `retrieval_mode` должен быть установлен в `RetrievalMode.HYBRID`.
- Значение [плотных эмбеддингов](https://python.langchain.com/docs/integrations/text_embedding/) должно быть передано в параметр `embedding`.
- Реализация интерфейса [`SparseEmbeddings`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/qdrant/langchain_qdrant/sparse_embeddings.py) с использованием любого провайдера разреженных эмбеддингов должна быть передана в параметр `sparse_embedding`.

Обратите внимание: если вы добавили документы в режиме `HYBRID`, вы можете переключаться на любой режим поиска, так как в коллекции доступны как плотные, так и разреженные векторы.

```python
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Create a Qdrant client for local storage
client = QdrantClient(path="/tmp/langchain_qdrant")

# Create a collection with both dense and sparse vectors
client.create_collection(
    collection_name="my_documents",
    vectors_config={"dense": VectorParams(size=3072, distance=Distance.COSINE)},
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
    },
)

qdrant = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)

qdrant.add_documents(documents=documents, ids=uuids)

query = "How much money did the robbers steal?"
found_docs = qdrant.similarity_search(query)
found_docs
```

Если вы хотите выполнить поиск по сходству и получить соответствующие оценки, вы можете запустить:

```python
results = vector_store.similarity_search_with_score(
    query="Will it be hot tomorrow", k=1
)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
```

```output
* [SIM=0.531834] The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees. [{'source': 'news', '_id': '9e6ba50c-794f-4b88-94e5-411f15052a02', '_collection_name': 'demo_collection'}]
```

Полный список всех функций поиска, доступных для `QdrantVectorStore`, см. в [справочнике API](https://python.langchain.com/api_reference/qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html).

### Фильтрация по метаданным

Qdrant имеет [расширенную систему фильтрации](https://qdrant.tech/documentation/concepts/filtering/) с поддержкой богатых типов. В LangChain также можно использовать фильтры, передавая дополнительный параметр в методы `similarity_search_with_score` и `similarity_search`.

```python
from qdrant_client import models

results = vector_store.similarity_search(
    query="Who are the best soccer players in the world?",
    k=1,
    filter=models.Filter(
        should=[
            models.FieldCondition(
                key="page_content",
                match=models.MatchValue(
                    value="The top 10 soccer players in the world right now."
                ),
            ),
        ]
    ),
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")
```

```output
* The top 10 soccer players in the world right now. [{'source': 'website', '_id': 'b0964ab5-5a14-47b4-a983-37fa5c5bd154', '_collection_name': 'demo_collection'}]
```

### Запрос через преобразование в retriever

Вы также можете преобразовать векторное хранилище в retriever для удобного использования в цепочках.

```python
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
retriever.invoke("Stealing from the bank is a crime")
```

```output
[Document(metadata={'source': 'news', '_id': '50d8d6ee-69bf-4173-a6a2-b254e9928965', '_collection_name': 'demo_collection'}, page_content='Robbers broke into the city bank and stole $1 million in cash.')]
```

## Использование для генерации с извлечением (RAG)

Руководства по использованию этого векторного хранилища для генерации с извлечением (retrieval-augmented generation, RAG) см. в следующих разделах:

- [Руководства](/oss/python/langchain/rag)
- [How-to: Вопросы и ответы с RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Концептуальные документы по извлечению](https://python.langchain.com/docs/concepts/retrieval)

## Настройка Qdrant

Существуют варианты использования существующей коллекции Qdrant в приложении LangChain. В таких случаях может потребоваться определить, как сопоставлять точку Qdrant с объектом LangChain `Document`.

### Именованные векторы

Qdrant поддерживает [несколько векторов на одну точку](https://qdrant.tech/documentation/concepts/collections/#collection-with-multiple-vectors) с помощью именованных векторов. Если вы работаете с коллекцией, созданной извне, или хотите использовать вектор с другим именем, вы можете настроить его, указав имя.

```python
from langchain_qdrant import RetrievalMode

QdrantVectorStore.from_documents(
    docs,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    location=":memory:",
    collection_name="my_documents_2",
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="custom_vector",
    sparse_vector_name="custom_sparse_vector",
)
```

### Метаданные

Qdrant хранит векторные эмбеддинги вместе с необязательным JSON-подобным payload. Payload необязателен, но поскольку LangChain предполагает, что эмбеддинги генерируются из документов, мы сохраняем контекстные данные, чтобы можно было извлечь исходные тексты.

По умолчанию ваш документ будет сохранен в следующей структуре payload:

```json
{
  "page_content": "Lorem ipsum dolor sit amet",
  "metadata": {
    "foo": "bar"
  }
}
```

Однако вы можете использовать другие ключи для содержимого страницы и метаданных. Это полезно, если у вас уже есть коллекция, которую вы хотите повторно использовать.

```python
QdrantVectorStore.from_documents(
    docs,
    embeddings,
    location=":memory:",
    collection_name="my_documents_2",
    content_payload_key="my_page_content_key",
    metadata_payload_key="my_meta",
)
```

---

## Справочник API

Подробная документация по всем функциям и конфигурациям `QdrantVectorStore` доступна в справочнике API: [python.langchain.com/api_reference/qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html](https://python.langchain.com/api_reference/qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html)

---

Source: https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant
