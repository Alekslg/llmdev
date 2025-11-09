---
title: "Embedding models"
topic: "Modules"
filename: "text-embedding.md"
source: "https://docs.langchain.com/oss/python/integrations/text_embedding"
author: "Перевод GPT"
date: "2025-11-10"
---

## Обзор

> Это описание охватывает **текстовые эмбеддинг-модели**. LangChain в настоящее время не поддерживает мультимодальные эмбеддинги.

Эмбеддинг-модели преобразуют необработанный текст — такой как предложение, абзац или твит — в вектор фиксированной длины, состоящий из чисел и отражающий его **семантический смысл**. Эти векторы позволяют машинам сравнивать и искать текст на основе смысла, а не точного совпадения слов.

На практике это означает, что тексты с похожими идеями располагаются близко друг к другу в векторном пространстве. Например, вместо поиска только по фразе _«machine learning»_, эмбеддинги могут находить документы, обсуждающие связанные концепции, даже если используется другая формулировка.

### Как это работает

1. **Векторизация** — модель кодирует каждую входную строку как многомерный вектор.
2. **Оценка схожести** — векторы сравниваются с использованием математических метрик для измерения степени близости смыслов исходных текстов.

### Метрики схожести

Для сравнения эмбеддингов обычно используются следующие метрики:

- **Косинусное сходство** (_Cosine similarity_) — измеряет угол между двумя векторами.
- **Евклидово расстояние** (_Euclidean distance_) — измеряет прямолинейное расстояние между точками.
- **Скалярное произведение** (_Dot product_) — измеряет, насколько один вектор проецируется на другой.

Вот пример вычисления косинусного сходства между двумя векторами:

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    return dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarity = cosine_similarity(query_embedding, document_embedding)
print("Cosine Similarity:", similarity)
```

## Интерфейс эмбеддингов в LangChain

LangChain предоставляет стандартный интерфейс для текстовых эмбеддинг-моделей (например, OpenAI, Cohere, Hugging Face) через интерфейс [Embeddings](https://reference.langchain.com/python/langchain_core/embeddings/#langchain_core.embeddings.embeddings.Embeddings).

Доступны два основных метода:

- `embed_documents(texts: List[str]) → List[List[float]]`: создаёт эмбеддинги для списка документов.
- `embed_query(text: str) → List[float]`: создаёт эмбеддинг для одного запроса.

> Интерфейс позволяет использовать разные стратегии для эмбеддингов запросов и документов, хотя большинство провайдеров на практике обрабатывают их одинаково.

## Популярные интеграции

| Provider                                                                      | Package                                                                                                                                                            |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [OpenAI](/oss/python/integrations/text_embedding/openai)                      | [`langchain-openai`](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)                              |
| [OpenAI on Azure](/oss/python/integrations/text_embedding/azure_openai)       | [`langchain-openai`](https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.azure.AzureOpenAIEmbeddings.html)                    |
| [Google Gemini](/oss/python/integrations/text_embedding/google_generative_ai) | [`langchain-google-genai`](https://python.langchain.com/api_reference/google_genai/embeddings/langchain_google_genai.embeddings.GoogleGenerativeAIEmbeddings.html) |
| [Ollama](/oss/python/integrations/text_embedding/ollama)                      | [`langchain-ollama`](https://python.langchain.com/api_reference/ollama/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html)                               |
| [Together](/oss/python/integrations/text_embedding/together)                  | [`langchain-together`](https://python.langchain.com/api_reference/together/embeddings/langchain_together.embeddings.TogetherEmbeddings.html)                       |
| [Fireworks](/oss/python/integrations/text_embedding/fireworks)                | [`langchain-fireworks`](https://python.langchain.com/api_reference/fireworks/embeddings/langchain_fireworks.embeddings.FireworksEmbeddings.html)                   |
| [MistralAI](/oss/python/integrations/text_embedding/mistralai)                | [`langchain-mistralai`](https://python.langchain.com/api_reference/mistralai/embeddings/langchain_mistralai.embeddings.MistralAIEmbeddings.html)                   |
| [Cohere](/oss/python/integrations/text_embedding/cohere)                      | [`langchain-cohere`](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.cohere.Cohere.html)                                        |
| [Nomic](/oss/python/integrations/text_embedding/nomic)                        | [`langchain-nomic`](https://python.langchain.com/api_reference/nomic/embeddings/langchain_nomic.embeddings.NomicEmbeddings.html)                                   |
| [Fake](/oss/python/integrations/text_embedding/fake)                          | [`langchain-core`](https://python.langchain.com/api_reference/core/embeddings/langchain_core.embeddings.fake.FakeEmbeddings.html)                                  |
| [Databricks](/oss/python/integrations/text_embedding/databricks)              | [`databricks-langchain`](https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_langchain.html#databricks_langchain.DatabricksEmbeddings)   |
| [IBM](/oss/python/integrations/text_embedding/ibm_watsonx)                    | [`langchain-ibm`](https://python.langchain.com/api_reference/ibm/embeddings/langchain_ibm.embeddings.WatsonxEmbeddings.html)                                       |
| [NVIDIA](/oss/python/integrations/text_embedding/nvidia_ai_endpoints)         | [`langchain-nvidia`](https://python.langchain.com/api_reference/nvidia_ai_endpoints/embeddings/langchain_nvidia_ai_endpoints.embeddings.NVIDIAEmbeddings.html)     |
| [AI/ML API](/oss/python/integrations/text_embedding/aimlapi)                  | [`langchain-aimlapi`](https://python.langchain.com/api_reference/aimlapi/embeddings/langchain_aimlapi.embeddings.AimlapiEmbeddings.html)                           |

## Кэширование

Эмбеддинги можно сохранять или временно кэшировать, чтобы избежать их повторного вычисления.

Кэширование эмбеддингов осуществляется с помощью `CacheBackedEmbeddings`. Эта обёртка сохраняет эмбеддинги в хранилище «ключ-значение», где текст хэшируется, а хэш используется в качестве ключа в кэше.

Основной поддерживаемый способ инициализации `CacheBackedEmbeddings` — метод `from_bytes_store`. Он принимает следующие параметры:

- **`underlying_embedder`**: эмбеддер, используемый для создания эмбеддингов.
- **`document_embedding_cache`**: любой [ByteStore](/oss/python/integrations/stores) для кэширования эмбеддингов документов.
- **`batch_size`**: (необязательный, по умолчанию `None`) количество документов для обработки между обновлениями хранилища.
- **`namespace`**: (необязательный, по умолчанию `""`) пространство имён для кэша документов. Помогает избежать коллизий (например, можно установить его равным имени модели эмбеддингов).
- **`query_embedding_cache`**: (необязательный, по умолчанию `None`) [ByteStore](/oss/python/integrations/stores) для кэширования эмбеддингов запросов или значение `True` для повторного использования того же хранилища, что и `document_embedding_cache`.

```python
import time
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

# Создайте базовую модель эмбеддингов
underlying_embeddings = ...  # например, OpenAIEmbeddings(), HuggingFaceEmbeddings() и т.д.

# Хранилище сохраняет эмбеддинги в локальной файловой системе
# Это не для продакшена, но удобно для локальной разработки
store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model
)

# Пример: кэширование эмбеддинга запроса
tic = time.time()
print(cached_embedder.embed_query("Hello, world!"))
print(f"First call took: {time.time() - tic:.2f} seconds")

# Последующие вызовы используют кэш
tic = time.time()
print(cached_embedder.embed_query("Hello, world!"))
print(f"Second call took: {time.time() - tic:.2f} seconds")
```

В продакшене обычно используют более надёжное постоянное хранилище, например базу данных или облачное хранилище. См. [интеграции хранилищ](/oss/python/integrations/stores).

Source: https://docs.langchain.com/oss/python/integrations/text_embedding
