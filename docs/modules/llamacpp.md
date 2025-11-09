---
title: "Llama.cpp"
topic: "Modules"
filename: "llamacpp.md"
source: "https://docs.langchain.com/oss/python/integrations/text_embedding/llamacpp"
author: "Перевод GPT"
date: "2025-11-10"
---

Библиотека Python [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) предоставляет простую обёртку на Python для [`llama.cpp`](https://github.com/ggerganov/llama.cpp) от `@ggerganov`.

Этот пакет поддерживает:

- Низкоуровневый доступ к C API через интерфейс ctypes.
- Высокоуровневый Python API для генерации текста:
  - API, совместимый с `OpenAI`,
  - Совместимость с `LangChain`,
  - Совместимость с `LlamaIndex`.
- Веб-сервер, совместимый с OpenAI:
  - Альтернатива локальному Copilot,
  - Поддержка вызова функций (Function Calling),
  - Поддержка Vision API,
  - Поддержка нескольких моделей.

## Установка

```bash
pip install -qU llama-cpp-python
```

## Быстрый старт

```python
from langchain_community.embeddings import LlamaCppEmbeddings
```

```python
llama = LlamaCppEmbeddings(model_path="/path/to/model/ggml-model-q4_0.bin")
```

```python
text = "This is a test document."
```

```python
query_result = llama.embed_query(text)
```

```python
doc_result = llama.embed_documents([text])
```

> Убедитесь, что ваша модель поддерживает генерацию эмбеддингов. Не все GGUF-модели подходят для этой задачи — используйте специализированные эмбеддинг-модели, такие как `nomic-embed-text` или `bge-small`.

Source: https://docs.langchain.com/oss/python/integrations/text_embedding/llamacpp
