---
title: "OpenAI"
topic: "integrations"
filename: "openai.md"
source: "https://python.langchain.com/docs/integrations/providers/openai/"
author: "Перевод GPT"
date: "2025-09-14"
---

# OpenAI

Вся функциональность, связанная с OpenAI

> **OpenAI** — американская лаборатория исследований в области искусственного интеллекта, состоящая из некоммерческой организации OpenAI Incorporated и её коммерческого подразделения OpenAI Limited Partnership. OpenAI проводит исследования в области ИИ с декларированной целью продвижения и разработки дружелюбного ИИ. Системы OpenAI работают на платформе суперкомпьютеров Azure от Microsoft.
> API **OpenAI** (OpenAI API) работает на множестве моделей с различными возможностями и ценами.
> **ChatGPT** — чат-бот на базе ИИ, разработанный OpenAI.

---

## Установка и настройка

Установите пакет интеграции:

```bash
pip install langchain-openai
```

Получите API-ключ OpenAI и установите его как переменную окружения (`OPENAI_API_KEY`) ([python.langchain.com][1])

---

## Chat модель

См. пример использования: [usage example](https://python.langchain.com/docs/integrations/providers/openai/#usage-example) ([python.langchain.com][1])

```python
from langchain_openai import ChatOpenAI
```

Если вы используете модель, размещённую на Azure, следует использовать другой обёртку:

```python
from langchain_openai import AzureChatOpenAI
```

Для подробного объяснения использования обёртки `Azure` см. [here](https://python.langchain.com/docs/integrations/providers/openai/#azure) ([python.langchain.com][1])

---

## LLM

См. пример использования: [usage example](https://python.langchain.com/docs/integrations/providers/openai/#usage-example) ([python.langchain.com][1])

```python
from langchain_openai import OpenAI
```

Если модель размещена на `Azure`, используйте:

```python
from langchain_openai import AzureOpenAI
```

Для детального руководства по обёртке `Azure` см. [here](https://python.langchain.com/docs/integrations/providers/openai/#azure) ([python.langchain.com][1])

---

## Модель эмбедингов (Embedding Model)

См. пример использования: [usage example](https://python.langchain.com/docs/integrations/providers/openai/#usage-example) ([python.langchain.com][1])

```python
from langchain_openai import OpenAIEmbeddings
```

---

## Загрузчик документов (Document Loader)

См. пример использования: [usage example](https://python.langchain.com/docs/integrations/providers/openai/#usage-example) ([python.langchain.com][1])

```python
from langchain_community.document_loaders.chatgpt import ChatGPTLoader
```

---

## Извлекатель (Retriever)

См. пример использования: [usage example](https://python.langchain.com/docs/integrations/providers/openai/#usage-example) ([python.langchain.com][1])

```python
from langchain.retrievers import ChatGPTPluginRetriever
```

---

## Инструменты (Tools)

### Генератор изображений Dall-E

> **OpenAI Dall-E** — модели text-to-image, разработанные OpenAI с использованием методов глубокого обучения, для генерации цифровых изображений по описанию на естественном языке, называемым «prompt». ([python.langchain.com][1])

См. пример использования: [usage example](https://python.langchain.com/docs/integrations/providers/openai/#usage-example) ([python.langchain.com][1])

```python
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
```

---

## Адаптер (Adapter)

См. пример использования: [usage example](https://python.langchain.com/docs/integrations/providers/openai/#usage-example) ([python.langchain.com][1])

```python
from langchain.adapters import openai as lc_openai
```

---

## Токенизатор

Есть несколько мест, где можно использовать токенизатор `tiktoken`. По умолчанию он используется для подсчёта токенов для OpenAI LLMs. ([python.langchain.com][1])

Вы можете также использовать его для подсчёта токенов при разбиении документов:

```python
from langchain.text_splitter import CharacterTextSplitter
CharacterTextSplitter.from_tiktoken_encoder(...)
```

Для более детального примера смотрите [this notebook](https://python.langchain.com/docs/integrations/providers/openai/#notebook) ([python.langchain.com][1])

---

## Цепочка (Chain)

См. пример использования: [usage example](https://python.langchain.com/docs/integrations/providers/openai/#usage-example) ([python.langchain.com][1])

```python
from langchain.chains import OpenAIModerationChain
```

---

Source: [https://python.langchain.com/docs/integrations/providers/openai/](https://python.langchain.com/docs/integrations/providers/openai/)

[1]: https://python.langchain.com/docs/integrations/providers/openai/ "OpenAI | ️ LangChain"
