---
title: "Введение"
topic: "Getting Started"
filename: "introduction.md"
source: "https://python.langchain.com/docs/introduction/"
author: "Перевод GPT"
date: "2025-09-11"
---

# Введение

**LangChain** — это фреймворк для разработки приложений, работающих на основе large language models (LLMs).

LangChain упрощает каждый этап жизненного цикла приложения на базе LLM:

- **Разработка (Development)**: Создавайте свои приложения с помощью open-source компонентов LangChain и интеграций со сторонними сервисами. Используйте LangGraph для построения stateful-агентов с нативной поддержкой streaming и human-in-the-loop.
- **Подготовка к продакшену (Productionization)**: Используйте LangSmith для инспектирования, мониторинга и оценки ваших приложений, чтобы вы могли постоянно оптимизировать их и уверенно разворачивать.
- **Развертывание (Deployment)**: Превращайте ваши приложения LangGraph в production-ready API и ассистенты с помощью LangGraph Platform.

![LangChain Framework Overview Diagram outlining the hierarchical organization of the LangChain framework, displaying the interconnected parts across multiple layers.](../assets/images/basics/introduction/getting-started/langchain_stack_112024_dark.svg)

LangChain реализует стандартный интерфейс для large language models и связанных технологий, таких как embedding models и vector stores, и интегрируется с сотнями провайдеров. Подробнее см. на странице [integrations](https://python.langchain.com/docs/integrations/providers/).

```bash
pip install -qU "langchain[google-genai]"
```

```python
import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
```

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
```

```python
model.invoke("Hello, world!")
```

Эта документация фокусируется на Python-библиотеке LangChain. Перейдите сюда для документации по JavaScript-библиотеке LangChain.

## Архитектура

Фреймворк LangChain состоит из нескольких open-source библиотек. Подробнее читайте на странице [Architecture](https://python.langchain.com/docs/architecture/).

- `langchain-core`
- **Пакеты интеграций** (например, `langchain-openai`, `langchain-anthropic` и т.д.): Ключевые интеграции были выделены в легковесные пакеты, которые совместно поддерживаются командой LangChain и разработчиками интеграций.
- `langchain`
- `langchain-community`
- `langgraph`

## Руководства

### Учебные пособия (Tutorials)

Если вы хотите создать что-то конкретное или предпочитаете обучаться на практике, ознакомьтесь с нашим разделом [tutorials](https://python.langchain.com/docs/tutorials/). Это лучшее место для начала.

Вот лучшие руководства, с которых стоит начать:

Ознакомьтесь со [всем списком учебных пособий LangChain здесь](https://python.langchain.com/docs/tutorials/), а также с другими [учебниками по LangGraph здесь](https://python.langchain.com/docs/tutorials/). Чтобы глубже изучить LangGraph, пройдите наш первый курс LangChain Academy, _Introduction to LangGraph_, доступный [здесь](https://learn.langchain.com/courses/introduction-to-langgraph).

### How-to руководства

Здесь вы найдете краткие ответы на вопросы типа «Как мне…?». Эти how-to руководства не охватывают темы глубоко — такую информацию вы найдете в разделах Tutorials и API Reference. Однако эти руководства помогут вам быстро выполнить распространенные задачи с использованием chat models, vector stores и других общих компонентов LangChain.

Ознакомьтесь с [how-to, специфичными для LangGraph, здесь](https://python.langchain.com/docs/how_to/).

### Концептуальные руководства

Знакомство со всеми ключевыми частями LangChain, которые вам необходимо знать! Здесь вы найдете высокоуровневые объяснения всех концепций LangChain.

Для более глубокого погружения в концепции LangGraph ознакомьтесь с [этой страницей](https://python.langchain.com/docs/concepts/langgraph/).

### Интеграции

LangChain является частью богатой экосистемы инструментов, которые интегрируются с нашим фреймворком и строятся поверх него. Если вы хотите быстро начать работу с chat models, vector stores или другими компонентами LangChain от конкретного провайдера, ознакомьтесь с нашим растущим списком [integrations](https://python.langchain.com/docs/integrations/).

### Справочник по API

Перейдите в раздел [reference](https://api.python.langchain.com/), чтобы ознакомиться с полной документацией по всем классам и методам в Python-пакетах LangChain.

## Экосистема

### ️ LangSmith

Трассировка и оценка ваших приложений на основе языковых моделей и интеллектуальных агентов, чтобы помочь вам перейти от прототипа к продакшену.

### ️ LangGraph

Создание stateful, multi-actor приложений с LLM. Плавно интегрируется с LangChain, но может использоваться и без него. LangGraph обеспечивает работу production-grade агентов, которым доверяют LinkedIn, Uber, Klarna, GitLab и многие другие.

## Дополнительные ресурсы

### Версии

Узнайте, что изменилось в v0.3, как перенести legacy-код, ознакомьтесь с нашей политикой версионирования и многое другое.

### Безопасность

Ознакомьтесь с лучшими практиками безопасности, чтобы убедиться, что вы безопасно разрабатываете с использованием LangChain.

### Вклад в проект

Ознакомьтесь с [руководством для разработчиков](https://python.langchain.com/docs/contributing/) с инструкциями по внесению вклада и настройке вашей среды разработки.

Source: https://python.langchain.com/docs/introduction/
