---
title: "Text splitters"
topic: "Modules"
filename: "text-splitters.md"
source: "https://docs.langchain.com/oss/python/integrations/splitters"
author: "Перевод GPT"
date: "2025-11-10"
---

# Text splitters

**Text splitters** разбивают большие документы на более мелкие фрагменты, которые можно извлекать по отдельности и которые укладываются в ограничение контекстного окна модели.

Существует несколько стратегий разбиения документов, каждая из которых имеет свои преимущества.

<Tip>
  Для большинства случаев использования начните с [RecursiveCharacterTextSplitter](/oss/python/integrations/splitters/recursive_text_splitter). Он обеспечивает хорошее соотношение между сохранением контекста и управлением размером фрагмента. Эта стратегия по умолчанию хорошо работает «из коробки», и вы должны рассматривать её изменение только в том случае, если вам нужно точно настроить производительность для вашего конкретного приложения.
</Tip>

## На основе структуры текста

Текст естественным образом организован в иерархические единицы, такие как абзацы, предложения и слова. Мы можем использовать эту врождённую структуру для определения стратегии разбиения, создавая фрагменты, которые сохраняют естественный языковой поток, семантическую целостность внутри фрагмента и адаптируются к различным уровням детализации текста. `RecursiveCharacterTextSplitter` из LangChain реализует эту концепцию:

- [RecursiveCharacterTextSplitter](/oss/python/integrations/splitters/recursive_text_splitter) пытается сохранить более крупные единицы (например, абзацы) нетронутыми.
- Если единица превышает размер фрагмента, он переходит на следующий уровень (например, предложения).
- Этот процесс продолжается вплоть до уровня слов, если это необходимо.

Пример использования:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_text(document)
```

**Доступные text splitters**:

- [Recursively split text](/oss/python/integrations/splitters/recursive_text_splitter)

## На основе длины

Интуитивная стратегия — разбивать документы на основе их длины. Этот простой, но эффективный подход гарантирует, что каждый фрагмент не превышает заданного ограничения по размеру. Основные преимущества разбиения на основе длины:

- Простота реализации
- Последовательные размеры фрагментов
- Лёгкая адаптация под требования различных моделей

Типы разбиения на основе длины:

- На основе токенов: разбиение текста по количеству токенов, что полезно при работе с языковыми моделями.
- На основе символов: разбиение текста по количеству символов, что может быть более последовательным для разных типов текста.

Пример реализации с использованием `CharacterTextSplitter` из LangChain с разбиением по токенам:

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(document)
```

**Доступные text splitters**:

- [Split by tokens](/oss/python/integrations/splitters/split_by_token)
- [Split by characters](/oss/python/integrations/splitters/character_text_splitter)

## На основе структуры документа

Некоторые документы имеют встроенную структуру, например HTML-, Markdown- или JSON-файлы. В таких случаях выгодно разбивать документ в соответствии с его структурой, так как она часто естественным образом группирует семантически связанный текст. Основные преимущества разбиения на основе структуры:

- Сохранение логической организации документа
- Поддержание контекста внутри каждого фрагмента
- Может быть эффективнее для последующих задач, таких как поиск или суммаризация

Примеры разбиения на основе структуры:

- Markdown: разбиение по заголовкам (например, `#`, `##`, `###`)
- HTML: разбиение по тегам
- JSON: разбиение по элементам объекта или массива
- Код: разбиение по функциям, классам или логическим блокам

**Доступные text splitters**:

- [Split Markdown](/oss/python/integrations/splitters/markdown_header_metadata_splitter)
- [Split JSON](/oss/python/integrations/splitters/recursive_json_splitter)
- [Split code](/oss/python/integrations/splitters/code_splitter)
- [Split HTML](/oss/python/integrations/splitters/split_html)

## Специфичные для провайдера

<Columns cols={3}>
  <Card title="WRITER" icon="link" href="/oss/python/integrations/splitters/writer" arrow="true" cta="View guide" />
</Columns>

---

Source: https://docs.langchain.com/oss/python/integrations/splitters
