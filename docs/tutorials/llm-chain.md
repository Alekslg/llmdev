---
title: "Создайте простое приложение LLM с чат-моделями и шаблонами промптов"
topic: "tutorials"
filename: "llm-chain"
source: "https://python.langchain.com/docs/tutorials/llm_chain/"
author: "Перевод GPT"
date: "2025-04-05"
---

# Создайте простое приложение LLM с чат-моделями и шаблонами промптов

В этом быстром старте мы покажем вам, как создать простое приложение LLM с помощью LangChain. Это приложение будет переводить текст с английского на другой язык. Это относительно простое приложение LLM — это всего лишь один вызов LLM плюс некоторая настройка промпта. Тем не менее, это отличный способ начать работу с LangChain — множество функций можно построить, используя только настройку промпта и вызов LLM!

После прочтения этого руководства вы получите общее представление о:

- Использовании языковых моделей
- Использовании шаблонов промптов
- Отладке и трассировке вашего приложения с помощью LangSmith

Давайте начнем!

## Настройка

### Jupyter Notebook

Это и другие руководства, возможно, наиболее удобно запускать в Jupyter notebooks. Прохождение руководств в интерактивной среде — отличный способ лучше их понять. Инструкции по установке см. здесь.

### Установка

Для установки LangChain выполните:

- Pip
- Conda

```pip
pip install langchain
```

```conda
conda install langchain -c conda-forge
```

Более подробную информацию см. в нашем руководстве по установке.

### LangSmith

Многие приложения, которые вы создаете с помощью LangChain, будут содержать несколько шагов с несколькими вызовами LLM. По мере того как эти приложения становятся все более и более сложными, становится крайне важно иметь возможность проверять, что именно происходит внутри вашей цепочки или агента. Лучший способ сделать это — с помощью LangSmith.

После регистрации по ссылке выше убедитесь, что вы установили переменные среды для начала записи трассировок:

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
export LANGSMITH_PROJECT="default" # или любое другое имя проекта
```

Или, если вы находитесь в ноутбуке, вы можете установить их с помощью:

```python
import getpass
import os

try:
    # загрузить переменные среды из файла .env (требуется `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Введите ваш ключ API LangSmith (необязательно): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Введите имя вашего проекта LangSmith (по умолчанию = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"
```

## Использование языковых моделей

Прежде всего, давайте узнаем, как использовать языковую модель саму по себе. LangChain поддерживает множество различных языковых моделей, которые можно использовать взаимозаменяемо. Подробную информацию о начале работы с конкретной моделью см. в разделе поддерживаемых интеграций.

```pip
pip install -qU "langchain[google-genai]"
```

```python
import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Введите ключ API для Google Gemini: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
```

Давайте сначала воспользуемся моделью напрямую. ChatModels являются экземплярами LangChain Runnables, что означает, что они предоставляют стандартный интерфейс для взаимодействия с ними. Чтобы просто вызвать модель, мы можем передать список сообщений методу `.invoke`.

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

model.invoke(messages)
```

```python
AIMessage(content='Ciao!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 3, 'prompt_tokens': 20, 'total_tokens': 23, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0705bf87c0', 'finish_reason': 'stop', 'logprobs': None}, id='run-32654a56-627c-40e1-a141-ad9350bbfd3e-0', usage_metadata={'input_tokens': 20, 'output_tokens': 3, 'total_tokens': 23, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

Если мы включили LangSmith, мы можем увидеть, что этот запуск регистрируется в LangSmith, и можем увидеть трассировку LangSmith. Трассировка LangSmith сообщает информацию об использовании токенов, задержке, стандартных параметрах модели (например, температуре) и другой информации.

Обратите внимание, что ChatModels получают на вход объекты сообщений и генерируют на выходе объекты сообщений. Помимо текстового содержимого, объекты сообщений передают роли в диалоге и содержат важные данные, такие как вызовы инструментов и количество использованных токенов.

LangChain также поддерживает входные данные чат-модели через строки или формат OpenAI. Следующие варианты эквивалентны:

```python
model.invoke("Hello")
```

```python
model.invoke([{"role": "user", "content": "Hello"}])
```

```python
model.invoke([HumanMessage("Hello")])
```

### Потоковая передача

Поскольку чат-модели являются Runnables, они предоставляют стандартный интерфейс, который включает асинхронные и потоковые режимы вызова. Это позволяет нам передавать отдельные токены из чат-модели в потоковом режиме:

```python
for token in model.stream(messages):
    print(token.content, end="|")
```

```python
|C|iao|!||
```

Более подробную информацию о потоковой передаче вывода чат-модели см. в этом руководстве.

## Шаблоны промптов

Сейчас мы передаем список сообщений напрямую в языковую модель. Откуда берется этот список сообщений? Обычно он формируется из комбинации пользовательского ввода и логики приложения. Эта логика приложения обычно берет необработанный пользовательский ввод и преобразует его в список сообщений, готовых для передачи в языковую модель. Распространенные преобразования включают добавление системного сообщения или форматирование шаблона с пользовательским вводом.

Шаблоны промптов — это концепция в LangChain, предназначенная для помощи в этом преобразовании. Они принимают необработанный пользовательский ввод и возвращают данные (промпт), готовые для передачи в языковую модель.

Давайте создадим здесь шаблон промпта. Он будет принимать две пользовательские переменные:

- `language`: Язык, на который нужно перевести текст
- `text`: Текст для перевода

```python
from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
```

**Справочник по API:** [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.ChatPromptTemplate.html)

Обратите внимание, что `ChatPromptTemplate` поддерживает несколько ролей сообщений в одном шаблоне. Мы форматируем параметр `language` в системное сообщение, а пользовательский `text` — в сообщение пользователя.

Входными данными для этого шаблона промпта является словарь. Мы можем поэкспериментировать с этим шаблоном промпта самостоятельно, чтобы увидеть, что он делает сам по себе.

```python
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

prompt
```

```python
ChatPromptValue(messages=[SystemMessage(content='Translate the following from English into Italian', additional_kwargs={}, response_metadata={}), HumanMessage(content='hi!', additional_kwargs={}, response_metadata={})])
```

Мы видим, что он возвращает `ChatPromptValue`, состоящий из двух сообщений. Если мы хотим получить доступ к сообщениям напрямую, мы делаем:

```python
prompt.to_messages()
```

```python
[SystemMessage(content='Translate the following from English into Italian', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='hi!', additional_kwargs={}, response_metadata={})]
```

Наконец, мы можем вызвать чат-модель на отформатированном промпте:

```python
response = model.invoke(prompt)
print(response.content)
```

```python
Ciao!
```

Содержимое `content` сообщения может содержать как текст, так и блоки контента с дополнительной структурой. Дополнительную информацию см. в этом руководстве.

Если мы посмотрим на трассировку LangSmith, мы сможем точно увидеть, какой промпт получает чат-модель, а также информацию об использовании токенов, задержке, стандартных параметрах модели (например, температуре) и другой информации.

## Заключение

Вот и все! В этом руководстве вы узнали, как создать свое первое простое приложение LLM. Вы узнали, как работать с языковыми моделями, как создавать шаблон промпта и как получать отличную наблюдаемость за приложениями, которые вы создаете с помощью LangSmith.

Это лишь поверхностное знакомство с тем, что вам нужно будет изучить, чтобы стать опытным AI Engineer. К счастью, у нас есть много других ресурсов!

Для дальнейшего чтения по основным концепциям LangChain у нас есть подробные Conceptual Guides.

Если у вас есть более конкретные вопросы по этим концепциям, ознакомьтесь со следующими разделами руководств how-to:

И документацией LangSmith:

Source: https://python.langchain.com/docs/tutorials/llm_chain/
