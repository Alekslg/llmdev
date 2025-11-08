---
title: "Создайте чат-бота"
topic: "tutorials"
filename: "chatbot"
source: "https://python.langchain.com/docs/tutorials/chatbot/"
author: "Перевод GPT"
date: "2025-04-05"
---

# Создайте чат-бота

Это руководство ранее использовало абстракцию `RunnableWithMessageHistory`. Вы можете получить доступ к той версии документации в документации v0.2.

Начиная с выпуска LangChain v0.3, мы рекомендуем пользователям LangChain использовать персистентность LangGraph для включения `memory` (памяти) в новые приложения LangChain.

Если ваш код уже использует `RunnableWithMessageHistory` или `BaseChatMessageHistory`, вам **не** нужно вносить какие-либо изменения. Мы не планируем отказываться от этой функциональности в ближайшем будущем, поскольку она работает для простых чат-приложений, и любой код, использующий `RunnableWithMessageHistory`, будет продолжать работать как ожидалось.

Пожалуйста, смотрите раздел [How to migrate to LangGraph Memory](https://python.langchain.com/docs/how_to/migrate_to_langgraph_memory/) для получения более подробной информации.

## Обзор

Мы рассмотрим пример того, как спроектировать и реализовать чат-бота на основе LLM. Этот чат-бот сможет вести беседу и запоминать предыдущие взаимодействия с чат-моделью.

Обратите внимание, что этот чат-бот, который мы создадим, будет использовать языковую модель только для ведения беседы. Существует несколько других связанных концепций, которые могут вас интересовать:

- Conversational RAG: Включить опыт чат-бота над внешним источником данных
- Agents: Создать чат-бота, который может выполнять действия

Это руководство охватывает основы, которые будут полезны для освоения этих двух более продвинутых тем, но вы можете сразу перейти к ним, если пожелаете.

## Настройка

### Jupyter Notebook

Это руководство (и большинство других руководств в документации) использует Jupyter notebooks и предполагает, что читатель также использует их. Jupyter notebooks идеально подходят для изучения работы с системами LLM, потому что часто что-то может пойти не так (неожиданный вывод, API недоступен и т.д.), и прохождение руководств в интерактивной среде — отличный способ лучше их понять.

Это и другие руководства, возможно, наиболее удобно запускать в Jupyter notebook. Инструкции по установке см. здесь.

### Установка

Для этого руководства нам понадобятся `langchain-core` и `langgraph`. Это руководство требует `langgraph >= 0.2.28`.

- Pip
- Conda

```pip
pip install langchain-core langgraph>0.2.27
```

```conda
conda install langchain-core langgraph>0.2.27 -c conda-forge
```

Для получения более подробной информации см. наше руководство по установке.

### LangSmith

Многие приложения, которые вы создаете с помощью LangChain, будут содержать несколько шагов с несколькими вызовами LLM. По мере того как эти приложения становятся все более и более сложными, становится крайне важно иметь возможность проверять, что именно происходит внутри вашей цепочки или агента. Лучший способ сделать это — с помощью LangSmith.

После регистрации по ссылке выше, **(вам нужно будет создать ключ API на странице Settings -> API Keys на веб-сайте LangSmith)**, убедитесь, что вы установили переменные среды для начала записи трассировок:

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

Или, если вы находитесь в ноутбуке, вы можете установить их с помощью:

```python
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

## Быстрый старт

Прежде всего, давайте узнаем, как использовать языковую модель саму по себе. LangChain поддерживает множество различных языковых моделей, которые можно использовать взаимозаменяемо — выберите ту, которую хотите использовать ниже!

```pip
pip install -qU "langchain[google-genai]"
```

```python
import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Введите ключ API для Google Gemini: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
```

Давайте сначала воспользуемся моделью напрямую. `ChatModel` являются экземплярами "Runnables" LangChain, что означает, что они предоставляют стандартный интерфейс для взаимодействия с ними. Чтобы просто вызвать модель, мы можем передать список сообщений методу `.invoke`.

```python
from langchain_core.messages import HumanMessage

model.invoke([HumanMessage(content="Hi! I'm Bob")])
```

**Справочник по API:** [HumanMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.HumanMessage.html)

```python
AIMessage(content='Hi Bob! How can I assist you today?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 11, 'total_tokens': 21, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0705bf87c0', 'finish_reason': 'stop', 'logprobs': None}, id='run-5211544f-da9f-4325-8b8e-b3d92b2fc71a-0', usage_metadata={'input_tokens': 11, 'output_tokens': 10, 'total_tokens': 21, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

Модель сама по себе не имеет никакого понятия о состоянии. Например, если вы зададите последующий вопрос:

```python
model.invoke([HumanMessage(content="What's my name?")])
```

```python
AIMessage(content="I'm sorry, but I don't have access to personal information about users unless it has been shared with me in the course of our conversation. How can I assist you today?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 11, 'total_tokens': 45, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0705bf87c0', 'finish_reason': 'stop', 'logprobs': None}, id='run-a2d13a18-7022-4784-b54f-f85c097d1075-0', usage_metadata={'input_tokens': 11, 'output_tokens': 34, 'total_tokens': 45, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

Давайте посмотрим на пример трассировки LangSmith.

Мы видим, что модель не учитывает предыдущий ход беседы и не может ответить на вопрос. Это создает ужасный опыт чат-бота!

Чтобы обойти это, нам нужно передать всю историю беседы в модель. Посмотрим, что произойдет, когда мы это сделаем:

```python
from langchain_core.messages import AIMessage

model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)
```

**Справочник по API:** [AIMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.AIMessage.html)

```python
AIMessage(content='Your name is Bob! How can I help you today, Bob?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 33, 'total_tokens': 47, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0705bf87c0', 'finish_reason': 'stop', 'logprobs': None}, id='run-34bcccb3-446e-42f2-b1de-52c09936c02c-0', usage_metadata={'input_tokens': 33, 'output_tokens': 14, 'total_tokens': 47, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

И теперь мы видим, что получаем хороший ответ!

Это основная идея, лежащая в основе способности чат-бота вести беседу. Так как же нам лучше всего это реализовать?

## Персистентность сообщений

LangGraph реализует встроенный слой персистентности, что делает его идеальным для чат-приложений, поддерживающих несколько ходов беседы.

Обертывание нашей чат-модели в минимальное приложение LangGraph позволяет нам автоматически сохранять историю сообщений, упрощая разработку приложений с несколькими ходами.

LangGraph поставляется с простым чекпоинтером в памяти, который мы используем ниже. См. его документацию для получения более подробной информации, включая использование различных бэкендов персистентности (например, SQLite или Postgres).

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Определите новый граф
workflow = StateGraph(state_schema=MessagesState)


# Определите функцию, которая вызывает модель
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


# Определите (единственный) узел в графе
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Добавьте память
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

Теперь нам нужно создать `config`, который мы передаем в runnable каждый раз. Этот config содержит информацию, которая не является частью входных данных напрямую, но все же полезна. В данном случае мы хотим включить `thread_id`. Это должно выглядеть так:

```python
config = {"configurable": {"thread_id": "abc123"}}
```

Это позволяет нам поддерживать несколько потоков беседы с одним приложением, что является распространенным требованием, когда у вашего приложения есть несколько пользователей.

Затем мы можем вызвать приложение:

```python
query = "Hi! I'm Bob."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()  # output содержит все сообщения в состоянии
```

```python
==================================[1m Ai Message [0m==================================

Hi Bob! How can I assist you today?
```

```python
query = "What's my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

```python
==================================[1m Ai Message [0m==================================

Your name is Bob! How can I help you today, Bob?
```

Отлично! Наш чат-бот теперь помнит о нас. Если мы изменим config, чтобы сослаться на другой `thread_id`, мы увидим, что он начинает беседу заново.

```python
config = {"configurable": {"thread_id": "abc234"}}

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

```python
==================================[1m Ai Message [0m==================================

I'm sorry, but I don't have access to personal information about you unless you've shared it in this conversation. How can I assist you today?
```

Однако мы всегда можем вернуться к исходной беседе (поскольку мы сохраняем ее в базе данных).

```python
config = {"configurable": {"thread_id": "abc123"}}

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

```python
==================================[1m Ai Message [0m==================================

Your name is Bob. What would you like to discuss today?
```

Вот как мы можем поддерживать чат-бота, ведущего беседы со многими пользователями!

Для асинхронной поддержки обновите узел `call_model`, чтобы он был асинхронной функцией, и используйте `.ainvoke` при вызове приложения:

```python
# Асинхронная функция для узла:
async def call_model(state: MessagesState):
    response = await model.ainvoke(state["messages"])
    return {"messages": response}


# Определите граф как раньше:
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=MemorySaver())

# Асинхронный вызов:
output = await app.ainvoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

На данный момент все, что мы сделали, — это добавили простой слой персистентности вокруг модели. Мы можем начать усложнять и персонализировать чат-бота, добавив шаблон промпта.

## Шаблоны промптов

Шаблоны промптов (Prompt Templates) помогают преобразовать необработанную пользовательскую информацию в формат, с которым может работать LLM. В данном случае необработанный пользовательский ввод — это просто сообщение, которое мы передаем LLM. Теперь давайте сделаем это немного сложнее. Во-первых, добавим системное сообщение с некоторыми пользовательскими инструкциями (но по-прежнему принимающее сообщения в качестве входных данных). Затем добавим больше входных данных, кроме самих сообщений.

Чтобы добавить системное сообщение, мы создадим `ChatPromptTemplate`. Мы будем использовать `MessagesPlaceholder`, чтобы передать все сообщения.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
```

Теперь мы можем обновить наше приложение, чтобы включить этот шаблон:

```python
workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

Мы вызываем приложение тем же способом:

```python
config = {"configurable": {"thread_id": "abc345"}}
query = "Hi! I'm Jim."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

```python
==================================[1m Ai Message [0m==================================

Ahoy there, Jim! What brings ye to these waters today? Be ye seekin' treasure, knowledge, or perhaps a good tale from the high seas? Arrr!
```

```python
query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

```python
==================================[1m Ai Message [0m==================================

Ye be called Jim, matey! A fine name fer a swashbuckler such as yerself! What else can I do fer ye? Arrr!
```

Отлично! Теперь давайте сделаем наш промпт немного сложнее. Предположим, что шаблон промпта теперь выглядит примерно так:

```python
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
```

Обратите внимание, что мы добавили новый входной параметр `language` в промпт. Наше приложение теперь имеет два параметра — входные данные `messages` и `language`. Мы должны обновить состояние нашего приложения, чтобы отразить это:

```python
from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


workflow = StateGraph(state_schema=State)


def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

```python
config = {"configurable": {"thread_id": "abc456"}}
query = "Hi! I'm Bob."
language = "Spanish"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()
```

```python
==================================[1m Ai Message [0m==================================

¡Hola, Bob! ¿Cómo puedo ayudarte hoy?
```

Обратите внимание, что все состояние сохраняется, поэтому мы можем опустить параметры, такие как `language`, если не требуется вносить изменения:

```python
query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages},
    config,
)
output["messages"][-1].pretty_print()
```

```python
==================================[1m Ai Message [0m==================================

Tu nombre es Bob. ¿Hay algo más en lo que pueda ayudarte?
```

Чтобы лучше понять, что происходит внутри, ознакомьтесь с этой трассировкой LangSmith.

## Управление историей беседы

Одна важная концепция, которую необходимо понимать при создании чат-ботов, — это управление историей беседы. Если ею не управлять, список сообщений будет расти неограниченно и потенциально может превысить окно контекста LLM. Поэтому важно добавить шаг, который ограничивает размер передаваемых сообщений.

**Важно: вы захотите сделать это ДО шаблона промпта, но ПОСЛЕ загрузки предыдущих сообщений из истории сообщений.**

Мы можем сделать это, добавив простой шаг перед промптом, который соответствующим образом изменяет ключ `messages`, а затем обернуть эту новую цепочку в класс истории сообщений.

LangChain поставляется с несколькими встроенными помощниками для управления списком сообщений. В данном случае мы будем использовать помощник `trim_messages`, чтобы уменьшить количество сообщений, отправляемых модели. Триммер позволяет нам указать, сколько токенов мы хотим сохранить, а также другие параметры, например, всегда ли сохранять системное сообщение и разрешать ли частичные сообщения:

```python
from langchain_core.messages import SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages)
```

```python
[SystemMessage(content="you're a good assistant", additional_kwargs={}, response_metadata={}),
 HumanMessage(content='whats 2 + 2', additional_kwargs={}, response_metadata={}),
 AIMessage(content='4', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='thanks', additional_kwargs={}, response_metadata={}),
 AIMessage(content='no problem!', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='having fun?', additional_kwargs={}, response_metadata={}),
 AIMessage(content='yes!', additional_kwargs={}, response_metadata={})]
```

Чтобы использовать его в нашей цепочке, нам просто нужно запустить триммер перед передачей входных данных `messages` нашему промпту.

```python
workflow = StateGraph(state_schema=State)


def call_model(state: State):
    print(f"Messages before trimming: {len(state['messages'])}")
    trimmed_messages = trimmer.invoke(state["messages"])
    print(f"Messages after trimming: {len(trimmed_messages)}")
    print("Remaining messages:")
    for msg in trimmed_messages:
        print(f"  {type(msg).__name__}: {msg.content}")
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

Теперь, если мы попытаемся спросить у модели наше имя, она его не узнает, поскольку мы обрезали эту часть истории чата. (Определив нашу стратегию обрезки как `'last'`, мы сохраняем только самые последние сообщения, которые помещаются в `max_tokens`.)

```python
config = {"configurable": {"thread_id": "abc567"}}
query = "What is my name?"
language = "English"

input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()
```

```python
Messages before trimming: 12
Messages after trimming: 8
Remaining messages:
  SystemMessage: you're a good assistant
  HumanMessage: whats 2 + 2
  AIMessage: 4
  HumanMessage: thanks
  AIMessage: no problem!
  HumanMessage: having fun?
  AIMessage: yes!
  HumanMessage: What is my name?
==================================[1m Ai Message [0m==================================

I don't know your name. If you'd like to share it, feel free!
```

Но если мы спросим о информации, которая содержится в последних нескольких сообщениях, она помнит:

```python
config = {"configurable": {"thread_id": "abc678"}}

query = "What math problem was asked?"
language = "English"

input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()
```

```python
Messages before trimming: 12
Messages after trimming: 8
Remaining messages:
  SystemMessage: you're a good assistant
  HumanMessage: whats 2 + 2
  AIMessage: 4
  HumanMessage: thanks
  AIMessage: no problem!
  HumanMessage: having fun?
  AIMessage: yes!
  HumanMessage: What math problem was asked?
==================================[1m Ai Message [0m==================================

The math problem that was asked was "what's 2 + 2."
```

Если вы посмотрите на LangSmith, вы сможете точно увидеть, что происходит под капотом, в трассировке LangSmith.

## Потоковая передача

Теперь у нас есть работающий чат-бот. Однако одно _действительно_ важное соображение UX для приложений чат-ботов — это потоковая передача. LLM иногда могут отвечать довольно долго, и поэтому, чтобы улучшить пользовательский опыт, большинство приложений передают каждый токен по мере его генерации. Это позволяет пользователю видеть прогресс.

На самом деле это очень просто сделать!

По умолчанию `.stream` в нашем приложении LangGraph передает шаги приложения — в данном случае единственный шаг ответа модели. Установка `stream_mode="messages"` позволяет нам передавать выходные токены вместо этого:

```python
config = {"configurable": {"thread_id": "abc789"}}
query = "Hi I'm Todd, please tell me a joke."
language = "English"

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Фильтр только для ответов модели
        print(chunk.content, end="|")
```

```python
|Hi| Todd|!| Here|’s| a| joke| for| you|:
|Why| don't| scientists| trust| atoms|?
|Because| they| make| up| everything|!||
```

## Следующие шаги

Теперь, когда вы понимаете основы создания чат-бота в LangChain, вас могут заинтересовать некоторые более продвинутые руководства:

- [Conversational RAG](https://python.langchain.com/docs/tutorials/rag/): Включить опыт чат-бота над внешним источником данных
- [Agents](https://python.langchain.com/docs/tutorials/agents/): Создать чат-бота, который может выполнять действия

Если вы хотите глубже погрузиться в детали, стоит изучить следующее:

- [Streaming](https://python.langchain.com/docs/how_to/streaming/): потоковая передача _критически важна_ для чат-приложений
- [How to add message history](https://python.langchain.com/docs/how_to/message_history/): для более глубокого погружения во все, что связано с историей сообщений
- [How to manage large message history](https://python.langchain.com/docs/how_to/chat_history/): дополнительные методы управления большой историей чата
- [LangGraph main docs](https://langchain-ai.github.io/langgraph/): для получения более подробной информации о создании с помощью LangGraph

Source: https://python.langchain.com/docs/tutorials/chatbot/
