---
title: "Создайте агента"
topic: "tutorials"
filename: "agents"
source: "https://python.langchain.com/docs/tutorials/agents/"
author: "Перевод GPT"
date: "2025-04-05"
---

# Создайте агента

LangChain поддерживает создание агентов — систем, которые используют LLM в качестве движка рассуждений для определения того, какие действия следует предпринять, и каких входных данных необходимо для выполнения действия. После выполнения действий результаты могут быть возвращены обратно в LLM, чтобы определить, нужны ли дополнительные действия, или можно ли завершить работу. Это часто достигается с помощью вызовов инструментов (tool-calling).

В этом руководстве мы создадим агента, который может взаимодействовать с поисковой системой. Вы сможете задавать этому агенту вопросы, наблюдать, как он вызывает инструмент поиска, и вести с ним беседы.

## Агент "под ключ"

Приведенный ниже фрагмент кода представляет собой полностью функционального агента, который использует LLM для принятия решений о том, какие инструменты использовать. Он оснащен универсальным инструментом поиска. У него есть разговорная память — это означает, что его можно использовать как многотурового чат-бота.

В остальной части руководства мы рассмотрим отдельные компоненты и то, что делает каждая часть, — но если вы хотите просто взять код и начать работу, не стесняйтесь использовать этот!

```python
# Импортируем соответствующую функциональность
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Создаем агента
memory = MemorySaver()
model = init_chat_model("anthropic:claude-3-5-sonnet-latest")
search = TavilySearch(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)
```

```python
# Используем агента
config = {"configurable": {"thread_id": "abc123"}}

input_message = {
    "role": "user",
    "content": "Hi, I'm Bob and I live in SF.",
}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

```python
================================[1m Human Message [0m=================================

Hi, I'm Bob and I live in SF.
==================================[1m Ai Message [0m==================================

Hello Bob! I notice you've introduced yourself and mentioned you live in SF (San Francisco), but you haven't asked a specific question or made a request that requires the use of any tools. Is there something specific you'd like to know about San Francisco or any other topic? I'd be happy to help you find information using the available search tools.
```

```python
input_message = {
    "role": "user",
    "content": "What's the weather where I live?",
}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

```python
================================[1m Human Message [0m=================================

What's the weather where I live?
==================================[1m Ai Message [0m==================================

[{'text': 'Let me search for current weather information in San Francisco.', 'type': 'text'}, {'id': 'toolu_011kSdheoJp8THURoLmeLtZo', 'input': {'query': 'current weather San Francisco CA'}, 'name': 'tavily_search', 'type': 'tool_use'}]
Tool Calls:
  tavily_search (toolu_011kSdheoJp8THURoLmeLtZo)
 Call ID: toolu_011kSdheoJp8THURoLmeLtZo
  Args:
    query: current weather San Francisco CA
=================================[1m Tool Message [0m=================================
Name: tavily_search

{"query": "current weather San Francisco CA", "follow_up_questions": null, "answer": null, "images": [], "results": [{"title": "Weather in San Francisco, CA", "url": "https://www.weatherapi.com/", "content": "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1750168606, 'localtime': '2025-06-17 06:56'}, 'current': {'last_updated_epoch': 1750167900, 'last_updated': '2025-06-17 06:45', 'temp_c': 11.7, 'temp_f': 53.1, 'is_day': 1, 'condition': {'text': 'Fog', 'icon': '//cdn.weatherapi.com/weather/64x64/day/248.png', 'code': 1135}, 'wind_mph': 4.0, 'wind_kph': 6.5, 'wind_degree': 215, 'wind_dir': 'SW', 'pressure_mb': 1017.0, 'pressure_in': 30.02, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 86, 'cloud': 0, 'feelslike_c': 11.3, 'feelslike_f': 52.4, 'windchill_c': 8.7, 'windchill_f': 47.7, 'heatindex_c': 9.8, 'heatindex_f': 49.7, 'dewpoint_c': 9.6, 'dewpoint_f': 49.2, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 0.0, 'gust_mph': 6.3, 'gust_kph': 10.2}}", "score": 0.944705, "raw_content": null}, {"title": "Weather in San Francisco in June 2025", "url": "https://world-weather.info/forecast/usa/san_francisco/june-2025/", "content": "Detailed ⚡ San Francisco Weather Forecast for June 2025 - day/night ️ temperatures, precipitations - World-Weather.info. Add the current city. Search. Weather; Archive; Weather Widget °F. World; United States; California; Weather in San Francisco; ... 17 +64° +54° 18 +61° +54° 19", "score": 0.86441374, "raw_content": null}], "response_time": 2.34}
==================================[1m Ai Message [0m==================================

Based on the search results, here's the current weather in San Francisco:
- Temperature: 53.1°F (11.7°C)
- Condition: Foggy
- Wind: 4.0 mph from the Southwest
- Humidity: 86%
- Visibility: 9 miles

This is quite typical weather for San Francisco, with the characteristic fog that the city is known for. Would you like to know anything else about the weather or San Francisco in general?
```

## Настройка

### Jupyter Notebook

Это руководство (и большинство других руководств в документации) использует Jupyter notebooks и предполагает, что читатель также использует их. Jupyter notebooks идеально подходят для изучения работы с системами LLM, потому что часто что-то может пойти не так (неожиданный вывод, API недоступен и т.д.), и прохождение руководств в интерактивной среде — отличный способ лучше их понять.

Это и другие руководства, возможно, наиболее удобно запускать в Jupyter notebook. Инструкции по установке см. здесь.

### Установка

Для установки LangChain выполните:

```pip
%pip install -U langgraph langchain-tavily langgraph-checkpoint-sqlite
```

Для получения более подробной информации см. наше руководство по установке.

### LangSmith

Многие приложения, которые вы создаете с помощью LangChain, будут содержать несколько шагов с несколькими вызовами LLM. По мере того как эти приложения становятся все более и более сложными, становится крайне важно иметь возможность проверять, что именно происходит внутри вашей цепочки или агента. Лучший способ сделать это — с помощью LangSmith.

После регистрации по ссылке выше, убедитесь, что вы установили переменные среды для начала записи трассировок:

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

### Tavily

Мы будем использовать Tavily (поисковую систему) в качестве инструмента. Чтобы использовать ее, вам нужно получить и установить ключ API:

```bash
export TAVILY_API_KEY="..."
```

Или, если вы находитесь в ноутбуке, вы можете установить его с помощью:

```python
import getpass
import os

os.environ["TAVILY_API_KEY"] = getpass.getpass()
```

## Определение инструментов

Сначала нам нужно создать инструменты, которые мы хотим использовать. Нашим основным инструментом выбора будет Tavily — поисковая система. Мы можем использовать специальный пакет интеграции `langchain-tavily`, чтобы легко использовать поисковую систему Tavily в качестве инструмента с LangChain.

```python
from langchain_tavily import TavilySearch

search = TavilySearch(max_results=2)
search_results = search.invoke("What is the weather in SF")
print(search_results)
# Если захотим, мы можем создать другие инструменты.
# Как только у нас будут все нужные инструменты, мы можем поместить их в список, на который будем ссылаться позже.
tools = [search]
```

```python
{'query': 'What is the weather in SF', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': 'Weather in San Francisco, CA', 'url': 'https://www.weatherapi.com/', 'content': "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1750168606, 'localtime': '2025-06-17 06:56'}, 'current': {'last_updated_epoch': 1750167900, 'last_updated': '2025-06-17 06:45', 'temp_c': 11.7, 'temp_f': 53.1, 'is_day': 1, 'condition': {'text': 'Fog', 'icon': '//cdn.weatherapi.com/weather/64x64/day/248.png', 'code': 1135}, 'wind_mph': 4.0, 'wind_kph': 6.5, 'wind_degree': 215, 'wind_dir': 'SW', 'pressure_mb': 1017.0, 'pressure_in': 30.02, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 86, 'cloud': 0, 'feelslike_c': 11.3, 'feelslike_f': 52.4, 'windchill_c': 8.7, 'windchill_f': 47.7, 'heatindex_c': 9.8, 'heatindex_f': 49.7, 'dewpoint_c': 9.6, 'dewpoint_f': 49.2, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 0.0, 'gust_mph': 6.3, 'gust_kph': 10.2}}", 'score': 0.9185379, 'raw_content': None}, {'title': 'Weather in San Francisco in June 2025', 'url': 'https://world-weather.info/forecast/usa/san_francisco/june-2025/', 'content': "Weather in San Francisco in June 2025 (California) - Detailed Weather Forecast for a Month *   Weather in San Francisco Weather in San Francisco in June 2025 *   1 +63° +55° *   2 +66° +54° *   3 +66° +55° *   4 +66° +54° *   5 +66° +55° *   6 +66° +57° *   7 +64° +55° *   8 +63° +55° *   9 +63° +54° *   10 +59° +54° *   11 +59° +54° *   12 +61° +54° Weather in Washington, D.C.**+68°** Sacramento**+81°** Pleasanton**+72°** Redwood City**+68°** San Leandro**+61°** San Mateo**+64°** San Rafael**+70°** San Ramon**+64°** South San Francisco**+61°** Daly City**+59°** Wilder**+66°** Woodacre**+70°** world's temperature today Colchani day+50°F night+16°F Az Zubayr day+124°F night+93°F Weather forecast on your site Install _San Francisco_ +61° Temperature units", 'score': 0.7978881, 'raw_content': None}], 'response_time': 2.62}
```

Во многих приложениях вы можете захотеть определить пользовательские инструменты. LangChain поддерживает создание пользовательских инструментов через функции Python и другими способами. Подробности см. в руководстве "How to create tools".

## Использование языковых моделей

Далее давайте узнаем, как использовать языковую модель для вызова инструментов. LangChain поддерживает множество различных языковых моделей, которые можно использовать взаимозаменяемо — выберите ту, которую хотите использовать ниже!

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

Вы можете вызвать языковую модель, передав список сообщений. По умолчанию ответом является строка `content`.

```python
query = "Hi!"
response = model.invoke([{"role": "user", "content": query}])
response.text()
```

```python
'Hello! How can I help you today?'
```

Теперь мы можем увидеть, как выглядит включение у этой модели возможности вызова инструментов. Для этого мы используем `.bind_tools`, чтобы дать языковой модели знания об этих инструментах.

```python
model_with_tools = model.bind_tools(tools)
```

Теперь мы можем вызвать модель. Сначала давайте вызовем ее с обычным сообщением и посмотрим, как она отвечает. Мы можем посмотреть как на поле `content`, так и на поле `tool_calls`.

```python
query = "Hi!"
response = model_with_tools.invoke([{"role": "user", "content": query}])

print(f"Message content: {response.text()}
")
print(f"Tool calls: {response.tool_calls}")
```

```python
Message content: Hello! I'm here to help you. I have access to a powerful search tool that can help answer questions and find information about various topics. What would you like to know about?

Feel free to ask any question or request information, and I'll do my best to assist you using the available tools.

Tool calls: []
```

Теперь давайте попробуем вызвать ее с вводом, при котором ожидается вызов инструмента.

```python
query = "Search for the weather in SF"
response = model_with_tools.invoke([{"role": "user", "content": query}])

print(f"Message content: {response.text()}
")
print(f"Tool calls: {response.tool_calls}")
```

```python
Message content: I'll help you search for information about the weather in San Francisco.

Tool calls: [{'name': 'tavily_search', 'args': {'query': 'current weather San Francisco'}, 'id': 'toolu_015gdPn1jbB2Z21DmN2RAnti', 'type': 'tool_call'}]
```

Мы видим, что теперь нет текстового содержимого, но есть вызов инструмента! Он хочет, чтобы мы вызвали инструмент Tavily Search.

Это еще не вызывает сам инструмент — он просто говорит нам об этом. Чтобы на самом деле вызвать его, нам нужно создать нашего агента.

## Создание агента

Теперь, когда мы определили инструменты и LLM, мы можем создать агента. Мы будем использовать LangGraph для построения агента. В настоящее время мы используем высокоуровневый интерфейс для построения агента, но приятная особенность LangGraph заключается в том, что этот высокоуровневый интерфейс поддерживается низкоуровневым, высокоуправляемым API на случай, если вы захотите изменить логику агента.

Теперь мы можем инициализировать агента с помощью LLM и инструментов.

Обратите внимание, что мы передаем `model`, а не `model_with_tools`. Это потому, что `create_react_agent` вызовет `.bind_tools` для нас под капотом.

```python
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)
```

**Справочник по API:** [create_react_agent](https://api.python.langchain.com/en/latest/agents/langgraph.prebuilt.create_react_agent.html)

## Запуск агента

Теперь мы можем запустить агента с несколькими запросами! Обратите внимание, что пока это все **бесшовные** запросы (он не будет помнить предыдущие взаимодействия). Обратите внимание, что агент вернет **финальное** состояние в конце взаимодействия (которое включает любые входные данные, позже мы увидим, как получить только выходные данные).

Сначала давайте посмотрим, как он отвечает, когда нет необходимости вызывать инструмент:

```python
input_message = {"role": "user", "content": "Hi!"}
response = agent_executor.invoke({"messages": [input_message]})

for message in response["messages"]:
    message.pretty_print()
```

```python
================================[1m Human Message [0m=================================

Hi!
==================================[1m Ai Message [0m==================================

Hello! I'm here to help you with your questions using the available search tools. Please feel free to ask any question, and I'll do my best to find relevant and accurate information for you.
```

Чтобы точно увидеть, что происходит под капотом (и убедиться, что он не вызывает инструмент), мы можем посмотреть трассировку LangSmith.

Теперь давайте попробуем его на примере, когда он должен вызывать инструмент.

```python
input_message = {"role": "user", "content": "Search for the weather in SF"}
response = agent_executor.invoke({"messages": [input_message]})

for message in response["messages"]:
    message.pretty_print()
```

```python
================================[1m Human Message [0m=================================

Search for the weather in SF
==================================[1m Ai Message [0m==================================

[{'text': "I'll help you search for weather information in San Francisco. Let me use the search engine to find current weather conditions.", 'type': 'text'}, {'id': 'toolu_01WWcXGnArosybujpKzdmARZ', 'input': {'query': 'current weather San Francisco SF'}, 'name': 'tavily_search', 'type': 'tool_use'}]
Tool Calls:
  tavily_search (toolu_01WWcXGnArosybujpKzdmARZ)
 Call ID: toolu_01WWcXGnArosybujpKzdmARZ
  Args:
    query: current weather San Francisco SF
=================================[1m Tool Message [0m=================================
Name: tavily_search

{"query": "current weather San Francisco SF", "follow_up_questions": null, "answer": null, "images": [], "results": [{"title": "Weather in San Francisco, CA", "url": "https://www.weatherapi.com/", "content": "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1750168606, 'localtime': '2025-06-17 06:56'}, 'current': {'last_updated_epoch': 1750167900, 'last_updated': '2025-06-17 06:45', 'temp_c': 11.7, 'temp_f': 53.1, 'is_day': 1, 'condition': {'text': 'Fog', 'icon': '//cdn.weatherapi.com/weather/64x64/day/248.png', 'code': 1135}, 'wind_mph': 4.0, 'wind_kph': 6.5, 'wind_degree': 215, 'wind_dir': 'SW', 'pressure_mb': 1017.0, 'pressure_in': 30.02, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 86, 'cloud': 0, 'feelslike_c': 11.3, 'feelslike_f': 52.4, 'windchill_c': 8.7, 'windchill_f': 47.7, 'heatindex_c': 9.8, 'heatindex_f': 49.7, 'dewpoint_c': 9.6, 'dewpoint_f': 49.2, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 0.0, 'gust_mph': 6.3, 'gust_kph': 10.2}}", "score": 0.885373, "raw_content": null}, {"title": "Weather in San Francisco in June 2025", "url": "https://world-weather.info/forecast/usa/san_francisco/june-2025/", "content": "Detailed ⚡ San Francisco Weather Forecast for June 2025 - day/night ️ temperatures, precipitations - World-Weather.info. Add the current city. Search. Weather; Archive; Weather Widget °F. World; United States; California; Weather in San Francisco; ... 17 +64° +54° 18 +61° +54° 19", "score": 0.8830044, "raw_content": null}], "response_time": 2.6}
==================================[1m Ai Message [0m==================================

Based on the search results, here's the current weather in San Francisco:
- Temperature: 53.1°F (11.7°C)
- Conditions: Foggy
- Wind: 4.0 mph from the SW
- Humidity: 86%
- Visibility: 9.0 miles

The weather appears to be typical for San Francisco, with morning fog and mild temperatures. The "feels like" temperature is 52.4°F (11.3°C).
```

Мы можем проверить трассировку LangSmith, чтобы убедиться, что он эффективно вызывает поисковый инструмент.

## Потоковая передача сообщений

Мы видели, как агент может быть вызван с помощью `.invoke`, чтобы получить финальный ответ. Если агент выполняет несколько шагов, это может занять некоторое время. Чтобы показать промежуточный прогресс, мы можем передавать сообщения обратно по мере их появления.

```python
for step in agent_executor.stream({"messages": [input_message]}, stream_mode="values"):
    step["messages"][-1].pretty_print()
```

```python
================================[1m Human Message [0m=================================

Search for the weather in SF
==================================[1m Ai Message [0m==================================

[{'text': "I'll help you search for information about the weather in San Francisco.", 'type': 'text'}, {'id': 'toolu_01DCPnJES53Fcr7YWnZ47kDG', 'input': {'query': 'current weather San Francisco'}, 'name': 'tavily_search', 'type': 'tool_use'}]
Tool Calls:
  tavily_search (toolu_01DCPnJES53Fcr7YWnZ47kDG)
 Call ID: toolu_01DCPnJES53Fcr7YWnZ47kDG
  Args:
    query: current weather San Francisco
=================================[1m Tool Message [0m=================================
Name: tavily_search

{"query": "current weather San Francisco", "follow_up_questions": null, "answer": null, "images": [], "results": [{"title": "Weather in San Francisco", "url": "https://www.weatherapi.com/", "content": "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1750168506, 'localtime': '2025-06-17 06:55'}, 'current': {'last_updated_epoch': 1750167900, 'last_updated': '2025-06-17 06:45', 'temp_c': 11.7, 'temp_f': 53.1, 'is_day': 1, 'condition': {'text': 'Fog', 'icon': '//cdn.weatherapi.com/weather/64x64/day/248.png', 'code': 1135}, 'wind_mph': 4.0, 'wind_kph': 6.5, 'wind_degree': 215, 'wind_dir': 'SW', 'pressure_mb': 1017.0, 'pressure_in': 30.02, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 86, 'cloud': 0, 'feelslike_c': 11.3, 'feelslike_f': 52.4, 'windchill_c': 8.7, 'windchill_f': 47.7, 'heatindex_c': 9.8, 'heatindex_f': 49.7, 'dewpoint_c': 9.6, 'dewpoint_f': 49.2, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 0.0, 'gust_mph': 6.3, 'gust_kph': 10.2}}", "score": 0.9542825, "raw_content": null}, {"title": "Weather in San Francisco in June 2025", "url": "https://world-weather.info/forecast/usa/san_francisco/june-2025/", "content": "Detailed ⚡ San Francisco Weather Forecast for June 2025 - day/night ️ temperatures, precipitations - World-Weather.info. Add the current city. Search. Weather; Archive; Weather Widget °F. World; United States; California; Weather in San Francisco; ... 17 +64° +54° 18 +61° +54° 19", "score": 0.8638634, "raw_content": null}], "response_time": 2.57}
==================================[1m Ai Message [0m==================================

Based on the search results, here's the current weather in San Francisco:
- Temperature: 53.1°F (11.7°C)
- Condition: Foggy
- Wind: 4.0 mph from the Southwest
- Humidity: 86%
- Visibility: 9.0 miles
- Feels like: 52.4°F (11.3°C)

This is quite typical weather for San Francisco, which is known for its fog, especially during the morning hours. The city's proximity to the ocean and unique geographical features often result in mild temperatures and foggy conditions.
```

## Потоковая передача токенов

В дополнение к потоковой передаче сообщений, также полезно передавать токены.

Мы можем сделать это, указав `stream_mode="messages"`.

::: note
Ниже мы используем `message.text()`, что требует `langchain-core>=0.3.37`.
:::

```python
for step, metadata in agent_executor.stream(
    {"messages": [input_message]}, stream_mode="messages"
):
    if metadata["langgraph_node"] == "agent" and (text := step.text()):
        print(text, end="|")
```

```python
I|'ll help you search for information| about the weather in San Francisco.|Base|d on the search results, here|'s the current weather in| San Francisco:
-| Temperature: 53.1°F (|11.7°C)
-| Condition: Foggy
- Wind:| 4.0 mph from| the Southwest
- Humidity|: 86%|
- Visibility: 9|.0 miles
- Pressure: |30.02 in|Hg

The weather| is characteristic of San Francisco, with| foggy conditions and mild temperatures|. The "feels like" temperature is slightly| lower at 52.4|°F (11.|3°C)| due to the wind chill effect|.|
```

## Добавление памяти

Как упоминалось ранее, этот агент является бесшовным. Это означает, что он не помнит предыдущие взаимодействия. Чтобы дать ему память, нам нужно передать checkpointer. При передаче checkpointer нам также необходимо передать `thread_id` при вызове агента (чтобы он знал, какой поток/беседу возобновлять).

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
```

**Справочник по API:** [MemorySaver](https://api.python.langchain.com/en/latest/checkpoint/langgraph.checkpoint.memory.MemorySaver.html)

```python
agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}
```

```python
input_message = {"role": "user", "content": "Hi, I'm Bob!"}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

```python
================================[1m Human Message [0m=================================

Hi, I'm Bob!
==================================[1m Ai Message [0m==================================

Hello Bob! I'm an AI assistant who can help you search for information using specialized search tools. Is there anything specific you'd like to know about or search for? I'm happy to help you find accurate and up-to-date information on various topics.
```

```python
input_message = {"role": "user", "content": "What's my name?"}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

```python
================================[1m Human Message [0m=================================

What's my name?
==================================[1m Ai Message [0m==================================

Your name is Bob, as you introduced yourself earlier. I can remember information shared within our conversation without needing to search for it.
```

Пример трассировки LangSmith

Если вы хотите начать новую беседу, вам просто нужно изменить используемый `thread_id`.

```python
config = {"configurable": {"thread_id": "xyz123"}}

input_message = {"role": "user", "content": "What's my name?"}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

```python
================================[1m Human Message [0m=================================

What's my name?
==================================[1m Ai Message [0m==================================

I apologize, but I don't have access to any tools that would tell me your name. I can only assist you with searching for publicly available information using the tavily_search function. I don't have access to personal information about users. If you'd like to tell me your name, I'll be happy to address you by it.
```

## Заключение

Вот и все! В этом быстром старте мы рассмотрели, как создать простого агента. Затем мы показали, как передавать ответ в потоковом режиме — не только с промежуточными шагами, но и с токенами! Мы также добавили память, чтобы вы могли вести с ними беседу. Агенты — это сложная тема, в которой многое нужно изучить!

Для получения дополнительной информации об агентах, пожалуйста, ознакомьтесь с документацией LangGraph. Она имеет свой собственный набор концепций, руководств и практических руководств.

Source: https://python.langchain.com/docs/tutorials/agents/

```

```
