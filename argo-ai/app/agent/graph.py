from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools import ml_predict_tool, rag_recommend_tool
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv('secret_api_key')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    api_key=api_key
)

def agent_node(state):
    data = state["data"]

    ml = ml_predict_tool.invoke({"data": data})
    rag = rag_recommend_tool.invoke(
        "арбуз оптимальная температура полив риск засуха жара"
    )

    prompt = f"""
Ты — агро-помощник. Говори просто и по делу.

Результаты модели:
- Индекс урожайности: {ml['yield_index']}
- Уровень риска: {ml['risk_level']}

Экспертные рекомендации:
{rag}

Сформируй:
1) Короткое объяснение (2–3 предложения)
2) 2 практических совета фермеру
"""

    res = llm.invoke(prompt)
    return {"output": res.content}

graph = StateGraph(dict)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.set_finish_point("agent")

app_graph = graph.compile()
