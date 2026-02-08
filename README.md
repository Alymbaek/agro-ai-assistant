# Agro AI Assistant 🌱🍉

An AI-powered service for estimating watermelon yield and providing
practical recommendations for farmers based on weather data,
machine learning, and an AI agent with RAG.

The project is built as a demo decision-support system for agriculture,
with a focus on real-world applicability in Kyrgyzstan.

---

## 🚜 Features
- 📊 Watermelon yield index prediction
- ⚠️ Risk level assessment (low / medium / high)
- 🧠 AI agent that explains results in simple, human-friendly language
- 📚 RAG (Vector Store) with agronomic knowledge
- 🌦 Uses real weather data (Open-Meteo)

---

## 🧩 Architecture
- ML model (offline training → online inference)
- FastAPI backend
- LangGraph AI agent (orchestration layer)
- FAISS vector store (RAG)
- Gemini 1.5 Flash for natural language explanations

Machine learning is responsible for calculations,  
LLM is used **only** for explanation and recommendations.

---

## 🛠 Tech Stack
- FastAPI
- Scikit-learn
- LangGraph
- LangChain
- FAISS
- SentenceTransformers
- Google Gemini 1.5 Flash

---

## ▶️ Run the Project
uvicorn app.main:app --reload

after run: 
Main endpoint: /analyze-with-agent

## 📌 Example Request
{
  "region": "Osh",
  "month": 6,
  "avg_temp": 24.8,
  "max_temp": 31.2,
  "precipitation": 1.3
}


