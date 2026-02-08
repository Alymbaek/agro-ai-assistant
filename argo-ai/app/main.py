from fastapi import FastAPI
from .schemas import AnalyzeResponse, AnalyzeRequest
from app.ml import predict
import uvicorn
from app.agent.graph import app_graph



app = FastAPI(title='Argo AI - Watermelon Assisten')



@app.post('/analyze', response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    yield_index, risk = predict(req.dict())

    return AnalyzeResponse(
        yield_index=yield_index,
        risk_level=risk
    )

@app.post('/analyze_agent')
def analyze_agent(req: AnalyzeRequest):
    result = app_graph.invoke({'data': req.dict()})
    return {'analyze': result['output']}

@app.get('/check')
def check():
    return {'status': 'Ok 200'}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)
