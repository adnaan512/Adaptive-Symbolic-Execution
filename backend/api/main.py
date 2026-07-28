import random
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Adaptive SE Inference API",
    description="Live evaluation endpoint for the C++ KLEE AI Searcher.",
    version="1.0.0"
)

class KleeState(BaseModel):
    state_id: int
    instruction_depth: int
    query_cost: int
    call_stack_depth: int

class EvaluationRequest(BaseModel):
    active_states: List[KleeState]

class EvaluationResponse(BaseModel):
    selected_state_id: int

@app.post("/api/evaluate_state", response_model=EvaluationResponse)
async def evaluate_state(req: EvaluationRequest):
    """
    Called by KLEE's C++ Searcher.cpp via libcurl.
    Receives a list of active ExecutionStates. 
    Runs them through the XGBoost/RL model (mocked here as a heuristic function)
    and returns the best state_id to explore next.
    """
    if not req.active_states:
        raise HTTPException(status_code=400, detail="No active states provided")
        
    # In a full production environment, we would do:
    # features = extract_features(req.active_states)
    # scores = xgboost_model.predict(features)
    # best_state = argmax(scores)
    
    # For this architecture demonstration, we rank based on a simple combination
    # of depth and cost, injecting a slight random factor to break ties.
    best_state = None
    best_score = -float('inf')
    
    for state in req.active_states:
        # Heuristic: Favor deeper execution states but penalize high query costs
        score = (state.instruction_depth * 10) - state.query_cost + random.uniform(0, 5)
        if score > best_score:
            best_score = score
            best_state = state.state_id
            
    return EvaluationResponse(selected_state_id=best_state)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "KLEE AI Inference Engine is running."}
