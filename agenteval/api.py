from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .runner import load_tasks, run_suite

app = FastAPI(title="AgentEval")
RUNS: dict[str, dict] = {}  # ponytail: in-memory; MLflow holds the durable copy


class RunRequest(BaseModel):
    model: str = "scripted:oracle"
    seed: int = 0
    adversarial: bool = True
    defense: Literal["none", "prompt", "firewall"] = "prompt"
    judge_model: str | None = None


@app.get("/tasks")
def tasks():
    return [{"id": t["id"], "category": t["category"], "prompt": t["prompt"]} for t in load_tasks()]


@app.post("/runs")
def create_run(req: RunRequest):
    run_id, metrics, records = run_suite(req.model, seed=req.seed, adversarial=req.adversarial,
                                         defense=req.defense, judge_model=req.judge_model)
    RUNS[run_id] = {"run_id": run_id, "config": req.model_dump(), "metrics": metrics, "records": records}
    return {"run_id": run_id, "metrics": metrics}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "run not found")
    return RUNS[run_id]
