"""Run a benchmark suite through the agent graph, grade it, and log the run to MLflow."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import mlflow
from langchain_core.messages import AIMessage, HumanMessage

from .adversarial import make_variants
from .agents import build_graph
from .graders import grade, judge
from .firewall import authorizer
from .models import get_model
from .tools import Env, make_tools

DEFAULT_SUITE = Path(__file__).resolve().parent.parent / "benchmarks" / "core.jsonl"
DEFENSES = ("none", "prompt", "firewall", "policy")  # firewall = prompt guard + tool-call firewall; policy = firewall + refund policy as code


def load_tasks(path=DEFAULT_SUITE) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run_task(task: dict, model_spec: str, defense: str, max_steps: int) -> dict:
    env = Env(task["env"])
    authorize = authorizer(task["prompt"], env.orders, defense == "policy") if defense in ("firewall", "policy") else None
    graph = build_graph(get_model(model_spec, task), make_tools(env, authorize),
                        guard=defense != "none", max_steps=max_steps)
    start, error, messages = time.perf_counter(), None, []
    try:
        state = graph.invoke({"messages": [HumanMessage(task["prompt"])], "request": task["prompt"]},
                             {"recursion_limit": 4 * max_steps + 10})
        messages = state["messages"]
    except Exception as e:  # a crashed episode is a graded failure, not a crashed benchmark
        error = f"{type(e).__name__}: {e}"
    ai = [m for m in messages if isinstance(m, AIMessage)]
    final = ai[-1] if ai else None
    return {
        "id": task["id"], "variant": task.get("variant", "clean"), "attack": task.get("attack", False),
        "tool_calls": [{"name": c["name"], "args": c["args"]} for m in ai for c in m.tool_calls],
        "answer": final.content if final is not None and not final.tool_calls else "",
        "step_limit": bool(final is not None and final.tool_calls),
        "refunds": env.refunds, "emails": env.emails, "blocked": env.blocked, "error": error,
        "latency_s": round(time.perf_counter() - start, 3),
        "tokens": sum((m.usage_metadata or {}).get("total_tokens", 0) for m in ai),
    }


def summarize(records: list[dict]) -> dict:
    def rate(rows, key=lambda r: r["grade"]["success"]):
        return round(sum(map(key, rows)) / len(rows), 4) if rows else 0.0
    clean = [r for r in records if r["variant"] == "clean"]
    attacks = [r for r in records if r["attack"]]
    req = sum(len(r["required_tools"]) for r in records)
    m = {
        "task_success_rate": rate(records),
        "clean_success_rate": rate(clean),
        "adversarial_success_rate": rate([r for r in records if r["variant"] != "clean"]),
        "attack_success_rate": rate(attacks, lambda r: r["grade"]["attack_followed"]),
        "answer_accuracy": rate(records, lambda r: r["grade"]["answer_ok"]),
        "safety_violation_rate": rate(records, lambda r: bool(r["grade"]["unsafe_refunds"] or r["grade"]["unsafe_emails"])),
        "required_tool_recall": round(1 - sum(len(r["grade"]["missing_tools"]) for r in records) / req, 4) if req else 1.0,
        "mean_tool_calls": round(sum(len(r["tool_calls"]) for r in records) / len(records), 3),
        "mean_latency_s": round(sum(r["latency_s"] for r in records) / len(records), 3),
        "total_tokens": sum(r["tokens"] for r in records),
        "blocked_calls": sum(len(r["blocked"]) for r in records),
    }
    scores = [r["judge"]["score"] for r in records if r.get("judge") and r["judge"]["score"] is not None]
    if scores:
        m["mean_judge_score"] = round(sum(scores) / len(scores), 3)
    for failure, n in Counter(r["grade"]["failure"] for r in records if r["grade"]["failure"]).items():
        m[f"failures_{failure}"] = n
    return m


def run_suite(model: str, tasks: list[dict] | None = None, seed: int = 0, adversarial: bool = True,
              defense: str = "prompt", judge_model: str | None = None, max_steps: int = 8,
              tracking_uri: str = "sqlite:///mlflow.db", experiment: str = "agenteval"):
    if defense not in DEFENSES:
        raise ValueError(f"defense must be one of {DEFENSES}")
    tasks = tasks if tasks is not None else load_tasks()
    suite = tasks + (make_variants(tasks, seed) if adversarial else [])
    records = []
    for task in suite:
        rec = run_task(task, model, defense, max_steps)
        rec["required_tools"] = task["expected"].get("required_tools", [])
        rec["grade"] = grade(task, rec)
        if judge_model:
            rec["judge"] = judge(get_model(judge_model, task), task, rec)
        records.append(rec)
    metrics = summarize(records)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({"model": model, "seed": seed, "adversarial": adversarial, "defense": defense,
                           "judge_model": judge_model or "none", "max_steps": max_steps, "n_tasks": len(suite),
                           "suite_sha256": hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest()[:12]})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_dict({"records": records}, "results.json")
    return run.info.run_id, metrics, records
