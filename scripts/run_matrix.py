"""Run every model under each defense; save metrics with 95% bootstrap CIs and per-task records."""
import json
import random
import sys
from pathlib import Path

from agenteval.runner import DEFENSES, run_suite

MODELS = sys.argv[1:] or ["ollama:qwen2.5:7b", "ollama:llama3.1:8b", "ollama:mistral-nemo"]
OUT = Path("results")
OUT.mkdir(exist_ok=True)
RATES = {  # metric -> (rows it is measured on, per-row outcome)
    "task_success_rate": (lambda r: True, lambda r: r["grade"]["success"]),
    "attack_success_rate": (lambda r: r["attack"], lambda r: r["grade"]["attack_followed"]),
    "safety_violation_rate": (lambda r: True, lambda r: bool(r["grade"]["unsafe_refunds"] or r["grade"]["unsafe_emails"])),
}


def ci95(values: list[bool], n_boot: int = 2000, seed: int = 0) -> list[float]:
    """Percentile bootstrap over episodes."""
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_boot))
    return [round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot) - 1], 3)]


summary = []
for model in MODELS:
    for defense in DEFENSES:
        run_id, metrics, records = run_suite(model, seed=0, defense=defense)
        cis = {f"{k}_ci95": ci95([bool(out(r)) for r in records if keep(r)]) for k, (keep, out) in RATES.items()}
        tag = f"{model.replace(':', '_').replace('/', '_')}_{defense}"
        (OUT / f"{tag}.json").write_text(json.dumps({"model": model, "defense": defense, "mlflow_run_id": run_id,
                                                     "metrics": metrics, **cis, "records": records}, indent=2))
        summary.append({"model": model, "defense": defense, "mlflow_run_id": run_id, **metrics, **cis})
        print(json.dumps(summary[-1]), flush=True)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))  # partial results survive a crash
