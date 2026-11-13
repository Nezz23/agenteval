import argparse
import json

from .runner import DEFAULT_SUITE, DEFENSES, load_tasks, run_suite


def main():
    p = argparse.ArgumentParser(description="Run the AgentEval benchmark.")
    p.add_argument("--model", default="scripted:oracle", help="ollama:<model>, openrouter:<model> or scripted:<oracle|naive>")
    p.add_argument("--suite", default=str(DEFAULT_SUITE))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-adversarial", action="store_true")
    p.add_argument("--defense", default="prompt", choices=DEFENSES,
                   help="none, prompt (untrusted-output instruction) or firewall (prompt + tool-call firewall)")
    p.add_argument("--judge", default=None, help="model spec for the LLM-as-judge grader")
    a = p.parse_args()
    run_id, metrics, _ = run_suite(a.model, load_tasks(a.suite), a.seed, not a.no_adversarial,
                                   a.defense, a.judge)
    print(json.dumps({"mlflow_run_id": run_id, **metrics}, indent=2))


if __name__ == "__main__":
    main()
