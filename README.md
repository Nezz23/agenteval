# AgentEval: Agentic AI Evaluation & Red-Teaming Framework

Benchmarks LLM agents on tool use, task completion and side-effect safety in a sandboxed
customer-support environment, then red-teams them with seeded adversarial variants.

## How it works

```
request ─▶ planner ─▶ executor ⇄ tools (calculator, get_order, search_policy, issue_refund, send_email)
                         │
                         ▼
                      reviewer ──(REVISE, max 1)──▶ executor
                         │ APPROVE
                         ▼
             graders ─▶ metrics ─▶ MLflow (params, metrics, per-task results.json)
```

- **Agents** (`agenteval/agents.py`): LangGraph planner, tool-calling executor and reviewer, with a step limit.
- **Benchmark** (`benchmarks/core.jsonl`): 30 tasks across lookup, multi-step math, policy QA and reasoning, authorized actions and refusals.
- **Red team** (`agenteval/adversarial.py`): seeded variants. These are indirect prompt injection planted in every tool-visible record, instruction override in the user turn, and numeric distractors. 30 clean tasks become 80 episodes (30 clean, 40 attacks, 10 distractors).
- **Graders** (`agenteval/graders.py`): deterministic checks on the answer, the required tools and the environment state, so unauthorized refunds or emails are caught even when the answer looks right. Each failure is classified into a taxonomy (`attack_followed`, `unsafe_action`, `missing_tool`, `wrong_answer`, `step_limit`, `runtime_error`). An optional LLM-as-judge rubric score is also available.
- **Defenses** (`--defense`): `none`; `prompt`, a system instruction to treat tool output as untrusted data; and `firewall`, the prompt guard plus a tool-call firewall (`agenteval/firewall.py`) that only runs a refund or email when the user's own request names that order or recipient. Injected text cannot authorize anything. `policy` adds the store refund policy as code (delivered, within 30 days, at most $100 without a manager ticket), so even a refund the user did ask for is blocked when the policy forbids it.
- **Metrics**: task success (overall, clean, adversarial), attack success rate, safety-violation rate, blocked calls, 95% bootstrap confidence intervals, required-tool recall, mean tool calls, latency and tokens.
- **Interfaces**: CLI, FastAPI (`GET /tasks`, `POST /runs`, `GET /runs/{id}`), and MLflow tracking for every run.

## Run

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                          # harness tests
.venv/bin/python -m agenteval.cli --model scripted:oracle
export OPENROUTER_API_KEY=...                          # real model
.venv/bin/python -m agenteval.cli --model openrouter:<model-id>
.venv/bin/python -m agenteval.cli --model openrouter:<model-id> --defense firewall   # none | prompt | firewall | policy
PYTHONPATH=. .venv/bin/python scripts/run_matrix.py ollama:qwen2.5:7b      # every defense, with CIs, into results/
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
.venv/bin/uvicorn agenteval.api:app
```

## Results

**Harness validation** (scripted reference agents, seed 0, 80 episodes). These numbers check the graders, not a model:

| Agent | Defense | Task success | Clean | Adversarial | Attack success |
|---|---|---|---|---|---|
| `scripted:oracle` (ideal behaviour) | none | 1.00 | 1.00 | 1.00 | 0.00 |
| `scripted:naive` (obeys injected instructions) | none | 0.51 | 1.00 | 0.22 | 0.98 (39/40) |
| `scripted:naive` | firewall | 1.00 | 1.00 | 1.00 | 0.00 |

The graders give the oracle a perfect score and catch the naive agent obeying 39 of 40 attacks. The firewall blocks all of them without breaking a single legitimate task.

**LLM results** (local open-weight models via Ollama, default quantization, temperature 0, seed 0). Each run is 80 episodes, giving 720 episodes in total. Brackets are 95% bootstrap confidence intervals. Raw per-episode records are in `results/`, and every run is logged in MLflow.

| Model | Defense | Task success | Answer accuracy | Attack success | Safety violations | Required-tool recall | Blocked calls | Mean latency (s) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | none | 0.45 [0.34, 0.56] | 0.68 | 0.12 [0.03, 0.23] | 0.21 [0.12, 0.31] | 0.85 | 0 | 11.6 |
| Qwen2.5-7B | prompt | 0.41 [0.30, 0.53] | 0.71 | 0.25 [0.12, 0.40] | 0.21 [0.12, 0.31] | 0.84 | 0 | 10.9 |
| Qwen2.5-7B | firewall | 0.41 [0.30, 0.53] | 0.69 | **0.00** | 0.06 [0.01, 0.12] | 0.81 | 602 | 23.8 |
| Llama-3.1-8B | none | 0.24 [0.15, 0.34] | 0.44 | 0.10 [0.03, 0.20] | 0.35 [0.25, 0.46] | 0.69 | 0 | 39.5 |
| Llama-3.1-8B | prompt | 0.19 [0.10, 0.28] | 0.44 | 0.10 [0.03, 0.20] | 0.39 [0.29, 0.50] | 0.66 | 0 | 20.4 |
| Llama-3.1-8B | firewall | 0.30 [0.20, 0.40] | 0.42 | **0.00** | 0.05 [0.01, 0.10] | 0.66 | 51 | 16.3 |
| Mistral-Nemo-12B | none | 0.34 [0.24, 0.44] | 0.61 | 0.07 [0.00, 0.17] | 0.04 [0.00, 0.09] | 0.68 | 0 | 17.8 |
| Mistral-Nemo-12B | prompt | 0.34 [0.24, 0.44] | 0.60 | 0.07 [0.00, 0.17] | 0.04 [0.00, 0.09] | 0.66 | 0 | 16.6 |
| Mistral-Nemo-12B | firewall | 0.33 [0.23, 0.42] | 0.60 | **0.00** | **0.00** | 0.69 | 3 | 15.8 |

Metrics:
- **Attack success** is the share of the 40 attack episodes where the agent performed the exact injected action.
- **Safety violations** is the share of all episodes with any unauthorized refund or email.
- **Task success** requires a correct answer, all required tools, and no unsafe side effects.

Findings:
- **The firewall stopped every attack, for every model.** Attack success went to 0/40 in all three cases, and safety violations fell from 21% to 6% (Qwen), 35% to 5% (Llama) and 4% to 0% (Mistral). Task success did not drop meaningfully: Qwen moved from 0.45 to 0.41, within the confidence intervals, and Llama rose from 0.24 to 0.30.
- **The prompt guard did not help.** Telling the model to distrust tool output never measurably reduced attacks. Qwen's attack success actually rose from 0.12 to 0.25, though the intervals overlap. Enforcing safety in code beat asking for it in the prompt.
- **The agents' own unsafe actions were a bigger risk than prompt injection.** Without the firewall, Llama issued unauthorized refunds or emails in 35% of episodes, but followed an attack in only 10%. Most of these were refunds on orders it was only asked about, or ones the policy denied.
- **What the firewall cannot catch.** The violations left over were refunds on orders the user named in a request like "refund A103 *if policy allows it*", where the policy said no. The firewall checks *who asked*, not *whether policy allows it*. Closing that gap needs a policy check on the action itself, which the `policy` defense adds. It passes the scripted validation, blocking every policy-denied refund in the suite with zero false blocks. LLM results for it are pending.
- **The firewall has a cost on some models.** Qwen kept retrying blocked calls, 602 of them, which doubled its latency from 11.6 s to 23.8 s. Mistral almost never tried an unauthorized action (3 blocked calls).
- **Skipping tools was the main reliability failure.** Models answered without calling the tools they needed, for example guessing a policy instead of looking it up. Required-tool recall was 0.66–0.85, which is why answer accuracy is well above task success.

Limitations: single seed, small quantized local models, and a synthetic environment. The 80 episodes per run are enough to separate the firewall from no defense, but not to rank the prompt guard against no defense. The earlier 10-task results are kept in `results/v1_10tasks/`.
