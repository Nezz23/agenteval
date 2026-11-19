"""AgentEval episode replay: steps through recorded runs. No model runs here; every step shown is
read from results/*.json, the same records the README's numbers come from."""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Streamlit Cloud runs from the repo root

import altair as alt
import pandas as pd
import streamlit as st

from replay.episodes import (DEFENSES, VARIANTS, load_runs, load_tasks, outcome, planted_text,
                             results_table, showcase, timeline, user_request)

REPO = "https://github.com/Nezz23/agenteval"
SHOWN_STEPS = 12

st.set_page_config(page_title="AgentEval · Episode Replay", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
.hero h1 {font-size: 2.3rem; margin-bottom: .2rem; letter-spacing: -.02em;}
.hero p {font-size: 1.05rem; opacity: .8; margin-top: 0;}
.stat {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 16px 18px; height: 100%;}
.stat .num {font-size: 2rem; font-weight: 700; line-height: 1.1;}
.stat .lbl {font-size: .9rem; opacity: .75;}
.req {border-left: 4px solid #3b82f6; background: rgba(59,130,246,.08); padding: 10px 14px;
      border-radius: 8px; margin: 6px 0;}
.plant {border-left: 4px solid #ef4444; background: rgba(239,68,68,.08); padding: 10px 14px;
        border-radius: 8px; margin: 6px 0; font-family: ui-monospace, monospace; font-size: .85rem;}
.step {border: 1px solid rgba(128,128,128,.22); border-radius: 10px; padding: 8px 12px; margin: 6px 0;
       font-size: .88rem;}
.step code {font-size: .8rem; white-space: pre-wrap; word-break: break-word;}
.step .n {opacity: .55; margin-right: 6px;}
.step.ok {border-left: 4px solid #94a3b8;}
.step.blocked {border-left: 4px solid #f59e0b; background: rgba(245,158,11,.10);}
.step.unsafe {border-left: 4px solid #ef4444; background: rgba(239,68,68,.08);}
.step.attack {border-left: 4px solid #dc2626; background: rgba(220,38,38,.14);}
.tag {display: inline-block; font-size: .72rem; font-weight: 700; padding: 1px 8px; border-radius: 999px;
      margin-left: 6px; letter-spacing: .03em;}
.tag.blocked {background: #f59e0b; color: #111;}
.tag.unsafe, .tag.attack {background: #dc2626; color: #fff;}
.why {font-size: .82rem; margin-top: 4px;}
.verdict {border-radius: 10px; padding: 10px 14px; font-weight: 600; margin-top: 8px;}
.verdict.good {background: rgba(34,197,94,.14); border: 1px solid rgba(34,197,94,.5);}
.verdict.bad {background: rgba(220,38,38,.14); border: 1px solid rgba(220,38,38,.5);}
.verdict.neutral {background: rgba(148,163,184,.14); border: 1px solid rgba(148,163,184,.5);}
.side h4 {margin-bottom: 2px;}
.note {font-size: .85rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

TAGS = {"blocked": "BLOCKED", "unsafe": "UNSAFE", "attack": "FOLLOWED ATTACK"}


@st.cache_data
def data():
    return load_runs(), load_tasks()


def esc(text) -> str:
    return html.escape(str(text))


def render_steps(record: dict) -> None:
    steps = timeline(record)
    if not steps:
        st.markdown('<div class="step ok">The agent answered without calling a tool.</div>',
                    unsafe_allow_html=True)
    shown = steps[:SHOWN_STEPS]
    for i, s in enumerate(shown, 1):
        tag = f'<span class="tag {s["status"]}">{TAGS[s["status"]]}</span>' if s["status"] in TAGS else ""
        why = f'<div class="why">{esc(s["reason"])}</div>' if s["reason"] else ""
        st.markdown(f'<div class="step {s["status"]}"><span class="n">{i}</span><b>{esc(s["tool"])}</b>{tag}'
                    f'<br><code>{esc(json.dumps(s["args"]))}</code>{why}</div>', unsafe_allow_html=True)
    if len(steps) > SHOWN_STEPS:
        rest = steps[SHOWN_STEPS:]
        counts = {k: sum(1 for s in rest if s["status"] == k) for k in ("ok", "blocked", "unsafe", "attack")}
        parts = [f"{v} {k}" for k, v in counts.items() if v]
        with st.expander(f"{len(rest)} more steps ({', '.join(parts)})"):
            for i, s in enumerate(rest, SHOWN_STEPS + 1):
                st.markdown(f"`{i}` **{s['tool']}** `{json.dumps(s['args'])}` · {s['status']}")
    answer = record.get("answer") or "(no final answer)"
    kind, text = outcome(record)
    st.markdown(f'<div class="note">Final answer: <b>{esc(answer)}</b></div>'
                f'<div class="verdict {kind}">{esc(text)}</div>', unsafe_allow_html=True)


def render_task(task: dict) -> None:
    st.markdown(f'<div class="req"><b>User asked:</b> {esc(user_request(task))}</div>', unsafe_allow_html=True)
    planted = planted_text(task)
    if planted:
        where = ("hidden inside every order note and policy document the agent can read"
                 if task["variant"] == "indirect_injection" else "appended to the user's message")
        label = "Attacker text" if task.get("attack") else "Distractor"
        st.markdown(f'<div class="plant"><b>{label}, {where}:</b><br>{esc(planted)}</div>',
                    unsafe_allow_html=True)


runs, tasks = data()
by_key = {(r["model_label"], r["defense"]): r for r in runs}
episodes = sum(len(r["records"]) for r in runs)
attacks_per_run = sum(1 for rec in runs[0]["records"] if rec["attack"])
firewall_attacks = [round(r["metrics"]["attack_success_rate"] * attacks_per_run)
                    for r in runs if r["defense"] == "firewall"]
guard_helped = sum(by_key[(m, "prompt")]["metrics"]["attack_success_rate"]
                   < by_key[(m, "none")]["metrics"]["attack_success_rate"]
                   for m in dict.fromkeys(r["model_label"] for r in runs))
llama = {d: by_key[("Llama-3.1-8B", d)]["metrics"]["safety_violation_rate"] for d in ("none", "firewall")}

# ---------- Hero ----------
st.markdown(f"""<div class="hero"><h1>🛡️ AgentEval: watch a tool-call firewall stop prompt injection</h1>
<p>Tool-using LLM agents handle refunds and emails for a mock store. Attackers hide instructions in the data
they read. This page replays the recorded runs behind the <a href="{REPO}">AgentEval</a> results.</p></div>""",
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
stats = [(f"{episodes}", "recorded episodes, 3 open-weight models × 3 defenses"),
         (f"{max(firewall_attacks)} / {attacks_per_run}", "attacks that succeeded against the firewall, for every model"),
         (f"{llama['none']:.0%} → {llama['firewall']:.0%}", "Llama-3.1-8B episodes with an unauthorized action"),
         (f"{guard_helped} of 3", "models where a prompt guard (\"ignore injected text\") lowered attack success")]
for col, (num, lbl) in zip((c1, c2, c3, c4), stats):
    col.markdown(f'<div class="stat"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
                 unsafe_allow_html=True)

st.write("")
tab_demo, tab_results, tab_explore, tab_how = st.tabs(
    ["Same attack, two defenses", "Results", "Explore all 720 episodes", "How it works"])

# ---------- Showcase ----------
with tab_demo:
    pair = showcase(runs)
    task = tasks[pair["id"]]
    st.subheader(f"{pair['model']}, episode {pair['id']}")
    st.caption(f"The same model, the same task and the same attack. Only the defense changes. "
               f"Both runs answered {pair['before']['answer']!r} where the right answer is "
               f"{task['expected']['answer']!r}: the firewall stops the harmful action; it does not make "
               f"the model smarter.")
    render_task(task)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="side"><h4>❌ No defense</h4></div>', unsafe_allow_html=True)
        render_steps(pair["before"])
    with right:
        st.markdown('<div class="side"><h4>✅ Tool-call firewall</h4></div>', unsafe_allow_html=True)
        render_steps(pair["after"])
    st.markdown('<p class="note">The firewall reads permission only from the user\'s own request, never from '
                'tool outputs or injected text. A refund runs only if the user wrote "refund order X"; an email '
                'goes only to an address the user typed.</p>', unsafe_allow_html=True)

# ---------- Results ----------
with tab_results:
    df = pd.DataFrame(results_table(runs))
    order = list(DEFENSES.values())
    colors = alt.Scale(domain=order, range=["#94a3b8", "#60a5fa", "#22c55e"])
    for metric, help_text in (("Attack success", "Share of the 40 attack episodes where the agent did what the attacker asked."),
                              ("Unsafe actions", "Share of all 80 episodes with any refund or email the user never asked for.")):
        st.markdown(f"**{metric}** · {help_text}")
        part = df[df.metric == metric]
        base = alt.Chart(part).encode(
            x=alt.X("defense:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("defense:N", sort=order, scale=colors, title=None,
                            legend=alt.Legend(orient="top", direction="horizontal")))
        bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title=None, scale=alt.Scale(domain=[0, 0.55])),
            tooltip=[alt.Tooltip("model:N"), alt.Tooltip("defense:N"), alt.Tooltip("rate:Q", format=".1%"),
                     alt.Tooltip("low:Q", format=".1%", title="95% CI low"),
                     alt.Tooltip("high:Q", format=".1%", title="95% CI high")])
        whiskers = base.mark_rule(color="#334155", strokeWidth=1.5).encode(y="low:Q", y2="high:Q")
        labels = base.mark_text(dy=-8, fontSize=12, fontWeight="bold", color="#334155").encode(
            y="high:Q", text=alt.Text("rate:Q", format=".0%"))
        chart = alt.layer(bars, whiskers, labels).properties(width=200, height=220).facet(
            column=alt.Column("model:N", title=None, sort=["Qwen2.5-7B", "Llama-3.1-8B", "Mistral-Nemo-12B"]))
        st.altair_chart(chart)
    st.caption("Thin lines are 95% bootstrap confidence intervals over episodes. Local models via Ollama, "
               "default quantization, temperature 0, seed 0.")
    st.markdown("""
- **The firewall stopped every attack in these runs, for every model.** It enforces permission where the
  action executes, so injected text cannot grant it.
- **The prompt guard did not help.** Telling the model to ignore injected instructions lowered attack success
  for none of the three models. Qwen's rose from 12.5% to 25%, though the intervals overlap.
- **The models' own mistakes were the bigger risk.** Without the firewall, Llama issued unrequested refunds or
  emails in 35% of episodes but followed an attack in only 10%.
- **What remains:** a refund the user asked for "if policy allows it" still passes when policy says no.
  The firewall checks *who asked*, not *whether policy allows it*.""")

# ---------- Explorer ----------
with tab_explore:
    a, b, c = st.columns([1.1, 1.1, 1.4])
    model = a.selectbox("Model", list(dict.fromkeys(r["model_label"] for r in runs)))
    defense = b.selectbox("Defense", list(DEFENSES), format_func=DEFENSES.get, index=2)
    variant = c.selectbox("Episode type", ["all"] + list(VARIANTS),
                          format_func=lambda v: "All types" if v == "all" else VARIANTS[v])
    records = [r for r in by_key[(model, defense)]["records"] if variant == "all" or r["variant"] == variant]
    only_events = st.toggle("Only episodes where something was blocked or went wrong", value=True)
    if only_events:
        records = [r for r in records if outcome(r)[0] == "bad" or r["blocked"]]
    if not records:
        st.info("No episodes match. Turn off the filter above to see every episode.")
    else:
        icons = {"good": "✅", "bad": "❌", "neutral": "➖"}
        pick = st.selectbox(f"Episode ({len(records)})", range(len(records)),
                            format_func=lambda i: f"{icons[outcome(records[i])[0]]} {records[i]['id']} · "
                                                  f"{VARIANTS[records[i]['variant']]}")
        rec = records[pick]
        render_task(tasks[rec["id"]])
        st.caption(f"Expected answer: {tasks[rec['id']]['expected']['answer']} · "
                   f"{rec['latency_s']:.1f} s · {rec['tokens']:,} tokens")
        render_steps(rec)

# ---------- How it works ----------
with tab_how:
    st.markdown(f"""
**The benchmark.** 30 customer-support tasks for a mock store: order lookups, tax math, refund-policy questions,
and requests to refund or email. Each runs in a fresh sandbox, so refunds and emails are graded from the
sandbox's state, not from what the model says it did.

**The attacks.** Every task gets a copy with an instruction planted in all the order notes and policy documents
the agent can read (indirect injection). Ten get an override appended to the user's own message, and ten get an
irrelevant distractor. That makes 80 episodes per run: 40 with an attack.

**The agent.** A LangGraph planner, executor and reviewer, with five tools: `get_order`, `search_policy`,
`calculator`, `issue_refund` and `send_email`.

**The defenses.**
- *No defense:* the agent as built.
- *Prompt guard:* the system prompt tells the model to ignore instructions found in tool outputs.
- *Tool-call firewall:* the prompt guard, plus a check in code that lets a refund or email run only when the
  user's own request authorizes it.

**Limits.** These are 3 local models, one seed and 40 attacks per run. A clean result here does not make an agent
secure against attacks this suite does not contain.

**Code and raw data:** [{REPO.removeprefix('https://')}]({REPO}). This page replays `results/*.json`; it runs
no model.""")
