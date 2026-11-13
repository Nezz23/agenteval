"""Sandboxed tool environment. Each task gets a fresh Env so side effects are graded from state."""
import ast
import copy
import json
import operator

from langchain_core.tools import tool

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def safe_eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression: {expr}")
    return ev(ast.parse(expr, mode="eval").body)


class Env:
    def __init__(self, data: dict):
        self.orders = copy.deepcopy(data.get("orders", {}))
        self.docs = copy.deepcopy(data.get("docs", []))
        self.refunds: list[tuple[str, float]] = []
        self.emails: list[dict] = []
        self.blocked: list[dict] = []


def make_tools(env: Env, authorize=None):
    def blocked(name, args):
        reason = authorize and authorize(name, args)
        if reason:
            env.blocked.append({"tool": name, "args": args, "reason": reason})
            return f"BLOCKED by policy: {reason}."

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression, e.g. '50*1.08'."""
        return str(round(safe_eval(expression), 4))

    @tool
    def get_order(order_id: str) -> str:
        """Look up an order by id. Returns JSON with subtotal, status, delivered_days_ago and note."""
        order = env.orders.get(order_id)
        return json.dumps(order) if order else f"Order {order_id} not found."

    @tool
    def search_policy(query: str) -> str:
        """Search store policy documents by keyword."""
        words = {w for w in query.lower().split() if len(w) > 2}
        scored = sorted(env.docs, key=lambda d: -len(words & set(d.lower().split())))
        hits = [d for d in scored if words & set(d.lower().split())][:2]
        return "\n".join(hits) or "No matching policy."

    @tool
    def issue_refund(order_id: str, amount: float) -> str:
        """Issue a refund for an order. Irreversible."""
        if msg := blocked("issue_refund", {"order_id": order_id, "amount": float(amount)}):
            return msg
        env.refunds.append((order_id, float(amount)))
        return f"Refunded {amount} for {order_id}."

    @tool
    def send_email(to: str, body: str) -> str:
        """Send an email to a recipient."""
        if msg := blocked("send_email", {"to": to}):
            return msg
        env.emails.append({"to": to, "body": body})
        return f"Email sent to {to}."

    return [calculator, get_order, search_policy, issue_refund, send_email]
