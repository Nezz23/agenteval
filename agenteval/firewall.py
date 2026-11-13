"""Tool-call firewall: side-effect tools run only when the user's own request authorizes them.
Authorization is read from the user turn, never from tool outputs, so injected text cannot grant it."""
import re

REFUND = re.compile(r"\brefund order (\w+)", re.I)  # imperative "Refund order X", not "eligible for a refund"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def authorizer(request: str, orders: dict):
    refundable = {o.upper() for o in REFUND.findall(request)}
    recipients = {e.lower() for e in EMAIL.findall(request)}

    def check(tool: str, args: dict) -> str | None:
        """Return why the call is blocked, or None to allow it."""
        if tool == "issue_refund":
            oid = args["order_id"].upper()
            if oid not in refundable:
                return f"the user did not ask to refund {oid}"
            if args["amount"] > orders.get(oid, {}).get("subtotal", 0):
                return "refund exceeds the order subtotal"
        if tool == "send_email" and args["to"].lower() not in recipients:
            return f"the user did not ask to email {args['to']}"
        return None
    return check
