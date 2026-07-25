import json


def make_status(enabled, healthy, reason, **fields):
    payload = {
        "enabled": bool(enabled),
        "healthy": bool(healthy),
        "reason": str(reason),
    }
    payload.update(fields)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
