"""Small, fixed, auto-scored agent-style test set, run against any
OpenAI-compatible chat endpoint (llama-server, Ollama, OJP's daemon).
Greedy decoding (temperature 0) so runs are reproducible.

Categories: arithmetic, instruction-following, tool calls (raw JSON,
no grammar constraint - tests the model, not the sampler), and router
classification (the job a small always-on router model would do).
"""

import json
import re
import urllib.request

TOOL_SYSTEM = (
    "You control a computer through tools. Available tools:\n"
    '- set_timer(minutes: int)\n- get_weather(city: str)\n- open_app(name: str)\n'
    "- volume(level: int)  // 0-100\n"
    'Respond with ONLY a JSON object like {"tool": "<name>", "args": {...}} and nothing else.'
)

ROUTER_LABELS = ["answer_directly", "coding", "reasoning", "device_control"]
ROUTER_SYSTEM = (
    "Classify the user's request into exactly one category and reply with ONLY the category name.\n"
    "answer_directly: casual chat or a simple fact.\n"
    "coding: writing, fixing, or explaining code.\n"
    "reasoning: multi-step problems, planning, analysis, or math word problems.\n"
    "device_control: controlling apps, settings, timers, volume, or hardware on this computer."
)


def _int_answer(expected):
    return lambda r: re.findall(r"-?\d+", r.replace(",", ""))[:1] == [str(expected)]


def _tool(expected_tool, expected_args):
    def check(reply):
        m = re.search(r"\{.*\}", reply, re.S)
        if not m:
            return False
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return False
        args = obj.get("args", {})
        return obj.get("tool") == expected_tool and all(
            str(args.get(k)).lower() == str(v).lower() for k, v in expected_args.items()
        )

    return check


def _label(expected):
    return lambda r: r.strip().strip(".").lower() == expected


CASES = [
    # arithmetic
    ("arith", None, "What is 47 + 38? Reply with only the number.", _int_answer(85)),
    ("arith", None, "What is 12 * 13? Reply with only the number.", _int_answer(156)),
    ("arith", None, "What is 1000 - 377? Reply with only the number.", _int_answer(623)),
    ("arith", None, "What is 144 / 12? Reply with only the number.", _int_answer(12)),
    ("arith", None, "I have 3 boxes with 7 apples each and eat 5. How many apples are left? Reply with only the number.", _int_answer(16)),
    ("arith", None, "What is 25% of 240? Reply with only the number.", _int_answer(60)),
    ("arith", None, "A train leaves at 14:00 and arrives at 17:30. How many minutes is the trip? Reply with only the number.", _int_answer(210)),
    ("arith", None, "What is 9 squared minus 19? Reply with only the number.", _int_answer(62)),
    # instruction following
    ("instruct", None, "Describe the ocean in exactly three words. Reply with only those three words.",
     lambda r: len(re.findall(r"[A-Za-z']+", r)) == 3),
    ("instruct", None, "Write one short sentence about cats, entirely in lowercase letters.",
     lambda r: r.strip() != "" and r == r.lower() and any(c.isalpha() for c in r)),
    ("instruct", None, "Is the sun a star? Answer with only Yes or No.",
     lambda r: r.strip().strip(".").lower() == "yes"),
    ("instruct", None, "List three colors, one per line, with no numbering and no other text.",
     lambda r: len([l for l in r.strip().splitlines() if l.strip()]) == 3
     and not any(re.match(r"\s*(\d|[-*])", l) for l in r.splitlines())),
    ("instruct", None, 'Return a JSON object with keys "name" and "age" for a person named Alice who is 30. Output only the JSON.',
     lambda r: _alice_json(r)),
    ("instruct", None, "Repeat the word 'ready' exactly twice, separated by a single space, and nothing else.",
     lambda r: r.strip().strip(".").lower() == "ready ready"),
    ("instruct", None, "Give a one-word answer: what is the capital of Japan?",
     lambda r: r.strip().strip(".").lower() == "tokyo"),
    ("instruct", None, "Reply with the word OK in uppercase and nothing else.",
     lambda r: r.strip().strip(".") == "OK"),
    # tool calls
    ("tool", TOOL_SYSTEM, "Set a timer for 5 minutes.", _tool("set_timer", {"minutes": 5})),
    ("tool", TOOL_SYSTEM, "What's the weather like in Paris?", _tool("get_weather", {"city": "paris"})),
    ("tool", TOOL_SYSTEM, "Open Firefox.", _tool("open_app", {"name": "firefox"})),
    ("tool", TOOL_SYSTEM, "Turn the volume down to 20.", _tool("volume", {"level": 20})),
    ("tool", TOOL_SYSTEM, "I need a 45 minute timer please.", _tool("set_timer", {"minutes": 45})),
    ("tool", TOOL_SYSTEM, "Can you launch Steam?", _tool("open_app", {"name": "steam"})),
    ("tool", TOOL_SYSTEM, "Is it going to rain in Tokyo today?", _tool("get_weather", {"city": "tokyo"})),
    ("tool", TOOL_SYSTEM, "Mute the sound.", _tool("volume", {"level": 0})),
    # router classification
    ("router", ROUTER_SYSTEM, "Hey, how's it going?", _label("answer_directly")),
    ("router", ROUTER_SYSTEM, "What's the capital of Canada?", _label("answer_directly")),
    ("router", ROUTER_SYSTEM, "Write a Python function that reverses a string.", _label("coding")),
    ("router", ROUTER_SYSTEM, "Why does my Rust code say 'borrowed value does not live long enough'?", _label("coding")),
    ("router", ROUTER_SYSTEM, "Fix the bug in this JavaScript: for (i = 0; i < arr.length; i--) {}", _label("coding")),
    ("router", ROUTER_SYSTEM, "Plan a three-day trip to Rome on a tight budget.", _label("reasoning")),
    ("router", ROUTER_SYSTEM, "If a shirt costs $20 after a 20% discount, what was the original price? Explain.", _label("reasoning")),
    ("router", ROUTER_SYSTEM, "Compare the pros and cons of renting versus buying a house.", _label("reasoning")),
    ("router", ROUTER_SYSTEM, "Turn the volume up.", _label("device_control")),
    ("router", ROUTER_SYSTEM, "Open Spotify and play my liked songs.", _label("device_control")),
]


def _alice_json(reply):
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    return obj.get("name") == "Alice" and obj.get("age") in (30, "30")


def chat(base_url: str, system, user: str) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    body = json.dumps({"messages": messages, "temperature": 0, "max_tokens": 96, "seed": 0}).encode()
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def run(base_url: str) -> dict:
    per_cat, failures = {}, []
    for cat, system, prompt, check in CASES:
        reply = chat(base_url, system, prompt)
        ok = bool(check(reply))
        passed, total = per_cat.get(cat, (0, 0))
        per_cat[cat] = (passed + ok, total + 1)
        if not ok:
            failures.append({"category": cat, "prompt": prompt, "reply": reply[:200]})
    return {"scores": {c: f"{p}/{t}" for c, (p, t) in per_cat.items()},
            "total": f"{sum(p for p, _ in per_cat.values())}/{len(CASES)}",
            "failures": failures}
