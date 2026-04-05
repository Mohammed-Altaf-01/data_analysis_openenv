import json
import re
from typing import Any

FALLBACK_ACTION = json.dumps({"action": "submit_answer", "answer": "unknown"})


def _sanitize_string_value(match: re.Match) -> str:
    """
    Receives a regex match of ("key": "value") and cleans only the value part.
    Escapes unescaped newlines, tabs, carriage returns, and inner double quotes.
    NOTE: This is the core trick LangChain uses in _replace_new_line / _custom_parser.
    """
    opening = match.group(1)
    value = match.group(2)
    closing = match.group(3)

    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r'\\"', value)  # escape unescaped inner quotes

    return opening + value + closing


def _sanitize_all_string_values(text: str) -> str:
    """
    Apply _sanitize_string_value to every JSON string value in the text.
    Uses re.DOTALL so values that span multiple lines are handled correctly.
    NOTE: Generalised version of LangChain's _custom_parser (which only targeted action_input).
    """
    return re.sub(
        r'("[\w]+"\s*:\s*")(.*?)(")',
        _sanitize_string_value,
        text,
        flags=re.DOTALL,
    )


def _preprocess(text: str) -> str:
    """Fix common LLM response quirks before attempting JSON parsing."""

    # Strip markdown code fences  (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Double curly braces  {{"k": "v"}}  →  {"k": "v"}
    text = text.replace("{{", "{").replace("}}", "}")
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Outer single-quote wrap  '{"k": "v"}'  →  {"k": "v"}
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1].replace("\\'", "'")

    return text.strip()


def _extract_json_blob(text: str) -> str:
    """
    Pull out the first {...} or [...] blob from text that has prose around it.
    Inspired by LangChain's _json_markdown_re fallback in parse_json_markdown.
    """
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    return match.group(1) if match else text


def _parse_partial_json(s: str) -> Any:
    """
    Parse JSON that may be truncated / missing closing brackets.
    Adapted from LangChain's parse_partial_json (originally from open-interpreter).
    Uses a stack to track open containers and closes them before parsing.
    """
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    stack = []
    is_inside = False
    position = 0

    for i, char in enumerate(s):
        if is_inside:
            if char == '"' and s[i - 1] != "\\":
                is_inside = False
        else:
            if char == '"':
                is_inside = True
                stack.append('"')
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if stack and stack[-1] in "{[":
                    stack.pop()
        position = i

    completed = s[: position + 1]
    for bracket in reversed(stack):
        if bracket == '"':
            completed += '"'
        elif bracket == "{":
            completed += "}"
        elif bracket == "[":
            completed += "]"

    return json.loads(completed)


def _extract_fields_direct(text: str) -> dict:
    """Extract action fields using greedy regex anchored to the last closing quote.

    Handles the case where the model emits unescaped double-quote characters inside
    a "code" or "answer" value (e.g. df["col"]).  The non-greedy `(.*?)` in
    _sanitize_all_string_values stops at the *first* inner quote and corrupts the
    output.  By using a greedy `(.*)` anchored with a lookahead for the last `"}`
    boundary we capture the full value regardless of inner quotes.

    Args:
        text: Pre-processed JSON-like string.

    Returns:
        Dict with 'action' and 'code'/'answer' keys.

    Raises:
        ValueError: If the action field cannot be found or the value cannot be
            extracted for the detected action type.
    """
    action_match = re.search(r'"action"\s*:\s*"(\w+)"', text)
    if not action_match:
        raise ValueError("No 'action' field found")
    action_type = action_match.group(1)

    if action_type == "execute_code":
        m = re.search(r'"code"\s*:\s*"(.*)"(?=\s*})', text, re.DOTALL)
        if m:
            return {"action": "execute_code", "code": m.group(1)}
    elif action_type == "submit_answer":
        m = re.search(r'"answer"\s*:\s*"(.*)"(?=\s*})', text, re.DOTALL)
        if m:
            return {"action": "submit_answer", "answer": m.group(1)}

    raise ValueError(f"Could not extract value for action_type={action_type!r}")


def parse_model_action(response_text: str) -> dict:
    """
    Parse a raw LLM response into an action dict.

    Pipeline (mirrors LangChain's JsonOutputParser internals):
      1. _preprocess      – fix markdown fences, double braces, Python literals …
      2. _sanitize_all_string_values – escape unescaped quotes/newlines inside values
      3. _extract_json_blob           – strip surrounding prose
      4. _parse_partial_json          – close truncated JSON with a stack algorithm

    Each strategy is tried independently so a failure in one doesn't block others.
    """
    text = response_text.strip()

    strategies = [
        lambda t: _parse_partial_json(t),
        lambda t: _parse_partial_json(_sanitize_all_string_values(_preprocess(t))),
        lambda t: _parse_partial_json(_sanitize_all_string_values(_preprocess(_extract_json_blob(t)))),
        lambda t: _parse_partial_json(_sanitize_all_string_values(_extract_json_blob(_preprocess(t)))),
        lambda t: _parse_partial_json(_sanitize_all_string_values(t)),
        lambda t: _extract_fields_direct(_preprocess(_extract_json_blob(t))),
        lambda t: _extract_fields_direct(_extract_json_blob(t)),
    ]

    for strategy in strategies:
        try:
            return strategy(text)
        except (json.JSONDecodeError, ValueError):
            continue

    print(f"JSON Decoding Error while parsing action in response text: {response_text}")
    return json.loads(FALLBACK_ACTION)
