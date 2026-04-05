import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from client import DataAnalysisClient
from models import DataAction

load_dotenv()
TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_STEPS = 15
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL") or "https://mohammed-altaf-dataanalysis-env.hf.space"

SYSTEM_PROMPT = """
<ROLE>
You are a data analyst. You are given a dataset loaded as a pandas DataFrame called `df`.
You can execute Python/pandas code to explore the dataset and answer the question.
</ROLE>

<RULES>
- Use `print()` to see results of your code
- The DataFrame `df` is pre-loaded with pandas as `pd` and numpy as `np`
- When you have the answer, submit it in the exact format requested
- Be precise with numbers and formatting
</RULES>

<RESPONSE>
Respond with JSON in one of these formats:
1. To execute code: {"action": "execute_code", "code": "your python code here"}
2. To submit answer: {"action": "submit_answer", "answer": "your answer here"}
</RESPONSE>

<NOTE>
Respond with ONLY the JSON, no other text.
</NOTE>
"""

FALLBACK_ACTION = json.dumps({"action": "submit_answer", "answer": "unknown"})


def parse_model_action(response_text: str) -> dict:
    """Parse the model's raw text response into an action dict.

    Handles multiple LLM response edge cases:
    - Markdown code blocks (```json ... ``` or ``` ... ```)
    - Double curly braces e.g. {{"key": "value"}}
    - Single quotes instead of double quotes e.g. {'key': 'value'}
    - Python literals: True/False/None → true/false/null
    - Trailing commas in objects/arrays e.g. {"key": "value",}
    - Extra text/prose before or after the JSON blob
    - Escaped single quotes inside single-quoted strings
    - Whitespace and newline noise

    Args:
        response_text: Raw string returned by the model.

    Returns:
        Parsed action dict, or a fallback submit_answer on failure.
    """

    def attempt_parse(text: str) -> dict:
        return json.loads(text)

    def apply_fixes(text: str) -> str:
        # Strip markdown code blocks
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

        # Double curly braces → single
        text = text.replace("{{", "{").replace("}}", "}")

        # Python literals → JSON literals
        text = re.sub(r"\bTrue\b", "true", text)
        text = re.sub(r"\bFalse\b", "false", text)
        text = re.sub(r"\bNone\b", "null", text)

        # Trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Single quote handling — two distinct cases:
        #
        # Case 1: Entire JSON is wrapped in outer single quotes
        #   e.g. '{"action": "x", "code": "df[\'col\']"}'
        #   → strip the outer quotes and unescape internal \'
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1].replace("\\'", "'")

        # Case 2: JSON itself uses single quotes as delimiters
        #   e.g. {'action': 'execute_code', 'code': 'print()'}
        #   → only apply when structure looks single-quote delimited
        #   → avoids corrupting double-quoted values that contain bracket notation
        elif text.startswith("{'") or ("': " in text and '": ' not in text):
            text = re.sub(
                r"'((?:\\'|[^'])*)'", lambda m: '"' + m.group(1).replace("\\'", "'").replace('"', '\\"') + '"', text
            )

        return text

    def extract_json_blob(text: str) -> str:
        """Extract the first {...} or [...] blob from text with surrounding prose."""
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return match.group(1)
        return text

    text = response_text.strip()

    try:
        return attempt_parse(text)
    except json.JSONDecodeError:
        pass

    try:
        return attempt_parse(apply_fixes(text))
    except json.JSONDecodeError:
        pass

    try:
        blob = extract_json_blob(text)
        return attempt_parse(apply_fixes(blob))
    except json.JSONDecodeError:
        pass

    try:
        fixed = apply_fixes(text)
        blob = extract_json_blob(fixed)
        return attempt_parse(blob)
    except json.JSONDecodeError:
        pass

    print(f"JSON Decoding Error while parsing action in response text: {response_text}")
    return json.loads(FALLBACK_ACTION)


def run_task(openai_client: OpenAI, env_client: DataAnalysisClient, task_id: int) -> float:
    """Run a single task episode using the language model as the agent.

    Args:
        openai_client: Configured OpenAI-compatible client.
        env_client: Connected DataAnalysisClient (sync wrapper).
        task_id: Task to evaluate (1 = easy, 2 = medium, 3 = hard).

    Returns:
        Final score for this task between 0.0 and 1.0.
    """
    difficulties = {1: "Easy", 2: "Medium", 3: "Hard"}
    result = env_client.reset(task_id=task_id)
    obs = result.observation
    history = []

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Task: {obs.task_description}\n\nDataset Info:\n{obs.dataset_info}",
                }
            ],
        },
    ]

    print(f"\n{'=' * 55}")
    print(f"Episode Start — Task {task_id} ({difficulties.get(task_id, 'Unknown')})")
    print(f"Question: {obs.task_description}")
    print(f"{'=' * 55}")

    for step in range(MAX_STEPS):
        try:
            completion = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )

            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            failure_msg = f"Model request failed ({exc}). Using fallback action."
            print(f"  {failure_msg}")
            response_text = FALLBACK_ACTION

        action = parse_model_action(response_text)
        action_type = action.get("action", "")
        print(f"  Step {step + 1}: model suggested -> {action_type}")

        if action_type == "execute_code":
            step_result = env_client.step(DataAction(action_type="execute_code", code=action.get("code", "")))
            step_obs = step_result.observation
            reward = step_result.reward or 0.0
            result_text = f"Output: {step_obs.output}" if not step_obs.error else f"Error: {step_obs.error}"
            history_line = f"Step {step + 1}: execute_code -> reward {reward:+.2f}"
            history.append(history_line)
            print(f"    Reward: {reward:+.2f} | {result_text[:100]}")

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": [{"type": "text", "text": result_text}]})

        elif action_type == "submit_answer":
            step_result = env_client.step(DataAction(action_type="submit_answer", answer=action.get("answer", "")))
            step_obs = step_result.observation
            score = step_obs.metadata.get("score", 0.0) if step_obs.metadata else step_result.reward
            history_line = f"Step {step + 1}: submit_answer '{action.get('answer', '')}' -> score {score:.2f}"
            history.append(history_line)
            print(f"    Reward: {score:+.2f} | Done: {step_result.done}")
            print("  Episode complete.")
            _log_episode_end(task_id, step + 1, float(score), history)
            return float(score)

        else:
            history_line = f"Step {step + 1}: unknown action '{action_type}'"
            history.append(history_line)
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Unknown action '{action_type}'. Use 'execute_code' or 'submit_answer'.",
                        }
                    ],
                }
            )

    print(f"  Reached max steps ({MAX_STEPS}).")
    _log_episode_end(task_id, MAX_STEPS, 0.0, history)
    return 0.0


def _log_episode_end(task_id: int, steps_taken: int, final_score: float, history: list) -> None:
    """Print a structured end-of-episode summary.

    Args:
        task_id: The task that just finished.
        steps_taken: Total steps taken in the episode.
        final_score: Final graded score (0.0 to 1.0).
        history: List of history_line strings logged during the episode.
    """
    print(f"\n--- Episode End — Task {task_id} ---")
    print(f"  Steps taken  : {steps_taken}")
    print(f"  Final score  : {final_score:.2f}")
    print("  Step history :")
    for line in history:
        print(f"    {line}")


def main():
    print("Executing Data Analysis Environment")
    openai_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    scores = {}
    difficulties = {1: "Easy", 2: "Medium", 3: "Hard"}

    with DataAnalysisClient(base_url=ENV_SERVER_URL).sync() as env_client:
        for task_id in [1, 2, 3]:
            score = run_task(openai_client=openai_client, env_client=env_client, task_id=task_id)
            scores[task_id] = score

    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    for task_id, score in scores.items():
        print(f"  Task {task_id} ({difficulties[task_id]:6s}): {score:.2f}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average Score : {avg:.2f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
