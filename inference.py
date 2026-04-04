import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from client import DataAnalysisClient
from models import DataAction

load_dotenv()
TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_STEPS = 15
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen3.5-9B"
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
1. To execute code: {{"action": "execute_code", "code": "your python code here"}}
2. To submit answer: {{"action": "submit_answer", "answer": "your answer here"}}
</RESPONSE>

<NOTE>
Respond with ONLY the JSON, no other text.
</NOTE>
"""

FALLBACK_ACTION = json.dumps({"action": "submit_answer", "answer": "unknown"})


def parse_model_action(response_text: str) -> dict:
    """Parse the model's raw text response into an action dict.

    Handles plain JSON and markdown code block wrapping.

    Args:
        response_text: Raw string returned by the model.

    Returns:
        Parsed action dict, or a fallback submit_answer on failure.
    """
    text = response_text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
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
    result = env_client.reset(task_id=task_id)
    obs = result.observation

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

    print(f"\n--- Task {task_id} ---")
    print(f"Question: {obs.task_description}")

    for step in range(MAX_STEPS):
        try:
            completion = openai_client.chat.completions.create(
                model=os.environ["MODEL_NAME"],
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  Model request failed ({exc}). Using fallback action.")
            response_text = FALLBACK_ACTION

        action = parse_model_action(response_text)
        action_type = action.get("action", "")
        print(f"  Step {step + 1}: model suggested -> {action_type}")

        if action_type == "execute_code":
            step_result = env_client.step(DataAction(action_type="execute_code", code=action.get("code", "")))
            step_obs = step_result.observation
            result_text = f"Output: {step_obs.output}" if not step_obs.error else f"Error: {step_obs.error}"
            print(f"    -> {result_text[:120]}")

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": [{"type": "text", "text": result_text}]})

        elif action_type == "submit_answer":
            step_result = env_client.step(DataAction(action_type="submit_answer", answer=action.get("answer", "")))
            step_obs = step_result.observation
            score = step_obs.metadata.get("score", 0.0) if step_obs.metadata else step_result.reward
            print(f"    -> submitted: '{action.get('answer', '')}' | score: {score:.2f}")
            return float(score)

        else:
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

    print(f"  Reached max steps ({MAX_STEPS}). No answer submitted.")
    return 0.0


def main():
    print("Executing Data Analysis Environment")
    openai_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    scores = {}
    difficulties = {1: "Easy", 2: "Medium", 3: "Hard"}

    for task_id in [1, 2, 3]:
        with DataAnalysisClient(base_url=ENV_SERVER_URL).sync() as env_client:
            score = run_task(openai_client, env_client, task_id)
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
