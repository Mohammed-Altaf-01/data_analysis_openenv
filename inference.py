"""Inference script for the Data Analysis Agent environment.

Runs a language model agent against all 3 tasks and reports scores.
Uses the OpenAI-compatible client pointed at API_BASE_URL.

Required environment variables (set in .env or shell):
    API_BASE_URL   OpenAI-compatible LLM API endpoint
    MODEL_NAME     Model identifier to use for inference
    HF_TOKEN       API key (Hugging Face token or other provider key)

Optional:
    ENV_SERVER_URL  Environment server URL (default: http://localhost:7860)

Usage:
    uv run python inference.py
    uv run python inference.py --env-url http://localhost:8000
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from client import DataAnalysisClient
from models import DataAction

# Load .env file if present (safe — does not override already-set shell vars)
load_dotenv()

TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_STEPS = 15  # Per task — keeps total runtime well under 20 min

SYSTEM_PROMPT = """You are a data analyst. You are given a dataset loaded as a pandas DataFrame called `df`.
You can execute Python/pandas code to explore the dataset and answer the question.

Rules:
- Use `print()` to see results of your code
- The DataFrame `df` is pre-loaded with pandas as `pd` and numpy as `np`
- When you have the answer, submit it in the exact format requested
- Be precise with numbers and formatting

Respond with JSON in one of these formats:
1. To execute code: {"action": "execute_code", "code": "your python code here"}
2. To submit answer: {"action": "submit_answer", "answer": "your answer here"}

Respond with ONLY the JSON, no other text."""

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
            step_result = env_client.step(
                DataAction(action_type="execute_code", code=action.get("code", ""))
            )
            step_obs = step_result.observation
            result_text = f"Output: {step_obs.output}" if not step_obs.error else f"Error: {step_obs.error}"
            print(f"    -> {result_text[:120]}")

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": [{"type": "text", "text": result_text}]})

        elif action_type == "submit_answer":
            step_result = env_client.step(
                DataAction(action_type="submit_answer", answer=action.get("answer", ""))
            )
            step_obs = step_result.observation
            score = step_obs.metadata.get("score", 0.0) if step_obs.metadata else step_result.reward
            print(f"    -> submitted: '{action.get('answer', '')}' | score: {score:.2f}")
            return float(score)

        else:
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"Unknown action '{action_type}'. Use 'execute_code' or 'submit_answer'."}],
            })

    print(f"  Reached max steps ({MAX_STEPS}). No answer submitted.")
    return 0.0


def main():
    """Run inference across all 3 tasks and print final scores."""
    parser = argparse.ArgumentParser(description="Data Analysis Agent inference script")
    parser.add_argument(
        "--env-url",
        default=os.environ.get("ENV_SERVER_URL", "http://localhost:7860"),
        help="Environment server URL (default: http://localhost:7860)",
    )
    args = parser.parse_args()

    # Validate required environment variables
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.environ.get(v)]
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("Set them in your shell or create a .env file (see .env.example).")
        sys.exit(1)

    openai_client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["HF_TOKEN"],
    )

    print("=" * 55)
    print("Data Analysis Agent — Inference")
    print(f"Server : {args.env_url}")
    print(f"Model  : {os.environ['MODEL_NAME']}")
    print(f"API    : {os.environ['API_BASE_URL']}")
    print("=" * 55)

    scores = {}
    difficulties = {1: "Easy", 2: "Medium", 3: "Hard"}

    # Each task gets its own isolated WebSocket session
    for task_id in [1, 2, 3]:
        with DataAnalysisClient(base_url=args.env_url).sync() as env_client:
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
