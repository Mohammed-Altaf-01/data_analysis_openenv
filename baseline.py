"""Baseline inference script for the Data Analysis Agent environment.

Uses the OpenAI API to run a model (gpt-4o-mini) against all 3 tasks
and produces reproducible baseline scores.

The script uses DataAnalysisClient (WebSocket) because the HTTP endpoints
are stateless — each request gets a fresh env instance. State (namespace,
task, dataset) only persists within a WebSocket session.

Usage:
    OPENAI_API_KEY=sk-... uv run python baseline.py
    OPENAI_API_KEY=sk-... uv run python baseline.py --base-url http://localhost:8000
"""

import argparse
import json
import os
import sys

from openai import OpenAI
from client import DataAnalysisClient
from models import DataAction

SYSTEM_PROMPT = """Y
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


def run_task(openai_client: OpenAI, env_client: DataAnalysisClient, task_id: int, max_steps: int = 15) -> float:
    """Run a single task using the OpenAI API as the agent.

    Args:
        openai_client: The OpenAI client instance.
        env_client: The connected DataAnalysisClient (sync wrapper).
        task_id: Which task to run (1, 2, or 3).
        max_steps: Maximum agent steps before giving up.

    Returns:
        The final score for this task (0.0 to 1.0).
    """
    result = env_client.reset(task_id=task_id)
    obs = result.observation

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Task: {obs.task_description}\n\nDataset Info:\n{obs.dataset_info}",
        },
    ]

    print(f"\n--- Task {task_id} ---")
    print(f"Question: {obs.task_description}")

    for step in range(max_steps):
        response = openai_client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=messages,
            temperature=0.0,
        )
        assistant_msg = response.choices[0].message.content.strip()

        # Parse the agent's JSON response
        try:
            # Handle markdown code blocks if present
            if assistant_msg.startswith("```"):
                assistant_msg = assistant_msg.split("```")[1]
                if assistant_msg.startswith("json"):
                    assistant_msg = assistant_msg[4:]
                assistant_msg = assistant_msg.strip()
            action = json.loads(assistant_msg)
        except json.JSONDecodeError:
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid JSON. Please respond with valid JSON only.",
                }
            )
            continue

        action_type = action.get("action", "")

        if action_type == "execute_code":
            result = env_client.step(DataAction(action_type="execute_code", code=action.get("code", "")))
            obs = result.observation
            result_text = f"Output: {obs.output}" if not obs.error else f"Error: {obs.error}"
            print(f"  Step {step + 1}: execute_code -> {result_text[:120]}")
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": result_text})

        elif action_type == "submit_answer":
            result = env_client.step(DataAction(action_type="submit_answer", answer=action.get("answer", "")))
            obs = result.observation
            score = obs.metadata.get("score", 0.0) if obs.metadata else result.reward
            print(f"  Step {step + 1}: submit_answer -> '{action.get('answer', '')}'")
            print(f"  Score: {score:.2f}")
            return score

        else:
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append(
                {
                    "role": "user",
                    "content": f"Unknown action '{action_type}'. Use 'execute_code' or 'submit_answer'.",
                }
            )

    print("  Max steps reached without submitting an answer.")
    return 0.0


def main():
    """Run baseline inference across all 3 tasks and report scores."""
    parser = argparse.ArgumentParser(description="Baseline inference for Data Analysis Env")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Environment server URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required.")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)

    print("=" * 50)
    print("Data Analysis Agent - Baseline Inference")
    print(f"Server: {args.base_url}")
    print("Model: gpt-4o-mini")
    print("=" * 50)

    scores = {}
    difficulties = {1: "Easy", 2: "Medium", 3: "Hard"}

    for task_id in [1, 2, 3]:
        with DataAnalysisClient(base_url=args.base_url).sync() as env_client:
            score = run_task(openai_client, env_client, task_id)
            scores[task_id] = score

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for task_id, score in scores.items():
        print(f"  Task {task_id} ({difficulties[task_id]}): {score:.2f}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average Score: {avg:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
