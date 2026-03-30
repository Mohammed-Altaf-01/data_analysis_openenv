"""Baseline inference script for the Data Analysis Agent environment.

Uses the OpenAI API to run a model (gpt-4o-mini) against all 3 tasks
and produces reproducible baseline scores.

Usage:
    OPENAI_API_KEY=sk-... uv run python baseline.py
    OPENAI_API_KEY=sk-... uv run python baseline.py --base-url http://localhost:8000
"""

import argparse
import json
import os
import sys

import requests
from openai import OpenAI

SYSTEM_PROMPT = """You are a data analyst. You are given a dataset loaded as a pandas DataFrame called `df`.
You can execute Python/pandas code to explore the dataset and answer the question.

Rules:
- Use `print()` to see results of your code
- The DataFrame `df` is pre-loaded with pandas as `pd` and numpy as `np`
- When you have the answer, submit it in the exact format requested
- Be precise with numbers and formatting

Respond with JSON in one of these formats:
1. To execute code: {{"action": "execute_code", "code": "your python code here"}}
2. To submit answer: {{"action": "submit_answer", "answer": "your answer here"}}

Respond with ONLY the JSON, no other text."""


def run_task(client: OpenAI, base_url: str, task_id: int, max_steps: int = 15) -> float:
    """Run a single task using the OpenAI API as the agent.

    Args:
        client: The OpenAI client instance.
        base_url: The environment server base URL.
        task_id: Which task to run (1, 2, or 3).
        max_steps: Maximum agent steps before giving up.

    Returns:
        The final score for this task (0.0 to 1.0).
    """
    # Reset environment with the specified task
    reset_resp = requests.post(
        f"{base_url}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    reset_data = reset_resp.json()
    obs = reset_data.get("observation", reset_data)

    task_desc = obs.get("task_description", "")
    dataset_info = obs.get("dataset_info", "")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Task: {task_desc}\n\nDataset Info:\n{dataset_info}",
        },
    ]

    print(f"\n--- Task {task_id} ---")
    print(f"Question: {task_desc}")

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
            # Send code execution to environment
            step_resp = requests.post(
                f"{base_url}/step",
                json={
                    "action_type": "execute_code",
                    "code": action.get("code", ""),
                },
                timeout=30,
            )
            step_data = step_resp.json()
            step_obs = step_data.get("observation", step_data)

            output = step_obs.get("output", "")
            error = step_obs.get("error", "")
            result_text = f"Output: {output}" if not error else f"Error: {error}"
            print(f"  Step {step + 1}: execute_code -> {result_text[:100]}")

            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": result_text})

        elif action_type == "submit_answer":
            # Submit final answer
            step_resp = requests.post(
                f"{base_url}/step",
                json={
                    "action_type": "submit_answer",
                    "answer": action.get("answer", ""),
                },
                timeout=30,
            )
            step_data = step_resp.json()
            step_obs = step_data.get("observation", step_data)

            score = step_obs.get("metadata", {}).get("score", 0.0)
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

    client = OpenAI(api_key=api_key)

    print("=" * 50)
    print("Data Analysis Agent - Baseline Inference")
    print(f"Server: {args.base_url}")
    print(f"Model: gpt-4o-mini")
    print("=" * 50)

    scores = {}
    for task_id in [1, 2, 3]:
        score = run_task(client, args.base_url, task_id)
        scores[task_id] = score

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    difficulties = {1: "Easy", 2: "Medium", 3: "Hard"}
    for task_id, score in scores.items():
        print(f"  Task {task_id} ({difficulties[task_id]}): {score:.2f}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average Score: {avg:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
