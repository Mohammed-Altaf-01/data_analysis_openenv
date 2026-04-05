import json
import os
import re
from typing import Any, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

from client import DataAnalysisClient
from helpers.response_parser import FALLBACK_ACTION, parse_model_action
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
You are a data analyst. You have two data sources available:
1. `df` — a pandas DataFrame (sales CSV, pre-loaded)
2. A SQLite database at `db_path` — contains additional tables (e.g. customer_profiles, product_catalog)
</ROLE>

<RULES>
- Use `print()` to output results
- `pd`, `np`, `sqlite3`, and `db_path` are already in scope — NEVER use import statements (they will fail)
- `df` is a pandas DataFrame — use pandas operations on it, NEVER SQL
- To query the SQLite database use: `conn = sqlite3.connect(db_path)` then `pd.read_sql(query, conn)`
- For cross-source tasks: query SQLite for the extra data, then merge with `df` using pandas
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


def log_start(task: str, env: str, model: str) -> None:
    """Log the start of a task episode.

    Args:
        task: Task identifier string.
        env: Environment name or URL.
        model: Model name used for inference.
    """
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: Union[dict, str], reward: float, done: bool, error: Optional[str]) -> None:
    """Log a single environment step.

    Args:
        step: Current step number.
        action: Action type executed.
        reward: Reward received from the environment.
        done: Whether the episode is complete.
        error: Error message if the step failed, else None.
    """
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log the end of a task episode.

    Args:
        success: Whether the agent submitted a correct answer.
        steps: Total number of steps taken.
        score: Final graded score (0.0 to 1.0).
        rewards: List of per-step rewards collected during the episode.
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}\n", flush=True)


def run_task(openai_client: OpenAI, env_client: Any, task_id: int) -> float:
    """Run a single task episode using the language model as the agent.

    Args:
        openai_client: Configured OpenAI-compatible client.
        env_client: Connected DataAnalysisClient (sync wrapper).
        task_id: Task to evaluate (1 - 6)

    Returns:
        Final score for this task between 0.0 and 1.0.
    """
    try:
        result = env_client.reset(task_id=task_id)
    except Exception as exc:
        print(f"[DEBUG] env reset failed: {exc}", flush=True)
        return 0.0

    obs = result.observation
    rewards: List[float] = []

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

    log_start(task=str(task_id), env=ENV_SERVER_URL, model=MODEL_NAME)

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
            print(f"[DEBUG] Model request failed: {exc}", flush=True)
            response_text = FALLBACK_ACTION
        action = parse_model_action(response_text)
        action_type = action.get("action", "")

        if action_type == "execute_code":
            try:
                exec_result = env_client.step(DataAction(action_type="execute_code", code=action.get("code", "")))
                exec_obs = exec_result.observation
                reward = exec_result.reward or 0.0
                done = exec_result.done
            except Exception as exc:
                print(f"[DEBUG] env step failed: {exc}", flush=True)
                log_step(step=step + 1, action=action, reward=0.0, done=False, error=str(exc))
                rewards.append(0.0)
                continue

            rewards.append(reward)
            error = exec_obs.error if not exec_obs.success else None
            result_text = f"Output: {exec_obs.output}" if not exec_obs.error else f"Error: {exec_obs.error}"
            log_step(step=step + 1, action=action, reward=reward, done=done, error=error)

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": [{"type": "text", "text": result_text}]})

        elif action_type == "submit_answer":
            try:
                submit_result = env_client.step(
                    DataAction(action_type="submit_answer", answer=action.get("answer", ""))
                )
                submit_obs = submit_result.observation
                score = float(submit_obs.metadata.get("score", 0.0) if submit_obs.metadata else submit_result.reward)
            except Exception as exc:
                print(f"[DEBUG] env step failed: {exc}", flush=True)
                log_step(step=step + 1, action=action, reward=0.0, done=True, error=str(exc))
                log_end(success=False, steps=step + 1, score=0.0, rewards=rewards)
                return 0.0

            rewards.append(score)
            log_step(step=step + 1, action=action, reward=score, done=True, error=None)
            log_end(success=score > 0.0, steps=step + 1, score=score, rewards=rewards)
            return score

        else:
            log_step(
                step=step + 1,
                action=action_type or "unknown",
                reward=0.0,
                done=False,
                error=f"unknown action '{action_type}'",
            )
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

    log_end(success=False, steps=MAX_STEPS, score=0.0, rewards=rewards)
    return 0.0


def main():
    print("Executing Data Analysis Environment")
    openai_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    scores = {}
    difficulties = {
        1: "Easy_TopRevenueCategoryTask",
        2: "Medium_CityRevenueShareTask",
        3: "Medium_RepeatCustomerCohortTask",
        4: "Hard_MonthlyRevenueRatioTask",
        5: "Hard_CustomerLoyaltyRevenueTask",
        6: "Hard_SupplierProfitabilityTask",
    }

    with DataAnalysisClient(base_url=ENV_SERVER_URL).sync() as env_client:
        for task_id in difficulties.keys():
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
