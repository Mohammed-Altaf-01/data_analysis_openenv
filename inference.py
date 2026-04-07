from typing import Any, List

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from client import DataAnalysisClient
from helpers.constants import *
from helpers.logging import log_end, log_start, log_step
from helpers.prompts import SYSTEM_PROMPT
from helpers.response_parser import FALLBACK_ACTION, parse_model_action
from models import DataAction


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
                log_step(step=step + 1, action=action_type, reward=0.0, done=False, error=str(exc))
                rewards.append(0.0)
                continue

            rewards.append(reward)
            error = exec_obs.error if not exec_obs.success else None
            result_text = f"Output: {exec_obs.output}" if not exec_obs.error else f"Error: {exec_obs.error}"
            log_step(step=step + 1, action=action_type, reward=reward, done=done, error=error)

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
                log_step(step=step + 1, action=action_type, reward=0.0, done=True, error=str(exc))
                log_end(success=False, steps=step + 1, rewards=rewards)
                return 0.0

            score = max(0.01, min(0.99, score))
            rewards.append(score)
            log_step(step=step + 1, action=action_type, reward=score, done=True, error=None)
            log_end(success=score > 0.01, steps=step + 1, rewards=rewards)
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

    log_end(success=False, steps=MAX_STEPS, rewards=rewards)
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
