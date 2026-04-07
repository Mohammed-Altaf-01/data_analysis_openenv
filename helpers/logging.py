from typing import List, Optional


def log_start(task: str, env: str, model: str) -> None:
    """Emit the [START] line at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit one [STEP] line immediately after env.step() returns.

    Args:
        step: 1-based step number.
        action: Compact single-line action label (e.g. 'execute_code').
        reward: Step reward, formatted to 2 decimal places.
        done: Whether the episode ended after this step.
        error: Raw error string from the env, or None.
    """
    error_val = error.replace("\n", " ") if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """Emit the [END] line after env.close(), always emitted even on exception.

    Args:
        success: Whether the episode was successful.
        steps: Total number of steps taken.
        rewards: List of per-step rewards, each formatted to 2 decimal places.
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)
