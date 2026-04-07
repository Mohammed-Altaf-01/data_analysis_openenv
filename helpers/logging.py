from typing import List, Optional


def safe_score(raw: float) -> float:
    """Clamp a raw score to the strictly-open interval (0.05, 0.95).

    Args:
        raw: Unclamped score value.

    Returns:
        Score guaranteed to be in [0.05, 0.95].
    """
    return max(0.05, min(0.95, float(raw)))


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


def log_end(task_id: int, score: float, steps: int) -> None:
    """Emit the [END] line after the episode completes.

    Args:
        task_id: The task number that just ran.
        score: Final clamped score in [0.05, 0.95].
        steps: Total number of steps taken.
    """
    print(f"[END] task={task_id} score={score:.2f} steps={steps}", flush=True)
