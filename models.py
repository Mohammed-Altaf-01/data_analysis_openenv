"""Pydantic models for the Data Analysis Agent environment.

Defines the action, observation, and state types used for communication
between the RL agent and the environment server.
"""

from typing import Literal, Optional

from openenv.core.env_server import Action, Observation, State


class DataAction(Action):
    """Agent action for the data analysis environment.

    The agent can either execute pandas code against the loaded dataset
    or submit a final answer to be graded.

    Attributes:
        action_type: Whether to execute code or submit an answer.
        code: Python/pandas code to execute (required when action_type is "execute_code").
        answer: Final answer string (required when action_type is "submit_answer").
    """

    action_type: Literal["execute_code", "submit_answer"]
    code: Optional[str] = None
    answer: Optional[str] = None


class DataObservation(Observation):
    """Observation returned after each step or reset.

    Attributes:
        output: String output from code execution or environment messages.
        success: Whether the last action executed without errors.
        error: Error message if the last action failed.
        task_description: The task question, populated on reset.
        dataset_info: Column names and dtypes summary, populated on reset.
    """

    output: str = ""
    success: bool = True
    error: Optional[str] = None
    task_description: str = ""
    dataset_info: str = ""


class DataState(State):
    """Episode state for the data analysis environment.

    Attributes:
        task_id: The current task being evaluated (1, 2, or 3).
        answer_submitted: Whether the agent has submitted a final answer.
        final_score: The graded score after answer submission (0.0 to 1.0).
    """

    task_id: int = 1
    answer_submitted: bool = False
    final_score: float = 0.0
