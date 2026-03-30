"""Data Analysis Agent environment implementation.

Provides an RL environment where an agent executes pandas code against
a business dataset to answer analytical questions. Each episode presents
a task with a programmatic grader that scores performance 0.0-1.0.
"""

import io
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from models import DataAction, DataObservation, DataState
from openenv.core.env_server import Environment
from tasks import TASKS

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "sales.csv"


class DataAnalysisEnv(Environment):
    """Environment for training data analysis agents on business datasets.

    The agent receives a task question and can execute pandas code against
    a pre-loaded DataFrame. The episode ends when the agent submits an answer
    or exceeds the maximum number of steps.

    Attributes:
        MAX_STEPS: Maximum steps before forced episode termination.
    """

    MAX_STEPS = 20

    def __init__(self):
        """Initialize the environment with default state."""
        super().__init__()
        self._source_df = pd.read_csv(DATASET_PATH)
        self._df = self._source_df.copy()
        self._state = DataState()
        self._task = None
        self._exec_namespace = {}

    def _build_namespace(self) -> dict:
        """Build a restricted execution namespace for agent code.

        The namespace includes only pandas, numpy, and the dataset copy.
        Dangerous builtins like open, exec, eval, and __import__ are removed.

        Returns:
            A dictionary to use as the globals for exec().
        """
        safe_builtins = (
            {
                k: v
                for k, v in __builtins__.items()
                if k not in ("open", "exec", "eval", "__import__", "compile", "exit", "quit")
            }
            if isinstance(__builtins__, dict)
            else {
                k: getattr(__builtins__, k)
                for k in dir(__builtins__)
                if k not in ("open", "exec", "eval", "__import__", "compile", "exit", "quit") and not k.startswith("_")
            }
        )
        return {
            "__builtins__": safe_builtins,
            "df": self._df.copy(),
            "pd": pd,
            "np": np,
        }

    def _dataset_info(self) -> str:
        """Generate a summary of the dataset schema for the agent.

        Returns:
            A string describing column names, dtypes, row count, and a sample.
        """
        buf = io.StringIO()
        self._df.info(buf=buf)
        info_str = buf.getvalue()
        sample = self._df.head(3).to_string()
        return f"Dataset shape: {self._df.shape}\n\n{info_str}\nSample rows:\n{sample}"

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DataObservation:
        """Reset the environment for a new episode.

        Args:
            seed: Optional random seed (unused, kept for interface compliance).
            episode_id: Optional episode identifier; generated if not provided.
            **kwargs: Additional keyword arguments. Supports 'task_id' (int, 1-3).

        Returns:
            An initial observation with the task description and dataset info.
        """
        task_id = kwargs.get("task_id", 1)
        eid = episode_id or str(uuid.uuid4())

        self._df = self._source_df.copy()
        self._state = DataState(episode_id=eid, step_count=0, task_id=task_id)
        self._exec_namespace = self._build_namespace()

        task_cls = TASKS.get(task_id)
        if task_cls is None:
            return DataObservation(
                done=True,
                reward=0.0,
                success=False,
                error=f"Invalid task_id: {task_id}. Must be 1, 2, or 3.",
            )
        self._task = task_cls(self._df)

        return DataObservation(
            done=False,
            reward=0.0,
            output="Environment ready. Use 'execute_code' actions to explore the dataset, then 'submit_answer' with your result.",
            task_description=self._task.description,
            dataset_info=self._dataset_info(),
            metadata={"task_id": task_id, "difficulty": self._task.difficulty},
        )

    def step(
        self,
        action: DataAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DataObservation:
        """Execute one step in the environment.

        Handles two action types:
        - execute_code: runs pandas code in a sandboxed namespace
        - submit_answer: grades the agent's final answer and ends the episode

        Args:
            action: The agent's action (execute_code or submit_answer).
            timeout_s: Optional timeout in seconds (unused).
            **kwargs: Additional keyword arguments.

        Returns:
            An observation with execution output, reward, and done flag.
        """
        self._state.step_count += 1

        if self._state.answer_submitted:
            return DataObservation(
                done=True,
                reward=0.0,
                output="Episode is already finished. Call reset() to start a new one.",
                success=False,
            )

        # Check max steps
        if self._state.step_count >= self.MAX_STEPS and action.action_type != "submit_answer":
            self._state.answer_submitted = True
            return DataObservation(
                done=True,
                reward=0.0,
                output=f"Maximum steps ({self.MAX_STEPS}) exceeded without submitting an answer.",
                success=False,
                metadata={"reason": "max_steps_exceeded"},
            )

        if action.action_type == "execute_code":
            return self._handle_execute_code(action)
        elif action.action_type == "submit_answer":
            return self._handle_submit_answer(action)
        else:
            return DataObservation(
                done=False,
                reward=-0.05,
                success=False,
                error=f"Unknown action_type: {action.action_type}",
            )

    def _handle_execute_code(self, action: DataAction) -> DataObservation:
        """Execute pandas code in the sandboxed namespace.

        Args:
            action: The action containing the code to execute.

        Returns:
            An observation with stdout output or error message.
        """
        if not action.code:
            return DataObservation(
                done=False,
                reward=-0.05,
                success=False,
                error="No code provided for execute_code action.",
            )

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = stdout_capture
            exec(action.code, self._exec_namespace)
            sys.stdout = old_stdout
            output = stdout_capture.getvalue()

            # If code produced no print output, try to get the last expression value
            if not output.strip():
                try:
                    result = eval(action.code.strip().split("\n")[-1], self._exec_namespace)
                    if result is not None:
                        output = str(result)
                except Exception:
                    output = "(Code executed successfully with no output)"

            return DataObservation(
                done=False,
                reward=0.05,
                output=output[:5000],
                success=True,
                metadata={"steps_remaining": self.MAX_STEPS - self._state.step_count},
            )
        except Exception as e:
            sys.stdout = old_stdout
            return DataObservation(
                done=False,
                reward=-0.05,
                success=False,
                error=f"{type(e).__name__}: {e}",
                output="",
                metadata={"steps_remaining": self.MAX_STEPS - self._state.step_count},
            )

    def _handle_submit_answer(self, action: DataAction) -> DataObservation:
        """Grade the agent's submitted answer and end the episode.

        Args:
            action: The action containing the answer to grade.

        Returns:
            An observation with the final score and done=True.
        """
        if not action.answer:
            return DataObservation(
                done=False,
                reward=-0.05,
                success=False,
                error="No answer provided for submit_answer action.",
            )

        self._state.answer_submitted = True
        score = self._task.grade(action.answer)
        self._state.final_score = score

        return DataObservation(
            done=True,
            reward=score,
            output=f"Answer submitted. Score: {score:.2f}/1.00",
            success=True,
            metadata={
                "score": score,
                "expected_answer": self._task.expected_answer(),
                "submitted_answer": action.answer,
            },
        )

    @property
    def state(self) -> DataState:
        """Return the current episode state.

        Returns:
            The current DataState with episode_id, step_count, task_id, etc.
        """
        return self._state
