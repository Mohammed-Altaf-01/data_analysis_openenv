"""Abstract base class for data analysis tasks.

Each task defines a question, computes the expected answer from the dataset,
and provides a grader that scores agent responses from 0.0 to 1.0.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseTask(ABC):
    """Base class for all data analysis tasks.

    Subclasses must implement the question, compute the expected answer
    from the dataset, and provide a grading function.

    Attributes:
        df: The pandas DataFrame containing the dataset.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize the task with a dataset.

        Args:
            df: The pandas DataFrame to analyze.
        """
        self.df = df

    @property
    @abstractmethod
    def task_id(self) -> int:
        """Return the unique task identifier."""

    @property
    @abstractmethod
    def difficulty(self) -> str:
        """Return the difficulty level: 'easy', 'medium', or 'hard'."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the task question shown to the agent."""

    @abstractmethod
    def expected_answer(self) -> str:
        """Compute and return the ground-truth answer from the dataset.

        Returns:
            The expected answer as a formatted string.
        """

    @abstractmethod
    def grade(self, answer: str) -> float:
        """Grade the agent's submitted answer.

        Args:
            answer: The agent's submitted answer string.

        Returns:
            A score between 0.0 and 1.0.
        """
