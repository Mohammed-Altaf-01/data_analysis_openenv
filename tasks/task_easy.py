"""Task 1 (Easy): Identify the top-selling product category by total revenue.

Requires a single groupby + sum + idxmax operation.
"""

import pandas as pd

from tasks.base_task import BaseTask


class TopRevenueCategoryTask(BaseTask):
    """Easy task: find the product category with the highest total revenue.

    The agent must group the dataset by category, sum the total_price column,
    and identify which category has the highest revenue.
    """

    @property
    def task_id(self) -> int:
        """Return the task identifier."""
        return 1

    @property
    def difficulty(self) -> str:
        """Return the difficulty level."""
        return "easy"

    @property
    def description(self) -> str:
        """Return the task question."""
        return (
            "What is the top-selling product category by total revenue? "
            "Submit just the category name as your answer."
        )

    def expected_answer(self) -> str:
        """Compute the top revenue category from the dataset.

        Returns:
            The name of the category with the highest total_price sum.
        """
        return self.df.groupby("category")["total_price"].sum().idxmax()

    def grade(self, answer: str) -> float:
        """Grade the answer by case-insensitive string match.

        Args:
            answer: The agent's submitted category name.

        Returns:
            1.0 if the answer matches the expected category, 0.0 otherwise.
        """
        expected = self.expected_answer().strip().lower()
        submitted = answer.strip().lower()
        return 1.0 if submitted == expected else 0.0
