"""Task 2 (Medium): Find the top revenue city and its share of total revenue.

Requires groupby + aggregation + percentage calculation + formatting.
"""

import re

import pandas as pd

from tasks.base_task import BaseTask


class CityRevenueShareTask(BaseTask):
    """Medium task: identify the city with the highest revenue and its percentage share.

    The agent must group by city, compute total revenue per city,
    find the top city, and calculate what percentage of overall revenue it represents.
    """

    @property
    def task_id(self) -> int:
        """Return the task identifier."""
        return 2

    @property
    def difficulty(self) -> str:
        """Return the difficulty level."""
        return "medium"

    @property
    def description(self) -> str:
        """Return the task question."""
        return (
            "Which city generates the most revenue? What percentage of total revenue "
            "does it represent? Round to 2 decimal places. "
            "Submit your answer in the format: 'City: <name>, Percentage: <X.XX>%'"
        )

    def expected_answer(self) -> str:
        """Compute the top city and its revenue share.

        Returns:
            Formatted string like 'City: London, Percentage: 10.81%'.
        """
        city_rev = self.df.groupby("city")["total_price"].sum()
        top_city = city_rev.idxmax()
        pct = round(city_rev[top_city] / city_rev.sum() * 100, 2)
        return f"City: {top_city}, Percentage: {pct}%"

    def grade(self, answer: str) -> float:
        """Grade the answer with partial credit for city and percentage.

        Scoring:
            - 0.5 for correct city name (case-insensitive)
            - 0.5 for percentage within ±0.1 of expected

        Args:
            answer: The agent's submitted answer string.

        Returns:
            A score between 0.0 and 1.0.
        """
        score = 0.0
        city_rev = self.df.groupby("city")["total_price"].sum()
        expected_city = city_rev.idxmax()
        expected_pct = round(city_rev[expected_city] / city_rev.sum() * 100, 2)

        # Check city
        city_match = re.search(r"City:\s*([^,]+)", answer, re.IGNORECASE)
        if city_match:
            submitted_city = city_match.group(1).strip()
            if submitted_city.lower() == expected_city.lower():
                score += 0.5

        # Check percentage
        pct_match = re.search(r"Percentage:\s*([\d.]+)%?", answer, re.IGNORECASE)
        if pct_match:
            try:
                submitted_pct = float(pct_match.group(1))
                if abs(submitted_pct - expected_pct) <= 0.1:
                    score += 0.5
            except ValueError:
                pass

        return score
