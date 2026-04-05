import re

import pandas as pd

from tasks.base_task import BaseTask


class MonthlyRevenueRatioTask(BaseTask):
    """Medium task: find the best and worst months by revenue and compute their ratio.

    The agent must parse order_date, group by month, find the extremes,
    and compute how many times larger the best month is versus the worst.
    """

    @property
    def task_id(self) -> int:
        return 4

    @property
    def difficulty(self) -> str:
        return "medium"

    @property
    def description(self) -> str:
        return (
            "What is the best and worst performing month by total revenue in 2024? "
            "What is the ratio of best to worst month revenue? Round ratio to 2 decimal places. "
            "Submit your answer in the format: "
            "'Best: YYYY-MM ($X.XX), Worst: YYYY-MM ($X.XX), Ratio: X.XX'"
        )

    def _compute(self) -> tuple:
        """Compute the best month, worst month, and their revenue ratio.

        Returns:
            A tuple of (best_month_str, best_rev, worst_month_str, worst_rev, ratio).
        """
        df = self.df.copy()
        df["order_date"] = pd.to_datetime(df["order_date"])
        monthly = df.groupby(df["order_date"].dt.to_period("M"))["total_price"].sum()
        best = monthly.idxmax()
        worst = monthly.idxmin()
        ratio = round(monthly[best] / monthly[worst], 2)
        return str(best), monthly[best], str(worst), monthly[worst], ratio

    def expected_answer(self) -> str:
        """Compute the expected formatted answer.

        Returns:
            Formatted string like 'Best: 2024-12 ($23025.82), Worst: 2024-05 ($16871.48), Ratio: 1.36'.
        """
        best, best_rev, worst, worst_rev, ratio = self._compute()
        return f"Best: {best} (${best_rev:.2f}), Worst: {worst} (${worst_rev:.2f}), Ratio: {ratio}"

    def grade(self, answer: str) -> float:
        """Grade with partial credit for each of the three fields.

        Scoring:
            - 0.33 for correct best month (exact YYYY-MM match)
            - 0.33 for correct worst month (exact YYYY-MM match)
            - 0.34 for ratio within ±0.01 of expected

        Args:
            answer: The agent's submitted answer string.

        Returns:
            A score between 0.0 and 1.0.
        """
        best, _, worst, _, expected_ratio = self._compute()
        score = 0.0

        best_match = re.search(r"Best:\s*([\d]{4}-[\d]{2})", answer, re.IGNORECASE)
        if best_match and best_match.group(1).strip() == best:
            score += 0.33

        worst_match = re.search(r"Worst:\s*([\d]{4}-[\d]{2})", answer, re.IGNORECASE)
        if worst_match and worst_match.group(1).strip() == worst:
            score += 0.33

        ratio_match = re.search(r"Ratio:\s*([\d.]+)", answer, re.IGNORECASE)
        if ratio_match:
            try:
                if abs(float(ratio_match.group(1)) - expected_ratio) <= 0.01:
                    score += 0.34
            except ValueError:
                pass

        return score
