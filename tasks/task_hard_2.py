import re
import sqlite3
from pathlib import Path

import pandas as pd

from tasks.base_task import BaseTask

DB_PATH = Path(__file__).resolve().parent.parent / "datasets" / "store_data.db"


class CustomerLoyaltyRevenueTask(BaseTask):
    """Hard task: find the highest-revenue customer loyalty tier using cross-source data.

    The agent must query the customer_profiles table from the SQLite database,
    join it with the sales DataFrame on customer_id, and compute revenue by tier.
    The database is accessible via sqlite3.connect(db_path) in the sandbox.
    """

    @property
    def task_id(self) -> int:
        return 5

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def description(self) -> str:
        return (
            "Using the customer profiles database (connect with sqlite3.connect(db_path)), "
            "which customer loyalty tier generates the highest total revenue? "
            "What percentage of total revenue does it represent? "
            "Round percentage to 2 decimal places. "
            "Submit your answer in the format: "
            "'Top tier: <name>, Revenue: $X.XX, Percentage: X.XX%'"
        )

    def _compute(self) -> tuple:
        """Compute the top loyalty tier and its revenue share.

        Returns:
            A tuple of (top_tier, tier_revenue, percentage).
        """
        conn = sqlite3.connect(DB_PATH)
        profiles = pd.read_sql("SELECT customer_id, loyalty_tier FROM customer_profiles", conn)
        conn.close()
        merged = self.df.merge(profiles, on="customer_id", how="left")
        tier_rev = merged.groupby("loyalty_tier")["total_price"].sum()
        total = merged["total_price"].sum()
        top = tier_rev.idxmax()
        rev = tier_rev[top]
        pct = rev / total * 100
        return top, rev, pct

    def expected_answer(self) -> str:
        """Compute the expected formatted answer.

        Returns:
            Formatted string like 'Top tier: Bronze, Revenue: $97210.91, Percentage: 39.28%'.
        """
        top, rev, pct = self._compute()
        return f"Top tier: {top}, Revenue: ${rev:.2f}, Percentage: {round(pct, 2)}%"

    def grade(self, answer: str) -> float:
        """Grade with partial credit for each of the three fields.

        Scoring:
            - 0.33 for correct tier name (case-insensitive)
            - 0.33 for revenue within ±0.5% of expected
            - 0.34 for percentage within ±0.1 of expected

        Args:
            answer: The agent's submitted answer string.

        Returns:
            A score between 0.0 and 1.0.
        """
        top, expected_rev, expected_pct = self._compute()
        score = 0.0

        tier_match = re.search(r"Top tier:\s*([^,]+)", answer, re.IGNORECASE)
        if tier_match and tier_match.group(1).strip().lower() == top.lower():
            score += 0.33

        rev_match = re.search(r"Revenue:\s*\$?([\d.]+)", answer, re.IGNORECASE)
        if rev_match:
            try:
                submitted = float(rev_match.group(1))
                if abs(submitted - expected_rev) <= expected_rev * 0.005:
                    score += 0.33
            except ValueError:
                pass

        pct_match = re.search(r"Percentage:\s*([\d.]+)%?", answer, re.IGNORECASE)
        if pct_match:
            try:
                if abs(float(pct_match.group(1)) - expected_pct) <= 0.1:
                    score += 0.34
            except ValueError:
                pass

        return max(0.05, min(0.95, score))
