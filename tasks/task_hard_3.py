import re
import sqlite3
from pathlib import Path

import pandas as pd

from tasks.base_task import BaseTask

DB_PATH = Path(__file__).resolve().parent.parent / "datasets" / "store_data.db"


class SupplierProfitabilityTask(BaseTask):
    """Hard task: find the most profitable supplier using cross-source data.

    The agent must query the product_catalog table from the SQLite database,
    join it with the sales DataFrame on product_name, compute per-order profit
    and margin, then aggregate by supplier.
    The database is accessible via sqlite3.connect(db_path) in the sandbox.
    """

    @property
    def task_id(self) -> int:
        return 6

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def description(self) -> str:
        return (
            "Using the product catalog database (connect with sqlite3.connect(db_path)), "
            "which supplier has the highest total profit from orders? "
            "(profit per order = (unit_price - cost_price) * quantity) "
            "What is their total profit and average profit margin? "
            "(margin % = (unit_price - cost_price) / unit_price * 100, "
            "averaged across all their orders) "
            "Round total profit to 2 decimal places and avg margin to 2 decimal places. "
            "Submit your answer in the format: "
            "'Supplier: <name>, Total profit: $X.XX, Avg margin: X.XX%'"
        )

    def _compute(self) -> tuple:
        """Compute the top supplier by profit and their average margin.

        Returns:
            A tuple of (supplier_name, total_profit, avg_margin_pct).
        """
        conn = sqlite3.connect(DB_PATH)
        catalog = pd.read_sql("SELECT product_name, supplier, cost_price FROM product_catalog", conn)
        conn.close()
        merged = self.df.merge(catalog, on="product_name", how="left")
        merged["profit"] = (merged["unit_price"] - merged["cost_price"]) * merged["quantity"]
        merged["margin"] = (merged["unit_price"] - merged["cost_price"]) / merged["unit_price"] * 100
        sup_profit = merged.groupby("supplier")["profit"].sum()
        sup_margin = merged.groupby("supplier")["margin"].mean()
        top = sup_profit.idxmax()
        return top, sup_profit[top], sup_margin[top]

    def expected_answer(self) -> str:
        """Compute the expected formatted answer.

        Returns:
            Formatted string like 'Supplier: FashionWorld, Total profit: $38292.08, Avg margin: 52.08%'.
        """
        top, profit, margin = self._compute()
        return f"Supplier: {top}, Total profit: ${profit:.2f}, Avg margin: {round(margin, 2)}%"

    def grade(self, answer: str) -> float:
        """Grade with partial credit for each of the three fields.

        Scoring:
            - 0.33 for correct supplier name (case-insensitive)
            - 0.34 for total profit within ±0.5% of expected
            - 0.33 for avg margin within ±0.1 of expected

        Args:
            answer: The agent's submitted answer string.

        Returns:
            A score between 0.0 and 1.0.
        """
        top, expected_profit, expected_margin = self._compute()
        score = 0.0

        sup_match = re.search(r"Supplier:\s*([^,]+)", answer, re.IGNORECASE)
        if sup_match and sup_match.group(1).strip().lower() == top.lower():
            score += 0.33

        profit_match = re.search(r"Total profit:\s*\$?([\d.]+)", answer, re.IGNORECASE)
        if profit_match:
            try:
                submitted = float(profit_match.group(1))
                if abs(submitted - expected_profit) <= expected_profit * 0.005:
                    score += 0.34
            except ValueError:
                pass

        margin_match = re.search(r"Avg margin:\s*([\d.]+)%?", answer, re.IGNORECASE)
        if margin_match:
            try:
                if abs(float(margin_match.group(1)) - expected_margin) <= 0.1:
                    score += 0.33
            except ValueError:
                pass

        return score
