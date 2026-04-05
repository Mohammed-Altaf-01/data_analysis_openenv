import re

import pandas as pd

from tasks.base_task import BaseTask


class RepeatCustomerCohortTask(BaseTask):
    """Hard task: find customers who ordered in both January and December.

    The agent must identify customers present in both months, count them,
    and compare their average order value to all other customers.
    """

    @property
    def task_id(self) -> int:
        return 3

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def description(self) -> str:
        return (
            "How many unique customers placed orders in BOTH January and December? "
            "What is their average order value compared to all other customers? "
            "Submit your answer in the format: "
            "'Cohort: N customers, Cohort AOV: $X.XX, Other AOV: $X.XX'"
        )

    def _compute_cohort(self) -> tuple[set, float, float]:
        """Compute the cohort of customers ordering in both January and December.

        Returns:
            A tuple of (cohort_customer_ids, cohort_aov, other_aov).
        """
        df = self.df.copy()
        df["order_date"] = pd.to_datetime(df["order_date"])
        jan_customers = set(df[df["order_date"].dt.month == 1]["customer_id"])
        dec_customers = set(df[df["order_date"].dt.month == 12]["customer_id"])
        cohort = jan_customers & dec_customers

        cohort_aov = df[df["customer_id"].isin(cohort)]["total_price"].mean()
        other_aov = df[~df["customer_id"].isin(cohort)]["total_price"].mean()
        return cohort, round(cohort_aov, 2), round(other_aov, 2)

    def expected_answer(self) -> str:
        """Compute the expected cohort analysis answer.

        Returns:
            Formatted string like 'Cohort: 57 customers, Cohort AOV: $126.57, Other AOV: $122.94'.
        """
        cohort, cohort_aov, other_aov = self._compute_cohort()
        return f"Cohort: {len(cohort)} customers, Cohort AOV: ${cohort_aov}, Other AOV: ${other_aov}"

    def grade(self, answer: str) -> float:
        """Grade the answer with partial credit for each of the three fields.

        Scoring:
            - 0.33 for correct customer count (exact match)
            - 0.33 for cohort AOV within ±0.5% of expected
            - 0.34 for other AOV within ±0.5% of expected

        Args:
            answer: The agent's submitted answer string.

        Returns:
            A score between 0.0 and 1.0.
        """
        cohort, expected_cohort_aov, expected_other_aov = self._compute_cohort()
        expected_count = len(cohort)
        score = 0.0

        # Check customer count
        count_match = re.search(r"Cohort:\s*(\d+)\s*customers?", answer, re.IGNORECASE)
        if count_match:
            if int(count_match.group(1)) == expected_count:
                score += 0.33

        # Check cohort AOV
        cohort_aov_match = re.search(r"Cohort\s+AOV:\s*\$?([\d.]+)", answer, re.IGNORECASE)
        if cohort_aov_match:
            try:
                submitted = float(cohort_aov_match.group(1))
                tolerance = expected_cohort_aov * 0.005
                if abs(submitted - expected_cohort_aov) <= tolerance:
                    score += 0.33
            except ValueError:
                pass

        # Check other AOV
        other_aov_match = re.search(r"Other\s+AOV:\s*\$?([\d.]+)", answer, re.IGNORECASE)
        if other_aov_match:
            try:
                submitted = float(other_aov_match.group(1))
                tolerance = expected_other_aov * 0.005
                if abs(submitted - expected_other_aov) <= tolerance:
                    score += 0.34
            except ValueError:
                pass

        return score
