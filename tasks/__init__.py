"""Task definitions for the Data Analysis Agent environment."""

from tasks.base_task import BaseTask
from tasks.task_easy import TopRevenueCategoryTask
from tasks.task_hard import RepeatCustomerCohortTask
from tasks.task_hard_2 import CustomerLoyaltyRevenueTask
from tasks.task_hard_3 import SupplierProfitabilityTask
from tasks.task_medium import CityRevenueShareTask
from tasks.task_medium_2 import MonthlyRevenueRatioTask

TASKS = {
    1: TopRevenueCategoryTask,
    2: CityRevenueShareTask,
    3: RepeatCustomerCohortTask,
    4: MonthlyRevenueRatioTask,
    5: CustomerLoyaltyRevenueTask,
    6: SupplierProfitabilityTask,
}

__all__ = [
    "BaseTask",
    "TASKS",
    "TopRevenueCategoryTask",
    "CityRevenueShareTask",
    "RepeatCustomerCohortTask",
    "MonthlyRevenueRatioTask",
    "CustomerLoyaltyRevenueTask",
    "SupplierProfitabilityTask",
]
