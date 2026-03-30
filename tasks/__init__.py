"""Task definitions for the Data Analysis Agent environment."""

from tasks.base_task import BaseTask
from tasks.task_easy import TopRevenueCategoryTask
from tasks.task_hard import RepeatCustomerCohortTask
from tasks.task_medium import CityRevenueShareTask

TASKS = {
    1: TopRevenueCategoryTask,
    2: CityRevenueShareTask,
    3: RepeatCustomerCohortTask,
}

__all__ = ["BaseTask", "TASKS", "TopRevenueCategoryTask", "CityRevenueShareTask", "RepeatCustomerCohortTask"]
