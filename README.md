# Data Analysis Agent Environment

An OpenEnv-compliant RL environment for training and evaluating data analysis agents. Agents execute pandas code against a business dataset to answer analytical questions, graded by deterministic programmatic graders.

## Motivation

Data analysis is a universal real-world task. Every business needs analysts who can query datasets, compute metrics, and extract insights. This environment lets RL agents practice that exact workflow — explore a dataset with code, then submit a precise answer — with automatic scoring.

## Action & Observation Spaces

### Action (`DataAction`)

| Field | Type | Description |
|---|---|---|
| `action_type` | `"execute_code"` or `"submit_answer"` | What the agent wants to do |
| `code` | `str` (optional) | Python/pandas code to execute |
| `answer` | `str` (optional) | Final answer to submit for grading |

### Observation (`DataObservation`)

| Field | Type | Description |
|---|---|---|
| `output` | `str` | Stdout from code execution or environment messages |
| `success` | `bool` | Whether the action succeeded |
| `error` | `str` (optional) | Error message if action failed |
| `task_description` | `str` | The question to answer (set on reset) |
| `dataset_info` | `str` | Dataset schema summary (set on reset) |
| `done` | `bool` | Whether the episode is over |
| `reward` | `float` | Step reward |

### State (`DataState`)

| Field | Type | Description |
|---|---|---|
| `episode_id` | `str` | Unique episode identifier |
| `step_count` | `int` | Current step number |
| `task_id` | `int` | Active task (1, 2, or 3) |
| `answer_submitted` | `bool` | Whether final answer was submitted |
| `final_score` | `float` | Graded score after submission |

## Tasks

All tasks use a synthetic e-commerce dataset (~2000 orders) with columns: `order_id`, `customer_id`, `product_name`, `category`, `quantity`, `unit_price`, `total_price`, `order_date`, `city`, `country`.

### Task 1 — Easy: Top Revenue Category
- **Question**: What is the top-selling product category by total revenue?
- **Grading**: Exact match (case-insensitive) → 1.0 or 0.0
- **Expected difficulty**: Single groupby + sum + argmax

### Task 2 — Medium: City Revenue Share
- **Question**: Which city generates the most revenue? What percentage of total revenue does it represent?
- **Grading**: 0.5 for correct city + 0.5 for percentage within ±0.1%
- **Expected difficulty**: Groupby + percentage calculation + formatting

### Task 3 — Hard: Repeat Customer Cohort Analysis
- **Question**: How many unique customers ordered in both January and December? Compare their average order value to all other customers.
- **Grading**: 0.33 per correct field (count, cohort AOV, other AOV)
- **Expected difficulty**: Temporal filtering, set intersection, conditional aggregation

## Reward Function

| Event | Reward |
|---|---|
| Successful code execution | +0.05 |
| Code execution error | -0.05 |
| Final answer (graded) | 0.0 — 1.0 based on task grader |
| Max steps (20) exceeded | 0.0 |

## Setup & Usage

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Install
```bash
uv sync
```

### Run the server
```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run the baseline
```bash
OPENAI_API_KEY=sk-... uv run python baseline.py
```

### Docker
```bash
docker build -t data-analysis-env -f server/Dockerfile .
docker run -p 8000:8000 data-analysis-env
```

### Client usage (Python)
```python
from client import DataAnalysisClient
from models import DataAction

# Async
async with DataAnalysisClient(base_url="http://localhost:8000") as client:
    result = await client.reset(task_id=1)
    result = await client.step(DataAction(action_type="execute_code", code="print(df.head())"))
    result = await client.step(DataAction(action_type="submit_answer", answer="Electronics"))

# Sync
with DataAnalysisClient(base_url="http://localhost:8000").sync() as client:
    result = client.reset(task_id=2)
    result = client.step(DataAction(action_type="execute_code", code="print(df.groupby('city')['total_price'].sum())"))
```

## Baseline Scores

| Task | Difficulty | gpt-4o-mini Score |
|---|---|---|
| 1 | Easy | TBD |
| 2 | Medium | TBD |
| 3 | Hard | TBD |
| **Average** | | **TBD** |

*(Run the baseline script to populate these scores)*

## Project Structure

```
├── models.py               # DataAction, DataObservation, DataState
├── client.py               # DataAnalysisClient (EnvClient subclass)
├── baseline.py             # OpenAI baseline inference script
├── tasks/
│   ├── base_task.py        # Task ABC with grade() interface
│   ├── task_easy.py        # Task 1: Top revenue category
│   ├── task_medium.py      # Task 2: City revenue share
│   └── task_hard.py        # Task 3: Repeat customer cohort
├── datasets/
│   └── sales.csv           # Synthetic e-commerce dataset
├── server/
│   ├── app.py              # FastAPI app (create_app)
│   ├── data_analysis_env.py # Environment implementation
│   └── Dockerfile          # Container build
├── openenv.yaml            # OpenEnv spec metadata
└── pyproject.toml          # Dependencies and project config
```
