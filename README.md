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
| `task_id` | `int` | Active task (1–6) |
| `answer_submitted` | `bool` | Whether final answer was submitted |
| `final_score` | `float` | Graded score after submission |

## Tasks

Tasks use two data sources:
- **`df`** — synthetic e-commerce sales CSV (~2000 orders): `order_id`, `customer_id`, `product_name`, `category`, `quantity`, `unit_price`, `total_price`, `order_date`, `city`, `country`
- **SQLite DB** (`store_data.db`) — additional tables for cross-source tasks: `customer_profiles` (300 rows), `product_catalog` (25 rows)

### Task 1 — Easy: Top Revenue Category
- **Question**: What is the top-selling product category by total revenue?
- **Grading**: Containment match (case-insensitive) → 1.0 or 0.0
- **Expected difficulty**: Single groupby + sum + argmax

### Task 2 — Medium: City Revenue Share
- **Question**: Which city generates the most revenue? What percentage of total revenue does it represent?
- **Grading**: 0.5 for correct city + 0.5 for percentage within ±0.1%
- **Expected difficulty**: Groupby + percentage calculation + formatting

### Task 3 — Medium: Repeat Customer Cohort Analysis
- **Question**: How many unique customers ordered in both January and December? Compare their average order value to all other customers.
- **Grading**: 0.33 per correct field (count, cohort AOV, other AOV)
- **Expected difficulty**: Temporal filtering, set intersection, conditional aggregation

### Task 4 — Hard: Monthly Revenue Ratio
- **Question**: Which month had the highest vs. lowest total revenue? What is the ratio between them?
- **Grading**: 0.33 for best month + 0.33 for worst month + 0.34 for ratio within ±0.01
- **Expected difficulty**: Monthly resample/groupby, min/max comparison, ratio formatting

### Task 5 — Hard: Customer Loyalty Tier Revenue (cross-source)
- **Question**: Which customer loyalty tier generates the highest total revenue and what percentage does it represent?
- **Data**: Requires joining `df` with `customer_profiles` table from SQLite on `customer_id`
- **Grading**: 0.33 for tier name + 0.33 for revenue within ±0.5% + 0.34 for percentage within ±0.1
- **Expected difficulty**: SQLite query → pandas merge → groupby aggregation

### Task 6 — Hard: Supplier Profitability (cross-source)
- **Question**: Which supplier has the highest total profit? What is their average profit margin?
- **Data**: Requires joining `df` with `product_catalog` table from SQLite on `product_name`
- **Grading**: 0.33 for supplier name + 0.34 for total profit within ±0.5% + 0.33 for avg margin within ±0.1
- **Expected difficulty**: SQLite query → pandas merge → per-order profit/margin calculation → group aggregation

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

### Run the inference
- First export all the required env variables mentioned in the .env.example. Then run below command
```bash
uv run python inference.py
```

### Run the baseline
```bash
OPENAI_API_KEY=sk-... uv run python baseline.py
# Against a deployed HF Space:
OPENAI_API_KEY=sk-... uv run python baseline.py --base-url https://<your-username>-<space-name>.hf.space
```

### Docker (local)
```bash
docker build -t data-analysis-env .
docker run -p 7860:7860 data-analysis-env
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

## Project Structure

```
├── models.py                  # DataAction, DataObservation, DataState
├── client.py                  # DataAnalysisClient (EnvClient subclass)
├── inference.py               # HF inference script (uses HF Inference API)
├── baseline.py                # OpenAI baseline inference script
├── helpers/
│   └── response_parser.py     # Robust LLM JSON response parser
├── tasks/
│   ├── base_task.py           # Task ABC with grade() interface
│   ├── task_easy.py           # Task 1 (Easy): Top revenue category
│   ├── task_medium.py         # Task 2 (Medium): City revenue share
│   ├── task_medium_2.py       # Task 4 (Hard): Monthly revenue ratio
│   ├── task_hard.py           # Task 3 (Medium): Repeat customer cohort
│   ├── task_hard_2.py         # Task 5 (Hard): Customer loyalty tier revenue
│   └── task_hard_3.py         # Task 6 (Hard): Supplier profitability
├── datasets/
│   ├── sales.csv              # Synthetic e-commerce sales dataset
│   └── store_data.db          # SQLite DB: customer_profiles, product_catalog
├── server/
│   ├── app.py                 # FastAPI app entry point
│   └── data_analysis_env.py   # Environment implementation
├── Dockerfile                 # HF Spaces Docker build (port 7860)
├── openenv.yaml               # OpenEnv spec metadata
└── pyproject.toml             # Dependencies and project config
```
