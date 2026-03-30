"""FastAPI application for the Data Analysis Agent environment.

Creates the OpenEnv-compliant HTTP/WebSocket server that wraps
the DataAnalysisEnv environment.
"""

from openenv.core.env_server import create_app

from models import DataAction, DataObservation
from server.data_analysis_env import DataAnalysisEnv

app = create_app(DataAnalysisEnv, DataAction, DataObservation, env_name="data_analysis_env")


def main():
    """Run the environment server with uvicorn."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
