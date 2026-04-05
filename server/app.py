from models import DataAction, DataObservation
from openenv.core.env_server import create_app
from server.data_analysis_env import DataAnalysisEnv

app = create_app(DataAnalysisEnv, DataAction, DataObservation, env_name="data_analysis_env", max_concurrent_envs=3)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
