"""Client for the Data Analysis Agent environment.

Provides a typed async/sync client for interacting with the
data analysis environment server over HTTP/WebSocket.
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import DataAction, DataObservation, DataState


class DataAnalysisClient(EnvClient[DataAction, DataObservation, DataState]):
    """Client for interacting with the Data Analysis environment server.

    Supports both async and sync usage patterns:
        - Async: ``async with DataAnalysisClient(base_url=...) as client:``
        - Sync: ``with DataAnalysisClient(base_url=...).sync() as client:``
    """

    def _step_payload(self, action: DataAction) -> dict:
        """Convert a DataAction into a JSON-serializable payload.

        Args:
            action: The action to send to the server.

        Returns:
            A dictionary representation of the action.
        """
        payload = {"action_type": action.action_type}
        if action.code is not None:
            payload["code"] = action.code
        if action.answer is not None:
            payload["answer"] = action.answer
        return payload

    def _parse_result(self, payload: dict) -> StepResult[DataObservation]:
        """Parse the server's JSON response into a StepResult.

        Args:
            payload: The raw JSON response from the server.

        Returns:
            A StepResult containing the parsed observation, reward, and done flag.
        """
        obs_data = payload.get("observation", payload)
        obs = DataObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: dict) -> DataState:
        """Parse the server's state response into a DataState.

        Args:
            payload: The raw JSON state response from the server.

        Returns:
            A DataState object reflecting the current episode state.
        """
        return DataState(**payload)
