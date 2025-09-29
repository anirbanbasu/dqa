# A practically useless test to test the useless hello world example.

import time
from typing import List
from uuid import uuid4
from pydantic import TypeAdapter
import pytest
from typer.testing import CliRunner
from py_a2a_dapr.client.a2a import cli_app as app
from py_a2a_dapr.model.echo_task import EchoResponse, EchoResponseWithHistory
import subprocess


class TestCLI:
    # TODO: Should these be fixtures instead?
    thread_id = str(uuid4())
    echo_iteratons = 5  # Number of echo iterations to create a history

    @pytest.fixture(scope="class", autouse=True)
    def manage_dapr_sidecars(self):
        """
        Fixture to run Dapr sidecars before tests and clean up afterwards.
        """
        # The start script MUST fork out and return immediately.
        dapr_start_script = "./start_dapr_multi.sh"
        dapr_stop_script = "./stop_dapr_multi.sh"

        try:
            subprocess.run([dapr_start_script], shell=True, check=True, text=True)
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Start script failed with error. {e}")
        except FileNotFoundError:
            pytest.fail(f"Start script not found at: {dapr_start_script}")

        # Wait for Dapr sidecars to be up and running
        time.sleep(8)
        # --- Yield control to tests ---
        yield

        # --- Teardown: Optional cleanup after tests ---
        # Stop Dapr sidecars
        try:
            subprocess.run([dapr_stop_script], shell=True, text=True)
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Stop script failed with error {e}")
        except FileNotFoundError:
            pytest.fail(f"Stop script not found at: {dapr_stop_script}")

    def test_hello(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["hello", "Tester"])
        assert result.exit_code == 0
        assert "Hello, Tester!" in result.stdout

    @pytest.mark.parametrize("iterations", range(echo_iteratons))
    def test_echo_a2a_echo(self, manage_dapr_sidecars, iterations: int) -> None:
        # Iterations are there to create a history in the response.
        runner = CliRunner()
        message = f"Hello there! {iterations}"
        result = runner.invoke(
            app, ["echo-a2a-echo", "--thread-id", self.thread_id, message]
        )
        assert result.exit_code == 0
        validated_response = EchoResponseWithHistory.model_validate_json(result.stdout)
        assert validated_response.current.user_input == message
        assert message in validated_response.current.output
        if iterations > 1:
            assert len(validated_response.past) >= (iterations - 1)

    def test_echo_a2a_history(self, manage_dapr_sidecars) -> None:
        # Iterations are there to create a history in the response.
        runner = CliRunner()
        result = runner.invoke(app, ["echo-a2a-history", "--thread-id", self.thread_id])
        assert result.exit_code == 0
        response_adapter = TypeAdapter(List[EchoResponse])
        validated_response = response_adapter.validate_json(result.stdout)
        # The history could be empty if this test runs independently without any preceding echo tests.
        assert (
            len(validated_response) == 0
            or len(validated_response) == self.echo_iteratons
        )
        if len(validated_response) > 0:
            assert len(validated_response) == self.echo_iteratons
        for resp in validated_response:
            assert isinstance(resp, EchoResponse)

    def test_echo_a2a_delete_history(self, manage_dapr_sidecars) -> None:
        # Iterations are there to create a history in the response.
        runner = CliRunner()
        result = runner.invoke(
            app, ["echo-a2a-delete-history", "--thread-id", self.thread_id]
        )
        assert result.exit_code == 0
        assert "deleted successfully" in result.stdout
        assert self.thread_id in result.stdout
