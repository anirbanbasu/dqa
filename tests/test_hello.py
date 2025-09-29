# A practically useless test to test the useless hello world example.

import pytest
from py_a2a_dapr.hello import HELLO_MESSAGE, main


class TestHello:
    def test_main(self) -> None:
        assert main() == 0

    def test_run_main(self, capsys: "pytest.CaptureFixture[str]") -> None:
        main()
        captured = capsys.readouterr()
        assert HELLO_MESSAGE in captured.out.strip()
