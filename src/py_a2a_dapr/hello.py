# A practically useless hello world example to get started.

from datetime import datetime

HELLO_MESSAGE = "Hello from py-a2a-dapr!"


def main() -> int:
    print(f"{HELLO_MESSAGE} @ {datetime.now()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
