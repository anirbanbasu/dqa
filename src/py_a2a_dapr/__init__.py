import logging
from rich.logging import RichHandler
from environs import Env

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # pragma: no cover
    # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

env = Env()
env.read_env()

logging.basicConfig(
    level=env.str("APP_LOG_LEVEL", default="INFO").upper(),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
