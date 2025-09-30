# Based on https://github.com/ollama/ollama-python/blob/main/examples/web-search-mcp.py


from dqa.mcp.mixin import MCPMixin

from importlib.metadata import version
from fastmcp import FastMCP


class BasicArithmeticMCP(MCPMixin):
    """
    A mixin-based MCP server for basic arithmetic operations.
    """

    tools = [
        {
            "fn": "add",
            "name": "add",
            "description": "Returns the sum of two numbers.",
            "tags": ["arithmetic", "math", "addition"],
        },
        {
            "fn": "subtract",
            "name": "subtract",
            "description": "Returns the difference of two numbers.",
            "tags": ["arithmetic", "math", "subtraction"],
        },
        {
            "fn": "multiply",
            "name": "multiply",
            "description": "Returns the product of two numbers.",
            "tags": ["arithmetic", "math", "multiplication"],
        },
        {
            "fn": "divide",
            "name": "divide",
            "description": "Returns the quotient of two numbers.",
            "tags": ["arithmetic", "math", "division"],
        },
        {
            "fn": "modulus",
            "name": "modulus",
            "description": "Returns the modulus of two numbers.",
            "tags": ["arithmetic", "math", "modulus"],
        },
        {
            "fn": "power",
            "name": "power",
            "description": "Returns the result of raising a base to an exponent.",
            "tags": ["arithmetic", "math", "exponentiation"],
        },
    ]

    def add(self, augend: float, summand: float) -> float:
        """
        Returns the sum of two numbers.
        """

        return augend + summand

    def subtract(self, minuend: float, subtrahend: float) -> float:
        """
        Returns the difference of two numbers.
        """

        return minuend - subtrahend

    def multiply(self, multiplicand: float, multiplier: float) -> float:
        """
        Returns the product of two numbers.
        """

        return multiplicand * multiplier

    def divide(self, dividend: float, divisor: float) -> float:
        """
        Returns the quotient of two numbers.
        """

        if divisor == 0:
            raise ValueError("Division by zero is not allowed.")
        return dividend / divisor

    def modulus(self, dividend: float, divisor: float) -> float:
        """
        Returns the modulus of two numbers.
        """

        if divisor == 0:
            raise ValueError("Division by zero is not allowed.")
        return dividend % divisor

    def power(self, base: float, exponent: float) -> float:
        """
        Returns the result of raising a base to an exponent.
        """

        return base**exponent


def app() -> FastMCP:  # pragma: no cover
    app = FastMCP(
        name="dqa-datetime",
        version=version("dqa"),
        instructions="A MCP for basic arithmetic operations.",
        on_duplicate_prompts="error",
        on_duplicate_resources="error",
        on_duplicate_tools="error",
    )
    mcp_obj = BasicArithmeticMCP()
    app_with_features = mcp_obj.register_features(app)
    return app_with_features
