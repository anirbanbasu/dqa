from typing import Annotated
from fastmcp import FastMCP

app = FastMCP(
    instructions="A collection of arithmetic tools.",
    on_duplicate_prompts="error",
    on_duplicate_resources="error",
    on_duplicate_tools="error",
)


@app.tool(
    tags=["arithmetic", "math", "addition"],
)
def add(
    a: Annotated[float, "The first number"], b: Annotated[float, "The summand"]
) -> float:
    """Adds two numbers together."""
    return a + b


@app.tool(
    tags=["arithmetic", "math", "addition", "multiple"],
)
def add_multiple(
    numbers: Annotated[list[float], "A list of numbers to add together"],
) -> float:
    """
    Adds multiple numbers together.
    """
    return sum(numbers)


@app.tool(
    tags=["arithmetic", "math", "subtraction"],
)
def subtract(
    a: Annotated[float, "The minuend"], b: Annotated[float, "The subtrahend"]
) -> float:
    """
    Subtracts one number from another.
    """
    return a - b


@app.tool(
    tags=["arithmetic", "math", "multiplication"],
)
def multiply(
    a: Annotated[float, "The multiplicand"], b: Annotated[float, "The multiplier"]
) -> float:
    """
    Multiplies two numbers together.
    """
    return a * b


@app.tool(
    tags=["arithmetic", "math", "division"],
)
def divide(
    a: Annotated[float, "The dividend"], b: Annotated[float, "The divisor"]
) -> float:
    """
    Divides one number by another.
    Raises an error if the divisor is zero.
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@app.tool(
    tags=["arithmetic", "math", "modulus"],
)
def modulus(
    a: Annotated[float, "The dividend"], b: Annotated[float, "The divisor"]
) -> float:
    """
    Returns the modulus of one number by another.
    Raises an error if the divisor is zero."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a % b


@app.tool(
    tags=["arithmetic", "math", "exponentiation"],
)
def power(
    base: Annotated[float, "The base number"],
    exponent: Annotated[float, "The exponent"],
) -> float:
    """Raises the base number to the power of the exponent."""
    return base**exponent
