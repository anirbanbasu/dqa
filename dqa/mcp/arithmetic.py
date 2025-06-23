from typing import Annotated
from fastmcp import FastMCP

app = FastMCP(
    instructions="A collection of arithmetic tools.",
    on_duplicate_prompts="error",
    on_duplicate_resources="error",
    on_duplicate_tools="error",
)


@app.tool(
    name="add",
    description="Adds two numbers together.",
    tags=["arithmetic", "math", "addition"],
)
def add(
    a: Annotated[float, "The first number"], b: Annotated[float, "The summand"]
) -> float:
    return a + b


@app.tool(
    name="subtract",
    description="Subtracts one number from another.",
    tags=["arithmetic", "math", "subtraction"],
)
def subtract(
    a: Annotated[float, "The minuend"], b: Annotated[float, "The subtrahend"]
) -> float:
    return a - b


@app.tool(
    name="multiply",
    description="Multiplies two numbers together.",
    tags=["arithmetic", "math", "multiplication"],
)
def multiply(
    a: Annotated[float, "The multiplicand"], b: Annotated[float, "The multiplier"]
) -> float:
    return a * b


@app.tool(
    name="divide",
    description="Divides one number by another.",
    tags=["arithmetic", "math", "division"],
)
def divide(
    a: Annotated[float, "The dividend"], b: Annotated[float, "The divisor"]
) -> float:
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@app.tool(
    name="modulus",
    description="Calculates the modulus of one number by another.",
    tags=["arithmetic", "math", "modulus"],
)
def modulus(
    a: Annotated[float, "The dividend"], b: Annotated[float, "The divisor"]
) -> float:
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a % b


@app.tool(
    name="power",
    description="Raises one number to the power of another.",
    tags=["arithmetic", "math", "exponentiation"],
)
def power(
    base: Annotated[float, "The base number"],
    exponent: Annotated[float, "The exponent"],
) -> float:
    return base**exponent
