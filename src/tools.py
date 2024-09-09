# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions wrapped as tools used by LLMs and agents for various tasks."""

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

import math

from utils import EMPTY_STRING


class DuckDuckGoFullSearchOnlyToolSpec(DuckDuckGoSearchToolSpec):
    """Modified version of DuckDuckGoSearch tool spec because we do not need the duckduckgo_instant_search."""

    spec_functions = ["duckduckgo_full_search"]


class StringFunctionsToolSpec(BaseToolSpec):
    """Tool spec for some string manipulation functions."""

    def __init__(self):
        self.spec_functions = [
            method
            for method in self.__class__.__dict__
            if callable(getattr(self.__class__, method)) and not method.startswith("__")
        ]

    def sf_count_substrings(self, string: str, substring: str) -> int:
        """
        StringFunctions: Counts the number of times a substring appears in a string.

        Args:
            string (str): The string to search.
            substring (str): The substring to search for. This can be a single character or a sequence of characters.

        Returns:
            int: The number of times the substring appears in the string.
        """
        if not string or not substring:
            raise ValueError("Both the string and the substring must be provided.")
        if string == EMPTY_STRING or substring == EMPTY_STRING:
            return 0
        return string.count(substring)

    def sf_is_palindrome(self, string: str) -> bool:
        """
        StringFunctions: Checks if a string is a palindrome.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the string is a palindrome, False otherwise.
        """
        if not string:
            raise ValueError("The string must be provided.")
        return string == string[::-1]


class BasicArithmeticCalculatorSpec(BaseToolSpec):
    """Tool spec for basic arithmetic operations and number comparison."""

    def __init__(self):
        self.spec_functions = [
            method
            for method in self.__class__.__dict__
            if callable(getattr(self.__class__, method)) and not method.startswith("__")
        ]

    def bac_add(self, a: int | float, b: int | float) -> int | float:
        """
        BasicArithmeticCalculator: Adds two numbers.

        Args:
            a (int | float): The first number.
            b (int | float): The second number.

        Returns:
            int | float: The sum of the two numbers.
        """
        return a + b

    def bac_subtract(self, a: int | float, b: int | float) -> int | float:
        """
        BasicArithmeticCalculator: Subtracts one number from another.

        Args:
            a (int | float): The number to subtract from.
            b (int | float): The number to subtract.

        Returns:
            int | float: The result of the subtraction.
        """
        return a - b

    def bac_multiply(self, a: int | float, b: int | float) -> int | float:
        """
        BasicArithmeticCalculator: Multiplies two numbers.

        Args:
            a (int | float): The first number.
            b (int | float): The second number.

        Returns:
            int | float: The product of the two numbers.
        """
        return a * b

    def bac_divide(self, a: int | float, b: int | float) -> int | float:
        """
        BasicArithmeticCalculator: Divides one number by another.

        Args:
            a (int | float): The dividend.
            b (int | float): The divisor.

        Returns:
            int | float: The result of the division.
        """
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b

    def bac_modulo(self, a: int, b: int) -> int:
        """
        BasicArithmeticCalculator: Computes the modulo of one number by another.

        Args:
            a (int): The number to find the modulo of.
            b (int): The modulo.

        Returns:
            int: The result of the modulo operation.
        """
        if b == 0:
            raise ValueError("Modulo by zero is not allowed.")
        return a % b

    def bac_power(self, base: int | float, exponent: int | float) -> int | float:
        """
        BasicArithmeticCalculator: Raises one number to the power of another.

        Args:
            base (int | float): The base. Must not be zero if the exponent is negative.
            exponent (int | float): The exponent.

        Returns:
            int | float: The result of the power operation.
        """
        if base == 0 and exponent < 0:
            raise ValueError("Zero raised to a negative power is undefined.")
        return base**exponent

    def bac_floor_divide(self, a: int | float, b: int | float) -> int:
        """
        BasicArithmeticCalculator: Divides one number by another and returns the floor of the quotient.

        Args:
            a (int | float): The dividend.
            b (int | float): The divisor.

        Returns:
            int: The floor of the result of the division.
        """
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a // b

    def bac_compare(self, a: int | float, b: int | float) -> int:
        """
        BasicArithmeticCalculator: Compares two numbers.

        Args:
            a (int | float): The first number.
            b (int | float): The second number.

        Returns:
            int: The comparison result, which is 0 if the numbers are equal,
            1 if the first number is greater, and -1 if the second number is greater.
        """
        return (a > b) - (a < b)


class MathematicalFunctionsSpec(BaseToolSpec):
    """Tool spec for selected mathematical functions from the math library."""

    def __init__(self):
        self.spec_functions = [
            method
            for method in self.__class__.__dict__
            if callable(getattr(self.__class__, method)) and not method.startswith("__")
        ]

    def math_factorial(self, n: int) -> int:
        """
        MathematicalFunctions: Computes the factorial of a non-negative integer.

        Args:
            n (int): The number to find the factorial of.

        Returns:
            int: The factorial of the number.
        """
        return math.factorial(n)

    def math_combinations(self, n: int, r: int) -> int:
        """
        MathematicalFunctions: Computes the number of combinations of n items taken r at a time.
        Both n and r must be non-negative integers.
        The function evaluates to n! / (k! * (n - k)!) when k <= n and evaluates to zero when k > n.

        Args:
            n (int): The total number of items.
            r (int): The number of items to choose.

        Returns:
            int: The number of combinations.
        """
        return math.comb(n, r)

    def math_permutations(self, n: int, r: int) -> int:
        """
        MathematicalFunctions: Computes the number of permutations of n items taken r at a time.
        Both n and r must be non-negative integers.
        Evaluates to n! / (n - k)! when k <= n and evaluates to zero when k > n.


        Args:
            n (int): The total number of items.
            r (int): The number of items to choose.

        Returns:
            int: The number of permutations.
        """
        return math.perm(n, r)

    def math_gcd(self, *integers: int) -> int:
        """
        MathematicalFunctions: Computes the greatest common divisor of a set of integers.

        Args:
            *integers (int): The integers to find the greatest common divisor of.

        Returns:
            int: The greatest common divisor.
        """
        return math.gcd(*integers)

    def math_lcm(self, *integers: int) -> int:
        """
        MathematicalFunctions: Computes the least common multiple of a set of integers.

        Args:
            *integers (int): The integers to find the least common multiple of.

        Returns:
            int: The least common multiple.
        """
        return math.lcm(*integers)

    def math_sqrt(self, n: int | float) -> float:
        """
        MathematicalFunctions: Computes the square root of a number.

        Args:
            n (int | float): The number to find the square root of.

        Returns:
            float: The square root of the number.
        """
        return math.sqrt(n)

    def math_cube_root(self, n: int | float) -> float:
        """
        MathematicalFunctions: Computes the cube root of a number.

        Args:
            n (int | float): The number to find the cube root of.

        Returns:
            float: The cube root of the number.
        """
        return math.cbrt(n)

    def math_nth_root(self, x: int | float, n: int) -> float:
        """
        MathematicalFunctions: Computes the nth root of a number.

        Args:
            x (int | float): The number to find the nth root of.
            n (int): The root to find.

        Returns:
            float: The nth root of the number.
        """
        # Alternatively: x ** (1 / n)
        return math.pow(x, 1 / n)

    def math_exponential(self, n: int | float) -> float:
        """
        MathematicalFunctions: Computes the value of e raised to the power n,
        where e = 2.718281… is the base of natural logarithms.

        Args:
            n (int | float): The number to find the exponential of.

        Returns:
            float: The exponential of the number.
        """
        return math.exp(n)

    def math_logarithm(self, n: int | float, base: int | float = math.e) -> float:
        """
        MathematicalFunctions: Computes the logarithm of a number to a given base.
        To calculate the natural logarithm of the number, do not provide a base.

        Args:
            n (int | float): The number to find the logarithm of.
            base (int | float): The base of the logarithm. Defaults to the natural logarithm base e.

        Returns:
            float: The logarithm of the number to the given base.
        """
        return math.log(n, base)

    def math_sine(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the sine of an angle in radians.

        Args:
            x (int | float): The angle in radians.

        Returns:
            float: The sine of the angle.
        """
        return math.sin(x)

    def math_cosine(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the cosine of an angle in radians.

        Args:
            x (int | float): The angle in radians.

        Returns:
            float: The cosine of the angle.
        """
        return math.cos(x)

    def math_tangent(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the tangent of an angle in radians.

        Args:
            x (int | float): The angle in radians.

        Returns:
            float: The tangent of the angle.
        """
        return math.tan(x)

    def math_arc_sine(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the inverse sine of a number.

        Args:
            x (int | float): The number to find the inverse sine of.

        Returns:
            float: The inverse sine of the number.
        """
        return math.asin(x)

    def math_arc_cosine(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the inverse cosine of a number.

        Args:
            x (int | float): The number to find the inverse cosine of.

        Returns:
            float: The inverse cosine of the number.
        """
        return math.acos(x)

    def math_arc_tangent(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the inverse tangent of a number.

        Args:
            x (int | float): The number to find the inverse tangent of.

        Returns:
            float: The inverse tangent of the number.
        """
        return math.atan(x)

    def math_radians_to_degrees(self, x: int | float) -> float:
        """
        MathematicalFunctions: Converts an angle from radians to degrees.

        Args:
            x (int | float): The angle in radians.

        Returns:
            float: The angle in degrees.
        """
        return math.degrees(x)

    def math_degrees_to_radians(self, x: int | float) -> float:
        """
        MathematicalFunctions: Converts an angle from degrees to radians.

        Args:
            x (int | float): The angle in degrees.

        Returns:
            float: The angle in radians.
        """
        return math.radians(x)

    def math_distance(
        self, x1: int | float, y1: int | float, x2: int | float, y2: int | float
    ) -> float:
        """
        MathematicalFunctions: Computes the Euclidean distance between two points in a plane.

        Args:
            x1 (int | float): The x-coordinate of the first point.
            y1 (int | float): The y-coordinate of the first point.
            x2 (int | float): The x-coordinate of the second point.
            y2 (int | float): The y-coordinate of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return math.dist([x1, y1], [x2, y2])

    # TODO: Wrap math.hypot

    def math_hyperbolic_sine(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the hyperbolic sine of a number.

        Args:
            x (int | float): The number to find the hyperbolic sine of.

        Returns:
            float: The hyperbolic sine of the number.
        """
        return math.sinh(x)

    def math_hyperbolic_cosine(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the hyperbolic cosine of a number.

        Args:
            x (int | float): The number to find the hyperbolic cosine of.

        Returns:
            float: The hyperbolic cosine of the number.
        """
        return math.cosh(x)

    def math_hyperbolic_tangent(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the hyperbolic tangent of a number.

        Args:
            x (int | float): The number to find the hyperbolic tangent of.

        Returns:
            float: The hyperbolic tangent of the number.
        """
        return math.tanh(x)

    def math_hyperbolic_arc_sine(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the inverse hyperbolic sine of a number.

        Args:
            x (int | float): The number to find the inverse hyperbolic sine of.

        Returns:
            float: The inverse hyperbolic sine of the number.
        """
        return math.asinh(x)

    def math_hyperbolic_arc_cosine(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the inverse hyperbolic cosine of a number.

        Args:
            x (int | float): The number to find the inverse hyperbolic cosine of.

        Returns:
            float: The inverse hyperbolic cosine of the number.
        """
        return math.acosh(x)

    def math_hyperbolic_arc_tangent(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the inverse hyperbolic tangent of a number.

        Args:
            x (int | float): The number to find the inverse hyperbolic tangent of.

        Returns:
            float: The inverse hyperbolic tangent of the number.
        """
        return math.atanh(x)

    def math_gamma(self, x: int | float) -> float:
        """
        MathematicalFunctions: Computes the gamma function of a number.

        Args:
            x (int | float): The number to find the gamma function of.

        Returns:
            float: The gamma function of the number.
        """
        return math.gamma(x)
