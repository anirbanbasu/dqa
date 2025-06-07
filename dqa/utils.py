import os
from typing import Any

SPACE_STRING = " "
TRUE_VALUES_LIST = ["true", "1", "yes", "on", "yes", "y", "t"]
FALSE_VALUES_LIST = ["false", "0", "no", "off", "n", "f"]


def parse_env(
    var_name: str,
    default_value: str | None = None,
    allowed_values: list[str] | None = None,
    type_cast=str,
    convert_to_list=False,
    list_split_char=SPACE_STRING,
) -> Any | list[Any]:
    """
    Parse the environment variable and return the value.

    Args:
        var_name (str): The name of the environment variable.
        default_value (str | None): The default value to use if the environment variable is not set. Defaults to None.
        allowed_values (list[str] | None): A list of allowed values for the environment variable. If provided, the
            value will be checked against this list. This option is ignored if type_cast is bool.
        type_cast (str): The type to cast the value to.
        convert_to_list (bool): Whether to convert the value to a list.
        list_split_char (str): The character to split the list on.

    Returns:
        (Any | list[Any]) The parsed value, either as a single value or a list. The type of the returned single
        value or individual elements in the list depends on the supplied type_cast parameter.
    """
    if os.getenv(var_name) is None and default_value is None:
        raise ValueError(
            f"Environment variable {var_name} does not exist and a default value has not been provided."
        )
    parsed_value = None
    if type_cast is bool:
        parsed_value = os.getenv(var_name, default_value).lower() in TRUE_VALUES_LIST
    else:
        parsed_value = os.getenv(var_name, default_value)
        if allowed_values is not None:
            if parsed_value not in allowed_values:
                raise ValueError(
                    f"Environment variable {var_name} has value '{parsed_value}', "
                    f"which is not in the allowed values: {allowed_values}."
                )

    value: Any | list[Any] = (
        type_cast(parsed_value)
        if not convert_to_list
        else [type_cast(v) for v in parsed_value.split(list_split_char)]
    )
    return value
