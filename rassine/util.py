from typing import NoReturn


def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError(f"Invalid value: {x!r}")
