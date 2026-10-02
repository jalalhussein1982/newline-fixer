import pytest

from newline_fixer.example import EXAMPLE_INPUT, EXAMPLE_OUTPUT


@pytest.fixture
def example_input() -> str:
    return EXAMPLE_INPUT


@pytest.fixture
def example_output() -> str:
    return EXAMPLE_OUTPUT
