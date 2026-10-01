import newline_fixer


def test_version_is_a_string() -> None:
    assert isinstance(newline_fixer.__version__, str)
