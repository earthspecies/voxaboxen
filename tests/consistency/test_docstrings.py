"""Tests for checking the docstrings of functions and classes.

Header info specific to ESP
"""

from tests.utils.check_docstrings import check_docstrings


def test_docstrings_exist(base_folder: str, skip_files_list: list) -> None:
    """Check that all class and functions contain a docstring.
    Numpy-style is used.

    Arguments
    ---------
    base_folder: str, optional
        Path to the base folder where this function is executed.
    skip_files_list: list[str], optional
        List of filename that should be skipped.
    """
    print(skip_files_list)
    assert check_docstrings(base_folder, None, skip_files_list=skip_files_list)
