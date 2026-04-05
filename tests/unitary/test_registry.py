import pytest
from src.representations import get_representation


def test_invalid_name():
    with pytest.raises(ValueError):
        get_representation("invalid_name")