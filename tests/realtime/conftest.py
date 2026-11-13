import pytest
from synthetic_dhm import SyntheticDHM


@pytest.fixture
def dhm():
    return SyntheticDHM()
