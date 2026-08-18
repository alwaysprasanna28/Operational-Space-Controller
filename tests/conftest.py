import numpy as np
import pytest

from khatib_osc.robot import load_robot


@pytest.fixture
def robot():
    r = load_robot()
    r.reset_home()
    return r


@pytest.fixture
def rng():
    return np.random.default_rng(0)
