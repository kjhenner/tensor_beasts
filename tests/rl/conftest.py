import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_default_device():
    """Keep RL tests on CPU regardless of the machine's accelerators.

    Restored afterwards: the default device is global, so leaving it set makes
    the result of a later test depend on whether these ran first.
    """
    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        torch.set_default_device(previous)
