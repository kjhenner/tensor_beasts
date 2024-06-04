import numpy as np
import torch

VALUE_STORE = {}


def compare(key, value):
    global VALUE_STORE
    if type(value) == torch.Tensor:
        value = value.cpu().numpy()
    if VALUE_STORE.get(key, None) is not None:
        if not np.allclose(VALUE_STORE[key], value):
            print(f"Values for key {key} are not equal")
            print(f"Old value: {VALUE_STORE[key]}")
            print(f"New value: {value}")
            print(f"Difference: {np.sum(np.abs(VALUE_STORE[key] - value))}")
            assert False
        else:
            VALUE_STORE[key] = None
    else:
        VALUE_STORE[key] = value
