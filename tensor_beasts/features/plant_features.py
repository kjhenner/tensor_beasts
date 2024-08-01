import torch

from tensor_beasts.features.feature import Feature


class Seed(Feature):
    name = "seed"
    dtype = torch.uint8


class Crowding(Feature):
    name = "crowding"
    dtype = torch.float32
