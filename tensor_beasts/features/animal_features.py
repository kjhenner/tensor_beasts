import torch

from tensor_beasts.features.feature import Feature


class IdFeature(Feature):
    name = "id"
    dtype = torch.int32


class OffspringCount(Feature):
    name = "offspring_count"
    dtype = torch.int32
