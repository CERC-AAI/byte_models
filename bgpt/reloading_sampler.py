import math
from typing import Optional, Iterator

import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, start_index: int = 0) -> None:
        # Initialize the parent DistributedSampler class with the provided arguments
        super(CustomDistributedSampler, self).__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle,
                                                       seed=seed, drop_last=drop_last)

        # Additional attribute for custom functionality
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        # Generate a list of indices as the DistributedSampler would
        indices = super(CustomDistributedSampler, self).__iter__()

        # Convert iterator to list to manipulate the starting index
        indices = list(indices)

        # Adjust for the custom starting index by rotating the list
        total_length = len(indices)
        adjusted_start_index = self.start_index % total_length
        indices = indices[adjusted_start_index:]

        return iter(indices)

    def set_index(self, index: int):
        # Set the starting index for sampling
        self.start_index = index
