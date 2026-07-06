import os 
from abc import ABC
from typing import Any, Dict, List, Tuple, Optional, Union

import torch 
from torch.utils.data import Dataset, DataLoader 
from .abstract_task import AbstractTask


class Drop(AbstractTask): 
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    
    def get_datasets(self, list_splits: List[str]) -> List[Dataset]:
        raise NotImplementedError() 

    def get_dataloaders(self, list_splits: List[str]) -> List[DataLoader]:
        raise NotImplementedError()
    
    def get_dataset(self, split: str) -> Dataset: 
        raise NotImplementedError()

    def preprocess_sample(self, input_sample: Any): 
        raise NotImplementedError()

    def preprocess_dataset(self, dataset: Dataset) -> Dataset: 
        raise NotImplementedError() 

    # def collate_fn(self, batch) -> Any: 
    #     raise NotImplementedError()

if __name__ == "__main__": 
    pass