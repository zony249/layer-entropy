import os 
from abc import ABC
from typing import Any, Dict, List, Tuple, Optional, Union

import torch 
from torch.utils.data import Dataset, DataLoader 


class AbstractTask(ABC): 
    def __init__(self, 
                 list_splits: List[str], 
                 batch_size: Optional[int] = 8, 
                 local_dir: Optional[str] = None, 
                 load_local: Optional[bool] = False): 

        self.local_dir = local_dir
        self.load_local = load_local
        self.batch_size = batch_size

        self.datasets = None 
        self.dataloaders = None

        if load_local: 
            assert self.local_dir is not None 

        self.datasets = self.get_datasets(list_splits=list_splits) 
        # pre-process dataset
        for k, dataset in self.datasets.items(): 
            self.datasets[k] = self.preprocess_dataset(dataset)            
        
        self.dataloaders = self.get_dataloaders(list_splits=list_splits) 

    
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