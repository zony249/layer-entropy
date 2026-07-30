import os 
import sys 
from typing import List, Dict, Union, Any, Optional, Tuple 
import math 

import numpy as np 
import torch 
from torch import nn 
from torch.utils.data import Dataset, DataLoader 
from datasets import load_dataset, load_from_disk

from .abstract_task import AbstractTask 

class Squad(AbstractTask): 
    def __init__(
        self,
        list_splits: List[str], 
        batch_size: Optional[int] = 8, 
        local_dir: Optional[str] = None, 
        load_local: Optional[bool] = False, 
        preprocess_validation: bool = False,
    ): 
        self.preprocess_validation = preprocess_validation
        super().__init__(list_splits, batch_size, local_dir, load_local) 
        self.local_dir = "squad-local" if self.local_dir is None else self.local_dir

    def get_datasets(self, list_splits: List[str]) -> Dict[str, Dataset]:
        if self.datasets is not None: 
            return {split:self.datasets[split] for split in list_splits}
        
        if self.load_local: 
            all_dset = load_from_disk(self.local_dir)
        else: 
            all_dset = load_dataset("rajpurkar/squad_v2")

        dataset = {split:all_dset[split] for split in list_splits}

        return dataset

    def get_dataloaders(self, list_splits: List[str]) -> Dict[str, DataLoader]:
        assert hasattr(self, "datasets"), "self.datasets does not exist!" 
        assert self.datasets is not None, "self.datasets is None..." 

        if self.dataloaders is not None: 
            return {split:self.dataloaders[split] for split in list_splits} 

        dataloaders = {}
        for split in list_splits: 
            if split == "train": 
                dataloader_kwargs = {"shuffle": True}
            elif split == "validation":
                dataloader_kwargs = {"shuffle": False}

            dataloader = DataLoader(self.datasets[split], 
                                    batch_size=self.batch_size, 
                                    # collate_fn=self.collate_fn, 
                                    **dataloader_kwargs)
            dataloaders[split] = dataloader 

        return dataloaders
    
    def get_dataset(self, split: str) -> Dataset: 
        if self.datasets is not None: 
            return self.datasets[split]
        if self.load_local: 
            dataset = load_from_disk(self.load_local)[split]
        else: 
            dataset = load_dataset("rajpurkar/squad_v2")[split] 
        return dataset

    def preprocess_sample(self, example: Any): 
        """
        takes data samples, converts them into strings (or tuples/dictionaries of strings)
        """
        return {"context": example["context"], 
                "question": example["question"], 
                "answers": example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else "Not enough information."}    

    # def remove_no_answer(self, example: Any): 
    #     return len(example["answers"]) > 0

    def preprocess_dataset(self, split:str, dataset: Dataset) -> Dataset: 
        
        if split == "train" or self.preprocess_validation: 
            dataset = dataset.map(self.preprocess_sample, remove_columns=["context", "question", "answers"])#.filter(self.remove_no_answer)
        return dataset

    # def collate_fn(self, batch: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    #     """
    #     unwraps the list of tuples input into a tuple of two lists.
    #     """ 
    #     raise NotImplementedError()





if __name__ == "__main__": 
    pass
    ds = load_dataset("rajpurkar/squad_v2")
    ds.save_to_disk("squad-local")
    ds.load_from_disk("squad-local")

    print(ds)

    # tset = ds["train"]
    # maxlen = -1
    # for ex in tset: 
    #     length = len(ex["answers"]["text"])
    #     if length > maxlen: 
    #         maxlen = length 
    # print(maxlen)

    squad = Squad(list_splits = ["train", "validation"])
    tset = squad.get_dataset("train")
    print(tset) 
    print(tset[0]) 
