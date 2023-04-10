import unittest
import os, sys
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from bilberry_dataset import get_training_dataset, get_validation_dataset

class TestBilberryDataset(unittest.TestCase):
    def __init__(self, TestBilberryDataset) -> None:
        super().__init__(TestBilberryDataset)
        self.size = 224
        self.batch_size = 32
        dataset_train = get_training_dataset(self.batch_size, ratio=1)        
        dataset_val = get_validation_dataset(self.batch_size, ratio=1)        

        self.output_train = next(iter(dataset_train))
        self.output_val = next(iter(dataset_val))

    def test_bilberry_dataset(self):
        ### Test on output type/size
        self.assertIs(type(self.output_train), list)
        self.assertEqual(len(self.output_train), 2)

        ### Test on output image shape
        self.assertEqual(len(self.output_train[0].shape), 4)
        self.assertEqual(self.output_train[0].shape[0], self.batch_size)
        self.assertEqual(self.output_train[0].shape[1], 3)
        self.assertEqual(self.output_train[0].shape[2], self.output_train[0].shape[3])
        
        ### Test on output target shape
        self.assertEqual(len(self.output_train[1].shape), 1)
        self.assertEqual(self.output_train[1].shape[0], self.batch_size)
        

if __name__ == "__main__":
    unittest.main()
