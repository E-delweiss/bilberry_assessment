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
        self.SIZE = 448
        self.S = 7
        self.B = 1
        self.C = 8
        self.CELL_SIZE = 1/self.S

        dataset_train = get_training_dataset()        
        dataset_val = get_validation_dataset()        

        output_train = next(iter(dataset_train))
        output_val = next(iter(dataset_val))

    def test_my_mealtrays_dataset(self):
        ###### TODO BILBERRY


        ### Test on output type/size
        self.assertIs(type(self.output), tuple)
        self.assertEqual(len(self.output), 2)

        ### Test on output image shape
        self.assertEqual(len(self.output[0].shape), 3)
        self.assertEqual(self.output[0].shape[1], self.output[0].shape[2])
        
        ### Test on output target shape
        self.assertEqual(len(self.output[1].shape), 3)
        self.assertEqual(self.output[1].shape[0], self.S)
        self.assertEqual(self.output[1].shape[0], self.output[1].shape[1])
        self.assertEqual(self.output[1].shape[2], self.B*(4+1) + self.C)

    

if __name__ == "__main__":
    unittest.main()
