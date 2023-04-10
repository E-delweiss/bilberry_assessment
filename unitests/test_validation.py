import unittest
import os, sys
from pathlib import Path

import torch
import numpy as np

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from bilberry_dataset import get_validation_dataset
from validation import validation_loop
from modelBuilder import resNetBilberry

class TestValidation(unittest.TestCase):
    def __init__(self, TestUtils) -> None:
        super().__init__(TestUtils)
        BATCH_SIZE = 32
        validation_loader = get_validation_dataset(BATCH_SIZE, ratio=1, isAugment=False)
        criterion = torch.nn.BCELoss()
        model = resNetBilberry(pretrained=False)
        self.output = validation_loop(model, validation_loader, criterion, DO_VALIDATION=True)


    def test_validation(self):  
        val_loss, val_acc, confmatrix_dict = self.output

        self.assertIs(type(val_loss.item()), float)
        self.assertIs(type(val_acc.item()), float)
        self.assertIs(type(confmatrix_dict), dict)

        self.assertGreater(val_loss, 0)
        self.assertGreaterEqual(val_acc, 0)
        self.assertLessEqual(val_acc, 1)
        

if __name__ == "__main__":
    unittest.main()
        
