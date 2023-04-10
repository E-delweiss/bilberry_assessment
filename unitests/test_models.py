import unittest
import os, sys
from pathlib import Path

import torch
from torchinfo import summary

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from modelBuilder import resNetBilberry, efficientNetBilberry

class TestModel(unittest.TestCase):
    def __init__(self, TestModel) -> None:
        super().__init__(TestModel)
        self.size = 224
        self.channel_img = 3
        self.BATCH_SIZE = 32


    def test_resNetBilberry(self):
        model = resNetBilberry(load_resNetBilberry_weights=False, pretrained=True)
        img_test = torch.rand(self.BATCH_SIZE, self.channel_img, self.size, self.size)
        output = model(img_test)
        
        self.assertIs(type(output), torch.Tensor)
        self.assertIs(type(output[0].item()), float)
        # summary(model, input_size = img_test.shape)

    def test_efficientNetBilberry(self):
        model = efficientNetBilberry(load_efficientNetBilberry_weights=False, pretrained=True)
        img_test = torch.rand(self.BATCH_SIZE, self.channel_img, self.size, self.size)
        output = model(img_test)
        
        self.assertIs(type(output), torch.Tensor)
        self.assertIs(type(output[0].item()), float)
        # summary(model, input_size = img_test.shape)

if __name__ == "__main__":
    unittest.main()
