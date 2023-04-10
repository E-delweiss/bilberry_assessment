import unittest
import os, sys
from pathlib import Path

import torch
import numpy as np

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from modelBuilder import resNetBilberry
from class_activation_map import Cam_model, plot_CAM, convert_and_classify

class TestClassActivationMap(unittest.TestCase):
    def __init__(self, TestClassActivationMap):
        super().__init__(TestClassActivationMap)
        self.BATCH_SIZE = 1
        self.input_tensor = torch.rand(self.BATCH_SIZE, 3, 224, 224)
        self.model = resNetBilberry(pretrained=False)

    def test_cam_model(self):  
        cam_model = Cam_model(self.model)
        output = cam_model(self.input_tensor)

        self.assertIs(type(output), tuple)
        self.assertEqual(output[0].shape, torch.Size([self.BATCH_SIZE, 512, 7, 7]))
        self.assertEqual(output[1].shape, torch.Size([self.BATCH_SIZE, 1]))

if __name__ == "__main__":
    unittest.main()
