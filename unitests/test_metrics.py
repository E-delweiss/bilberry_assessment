import os
import unittest
import os, sys
from pathlib import Path

import torch

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from metrics import class_acc, metrics

class TestMetrics(unittest.TestCase):
    def __init__(self, TestMetrics) -> None:
        super().__init__(TestMetrics)
        BATCH_SIZE = 32

        self.target = torch.zeros(BATCH_SIZE)
        self.target[:BATCH_SIZE//2] = torch.ones(BATCH_SIZE//2)
        self.prediction = torch.zeros(BATCH_SIZE)
    
    def test_class_acc(self):
        acc = class_acc(self.target, self.prediction)
        self.assertIs(type(acc), float)
        self.assertAlmostEqual(acc, 0.5)

    def test_metrics(self):
        metrics_dict = metrics(self.target, self.prediction)
        self.assertIs(type(metrics_dict), dict)
        self.assertGreaterEqual(metrics_dict["F1_score"], 0.)
        self.assertLessEqual(metrics_dict["F1_score"],1.)



if __name__ == "__main__":
    unittest.main()
        
