import unittest
import torch.nn as nn
from src.flops_calculator.flops_calculator import FLOPsCalculator


class TestFLOPsCalculatorLinear(unittest.TestCase):
    def test_linear_flops(self):
        model = nn.Linear(128, 256)
        input_shape = (128,)
        calculator = FLOPsCalculator(model, input_shape)
        result = calculator.calculate()
        self.assertGreater(result["total_flops"], 0)
        self.assertGreater(result["total_macs"], 0)
        self.assertIn("Linear", result["layer_flops"])


if __name__ == "__main__":
    unittest.main()
