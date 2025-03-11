import unittest
import torch.nn as nn
from src.flops_calculator.flops_calculator import FLOPsCalculator


class TestFLOPsCalculatorConv(unittest.TestCase):
    def test_conv2d_flops(self):
        model = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        input_shape = (3, 224, 224)
        calculator = FLOPsCalculator(model, input_shape)
        result = calculator.calculate()
        self.assertGreater(result["total_flops"], 0)
        self.assertGreater(result["total_macs"], 0)
        self.assertIn("Conv2d", result["layer_flops"])

    def test_conv1d_flops(self):
        model = nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1)
        input_shape = (3, 224)
        calculator = FLOPsCalculator(model, input_shape)
        result = calculator.calculate()
        self.assertGreater(result["total_flops"], 0)
        self.assertGreater(result["total_macs"], 0)
        self.assertIn("Conv1d", result["layer_flops"])

    def test_conv3d_flops(self):
        model = nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1)
        input_shape = (3, 16, 16, 16)
        calculator = FLOPsCalculator(model, input_shape)
        result = calculator.calculate()
        self.assertGreater(result["total_flops"], 0)
        self.assertGreater(result["total_macs"], 0)
        self.assertIn("Conv3d", result["layer_flops"])


if __name__ == "__main__":
    unittest.main()
