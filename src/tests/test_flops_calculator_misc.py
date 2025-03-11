import unittest
import torch.nn as nn
from src.flops_calculator.flops_calculator import FLOPsCalculator


class TestFLOPsCalculatorMisc(unittest.TestCase):
    def test_batchnorm_flops(self):
        model = nn.BatchNorm2d(16)
        input_shape = (16, 224, 224)
        calculator = FLOPsCalculator(model, input_shape)
        result = calculator.calculate()
        self.assertGreater(result["total_flops"], 0)
        self.assertGreater(result["total_macs"], 0)
        self.assertIn("BatchNorm2d", result["layer_flops"])

    def test_relu_flops(self):
        model = nn.ReLU()
        input_shape = (16, 224, 224)
        calculator = FLOPsCalculator(model, input_shape)
        result = calculator.calculate()
        self.assertGreater(result["total_flops"], 0)
        self.assertEqual(result["total_macs"], 0)
        self.assertIn("ReLU", result["layer_flops"])

    def test_maxpool_flops(self):
        model = nn.MaxPool2d(kernel_size=2, stride=2)
        input_shape = (16, 224, 224)
        calculator = FLOPsCalculator(model, input_shape)
        result = calculator.calculate()
        self.assertGreater(result["total_flops"], 0)
        self.assertEqual(result["total_macs"], 0)
        self.assertIn("MaxPool2d", result["layer_flops"])


if __name__ == "__main__":
    unittest.main()
