import unittest
import torch.nn as nn
from src.flops_calculator.flops_calculator import FLOPsCalculator


class TestFLOPsCalculatorInit(unittest.TestCase):
    def test_valid_initialization(self):
        model = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        input_shape = (3, 224, 224)
        calculator = FLOPsCalculator(model, input_shape)
        self.assertEqual(calculator.input_shape, input_shape)
        self.assertEqual(calculator.device, "cpu")
        self.assertFalse(calculator.verbose)
        self.assertFalse(calculator.count_mac)

    def test_invalid_model_type(self):
        with self.assertRaises(TypeError):
            FLOPsCalculator("not_a_model", (3, 224, 224))

    def test_invalid_input_shape_type(self):
        model = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        with self.assertRaises(TypeError):
            FLOPsCalculator(model, [3, 224, 224])

    def test_invalid_device(self):
        model = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        with self.assertRaises(ValueError):
            FLOPsCalculator(model, (3, 224, 224), device="invalid_device")


if __name__ == "__main__":
    unittest.main()
