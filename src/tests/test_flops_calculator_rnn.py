import unittest
import torch.nn as nn
from src.flops_calculator.flops_calculator import FLOPsCalculator


class TestFLOPsCalculatorRNN(unittest.TestCase):
    def test_lstm_flops(self):
        model = nn.LSTM(input_size=128, hidden_size=256, num_layers=2)
        input_shape = (10, 128)  # (sequence_length, input_size)
        calculator = FLOPsCalculator(model, input_shape)
        result = calculator.calculate()
        self.assertGreater(result["total_flops"], 0)
        self.assertGreater(result["total_macs"], 0)
        self.assertIn("LSTM", result["layer_flops"])

    def test_gru_flops(self):
        model = nn.GRU(input_size=128, hidden_size=256, num_layers=2)
        input_shape = (10, 128)  # (sequence_length, input_size)
        calculator = FLOPsCalculator(model, input_shape)
        result = calculator.calculate()
        self.assertGreater(result["total_flops"], 0)
        self.assertGreater(result["total_macs"], 0)
        self.assertIn("GRU", result["layer_flops"])


if __name__ == "__main__":
    unittest.main()
