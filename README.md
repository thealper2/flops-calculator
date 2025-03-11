# flops-calculator

flops-calculator is a Python library designed to calculate the number of Floating Point Operations (FLOPs) and Multiply-Accumulate operations (MACs) for PyTorch models. It provides a simple and efficient way to analyze the computational complexity of neural networks without relying on external libraries.

---

## :dart: Features

- **FLOPs and MACs Calculation**: Accurately calculates FLOPs and MACs for various PyTorch layers, including convolutional, linear, recurrent, and normalization layers.
- **Layer-wise Breakdown**: Provides a detailed breakdown of FLOPs for each layer in the model.
- **Custom Input Shapes**: Supports custom input shapes for flexible analysis.
- **Device Support**: Works on both CPU and CUDA devices.
- **Verbose Mode**: Optionally prints detailed layer-by-layer information during calculation.

## :hammer_and_wrench: Installation

```bash
git clone https://github.com/thealper2/flops-calculator.git
cd flops-calculator
pip install .
```

## :joystick: Usage

### Basic Usage

To calculate FLOPs for a PyTorch model, simply create an instance of flops-calculator and call the calculate method:

```python
import torch
import torch.nn as nn
from flops_calculator import FLOPsCalculator

# Define a simple model
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 112 * 112, 10)
)

# Initialize the FLOPs calculator
input_shape = (3, 224, 224)  # Input shape (C, H, W)
calculator = FLOPsCalculator(model, input_shape)

# Calculate FLOPs
result = calculator.calculate()
print(f"Total FLOPs: {result['total_flops']}")
print(f"Total MACs: {result['total_macs']}")
print(f"Total Parameters: {result['total_params']}")
```

### Command Line Interface (CLI)

You can also use the flops-calculator from the command line:

```bash
flops-calculator --model resnet18 --input_shape 3 224 224 --device cpu --verbose
```

### Arguments

- **--model**: Path to a saved PyTorch model or a torchvision model name (e.g., `resnet18`).
- **--input_shape**: Input shape without the batch dimension (e.g., `3 224 224` for RGB images).
- **--device**: Device to run calculations on (`cpu` or `cuda`).
- **--verbose**: Print detailed layer-by-layer information.
- **--detailed**: Print a detailed breakdown of FLOPs per layer.
- **--mac**: Count Multiply-Accumulate operations (MACs) instead of FLOPs.

## :clipboard:Supported Layers

flops-calculator supports the following PyTorch layers:

- **Convolutional Layers**: `Conv1d`, `Conv2d`, `Conv3d`
- **Linear Layers**: `Linear`
- **Recurrent Layers**: `LSTM`, `GRU`
- **Normalization Layers**: `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`
- **Activation Functions**: `ReLU`, `ReLU6`, `LeakyReLU`, `Sigmoid`, `Tanh`
- **Pooling Layers**: `MaxPool1d`, `MaxPool2d`, `MaxPool3d`, `AvgPool1d`, `AvgPool2d`, `AvgPool3d`, `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveAvgPool3d`
- **Dropout Layers**: `Dropout`, `Dropout2d`, `Dropout3d`

## :handshake: Contributing

We welcome contributions! If you'd like to contribute to flops-calculator, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your fork.
4. Submit a pull request with a detailed description of your changes.

## :scroll: License

flops-calculator is licensed under the MIT License. See the LICENSE file for more details.
