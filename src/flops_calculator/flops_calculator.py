import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Set, Any

from src.flops_calculator.utils import format_flops


class FLOPsCalculator:
    """
    A class to calculate FLOPs (Floating Point Operations) for PyTorch neural networks models.

    Attributes:
        model (nn.Module): The PyTorch model to analyze
        input_shape (Tuple[int, ...]): The shape of the input tensor (excluding batch dimension)
        device (str): Device to run calculations on ('cpu' or 'cuda')
        verbose (bool): Whether to print detailed layer-by-layer information
        count_mac (bool): Whether to count Multiply-Accumulate operations instaed of FLOPs
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cpu",
        verbose: bool = False,
        count_mac: bool = False,
    ):
        """
        Initialize the FLOPs calculator.

        Args:
            model (nn.Module): The PyTorch model to analyze
            input_shape (Tuple[int, ...]): The shape of the input tensor (excluding batch dimension)
            device (str): Device to run calculations on ('cpu' or 'cuda')
            verbose (bool): Whether to print detailed layer-by-layer information
            count_mac (bool): Whether to count Multiply-Accumulate operations instaed of FLOPs

        Raises:
            TypeError: If model is not a PyTorch module or input_shape is not a tuple
            ValueError: If device is not 'cpu' or 'cuda'
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be a PyTorch nn.Module")

        if not isinstance(input_shape, tuple):
            raise TypeError("Input shape must be a tuple of integers")

        self.model = model
        self.input_shape = input_shape
        self.device = device
        self.verbose = verbose
        self.count_mac = count_mac

        # Move model to specified device
        try:
            self.model.to(self.device)
        except RuntimeError as e:
            print(f"Error moving model to {self.device}: {str(e)}")
            print("Falling back to CPU")
            self.device = "cpu"
            self.model.to(self.device)

        self.model.eval()  # Set model to evaluation mode

        # Handlers for each supported layer type
        self.handlers = {
            nn.Conv1d: self._conv1d_flops,
            nn.Conv2d: self._conv2d_flops,
            nn.Conv3d: self._conv3d_flops,
            nn.Linear: self._linear_flops,
            nn.LSTM: self._lstm_flops,
            nn.GRU: self._gru_flops,
            nn.BatchNorm1d: self._bn_flops,
            nn.BatchNorm2d: self._bn_flops,
            nn.BatchNorm3d: self._bn_flops,
            nn.ReLU: self._activation_flops,
            nn.ReLU6: self._activation_flops,
            nn.LeakyReLU: self._activation_flops,
            nn.Sigmoid: self._activation_flops,
            nn.Tanh: self._activation_flops,
            nn.MaxPool1d: self._pool_flops,
            nn.MaxPool2d: self._pool_flops,
            nn.MaxPool3d: self._pool_flops,
            nn.AvgPool1d: self._pool_flops,
            nn.AvgPool2d: self._pool_flops,
            nn.AvgPool3d: self._pool_flops,
            nn.AdaptiveAvgPool1d: self._adaptive_pool_flops,
            nn.AdaptiveAvgPool2d: self._adaptive_pool_flops,
            nn.AdaptiveAvgPool3d: self._adaptive_pool_flops,
            nn.Dropout: self._zero_flops,
            nn.Dropout2d: self._zero_flops,
            nn.Dropout3d: self._zero_flops,
        }

        # Initialize tensors dictionary to track tensor shapes
        self.tensor_shapes: Dict[str, Tuple[int, ...]] = {}
        self.total_flops = 0
        self.layer_flops: Dict[str, int] = {}

        # Keep track of processed modules to avoid double-counting in complex models with shared modules
        self.processed_modules: Set[int] = set()

    def _register_hook(self) -> List:
        """
        Register forward hooks on model layers to calculate FLOPs.

        This method adds hooks to all model layers to track input and output tensor shapes
        and calculate FLOPs for each operation.

        Returns:
            List: List of hooks for later removal
        """

        def _pre_hook(
            module: nn.Module, input_tensors: Tuple[torch.Tensor, ...]
        ) -> None:
            if input_tensors:
                module_name = str(module).split("(")[0]
                input_tensor = input_tensors[0]
                self.tensor_shapes[module_name + "_input"] = input_tensor.shape

        def _post_hook(
            module: nn.Module,
            input_tensors: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            # Skip if this module has already been processed (for shared modules)
            module_id = id(module)
            if module_id in self.processed_modules:
                return

            self.processed_modules.add(module_id)

            module_name = str(module).split("(")[0]

            # Handle case where output is a tuple
            if isinstance(output, tuple):
                output_shape = output[0].shape
            else:
                output_shape = output.shape

            self.tensor_shapes[module_name + "_output"] = output_shape

            # Calculate FLOPs for this module
            module_type = type(module)
            if module_type in self.handlers:
                flops = self.handlers[module_type](module, input_tensors[0], output)
                module_id = module_name + "_" + str(id(module))[-6:]
                self.layer_flops[module_id] = flops
                self.total_flops += flops

                if self.verbose:
                    print(f"{module_id:<30} | {module_type.__name__:<15} | {flops:,}")

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if type(module) in self.handlers:
                pre_hook = module.register_forward_pre_hook(_pre_hook)
                post_hook = module.register_forward_hook(_post_hook)
                hooks.append(pre_hook)
                hooks.append(post_hook)

        return hooks

    def _remove_hooks(self, hooks: List) -> None:
        """
        Remove all registered hooks.
        """
        for hook in hooks:
            hook.remove()

    def calculate(self) -> Dict[str, Any]:
        """
        Calculate FLOPs for the model.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - total_flops: Total FLOPs for the model
                - total_macs: Total MACs for the model
                - layer_flops: FLOPs per layer
                - total_params: Total number of parameters

        Raises:
            RuntimeError: If an error occurs during the forward pass
        """
        if self.verbose:
            print(
                f"\n{'Layer':<30} | {'Type':<15} |  {'FLOPs' if not self.count_mac else 'MACs'}"
            )
            print("-" * 60)

        # Reset tracking variables
        self.total_flops = 0
        self.layer_flops = {}
        self.processed_modules = set()

        # Register hooks
        hooks = self._register_hook()

        try:
            # Create a dummy input
            x = torch.rand(1, *self.input_shape).to(self.device)

            # Run the model
            with torch.no_grad():
                self.model(x)

        except Exception as e:
            raise RuntimeError(f"Error during model forward pass: {str(e)}")

        finally:
            # Remove hooks
            self._remove_hooks(hooks)

        # Count parameters
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Calculate MACs (half of FLOPs for most operations)
        total_macs = self.total_flops // 2 if not self.count_mac else self.total_flops

        result = {
            "total_flops": self.total_flops
            if not self.count_mac
            else self.total_flops * 2,
            "total_macs": total_macs,
            "layer_flops": self.layer_flops,
            "total_params": total_params,
        }
        return result

    def print_summary(self, detailed: bool = False) -> None:
        """
        Print a summary of the FLOPs calculation.

        Args:
            detailed (bool): Whether to print detailed layer-by-layer breakdown
        """
        result = self.calculate()

        print("\n" + "=" * 50)
        print("MODEL COMPLEXITY ANALYSIS")
        print("=" * 50)

        if self.count_mac:
            print(
                f"Computational complexity: {format_flops(result['total_macs'], mac=True)}"
            )
        else:
            print(f"Computational complexity: {format_flops(result['total_flops'])}")
            print(
                f"                          {format_flops(result['total_macs'], mac=True)}"
            )

        print(f"Number of parameters:     {result['total_params'] / 1e6:.2f} M")

        if detailed:
            print("\nLayer-by-layer breakdown:")
            measure = "MACs" if self.count_mac else "FLOPs"
            print(f"{'Layer':<30} | {measure:<15} | {'% of Total'}")
            print("-" * 60)

            # Sort layers by FLOPs (descending)
            sorted_layers = sorted(
                result["layer_flops"].items(), key=lambda x: x[1], reverse=True
            )

            total = result["total_macs"] if self.count_mac else result["total_flops"]
            for layer_name, ops in sorted_layers:
                percentage = (ops / total) * 100
                print(f"{layer_name:<30} | {ops:,} | {percentage:.2f}%")

    # FLOPs calculation handlers for different layer types
    def _conv2d_flops(
        self, module: nn.Conv2d, input_tensor: torch.Tensor, output: torch.Tensor
    ) -> int:
        """Calculate FLOPs for 2D convolution."""
        batch_size = input_tensor.shape[0]
        output_h, output_w = output.shape[2:]

        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups

        # Each output element needs kernel_h * kernel_w * in_channels/groups * out_channels MACs
        flops_per_element = kernel_h * kernel_w * (in_channels // groups) * out_channels

        # Total MACs: MACs_per_element * output_elements
        flops = batch_size * output_h * output_w * flops_per_element

        # Add bias if present (one addition per output element)
        if module.bias is not None:
            bias_flops = batch_size * output_h * output_w * out_channels
            # Only count bias for FLOPs, not for MACs
            if not self.count_mac:
                flops += bias_flops

        # For FLOPs count, multiply by 2 (multiply + add)
        if not self.count_mac:
            flops *= 2

        return flops

    def _conv1d_flops(
        self, module: nn.Conv1d, input_tensor: torch.Tensor, output: torch.Tensor
    ) -> int:
        """Calculate FLOPs for 1D convolution."""
        batch_size = input_tensor.shape[0]
        output_length = output.shape[2]

        kernel_size = module.kernel_size[0]
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups

        flops_per_element = kernel_size * (in_channels // groups) * out_channels
        flops = batch_size * output_length * flops_per_element

        if module.bias is not None:
            bias_flops = batch_size * output_length * out_channels
            if not self.count_mac:
                flops += bias_flops

        if not self.count_mac:
            flops *= 2

        return flops

    def _conv3d_flops(
        self, module: nn.Conv3d, input_tensor: torch.Tensor, output: torch.Tensor
    ) -> int:
        """Calculate FLOPs for 3D convolution."""
        batch_size = input_tensor.shape[0]
        output_d, output_h, output_w = output.shape[2:]

        kernel_d, kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups

        flops_per_element = (
            kernel_d * kernel_h * kernel_w * (in_channels // groups) * out_channels
        )
        flops = batch_size * output_d * output_h * output_w * flops_per_element

        if module.bias is not None:
            bias_flops = batch_size * output_d * output_h * output_w * out_channels
            if not self.count_mac:
                flops += bias_flops

        if not self.count_mac:
            flops *= 2

        return flops

    def _linear_flops(
        self, module: nn.Linear, input_tensor: torch.Tensor, output: torch.Tensor
    ) -> int:
        """Calculate FLOPs for fully connected layer."""
        batch_size = input_tensor.shape[0]
        in_features = module.in_features
        out_features = module.out_features

        # MAC operations
        flops = batch_size * in_features * out_features

        # Add bias if present
        if module.bias is not None:
            bias_flops = batch_size * out_features
            if not self.count_mac:
                flops += bias_flops

        if not self.count_mac:
            flops *= 2

        return flops

    def _lstm_flops(
        self, module: nn.LSTM, input_tensor: torch.Tensor, output: torch.Tensor
    ) -> int:
        """Calculate FLOPs for LSTM layer."""
        batch_size = input_tensor.shape[0]
        seq_length = input_tensor.shape[1] if len(input_tensor.shape) > 2 else 1
        input_size = (
            input_tensor.shape[2]
            if len(input_tensor.shape) > 2
            else input_tensor.shape[1]
        )
        hidden_size = module.hidden_size
        num_layers = module.num_layers
        bidirectional = 2 if module.bidirectional else 1

        # LSTM gate equations (simplified):
        # 4 gates (input, forget, cell, output) with input and hidden state connections
        gates_per_cell = 4
        ops_per_layer = (
            batch_size
            * seq_length
            * (
                # Input to hidden connections
                gates_per_cell * input_size * hidden_size
                +
                # Hidden to hidden connections
                gates_per_cell * hidden_size * hidden_size
                +
                # Gate activations and cell updates
                gates_per_cell
                * hidden_size
                * 1.5  # Approximation for the activation functions
            )
        )

        # Account for multiple layers and bidirectional
        flops = ops_per_layer * num_layers * bidirectional

        # Convert to FLOPs if needed
        if not self.count_mac:
            flops *= 2

        return int(flops)

    def _gru_flops(
        self, module: nn.GRU, input_tensor: torch.Tensor, output: torch.Tensor
    ) -> int:
        """Calculate FLOPs for GRU layer."""
        batch_size = input_tensor.shape[0]
        seq_length = input_tensor.shape[1] if len(input_tensor.shape) > 2 else 1
        input_size = (
            input_tensor.shape[2]
            if len(input_tensor.shape) > 2
            else input_tensor.shape[1]
        )
        hidden_size = module.hidden_size
        num_layers = module.num_layers
        bidirectional = 2 if module.bidirectional else 1

        # GRU has 3 gates instead of LSTM's 4
        gates_per_cell = 3
        ops_per_layer = (
            batch_size
            * seq_length
            * (
                # Input to hidden connections
                gates_per_cell * input_size * hidden_size
                +
                # Hidden to hidden connections
                gates_per_cell * hidden_size * hidden_size
                +
                # Gate activations and cell updates
                gates_per_cell
                * hidden_size
                * 1.5  # Approximation for the activation functions
            )
        )

        # Account for multiple layers and bidirectional
        flops = ops_per_layer * num_layers * bidirectional

        # Convert to FLOPs if needed
        if not self.count_mac:
            flops *= 2

        return int(flops)

    def _bn_flops(
        self,
        module: Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d],
        input_tensor: torch.Tensor,
        output: torch.Tensor,
    ) -> int:
        """Calculate FLOPs for batch normalization."""
        numel = input_tensor.numel()

        # Each element needs normalization (subtract mean, divide by std, scale, shift)
        # In inference mode, this is just 2 operations per element
        flops = numel * 2

        # For FLOPs (not MACs), we double count
        if not self.count_mac:
            flops *= 2

        return flops

    def _activation_flops(
        self, module: nn.Module, input_tensor: torch.Tensor, output: torch.Tensor
    ) -> int:
        """Calculate FLOPs for activation functions."""
        # Most activation functions are approximately 1 operation per element
        # For MACs this should be 0, but we'll count it as 1 for consistency with other counters
        return input_tensor.numel() if not self.count_mac else 0

    def _pool_flops(
        self, module: nn.Module, input_tensor: torch.Tensor, output: torch.Tensor
    ) -> int:
        """Calculate FLOPs for pooling layers."""
        # For max pooling: 1 comparison per element in kernel
        # For avg pooling: 1 add per element in kernel + 1 divide
        # These aren't MAC operations, so for MAC counting they're 0
        if self.count_mac:
            return 0

        batch_size = input_tensor.shape[0]

        if hasattr(module, "kernel_size"):
            if isinstance(module.kernel_size, int):
                kernel_size = module.kernel_size
            else:
                kernel_size = module.kernel_size[0] * module.kernel_size[1]
        else:
            kernel_size = 1

        output_size = output.numel() // batch_size

        # FLOPs: output elements * kernel size
        flops = batch_size * output_size * kernel_size

        return flops

    def _adaptive_pool_flops(
        self, module: nn.Module, input_tensor: torch.Tensor, output: torch.Tensor
    ) -> int:
        """Calculate FLOPs for adaptive pooling layers."""
        # For MAC counting these aren't multiply-accumulate operations
        if self.count_mac:
            return 0

        # Similar to regular pooling but kernel size varies
        batch_size = input_tensor.shape[0]
        input_size = input_tensor.numel() // (batch_size * input_tensor.shape[1])
        output_size = output.numel() // (batch_size * output.shape[1])

        # Approximate adaptive pooling as input_size/output_size operations per output element
        avg_kernel_size = input_size / output_size
        flops = batch_size * output.numel() * avg_kernel_size

        return int(flops)

    def _zero_flops(
        self, module: nn.Module, input_tensor: torch.Tensor, output: torch.Tensor
    ) -> int:
        """Return zero FLOPs for layers like Dropout that don't contribute to FLOPs in inference."""
        return 0
