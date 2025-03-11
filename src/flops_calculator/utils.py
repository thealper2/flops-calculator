import argparse


def format_flops(flops: int, mac: bool = False) -> str:
    """
    Format FLOPs or MACs into a human-readable string.

    Args:
        flops (int): Number of operations
        mac (bool): If True, format as MACs instead of FLOPs

    Returns:
        str: Formatted string (e.g., '1.23 GFLOPs' or '1.23 GMac')
    """
    unit = "Mac" if mac else "FLOPs"

    if flops < 1e3:
        return f"{flops} {unit}"
    elif flops < 1e6:
        return f"{flops / 1e3:.2f} K{unit}"
    elif flops < 1e9:
        return f"{flops / 1e6:.2f} M{unit}"
    else:
        return f"{flops / 1e9:.2f} G{unit}"


def setup_argparser() -> argparse.ArgumentParser:
    """
    Main function to handle command-line arguments."

    Returns:
        argparse.ArgumentParser: Argument parser
    """
    parser = argparse.ArgumentParser(description="Calculate FLOPs for PyTorch models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to saved PyTorch model (.pth or .pt file) or torchvision model name",
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        required=True,
        help="Input shape without batch dimension (e.g., 3 224 224 for RGB images)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run calculations on",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed layer-by-layer information during calculation",
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Print detailed breakdown in summary"
    )
    parser.add_argument(
        "--mac", action="store_true", help="Count in MAC operations instead of FLOPs"
    )
    return parser
