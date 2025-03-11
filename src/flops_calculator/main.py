import sys
import torch
import torch.nn as nn

from src.flops_calculator.flops_calculator import FLOPsCalculator
from src.flops_calculator.utils import setup_argparser


def main():
    """Main function to run the FLOPs calculator from command line."""
    parser = setup_argparser()
    args = parser.parse_args()

    try:
        # Check if CUDA is available if requested
        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            args.device = "cpu"

        # Check if the model is a torchvision model
        if (
            "/" not in args.model
            and "\\" not in args.model
            and not args.model.endswith((".pth", ".pt"))
        ):
            try:
                import torchvision.models as models

                if not hasattr(models, args.model):
                    print(f"Error: {args.model} is not a valid torchvision model name.")
                    print(
                        "Available models:",
                        ", ".join(
                            name
                            for name in dir(models)
                            if not name.startswith("_") and name[0].islower()
                        ),
                    )
                    sys.exit(1)

                # Get the model from torchvision
                model_fn = getattr(models, args.model)
                model = model_fn(pretrained=False)
                print(f"Loaded {args.model} from torchvision models.")
            except ImportError:
                print(
                    "Could not import torchvision. Please provide a path to a saved model file."
                )
                sys.exit(1)
        else:
            # Load model from file
            try:
                model = torch.load(args.model, map_location=args.device)

                # Handle cases where the saved file contains more than just the model
                if not isinstance(model, nn.Module):
                    if hasattr(model, "model"):
                        model = model.model
                    elif isinstance(model, dict) and "model" in model:
                        model = model["model"]
                    elif isinstance(model, dict) and "state_dict" in model:
                        # We need a model architecture to load the state dict into
                        print(
                            "Error: Loaded file contains only a state dictionary, not a model architecture."
                        )
                        print("Please provide a model architecture file instead.")
                        sys.exit(1)
                    else:
                        print(
                            "Error: Could not extract a PyTorch model from the provided file."
                        )
                        sys.exit(1)
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                sys.exit(1)

        # Create input shape tuple
        input_shape = tuple(args.input_shape)

        # Initialize calculator
        calculator = FLOPsCalculator(
            model=model,
            input_shape=input_shape,
            device=args.device,
            verbose=args.verbose,
            count_mac=args.mac,
        )

        # Print summary
        calculator.print_summary(detailed=args.detailed)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
