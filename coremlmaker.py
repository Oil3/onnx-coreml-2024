import argparse
from ultralytics import YOLO

def summarize_model_info(model_path):
    """
    Load a YOLO model and display a summarized table of its layers, including shapes.
    
    Args:
        model_path (str): Path to the PyTorch model file.
    """
    try:
        # Load the YOLO model
        model = YOLO(model_path)

        # Header
        header = f"{'from':<10}{'n':<5}{'params':<10}{'module':<40}{'shape':<30}"
        print("\nSummarized Model Information:\n")
        print(header)
        print("-" * len(header))

        # Use ultralytics' utility to extract detailed layer information
        layers = model.model.model  # Access model layers

        for i, layer in enumerate(layers):
            # Extract details
            from_layer = getattr(layer, 'from_', "N/A")  # Layer input source
            n = getattr(layer, 'n', "N/A")              # Number of repeats
            params = sum(p.numel() for p in layer.parameters()) if hasattr(layer, 'parameters') else "N/A"
            module = type(layer).__name__              # Module type
            shape = getattr(layer, 'shape', "N/A")     # Shape attribute (if exists)
            
            # Format shape
            shape = str(shape) if shape != "N/A" else "Unknown"

            # Print the formatted row
            print(f"{from_layer:<10}{n:<5}{params:<10}{module:<40}{shape:<30}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Summarize YOLO model layers with shapes.")
    parser.add_argument("model_path", type=str, help="Path to the PyTorch model file.")
    args = parser.parse_args()

    # Summarize the model information
    summarize_model_info(args.model_path)