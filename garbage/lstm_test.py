import torch
import numpy as np
import cv2
from torchvision import transforms
from garbage.convlstm import ConvLSTM

# Paths
model_path = r"checkpoint.pth"
image_path = r"/predict/00131.jpg"
output_path = r"/output_image.jpg"

# Model parameters
input_dim = 3  # Update if the model expects a different number of input channels
hidden_dim = [64, 32]  # Adjust to match your ConvLSTM hidden dimensions
kernel_size = [(3, 3), (3, 3)]  # Adjust to match your ConvLSTM kernel sizes
num_layers = 2  # Number of ConvLSTM layers
batch_first = True
bias = True
return_all_layers = False

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTM(input_dim=input_dim,
                 hidden_dim=hidden_dim,
                 kernel_size=kernel_size,
                 num_layers=num_layers,
                 batch_first=batch_first,
                 bias=bias,
                 return_all_layers=return_all_layers)

# Load the checkpoint and remap keys
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint["model_state_dict"]  # Extract the state_dict
new_state_dict = {}
for key, value in state_dict.items():
    if "convlstm1" in key or "convlstm2" in key or "convlstm3" in key or "convlstm4" in key:
        new_key = key.split(".", 1)[-1]  # Remove prefixes like 'convlstm1.'
        if "conv.weight" in new_key and value.shape != model.state_dict()[new_key].shape:
            print(f"Reshaping {new_key} from {value.shape} to {model.state_dict()[new_key].shape}")
            value = value[:, :model.state_dict()[new_key].shape[1], :, :]  # Fix input channel mismatch
        new_state_dict[new_key] = value
model.load_state_dict(new_state_dict, strict=False)  # Allow partial loading
model = model.to(device)
model.eval()

# Load and preprocess the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
image_resized = cv2.resize(image, (128, 128))  # Resize to match input dimensions
image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0)  # Add batch dimension
image_tensor = image_tensor.unsqueeze(1).to(device)  # Add time dimension (B, T, C, H, W)

# Perform inference
with torch.no_grad():
    layer_output_list, _ = model(image_tensor)

# Extract predictions
output = layer_output_list[-1].squeeze(1)  # Get the last layer's output (B, C, H, W)

# Check output type and handle accordingly
if output.shape[1] > 1:  # Multi-class prediction (e.g., segmentation)
    predicted_classes = torch.argmax(output, dim=1)  # Pixel-wise class predictions
    predicted_image = predicted_classes[0].cpu().numpy()  # Convert to numpy
    predicted_image = (predicted_image * 255).astype(np.uint8)  # Scale for visualization
    cv2.imwrite(output_path, predicted_image)
    print(f"Segmentation result saved to {output_path}")

elif output.shape[1] == 1:  # Single-class prediction (e.g., heatmap or binary mask)
    heatmap = output[0, 0, :, :].cpu().numpy()  # Extract the first channel
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize
    heatmap = (heatmap * 255).astype(np.uint8)  # Scale to 0-255
    cv2.imwrite(output_path, heatmap)
    print(f"Heatmap result saved to {output_path}")

else:  # Classification or flattened tensor output
    predicted_values = output.mean(axis=(2, 3))  # Reduce spatial dimensions
    predicted_class = torch.argmax(predicted_values, dim=1).item()
    print(f"Predicted class: {predicted_class}")
