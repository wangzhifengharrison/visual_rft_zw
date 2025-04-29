import os
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("laolao77/ViRFT_COCO_base65")

# Get current directory path using os
current_dir = os.getcwd()
print(f"Saving dataset to: {current_dir}")

# Save to current directory
dataset.save_to_disk(current_dir)

# Optional: Verify files were created
print("Saved files:", os.listdir(current_dir))