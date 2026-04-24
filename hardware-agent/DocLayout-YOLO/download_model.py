from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
import os

print("Downloading DocLayout-YOLO-DocStructBench model from HuggingFace...")
print("This model is capable of handling various document types.\n")

# Download the model file from HuggingFace
try:
    filepath = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt"
    )
    print(f"✓ Model downloaded to: {filepath}\n")

    # Copy to local models directory
    import shutil
    local_path = "./models/doclayout_docstructbench.pt"
    shutil.copy(filepath, local_path)
    print(f"✓ Model copied to: {local_path}\n")

    # Load and check the model
    model = YOLOv10(local_path)
    print(f"✓ Model loaded successfully!")
    print(f"\nModel classes ({len(model.names)}): {model.names}")

except Exception as e:
    print(f"Error: {e}")
