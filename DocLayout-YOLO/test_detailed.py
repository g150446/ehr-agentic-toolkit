import cv2
import time
import argparse
import numpy as np
from doclayout_yolo import YOLOv10

# Parse command line arguments
parser = argparse.ArgumentParser(description='DocLayout YOLO Detection Test')
parser.add_argument('--skip-viz', action='store_true', help='Skip visualization to save time')
args = parser.parse_args()

start_time = time.time()

# Load the pre-trained model
model = YOLOv10("./models/doclayout_docstructbench.pt")

# Perform prediction with adjusted parameters
det_res = model.predict(
    "./page1.png",
    imgsz=1024,        # Try larger sizes: 1280, 1536 for better detection
    conf=0.1,          # Lowered from 0.2 to detect more objects
    device="cpu"
)

# Print detection results
print("\n=== Detection Results ===")
print(f"Number of detections: {len(det_res[0].boxes)}")
print("\nDetected objects:")

# time checkpoint 1
elapsed_time = time.time() - start_time
print(f"Time since start: {elapsed_time:.2f} seconds")

for i, box in enumerate(det_res[0].boxes):
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    coords = box.xyxy[0].tolist()
    class_name = model.names[class_id]

    print(f"\n{i+1}. {class_name}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Bbox: [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")

# Save annotated result
if not args.skip_viz:
    annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
    
    # Convert PIL image to numpy array for drawing
    if hasattr(annotated_frame, 'mode'):  # PIL Image
        annotated_frame = np.array(annotated_frame)
        # Convert RGB to BGR for cv2
        if annotated_frame.shape[2] == 3:
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    
    # Make a writable copy to avoid read-only array issues
    annotated_frame = annotated_frame.copy()
    
    # Add section numbers to match the printed results
    for i, box in enumerate(det_res[0].boxes):
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        section_num = str(i + 1)
        
        # Draw section number at top-left corner of bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        color = (0, 255, 0)  # Green color for section numbers
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(section_num, font, font_scale, thickness)
        
        # Draw background rectangle for better visibility
        cv2.rectangle(annotated_frame, 
                     (x1, y1 - text_height - 10), 
                     (x1 + text_width + 10, y1), 
                     (0, 0, 0), -1)
        
        # Draw section number
        cv2.putText(annotated_frame, section_num, 
                   (x1 + 5, y1 - 5), 
                   font, font_scale, color, thickness, cv2.LINE_AA)
    
    cv2.imwrite("result_detailed.jpg", annotated_frame)
    print("\n✓ Annotated image saved as 'result_detailed.jpg'")
else:
    print("\n⊗ Visualization skipped")

# Print model info
print(f"\nModel classes: {model.names}")

elapsed_time = time.time() - start_time
print(f"Time since start: {elapsed_time:.2f} seconds")
