import cv2
import os
import argparse
import ssl
import certifi
from doclayout_yolo import YOLOv10

def main():
    parser = argparse.ArgumentParser(description="Document layout analysis with OCR and position information")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--model", default="./models/doclayout_docstructbench.pt", help="Path to the model file (default: ./models/doclayout_docstructbench.pt)")
    parser.add_argument("--output-dir", default="outputs/extracted_regions", help="Output directory for cropped regions (default: outputs/extracted_regions)")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold (default: 0.1)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for inference (default: 1024)")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR processing")
    args = parser.parse_args()

    # Fix SSL certificate issue for EasyOCR model download
    ssl._create_default_https_context = ssl._create_unverified_context

    # Load the pre-trained model
    model = YOLOv10(args.model)

    # Read the image
    image_path = args.image
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Perform prediction
    det_res = model.predict(
        image_path,
        imgsz=args.imgsz,
        conf=args.conf,
        device="cpu"
    )

    print("\n=== Detection Results ===")
    print(f"Image size: {image_width} x {image_height} pixels")
    print(f"Number of detections: {len(det_res[0].boxes)}\n")

    # Create output directory for cropped regions
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a visualization image with numbered regions
    vis_image = image.copy()

    # Extract and save cropped regions
    extracted_data = []
    for i, box in enumerate(det_res[0].boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        class_name = model.names[class_id]

        # Get coordinates
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        width = x2 - x1
        height = y2 - y1

        # Calculate center and position
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Crop the region
        cropped = image[y1:y2, x1:x2]

        # Save cropped region
        crop_filename = f"{args.output_dir}/{i+1:02d}_{class_name.replace(' ', '_')}.png"
        cv2.imwrite(crop_filename, cropped)

        # Draw numbered rectangle on visualization image
        color = (0, 255, 0)  # Green
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

        # Draw number label with background
        label = f"#{i+1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        # Draw label background
        label_x = x1
        label_y = y1 - 10
        cv2.rectangle(vis_image,
                      (label_x, label_y - text_size[1] - 10),
                      (label_x + text_size[0] + 10, label_y),
                      color, -1)

        # Draw label text
        cv2.putText(vis_image, label, (label_x + 5, label_y - 5),
                    font, font_scale, (0, 0, 0), thickness)

        extracted_data.append({
            'index': i+1,
            'class': class_name,
            'confidence': confidence,
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'size': {'width': width, 'height': height},
            'center': {'x': center_x, 'y': center_y},
            'crop_path': crop_filename
        })

        print(f"Region #{i+1}: {class_name}")
        print(f"  Position: Top-left ({x1}, {y1}) → Bottom-right ({x2}, {y2})")
        print(f"  Center: ({center_x}, {center_y})")
        print(f"  Size: {width} x {height} pixels")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Saved to: {crop_filename}")
        print()

    # Save visualization with numbered regions
    os.makedirs("outputs", exist_ok=True)
    output_positions = f"outputs/result_with_positions_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
    cv2.imwrite(output_positions, vis_image)
    print(f"✓ Position visualization saved as '{output_positions}'\n")

    # Sort regions by position (top to bottom, left to right)
    sorted_data = sorted(extracted_data, key=lambda x: (x['center']['y'], x['center']['x']))

    print("=== Regions sorted by position (top → bottom, left → right) ===")
    for idx, data in enumerate(sorted_data, 1):
        print(f"{idx}. Region #{data['index']}: {data['class']} at ({data['center']['x']}, {data['center']['y']})")

    # Save annotated result (original style)
    output_detailed = f"outputs/result_detailed_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
    annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
    cv2.imwrite(output_detailed, annotated_frame)
    print(f"\n✓ Original annotated image saved as '{output_detailed}'")
    print(f"✓ Cropped regions saved to '{args.output_dir}/' directory")

    # Perform OCR on the extracted regions
    if not args.no_ocr:
        print("\n=== Starting OCR (Japanese & English) ===")
        print("Loading EasyOCR reader...")

        try:
            import easyocr
            reader = easyocr.Reader(['ja', 'en'], gpu=False)
            print("✓ EasyOCR loaded\n")

            # Create output text file with position information
            output_text = f"outputs/extracted_text_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
            with open(output_text, "w", encoding="utf-8") as f:
                f.write(f"Document size: {image_width} x {image_height} pixels\n")
                f.write(f"Total regions detected: {len(extracted_data)}\n")
                f.write(f"\n{'='*100}\n")

                # Process in sorted order
                for data in sorted_data:
                    print(f"Processing Region #{data['index']}: {data['class']}")

                    # Write position information
                    f.write(f"\nREGION #{data['index']}: {data['class']} (Confidence: {data['confidence']:.2%})\n")
                    f.write(f"{'-'*100}\n")
                    f.write(f"Position:\n")
                    f.write(f"  Top-left: ({data['bbox']['x1']}, {data['bbox']['y1']})\n")
                    f.write(f"  Bottom-right: ({data['bbox']['x2']}, {data['bbox']['y2']})\n")
                    f.write(f"  Center: ({data['center']['x']}, {data['center']['y']})\n")
                    f.write(f"  Size: {data['size']['width']} x {data['size']['height']} pixels\n")
                    f.write(f"\nExtracted Text:\n")

                    # Perform OCR
                    results = reader.readtext(data['crop_path'])

                    if results:
                        for detection in results:
                            text = detection[1]
                            conf = detection[2]
                            f.write(f"  {text}\n")
                            print(f"  - {text[:60]}..." if len(text) > 60 else f"  - {text}")
                    else:
                        f.write("  (No text detected)\n")
                        print("  - (No text detected)")

                    f.write(f"\n{'='*100}\n")

            print(f"\n✓ OCR complete! Text with positions saved to '{output_text}'")

        except ImportError:
            print("\n⚠ EasyOCR not installed!")
            print("Please install it with: pip install easyocr")
        except Exception as e:
            print(f"\n✗ OCR error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nModel classes: {model.names}")

if __name__ == "__main__":
    main()
