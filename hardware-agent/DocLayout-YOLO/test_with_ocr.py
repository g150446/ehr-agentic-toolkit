import cv2
import os
import argparse
from doclayout_yolo import YOLOv10

def main():
    parser = argparse.ArgumentParser(description="Document layout analysis with OCR")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--model", default="./models/doclayout_docstructbench.pt", help="Path to the model file (default: ./models/doclayout_docstructbench.pt)")
    parser.add_argument("--output-dir", default="outputs/extracted_regions", help="Output directory for cropped regions (default: outputs/extracted_regions)")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold (default: 0.1)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for inference (default: 1024)")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR processing")
    args = parser.parse_args()

    # Load the pre-trained model
    model = YOLOv10(args.model)

    # Read the image
    image_path = args.image
    image = cv2.imread(image_path)

    # Perform prediction
    det_res = model.predict(
        image_path,
        imgsz=args.imgsz,
        conf=args.conf,
        device="cpu"
    )

    print("\n=== Detection Results ===")
    print(f"Number of detections: {len(det_res[0].boxes)}\n")

    # Create output directory for cropped regions
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract and save cropped regions
    extracted_data = []
    for i, box in enumerate(det_res[0].boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        class_name = model.names[class_id]

        # Get coordinates
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

        # Crop the region
        cropped = image[y1:y2, x1:x2]

        # Save cropped region
        crop_filename = f"{args.output_dir}/{i+1:02d}_{class_name.replace(' ', '_')}.png"
        cv2.imwrite(crop_filename, cropped)

        extracted_data.append({
            'index': i+1,
            'class': class_name,
            'confidence': confidence,
            'bbox': coords,
            'crop_path': crop_filename
        })

        print(f"{i+1}. {class_name}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Bbox: [{x1}, {y1}, {x2}, {y2}]")
        print(f"   Size: {x2-x1} x {y2-y1} pixels")
        print(f"   Saved to: {crop_filename}")

    # Save annotated result
    os.makedirs("outputs", exist_ok=True)
    annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
    output_image = f"outputs/result_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
    cv2.imwrite(output_image, annotated_frame)
    print(f"\n✓ Annotated image saved as '{output_image}'")
    print(f"✓ Cropped regions saved to '{args.output_dir}/' directory")

    # Perform OCR on the extracted regions
    if not args.no_ocr:
        print("\n=== Starting OCR (Japanese & English) ===")
        print("Loading EasyOCR reader...")

        try:
            import easyocr
            reader = easyocr.Reader(['ja', 'en'], gpu=False)
            print("✓ EasyOCR loaded\n")

            # Create output text file
            output_text = f"outputs/extracted_text_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
            with open(output_text, "w", encoding="utf-8") as f:
                for data in extracted_data:
                    print(f"Processing: {data['crop_path']}")

                    # Perform OCR
                    results = reader.readtext(data['crop_path'])

                    # Write to file
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Region {data['index']}: {data['class']} (Confidence: {data['confidence']:.2%})\n")
                    f.write(f"Bbox: {data['bbox']}\n")
                    f.write(f"{'-'*80}\n")

                    if results:
                        for detection in results:
                            text = detection[1]
                            conf = detection[2]
                            f.write(f"{text} (OCR conf: {conf:.2%})\n")
                            print(f"  - {text[:50]}..." if len(text) > 50 else f"  - {text}")
                    else:
                        f.write("(No text detected)\n")
                        print("  - (No text detected)")

                    f.write("\n")

            print(f"\n✓ OCR complete! Text saved to '{output_text}'")

        except ImportError:
            print("\n⚠ EasyOCR not installed!")
            print("Please install it with: pip install easyocr")
            print("\nAlternatively, you can use:")
            print("  - Tesseract: pip install pytesseract (requires tesseract binary)")
            print("  - PaddleOCR: pip install paddleocr")
        except Exception as e:
            print(f"\n✗ OCR error: {e}")

    print(f"\nModel classes: {model.names}")

if __name__ == "__main__":
    main()
