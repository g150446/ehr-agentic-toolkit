import AppKit
import Vision

// MARK: - Helpers

func getDocumentsDirectory() -> URL {
    return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
}

func loadImage(from path: String) -> CGImage? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
          let nsImage = NSImage(data: data),
          let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return nil
    }
    return cgImage
}

func performOCRWithObservations(on cgImage: CGImage) -> [(text: String, bbox: CGRect)]? {
    let request = VNRecognizeTextRequest()
    request.recognitionLanguages = ["ja-JP", "en-US"]
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
        try handler.perform([request])
    } catch {
        print("OCR Error: \(error)")
        return nil
    }

    guard let observations = request.results else {
        print("No OCR results")
        return nil
    }

    var results: [(text: String, bbox: CGRect)] = []
    for observation in observations {
        guard let candidate = observation.topCandidates(1).first else { continue }
        results.append((text: candidate.string, bbox: observation.boundingBox))
    }
    return results
}

func saveOverlay(original: CGImage, observations: [(text: String, bbox: CGRect)], matchedBBox: CGRect?, outputPath: String) -> Bool {
    let width = original.width
    let height = original.height
    let size = NSSize(width: width, height: height)
    let image = NSImage(size: size)

    image.lockFocus()

    let nsOriginal = NSImage(cgImage: original, size: size)
    nsOriginal.draw(in: NSRect(origin: .zero, size: size))

    // VisionのboundingBoxは左下原点の正規化座標
    // NSImageの描画は左上原点なのでY座標を反転する必要がある
    let flipY: (CGRect) -> CGRect = { bbox in
        let newY = CGFloat(height) - bbox.origin.y * CGFloat(height) - bbox.height * CGFloat(height)
        return CGRect(
            x: bbox.origin.x * CGFloat(width),
            y: newY,
            width: bbox.width * CGFloat(width),
            height: bbox.height * CGFloat(height)
        )
    }

    // 全テキスト領域（黄色）
    for obs in observations {
        let rect = flipY(obs.bbox)
        let path = NSBezierPath(rect: rect)
        NSColor.yellow.withAlphaComponent(0.15).setFill()
        path.fill()
        NSColor.yellow.withAlphaComponent(0.8).setStroke()
        path.lineWidth = 1.5
        path.stroke()

        // テキストラベルも描画
        let fontSize = max(10, min(rect.height * 0.8, 14))
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: fontSize),
            .foregroundColor: NSColor.yellow
        ]
        let label = obs.text
        label.draw(at: NSPoint(x: rect.minX, y: rect.maxY + 2), withAttributes: attrs)
    }

    // マッチした領域（赤）
    if let matched = matchedBBox {
        let rect = flipY(matched)
        let path = NSBezierPath(rect: rect)
        NSColor.red.withAlphaComponent(0.35).setFill()
        path.fill()
        NSColor.red.setStroke()
        path.lineWidth = 3
        path.stroke()

        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.boldSystemFont(ofSize: 18),
            .foregroundColor: NSColor.red
        ]
        "★ サマリ ★".draw(at: NSPoint(x: rect.minX, y: rect.maxY + 4), withAttributes: attrs)
    }

    image.unlockFocus()

    guard let cgResult = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("Failed to get CGImage from overlay")
        return false
    }

    let rep = NSBitmapImageRep(cgImage: cgResult)
    guard let data = rep.representation(using: .png, properties: [:]) else {
        print("Failed to convert to PNG")
        return false
    }

    do {
        try data.write(to: URL(fileURLWithPath: outputPath))
        return true
    } catch {
        print("Failed to write overlay: \(error)")
        return false
    }
}

// MARK: - Main

let inputPath = "swift-appkit/captures/summary_left_panel_20260504_225033.png"
let outputPath = "swift-appkit/captures/test_ocr_overlay.png"

print("Loading image: \(inputPath)")
guard let image = loadImage(from: inputPath) else {
    print("Failed to load image")
    exit(1)
}
print("Image size: \(image.width) x \(image.height)")

print("\nRunning Apple Vision OCR...")
guard let observations = performOCRWithObservations(on: image) else {
    print("OCR failed")
    exit(1)
}

print("\n--- Detected \(observations.count) text regions ---")
var matchedBBox: CGRect? = nil
for (i, obs) in observations.enumerated() {
    let isMatch = obs.text.contains("サマリ")
    let marker = isMatch ? " >>> MATCH <<<" : ""
    print("[\(String(format: "%2d", i))] \"\(obs.text)\"  bbox=(x:\(String(format: "%.4f", obs.bbox.origin.x)), y:\(String(format: "%.4f", obs.bbox.origin.y)), w:\(String(format: "%.4f", obs.bbox.width)), h:\(String(format: "%.4f", obs.bbox.height)))\(marker)")
    if isMatch && matchedBBox == nil {
        matchedBBox = obs.bbox
    }
}

if let matched = matchedBBox {
    print("\n✅ Found 'サマリ' at: \(matched)")
} else {
    print("\n❌ 'サマリ' NOT FOUND in any text region")
}

print("\nSaving overlay image: \(outputPath)")
if saveOverlay(original: image, observations: observations, matchedBBox: matchedBBox, outputPath: outputPath) {
    print("✅ Overlay saved successfully")
} else {
    print("❌ Failed to save overlay")
    exit(1)
}
