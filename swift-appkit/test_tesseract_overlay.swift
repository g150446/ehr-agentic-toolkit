import AppKit

// MARK: - Helpers

func loadImage(from path: String) -> CGImage? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
          let nsImage = NSImage(data: data),
          let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return nil
    }
    return cgImage
}

struct LineBox {
    let text: String
    let x: Int      // left
    let y: Int      // top (Tesseract top-left origin)
    let w: Int      // width
    let h: Int      // height
}

func performTesseractOCR(on cgImage: CGImage) -> (boxes: [LineBox], matched: LineBox?)? {
    let tempDir = FileManager.default.temporaryDirectory
    let inputPath = tempDir.appendingPathComponent("tess_input.png").path
    let outputPrefix = tempDir.appendingPathComponent("tess_out").path

    let rep = NSBitmapImageRep(cgImage: cgImage)
    guard let data = rep.representation(using: .png, properties: [:]) else {
        print("Failed to convert image to PNG")
        return nil
    }
    do {
        try data.write(to: URL(fileURLWithPath: inputPath))
    } catch {
        print("Failed to write temp image: \(error)")
        return nil
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/tesseract")
    process.arguments = [inputPath, outputPrefix, "-l", "jpn", "tsv"]
    if !FileManager.default.fileExists(atPath: "/opt/homebrew/bin/tesseract") {
        process.executableURL = URL(fileURLWithPath: "/usr/local/bin/tesseract")
    }

    do {
        try process.run()
        process.waitUntilExit()
    } catch {
        print("Failed to run tesseract: \(error)")
        try? FileManager.default.removeItem(atPath: inputPath)
        return nil
    }

    let tsvPath = outputPrefix + ".tsv"
    guard let tsvData = try? Data(contentsOf: URL(fileURLWithPath: tsvPath)),
          let tsvString = String(data: tsvData, encoding: .utf8) else {
        print("Failed to read TSV output")
        try? FileManager.default.removeItem(atPath: inputPath)
        try? FileManager.default.removeItem(atPath: tsvPath)
        return nil
    }

    // Parse TSV words (level 5)
    struct WordEntry {
        let text: String
        let left: Int
        let top: Int
        let width: Int
        let height: Int
        let block: Int
        let par: Int
        let line: Int
    }

    var words: [WordEntry] = []
    for line in tsvString.components(separatedBy: .newlines).dropFirst() {
        let cols = line.split(separator: "\t", omittingEmptySubsequences: false).map { String($0) }
        guard cols.count >= 12 else { continue }
        guard let level = Int(cols[0]), level == 5 else { continue }
        guard let block = Int(cols[2]),
              let par = Int(cols[3]),
              let lineNum = Int(cols[4]),
              let left = Int(cols[6]),
              let top = Int(cols[7]),
              let w = Int(cols[8]),
              let h = Int(cols[9]),
              let conf = Double(cols[10]) else { continue }
        let text = cols[11]
        guard !text.trimmingCharacters(in: .whitespaces).isEmpty else { continue }
        guard conf > 30 else { continue }

        words.append(WordEntry(
            text: text, left: left, top: top, width: w, height: h,
            block: block, par: par, line: lineNum
        ))
    }

    // Group by line
    var lineGroups: [String: [WordEntry]] = [:]
    for w in words {
        let key = "\(w.block)-\(w.par)-\(w.line)"
        lineGroups[key, default: []].append(w)
    }

    var boxes: [LineBox] = []
    var matched: LineBox? = nil

    for (_, lineWords) in lineGroups {
        let sorted = lineWords.sorted { $0.left < $1.left }
        let lineText = sorted.map { $0.text }.joined(separator: " ")

        let minLeft = sorted.map { $0.left }.min() ?? 0
        let minTop = sorted.map { $0.top }.min() ?? 0
        let maxRight = sorted.map { $0.left + $0.width }.max() ?? 0
        let maxBottom = sorted.map { $0.top + $0.height }.max() ?? 0

        let normalizedText = lineText.replacingOccurrences(of: " ", with: "")
        let box = LineBox(
            text: lineText,
            x: minLeft,
            y: minTop,
            w: maxRight - minLeft,
            h: maxBottom - minTop
        )
        boxes.append(box)

        if matched == nil && normalizedText.contains("サマリ") {
            matched = box
        }
    }

    try? FileManager.default.removeItem(atPath: inputPath)
    try? FileManager.default.removeItem(atPath: tsvPath)

    return (boxes, matched)
}

func saveOverlay(original: CGImage, boxes: [LineBox], matched: LineBox?, outputPath: String) -> Bool {
    let imgW = original.width
    let imgH = original.height
    let size = NSSize(width: imgW, height: imgH)
    let image = NSImage(size: size)

    image.lockFocus()

    // Draw original image
    let nsOriginal = NSImage(cgImage: original, size: size)
    nsOriginal.draw(in: NSRect(x: 0, y: 0, width: imgW, height: imgH))

    // Tesseract uses top-left origin (y increases downward)
    // NSImage/CG uses bottom-left origin (y increases upward)
    // So we must flip Y: nsY = imgH - (tessY + tessH)

    // All detected lines (yellow)
    for box in boxes {
        let nsY = imgH - box.y - box.h
        let rect = NSRect(x: box.x, y: nsY, width: box.w, height: box.h)
        let path = NSBezierPath(rect: rect)
        NSColor.yellow.withAlphaComponent(0.15).setFill()
        path.fill()
        NSColor.yellow.withAlphaComponent(0.8).setStroke()
        path.lineWidth = 1
        path.stroke()
    }

    // Matched line (red)
    if let box = matched {
        let nsY = imgH - box.y - box.h
        let rect = NSRect(x: box.x, y: nsY, width: box.w, height: box.h)
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
        let label = "★ \(box.text.prefix(30)) ★"
        label.draw(at: NSPoint(x: CGFloat(box.x), y: CGFloat(nsY + box.h + 4)), withAttributes: attrs)
    }

    image.unlockFocus()

    guard let cgResult = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("Failed to encode PNG")
        return false
    }
    let rep = NSBitmapImageRep(cgImage: cgResult)
    guard let data = rep.representation(using: .png, properties: [:]) else {
        print("Failed to encode PNG")
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

let inputPath = "swift-appkit/captures/summary_left_panel_20260505_084741.png"
let outputPath = "swift-appkit/captures/test_tesseract_overlay_real.png"

print("Loading image: \(inputPath)")
guard let image = loadImage(from: inputPath) else {
    print("Failed to load image")
    exit(1)
}
print("Image size: \(image.width) x \(image.height)")

print("\nRunning Tesseract OCR...")
guard let result = performTesseractOCR(on: image) else {
    print("OCR failed")
    exit(1)
}

print("\n--- Detected \(result.boxes.count) text lines ---")
for (i, box) in result.boxes.enumerated() {
    let isMatch = box.text.contains("サマ リ")
    let marker = isMatch ? " >>> MATCH <<<" : ""
    print("[\(String(format: "%2d", i))] \(box.text.prefix(40))  [\(box.x), \(box.y), \(box.w), \(box.h)]\(marker)")
}

if let m = result.matched {
    print("\n✅ Matched 'サマリ' at pixel rect: [\(m.x), \(m.y), \(m.w), \(m.h)]")
} else {
    print("\n❌ 'サマリ' NOT FOUND")
}

// Dump raw TSV for debugging
let tsvPath = FileManager.default.temporaryDirectory.appendingPathComponent("tess_out.tsv").path
if let tsvData = try? Data(contentsOf: URL(fileURLWithPath: tsvPath)),
   let tsvString = String(data: tsvData, encoding: .utf8) {
    print("\n--- Raw TSV output (first 50 lines) ---")
    for (i, line) in tsvString.components(separatedBy: .newlines).enumerated() {
        if i < 50 { print(line) }
    }
}

print("\nSaving overlay: \(outputPath)")
if saveOverlay(original: image, boxes: result.boxes, matched: result.matched, outputPath: outputPath) {
    print("✅ Overlay saved")
} else {
    print("❌ Failed to save overlay")
    exit(1)
}
