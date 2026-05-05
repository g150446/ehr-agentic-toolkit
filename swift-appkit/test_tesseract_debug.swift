import AppKit

func loadImage(from path: String) -> CGImage? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
          let nsImage = NSImage(data: data),
          let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return nil
    }
    return cgImage
}

func runTesseractDebug(on cgImage: CGImage) {
    let width = cgImage.width
    let height = cgImage.height

    let tempDir = FileManager.default.temporaryDirectory
    let inputPath = tempDir.appendingPathComponent("tesseract_input_\(Int(Date().timeIntervalSince1970)).png").path
    let outputPrefix = tempDir.appendingPathComponent("tesseract_out_\(Int(Date().timeIntervalSince1970))").path

    let rep = NSBitmapImageRep(cgImage: cgImage)
    guard let data = rep.representation(using: .png, properties: [:]) else {
        print("Failed to convert image to PNG")
        return
    }
    do {
        try data.write(to: URL(fileURLWithPath: inputPath))
    } catch {
        print("Failed to write temp image: \(error)")
        return
    }

    // Run Tesseract with TSV output
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/tesseract")
    process.arguments = [inputPath, outputPrefix, "-l", "jpn", "tsv"]
    if !FileManager.default.fileExists(atPath: "/opt/homebrew/bin/tesseract") {
        process.executableURL = URL(fileURLWithPath: "/usr/local/bin/tesseract")
    }

    do {
        try process.run()
        process.waitUntilExit()
        print("Tesseract exit code: \(process.terminationStatus)")
    } catch {
        print("Failed to run tesseract: \(error)")
        try? FileManager.default.removeItem(atPath: inputPath)
        return
    }

    let tsvPath = outputPrefix + ".tsv"
    if let tsvData = try? Data(contentsOf: URL(fileURLWithPath: tsvPath)),
       let tsvString = String(data: tsvData, encoding: .utf8) {
        print("\n=== TSV OUTPUT (first 50 lines) ===")
        let lines = tsvString.components(separatedBy: .newlines)
        for line in lines.prefix(50) {
            print(line)
        }
        print("=== END TSV ===")
    } else {
        print("Failed to read TSV output")
    }

    // Also try stdout output
    let stdoutProcess = Process()
    stdoutProcess.executableURL = process.executableURL
    stdoutProcess.arguments = [inputPath, "stdout", "-l", "jpn"]
    let pipe = Pipe()
    stdoutProcess.standardOutput = pipe
    do {
        try stdoutProcess.run()
        stdoutProcess.waitUntilExit()
        if let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) {
            print("\n=== STDOUT OUTPUT ===")
            print(output.prefix(1000))
            print("=== END STDOUT ===")
        }
    } catch {
        print("Failed to run stdout tesseract: \(error)")
    }

    // Cleanup
    try? FileManager.default.removeItem(atPath: inputPath)
    try? FileManager.default.removeItem(atPath: tsvPath)
}

let inputPath = "swift-appkit/captures/left_panel.png"
print("Loading image: \(inputPath)")
guard let image = loadImage(from: inputPath) else {
    print("Failed to load image")
    exit(1)
}
print("Image size: \(image.width) x \(image.height)")

print("\nRunning Tesseract debug...")
runTesseractDebug(on: image)
