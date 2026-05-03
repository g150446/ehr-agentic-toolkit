import AppKit
import Foundation
import CoreGraphics
import ApplicationServices
import Vision

// MARK: - ChatMessage
struct ChatMessage {
    let role: String
    var content: String
}

// MARK: - ChatViewController
class ChatViewController: NSViewController {
    private var scrollView: NSScrollView!
    private var textView: NSTextView!
    private var inputView: NSTextView!
    private var sendButton: NSButton!
    private var debugButton: NSButton!
    private var modelSelector: NSPopUpButton!
    private var messages: [ChatMessage] = []
    private var isStreaming = false
    private var debugMode = false

    private let apiBase = "http://localhost:8000/v1"
    private let apiKey = "penguin"
    private var currentModel = "gemma-4-26b-a4b-it-4bit"

    override func loadView() {
        view = NSView()
        view.autoresizesSubviews = true
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        fetchModels()
    }

    override func viewWillLayout() {
        super.viewWillLayout()
        layoutSubviews()
    }

    private func setupUI() {
        let topBar = NSView(frame: NSRect(x: 0, y: 0, width: 1, height: 40))
        topBar.autoresizingMask = [.width, .minYMargin]

        let modelLabel = NSTextField(labelWithString: "Model:")
        modelLabel.frame = NSRect(x: 10, y: 10, width: 50, height: 20)
        topBar.addSubview(modelLabel)

        modelSelector = NSPopUpButton(frame: NSRect(x: 60, y: 7, width: 300, height: 25))
        modelSelector.autoresizingMask = [.width]
        modelSelector.target = self
        modelSelector.action = #selector(modelChanged(_:))
        topBar.addSubview(modelSelector)

        view.addSubview(topBar)

        scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.borderType = .noBorder
        scrollView.autoresizingMask = [.width, .height]

        textView = NSTextView()
        textView.isEditable = false
        textView.isRichText = true
        textView.font = NSFont.systemFont(ofSize: 14)
        textView.backgroundColor = NSColor.textBackgroundColor
        textView.textContainer?.containerSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        textView.textContainer?.widthTracksTextView = true

        scrollView.documentView = textView

        let inputBar = NSView(frame: NSRect(x: 0, y: 0, width: 1, height: 100))
        inputBar.autoresizingMask = [.width, .maxYMargin]

        let inputScrollView = NSScrollView()
        inputScrollView.hasVerticalScroller = false
        inputScrollView.borderType = .bezelBorder
        inputScrollView.autoresizingMask = [.width]

        inputView = NSTextView()
        inputView.isEditable = true
        inputView.isRichText = false
        inputView.font = NSFont.systemFont(ofSize: 14)
        inputView.backgroundColor = NSColor.textBackgroundColor
        inputView.drawsBackground = true
        inputView.isAutomaticQuoteSubstitutionEnabled = false
        inputView.isAutomaticDashSubstitutionEnabled = false
        inputView.isAutomaticTextReplacementEnabled = false
        inputView.delegate = self
        inputView.textContainer?.containerSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        inputView.textContainer?.widthTracksTextView = true

        inputScrollView.documentView = inputView

        let buttonWidth: CGFloat = 70
        inputScrollView.frame = NSRect(x: 10, y: 10, width: 1, height: 80)
        inputBar.addSubview(inputScrollView)

        debugButton = NSButton(frame: NSRect(x: 0, y: 60, width: buttonWidth, height: 30))
        debugButton.title = "Debug"
        debugButton.bezelStyle = .rounded
        debugButton.autoresizingMask = [.minXMargin]
        debugButton.target = self
        debugButton.action = #selector(debugAction)
        debugButton.isHidden = true
        inputBar.addSubview(debugButton)

        sendButton = NSButton(frame: NSRect(x: 0, y: 10, width: buttonWidth, height: 30))
        sendButton.title = "Send"
        sendButton.bezelStyle = .rounded
        sendButton.autoresizingMask = [.minXMargin]
        sendButton.target = self
        sendButton.action = #selector(sendMessage)
        sendButton.keyEquivalent = "\r"
        inputBar.addSubview(sendButton)

        view.addSubview(scrollView)
        view.addSubview(inputBar)

        layoutSubviews()
    }

    private func layoutSubviews() {
        let viewFrame = view.frame
        let topBar = view.subviews[0]
        let inputBar = view.subviews[2]

        let inputBarHeight: CGFloat = 100
        topBar.frame = NSRect(x: 0, y: viewFrame.height - 40, width: viewFrame.width, height: 40)
        scrollView.frame = NSRect(x: 0, y: inputBarHeight, width: viewFrame.width, height: viewFrame.height - inputBarHeight - 40)
        inputBar.frame = NSRect(x: 0, y: 0, width: viewFrame.width, height: inputBarHeight)

        let buttonWidth: CGFloat = 70
        let inputScrollView = inputBar.subviews[0] as? NSScrollView
        inputScrollView?.frame = NSRect(x: 10, y: 10, width: viewFrame.width - buttonWidth - 20, height: 80)

        debugButton.frame = NSRect(x: viewFrame.width - buttonWidth - 10, y: 60, width: buttonWidth, height: 30)
        sendButton.frame = NSRect(x: viewFrame.width - buttonWidth - 10, y: 10, width: buttonWidth, height: 30)

        modelSelector.frame = NSRect(x: 60, y: 7, width: viewFrame.width - 70, height: 25)
    }

    func setDebugMode(_ enabled: Bool) {
        debugMode = enabled
        debugButton.isHidden = !enabled
        view.needsLayout = true
    }

    @objc private func modelChanged(_ sender: NSPopUpButton) {
        if let selected = sender.selectedItem?.title {
            currentModel = selected
        }
    }

    @objc private func debugAction() {
        debugButton.isEnabled = false
        debugButton.title = "Reading..."

        Task {
            do {
                try await runEHRReader()
            } catch {
                await MainActor.run {
                    appendMessage(role: "assistant", content: "Error: \(error.localizedDescription)")
                }
            }
            await MainActor.run {
                debugButton.isEnabled = true
                debugButton.title = "Debug"
            }
        }
    }

    private func authHeader() -> String {
        return "Bearer \(apiKey)"
    }

    private func fetchModels() {
        guard let url = URL(string: "\(apiBase)/models") else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        request.setValue(authHeader(), forHTTPHeaderField: "Authorization")

        URLSession.shared.dataTask(with: request) { [weak self] data, _, error in
            guard let data = data, error == nil,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let models = json["data"] as? [[String: Any]] else {
                DispatchQueue.main.async {
                    self?.modelSelector.addItem(withTitle: self?.currentModel ?? "gemma-4-26b-a4b-it-4bit")
                }
                return
            }

            let modelIds = models.compactMap { $0["id"] as? String }
            DispatchQueue.main.async {
                self?.modelSelector.removeAllItems()
                if modelIds.isEmpty {
                    self?.modelSelector.addItem(withTitle: self?.currentModel ?? "gemma-4-26b-a4b-it-4bit")
                } else {
                    for id in modelIds {
                        self?.modelSelector.addItem(withTitle: id)
                    }
                    let preferredModel = "gemma-4-26b-a4b-it-4bit"
                    if modelIds.contains(preferredModel) {
                        self?.currentModel = preferredModel
                    } else {
                        self?.currentModel = modelIds.first ?? self!.currentModel
                    }
                    self?.modelSelector.selectItem(withTitle: self?.currentModel ?? "")
                }
            }
        }.resume()
    }

    @objc private func sendMessage() {
        let text = inputView.string.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isStreaming else { return }

        messages.append(ChatMessage(role: "user", content: text))
        inputView.string = ""
        appendMessage(role: "user", content: text)

        isStreaming = true
        sendButton.isEnabled = false
        inputView.isEditable = false

        appendMessage(role: "assistant", content: "")

        streamChat()
    }

    private func appendMessage(role: String, content: String) {
        let attributed = NSMutableAttributedString()
        let color: NSColor = role == "user" ? NSColor.controlAccentColor : NSColor.labelColor
        let boldFont = NSFont.boldSystemFont(ofSize: 14)
        let normalFont = NSFont.systemFont(ofSize: 14)

        let roleAttr: [NSAttributedString.Key: Any] = [
            .font: boldFont,
            .foregroundColor: color
        ]
        let contentAttr: [NSAttributedString.Key: Any] = [
            .font: normalFont,
            .foregroundColor: NSColor.labelColor
        ]

        attributed.append(NSAttributedString(string: "\(role == "user" ? "You" : "AI"): ", attributes: roleAttr))
        attributed.append(NSAttributedString(string: content + "\n\n", attributes: contentAttr))

        textView.textStorage?.append(attributed)
        textView.scrollToEndOfDocument(nil)
    }

    private func updateLastMessage(content: String) {
        guard let storage = textView.textStorage else { return }
        let fullText = storage.string

        if let range = fullText.range(of: "AI: ", options: .backwards) {
            let startIdx = fullText.distance(from: fullText.startIndex, to: range.upperBound)
            let length = storage.length - startIdx
            if length > 0 {
                storage.replaceCharacters(in: NSMakeRange(startIdx, length), with: content + "\n\n")
            }
        }
        textView.scrollToEndOfDocument(nil)
    }

    private func streamChat() {
        guard let url = URL(string: "\(apiBase)/chat/completions") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(authHeader(), forHTTPHeaderField: "Authorization")

        let apiMessages: [[String: String]] = messages.map { ["role": $0.role, "content": $0.content] }
        let body: [String: Any] = [
            "model": currentModel,
            "messages": apiMessages,
            "stream": true
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        var assistantContent = ""
        let semaphore = DispatchSemaphore(value: 0)

        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            defer { semaphore.signal() }

            if let error = error {
                DispatchQueue.main.async {
                    self?.updateLastMessage(content: "Error: \(error.localizedDescription)")
                }
                return
            }

            guard let data = data,
                  let text = String(data: data, encoding: .utf8) else {
                DispatchQueue.main.async {
                    self?.updateLastMessage(content: "Error: No response")
                }
                return
            }

            for line in text.components(separatedBy: "\n") {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard trimmed.hasPrefix("data: ") else { continue }
                let jsonStr = String(trimmed.dropFirst(6))
                guard jsonStr != "[DONE]" else { break }

                if let jsonData = jsonStr.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                   let choices = json["choices"] as? [[String: Any]],
                   let first = choices.first,
                   let delta = first["delta"] as? [String: Any],
                   let content = delta["content"] as? String {
                    assistantContent += content
                    DispatchQueue.main.async {
                        self?.updateLastMessage(content: assistantContent)
                    }
                }
            }

            DispatchQueue.main.async {
                self?.messages.append(ChatMessage(role: "assistant", content: assistantContent))
                self?.isStreaming = false
                self?.sendButton.isEnabled = true
                self?.inputView.isEditable = true
                self?.inputView.becomeFirstResponder()
            }
        }
        task.resume()
    }

    // MARK: - EHR Reader (Scroll + VLM)

    private func runEHRReader() async throws {
        let logger = EHRLogger()
        logger.log("===== EHR Reader Started =====")

        await MainActor.run {
            appendMessage(role: "assistant", content: "診療録読み取りを開始します...")
        }

        // Check screen capture permission at start
        if !CGPreflightScreenCaptureAccess() {
            logger.log("Screen capture permission not granted. Requesting...")
            await MainActor.run {
                appendMessage(role: "assistant", content: "画面収録の権限が必要です。システム設定で許可してください。")
            }

            // Trigger permission dialog by attempting a dummy capture
            let tempFile = FileManager.default.temporaryDirectory.appendingPathComponent("dummy_screenshot_\(Int(Date().timeIntervalSince1970)).png")
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
            process.arguments = ["-x", tempFile.path]
            try? process.run()
            process.waitUntilExit()

            // Wait for user response
            var attempts = 0
            let maxAttempts = 300 // 30 seconds
            while !CGPreflightScreenCaptureAccess() && attempts < maxAttempts {
                try await Task.sleep(nanoseconds: 100_000_000)
                attempts += 1
            }

            if !CGPreflightScreenCaptureAccess() {
                logger.log("ERROR: Screen capture permission denied.")
                await MainActor.run {
                    appendMessage(role: "assistant", content: "画面収録の権限が拒否されました。処理を中止します。")
                }
                throw NSError(domain: "EHRReader", code: 18, userInfo: [NSLocalizedDescriptionKey: "画面収録の権限が拒否されました"])
            }

            logger.log("Screen capture permission granted.")
            try? FileManager.default.removeItem(at: tempFile)
        } else {
            logger.log("Screen capture permission already granted.")
        }

        let mainDisplay = CGMainDisplayID()
        let bounds = CGDisplayBounds(mainDisplay)
        let centerPoint = CGPoint(x: bounds.width / 2, y: bounds.height / 2)
        logger.log("Screen bounds: \(bounds)")
        logger.log("Center point: \(centerPoint)")

        CGDisplayMoveCursorToPoint(mainDisplay, centerPoint)
        try await Task.sleep(nanoseconds: 1_000_000_000)
        logger.log("Moved cursor to center")

        guard let clickDown = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: centerPoint, mouseButton: .left),
              let clickUp = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: centerPoint, mouseButton: .left) else {
            logger.log("ERROR: Failed to create click events")
            throw NSError(domain: "EHRReader", code: 10, userInfo: [NSLocalizedDescriptionKey: "Failed to create click events"])
        }
        clickDown.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 50_000_000)
        clickUp.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 500_000_000)
        logger.log("Posted click at center")

        var windowID: Int = 0
        if let windowInfo = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] {
            for info in windowInfo {
                if let layer = info[kCGWindowLayer as String] as? Int, layer == 0,
                   let ownerName = info[kCGWindowOwnerName as String] as? String,
                   ownerName != "Window Server",
                   ownerName != "Dock",
                   let winNum = info[kCGWindowNumber as String] as? Int {
                    windowID = winNum
                    logger.log("Found active window: \(ownerName) (ID: \(winNum))")
                    break
                }
            }
        }

        guard windowID != 0 else {
            logger.log("ERROR: No active window found")
            throw NSError(domain: "EHRReader", code: 11, userInfo: [NSLocalizedDescriptionKey: "No active window found"])
        }

        logger.log("Capturing initial screenshot...")
        guard let frame = captureActiveWindow(windowID: windowID) else {
            logger.log("ERROR: Failed to capture initial screenshot")
            throw NSError(domain: "EHRReader", code: 12, userInfo: [NSLocalizedDescriptionKey: "Failed to capture initial screenshot"])
        }
        logger.log("Initial screenshot captured: \(frame.width)x\(frame.height)")

        logger.log("Switching back to AI chat window...")
        postCommandTab()
        try await Task.sleep(nanoseconds: 500_000_000)

        logger.log("Detecting vertical divider...")
        guard let dividerX = detectVerticalDivider(image: frame) else {
            logger.log("ERROR: Vertical divider detection failed")
            throw NSError(domain: "EHRReader", code: 13, userInfo: [NSLocalizedDescriptionKey: "縦線を検出できませんでした"])
        }
        logger.log("Vertical divider detected at x=\(dividerX)")

        logger.log("Detecting horizontal divider...")
        guard let topY = detectHorizontalDivider(image: frame) else {
            logger.log("ERROR: Horizontal divider detection failed")
            throw NSError(domain: "EHRReader", code: 14, userInfo: [NSLocalizedDescriptionKey: "横線を検出できませんでした"])
        }
        logger.log("Horizontal divider detected at y=\(topY)")

        let fullPath = saveDebugImage(frame, name: "ehr_reader_full_initial")
        logger.log("Saved full screenshot: \(fullPath ?? "FAILED")")
        if let overlay = createOverlayImage(original: frame, dividerX: dividerX, topY: topY) {
            let overlayPath = saveDebugImage(overlay, name: "ehr_reader_overlay_initial")
            logger.log("Saved overlay image: \(overlayPath ?? "FAILED")")
        }

        logger.log("Extracting past chart region: dividerX=\(dividerX), topY=\(topY)")
        guard let cropped = extractPastChartRegion(image: frame, dividerX: dividerX, topY: topY) else {
            logger.log("ERROR: extractPastChartRegion returned nil")
            throw NSError(domain: "EHRReader", code: 15, userInfo: [NSLocalizedDescriptionKey: "過去診療録領域の切り出しに失敗しました"])
        }
        let cropPath = saveDebugImage(cropped, name: "ehr_reader_crop_initial")
        logger.log("Saved cropped region: \(cropPath ?? "FAILED") (size: \(cropped.width)x\(cropped.height))")

        logger.log("Performing OCR...")
        let ocrText = performOCR(on: cropped)
        logger.log("OCR result length: \(ocrText.count) characters")
        logger.log("OCR text preview: \(String(ocrText.prefix(200)).replacingOccurrences(of: "\n", with: " "))")

        guard let croppedData = cgImageToPNG(cropped) else {
            logger.log("ERROR: cgImageToPNG failed")
            throw NSError(domain: "EHRReader", code: 16, userInfo: [NSLocalizedDescriptionKey: "画像のPNG変換に失敗しました"])
        }
        logger.log("PNG conversion OK: \(croppedData.count) bytes")

        logger.log("Calling VLM (initial read)...")
        let rawResponse = try await callVLM(imageDataList: [croppedData], ocrText: ocrText, currentJSON: nil)
        logger.log("VLM raw response length: \(rawResponse.count) characters")
        logger.log("VLM raw response preview: \(String(rawResponse.prefix(500)).replacingOccurrences(of: "\n", with: " "))")

        guard var structured = parseVLMResponse(rawResponse) else {
            logger.log("ERROR: parseVLMResponse failed for initial read")
            throw NSError(domain: "EHRReader", code: 17, userInfo: [NSLocalizedDescriptionKey: "VLM応答のJSON解析に失敗しました"])
        }
        logger.log("Initial parse OK: \(structured.count) records")

        await MainActor.run {
            appendMessage(role: "assistant", content: "初回読み取り完了: \(structured.count) 件の診療録を検出")
        }

        let maxIterations = 20
        var prevFrame = frame
        var prevCropped = cropped
        var unchangedCount = 0

        for iteration in 1...maxIterations {
            logger.log("\n--- Scroll Set \(iteration) ---")
            try await Task.sleep(nanoseconds: 300_000_000)

            logger.log("Switching to EHR window for scroll...")
            postCommandTab()
            try await Task.sleep(nanoseconds: 500_000_000)

            let scrollAmount = Int32(bounds.height / 3)
            logger.log("Scrolling down by \(scrollAmount)px (screen height / 3)...")
            postScrollEvent(at: centerPoint, amount: -scrollAmount)
            try await Task.sleep(nanoseconds: 1_000_000_000)
            logger.log("Waited 1.0s after scroll")

            logger.log("Capturing screenshot after scroll...")
            guard let newFrame = captureActiveWindow(windowID: windowID) else {
                logger.log("WARNING: Screenshot capture failed after scroll")
                await MainActor.run {
                    appendMessage(role: "assistant", content: "[WARNING] スクリーンショット撮影に失敗しました。現在の結果で終了します。")
                }
                break
            }
            logger.log("Screenshot captured: \(newFrame.width)x\(newFrame.height)")

            let changeRatio = frameDiffRatio(prev: prevFrame, curr: newFrame)
            let changePercent = changeRatio * 100
            let isUnchanged = isFrameUnchanged(prev: prevFrame, curr: newFrame)
            let statusText = isUnchanged ? "変化なし (\(unchangedCount + 1)/2)" : "変化あり"
            let logLine = "[Set \(iteration)] 画面変化率: \(String(format: "%.4f", changePercent))% → \(statusText)"
            logger.log(logLine)
            await MainActor.run {
                appendMessage(role: "assistant", content: logLine)
            }

            if isUnchanged {
                unchangedCount += 1
                if unchangedCount >= 2 {
                    logger.log("Frame unchanged 2 times in a row. Stopping.")
                    await MainActor.run {
                        appendMessage(role: "assistant", content: "スクロール後の画面が変化しませんでした。自動終了します。")
                    }
                    break
                }
            } else {
                unchangedCount = 0
            }
            prevFrame = newFrame

            let newFullPath = saveDebugImage(newFrame, name: "ehr_reader_full_scroll_\(iteration)")
            logger.log("Saved full screenshot: \(newFullPath ?? "FAILED")")

            logger.log("Detecting vertical divider (scroll \(iteration))...")
            guard let newDividerX = detectVerticalDivider(image: newFrame) else {
                logger.log("ERROR: Vertical divider detection failed after scroll \(iteration)")
                await MainActor.run {
                    appendMessage(role: "assistant", content: "[ERROR] スクロール後の画面で縦線を検出できませんでした")
                }
                break
            }
            logger.log("Vertical divider at x=\(newDividerX)")

            logger.log("Detecting horizontal divider (scroll \(iteration))...")
            guard let newTopY = detectHorizontalDivider(image: newFrame) else {
                logger.log("ERROR: Horizontal divider detection failed after scroll \(iteration)")
                await MainActor.run {
                    appendMessage(role: "assistant", content: "[ERROR] スクロール後の画面で横線を検出できませんでした")
                }
                break
            }
            logger.log("Horizontal divider at y=\(newTopY)")

            if let overlay = createOverlayImage(original: newFrame, dividerX: newDividerX, topY: newTopY) {
                let overlayPath = saveDebugImage(overlay, name: "ehr_reader_overlay_scroll_\(iteration)")
                logger.log("Saved overlay: \(overlayPath ?? "FAILED")")
            }

            logger.log("Switching back to AI chat window...")
            postCommandTab()
            try await Task.sleep(nanoseconds: 500_000_000)

            logger.log("Extracting past chart region: dividerX=\(newDividerX), topY=\(newTopY)")
            guard let newCropped = extractPastChartRegion(image: newFrame, dividerX: newDividerX, topY: newTopY) else {
                logger.log("ERROR: extractPastChartRegion returned nil for scroll \(iteration)")
                await MainActor.run {
                    appendMessage(role: "assistant", content: "[ERROR] 過去診療録領域の切り出しに失敗しました")
                }
                break
            }
            let newCropPath = saveDebugImage(newCropped, name: "ehr_reader_crop_scroll_\(iteration)")
            logger.log("Saved cropped region: \(newCropPath ?? "FAILED") (size: \(newCropped.width)x\(newCropped.height))")

            logger.log("Performing OCR (scroll \(iteration))...")
            let newOcrText = performOCR(on: newCropped)
            logger.log("OCR result length: \(newOcrText.count) characters")

            guard let newCroppedData = cgImageToPNG(newCropped) else {
                logger.log("ERROR: cgImageToPNG failed for scroll \(iteration)")
                await MainActor.run {
                    appendMessage(role: "assistant", content: "[ERROR] 画像のPNG変換に失敗しました")
                }
                break
            }

            // Prepare previous cropped bottom 50% for overlap comparison
            let overlapHeight = prevCropped.height / 2
            let overlapRect = CGRect(x: 0, y: 0, width: prevCropped.width, height: overlapHeight)
            let prevBottomCrop = prevCropped.cropping(to: overlapRect)
            var imageDataList: [Data] = [newCroppedData]
            if let prevBottomCrop = prevBottomCrop,
               let prevData = cgImageToPNG(prevBottomCrop) {
                imageDataList.insert(prevData, at: 0)
                logger.log("Added previous cropped bottom half: \(prevBottomCrop.width)x\(prevBottomCrop.height)")
            }

            let currentJSONData = try? JSONSerialization.data(withJSONObject: structured, options: [.prettyPrinted, .sortedKeys])
            let currentJSONStr = currentJSONData.flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
            logger.log("Current JSON records: \(structured.count)")

            logger.log("Calling VLM (scroll \(iteration)) with \(imageDataList.count) images...")
            let mergeResponse = try await callVLM(imageDataList: imageDataList, ocrText: newOcrText, currentJSON: currentJSONStr)
            logger.log("VLM merge response length: \(mergeResponse.count) characters")

            guard let merged = parseVLMResponse(mergeResponse) else {
                logger.log("ERROR: parseVLMResponse failed for merge (scroll \(iteration))")
                await MainActor.run {
                    appendMessage(role: "assistant", content: "[ERROR] VLMマージ応答のJSON解析に失敗しました")
                }
                break
            }
            logger.log("Merge parse OK: \(merged.count) records")
            structured = merged
            
            prevCropped = newCropped

            await MainActor.run {
                appendMessage(role: "assistant", content: "セット \(iteration) 完了: 合計 \(structured.count) 件")
            }
        }

        logger.log("\n===== Processing Complete =====")
        logger.log("Total records: \(structured.count)")

        guard let finalJSONData = try? JSONSerialization.data(withJSONObject: structured, options: [.prettyPrinted, .sortedKeys]),
              let finalJSONStr = String(data: finalJSONData, encoding: .utf8) else {
            logger.log("ERROR: Final JSON serialization failed")
            await MainActor.run {
                appendMessage(role: "assistant", content: "[ERROR] 最終JSONのシリアライズに失敗しました")
            }
            return
        }

        logger.log("Final JSON:\n\(finalJSONStr)")
        logger.saveToFile()
        logger.log("Log saved to: \(logger.logFilePath)")

        await MainActor.run {
            appendMessage(role: "assistant", content: "過去診療録のスクロール読み取りが完了しました:\n```json\n\(finalJSONStr)\n```")
        }
    }

    private func postCommandTab() {
        let cmdDown = CGEvent(keyboardEventSource: nil, virtualKey: 0x37, keyDown: true)
        let tabDown = CGEvent(keyboardEventSource: nil, virtualKey: 0x30, keyDown: true)
        let tabUp = CGEvent(keyboardEventSource: nil, virtualKey: 0x30, keyDown: false)
        let cmdUp = CGEvent(keyboardEventSource: nil, virtualKey: 0x37, keyDown: false)

        cmdDown?.flags = .maskCommand
        tabDown?.flags = .maskCommand
        tabUp?.flags = .maskCommand

        cmdDown?.post(tap: .cghidEventTap)
        tabDown?.post(tap: .cghidEventTap)
        tabUp?.post(tap: .cghidEventTap)
        cmdUp?.post(tap: .cghidEventTap)
    }

    private func captureActiveWindow(windowID: Int) -> CGImage? {
        let capturesDir = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("captures")
        try? FileManager.default.createDirectory(at: capturesDir, withIntermediateDirectories: true)

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = formatter.string(from: Date())
        let filename = "ehr_capture_\(timestamp).png"
        let fileURL = capturesDir.appendingPathComponent(filename)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        process.arguments = ["-x", "-l", "\(windowID)", fileURL.path]
        try? process.run()
        process.waitUntilExit()

        guard let data = try? Data(contentsOf: fileURL),
              let image = NSImage(data: data),
              let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }
        return cgImage
    }

    private func postScrollEvent(at point: CGPoint, amount: Int32) {
        guard let scrollEvent = CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 1, wheel1: amount, wheel2: 0, wheel3: 0) else { return }
        scrollEvent.location = point
        scrollEvent.post(tap: .cghidEventTap)
    }

    private func getPixelData(from cgImage: CGImage) -> (pixels: [UInt8], width: Int, height: Int) {
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixels = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return (pixels, width, height)
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return (pixels, width, height)
    }

    private func detectVerticalDivider(image: CGImage) -> Int? {
        let (pixels, width, height) = getPixelData(from: image)
        let yStart = height * 5 / 100
        let yEnd = height * 95 / 100
        let bandHeight = yEnd - yStart

        var candidateRanges: [(start: Int, end: Int)] = []
        var currentStart: Int? = nil

        for x in 0..<width {
            var grayCount = 0
            for y in yStart..<yEnd {
                let idx = (y * width + x) * 4
                let r = Int(pixels[idx])
                let g = Int(pixels[idx + 1])
                let b = Int(pixels[idx + 2])
                let maxVal = max(r, g, b)
                let minVal = min(r, g, b)
                let value = (r + g + b) / 3
                if maxVal - minVal <= 14 && value >= 120 && value <= 235 {
                    grayCount += 1
                }
            }

            let ratio = Double(grayCount) / Double(bandHeight)
            if ratio >= 0.5 {
                if currentStart == nil {
                    currentStart = x
                }
            } else {
                if let start = currentStart {
                    candidateRanges.append((start: start, end: x - 1))
                    currentStart = nil
                }
            }
        }
        if let start = currentStart {
            candidateRanges.append((start: start, end: width - 1))
        }

        guard let firstRange = candidateRanges.first else { return nil }
        return (firstRange.start + firstRange.end) / 2
    }

    private func detectHorizontalDivider(image: CGImage) -> Int? {
        let (pixels, width, height) = getPixelData(from: image)
        let halfWidth = width / 2
        let minHalfLineWidth = Int(Double(halfWidth) * 0.80) // 半分の幅の80%以上
        let maxSearchY = Int(Double(height) * 0.25) // 上部25%のみ検索

        var lineYPositions: [Int] = []
        var currentGroupStart: Int? = nil
        var currentGroupEnd: Int? = nil

        for y in 0..<maxSearchY {
            // 左半分の最大連続グレー長を計算
            var leftMaxConsecutive = 0
            var leftConsecutive = 0
            for x in 0..<halfWidth {
                let idx = (y * width + x) * 4
                let r = Int(pixels[idx])
                let g = Int(pixels[idx + 1])
                let b = Int(pixels[idx + 2])
                let maxVal = max(r, g, b)
                let minVal = min(r, g, b)
                let value = (r + g + b) / 3

                if maxVal - minVal <= 14 && value >= 120 && value <= 235 {
                    leftConsecutive += 1
                } else {
                    leftMaxConsecutive = max(leftMaxConsecutive, leftConsecutive)
                    leftConsecutive = 0
                }
            }
            leftMaxConsecutive = max(leftMaxConsecutive, leftConsecutive)

            // 右半分の最大連続グレー長を計算
            var rightMaxConsecutive = 0
            var rightConsecutive = 0
            for x in halfWidth..<width {
                let idx = (y * width + x) * 4
                let r = Int(pixels[idx])
                let g = Int(pixels[idx + 1])
                let b = Int(pixels[idx + 2])
                let maxVal = max(r, g, b)
                let minVal = min(r, g, b)
                let value = (r + g + b) / 3

                if maxVal - minVal <= 14 && value >= 120 && value <= 235 {
                    rightConsecutive += 1
                } else {
                    rightMaxConsecutive = max(rightMaxConsecutive, rightConsecutive)
                    rightConsecutive = 0
                }
            }
            rightMaxConsecutive = max(rightMaxConsecutive, rightConsecutive)

            // 左右それぞれで80%以上の連続グレーがある行を線とみなす
            let isGrayLine = leftMaxConsecutive >= minHalfLineWidth && rightMaxConsecutive >= minHalfLineWidth

            if isGrayLine {
                if currentGroupStart == nil {
                    currentGroupStart = y
                }
                currentGroupEnd = y
            } else {
                // グループ終了 → 中心点を記録
                if let start = currentGroupStart, let end = currentGroupEnd {
                    let centerY = (start + end) / 2
                    lineYPositions.append(centerY)
                }
                currentGroupStart = nil
                currentGroupEnd = nil
            }
        }

        // 最後のグループを処理
        if let start = currentGroupStart, let end = currentGroupEnd {
            let centerY = (start + end) / 2
            lineYPositions.append(centerY)
        }

        // 最も下にある線（最後）を返す
        return lineYPositions.last
    }

    private func detectPatientInfoBarHeight(image: CGImage, topY: Int) -> Int {
        // 上部15%をスキャンして、患者情報バーの高さを検出
        // 青いボタン行 + 患者情報行の合計高さを返す
        let scanHeight = Int(CGFloat(image.height) * 0.15)
        let (pixels, width, height) = getPixelData(from: image)
        
        var blueButtonDetected = false
        var lastBlueRow = 0
        var lastGrayRow = 0
        
        for y in 0..<min(scanHeight, height) {
            var blueCount = 0
            var grayCount = 0
            
            for x in 0..<width {
                let idx = (y * width + x) * 4
                let r = Int(pixels[idx])
                let g = Int(pixels[idx + 1])
                let b = Int(pixels[idx + 2])
                
                // 青いボタン検出（R<100, G<100, B>150）
                if r < 100 && g < 100 && b > 150 {
                    blueCount += 1
                }
                
                // 薄いグレー背景検出（患者情報行）
                let value = (r + g + b) / 3
                if value > 240 && value < 250 && abs(r - g) < 5 && abs(g - b) < 5 {
                    grayCount += 1
                }
            }
            
            let blueRatio = Double(blueCount) / Double(width)
            let grayRatio = Double(grayCount) / Double(width)
            
            if blueRatio > 0.1 {
                blueButtonDetected = true
                lastBlueRow = y
            }
            if grayRatio > 0.3 && blueButtonDetected {
                lastGrayRow = y
            }
        }
        
        // 患者情報バーの高さを返す（青いボタン行 + 患者情報行）
        let barHeight = max(lastBlueRow, lastGrayRow) + 1
        return barHeight > 0 ? barHeight : 0
    }

    private func extractPastChartRegion(image: CGImage, dividerX: Int, topY: Int) -> CGImage? {
        // 患者情報バーの高さを検出して除外
        let patientInfoHeight = detectPatientInfoBarHeight(image: image, topY: topY)
        let adjustedTopY = topY + patientInfoHeight
        
        let width = image.width - dividerX
        let height = image.height - adjustedTopY
        guard width > 0, height > 0 else { return nil }
        // Core Graphics coordinate system: origin is bottom-left
        let cropRect = CGRect(x: dividerX, y: adjustedTopY, width: width, height: height)
        return image.cropping(to: cropRect)
    }

    private func frameDiffRatio(prev: CGImage?, curr: CGImage?) -> Double {
        guard let prev = prev, let curr = curr else { return 1.0 }
        guard prev.width == curr.width, prev.height == curr.height else { return 1.0 }

        let (prevPixels, width, height) = getPixelData(from: prev)
        let (currPixels, _, _) = getPixelData(from: curr)

        let totalPixels = width * height
        var diffCount = 0

        for i in 0..<totalPixels {
            let idx = i * 4
            let pr = Int(prevPixels[idx])
            let pg = Int(prevPixels[idx + 1])
            let pb = Int(prevPixels[idx + 2])
            let cr = Int(currPixels[idx])
            let cg = Int(currPixels[idx + 1])
            let cb = Int(currPixels[idx + 2])

            let prevGray = (pr + pg + pb) / 3
            let currGray = (cr + cg + cb) / 3

            if abs(prevGray - currGray) > 15 {
                diffCount += 1
            }
        }

        return Double(diffCount) / Double(totalPixels)
    }

    private func isFrameUnchanged(prev: CGImage?, curr: CGImage?) -> Bool {
        let ratio = frameDiffRatio(prev: prev, curr: curr)
        return ratio < 0.02
    }

    private func performOCR(on cgImage: CGImage) -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLanguages = ["ja-JP", "en-US"]
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return ""
        }

        guard let observations = request.results else {
            return ""
        }

        var lines: [(y: CGFloat, text: String)] = []
        for observation in observations {
            guard let candidate = observation.topCandidates(1).first else { continue }
            let confidence = candidate.confidence
            if confidence < 0.3 { continue }
            let text = candidate.string
            let bbox = observation.boundingBox
            let y = 1.0 - bbox.origin.y
            lines.append((y: y, text: text))
        }

        lines.sort { $0.y < $1.y }
        return lines.map { $0.text }.joined(separator: "\n")
    }

    private func logVLMRequest(prompt: String, ocrText: String, response: String) {
        let logsDir = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: logsDir, withIntermediateDirectories: true)
        let logFile = logsDir.appendingPathComponent("vlm_requests.log")

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let timestamp = formatter.string(from: Date())

        let entry = """
        === VLM Request @ \(timestamp) ===
        [Prompt]:
        \(prompt)
        [OCR Text]:
        \(ocrText)
        [Response]:
        \(response)
        === End ===

        """

        if FileManager.default.fileExists(atPath: logFile.path) {
            if let data = try? Data(contentsOf: logFile),
               var existing = String(data: data, encoding: .utf8) {
                existing.append(entry)
                try? existing.write(toFile: logFile.path, atomically: true, encoding: .utf8)
            }
        } else {
            try? entry.write(toFile: logFile.path, atomically: true, encoding: .utf8)
        }
    }

    private func callVLM(imageDataList: [Data], ocrText: String, currentJSON: String?) async throws -> String {
        let prompt: String
        if let currentJSON = currentJSON {
            prompt = """
            ### 指示
            添付された2枚の画像を比較し、以下の【現在のJSONデータ】と統合して、最新の診療録データを作成してください。

            ### 画像の説明
            - 「画像1」: スクロール前の画面の**最下部50%**です。この部分に含まれる日付ヘッダーが、直前の画面の最終日付です。
            - 「画像2」: スクロール後の画面**全体**です。

            ### 日付の継続性に関する最重要指示
            1. **画像2の最上部に日付ヘッダーがない場合**:
               - 画像2のテキストは、画像1に含まれる日付（【現在のJSONデータ】の最後の `date`）の**続き**です
               - **絶対に新しい日付として追加しないでください**
               - 必ず【現在のJSONデータ】の最後の `date` の `content` に追記してください
            2. **画像2の最上部に日付ヘッダーがある場合**:
               - 日付ヘッダーの**上側**のテキストは、画像1の日付の続きです
               - 日付ヘッダーの**下側**のテキストから、新しい日付として扱ってください
            3. **判断基準**:
               - 画像2内に「YYYY年MM月DD日(曜日)」形式の日付ヘッダーが**最上部にない**場合、そのテキストは前の日付の続きです
               - 新しい日付として扱うのは、画像2の最上部に日付ヘッダーが存在し、その下側のテキストのみです

            ### EasyOCR認識結果（参考）
            ```
            \(ocrText)
            ```

            ### 現在のJSONデータ
            \(currentJSON)

            ### 統合のルール
            1. **既存データの保護（絶対に改変しない）**:
               - 【現在のJSONデータ】の各 `content` に含まれるテキストは、絶対に改変・要約・再構成・削除しないでください。
            2. **新しい情報の追加**:
               - 【現在のJSONデータ】に存在しない日付の診療録が見つかった場合は、新しい要素として追加してください。
               - 既存の日付に追加の内容がある場合は、`content` に追記してください。
               - **画像2の最上部に日付ヘッダーがない場合**、そのテキストは【現在のJSONデータ】の最終日付の続きとして追記してください。
            3. **日付の統合**:
               - 同一日付が複数回出現する場合は、1つの要素に統合し、`content` を結合してください。
            4. **出力フォーマット**:
               - 以下の構造を維持したJSON形式のみを出力してください。
            [
              {
                "date": "YYYY年MM月DD日(曜日)",
                "content": "統合された本文テキスト"
              }
            ]

            ### 出力
            統合が完了した最新のJSONデータのみを出力してください。
            """
        } else {
            prompt = """
            ### 指示
            添付された画像は電子カルテシステムの「過去カルテ」領域のスクリーンショットです。
            画像の内容を読み取り、日付ごとに整理された診療録データ（JSON形式）を作成してください。

            ### EasyOCR認識結果（参考）
            ```
            \(ocrText)
            ```

            ### 処理のガイドライン
            1. **情報の抽出（すべての医療情報を漏らさず）**:
               - 画像とOCR結果の両方を参考にし、「日付」とそれに対応する診療録の本文を「すべて」抽出してください。
            2. **出力フォーマット**:
               - 必ず以下の構造のJSON形式のみを出力してください。
            [
              {
                "date": "YYYY年MM月DD日(曜日)",
                "content": "抽出された本文テキスト（改行を含む）"
              }
            ]

            ### 出力
            抽出・統合が完了したJSONデータのみを出力してください。
            """
        }

        var content: [[String: Any]] = [["type": "text", "text": prompt]]
        for imageData in imageDataList {
            let base64 = imageData.base64EncodedString()
            let dataUrl = "data:image/png;base64,\(base64)"
            content.append(["type": "image_url", "image_url": ["url": dataUrl]])
        }

        let body: [String: Any] = [
            "model": currentModel,
            "temperature": 0,
            "messages": [
                ["role": "user", "content": content]
            ],
            "stream": false,
            "max_tokens": 4096
        ]

        guard let url = URL(string: "\(apiBase)/chat/completions") else {
            throw NSError(domain: "EHRReader", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(authHeader(), forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 120

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw NSError(domain: "EHRReader", code: statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP Error \(statusCode)"])
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let first = choices.first,
              let message = first["message"] as? [String: Any],
              let contentStr = message["content"] as? String else {
            throw NSError(domain: "EHRReader", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])
        }

        logVLMRequest(prompt: prompt, ocrText: ocrText, response: contentStr)

        return contentStr
    }

    private func parseVLMResponse(_ raw: String) -> [[String: Any]]? {
        var cleaned = raw

        if let startRange = cleaned.range(of: "```json") {
            cleaned.removeSubrange(startRange)
        }
        if let startRange = cleaned.range(of: "```") {
            cleaned.removeSubrange(startRange)
        }

        if let regex = try? NSRegularExpression(pattern: "<think>.*?</think>", options: .dotMatchesLineSeparators) {
            let range = NSRange(cleaned.startIndex..., in: cleaned)
            cleaned = regex.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: "")
        }

        cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let start = cleaned.firstIndex(of: "["),
              let end = cleaned.lastIndex(of: "]") else {
            return nil
        }

        let jsonStr = String(cleaned[start...end])
        guard let data = jsonStr.data(using: .utf8),
              let result = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return nil
        }

        return result
    }

    private func cgImageToPNG(_ cgImage: CGImage) -> Data? {
        let rep = NSBitmapImageRep(cgImage: cgImage)
        return rep.representation(using: .png, properties: [:])
    }

    private func saveDebugImage(_ cgImage: CGImage, name: String) -> String? {
        let capturesDir = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("captures")
        try? FileManager.default.createDirectory(at: capturesDir, withIntermediateDirectories: true)

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = formatter.string(from: Date())
        let filename = "\(name)_\(timestamp).png"
        let fileURL = capturesDir.appendingPathComponent(filename)

        let rep = NSBitmapImageRep(cgImage: cgImage)
        guard let data = rep.representation(using: .png, properties: [:]) else { return nil }
        do {
            try data.write(to: fileURL)
            return fileURL.path
        } catch {
            return nil
        }
    }

    private func createOverlayImage(original: CGImage, dividerX: Int?, topY: Int?) -> CGImage? {
        let width = original.width
        let height = original.height
        let size = NSSize(width: width, height: height)
        let image = NSImage(size: size)

        image.lockFocus()

        let nsOriginal = NSImage(cgImage: original, size: size)
        nsOriginal.draw(in: NSRect(origin: .zero, size: size))

        if let dx = dividerX {
            let path = NSBezierPath()
            path.move(to: NSPoint(x: dx, y: 0))
            path.line(to: NSPoint(x: dx, y: height))
            NSColor.green.setStroke()
            path.lineWidth = 2
            path.stroke()
        }

        if let ty = topY {
            let path = NSBezierPath()
            let drawY = height - ty
            path.move(to: NSPoint(x: 0, y: drawY))
            path.line(to: NSPoint(x: width, y: drawY))
            NSColor.red.setStroke()
            path.lineWidth = 2
            path.stroke()
        }

        image.unlockFocus()

        return image.cgImage(forProposedRect: nil, context: nil, hints: nil)
    }
}

// MARK: - NSTextViewDelegate
extension ChatViewController: NSTextViewDelegate {
    func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.insertNewline(_:)) && NSEvent.modifierFlags.contains(.command) {
            sendMessage()
            return true
        }
        return false
    }
}

// MARK: - EHRLogger
class EHRLogger {
    private var logs: [String] = []
    private let formatter = DateFormatter()
    let logFilePath: String

    init() {
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        let logsDir = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: logsDir, withIntermediateDirectories: true)
        let timestamp = ISO8601DateFormatter().string(from: Date())
        logFilePath = logsDir.appendingPathComponent("ehr_reader_\(timestamp).log").path
    }

    func log(_ message: String) {
        let line = "[\(formatter.string(from: Date()))] \(message)"
        logs.append(line)
        print(line)
    }

    func saveToFile() {
        let content = logs.joined(separator: "\n")
        try? content.write(toFile: logFilePath, atomically: true, encoding: .utf8)
    }
}

// MARK: - AppDelegate
class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var chatVC: ChatViewController!
    var debugMenuItem: NSMenuItem!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let options: NSDictionary = [kAXTrustedCheckOptionPrompt.takeRetainedValue() as NSString: true]
        if !AXIsProcessTrustedWithOptions(options) {
            print("[Debug] Accessibility permission not granted, prompting user...")
        }

        setupMenuBar()

        let screen = NSScreen.main!
        let screenFrame = screen.frame
        let width = screenFrame.width / 3
        let height = screenFrame.height / 2
        let x = screenFrame.maxX - width
        let y = screenFrame.minY

        window = NSWindow(
            contentRect: NSRect(x: x, y: y, width: width, height: height),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "EHR-Agent"
        window.minSize = NSSize(width: 300, height: 200)

        chatVC = ChatViewController()
        window.contentViewController = chatVC

        window.setFrame(NSRect(x: x, y: y, width: width, height: height), display: true)
        window.makeKeyAndOrderFront(nil)
    }

    func setupMenuBar() {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        let appMenu = NSMenu()

        debugMenuItem = NSMenuItem(
            title: "Debug Mode",
            action: #selector(toggleDebugMode(_:)),
            keyEquivalent: "d"
        )
        debugMenuItem.keyEquivalentModifierMask = .command
        debugMenuItem.target = self
        appMenu.addItem(debugMenuItem)

        let quitMenuItem = NSMenuItem(
            title: "Quit EHR-Agent",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        appMenu.addItem(quitMenuItem)

        appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)

        NSApp.mainMenu = mainMenu
    }

    @objc private func toggleDebugMode(_ sender: NSMenuItem) {
        debugMenuItem.state = debugMenuItem.state == .on ? .off : .on
        chatVC?.setDebugMode(debugMenuItem.state == .on)
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
