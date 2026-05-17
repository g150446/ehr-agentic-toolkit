import AppKit
import Foundation
import CoreGraphics
import ApplicationServices
import Vision
import ScreenCaptureKit

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

    private var apiBase = "http://localhost:8000/v1"
    private var apiKey = "penguin"
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
                _ = try await runEHRReader()
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
        // ollama は固定モデルを使用
        if apiBase.contains(":11434") {
            DispatchQueue.main.async { [weak self] in
                self?.modelSelector.removeAllItems()
                self?.modelSelector.addItem(withTitle: "gemma4:26b")
                self?.currentModel = "gemma4:26b"
            }
            return
        }

        // omlx は /v1/models から取得
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

    func switchServer(to type: String) {
        if type == "ollama" {
            apiBase = "http://127.0.0.1:11434/v1"
            apiKey = "ollama"
            currentModel = "gemma4:26b"
        } else {
            apiBase = "http://localhost:8000/v1"
            apiKey = "penguin"
            currentModel = "gemma-4-26b-a4b-it-4bit"
        }
        modelSelector.removeAllItems()
        fetchModels()
    }

    @objc private func sendMessage() {
        let text = inputView.string.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isStreaming else { return }

        messages.append(ChatMessage(role: "user", content: text))
        inputView.string = ""
        appendMessage(role: "user", content: text)

        // Handle click(x,y) command
        if let clickMatch = text.range(of: "^click\\((\\d+),\\s*(\\d+)\\)$", options: .regularExpression) {
            let matchedText = String(text[clickMatch])
            // Extract x and y using NSRegularExpression for reliability
            let regex = try! NSRegularExpression(pattern: "click\\((\\d+),\\s*(\\d+)\\)")
            let nsRange = NSRange(matchedText.startIndex..., in: matchedText)
            if let result = regex.firstMatch(in: matchedText, range: nsRange),
               let xRange = Range(result.range(at: 1), in: matchedText),
               let yRange = Range(result.range(at: 2), in: matchedText),
               let x = Int(matchedText[xRange]),
               let y = Int(matchedText[yRange]) {
                isStreaming = true
                sendButton.isEnabled = false
                inputView.isEditable = false
                appendMessage(role: "assistant", content: "")
                Task {
                    await performClickAt(x: x, y: y)
                    await MainActor.run {
                        isStreaming = false
                        sendButton.isEnabled = true
                        inputView.isEditable = true
                        if inputView.window != nil {
                            inputView.becomeFirstResponder()
                        }
                    }
                }
                return
            }
        }

        if text.contains("診療情報提供書作成") {
            isStreaming = true
            sendButton.isEnabled = false
            inputView.isEditable = false

            appendMessage(role: "assistant", content: "")

            Task {
                do {
                    try await runCreateReferralLetterAction()
                } catch {
                    await MainActor.run {
                        updateLastMessage(content: "Error: \(error.localizedDescription)")
                    }
                }
                await MainActor.run {
                    isStreaming = false
                    sendButton.isEnabled = true
                    inputView.isEditable = true
                    if inputView.window != nil {
                        inputView.becomeFirstResponder()
                    }
                }
            }
            return
        }

        if text.contains("letter") || text.contains("情報提供書") {
            isStreaming = true
            sendButton.isEnabled = false
            inputView.isEditable = false

            appendMessage(role: "assistant", content: "")

            Task {
                do {
                    try await runLetterAction()
                } catch {
                    await MainActor.run {
                        updateLastMessage(content: "Error: \(error.localizedDescription)")
                    }
                }
                await MainActor.run {
                    isStreaming = false
                    sendButton.isEnabled = true
                    inputView.isEditable = true
                    if inputView.window != nil {
                        inputView.becomeFirstResponder()
                    }
                }
            }
            return
        }

        if text.contains("サマリ作成") {
            isStreaming = true
            sendButton.isEnabled = false
            inputView.isEditable = false

            appendMessage(role: "assistant", content: "")

            Task {
                do {
                    try await runCreateSummaryAction()
                } catch {
                    await MainActor.run {
                        updateLastMessage(content: "Error: \(error.localizedDescription)")
                    }
                }
                await MainActor.run {
                    isStreaming = false
                    sendButton.isEnabled = true
                    inputView.isEditable = true
                    if inputView.window != nil {
                        inputView.becomeFirstResponder()
                    }
                }
            }
            return
        }

        if text.contains("サマリ") {
            isStreaming = true
            sendButton.isEnabled = false
            inputView.isEditable = false

            appendMessage(role: "assistant", content: "")

            Task {
                do {
                    try await runSummaryAction(content: "テスト")
                } catch {
                    await MainActor.run {
                        updateLastMessage(content: "Error: \(error.localizedDescription)")
                    }
                }
                await MainActor.run {
                    isStreaming = false
                    sendButton.isEnabled = true
                    inputView.isEditable = true
                    if inputView.window != nil {
                        inputView.becomeFirstResponder()   
                    }
                }
            }
            return
        }

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

            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.messages.append(ChatMessage(role: "assistant", content: assistantContent))
                self.isStreaming = false
                self.sendButton.isEnabled = true
                self.inputView.isEditable = true
                // Safety check: only becomeFirstResponder if the view is in a window
                if self.inputView.window != nil {
                    self.inputView.becomeFirstResponder()
                }
            }
        }
        task.resume()
    }

    // MARK: - Template Matching (OpenCV)

    private func checkAndOpenPastRecordsIfNeeded(
        captureImage: CGImage,
        captureBounds: CGRect,
        logger: EHRLogger
    ) async {
        logger.log("checkAndOpenPastRecordsIfNeeded: start")
        logger.saveToFile()

        guard let templatePath = Bundle.main.path(
            forResource: "past_records_button_glay",
            ofType: "png",
            inDirectory: "match_templates"
        ) else {
            logger.log("gray template not found in bundle")
            logger.saveToFile()
            return
        }
        guard let nsImg = NSImage(contentsOfFile: templatePath),
              let template = nsImg.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            logger.log("gray template load failed")
            logger.saveToFile()
            return
        }
        logger.log("template loaded: \(template.width)x\(template.height)px")
        logger.saveToFile()

        let searchRegion = CGRect(x: 0, y: 0,
                                  width: CGFloat(captureImage.width),
                                  height: CGFloat(captureImage.height) / 2)
        logger.log("matchTemplate: source=\(captureImage.width)x\(captureImage.height) region=\(Int(searchRegion.width))x\(Int(searchRegion.height))")
        logger.saveToFile()

        let t0 = Date()
        guard let match = TemplateMatchingWrapper.matchSource(
            captureImage,
            template: template,
            searchRegion: searchRegion,
            threshold: 0.7
        ) else {
            logger.log("gray not matched (elapsed \(String(format: "%.2f", -t0.timeIntervalSinceNow))s) → already open or not found")
            logger.saveToFile()
            return
        }
        logger.log("gray matched at (\(Int(match.position.x)), \(Int(match.position.y))) score=\(String(format: "%.3f", match.score)) elapsed=\(String(format: "%.2f", -t0.timeIntervalSinceNow))s")
        logger.saveToFile()

        let scaleX = CGFloat(captureImage.width)  / captureBounds.width
        let scaleY = CGFloat(captureImage.height) / captureBounds.height
        let clickX = captureBounds.origin.x + (match.position.x + CGFloat(template.width)  / 2) / scaleX
        let clickY = captureBounds.origin.y + (match.position.y + CGFloat(template.height) / 2) / scaleY
        logger.log("activating Chrome window...")
        logger.saveToFile()
        activateChrome()
        try? await Task.sleep(nanoseconds: 500_000_000)

        logger.log("clicking at screen (\(Int(clickX)), \(Int(clickY)))")
        logger.saveToFile()

        await performClickAt(x: Int(clickX), y: Int(clickY))
        try? await Task.sleep(nanoseconds: 3_000_000_000)
        logger.log("waited 3s after clicking past_records button")
        logger.saveToFile()
    }

    // MARK: - EHR Reader (Scroll + VLM)

    private func runEHRReader() async throws -> String {
        let logger = EHRLogger()
        defer { logger.saveToFile() }
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

        logger.log("Finding Chrome window via ScreenCaptureKit...")
        await MainActor.run {
            appendMessage(role: "assistant", content: "Chromeウィンドウを検索中...")
        }
        guard let chromeResult = await findChromeWindowSCK() else {
            logger.log("ERROR: Chrome window not found")
            await MainActor.run {
                appendMessage(role: "assistant", content: "Google Chromeのウィンドウが見つかりませんでした。Chromeを開いて電子カルテを表示してください。")
            }
            throw NSError(domain: "EHRReader", code: 11, userInfo: [NSLocalizedDescriptionKey: "Chrome window not found"])
        }
        let scWindow = chromeResult.scWindow
        logger.log("Found Chrome window: ID=\(scWindow.windowID), bounds=\(chromeResult.bounds)")

        // Scroll to oldest records first (past_records button opens latest record by default)
        logger.log("\n=== Scrolling to oldest records first ===")
        logger.log("Switching to EHR window for initial upward scroll...")
        activateChrome()
        try await Task.sleep(nanoseconds: 200_000_000)

        let upwardScrollAmount: Int32 = 10
        logger.log("Scrolling up by \(upwardScrollAmount) lines for 20 times...")
        for i in 1...20 {
            postScrollEvent(at: centerPoint, amount: upwardScrollAmount)
            logger.log("Upward scroll \(i)/20 completed")
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        logger.log("Initial upward scroll completed - should be at oldest records now")
        try await Task.sleep(nanoseconds: 500_000_000)

        logger.log("Switching back to AI chat window after upward scroll...")
        activateSelf()
        try await Task.sleep(nanoseconds: 200_000_000)

        // Capture initial screenshot at oldest records position
        logger.log("Capturing initial screenshot at oldest records position via SCK (AI window in front)...")
        guard let captureResult = await captureWindowViaSCK(scWindow) else {
            logger.log("ERROR: Failed to capture initial screenshot")
            throw NSError(domain: "EHRReader", code: 12, userInfo: [NSLocalizedDescriptionKey: "Failed to capture initial screenshot"])
        }
        let frame = captureResult.image
        logger.log("Initial screenshot captured: \(frame.width)x\(frame.height)")

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

        logger.log("Calling VLM (initial read at oldest records)...")
        activateSelf()
        await MainActor.run {
            appendMessage(role: "assistant", content: "VLMへ送信中（1回目）... 画面を操作しないでください")
        }
        let rawResponse = try await callVLM(imageDataList: [croppedData], ocrText: ocrText, currentJSON: nil)
        logger.log("VLM raw response length: \(rawResponse.count) characters")
        logger.log("VLM raw response preview: \(String(rawResponse.prefix(500)).replacingOccurrences(of: "\n", with: " "))")

        guard var structured = parseVLMResponse(rawResponse) else {
            logger.log("ERROR: parseVLMResponse failed for initial read")
            throw NSError(domain: "EHRReader", code: 17, userInfo: [NSLocalizedDescriptionKey: "VLM応答のJSON解析に失敗しました"])
        }
        logger.log("Initial parse OK: \(structured.count) records")

        let maxIterations = 20
        var prevFrame = frame
        var prevCropped = cropped
        var unchangedCount = 0

        for iteration in 1...maxIterations {
            logger.log("\n--- Scroll Set \(iteration) ---")
            try await Task.sleep(nanoseconds: 100_000_000)

            logger.log("Switching to EHR window for scroll...")
            activateChrome()
            try await Task.sleep(nanoseconds: 200_000_000)

            let scrollAmount: Int32 = -10
            logger.log("Scrolling down by \(scrollAmount) lines...")
            postScrollEvent(at: centerPoint, amount: scrollAmount)
            try await Task.sleep(nanoseconds: 500_000_000)
            logger.log("Waited 0.5s after scroll")

            logger.log("Switching back to AI chat window after scroll...")
            activateSelf()
            try await Task.sleep(nanoseconds: 200_000_000)

            logger.log("Capturing screenshot after scroll via SCK (AI window in front)...")
            guard let newCaptureResult = await captureWindowViaSCK(scWindow) else {
                logger.log("WARNING: Screenshot capture failed after scroll")
                await MainActor.run {
                    appendMessage(role: "assistant", content: "[WARNING] スクリーンショット撮影に失敗しました。現在の結果で終了します。")
                }
                break
            }
            let newFrame = newCaptureResult.image
            logger.log("Screenshot captured: \(newFrame.width)x\(newFrame.height)")

            let changeRatio = frameDiffRatio(prev: prevFrame, curr: newFrame)
            let changePercent = changeRatio * 100
            let isUnchanged = isFrameUnchanged(prev: prevFrame, curr: newFrame)
            let statusText = isUnchanged ? "変化なし (\(unchangedCount + 1)/2)" : "変化あり"
            let logLine = "[Set \(iteration)] 画面変化率: \(String(format: "%.4f", changePercent))% → \(statusText)"
            logger.log(logLine)

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
            await MainActor.run {
                appendMessage(role: "assistant", content: "VLMへ送信中（\(iteration + 1)回目）... 画面を操作しないでください")
            }
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
            throw NSError(domain: "EHRReader", code: 19,
                userInfo: [NSLocalizedDescriptionKey: "最終JSONのシリアライズに失敗しました"])
        }

        logger.log("Final JSON:\n\(finalJSONStr)")
        logger.log("Log saved to: \(logger.logFilePath)")

        await MainActor.run {
            appendMessage(role: "assistant", content: "過去診療録のスクロール読み取りが完了しました:\n```json\n\(finalJSONStr)\n```")
        }

        return finalJSONStr
    }

    // MARK: - Manual Click Action

    private func performClickAt(x: Int, y: Int) async {
        let logger = EHRLogger()
        defer { logger.saveToFile() }
        logger.log("===== Manual Click Action Started =====")
        logger.log("Clicking at screen point coords: (\(x)pt, \(y)pt)")

        let point = CGPoint(x: x, y: y)
        let mainDisplay = CGMainDisplayID()

        CGDisplayMoveCursorToPoint(mainDisplay, point)
        try? await Task.sleep(nanoseconds: 500_000_000)

        guard let clickDown = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: point, mouseButton: .left),
              let clickUp = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: point, mouseButton: .left) else {
            logger.log("ERROR: Failed to create click events")
            await MainActor.run {
                updateLastMessage(content: "クリックイベントの作成に失敗しました")
            }
            return
        }
        clickDown.post(tap: .cghidEventTap)
        try? await Task.sleep(nanoseconds: 50_000_000)
        clickUp.post(tap: .cghidEventTap)
        try? await Task.sleep(nanoseconds: 500_000_000)
        logger.log("Click posted at (\(x), \(y))")

        // Type "テスト" via paste
        logger.log("Typing 'テスト' via clipboard paste...")
        await pasteText("テスト")
        logger.log("'テスト' typed and clipboard restored")

        await MainActor.run {
            updateLastMessage(content: "(\(x), \(y)) をクリックし、『テスト』を入力しました")
        }
    }

    // MARK: - Summary Action

    private func runSummaryAction(content: String) async throws {
        let logger = EHRLogger()
        defer { logger.saveToFile() }
        logger.log("===== Summary Action Started =====")

        await MainActor.run {
            updateLastMessage(content: "サマリボタンを検索・クリックします...")
        }

        // Switch to EHR app
        activateChrome()
        try await Task.sleep(nanoseconds: 500_000_000)

        let mainDisplay = CGMainDisplayID()
        let displayBounds = CGDisplayBounds(mainDisplay)
        let centerPoint = CGPoint(x: displayBounds.width / 2, y: displayBounds.height / 2)

        logger.log("Clicking center of screen to activate window: \(centerPoint)")

        CGDisplayMoveCursorToPoint(mainDisplay, centerPoint)
        try await Task.sleep(nanoseconds: 500_000_000)

        guard let clickDown = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: centerPoint, mouseButton: .left),
              let clickUp = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: centerPoint, mouseButton: .left) else {
            logger.log("ERROR: Failed to create center click events")
            throw NSError(domain: "SummaryAction", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create click events"])
        }
        clickDown.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 50_000_000)
        clickUp.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 1_000_000_000)
        logger.log("Center click posted")

        // Find active window ID and bounds
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
            throw NSError(domain: "SummaryAction", code: 2, userInfo: [NSLocalizedDescriptionKey: "No active window found"])
        }

        logger.log("Capturing active window (first screenshot)...")
        guard let firstCapture = captureActiveWindow(windowID: windowID) else {
            logger.log("ERROR: Failed to capture active window")
            throw NSError(domain: "SummaryAction", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to capture active window"])
        }
        let fullImage = firstCapture.image
        let windowBounds = firstCapture.bounds
        logger.log("Active window captured: \(fullImage.width)x\(fullImage.height), bounds: \(windowBounds)")

        // Switch back to AI chat window for progress display
        activateSelf()
        
        // Detect exact vertical divider to isolate side panel (VLM first, fallback to pixel)
        let gaugeTask1 = showProgressGauge { [self] msg in
            updateLastMessage(content: msg)
        }

        let panelWidth: Int
        if let dividerX = await detectVerticalDividerWithVLM(image: fullImage, logger: logger) {
            panelWidth = dividerX
            logger.log("VLM detected vertical divider at x=\(dividerX), cropping side panel")
        } else if let dividerX = detectVerticalDivider(image: fullImage) {
            panelWidth = dividerX
            logger.log("Pixel-based vertical divider detected at x=\(dividerX), cropping side panel")
        } else {
            panelWidth = fullImage.width / 2
            logger.log("Vertical divider not detected, falling back to half width")
        }
        gaugeTask1.cancel()
        let panelRect = CGRect(x: 0, y: 0, width: panelWidth, height: fullImage.height)
        guard let panelImage = fullImage.cropping(to: panelRect) else {
            logger.log("ERROR: Failed to crop left panel")
            throw NSError(domain: "SummaryAction", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to crop left panel"])
        }
        logger.log("Cropped left panel: \(panelImage.width)x\(panelImage.height)")

        let fullPanelPath = saveDebugImage(fullImage, name: "summary_fullscreen")
        let panelPath = saveDebugImage(panelImage, name: "summary_left_panel")
        logger.log("Saved debug images: full=\(fullPanelPath ?? "FAILED"), panel=\(panelPath ?? "FAILED")")

        logger.log("Searching for 'サマリ' button via Tesseract OCR...")
        let (matchedBoxOpt, allBoxes) = findTextLineBoxWithTesseract(on: panelImage, searchText: "サマリ", logger: logger)

        let overlayPath = saveTesseractOCROverlayImage(original: panelImage, boxes: allBoxes, matched: matchedBoxOpt, name: "summary_ocr_overlay")
        logger.log("Saved OCR overlay: \(overlayPath ?? "FAILED")")

        guard let matchedBox = matchedBoxOpt else {
            logger.log("ERROR: 'サマリ' text not found in left panel.")
            await MainActor.run {
                updateLastMessage(content: "左側パネルに「サマリ」ボタンが見つかりませんでした")
            }
            return
        }

        logger.log("Found 'サマリ' line box: x=\(matchedBox.x), y=\(matchedBox.y), w=\(matchedBox.w), h=\(matchedBox.h)")

        // Calculate scale: pixel -> point
        let scaleX = CGFloat(fullImage.width) / windowBounds.width
        let scaleY = CGFloat(fullImage.height) / windowBounds.height
        logger.log("Scale factors: scaleX=\(scaleX), scaleY=\(scaleY)")
        logger.log("Window bounds: \(windowBounds) (points)")

        // Click at 80% from the left of the matched line box (pixel coords)
        let clickPixelX = CGFloat(matchedBox.x) + CGFloat(matchedBox.w) * 0.8
        let clickPixelY = CGFloat(matchedBox.y) + CGFloat(matchedBox.h) * 0.5
        logger.log("Click position in pixel coords: (\(clickPixelX)px, \(clickPixelY)px)")

        // Convert pixel coords to screen point coords
        let screenX = windowBounds.origin.x + clickPixelX / scaleX
        let screenY = windowBounds.origin.y + clickPixelY / scaleY
        let buttonCenter = CGPoint(x: screenX, y: screenY)
        logger.log("Calculated button center in screen point coords: (\(screenX)pt, \(screenY)pt) (80% from left of line box)")

        logger.log("Clicking 'サマリ' button at \(buttonCenter)")
        await MainActor.run {
            updateLastMessage(content: "「サマリ」ボタンを発見（\(Int(buttonCenter.x)), \(Int(buttonCenter.y))）、クリックします...")
        }

        // Ensure EHR window is in foreground before clicking
        activateChrome()
        try await Task.sleep(nanoseconds: 500_000_000)

        CGDisplayMoveCursorToPoint(mainDisplay, buttonCenter)
        try await Task.sleep(nanoseconds: 500_000_000)

        guard let btnDown = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: buttonCenter, mouseButton: .left),
              let btnUp = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: buttonCenter, mouseButton: .left) else {
            logger.log("ERROR: Failed to create button click events")
            throw NSError(domain: "SummaryAction", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to create button click events"])
        }
        btnDown.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 50_000_000)
        btnUp.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 100_000_000)

        guard let btnDown2 = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: buttonCenter, mouseButton: .left),
              let btnUp2 = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: buttonCenter, mouseButton: .left) else {
            logger.log("ERROR: Failed to create second button click events")
            throw NSError(domain: "SummaryAction", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to create second button click events"])
        }
        btnDown2.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 50_000_000)
        btnUp2.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 500_000_000)
        logger.log("Double-click posted")

        await MainActor.run {
            updateLastMessage(content: "「サマリ」ボタンをダブルクリックしました。次に「サマリの本文」を検索します...")
        }

        // Wait for screen transition after clicking サマリ button
        logger.log("Waiting for screen transition after サマリ button click...")
        try await Task.sleep(nanoseconds: 1_500_000_000)

        // Switch to AI chat while capturing (screencapture -l works without Chrome in front)
        activateSelf()
        try await Task.sleep(nanoseconds: 500_000_000)

        // Re-capture active window for summary body analysis
        logger.log("Re-capturing active window for summary body analysis...")
        guard let captureResultBody = captureActiveWindow(windowID: windowID) else {
            logger.log("ERROR: Failed to re-capture active window for summary body")
            throw NSError(domain: "SummaryAction", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to re-capture active window after サマリ click"])
        }
        let summaryBodyImage = captureResultBody.image
        let summaryBodyBounds = captureResultBody.bounds
        logger.log("Re-captured window for summary body: \(summaryBodyImage.width)x\(summaryBodyImage.height) pixels, bounds: \(summaryBodyBounds) points")

        // Recalculate scale for the new capture
        let bodyScaleX = CGFloat(summaryBodyImage.width) / summaryBodyBounds.width
        let bodyScaleY = CGFloat(summaryBodyImage.height) / summaryBodyBounds.height
        logger.log("Summary body scale factors: scaleX=\(bodyScaleX), scaleY=\(bodyScaleY)")

        // AI is already in front for VLM analysis
        
        // Crop main panel and search for "サマリの本文"
        let gaugeTaskBody = showProgressGauge { [self] msg in
            updateLastMessage(content: msg)
        }

        let bodyDividerX: Int
        if let dividerX = await detectVerticalDividerWithVLM(image: summaryBodyImage, logger: logger) {
            bodyDividerX = dividerX
            logger.log("VLM detected vertical divider at x=\(dividerX) for summary body")
        } else if let dividerX = detectVerticalDivider(image: summaryBodyImage) {
            bodyDividerX = dividerX
            logger.log("Pixel-based vertical divider detected at x=\(dividerX) for summary body")
        } else {
            bodyDividerX = summaryBodyImage.width / 2
            logger.log("Vertical divider not detected, falling back to half width for summary body")
        }
        gaugeTaskBody.cancel()
        let bodyPanelRect = CGRect(x: bodyDividerX, y: 0, width: summaryBodyImage.width - bodyDividerX, height: summaryBodyImage.height)
        guard let bodyPanelImage = summaryBodyImage.cropping(to: bodyPanelRect) else {
            logger.log("ERROR: Failed to crop main panel for summary body")
            throw NSError(domain: "SummaryAction", code: 7, userInfo: [NSLocalizedDescriptionKey: "Failed to crop main panel for summary body"])
        }
        logger.log("Cropped main panel for summary body: \(bodyPanelImage.width)x\(bodyPanelImage.height)")

        let bodyPanelPath = saveDebugImage(bodyPanelImage, name: "summary_body_panel")
        logger.log("Saved summary body panel debug image: \(bodyPanelPath ?? "FAILED")")

        logger.log("Searching for 'サマリの本文' in main panel via Tesseract OCR...")
        let (summaryBodyBoxOpt, bodyPanelBoxes) = findTextLineBoxWithTesseract(on: bodyPanelImage, searchText: "サマリの本文", logger: logger)

        let bodyOverlayPath = saveTesseractOCROverlayImage(original: bodyPanelImage, boxes: bodyPanelBoxes, matched: summaryBodyBoxOpt, name: "summary_body_ocr_overlay")
        logger.log("Saved summary body OCR overlay: \(bodyOverlayPath ?? "FAILED")")

        guard let summaryBodyBox = summaryBodyBoxOpt else {
            logger.log("ERROR: 'サマリの本文' text not found in main panel.")
            await MainActor.run {
                updateLastMessage(content: "メインパネルに「サマリの本文」が見つかりませんでした")
            }
            return
        }

        logger.log("Found 'サマリの本文' line box: x=\(summaryBodyBox.x), y=\(summaryBodyBox.y), w=\(summaryBodyBox.w), h=\(summaryBodyBox.h)")

        // Calculate click position (bottom of bounding box)
        let bodyClickPixelX = CGFloat(summaryBodyBox.x) + CGFloat(summaryBodyBox.w) / 2.0
        let bodyClickPixelY = CGFloat(summaryBodyBox.y) + CGFloat(summaryBodyBox.h)
        logger.log("Summary body click position in main-panel pixel coords: (\(bodyClickPixelX)px, \(bodyClickPixelY)px)")

        let bodyPointX = bodyClickPixelX / bodyScaleX
        let bodyPointY = bodyClickPixelY / bodyScaleY
        let bodyDividerPointX = CGFloat(bodyDividerX) / bodyScaleX
        let bodyScreenX = summaryBodyBounds.origin.x + bodyDividerPointX + bodyPointX
        let bodyScreenY = summaryBodyBounds.origin.y + bodyPointY
        let bodyClickPoint = CGPoint(x: bodyScreenX, y: bodyScreenY)
        logger.log("Final summary body click in screen point coords: (\(bodyScreenX)pt, \(bodyScreenY)pt)")

        logger.log("Clicking 'サマリの本文' at \(bodyClickPoint)")
        await MainActor.run {
            updateLastMessage(content: "「サマリの本文」を発見、クリックします...")
        }

        activateChrome()
        try await Task.sleep(nanoseconds: 500_000_000)

        CGDisplayMoveCursorToPoint(mainDisplay, bodyClickPoint)
        try await Task.sleep(nanoseconds: 500_000_000)

        guard let bodyDown = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: bodyClickPoint, mouseButton: .left),
              let bodyUp = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: bodyClickPoint, mouseButton: .left) else {
            logger.log("ERROR: Failed to create summary body click events")
            throw NSError(domain: "SummaryAction", code: 8, userInfo: [NSLocalizedDescriptionKey: "Failed to create summary body click events"])
        }
        bodyDown.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 50_000_000)
        bodyUp.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 500_000_000)
        logger.log("Summary body click posted")

        await MainActor.run {
            updateLastMessage(content: "「サマリの本文」をクリックしました。入力します...")
        }

        // Type body text
        logger.log("Typing text in summary body field...")
        await pasteText(content)
        logger.log("Text typed in summary body field")

        await MainActor.run {
            updateLastMessage(content: "サマリ本文欄に入力しました。")
        }

        // Switch back to EHR-Agent
        activateSelf()
        logger.log("Switched back to EHR-Agent")
    }

    private func callTextLLM(prompt: String, logger: EHRLogger) async throws -> String {
        let body: [String: Any] = [
            "model": currentModel,
            "temperature": 0,
            "messages": [["role": "user", "content": prompt]],
            "stream": false,
            "max_tokens": 2048
        ]
        guard let url = URL(string: "\(apiBase)/chat/completions") else {
            throw NSError(domain: "TextLLM", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid API URL"])
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(authHeader(), forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 120

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NSError(domain: "TextLLM", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "HTTP Error"])
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let first = choices.first,
              let message = first["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw NSError(domain: "TextLLM", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])
        }
        logger.log("TextLLM response (\(content.count) chars): \(content.prefix(200))")
        return content
    }

    private func runCreateSummaryAction() async throws {
        let logger = EHRLogger()
        defer { logger.saveToFile() }
        logger.log("===== Create Summary Action Started =====")

        // Step 0: 過去診療録ページが開いていなければ開く
        await MainActor.run {
            appendMessage(role: "assistant", content: "過去診療録ページを確認中...")
        }
        if let chromeResult = await findChromeWindowSCK(),
           let captureResult = await captureWindowViaSCK(chromeResult.scWindow) {
            await checkAndOpenPastRecordsIfNeeded(
                captureImage: captureResult.image,
                captureBounds: captureResult.bounds,
                logger: logger
            )
        } else {
            logger.log("Chrome window not found for past_records check, proceeding anyway")
        }

        // Step 1: 診療録読み取り
        activateSelf()
        try await Task.sleep(nanoseconds: 300_000_000)
        await MainActor.run {
            updateLastMessage(content: "診療録を読み取ります...")
        }
        let ehrJSON = try await runEHRReader()

        // Step 2: LLM でサマリ生成（ehr_composer.py の _generate_summary() と同一構成）
        await MainActor.run {
            updateLastMessage(content: "診療録読み取り完了。サマリを生成中...")
        }
        let prompt = """
        ### 指示
        以下の過去診療録データを元に、退院時サマリを作成してください。

        ### 過去診療録データ
        ```json
        \(ehrJSON)
        ```

        ### 重要：日付の扱い
        - 診療録データの**最初の `date`** を入院日として扱ってください。
        - 診療録データの**最後の `date`** を退院日として扱ってください。
        - サマリ内では「本日」「今日」「現在」などの相対的な表現を**一切使わず**、必ず具体的な日付（例：YYYY年MM月DD日）を記載してください。
          - 悪い例：「本日退院の運びとなった」「本日午前中に退院とし」
          - 良い例：「YYYY年MM月DD日に退院となった」「YYYY年MM月DD日午前中に退院とし」

        ### 出力形式
        以下の7項目に分けて記載してください。各項目の内容が充実するよう詳細な経過・処方・指導内容を含めてください。全体でMicrosoft Wordの1〜2ページに収まる内容にしてください。

        1. **主訴**
        2. **現病歴**（発症から入院日（YYYY年MM月DD日）までの経過のみを記載すること。入院後の治療経過・退院に関する内容は書かず、「入院後経過」に委ねること）
        3. **既往歴**
        4. **入院後経過**（入院後の治療経過を詳細に記載。検査所見・検査値、投薬内容・薬剤名・用量、処置内容、治療反応を含めること）
        5. **退院時状況**
        6. **退院時方針**（退院日を具体的な日付で明記すること）
        7. **退院時処方**

        ### 出力の書式
        - 必ず各行の先頭に `[項目名]` を付けてください。例: `[主訴] 呼吸困難、喘鳴`
        - 項目間は1行の空行で区切ってください。
        - 各項目の内容は連続した文章として記載し、項目内での改行は避けてください。
        - 内容が短くなりすぎないよう、検査値・薬剤名・用量・治療反応などの詳細を漏らさず記載してください。

        ### 制約
        - 診療録に記載されている情報のみを使用し、推測や補完は行わないでください。
        - 日付順に診療経過を整理し、簡潔に記載してください。
        - 「本日」「今日」「現在」などの相対的な時間表現は絶対に使用しないでください。
        """
        let summaryText = try await callTextLLM(prompt: prompt, logger: logger)

        // Step 3: サマリ欄に入力
        await MainActor.run {
            updateLastMessage(content: "サマリ生成完了。サマリ欄に入力します...")
        }
        try await runSummaryAction(content: summaryText)
    }

    private func runCreateReferralLetterAction() async throws {
        let logger = EHRLogger()
        defer { logger.saveToFile() }
        logger.log("===== Create Referral Letter Action Started =====")

        // Step 0: 過去診療録ページが開いていなければ開く（サマリ作成と同様）
        await MainActor.run {
            appendMessage(role: "assistant", content: "過去診療録ページを確認中...")
        }
        if let chromeResult = await findChromeWindowSCK(),
           let captureResult = await captureWindowViaSCK(chromeResult.scWindow) {
            await checkAndOpenPastRecordsIfNeeded(
                captureImage: captureResult.image,
                captureBounds: captureResult.bounds,
                logger: logger
            )
        } else {
            logger.log("Chrome window not found for past_records check, proceeding anyway")
        }

        // Step 1: 診療録読み取り（サマリ作成と同様）
        activateSelf()
        try await Task.sleep(nanoseconds: 300_000_000)
        await MainActor.run {
            updateLastMessage(content: "診療録を読み取ります...")
        }
        let ehrJSON = try await runEHRReader()

        // Step 2: LLM で診療情報提供書生成
        await MainActor.run {
            updateLastMessage(content: "診療録読み取り完了。診療情報提供書を生成中...")
        }
        let prompt = """
        ### 指示
        以下の過去診療録データを元に、かかりつけ医宛の入院に関する診療情報提供書（紹介状）の本文を作成してください。

        ### 過去診療録データ
        ```json
        \(ehrJSON)
        ```

        ### 重要：日付・時刻の表現について
        - 診療録データの**最初の `date`** を入院日として扱ってください。
        - 診療録データの**最後の `date`** を退院日として扱ってください。
        - **相対的な時間表現は一切使用禁止**です。以下の語句はすべて禁止します：
          「本日」「今日」「昨日」「昨晩」「今朝」「翌日」「前日」「一昨日」「先日」「〇日後」「〇日前」「入院〇日目」「今週」「今月」「現在」「最近」など
        - 時間を指す際は、必ず診療録の `date` フィールドの具体的な日付（例：2025年4月10日）を使用してください。
        - 日付が診療録に明記されていない出来事（例：夜間の症状）は、「〇月〇日夜」のように当日の日付を基準に表現してください。

        ### 出力形式
        診療情報提供書の本文として、以下の項目を含めてください：

        1. **入院期間**（入院日〜退院日）
        2. **入院理由・主訴**
        3. **入院中の経過**（治療内容・検査所見・投薬を含む詳細な経過）
        4. **退院時診断**
        5. **退院時処方**（薬剤名・用量・用法）
        6. **今後の方針・お願い**（かかりつけ医へのフォローアップ依頼事項）

        ### 出力の書式
        - 各項目は「【項目名】」の形式で見出しを付けてください
        - 読みやすい文章形式で記載してください
        - 診療録に記載されている情報のみを使用し、推測や補完は行わないでください
        - 「本日」「今日」などの相対的な時間表現は絶対に使用しないでください
        """
        let letterText = try await callTextLLM(prompt: prompt, logger: logger)

        // Step 3: 情報提供書フォームに本文を貼り付け
        await MainActor.run {
            updateLastMessage(content: "診療情報提供書生成完了。フォームに入力します...")
        }
        try await runLetterAction(content: letterText)
    }

    private func runLetterAction(content: String? = nil) async throws {
        let logger = EHRLogger()
        defer { logger.saveToFile() }
        logger.log("===== Letter Action Started =====")

        await MainActor.run {
            updateLastMessage(content: "情報提供書ボタンを検索・クリックします...")
        }

        // Switch to EHR app
        activateChrome()
        try await Task.sleep(nanoseconds: 500_000_000)

        let mainDisplay = CGMainDisplayID()
        let displayBounds = CGDisplayBounds(mainDisplay)
        let centerPoint = CGPoint(x: displayBounds.width / 2, y: displayBounds.height / 2)

        logger.log("Clicking center of screen to activate window: \(centerPoint)")

        CGDisplayMoveCursorToPoint(mainDisplay, centerPoint)
        try await Task.sleep(nanoseconds: 500_000_000)

        guard let clickDown = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: centerPoint, mouseButton: .left),
              let clickUp = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: centerPoint, mouseButton: .left) else {
            logger.log("ERROR: Failed to create center click events")
            throw NSError(domain: "LetterAction", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create click events"])
        }
        clickDown.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 50_000_000)
        clickUp.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 1_000_000_000)
        logger.log("Center click posted")

        // Find active window ID and bounds
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
            throw NSError(domain: "LetterAction", code: 2, userInfo: [NSLocalizedDescriptionKey: "No active window found"])
        }

        logger.log("Capturing active window (first screenshot)...")
        guard let firstCapture = captureActiveWindow(windowID: windowID) else {
            logger.log("ERROR: Failed to capture active window")
            throw NSError(domain: "LetterAction", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to capture active window"])
        }
        let fullImage = firstCapture.image
        let windowBounds = firstCapture.bounds
        logger.log("Active window captured: \(fullImage.width)x\(fullImage.height), bounds: \(windowBounds)")

        let fullPanelPath = saveDebugImage(fullImage, name: "letter_fullscreen")
        logger.log("Saved debug image: full=\(fullPanelPath ?? "FAILED")")

        logger.log("Searching for '情報提供書' button via template matching...")
        guard let templatePath = Bundle.main.path(
            forResource: "letter_button_gray",
            ofType: "png",
            inDirectory: "match_templates"
        ) else {
            logger.log("ERROR: letter_button_gray template not found in bundle")
            await MainActor.run {
                updateLastMessage(content: "テンプレート画像が見つかりませんでした")
            }
            throw NSError(domain: "LetterAction", code: 4, userInfo: [NSLocalizedDescriptionKey: "Template not found"])
        }
        guard let nsTemplateImg = NSImage(contentsOfFile: templatePath),
              let template = nsTemplateImg.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            logger.log("ERROR: Failed to load letter_button_gray template")
            throw NSError(domain: "LetterAction", code: 4, userInfo: [NSLocalizedDescriptionKey: "Template load failed"])
        }
        logger.log("Template loaded: \(template.width)x\(template.height)px")

        let t0 = Date()
        guard let match = TemplateMatchingWrapper.matchSource(
            fullImage,
            template: template,
            searchRegion: CGRect(x: 0, y: 0, width: CGFloat(fullImage.width), height: CGFloat(fullImage.height)),
            threshold: 0.7
        ) else {
            logger.log("ERROR: '情報提供書' template not matched (elapsed \(String(format: "%.2f", -t0.timeIntervalSinceNow))s)")
            await MainActor.run {
                updateLastMessage(content: "左側パネルに「情報提供書」ボタンが見つかりませんでした")
            }
            return
        }
        logger.log("Template matched at (\(Int(match.position.x)), \(Int(match.position.y))) score=\(String(format: "%.3f", match.score)) elapsed=\(String(format: "%.2f", -t0.timeIntervalSinceNow))s")

        let scaleX = CGFloat(fullImage.width) / windowBounds.width
        let scaleY = CGFloat(fullImage.height) / windowBounds.height
        let clickPixelX = match.position.x + CGFloat(template.width) / 2
        let clickPixelY = match.position.y + CGFloat(template.height) / 2
        let screenX = windowBounds.origin.x + clickPixelX / scaleX
        let screenY = windowBounds.origin.y + clickPixelY / scaleY
        let buttonCenter = CGPoint(x: screenX, y: screenY)
        logger.log("Calculated button center: (\(Int(screenX))pt, \(Int(screenY))pt)")

        logger.log("Clicking '情報提供書' button at \(buttonCenter)")
        await MainActor.run {
            updateLastMessage(content: "「情報提供書」ボタンを発見（\(Int(buttonCenter.x)), \(Int(buttonCenter.y))）、クリックします...")
        }

        activateChrome()
        try await Task.sleep(nanoseconds: 500_000_000)

        CGDisplayMoveCursorToPoint(mainDisplay, buttonCenter)
        try await Task.sleep(nanoseconds: 500_000_000)

        guard let btnDown = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: buttonCenter, mouseButton: .left),
              let btnUp = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: buttonCenter, mouseButton: .left) else {
            logger.log("ERROR: Failed to create button click events")
            throw NSError(domain: "LetterAction", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to create button click events"])
        }
        btnDown.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 50_000_000)
        btnUp.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 500_000_000)
        logger.log("Button click posted")

        await MainActor.run {
            updateLastMessage(content: "「情報提供書」ボタンをクリックしました。次に入力エリアを検索します...")
        }

        try await Task.sleep(nanoseconds: 1_500_000_000)

        activateSelf()
        try await Task.sleep(nanoseconds: 500_000_000)

        logger.log("Re-capturing active window for main panel analysis...")
        guard let captureResult2 = captureActiveWindow(windowID: windowID) else {
            logger.log("ERROR: Failed to re-capture active window")
            throw NSError(domain: "LetterAction", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to re-capture active window after 情報提供書 click"])
        }
        let postClickImage = captureResult2.image
        let postClickBounds = captureResult2.bounds
        logger.log("Re-captured window: \(postClickImage.width)x\(postClickImage.height) pixels, bounds: \(postClickBounds) points")

        let postScaleX = CGFloat(postClickImage.width) / postClickBounds.width
        let postScaleY = CGFloat(postClickImage.height) / postClickBounds.height
        logger.log("Post-click scale factors: scaleX=\(postScaleX), scaleY=\(postScaleY)")

        activateSelf()

        let mainPanelDividerX: Int
        if let dividerX = await detectVerticalDividerWithVLM(image: postClickImage, logger: logger) {
            mainPanelDividerX = dividerX
        } else if let dividerX = detectVerticalDivider(image: postClickImage) {
            mainPanelDividerX = dividerX
        } else {
            mainPanelDividerX = postClickImage.width / 2
        }

        let mainPanelRect = CGRect(x: mainPanelDividerX, y: 0, width: postClickImage.width - mainPanelDividerX, height: postClickImage.height)
        guard let mainPanelImage = postClickImage.cropping(to: mainPanelRect) else {
            logger.log("ERROR: Failed to crop main panel")
            throw NSError(domain: "LetterAction", code: 7, userInfo: [NSLocalizedDescriptionKey: "Failed to crop main panel"])
        }
        logger.log("Cropped main panel: \(mainPanelImage.width)x\(mainPanelImage.height)")

        let mainPanelPath = saveDebugImage(mainPanelImage, name: "letter_main_panel")
        logger.log("Saved main panel debug image: \(mainPanelPath ?? "FAILED")")

        logger.log("Searching for '医療機関名を入力' in main panel via Tesseract OCR...")
        let (inputAreaBoxOpt, mainPanelBoxes) = findTextLineBoxWithTesseract(on: mainPanelImage, searchText: "医療機関名を入力", logger: logger)

        let mainOverlayPath = saveTesseractOCROverlayImage(original: mainPanelImage, boxes: mainPanelBoxes, matched: inputAreaBoxOpt, name: "letter_main_ocr_overlay")
        logger.log("Saved main panel OCR overlay: \(mainOverlayPath ?? "FAILED")")

        guard let inputAreaBox = inputAreaBoxOpt else {
            logger.log("ERROR: '医療機関名を入力' text not found in main panel.")
            await MainActor.run {
                updateLastMessage(content: "右側パネルに「医療機関名を入力」が見つかりませんでした")
            }
            return
        }

        logger.log("Found '医療機関名を入力' line box: x=\(inputAreaBox.x), y=\(inputAreaBox.y), w=\(inputAreaBox.w), h=\(inputAreaBox.h)")

        let inputClickPixelX = CGFloat(inputAreaBox.x) + CGFloat(inputAreaBox.w) / 2.0
        let inputClickPixelY = CGFloat(inputAreaBox.y) + CGFloat(inputAreaBox.h)

        let inputPointX = inputClickPixelX / postScaleX
        let inputPointY = inputClickPixelY / postScaleY
        let dividerPointX = CGFloat(mainPanelDividerX) / postScaleX
        let finalScreenX = postClickBounds.origin.x + dividerPointX + inputPointX
        let finalScreenY = postClickBounds.origin.y + inputPointY
        let inputClickPoint = CGPoint(x: finalScreenX, y: finalScreenY)
        logger.log("Final input area click in screen point coords: (\(finalScreenX)pt, \(finalScreenY)pt)")

        logger.log("Clicking input area at \(inputClickPoint)")
        // Ensure EHR is in foreground before clicking
        activateChrome()
        try await Task.sleep(nanoseconds: 500_000_000)
        await MainActor.run {
            updateLastMessage(content: "「医療機関名を入力」を発見、入力欄をクリックします...")
        }

        CGDisplayMoveCursorToPoint(mainDisplay, inputClickPoint)
        try await Task.sleep(nanoseconds: 500_000_000)

        guard let inputDown = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: inputClickPoint, mouseButton: .left),
              let inputUp = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: inputClickPoint, mouseButton: .left) else {
            logger.log("ERROR: Failed to create input area click events")
            throw NSError(domain: "LetterAction", code: 8, userInfo: [NSLocalizedDescriptionKey: "Failed to create input area click events"])
        }
        inputDown.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 50_000_000)
        inputUp.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 500_000_000)
        logger.log("Input area click posted")

        await MainActor.run {
            updateLastMessage(content: "「医療機関名を入力」入力欄をクリックしました。テスト医療機関名を入力します...")
        }

        logger.log("Typing 'テスト医療機関' in medical institution input field...")
        await pasteText("テスト医療機関")
        logger.log("'テスト医療機関' typed in medical institution field")
        try await Task.sleep(nanoseconds: 500_000_000)

        await MainActor.run {
            updateLastMessage(content: "テスト医療機関名を入力しました。次に「本文を入力」を検索します...")
        }

        // Re-capture for body input
        logger.log("Re-capturing active window for body detection...")
        activateSelf()
        guard let captureResult3 = captureActiveWindow(windowID: windowID) else {
            logger.log("ERROR: Failed to re-capture active window for body")
            throw NSError(domain: "LetterAction", code: 9, userInfo: [NSLocalizedDescriptionKey: "Failed to re-capture active window for body"])
        }
        let bodyImage = captureResult3.image
        let bodyBounds = captureResult3.bounds
        logger.log("Re-captured window for body: \(bodyImage.width)x\(bodyImage.height) pixels, bounds: \(bodyBounds) points")

        let bodyScaleX = CGFloat(bodyImage.width) / bodyBounds.width
        let bodyScaleY = CGFloat(bodyImage.height) / bodyBounds.height
        logger.log("Body scale factors: scaleX=\(bodyScaleX), scaleY=\(bodyScaleY)")

        activateSelf()

        let bodyDividerX: Int
        if let dividerX = await detectVerticalDividerWithVLM(image: bodyImage, logger: logger) {
            bodyDividerX = dividerX
        } else if let dividerX = detectVerticalDivider(image: bodyImage) {
            bodyDividerX = dividerX
        } else {
            bodyDividerX = bodyImage.width / 2
        }

        let bodyPanelRect = CGRect(x: bodyDividerX, y: 0, width: bodyImage.width - bodyDividerX, height: bodyImage.height)
        guard let bodyPanelImage = bodyImage.cropping(to: bodyPanelRect) else {
            logger.log("ERROR: Failed to crop main panel for body")
            throw NSError(domain: "LetterAction", code: 10, userInfo: [NSLocalizedDescriptionKey: "Failed to crop main panel for body"])
        }
        logger.log("Cropped main panel for body: \(bodyPanelImage.width)x\(bodyPanelImage.height)")

        let bodyPanelPath = saveDebugImage(bodyPanelImage, name: "letter_body_panel")
        logger.log("Saved body panel debug image: \(bodyPanelPath ?? "FAILED")")

        logger.log("Searching for '本文を入力' in main panel via Tesseract OCR...")
        let (bodyBoxOpt, bodyPanelBoxes) = findTextLineBoxWithTesseract(on: bodyPanelImage, searchText: "本文を入力", logger: logger)

        let bodyOverlayPath = saveTesseractOCROverlayImage(original: bodyPanelImage, boxes: bodyPanelBoxes, matched: bodyBoxOpt, name: "letter_body_ocr_overlay")
        logger.log("Saved body OCR overlay: \(bodyOverlayPath ?? "FAILED")")

        guard let bodyBox = bodyBoxOpt else {
            logger.log("ERROR: '本文を入力' text not found in main panel.")
            await MainActor.run {
                updateLastMessage(content: "メインパネルに「本文を入力」が見つかりませんでした")
            }
            return
        }

        logger.log("Found '本文を入力' line box: x=\(bodyBox.x), y=\(bodyBox.y), w=\(bodyBox.w), h=\(bodyBox.h)")

        let bodyClickPixelX = CGFloat(bodyBox.x) + CGFloat(bodyBox.w) / 2.0
        let bodyClickPixelY = CGFloat(bodyBox.y) + CGFloat(bodyBox.h)
        logger.log("Body click position in main-panel pixel coords: (\(bodyClickPixelX)px, \(bodyClickPixelY)px)")

        let bodyPointX = bodyClickPixelX / bodyScaleX
        let bodyPointY = bodyClickPixelY / bodyScaleY
        let bodyDividerPointX = CGFloat(bodyDividerX) / bodyScaleX
        let bodyScreenX = bodyBounds.origin.x + bodyDividerPointX + bodyPointX
        let bodyScreenY = bodyBounds.origin.y + bodyPointY
        let bodyClickPoint = CGPoint(x: bodyScreenX, y: bodyScreenY)
        logger.log("Final body click in screen point coords: (\(bodyScreenX)pt, \(bodyScreenY)pt)")

        logger.log("Clicking '本文を入力' at \(bodyClickPoint)")
        // Ensure EHR is in foreground before clicking
        activateChrome()
        try await Task.sleep(nanoseconds: 500_000_000)
        await MainActor.run {
            updateLastMessage(content: "「本文を入力」を発見、クリックします...")
        }

        CGDisplayMoveCursorToPoint(mainDisplay, bodyClickPoint)
        try await Task.sleep(nanoseconds: 500_000_000)

        guard let bodyDown = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: bodyClickPoint, mouseButton: .left),
              let bodyUp = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: bodyClickPoint, mouseButton: .left) else {
            logger.log("ERROR: Failed to create body click events")
            throw NSError(domain: "LetterAction", code: 11, userInfo: [NSLocalizedDescriptionKey: "Failed to create body click events"])
        }
        bodyDown.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 50_000_000)
        bodyUp.post(tap: .cghidEventTap)
        try await Task.sleep(nanoseconds: 500_000_000)
        logger.log("Body click posted")

        await MainActor.run {
            updateLastMessage(content: "「本文を入力」をクリックしました。テスト本文を入力します...")
        }

        logger.log("Typing body text in body field...")
        let bodyContent = content ?? "テスト本文"
        await pasteText(bodyContent)
        logger.log("Body text typed in body field (\(bodyContent.count) chars)")

        await MainActor.run {
            updateLastMessage(content: "テスト本文を入力しました。")
        }

        activateSelf()
        logger.log("Switched back to EHR-Agent")
    }

    private func showProgressGauge(estimatedSeconds: Int = 25, message: String = "レイアウト解析中...", updateMessage: @escaping (String) -> Void) -> Task<Void, Error> {
        return Task {
            let totalBlocks = 10
            for elapsed in 0...estimatedSeconds {
                try Task.checkCancellation()
                
                let progress = min(Double(elapsed) / Double(estimatedSeconds), 1.0)
                let filledBlocks = Int(progress * Double(totalBlocks))
                let emptyBlocks = totalBlocks - filledBlocks
                let bar = String(repeating: "█", count: filledBlocks) + String(repeating: "░", count: emptyBlocks)
                let percent = Int(progress * 100)
                let remaining = estimatedSeconds - elapsed
                
                let text = "\(message) [\(bar)] \(percent)% (あと\(remaining)秒)"
                
                await MainActor.run {
                    updateMessage(text)
                }
                
                try await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    private func activateSelf() {
        DispatchQueue.main.async {
            guard NSWorkspace.shared.frontmostApplication?.bundleIdentifier == "com.google.Chrome" else { return }
            let cmdDown = CGEvent(keyboardEventSource: nil, virtualKey: 0x37, keyDown: true)
            let tabDown = CGEvent(keyboardEventSource: nil, virtualKey: 0x30, keyDown: true)
            let tabUp   = CGEvent(keyboardEventSource: nil, virtualKey: 0x30, keyDown: false)
            let cmdUp   = CGEvent(keyboardEventSource: nil, virtualKey: 0x37, keyDown: false)
            cmdDown?.flags = .maskCommand
            tabDown?.flags = .maskCommand
            tabUp?.flags   = .maskCommand
            cmdDown?.post(tap: .cghidEventTap)
            tabDown?.post(tap: .cghidEventTap)
            tabUp?.post(tap: .cghidEventTap)
            cmdUp?.post(tap: .cghidEventTap)
        }
    }

    private func activateChrome() {
        DispatchQueue.main.async {
            NSWorkspace.shared.runningApplications
                .first(where: { $0.bundleIdentifier == "com.google.Chrome" })?
                .activate()
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

    private func pasteText(_ text: String) async {
        let pasteboard = NSPasteboard.general
        let oldContents = pasteboard.string(forType: .string)

        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)

        let cmdDown = CGEvent(keyboardEventSource: nil, virtualKey: 0x37, keyDown: true)
        let vDown = CGEvent(keyboardEventSource: nil, virtualKey: 0x09, keyDown: true)
        let vUp = CGEvent(keyboardEventSource: nil, virtualKey: 0x09, keyDown: false)
        let cmdUp = CGEvent(keyboardEventSource: nil, virtualKey: 0x37, keyDown: false)

        cmdDown?.flags = .maskCommand
        vDown?.flags = .maskCommand
        vUp?.flags = .maskCommand

        cmdDown?.post(tap: .cghidEventTap)
        vDown?.post(tap: .cghidEventTap)
        try? await Task.sleep(nanoseconds: 50_000_000)
        vUp?.post(tap: .cghidEventTap)
        cmdUp?.post(tap: .cghidEventTap)
        try? await Task.sleep(nanoseconds: 500_000_000)

        // Restore original clipboard
        pasteboard.clearContents()
        if let old = oldContents {
            pasteboard.setString(old, forType: .string)
        }
    }

    private func captureActiveWindow(windowID: Int) -> (image: CGImage, bounds: CGRect)? {
        let capturesDir = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("captures")
        try? FileManager.default.createDirectory(at: capturesDir, withIntermediateDirectories: true)

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = formatter.string(from: Date())
        let filename = "ehr_capture_\(timestamp).png"
        let fileURL = capturesDir.appendingPathComponent(filename)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        process.arguments = ["-x", "-o", "-l", "\(windowID)", fileURL.path]
        try? process.run()
        process.waitUntilExit()

        guard let data = try? Data(contentsOf: fileURL),
              let image = NSImage(data: data),
              let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }

        // Get window bounds on screen
        guard let windowBounds = getWindowBounds(windowID: windowID) else {
            return nil
        }

        return (cgImage, windowBounds)
    }

    private func findChromeWindowSCK() async -> (scWindow: SCWindow, bounds: CGRect)? {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            guard let window = content.windows
                .filter({ $0.owningApplication?.applicationName == "Google Chrome" })
                .max(by: { $0.frame.width < $1.frame.width }) else {
                return nil
            }
            return (window, window.frame)
        } catch {
            print("SCK findChrome error: \(error)")
            return nil
        }
    }

    private func captureWindowViaSCK(_ scWindow: SCWindow) async -> (image: CGImage, bounds: CGRect)? {
        do {
            let filter = SCContentFilter(desktopIndependentWindow: scWindow)
            let config = SCStreamConfiguration()
            let scale = NSScreen.main?.backingScaleFactor ?? 2.0
            config.width = Int(scWindow.frame.width * scale)
            config.height = Int(scWindow.frame.height * scale)
            let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
            return (image, scWindow.frame)
        } catch {
            print("SCK capture error: \(error)")
            return nil
        }
    }

    private func getWindowBounds(windowID: Int) -> CGRect? {
        guard let windowInfo = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] else {
            return nil
        }
        for info in windowInfo {
            if let winNum = info[kCGWindowNumber as String] as? Int, winNum == windowID,
               let boundsDict = info[kCGWindowBounds as String] as? [String: Any],
               let x = boundsDict["X"] as? CGFloat,
               let y = boundsDict["Y"] as? CGFloat,
               let width = boundsDict["Width"] as? CGFloat,
               let height = boundsDict["Height"] as? CGFloat {
                return CGRect(x: x, y: y, width: width, height: height)
            }
        }
        return nil
    }

    private func moveMouseTo(_ point: CGPoint) {
        guard let moveEvent = CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left) else { return }
        moveEvent.post(tap: .cghidEventTap)
    }

    private func postScrollEvent(at point: CGPoint, amount: Int32) {
        // Move mouse cursor to the target point first so the scroll targets the correct window
        moveMouseTo(point)
        usleep(50_000) // 50ms wait for cursor move
        
        guard let scrollEvent = CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 1, wheel1: amount, wheel2: 0, wheel3: 0) else { return }
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

        // Search only between 15% and 35% of screen width (left panel divider)
        let xStart = width * 15 / 100
        let xEnd = width * 35 / 100

        var candidateRanges: [(start: Int, end: Int)] = []
        var currentStart: Int? = nil

        for x in xStart..<xEnd {
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
            candidateRanges.append((start: start, end: xEnd - 1))
        }

        // Prefer the widest candidate (left panel divider is typically 2-5px wide)
        guard let bestRange = candidateRanges.max(by: {
            let w0 = $0.end - $0.start
            let w1 = $1.end - $1.start
            if w0 == w1 { return $0.start < $1.start }
            return w0 < w1
        }) else { return nil }
        return (bestRange.start + bestRange.end) / 2
    }

    private func detectVerticalDividerWithVLM(image: CGImage, logger: EHRLogger) async -> Int? {
        logger.log("Detecting vertical divider via VLM layout analysis...")
        
        // Convert CGImage to PNG Data
        let rep = NSBitmapImageRep(cgImage: image)
        guard let pngData = rep.representation(using: .png, properties: [:]) else {
            logger.log("ERROR: Failed to convert image to PNG for VLM")
            return nil
        }
        
        let base64 = pngData.base64EncodedString()
        let dataUrl = "data:image/png;base64,\(base64)"
        
        let prompt = """
        以下の患者カルテ画面のレイアウトを解析し、結果を以下のJSONフォーマットで出力してください。
        
        {
          "layout_analysis": {
            "target_element": "left_side_panel",
            "width_percentage": "数値（単位なし）"
          }
        }
        
        左サイドパネルの幅が画面全体の何%を占めるかを推定してください。例: 18, 20, 22 など。
        """
        
        let content: [[String: Any]] = [
            ["type": "text", "text": prompt],
            ["type": "image_url", "image_url": ["url": dataUrl]]
        ]
        
        let body: [String: Any] = [
            "model": currentModel,
            "temperature": 0,
            "messages": [
                ["role": "user", "content": content]
            ],
            "stream": false,
            "max_tokens": 512
        ]
        
        guard let url = URL(string: "\(apiBase)/chat/completions") else {
            logger.log("ERROR: Invalid API URL")
            return nil
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(authHeader(), forHTTPHeaderField: "Authorization")
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 30
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                logger.log("ERROR: VLM request failed")
                return nil
            }
            
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = json["choices"] as? [[String: Any]],
                  let first = choices.first,
                  let message = first["message"] as? [String: Any],
                  let contentStr = message["content"] as? String else {
                logger.log("ERROR: Invalid VLM response format")
                return nil
            }
            
            logger.log("VLM raw response: \(contentStr)")
            
            // Parse JSON from response
            guard let jsonData = contentStr.data(using: .utf8),
                  let responseJson = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                  let layout = responseJson["layout_analysis"] as? [String: Any],
                  let percentageStr = layout["width_percentage"] as? String,
                  let percentage = Double(percentageStr) else {
                logger.log("ERROR: Failed to parse width_percentage from VLM response")
                return nil
            }
            
            let dividerX = Int(Double(image.width) * percentage / 100.0)
            logger.log("VLM detected left panel width: \(percentage)% -> divider at x=\(dividerX)")
            return dividerX
            
        } catch {
            logger.log("ERROR: VLM request exception: \(error)")
            return nil
        }
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

    private func logVLMRequest(prompt: String, ocrText: String, response: String, error: String? = nil) {
        let logsDir = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: logsDir, withIntermediateDirectories: true)
        let logFile = logsDir.appendingPathComponent("vlm_requests.log")

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let timestamp = formatter.string(from: Date())

        var entry = """
        === VLM Request @ \(timestamp) ===
        [API Base]: \(apiBase)
        [Model]: \(currentModel)
        [Prompt]:
        \(prompt)
        [OCR Text]:
        \(ocrText)
        """
        
        if let error = error {
            entry += "\n[ERROR]:\n\(error)\n"
        } else {
            entry += "\n[Response]:\n\(response)\n"
        }
        
        entry += "=== End ===\n\n"

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
        request.timeoutInterval = 300

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            let errMsg = "Network error: \(error.localizedDescription)"
            logVLMRequest(prompt: prompt, ocrText: ocrText, response: "", error: errMsg)
            throw error
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            let errorMsg = "Invalid response type"
            logVLMRequest(prompt: prompt, ocrText: ocrText, response: "", error: errorMsg)
            throw NSError(domain: "EHRReader", code: 3, userInfo: [NSLocalizedDescriptionKey: errorMsg])
        }

        let responseString = String(data: data, encoding: .utf8) ?? "<binary data>"

        guard httpResponse.statusCode == 200 else {
            let errorMsg = "HTTP Error \(httpResponse.statusCode)\nResponse: \(responseString)"
            logVLMRequest(prompt: prompt, ocrText: ocrText, response: "", error: errorMsg)
            throw NSError(domain: "EHRReader", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP Error \(httpResponse.statusCode)"])
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let first = choices.first,
              let message = first["message"] as? [String: Any],
              let contentStr = message["content"] as? String else {
            let errorMsg = "Invalid response format\nResponse: \(responseString)"
            logVLMRequest(prompt: prompt, ocrText: ocrText, response: "", error: errorMsg)
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

    private func captureFullScreen() -> (image: CGImage, screenBounds: CGRect)? {
        let mainDisplay = CGMainDisplayID()
        let bounds = CGDisplayBounds(mainDisplay)

        let tempFile = FileManager.default.temporaryDirectory.appendingPathComponent("full_screenshot_\(Int(Date().timeIntervalSince1970)).png")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        process.arguments = ["-x", tempFile.path]
        try? process.run()
        process.waitUntilExit()

        guard let data = try? Data(contentsOf: tempFile),
              let nsImage = NSImage(data: data),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }
        try? FileManager.default.removeItem(at: tempFile)
        return (cgImage, bounds)
    }

    private func findTextBoundingBox(on cgImage: CGImage, searchText: String) -> CGRect? {
        let request = VNRecognizeTextRequest()
        request.recognitionLanguages = ["ja-JP", "en-US"]
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        guard let observations = request.results else { return nil }

        for observation in observations {
            for candidate in observation.topCandidates(10) {
                if candidate.string.contains(searchText) {
                    return observation.boundingBox
                }
            }
        }
        return nil
    }

    // MARK: - Tesseract OCR Helpers

    struct TesseractLineBox {
        let text: String
        let x: Int
        let y: Int
        let w: Int
        let h: Int
    }

    private func findTextLineBoxWithTesseract(on cgImage: CGImage, searchText: String, logger: EHRLogger) -> (match: TesseractLineBox?, all: [TesseractLineBox]) {
        let uuid = UUID().uuidString
        let tempDir = FileManager.default.temporaryDirectory
        let inputPath = tempDir.appendingPathComponent("tess_input_\(uuid).png").path
        let outputPrefix = tempDir.appendingPathComponent("tess_out_\(uuid)").path

        logger.log("Tesseract temp files: input=\(inputPath), outputPrefix=\(outputPrefix)")

        let rep = NSBitmapImageRep(cgImage: cgImage)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            logger.log("ERROR: Failed to convert image to PNG")
            return (nil, [])
        }
        do {
            try data.write(to: URL(fileURLWithPath: inputPath))
            logger.log("Wrote temp PNG: \(inputPath) (\(data.count) bytes)")
        } catch {
            logger.log("ERROR: Failed to write temp image: \(error)")
            return (nil, [])
        }

        let process = Process()
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/tesseract")
        process.arguments = [inputPath, outputPrefix, "-l", "jpn", "--psm", "11", "tsv"]
        if !FileManager.default.fileExists(atPath: "/opt/homebrew/bin/tesseract") {
            process.executableURL = URL(fileURLWithPath: "/usr/local/bin/tesseract")
            logger.log("Tesseract not found at /opt/homebrew/bin/tesseract, trying /usr/local/bin/tesseract")
        }
        logger.log("Running Tesseract: \(process.executableURL?.path ?? "UNKNOWN") \(process.arguments?.joined(separator: " ") ?? "")")

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            logger.log("ERROR: Failed to run tesseract: \(error)")
            try? FileManager.default.removeItem(atPath: inputPath)
            return (nil, [])
        }

        let exitCode = process.terminationStatus
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrStr = String(data: stderrData, encoding: .utf8) ?? ""
        logger.log("Tesseract exited with code \(exitCode)")
        if !stderrStr.isEmpty {
            logger.log("Tesseract stderr: \(stderrStr)")
        }

        guard exitCode == 0 else {
            logger.log("ERROR: Tesseract failed with exit code \(exitCode)")
            try? FileManager.default.removeItem(atPath: inputPath)
            return (nil, [])
        }

        let tsvPath = outputPrefix + ".tsv"
        guard let tsvData = try? Data(contentsOf: URL(fileURLWithPath: tsvPath)),
              let tsvString = String(data: tsvData, encoding: .utf8) else {
            logger.log("ERROR: Failed to read TSV output at \(tsvPath)")
            try? FileManager.default.removeItem(atPath: inputPath)
            try? FileManager.default.removeItem(atPath: tsvPath)
            return (nil, [])
        }
        logger.log("TSV output read: \(tsvString.count) chars")

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
        var parseCount = 0
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
            parseCount += 1
        }
        logger.log("Parsed \(parseCount) word entries from TSV")

        // Group by line
        var lineGroups: [String: [WordEntry]] = [:]
        for w in words {
            let key = "\(w.block)-\(w.par)-\(w.line)"
            lineGroups[key, default: []].append(w)
        }
        logger.log("Grouped into \(lineGroups.count) lines")

        var allBoxes: [TesseractLineBox] = []
        var matched: TesseractLineBox? = nil

        for (_, lineWords) in lineGroups {
            let sorted = lineWords.sorted { $0.left < $1.left }
            let lineText = sorted.map { $0.text }.joined(separator: " ")

            let minLeft = sorted.map { $0.left }.min() ?? 0
            let minTop = sorted.map { $0.top }.min() ?? 0
            let maxRight = sorted.map { $0.left + $0.width }.max() ?? 0
            let maxBottom = sorted.map { $0.top + $0.height }.max() ?? 0

            let normalizedText = lineText.replacingOccurrences(of: " ", with: "")
            let box = TesseractLineBox(
                text: lineText,
                x: minLeft,
                y: minTop,
                w: maxRight - minLeft,
                h: maxBottom - minTop
            )
            allBoxes.append(box)

            if matched == nil && normalizedText.contains(searchText) {
                matched = box
                logger.log("MATCHED line (normalized): '\(normalizedText)' at [\(box.x), \(box.y), \(box.w), \(box.h)]")
            }
        }

        logger.log("Total lines detected: \(allBoxes.count)")
        if matched == nil {
            logger.log("WARNING: '\(searchText)' not found in any line. All lines:")
            for (i, box) in allBoxes.enumerated() {
                let normalized = box.text.replacingOccurrences(of: " ", with: "")
                logger.log("  [\(i)] raw='\(box.text)' normalized='\(normalized)'")
            }
        }

        try? FileManager.default.removeItem(atPath: inputPath)
        try? FileManager.default.removeItem(atPath: tsvPath)

        return (matched, allBoxes)
    }

    private func saveTesseractOCROverlayImage(original: CGImage, boxes: [TesseractLineBox], matched: TesseractLineBox?, name: String) -> String? {
        let width = original.width
        let height = original.height
        let size = NSSize(width: width, height: height)
        let image = NSImage(size: size)

        image.lockFocus()

        let nsOriginal = NSImage(cgImage: original, size: size)
        nsOriginal.draw(in: NSRect(origin: .zero, size: size))

        // Tesseract uses top-left origin; NSImage/CG uses bottom-left origin
        // So we must flip Y: nsY = height - (box.y + box.h)

        // Draw all detected lines in yellow
        for box in boxes {
            let nsY = height - box.y - box.h
            let rect = NSRect(x: box.x, y: nsY, width: box.w, height: box.h)
            let path = NSBezierPath(rect: rect)
            NSColor.yellow.withAlphaComponent(0.15).setFill()
            path.fill()
            NSColor.yellow.withAlphaComponent(0.8).setStroke()
            path.lineWidth = 1
            path.stroke()
        }

        // Highlight matched box in red
        if let box = matched {
            let nsY = height - box.y - box.h
            let rect = NSRect(x: box.x, y: nsY, width: box.w, height: box.h)
            let path = NSBezierPath(rect: rect)
            NSColor.red.withAlphaComponent(0.35).setFill()
            path.fill()
            NSColor.red.setStroke()
            path.lineWidth = 3
            path.stroke()

            let attrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.boldSystemFont(ofSize: 16),
                .foregroundColor: NSColor.red
            ]
            let label = "サマリ"
            label.draw(at: NSPoint(x: CGFloat(box.x), y: CGFloat(nsY + box.h + 4)), withAttributes: attrs)
        }

        image.unlockFocus()

        guard let cgResult = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        return saveDebugImage(cgResult, name: name)
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
        let tsFormatter = ISO8601DateFormatter()
        tsFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let timestamp = tsFormatter.string(from: Date())
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

        // Server submenu
        let serverMenuItem = NSMenuItem(title: "Server", action: nil, keyEquivalent: "")
        let serverMenu = NSMenu()

        let ollxItem = NSMenuItem(title: "Use Omlx", action: #selector(switchToOmlx(_:)), keyEquivalent: "")
        ollxItem.target = self
        serverMenu.addItem(ollxItem)

        let ollamaItem = NSMenuItem(title: "Use Ollama", action: #selector(switchToOllama(_:)), keyEquivalent: "")
        ollamaItem.target = self
        serverMenu.addItem(ollamaItem)

        serverMenuItem.submenu = serverMenu
        appMenu.addItem(serverMenuItem)

        appMenu.addItem(NSMenuItem.separator())

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

    @objc private func switchToOmlx(_ sender: NSMenuItem) {
        chatVC?.switchServer(to: "omlx")
    }

    @objc private func switchToOllama(_ sender: NSMenuItem) {
        chatVC?.switchServer(to: "ollama")
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
