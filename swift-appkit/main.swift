import AppKit
import Foundation
import CoreGraphics
import ApplicationServices

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
        debugButton.title = "Wait..."

        let weakSelf: ChatViewController? = self
        let workItem = DispatchWorkItem {
            guard let strongSelf = weakSelf else { return }
            print("[Debug] Starting debug action")

            let mainDisplay = CGMainDisplayID()
            let bounds = CGDisplayBounds(mainDisplay)
            let centerPoint = CGPoint(x: bounds.width / 2, y: bounds.height / 2)
            print("[Debug] Moving cursor to center: \(centerPoint)")

            CGDisplayMoveCursorToPoint(mainDisplay, centerPoint)

            Thread.sleep(forTimeInterval: 1.0)

            guard let clickDown = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: centerPoint, mouseButton: .left) else {
                print("[Debug] Error: Failed to create clickDown event")
                DispatchQueue.main.async {
                    strongSelf.debugButton.isEnabled = true
                    strongSelf.debugButton.title = "Debug"
                }
                return
            }
            guard let clickUp = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: centerPoint, mouseButton: .left) else {
                print("[Debug] Error: Failed to create clickUp event")
                DispatchQueue.main.async {
                    strongSelf.debugButton.isEnabled = true
                    strongSelf.debugButton.title = "Debug"
                }
                return
            }

            print("[Debug] Posting clickDown")
            clickDown.post(tap: .cghidEventTap)
            Thread.sleep(forTimeInterval: 0.05)
            print("[Debug] Posting clickUp")
            clickUp.post(tap: .cghidEventTap)
            print("[Debug] Click events posted at screen center: \(centerPoint)")

            Thread.sleep(forTimeInterval: 0.5)

            var windowID: Int = 0
            if let windowInfo = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] {
                for info in windowInfo {
                    if let layer = info[kCGWindowLayer as String] as? Int, layer == 0,
                       let ownerName = info[kCGWindowOwnerName as String] as? String,
                       ownerName != "Window Server",
                       ownerName != "Dock",
                       let winNum = info[kCGWindowNumber as String] as? Int {
                        windowID = winNum
                        print("[Debug] Found active window: \(ownerName) (ID: \(winNum))")
                        break
                    }
                }
            }

            if windowID == 0 {
                print("[Debug] Error: No active window found")
                DispatchQueue.main.async {
                    let alert = NSAlert()
                    alert.messageText = "Debug Error"
                    alert.informativeText = "No active window found"
                    alert.addButton(withTitle: "OK")
                    alert.runModal()
                    strongSelf.debugButton.isEnabled = true
                    strongSelf.debugButton.title = "Debug"
                }
                return
            }

            let capturesDir = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("captures")
            print("[Debug] Captures directory path: \(capturesDir.path)")

            if !FileManager.default.fileExists(atPath: capturesDir.path) {
                do {
                    try FileManager.default.createDirectory(at: capturesDir, withIntermediateDirectories: true)
                    print("[Debug] Created captures directory: \(capturesDir.path)")
                } catch {
                    print("[Debug] Error creating directory: \(error)")
                }
            }

            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd_HHmmss"

            let timestamp1 = formatter.string(from: Date())
            let filename1 = "debug_\(timestamp1)_1.png"
            let fileURL1 = capturesDir.appendingPathComponent(filename1)
            print("[Debug] Capturing screenshot 1 to: \(fileURL1.path)")

            let process1 = Process()
            process1.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
            process1.arguments = ["-x", "-l", "\(windowID)", fileURL1.path]
            let pipe1 = Pipe()
            process1.standardError = pipe1
            try? process1.run()
            process1.waitUntilExit()

            let errorData1 = pipe1.fileHandleForReading.readDataToEndOfFile()
            if let errorOutput1 = String(data: errorData1, encoding: .utf8), !errorOutput1.isEmpty {
                print("[Debug] screencapture stderr: \(errorOutput1)")
            }

            Thread.sleep(forTimeInterval: 0.5)

            let scrollAmount = Int(bounds.height / 2)
            let scrollEvent = CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 1, wheel1: Int32(-scrollAmount), wheel2: 0, wheel3: 0)
            scrollEvent?.location = centerPoint
            scrollEvent?.post(tap: .cghidEventTap)
            print("[Debug] Scrolled down by \(scrollAmount) pixels")

            Thread.sleep(forTimeInterval: 0.5)

            let timestamp2 = formatter.string(from: Date())
            let filename2 = "debug_\(timestamp2)_2.png"
            let fileURL2 = capturesDir.appendingPathComponent(filename2)
            print("[Debug] Capturing screenshot 2 to: \(fileURL2.path)")

            let process2 = Process()
            process2.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
            process2.arguments = ["-x", "-l", "\(windowID)", fileURL2.path]
            let pipe2 = Pipe()
            process2.standardError = pipe2
            try? process2.run()
            process2.waitUntilExit()

            let errorData2 = pipe2.fileHandleForReading.readDataToEndOfFile()
            if let errorOutput2 = String(data: errorData2, encoding: .utf8), !errorOutput2.isEmpty {
                print("[Debug] screencapture stderr: \(errorOutput2)")
            }

            DispatchQueue.main.async {
                strongSelf.debugButton.isEnabled = true
                strongSelf.debugButton.title = "Debug"
            }
        }
        DispatchQueue.global(qos: .userInitiated).async(execute: workItem)
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
