import AppKit
import Foundation
import CoreGraphics

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

        DispatchQueue.global().async { [weak self] in
            Thread.sleep(forTimeInterval: 3)

            guard let event = CGEvent(source: nil) else {
                DispatchQueue.main.async {
                    self?.debugButton.isEnabled = true
                    self?.debugButton.title = "Debug"
                }
                return
            }
            let cursorPosition = event.location

            let clickDown = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: cursorPosition, mouseButton: .left)
            let clickUp = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: cursorPosition, mouseButton: .left)
            clickDown?.post(tap: .cghidEventTap)
            clickUp?.post(tap: .cghidEventTap)

            Thread.sleep(forTimeInterval: 0.5)

            let source = CGEventSource(stateID: .hidSystemState)
            for char in "test" {
                if let keyCode = self?.charToKeyCode(char) {
                    let keyDown = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: true)
                    let keyUp = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: false)
                    keyDown?.post(tap: .cghidEventTap)
                    keyUp?.post(tap: .cghidEventTap)
                }
            }

            Thread.sleep(forTimeInterval: 0.5)

            let desktopURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop/debug_screenshot.png")
            var windowID: Int = 0
            if let windowInfo = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] {
                for info in windowInfo {
                    if let layer = info[kCGWindowLayer as String] as? Int, layer == 0,
                       let ownerName = info[kCGWindowOwnerName as String] as? String,
                       ownerName != "Window Server",
                       ownerName != "Dock",
                       let winNum = info[kCGWindowNumber as String] as? Int {
                        windowID = winNum
                        break
                    }
                }
            }
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
            process.arguments = ["-x", "-l", "\(windowID)", desktopURL.path]
            try? process.run()
            process.waitUntilExit()

            DispatchQueue.main.async {
                self?.debugButton.isEnabled = true
                self?.debugButton.title = "Debug"
            }
        }
    }

    private func charToKeyCode(_ char: Character) -> CGKeyCode? {
        let keyMap: [Character: CGKeyCode] = [
            "t": 17, "e": 14, "s": 1,
        ]
        return keyMap[char]
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
                    self?.currentModel = modelIds.first ?? self!.currentModel
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
        window.title = "AI Chat"
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
            keyEquivalent: ""
        )
        debugMenuItem.target = self
        appMenu.addItem(debugMenuItem)

        let quitMenuItem = NSMenuItem(
            title: "Quit AI Chat",
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
