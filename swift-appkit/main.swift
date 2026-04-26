import AppKit
import Foundation

// MARK: - ChatMessage
struct ChatMessage {
    let role: String
    var content: String
}

// MARK: - ChatViewController
class ChatViewController: NSViewController {
    private var scrollView: NSScrollView!
    private var textView: NSTextView!
    private var inputField: NSTextField!
    private var sendButton: NSButton!
    private var modelSelector: NSPopUpButton!
    private var messages: [ChatMessage] = []
    private var isStreaming = false

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

        let inputBar = NSView(frame: NSRect(x: 0, y: 0, width: 1, height: 50))
        inputBar.autoresizingMask = [.width, .maxYMargin]

        inputField = NSTextField(frame: NSRect(x: 10, y: 10, width: 1, height: 30))
        inputField.autoresizingMask = [.width]
        inputField.placeholderString = "Type a message..."
        inputField.delegate = self
        inputField.focusRingType = .none
        inputBar.addSubview(inputField)

        sendButton = NSButton(frame: NSRect(x: 0, y: 10, width: 70, height: 30))
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

        topBar.frame = NSRect(x: 0, y: viewFrame.height - 40, width: viewFrame.width, height: 40)
        scrollView.frame = NSRect(x: 0, y: 50, width: viewFrame.width, height: viewFrame.height - 90)
        inputBar.frame = NSRect(x: 0, y: 0, width: viewFrame.width, height: 50)

        let sendButtonWidth: CGFloat = 70
        inputField.frame = NSRect(x: 10, y: 10, width: viewFrame.width - sendButtonWidth - 20, height: 30)
        sendButton.frame = NSRect(x: viewFrame.width - sendButtonWidth - 10, y: 10, width: sendButtonWidth, height: 30)

        modelSelector.frame = NSRect(x: 60, y: 7, width: viewFrame.width - 70, height: 25)
    }

    @objc private func modelChanged(_ sender: NSPopUpButton) {
        if let selected = sender.selectedItem?.title {
            currentModel = selected
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
                    self?.currentModel = modelIds.first ?? self!.currentModel
                }
            }
        }.resume()
    }

    @objc private func sendMessage() {
        let text = inputField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isStreaming else { return }

        messages.append(ChatMessage(role: "user", content: text))
        inputField.stringValue = ""
        appendMessage(role: "user", content: text)

        isStreaming = true
        sendButton.isEnabled = false
        inputField.isEnabled = false

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
                self?.inputField.isEnabled = true
                self?.inputField.becomeFirstResponder()
            }
        }
        task.resume()
    }
}

// MARK: - NSTextFieldDelegate
extension ChatViewController: NSTextFieldDelegate {
    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            sendMessage()
            return true
        }
        return false
    }
}

// MARK: - AppDelegate
class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!

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

        let chatVC = ChatViewController()
        window.contentViewController = chatVC

        window.setFrame(NSRect(x: x, y: y, width: width, height: height), display: true)
        window.makeKeyAndOrderFront(nil)
    }

    func setupMenuBar() {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        let appMenu = NSMenu()

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
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
