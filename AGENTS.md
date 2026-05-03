# EHR-Agentic-Toolkit - Agent Notes

## Swift AppKit (EHR-Agent)

### Build

```bash
cd swift-appkit
swiftc -o /tmp/EHR-Agent main.swift
cp /tmp/EHR-Agent EHR-Agent.app/Contents/MacOS/EHR-Agent
```

### Code Signing (Required)

**重要**: `swiftc` で再コンパイルしたバイナリを `.app` バンドル内に `cp` で上書きすると、macOS のコード署名が無効になります。起動時に以下のクラッシュが発生します：

```
Termination Reason: Namespace CODESIGNING, Code 2, Invalid Page
Exception Type: EXC_BAD_ACCESS (SIGKILL (Code Signature Invalid))
```

**必ず再署名してください**：

```bash
codesign --force --deep --sign - EHR-Agent.app
```

### Architecture

- Single-file Swift app (`main.swift`)
- AppKit UI with `NSViewController` + `NSTextView`
- EHR Reader feature: screen capture → scroll → VLM extraction

### Key Files

- `swift-appkit/main.swift` - Main application code
- `swift-appkit/EHR-Agent.app/` - App bundle
- `swift-appkit/captures/` - Debug screenshots
- `swift-appkit/logs/` - Execution logs and VLM request logs

### Dependencies

- macOS 14+
- Screen capture permission (requested at startup)
- Accessibility permission (requested for click/scroll simulation)
- Local VLM server at `http://localhost:8000/v1`

### Notes

- The binary inside `EHR-Agent.app/Contents/MacOS/` must be re-signed after every `cp` or modification.
- For distribution, use proper Developer ID signing: `codesign --force --deep --sign "Developer ID" EHR-Agent.app`
