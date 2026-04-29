# EHR-Agent (Swift AppKit)

macOS native AI chat application with debugging capabilities.

## Setup

### 1. Build

```bash
swiftc -o swift-appkit/EHR-Agent swift-appkit/main.swift
mkdir -p swift-appkit/EHR-Agent.app/Contents/MacOS
mv swift-appkit/EHR-Agent swift-appkit/EHR-Agent.app/Contents/MacOS/
codesign --force --deep --sign - swift-appkit/EHR-Agent.app
```

### 2. Grant Accessibility Permission

**Required for mouse click and keyboard event simulation.**

1. Open **System Settings** > **Privacy & Security** > **Accessibility**
2. Click the **+** button to add an application
3. Navigate to and select `EHR-Agent.app`
4. Toggle the switch to **enable** accessibility access

Without this permission, `CGEvent` mouse clicks and keyboard shortcuts will not be recognized by the system.

### 3. Run

```bash
open swift-appkit/EHR-Agent.app
```

## Debug Mode

- Press **Command+D** to toggle Debug Mode
- The Debug button appears in the input area
- Clicking Debug performs:
  1. Waits 3 seconds (position your cursor)
  2. Simulates a mouse click at the current cursor position
  3. Waits 0.5 seconds
  4. Captures a screenshot of the active window
  5. Saves to `swift-appkit/captures/debug_YYYYMMDD_HHMMSS.png`

## Configuration

- **API Base**: `http://localhost:8000/v1`
- **API Key**: `penguin`
- **Default Model**: `gemma-4-26b-a4b-it-4bit`
