# EHR-Agent (Swift AppKit)

macOS native AI chat application with debugging capabilities.

## Target EHR

This app targets a browser-based demo EHR. A sample patient record is available at:

https://dialog-ehr.vercel.app/patients/3

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

### 3. Reset Permissions (after rebuild)

After rebuilding, macOS treats the app as "new" and invalidates existing permissions. Run:

```bash
cd swift-appkit
./scripts/reset_permissions.sh
```

This resets Accessibility and ScreenCapture permissions via `tccutil`.

### 4. Build & Run (automated)

Use the provided script to build, re-sign, install to `~/Applications/`, register with Launch Services, reset permissions, and launch:

```bash
cd swift-appkit
./scripts/build_and_run.sh
```

**Why `~/Applications/`?**
macOS Launch Services must know about the app for TCC (permission database) to track it by bundle ID. If the app stays only in the development folder, permissions may be lost on restart. The script copies the app to `~/Applications/` and runs `lsregister` to ensure proper registration.

Then manually grant permissions in System Settings when prompted.

### 5. Run

```bash
open ~/Applications/EHR-Agent.app
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

## Coordinate Transformation (OCR → Click)

The app uses Tesseract OCR to detect UI elements on screen. Since Tesseract returns **pixel coordinates** but macOS `CGEvent` requires **point coordinates**, the following transformation is applied:

### 1. Scale Factors

```
scaleX = screenshotWidthPixels  / screenBoundsWidthPoints
scaleY = screenshotHeightPixels / screenBoundsHeightPoints
```

On Retina displays, `scaleX` and `scaleY` are typically `~2.0`.

### 2. Pixel → Point Conversion

For a detected bounding box `(x, y, w, h)` in **pixel coords**:

| Target | Calculation |
|--------|-------------|
| Center X | `x + w / 2` |
| Center Y + 1h | `y + h / 2 + h` |

These pixel values are divided by the scale factors to get **point coords**:

```
pointX = pixelX / scaleX
pointY = pixelY / scaleY
```

### 3. Screen Offset

When the target is in a **cropped sub-image** (e.g., right panel), add the crop origin offset (also in points):

```
screenX = screenBounds.origin.x + (cropOriginPixels / scaleX) + pointX
screenY = screenBounds.origin.y + pointY
```

### Example: "タイトル" → Input Area Click

1. Detect "タイトル" bounding box in right panel crop
2. Click X = `box.x + box.w / 2` (center of text)
3. Click Y = `box.y + box.h / 2 + box.h` (center + 1 height down)
4. Convert pixels → points via scale factors
5. Add right panel divider offset to X
6. Post `CGEvent` at final screen point

## Configuration

- **API Base**: `http://localhost:8000/v1`
- **API Key**: `penguin`
- **Default Model**: `gemma-4-26b-a4b-it-4bit`
