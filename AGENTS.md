# EHR-Agentic-Toolkit - Agent Notes

## Swift AppKit (EHR-Agent)

### Build

```bash
cd swift-appkit
swiftc -target arm64-apple-macos14.0 -o /tmp/EHR-Agent main.swift
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

### TCC Permission Reset After Rebuild

アドホック署名 (`--sign -`) で再ビルドした場合、macOSはアプリを「別アプリ」と認識し、既存の権限が無効になります。手動で System Settings から削除する代わりに、以下のスクリプトで自動削除できます：

```bash
./scripts/reset_permissions.sh
```

このスクリプトは以下を行います：
- TCC データベース内の **ゴーストエントリ**（bundle ID が空または競合する古いエントリ）を検出・削除
- `Accessibility` と `ScreenCapture` の権限を `tccutil reset` でクリア

**ゴーストエントリとは**: `Info.plist` の `CFBundleIdentifier` を追加・変更する前に起動したアプリの権限情報が、TCC データベース内に「ID なしのゴースト」として残る現象です。このゴーストが新しい ID 付きのエントリと競合し、再起動時に権限が失われる原因となります。

### Build & Run Automation

ビルド、再署名、権限リセット、アプリ起動を一括で行うスクリプトも用意しています：

```bash
./scripts/build_and_run.sh
```

このスクリプトは以下を行います：
1. `swiftc` でビルド
2. `EHR-Agent.app` にバイナリをコピーして再署名
3. `~/Applications/` にアプリをインストール（Launch Services 登録のため必須）
4. Launch Services に登録 (`lsregister`)
5. 権限をリセット (`./scripts/reset_permissions.sh`)
6. アプリと System Settings を開く

**重要**: `~/Applications/` にコピーしないと、Launch Services に正しく登録されず、TCC 権限が再起動時に失われます（白紙アイコンで表示される）。

実行後、System Settings の権限画面が自動で開くので、手動で許可してください。

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
