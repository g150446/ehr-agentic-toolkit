# 「サマリ作成」コマンド実装プラン

## 概要

EHR-Agent アプリ（`swift-appkit/main.swift`）に「サマリ作成」コマンドを追加する。

- 既存の「デバッグ」機能（過去診療録読み取り）と「サマリ」機能（サマリ欄へのフォーカス移動）を連携
- 過去診療録データを VLM で構造化 → LLM でサマリ文生成 → サマリ本文欄に自動入力
- サマリのプロンプト構成は `hardware-agent/automation/ehr_composer.py` の `_generate_summary()` と同一

---

## 変更ファイル

`swift-appkit/main.swift` のみ

---

## 変更内容

### 1. `runEHRReader()` の戻り値を `String` に変更（line 454）

```swift
// Before
private func runEHRReader() async throws {

// After
private func runEHRReader() async throws -> String {
```

- line 754–760 のエラー時 `return` を `throw` に変更:
  ```swift
  throw NSError(domain: "EHRReader", code: 19,
      userInfo: [NSLocalizedDescriptionKey: "最終JSONのシリアライズに失敗しました"])
  ```
- 関数末尾（`postCommandTab()` の後）に `return finalJSONStr` を追加

### 2. `debugAction()` で戻り値を破棄（line 165）

```swift
// Before
try await runEHRReader()

// After
_ = try await runEHRReader()
```

### 3. `runSummaryAction()` にコンテンツ引数を追加（line 816）

```swift
// Before
private func runSummaryAction() async throws {

// After
private func runSummaryAction(content: String) async throws {
```

- line 1087 の `await pasteText("テスト")` → `await pasteText(content)`
- line 1082 のメッセージ文字列を汎用的に変更:
  `"「サマリの本文」を発見、クリックします..."` はそのまま
- line 1090–1092 の完了メッセージを変更:
  ```swift
  updateLastMessage(content: "サマリ本文欄に入力しました。")
  ```

### 4. 既存「サマリ」コマンド呼び出しを更新（line 318）

```swift
// Before
try await runSummaryAction()

// After
try await runSummaryAction(content: "テスト")
```

### 5. `callTextLLM(prompt:logger:)` ヘルパー関数を追加

`runSummaryAction` の直後（line 1097 付近）に追加。テキストのみの非ストリーミング LLM 呼び出し。

```swift
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
```

### 6. `runCreateSummaryAction()` 関数を追加

`callTextLLM` の直後に追加。

```swift
private func runCreateSummaryAction() async throws {
    let logger = EHRLogger()
    defer { logger.saveToFile() }
    logger.log("===== Create Summary Action Started =====")

    // Step 1: 診療録読み取り
    await MainActor.run {
        appendMessage(role: "assistant", content: "診療録を読み取ります...")
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

    ### 出力形式
    以下の7項目に分けて記載してください。各項目は1〜3行程度で記載し、内容が充実するよう詳細な経過・処方・指導内容を含めてください。全体でMicrosoft Wordの1ページに収まる内容にしてください。

    1. **主訴**
    2. **現病歴**
    3. **既往歴**
    4. **入院後経過**
    5. **退院時状況**
    6. **退院時方針**
    7. **退院時処方**

    ### 出力の書式
    - 必ず各行の先頭に `[項目名]` を付けてください。例: `[主訴] 呼吸困難、喘鳴`
    - 項目間は1行の空行で区切ってください。
    - 各項目の内容は連続した文章として記載し、項目内での改行は避けてください。
    - 内容が短くなりすぎないよう、診療経過の詳細（検査所見、治療反応、経過日数など）を含めてください。

    ### 制約
    - 診療録に記載されている情報のみを使用し、推測や補完は行わないでください。
    - 日付順に診療経過を整理し、簡潔に記載してください。
    """
    let summaryText = try await callTextLLM(prompt: prompt, logger: logger)

    // Step 3: サマリ欄に入力
    await MainActor.run {
        updateLastMessage(content: "サマリ生成完了。サマリ欄に入力します...")
    }
    try await runSummaryAction(content: summaryText)
}
```

### 7. `sendMessage()` に「サマリ作成」チェックを追加（line 309 の前）

**重要**: 「サマリ作成」は「サマリ」を含む文字列なので、既存の `"サマリ"` チェック（line 309）より**前**に配置する。

```swift
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

// 既存の「サマリ」チェック（そのまま）
if text.contains("サマリ") {
    ...
}
```

---

## ビルド・確認方法

```bash
cd swift-appkit
swiftc -o /tmp/EHR-Agent main.swift && \
cp /tmp/EHR-Agent EHR-Agent.app/Contents/MacOS/EHR-Agent && \
codesign --force --deep --sign - EHR-Agent.app
```

または一括スクリプト:
```bash
./scripts/build_and_run.sh
```

動作確認手順:
1. EHR システムを過去診療録が表示されている状態で開く
2. EHR-Agent アプリを起動
3. チャット欄に「サマリ作成」と入力して送信
4. 診療録読み取り → サマリ生成 → サマリ本文欄への自動入力が順に実行されることを確認
