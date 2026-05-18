"""
過去カルテを読み取り、退院時サマリを生成してWordに入力するツール。

--summary オプションを指定すると:
1. 過去カルテ領域をスクロールしながら読み取る（ehr_reader --scroll 相当）
2. 読み取った内容から退院時サマリをVLMで生成
3. 退院時要約のWord文書とノートパッドを開く（ehr_reader --summary 相当）
4. ノートパッドで1行ずつIME変換し、Ctrl+A→Ctrl+X→Alt+Tab→Ctrl+V→Alt+TabでWordに貼り付け

--summary-no-scroll オプションを指定すると:
- Phase 1（過去カルテ読み取り）と Phase 2（サマリ生成）をスキップ
- 固定デバッグテキストを使用して Phase 3〜4 のみ実行
- ノートパッド最大化のデバッグに使用

実行方法:
  python -m automation.ehr_composer --summary
  python -m automation.ehr_composer --summary --omlx gemma-4-26b-a4b-it-4bit
  python -m automation.ehr_composer --summary-no-scroll
  python -m automation.ehr_composer --summary-no-scroll --movie
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from automation.config import load_config
from automation.screen_analyzer import capture_screen as _capture_screen_hdmi
from automation.mlx_vlm_ime import MLX_VLM_IME_TIMEOUT
from automation.ehr_input import type_japanese_sentence, _configure_runtime
import automation.ehr_input as _ehr_input_mod
from automation.video_recorder import VideoRecorder
from automation.ehr_reader import (
    _detect_all_dividers,
    _extract_past_chart_region,
    _extract_ocr_text,
    _find_letter_icon,
    _click_letter_icon,
    _find_text_position_ocr,
    _find_target_y_by_ocr,
    _find_word_return_mark_bottom,
    _find_word_return_mark_x,
    _is_frame_unchanged,
    _parse_vlm_response,
    _read_past_chart_with_vlm,
    _read_past_chart_with_vlm_merge,
    _save_debug_frame,
    _scroll_past_chart_down,
    _wait_for_ble_connected,
    _build_runtime_config,
    _OMLX_DEFAULT_MODEL,
)

_CAPTURES_DIR = Path(__file__).resolve().parent.parent / "captures"
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class _TeeStream:
    """Mirror writes to both the terminal stream and the run log file."""

    def __init__(self, console_stream, log_stream) -> None:
        self._console_stream = console_stream
        self._log_stream = log_stream

    def write(self, data: str) -> int:
        self._console_stream.write(data)
        self._log_stream.write(data)
        return len(data)

    def flush(self) -> None:
        self._console_stream.flush()
        self._log_stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._console_stream, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self._console_stream, "encoding", "utf-8")


@contextmanager
def _capture_run_output():
    """Tee stdout/stderr to a timestamped per-run log file."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _LOGS_DIR / f"composer_{ts}.txt"
    with log_path.open("w", encoding="utf-8") as f:
        tee_stdout = _TeeStream(sys.stdout, f)
        tee_stderr = _TeeStream(sys.stderr, f)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print(f"=== ehr_composer run log: {log_path.name} ===")
            yield log_path


def _save_summary(summary_text: str) -> Path:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _LOGS_DIR / f"summary_{ts}.txt"
    path.write_text(summary_text, encoding="utf-8")
    print(f"[INFO] サマリを保存しました: {path.name}")
    return path


def _load_latest_summary() -> tuple[str, str]:
    """最新のサマリファイルを読み込み (テキスト, ファイル名) を返す。"""
    files = sorted(_LOGS_DIR.glob("summary_*.txt"), key=lambda p: p.name, reverse=True)
    if not files:
        raise FileNotFoundError(f"保存されたサマリが見つかりません: {_LOGS_DIR}")
    path = files[0]
    print(f"[INFO] 保存済みサマリを読み込みます: {path.name}")
    return path.read_text(encoding="utf-8"), path.name


_RESUME_STATE_PATH = _LOGS_DIR / "resume_state.json"


def _save_resume_state(
    start_chunk: int,
    current_mode: Optional[str],
    summary_file: str,
    total_chunks: int,
) -> None:
    import json as _json
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "start_chunk": start_chunk,
        "current_mode": current_mode,
        "summary_file": summary_file,
        "total_chunks": total_chunks,
    }
    _RESUME_STATE_PATH.write_text(_json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 再開ステートを保存しました: チャンク {start_chunk+1}/{total_chunks} から再開可能")


def _load_resume_state() -> dict:
    import json as _json
    if not _RESUME_STATE_PATH.exists():
        raise FileNotFoundError(f"再開ステートが見つかりません: {_RESUME_STATE_PATH}")
    return _json.loads(_RESUME_STATE_PATH.read_text(encoding="utf-8"))


def _clear_resume_state() -> None:
    if _RESUME_STATE_PATH.exists():
        _RESUME_STATE_PATH.unlink()
        print("[INFO] 再開ステートをクリアしました（正常完了）")


_MOVIE_DIR = _CAPTURES_DIR / "movie"

# --summary-no-scroll 用の固定デバッグテキスト
_DEBUG_SUMMARY_TEXT = (
    "[主訴] 呼吸困難、喘鳴\n\n"
    "[現病歴] 2日前からの感冒症状に続き、昨晩から咳嗽が悪化し、"
    "今朝より吸気時の呼吸困難が出現したため救急外来を受診した。"
    "咳は痰を伴う乾性咳嗽で、夜間に悪化する傾向があった。"
    "発熱は38.5℃まで上昇し、全身倦怠感も認めた。\n\n"
    "[既往歴] 気管支喘息（小児期発症、35歳頃再発）、アレルギー性鼻炎。"
    "30歳時に花粉症と診断され、以降春〜秋にかけて鼻症状が出現。"
    "50歳時に職場のストレスを契機に喘息発作が再燃し、"
    "吸入ステロイド（フルティフォーム）を使用開始。\n\n"
    "[入院後経過] 上気道感染を契機とした気管支喘息の急性増悪（大発作）に対し、"
    "入院の上で酸素療法（ nasal cannula 2L/min ）および全身性ステロイド静注、"
    "頻回な気管支拡張薬（サルブタモール）の吸入を開始した。"
    "治療開始後2時間で呼吸苦は軽減し、SpO2は92%から96%に改善。"
    "入院2日目には歩行可能となり、聴診で喘鳴は減少。"
    "入院3日目には吸入薬のみで管理可能となり、経口ステロイドに切り替え。\n\n"
    "[退院時状況] 自覚症状および聴診上の喘鳴は完全に消失し、"
    "酸素投与なしでも良好な酸素化（SpO2 97% room air）を維持できている。"
    "歩行時の呼吸困難はなく、日常生活動作は自立可能。"
    "吸入手技の確認を行い、コントローラー継続の重要性について患者の理解を得た。"
    "アレルゲン回避（ダニ、花粉）についても指導済み。\n\n"
    "[退院時方針] 退院後はプレドニンを漸減スケジュールにて内服し、"
    "継続中の吸入薬（コントローラー）を適切に使用する。"
    "1週間後に一般内科外来を受診予定。"
    "喘息日誌の記録を開始し、ピークフロー値の自己管理を指導。"
    "症状悪化時は速やかに受診することを説明。\n\n"
    "[退院時処方] プレドニン錠5mg 6錠（30mg）分2朝夕食後 3日分、"
    "シムビコートタービュヘイラー 1回2吸入 1日2回、"
    "サルタノール吸入液 1回1吸入 1日4回（必要時）"
)


def _save_debug_frame_local(frame, name: str) -> str:
    """デバッグ用フレームを captures/ に保存する（composer専用プレフィックス）。"""
    _CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _CAPTURES_DIR / f"composer_{name}_{ts}.png"
    cv2.imwrite(str(path), frame)
    print(f"  [debug] 保存: {path}")
    return str(path)


# --movie 用のグローバル録画インスタンス（モンキーパッチでフレームを横流し）
_recorder: Optional[VideoRecorder] = None
_original_capture = _capture_screen_hdmi
_config = None  # main() で設定される（_maybe_capture_frame 用）


def _capture_with_recording(*args, **kwargs):
    """capture_screen のラッパー：戻り値フレームを VideoRecorder にも渡す。"""
    frame = _original_capture(*args, **kwargs)
    if frame is not None and _recorder is not None and _recorder.is_recording():
        _recorder.write(frame)
    return frame


def _maybe_capture_frame() -> None:
    """--movie モード中のみ現在画面を1フレームキャプチャしてレコーダーに記録する。"""
    if _recorder is None or not _recorder.is_recording() or _config is None:
        return
    _capture_screen_hdmi(
        device_index=_config.capture_device_index,
        width=_config.capture_width,
        height=_config.capture_height,
    )


def _generate_summary(
    chart_data: list[dict],
    *,
    model: str,
    url: str,
    api_key: str,
    timeout: float,
) -> str:
    """過去カルテJSONから退院時サマリを生成する。"""
    chart_json = json.dumps(chart_data, ensure_ascii=False, indent=2)

    prompt = (
        "### 指示\n"
        "以下の過去診療録データを元に、退院時サマリを作成してください。\n\n"
        "### 過去診療録データ\n"
        "```json\n"
        f"{chart_json}\n"
        "```\n\n"
        "### 重要：日付の扱い\n"
        "- 診療録データの**最初の `date`** を入院日として扱ってください。\n"
        "- 診療録データの**最後の `date`** を退院日として扱ってください。\n"
        "- サマリ内では「本日」「今日」「現在」などの相対的な表現を**一切使わず**、必ず具体的な日付（例：YYYY年MM月DD日）を記載してください。\n"
        "  - 悪い例：「本日退院の運びとなった」「本日午前中に退院とし」\n"
        "  - 良い例：「YYYY年MM月DD日に退院となった」「YYYY年MM月DD日午前中に退院とし」\n\n"
        "### 出力形式\n"
        "以下の7項目に分けて記載してください。各項目の内容が充実するよう詳細な経過・処方・指導内容を含めてください。"
        "全体でMicrosoft Wordの1〜2ページに収まる内容にしてください。\n\n"
        "1. **主訴**\n"
        "2. **現病歴**（発症から入院日（YYYY年MM月DD日）までの経過のみを記載すること。入院後の治療経過・退院に関する内容は書かず、「入院後経過」に委ねること）\n"
        "3. **既往歴**\n"
        "4. **入院後経過**（入院後の治療経過を詳細に記載。検査所見・検査値、投薬内容・薬剤名・用量、処置内容、治療反応を含めること）\n"
        "5. **退院時状況**\n"
        "6. **退院時方針**（退院日を具体的な日付で明記すること）\n"
        "7. **退院時処方**\n\n"
        "### 出力の書式\n"
        "- 必ず各行の先頭に `[項目名]` を付けてください。例: `[主訴] 呼吸困難、喘鳴`\n"
        "- 項目間は1行の空行で区切ってください。\n"
        "- 各項目の内容は連続した文章として記載し、項目内での改行は避けてください。\n"
        "- 内容が短くなりすぎないよう、検査値・薬剤名・用量・治療反応などの詳細を漏らさず記載してください。\n\n"
        "### 制約\n"
        "- 診療録に記載されている情報のみを使用し、推測や補完は行わないでください。\n"
        "- 日付順に診療経過を整理し、簡潔に記載してください。\n"
        "- 「本日」「今日」「現在」などの相対的な時間表現は絶対に使用しないでください。"
    )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "max_tokens": 2048,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())

    return result["choices"][0]["message"]["content"]


def _open_word_and_notepad(frame, dividers: list[int], config) -> None:
    """退院時要約のWord文書とノートパッドを開く（ehr_reader --summary 相当）。"""
    print("\nletter_icon.png を画面全体から検索中...")
    letter_pos = _find_letter_icon(frame, threshold=0.7)
    if letter_pos is None:
        raise RuntimeError("letter_icon.png が画面内に見つかりませんでした")
    click_x, click_y, _ = letter_pos
    print(f"letter_icon クリック: ({click_x}, {click_y})")
    _click_letter_icon(click_x, click_y)

    # ポップアップの位置を幾何学的に計算
    panel2_width = dividers[2] - dividers[1]
    popup_width = panel2_width // 3
    target_x = dividers[2] - popup_width // 2
    target_y = click_y

    print(f"ポップアップ上段メニュー中心を計算: ({target_x}, {target_y})")

    client = _wait_for_ble_connected()
    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")
    ok = client.move_mouse_to_position(target_x, target_y)
    print(f"moveto ({target_x}, {target_y}) -> {'OK' if ok else 'NG'}")

    # 1秒待機後、新しいポップアップが表示されるのを待つ
    print("1.0秒待機（新ポップアップ表示待ち）...")
    time.sleep(1.0)

    # パネル2中央へ移動
    panel2_center_x = dividers[1] + panel2_width // 2
    panel2_center_y = target_y
    print(f"パネル2中央へ移動: ({panel2_center_x}, {panel2_center_y})")
    ok = client.move_mouse_to_position(panel2_center_x, panel2_center_y)
    print(f"moveto ({panel2_center_x}, {panel2_center_y}) -> {'OK' if ok else 'NG'}")

    # 0.5秒待機後、ポップアップをOCRで認識
    print("0.5秒待機（ポップアップ安定）...")
    time.sleep(0.5)

    # クリック後の画面を再キャプチャ
    print("ポップアップ表示後の画面をキャプチャ中...")
    popup_frame = _capture_screen_hdmi(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if popup_frame is None:
        raise RuntimeError("ポップアップ表示後のキャプチャに失敗しました")
    _save_debug_frame_local(popup_frame, "full_screen_with_popup")

    # パネル2を切り出し
    x_start = dividers[1]
    x_end = dividers[2]
    panel2 = popup_frame[:, x_start:x_end]

    print("\nパネル2をOCRで認識中...")
    text_pos = _find_text_position_ocr(panel2, x_offset=x_start, search_text="退院時要約")
    if text_pos is None:
        raise RuntimeError("'退院時要約' が見つかりませんでした")

    text_x, text_y = text_pos
    print(f"'退院時要約' へカーソル移動: ({text_x}, {text_y})")
    ok = client.move_mouse_to_position(text_x, text_y)
    print(f"moveto ({text_x}, {text_y}) -> {'OK' if ok else 'NG'}")

    # 0.5秒待機後にクリック
    print("0.5秒待機...")
    time.sleep(0.5)
    ok = client.click()
    print(f"click (退院時要約) -> {'OK' if ok else 'NG'}")

    # 8.0秒待機（Microsoft Word が開くのを待つ）
    print("8.0秒待機（Word起動待ち）...")
    time.sleep(8.0)

    # キーボードモードに切り替えて Alt+Tab
    ok = client.switch_to_keyboard_mode()
    print(f"mode:keyboard -> {'OK' if ok else 'NG'}")
    ok = client.alt_tab()
    print(f"alt_tab -> {'OK' if ok else 'NG'}")

    # マウスモードに戻す
    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")

    # Word画面を再キャプチャ
    print("\nWord画面をキャプチャ中...")
    word_frame = _capture_screen_hdmi(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if word_frame is None:
        raise RuntimeError("Word画面のキャプチャに失敗しました")
    _save_debug_frame_local(word_frame, "word_screen")

    # OCRで "担当医" または "医師名" を検索
    print("Word画面をOCRで認識中...")
    target_y_coord = _find_target_y_by_ocr(word_frame, keywords=["担当医", "医師名"])
    if target_y_coord is None:
        raise RuntimeError("'担当医'/'医師名' が見つかりませんでした")

    # テンプレートマッチングで word_return_mark の x 座標を検出
    print("word_return_mark.jpg を検索中...")
    target_x_coord = _find_word_return_mark_x(word_frame, threshold=0.7)
    if target_x_coord is None:
        raise RuntimeError("'word_return_mark.jpg' が見つかりませんでした")

    print(f"最終カーソル位置: ({target_x_coord}, {target_y_coord})")
    ok = client.move_mouse_to_position(target_x_coord, target_y_coord)
    print(f"moveto ({target_x_coord}, {target_y_coord}) -> {'OK' if ok else 'NG'}")

    # 最も下の word_return_mark を検索（ドラッグ先）
    print("\n最下 word_return_mark を検索中...")
    bottom_pos = _find_word_return_mark_bottom(
        word_frame, screen_width=config.capture_width, threshold=0.7
    )
    if bottom_pos is None:
        raise RuntimeError("ドラッグ先が見つかりませんでした")

    bottom_x, bottom_y = bottom_pos
    print(f"ドラッグ先: ({bottom_x}, {bottom_y})")

    # ドラッグ実行
    ok = client.mouse_down()
    print(f"mouse_down -> {'OK' if ok else 'NG'}")
    time.sleep(0.1)

    ok = client.move_mouse_to_position(bottom_x, bottom_y)
    print(f"moveto ({bottom_x}, {bottom_y}) -> {'OK' if ok else 'NG'}")
    time.sleep(0.1)

    # 2秒待機（ドラッグ範囲を目視確認）
    print("2.0秒待機（ドラッグ範囲確認）...")
    time.sleep(2.0)
    _maybe_capture_frame()  # Wordのドラッグ選択範囲

    # 先にマウスボタンを離す
    ok = client.mouse_up()
    print(f"mouse_up -> {'OK' if ok else 'NG'}")

    # キーボードモードに切り替えて backspace
    ok = client.switch_to_keyboard_mode()
    print(f"mode:keyboard -> {'OK' if ok else 'NG'}")
    ok = client.press_key("backspace")
    print(f"press_key(backspace) -> {'OK' if ok else 'NG'}")

    # Ctrl+L (左寄せ)
    ok = client.press_key("ctrl_l")
    print(f"press_key(ctrl_l) -> {'OK' if ok else 'NG'}")
    _maybe_capture_frame()  # テンプレート削除・左寄せ後のWord

    # Windows キー
    ok = client.press_key("win")
    print(f"press_key(win) -> {'OK' if ok else 'NG'}")

    # 1.0秒待機
    print("1.0秒待機...")
    time.sleep(1.0)
    _maybe_capture_frame()  # Windowsスタートメニュー

    # "note" テキスト入力
    ok = client.type_text("note")
    print(f"type_text(note) -> {'OK' if ok else 'NG'}")

    # 0.5秒待機
    print("0.5秒待機...")
    time.sleep(0.5)

    # Enter 送信
    ok = client.press_key("enter")
    print(f"press_key(enter) -> {'OK' if ok else 'NG'}")

    # 3.0秒待機（Notepad起動・前面化を確実に待つ）
    print("3.0秒待機（Notepad起動・前面化待ち）...")
    time.sleep(3.0)
    _maybe_capture_frame()  # Notepad起動直後

    # スクリーン中央をクリックしてフォーカス
    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")
    center_x = config.capture_width // 2
    center_y = config.capture_height // 2
    ok = client.move_mouse_to_position(center_x, center_y)
    print(f"moveto ({center_x}, {center_y}) -> {'OK' if ok else 'NG'}")
    ok = client.click()
    print(f"click (focus) -> {'OK' if ok else 'NG'}")

    # 0.5秒待機後、ウィンドウ最大化 (Win+Up)
    print("0.5秒待機...")
    time.sleep(0.5)
    ok = client.switch_to_keyboard_mode()
    print(f"mode:keyboard -> {'OK' if ok else 'NG'}")
    ok = client.press_key("win_up")
    print(f"press_key(win_up) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)
    _maybe_capture_frame()  # Notepad最大化後


def _split_summary_chunks(summary_text: str) -> list[tuple[str, bool]]:
    """サマリを改行・読点・句点で分割し (チャンク文字列, 改行フラグ) を返す。

    改行フラグ=True はWordへの貼り付け後にEnterを押すことを意味する。
    読点・句点で分割されたチャンク（行途中）は False、行末チャンクは True。
    句点は分割後のチャンク末尾に付加する。
    """
    import re as _re
    chunks: list[tuple[str, bool]] = []
    for line in summary_text.strip().splitlines():
        if not line.strip():
            continue
        # 読点・句点の直後で分割し、区切り文字はそのチャンクの末尾に残す
        parts = _re.split(r'(?<=。)|(?<=、)', line)
        parts = [p for p in parts if p]
        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)
            if part.strip():
                chunks.append((part, is_last))
    return chunks


def _type_line_and_paste(
    line: str,
    is_first_line: bool,
    _current_mode: Optional[str] = None,
    press_enter: bool = True,
) -> Optional[str]:
    """ノートパッドに1チャンク入力し、切り取ってWordに貼り付ける。

    Args:
        line: 入力するテキスト（改行・読点で分割済みのチャンク）
        is_first_line: True の場合、最初のチャンクとしてクリアを実行
        _current_mode: 前のチャンクから引き継いだ IME モード。None の場合は内部で検出する。
        press_enter: True の場合、Wordへの貼り付け後にEnterを押す（改行位置のみ True）

    Returns:
        入力後の IME モード。次のチャンクに渡すことで IME 検出を省略できる。
    """
    print(f"\n--- 行入力 ({'最初' if is_first_line else '追記'}) ---")
    print(f"内容: {line!r}")

    # ノートパッドに入力（clear_field は最初の行のみ）
    current_mode = type_japanese_sentence(line + "\n", clear_field=is_first_line, _current_mode=_current_mode)

    # 入力完了後の待機（IME確定を待つ）
    time.sleep(0.5)

    client = _wait_for_ble_connected()

    # Ctrl+A で全選択
    ok = client.switch_to_keyboard_mode()
    print(f"mode:keyboard -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)
    ok = client.press_key("ctrl_a")
    print(f"press_key(ctrl_a) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)

    # Ctrl+X で切り取り
    ok = client.press_key("ctrl_x")
    print(f"press_key(ctrl_x) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)

    # Alt+Tab で Word に切替
    ok = client.alt_tab()
    print(f"alt_tab (to Word) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)
    _maybe_capture_frame()  # Word画面

    # Ctrl+V で貼り付け
    ok = client.press_key("ctrl_v")
    print(f"press_key(ctrl_v) -> {'OK' if ok else 'NG'}")
    time.sleep(1.0)
    _maybe_capture_frame()  # Wordへの貼り付け後

    # 貼り付け後の改行（元の文章の改行位置のみ）
    if press_enter:
        ok = client.press_key("enter")
        print(f"press_key(enter) -> {'OK' if ok else 'NG'}")
        time.sleep(0.5)

    # Alt+Tab でノートパッドに戻る
    ok = client.alt_tab()
    print(f"alt_tab (to Notepad) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)
    _maybe_capture_frame()  # Notepad（次行入力前）

    return current_mode


def _read_past_chart_scroll(config, runtime: dict[str, str]) -> list[dict]:
    """過去カルテ領域をスクロールしながら読み取る。"""
    print("HDMIデバイスからキャプチャ中...")
    frame = _capture_screen_hdmi(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    _save_debug_frame_local(frame, "full_screen")
    print("画面を解析中...")

    dividers = _detect_all_dividers(frame, debug=True)
    if dividers is None:
        raise RuntimeError("患者カルテ画面の区切り線を検出できませんでした")

    print(f"区切り線検出: x={dividers}")

    print("\n過去カルテ領域を切り出し中...")
    try:
        past_chart = _extract_past_chart_region(frame, dividers, debug=True)
    except RuntimeError as exc:
        raise RuntimeError(str(exc))
    print(f"切り出しサイズ: {past_chart.shape[1]}x{past_chart.shape[0]} px")

    print("\nOCR でテキスト抽出中...")
    ocr_text = _extract_ocr_text(past_chart)
    print(f"OCR抽出完了 ({len(ocr_text)} 文字)")

    print("\nVLM で過去カルテの内容を読み取り中...")
    raw_response = _read_past_chart_with_vlm(
        past_chart,
        ocr_text,
        model=runtime["model"],
        url=runtime["url"],
        api_key=runtime["api_key"],
        timeout=MLX_VLM_IME_TIMEOUT,
    )

    print(f"\n--- VLM 生応答 ---\n{raw_response}\n--- 終了 ---\n")

    try:
        structured = _parse_vlm_response(raw_response)
    except ValueError as exc:
        raise RuntimeError(str(exc))

    print("--- 構造化出力 (JSON) ---")
    print(json.dumps(structured, ensure_ascii=False, indent=2))
    print("--- 終了 ---")

    if structured:
        print(f"\n過去カルテ {len(structured)} 件を読み取りました。")
    else:
        print("\n過去カルテにテキストが見つかりませんでした。")

    # スクロール読み取り
    prev_frame = frame.copy()
    iteration = 0
    max_iterations = 20

    while True:
        iteration += 1
        if iteration > max_iterations:
            print("\n安全上限(20セット)に達しました。自動終了します。")
            break

        print(f"\n[セット {iteration}] 過去カルテ領域をスクロール中...")
        _scroll_past_chart_down(
            dividers=dividers,
            screen_width=config.capture_width,
            screen_height=config.capture_height,
            scroll_count=2,
        )

        # スクロール後に再キャプチャ
        print("スクロール後の画面をキャプチャ中...")
        frame = _capture_screen_hdmi(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height,
        )
        if frame is None:
            print("[WARNING] 再キャプチャに失敗しました。現在の結果で終了します。", file=sys.stderr)
            break

        print("画面変化を確認中...")
        if _is_frame_unchanged(prev_frame, frame):
            print("スクロール後の画面が変化しませんでした。自動終了します。")
            break
        prev_frame = frame.copy()

        _save_debug_frame_local(frame, f"full_screen_scroll_{iteration}")
        print("画面を解析中...")

        dividers = _detect_all_dividers(frame, debug=True)
        if dividers is None:
            print("[ERROR] スクロール後の画面で区切り線を検出できませんでした", file=sys.stderr)
            break

        print(f"区切り線検出: x={dividers}")
        try:
            past_chart = _extract_past_chart_region(frame, dividers, debug=True)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            break

        print("\nOCR でテキスト抽出中...")
        ocr_text = _extract_ocr_text(past_chart)
        print(f"OCR抽出完了 ({len(ocr_text)} 文字)")

        print(f"\n[セット {iteration}] VLM で統合読み取り中...")
        raw_response = _read_past_chart_with_vlm_merge(
            past_chart,
            structured,
            ocr_text,
            model=runtime["model"],
            url=runtime["url"],
            api_key=runtime["api_key"],
            timeout=MLX_VLM_IME_TIMEOUT,
        )

        print(f"\n--- VLM 生応答 (セット {iteration}) ---\n{raw_response}\n--- 終了 ---\n")

        try:
            structured = _parse_vlm_response(raw_response)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            break

        print("--- 構造化出力 (JSON) ---")
        print(json.dumps(structured, ensure_ascii=False, indent=2))
        print("--- 終了 ---")

        if structured:
            print(f"\n過去カルテ {len(structured)} 件を読み取りました。")
        else:
            print("\n過去カルテにテキストが見つかりませんでした。")

    return structured


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv

    # --summary / --summary-no-scroll / --continue チェック
    do_summary = "--summary" in args
    do_summary_no_scroll = "--summary-no-scroll" in args
    do_continue = "--continue" in args

    if not do_summary and not do_summary_no_scroll and not do_continue:
        print("[ERROR] --summary, --summary-no-scroll, または --continue オプションが必要です", file=sys.stderr)
        print("使用例: python -m automation.ehr_composer --summary", file=sys.stderr)
        print("       python -m automation.ehr_composer --summary-no-scroll", file=sys.stderr)
        print("       python -m automation.ehr_composer --continue", file=sys.stderr)
        print("       python -m automation.ehr_composer --summary --omlx gemma-4-26b-a4b-it-4bit", file=sys.stderr)
        return 1

    do_movie = "--movie" in args

    # --omlx が指定されていなくても自動的に有効化
    omlx_model: Optional[str] = None
    if "--omlx" in args:
        idx = args.index("--omlx")
        if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
            omlx_model = args[idx + 1]
        else:
            omlx_model = _OMLX_DEFAULT_MODEL
    elif any(arg.startswith("--omlx=") for arg in args):
        for arg in args:
            if arg.startswith("--omlx="):
                _, _, omlx_model = arg.partition("=")
                if not omlx_model:
                    omlx_model = _OMLX_DEFAULT_MODEL
                break
    else:
        print("[INFO] --omlx が指定されていないため、自動的に有効化します")
        omlx_model = _OMLX_DEFAULT_MODEL

    runtime = _build_runtime_config(omlx_model=omlx_model)
    _configure_runtime(omlx=True, omlx_model=omlx_model)
    print(f"VLM ランタイム: {runtime['model']} ({runtime['url']})")

    config = load_config(skip_password=True)
    global _config
    _config = config

    # --movie 指定時：モンキーパッチで capture_screen をラップ
    movie_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if do_movie:
        global _recorder
        _recorder = VideoRecorder(
            width=config.capture_width,
            height=config.capture_height,
            fps=10,
        )
        globals()["_capture_screen_hdmi"] = _capture_with_recording
        _ehr_input_mod._capture_screen_hdmi = _capture_with_recording
        print("[movie] HDMIキャプチャ動画記録を有効化しました")

    with _capture_run_output():
        summary_fname: str = ""
        start_chunk: int = 0
        current_mode: Optional[str] = None

        if do_continue:
            # --continue: resume_state.json を読んで中断チャンクから再開
            print("\n========== 再開モード（--continue） ==========")
            try:
                state = _load_resume_state()
            except FileNotFoundError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                return 1
            start_chunk = state["start_chunk"]
            current_mode = state.get("current_mode")
            summary_fname = state["summary_file"]
            summary_text = (_LOGS_DIR / summary_fname).read_text(encoding="utf-8")
            print(f"[INFO] サマリ: {summary_fname}")
            print(f"[INFO] チャンク {start_chunk+1}/{state['total_chunks']} から再開します")
            print(f"\n--- 再開サマリ ---\n{summary_text}\n--- 終了 ---\n")

        elif do_summary_no_scroll:
            # --summary-no-scroll: Phase 1 と Phase 2 をスキップし、保存済みサマリを使用
            print("\n========== Phase 1 & 2: スキップ（--summary-no-scroll） ==========")
            try:
                summary_text, summary_fname = _load_latest_summary()
            except FileNotFoundError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                return 1
            print(f"\n--- 保存済みサマリ ---\n{summary_text}\n--- 終了 ---\n")

        else:
            # Phase 1: 過去カルテ読み取り
            print("\n========== Phase 1: 過去カルテ読み取り ==========")
            if do_movie:
                _recorder.start_phase(_MOVIE_DIR / f"composer_scroll_{movie_ts}.mp4", frame_skip=3)
            try:
                chart_data = _read_past_chart_scroll(config, runtime)
            except RuntimeError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                return 1
            finally:
                if do_movie:
                    _recorder.stop_phase()

            if not chart_data:
                print("[ERROR] 過去カルテからデータを読み取れませんでした", file=sys.stderr)
                return 1

            # Phase 2: サマリ生成（画面変化なし → 録画しない）
            print("\n========== Phase 2: 退院時サマリ生成 ==========")
            try:
                summary_text = _generate_summary(
                    chart_data,
                    model=runtime["model"],
                    url=runtime["url"],
                    api_key=runtime["api_key"],
                    timeout=MLX_VLM_IME_TIMEOUT,
                )
            except Exception as exc:
                print(f"[ERROR] サマリ生成に失敗しました: {exc}", file=sys.stderr)
                return 1

            print(f"\n--- 生成されたサマリ ---\n{summary_text}\n--- 終了 ---\n")
            summary_fname = _save_summary(summary_text).name

        # サマリを改行・読点で分割
        summary_chunks = _split_summary_chunks(summary_text)
        if not summary_chunks:
            print("[ERROR] サマリが空です", file=sys.stderr)
            return 1

        print(f"サマリを {len(summary_chunks)} チャンクに分割しました。")

        # Phase 3: Word / ノートパッドを開く（--continue 時はスキップ）
        if not do_continue:
            print("\n========== Phase 3: Word・ノートパッド起動 ==========")
            if do_movie:
                _recorder.start_phase(_MOVIE_DIR / f"composer_other_{movie_ts}.mp4", frame_skip=1)
            frame = _capture_screen_hdmi(
                device_index=config.capture_device_index,
                width=config.capture_width,
                height=config.capture_height,
            )
            if frame is None:
                print("[ERROR] 画面キャプチャに失敗しました", file=sys.stderr)
                return 1

            dividers = _detect_all_dividers(frame, debug=True)
            if dividers is None:
                print("[ERROR] 区切り線を検出できませんでした", file=sys.stderr)
                return 1

            try:
                _open_word_and_notepad(frame, dividers, config)
            except RuntimeError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                return 1
            finally:
                if do_movie:
                    _recorder.stop_phase()

        # Phase 4: 1行ずつ入力（IME モードを継承）
        print("\n========== Phase 4: サマリ入力 ==========")
        if start_chunk > 0:
            print(f"[INFO] チャンク {start_chunk+1}/{len(summary_chunks)} から再開します")
        if do_movie:
            _recorder.start_phase(_MOVIE_DIR / f"composer_input_{movie_ts}.mp4", frame_skip=2)
        try:
            for i, (chunk, press_enter) in enumerate(summary_chunks):
                if i < start_chunk:
                    continue
                try:
                    current_mode = _type_line_and_paste(
                        chunk,
                        is_first_line=(i == 0 and start_chunk == 0),
                        _current_mode=current_mode,
                        press_enter=press_enter,
                    )
                except Exception as exc:
                    print(f"[ERROR] チャンク入力に失敗しました ({i+1}/{len(summary_chunks)}): {exc}", file=sys.stderr)
                    _save_resume_state(i, current_mode, summary_fname, len(summary_chunks))
                    return 1
        finally:
            if do_movie:
                _recorder.stop_phase()

        print("\n========== 完了 ==========")
        _clear_resume_state()
        if do_movie:
            print(f"[movie] 動画を {_MOVIE_DIR} に保存しました")
        return 0


if __name__ == "__main__":
    sys.exit(main())
