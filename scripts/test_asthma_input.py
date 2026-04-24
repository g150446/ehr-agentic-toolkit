#!/usr/bin/env python3
"""asthma_*.txt の原文を行単位で保持しながら ehr_input でテストする。

前提条件:
  - BLE サーバーが起動・接続済み
  - EHR の患者記録画面が開いている

実行方法:
  python scripts/test_asthma_input.py
  python scripts/test_asthma_input.py --record asthma_2
  python scripts/test_asthma_input.py --record 3
  python scripts/test_asthma_input.py --start 3        # 3番目から再開
  python scripts/test_asthma_input.py --fragment 3     # 3番目だけ実行
  python scripts/test_asthma_input.py --fireworks accounts/fireworks/models/gemma-4-26b-a4b-it
  python scripts/test_asthma_input.py --novita
  python scripts/test_asthma_input.py --novita google/gemma-4-31b-it
  python scripts/test_asthma_input.py --openrouter google/gemma-4-26b-a4b-it
  python scripts/test_asthma_input.py --google-ai-studio
  python scripts/test_asthma_input.py --dry-run        # 行と Enter の並びを表示
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_PATIENT_RECORDS_DIR = _PROJECT_ROOT / "data" / "patient_records"
_DEFAULT_RECORD = "asthma_1"
_DEFAULT_NOVITA_MODEL = "google/gemma-4-31b-it"
_SUPPORTED_RECORDS = ("asthma_1", "asthma_2", "asthma_3")


def _resolve_python_executable() -> str:
    venv_candidates = [
        _PROJECT_ROOT / "venv" / "bin" / "python",
        _PROJECT_ROOT / "venv" / "Scripts" / "python.exe",
    ]
    for candidate in venv_candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


_PYTHON = _resolve_python_executable()


def _resolve_record_path(record: str) -> Path:
    normalized = record.strip()
    if normalized in {"1", "2", "3"}:
        normalized = f"asthma_{normalized}"
    if normalized.endswith(".txt"):
        normalized = normalized[:-4]
    if normalized not in _SUPPORTED_RECORDS:
        supported = ", ".join(_SUPPORTED_RECORDS)
        raise ValueError(f"--record は {supported} のいずれかを指定してください")
    return _PATIENT_RECORDS_DIR / f"{normalized}.txt"


def _build_fragments(path: Path) -> list[str]:
    """指定テキストを行構造ごとに保持したフラグメントリストを返す。

    各行の本文は元テキストどおり保持し、行末改行は ``"\\n"`` マーカーとして
    別フラグメントにする。これにより、``[S]`` / ``#`` 見出しや句読点を含めて、
    原文の並びを可能な限りそのまま ``ehr_input`` へ渡せるようにする。
    """
    text = path.read_text(encoding="utf-8")
    fragments: list[str] = []
    for raw_line in text.splitlines(keepends=True):
        has_newline = raw_line.endswith(("\n", "\r"))
        line = raw_line.removesuffix("\n").removesuffix("\r")
        if line:
            fragments.append(line)
        if has_newline:
            fragments.append("\n")
    return fragments


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--record",
        default=_DEFAULT_RECORD,
        metavar="NAME",
        help="対象カルテ名（asthma_1 / asthma_2 / asthma_3、または 1 / 2 / 3）",
    )
    parser.add_argument("--start", type=int, default=None, metavar="N", help="N番目のフラグメントから開始（1始まり）")
    parser.add_argument("--end", type=int, default=None, metavar="N", help="N番目のフラグメントで終了（1始まり、含む）")
    parser.add_argument("--fragment", type=int, default=None, metavar="N", help="N番目のフラグメントだけを実行（1始まり）")
    parser.add_argument("--dry-run", action="store_true", help="フラグメント一覧のみ表示して終了")
    parser.add_argument("--fireworks", metavar="MODEL", help="ehr_input に --fireworks MODEL を渡す")
    parser.add_argument(
        "--novita",
        nargs="?",
        const=_DEFAULT_NOVITA_MODEL,
        metavar="MODEL",
        help=f"ehr_input に --novita [MODEL] を渡す（省略時: {_DEFAULT_NOVITA_MODEL}）",
    )
    parser.add_argument("--openrouter", metavar="MODEL", help="ehr_input に --openrouter MODEL を渡す")
    parser.add_argument("--google-ai-studio", action="store_true", help="ehr_input に --google-ai-studio を渡す")
    parser.add_argument("--delay", type=float, default=2.0, metavar="SEC", help="フラグメント間の待機秒数（デフォルト: 2.0）")
    parser.add_argument("--clear", action="store_true", help="各フラグメント前にフィールドをクリアする（--clear を ehr_input に渡す）")
    return parser


def _build_ehr_input_command(
    fragment: str,
    *,
    clear: bool = False,
    fireworks_model: str | None = None,
    novita_model: str | None = None,
    openrouter_model: str | None = None,
    google_ai_studio: bool = False,
) -> list[str]:
    provider_count = int(bool(google_ai_studio)) + int(bool(openrouter_model)) + int(bool(novita_model)) + int(bool(fireworks_model))
    if provider_count > 1:
        raise ValueError("--google-ai-studio / --novita / --openrouter / --fireworks は同時に指定できません")
    cmd = [_PYTHON, "-m", "automation.ehr_input"]
    if fireworks_model:
        cmd.extend(["--fireworks", fireworks_model])
    if google_ai_studio:
        cmd.append("--google-ai-studio")
    if novita_model:
        cmd.extend(["--novita", novita_model])
    if openrouter_model:
        cmd.extend(["--openrouter", openrouter_model])
    if clear:
        cmd.append("--clear")
    cmd.append(fragment)
    return cmd


def _select_target_fragments(
    fragments: list[str],
    *,
    start: int = 1,
    end: int | None = None,
    fragment: int | None = None,
) -> tuple[int, int, list[str]]:
    total = len(fragments)
    if fragment is not None:
        if not 1 <= fragment <= total:
            raise ValueError(f"--fragment は 1〜{total} の範囲で指定してください")
        return fragment, fragment, [fragments[fragment - 1]]

    start = max(1, start)
    end = min(total, end) if end else total
    if start > end:
        raise ValueError(f"実行範囲が不正です: start={start}, end={end}")
    return start, end, fragments[start - 1 : end]


def _run_fragment(
    fragment: str,
    index: int,
    total: int,
    *,
    clear: bool = False,
    fireworks_model: str | None = None,
    novita_model: str | None = None,
    openrouter_model: str | None = None,
    google_ai_studio: bool = False,
) -> bool:
    """1フラグメントを ehr_input で実行する。成功時は True を返す。

    fragment が ``"\\n"`` の場合は改行マーカーとして扱い、
    ehr_input にそのまま渡して Enter キーを送信させる。
    """
    label = repr(fragment) if fragment != "\n" else '"\\n" (Enter)'
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] {label}")
    print('='*60)
    cmd = _build_ehr_input_command(
        fragment,
        clear=clear,
        fireworks_model=fireworks_model,
        novita_model=novita_model,
        openrouter_model=openrouter_model,
        google_ai_studio=google_ai_studio,
    )
    result = subprocess.run(
        cmd,
        cwd=_PROJECT_ROOT,
        timeout=900,
    )
    ok = result.returncode == 0
    status = "✅ OK" if ok else f"❌ FAIL (returncode={result.returncode})"
    print(f"→ {status}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        record_path = _resolve_record_path(args.record)
    except ValueError as exc:
        parser.error(str(exc))

    record_label = record_path.stem
    fragments = _build_fragments(record_path)
    total = len(fragments)
    print(f"対象カルテ: {record_label}")
    print(f"フラグメント総数: {total}")

    if args.dry_run:
        for i, f in enumerate(fragments, 1):
            if f == "\n":
                print(f"{i:3d}: [Enter]")
            else:
                print(f"{i:3d}: {f}")
        return 0

    if args.fragment is not None and (args.start is not None or args.end is not None):
        parser.error("--fragment は --start/--end と同時に使えません")
    provider_count = int(bool(args.google_ai_studio)) + int(bool(args.openrouter)) + int(bool(args.novita)) + int(bool(args.fireworks))
    if provider_count > 1:
        parser.error("--google-ai-studio / --novita / --openrouter / --fireworks は同時に使えません")

    try:
        start, end, target = _select_target_fragments(
            fragments,
            start=args.start or 1,
            end=args.end,
            fragment=args.fragment,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"実行範囲: {start}〜{end} ({len(target)} フラグメント)")
    print(f"入力対象ファイル: {record_path.relative_to(_PROJECT_ROOT)}")
    print("BLE サーバーと EHR 患者記録画面が起動していることを確認してください。")
    print("3秒後に開始します...")
    time.sleep(3)

    results: list[tuple[int, str, bool]] = []
    for rel_i, fragment in enumerate(target):
        abs_i = start + rel_i
        try:
            ok = _run_fragment(
                fragment,
                abs_i,
                total,
                clear=args.clear,
                fireworks_model=args.fireworks,
                novita_model=args.novita,
                openrouter_model=args.openrouter,
                google_ai_studio=args.google_ai_studio,
            )
        except subprocess.TimeoutExpired:
            print(f"[{abs_i}/{total}] ⏰ TIMEOUT (900s)")
            ok = False
        except KeyboardInterrupt:
            print("\n中断されました。")
            break
        results.append((abs_i, fragment, ok))
        if rel_i < len(target) - 1:
            time.sleep(args.delay)

    # サマリー
    print(f"\n{'='*60}")
    print("テスト結果サマリー")
    print('='*60)
    passed = sum(1 for _, _, ok in results if ok)
    failed = [(i, f) for i, f, ok in results if not ok]
    print(f"✅ 成功: {passed}/{len(results)}")
    if failed:
        print(f"❌ 失敗: {len(failed)} 件")
        for i, f in failed:
            print(f"  [{i}] {f[:60]}")
    else:
        print("全フラグメント成功！")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
