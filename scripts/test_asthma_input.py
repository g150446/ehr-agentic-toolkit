#!/usr/bin/env python3
"""asthma_1.txt の句読点分割フラグメントを ehr_input でテストする。

前提条件:
  - BLE サーバーが起動・接続済み
  - EHR の患者記録画面が開いている

実行方法:
  python scripts/test_asthma_input.py
  python scripts/test_asthma_input.py --win10          # Windows 10 IME モード
  python scripts/test_asthma_input.py --start 3        # 3番目から再開
  python scripts/test_asthma_input.py --win10 --start 3
  python scripts/test_asthma_input.py --dry-run        # フラグメント一覧のみ表示
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_ASTHMA_FILE = _PROJECT_ROOT / "data" / "patient_records" / "asthma_1.txt"
_PYTHON = sys.executable


def _build_fragments(path: Path) -> list[str]:
    """asthma_1.txt を句読点（。・、）で分割してフラグメントリストを返す。"""
    text = path.read_text(encoding="utf-8")
    # セクションヘッダ [S][O][A][P] と # プレフィックスを除去
    text = re.sub(r'^\s*\[[SOAP]\]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*#\s*', '', text, flags=re.MULTILINE)
    # 句読点の直後で分割（句読点は各フラグメントの末尾に残す）
    raw = re.split(r'(?<=[。、])', text)
    fragments: list[str] = []
    for f in raw:
        # 複数行をスペースに変換、前後の空白を除去
        f = re.sub(r'\s+', ' ', f).strip()
        # 4文字以上 かつ 日本語文字を含むもののみ
        if len(f) >= 4 and re.search(r'[\u3040-\u9FFF]', f):
            fragments.append(f)
    return fragments


def _run_fragment(fragment: str, index: int, total: int, win10: bool = False, clear: bool = False) -> bool:
    """1フラグメントを ehr_input で実行する。成功時は True を返す。"""
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] {fragment!r}")
    print('='*60)
    cmd = [_PYTHON, "-m", "automation.ehr_input"]
    if win10:
        cmd.append("--win10")
    if clear:
        cmd.append("--clear")
    cmd.append(fragment)
    result = subprocess.run(
        cmd,
        cwd=_PROJECT_ROOT,
        timeout=900,
    )
    ok = result.returncode == 0
    status = "✅ OK" if ok else f"❌ FAIL (returncode={result.returncode})"
    print(f"→ {status}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=1, metavar="N", help="N番目のフラグメントから開始（1始まり）")
    parser.add_argument("--end", type=int, default=None, metavar="N", help="N番目のフラグメントで終了（1始まり、含む）")
    parser.add_argument("--dry-run", action="store_true", help="フラグメント一覧のみ表示して終了")
    parser.add_argument("--win10", action="store_true", help="Windows 10 の IME テンプレートを使用")
    parser.add_argument("--delay", type=float, default=2.0, metavar="SEC", help="フラグメント間の待機秒数（デフォルト: 2.0）")
    parser.add_argument("--clear", action="store_true", help="各フラグメント前にフィールドをクリアする（--clear を ehr_input に渡す）")
    args = parser.parse_args()

    fragments = _build_fragments(_ASTHMA_FILE)
    total = len(fragments)
    print(f"フラグメント総数: {total}")

    if args.dry_run:
        for i, f in enumerate(fragments, 1):
            print(f"{i:3d}: {f[:80]}")
        return 0

    start = max(1, args.start)
    end = min(total, args.end) if args.end else total
    target = fragments[start - 1 : end]

    print(f"実行範囲: {start}〜{end} ({len(target)} フラグメント)")
    print("BLE サーバーと EHR 患者記録画面が起動していることを確認してください。")
    print("3秒後に開始します...")
    time.sleep(3)

    results: list[tuple[int, str, bool]] = []
    for rel_i, fragment in enumerate(target):
        abs_i = start + rel_i
        try:
            ok = _run_fragment(fragment, abs_i, total, win10=args.win10, clear=args.clear)
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
