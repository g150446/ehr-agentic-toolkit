"""
HDMIキャプチャのフレームをフェーズ別に動画として記録するモジュール。

--movie オプション時に ehr_composer.py から使用される。
USB HDMIキャプチャデバイスは多重オープン非対応を前提とし、
既存の capture_screen() 戻り値フレームをフックして記録する方式。

使用方法:
    recorder = VideoRecorder(width=1920, height=1080, fps=10)
    recorder.start_phase(path, frame_skip=1)
    # capture_screen() 等からフレームが渡される
    recorder.write(frame)
    recorder.stop_phase()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2


class VideoRecorder:
    """フェーズ別にフレームを書き出す動画レコーダー。

    frame_skip > 1 を指定すると、記録フレームを間引いて「早送り」効果を得る。
    例: frame_skip=3 は 3フレームに1回書き出し → 3倍速動画。
    """

    def __init__(self, width: int, height: int, fps: int = 10) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_skip: int = 1
        self._frame_count: int = 0

    def start_phase(self, path: Path, frame_skip: int = 1) -> None:
        """新しいフェーズの動画書き出しを開始する。

        Args:
            path: 出力先ファイルパス（.mp4）
            frame_skip: 1=通常速度, 2=2倍速, 3=3倍速
        """
        self.stop_phase()
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = _create_video_writer(path, self.width, self.height, self.fps)
        self._writer = writer
        self._frame_skip = max(1, frame_skip)
        self._frame_count = 0
        print(f"[movie] フェーズ録画開始: {path.name} (frame_skip={self._frame_skip})")

    def write(self, frame) -> bool:
        """フレームを書き出す。記録中でなければ何もしない。

        Returns:
            書き出しが行われたかどうか
        """
        if self._writer is None or frame is None:
            return False
        written = False
        if self._frame_count % self._frame_skip == 0:
            self._writer.write(frame)
            written = True
        self._frame_count += 1
        return written

    def stop_phase(self) -> None:
        """現在のフェーズの動画書き出しを終了する。"""
        if self._writer is not None:
            self._writer.release()
            print("[movie] フェーズ録画終了")
            self._writer = None

    def is_recording(self) -> bool:
        """現在記録中かどうかを返す。"""
        return self._writer is not None


def _create_video_writer(path: Path, width: int, height: int, fps: int) -> cv2.VideoWriter:
    """利用可能な fourcc で VideoWriter を作成する（フォールバック付き）。"""
    for fourcc_name in ("mp4v", "avc1", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"[movie] fourcc={fourcc_name} で書き出し開始")
            return writer
    raise RuntimeError(
        f"動画コーデックが見つかりません。OpenCV が MP4 書き出しに対応しているか確認してください。"
    )
