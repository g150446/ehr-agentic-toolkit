import sys
from types import SimpleNamespace

import numpy as np

from automation import medgemma_cxr_compare


def _run_main(monkeypatch, *args):
    monkeypatch.setattr(
        sys,
        "argv",
        ["automation.medgemma_cxr_compare", *args],
    )
    return medgemma_cxr_compare.main()


def test_main_requires_explicit_input_mode(monkeypatch, capsys):
    assert _run_main(monkeypatch) == 1
    assert "Specify one input mode" in capsys.readouterr().err


def test_main_rejects_device_without_capture(monkeypatch, capsys):
    assert _run_main(monkeypatch, "--device", "1") == 1
    assert "--device requires --capture" in capsys.readouterr().err


def test_main_rejects_multiple_input_modes(monkeypatch, capsys):
    assert _run_main(
        monkeypatch,
        "--new-image",
        "new.png",
        "--old-image",
        "old.png",
        "--capture",
    ) == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_main_defaults_capture_device_to_zero(monkeypatch, tmp_path):
    captured = {}
    config = SimpleNamespace(
        capture_device_index=9,
        log_level="INFO",
        log_dir=tmp_path,
        output_dir=tmp_path,
    )

    monkeypatch.setattr(
        medgemma_cxr_compare,
        "load_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        medgemma_cxr_compare,
        "setup_logging",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        medgemma_cxr_compare,
        "select_compute_device",
        lambda _preference: "cpu",
    )

    def fake_acquire(args, _config):
        captured["device"] = args.device
        return None, None, ""

    monkeypatch.setattr(
        medgemma_cxr_compare,
        "acquire_cxr_pair",
        fake_acquire,
    )

    assert _run_main(monkeypatch, "--capture") == 1
    assert captured["device"] == 0


def test_acquire_capture_reads_live_hdmi(monkeypatch):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    new_image = np.zeros((100, 80, 3), dtype=np.uint8)
    old_image = np.zeros((100, 80, 3), dtype=np.uint8)
    calls = {}

    def fake_capture_screen(**kwargs):
        calls.update(kwargs)
        return frame

    monkeypatch.setattr(
        "automation.screen_analyzer.capture_screen",
        fake_capture_screen,
    )
    monkeypatch.setattr(
        medgemma_cxr_compare,
        "crop_dual_cxr",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                image=new_image,
                bbox=(10, 20, 80, 100),
                method="manual",
            ),
            SimpleNamespace(
                image=old_image,
                bbox=(110, 20, 80, 100),
                method="manual",
            ),
        ),
    )

    args = SimpleNamespace(
        new_image=None,
        old_image=None,
        image=None,
        capture=True,
        device=2,
        debug_dir_path=None,
        roi1=None,
        roi2=None,
        no_mask_overlay=False,
    )
    config = SimpleNamespace(capture_width=1920, capture_height=1080)

    actual_new, actual_old, source = medgemma_cxr_compare.acquire_cxr_pair(
        args,
        config,
    )

    assert actual_new is new_image
    assert actual_old is old_image
    assert source.startswith("hdmi device=2")
    assert calls == {"device_index": 2, "width": 1920, "height": 1080}
