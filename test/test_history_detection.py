from types import SimpleNamespace
import json

import automation.ehr_input as ehr_input
import automation.mlx_vlm_history as mlx_vlm_history


def test_date_matches_text_accepts_extra_leading_one_for_10th_day():
    assert mlx_vlm_history._date_matches_text("2026年日4月110日12:51", 2026, 4, 10)
    assert not mlx_vlm_history._date_matches_text("2026年日4月110日12:51", 2026, 4, 11)


def test_find_history_date_with_vlm_handles_110_day_ocr_without_server():
    ocr_results = [
        ([[560, 20], [592, 20], [592, 42], [560, 42]], "1932(昭071年10月13日", 0.90),
        ([[1080, 160], [1182, 160], [1182, 184], [1080, 184]], "2026年14月1日日15：内科診療記録1版", 0.91),
        ([[290, 586], [380, 586], [380, 608], [290, 608]], "2026年日4月07日17:2日", 0.93),
        ([[290, 695], [380, 695], [380, 717], [290, 717]], "2026年日4月0日日15:52", 0.92),
        ([[352, 803], [442, 803], [442, 825], [352, 825]], "2026年14月09日17：42内科診療記録", 0.94),
        ([[290, 913], [380, 913], [380, 935], [290, 935]], "2026年日4月110日12:51", 0.95),
    ]

    assert mlx_vlm_history.find_history_date_with_vlm("20260410", ocr_results) == (335, 924)


def test_find_history_date_with_vlm_prefers_topmost_match_when_multiple_rows_match():
    ocr_results = [
        ([[560, 20], [592, 20], [592, 42], [560, 42]], "1932(昭071年10月13日", 0.90),
        ([[1080, 160], [1182, 160], [1182, 184], [1080, 184]], "2026年14月7日17：2日内科1診療記録1版", 0.91),
        ([[289, 643], [379, 643], [379, 665], [289, 665]], "2026年日4月03日15:日1", 0.93),
        ([[290, 719], [380, 719], [380, 741], [290, 741]], "2026年日4月03日15:02", 0.92),
        ([[308, 836], [398, 836], [398, 858], [308, 858]], "2026年日4月日4日", 0.90),
        ([[290, 886], [380, 886], [380, 908], [290, 908]], "2026年04月03日15:02", 0.95),
    ]

    assert mlx_vlm_history.find_history_date_with_vlm("20260403", ocr_results) == (334, 654)


def test_find_history_date_with_vlm_ignores_verbose_non_numeric_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "画像を確認すると、以下の候補が該当します。\n\n- 候補 [3]"
                            }
                        }
                    ]
                }
            ).encode()

    monkeypatch.setattr(mlx_vlm_history.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(mlx_vlm_history, "_encode_image_data_url", lambda image: "data:image/png;base64,abc")
    monkeypatch.setattr(mlx_vlm_history, "_build_history_crop", lambda image, candidates: image)

    ocr_results = [
        ([[560, 20], [592, 20], [592, 42], [560, 42]], "1932（昭07）年10月13日", 0.90),
        ([[290, 886], [380, 886], [380, 908], [290, 908]], "2026年03月02日13:20", 0.95),
    ]

    assert mlx_vlm_history.find_history_date_with_vlm("20260311", ocr_results, image=object()) is None


def test_find_history_date_with_vlm_maps_qwen_date_rank_to_easyocr_coords(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"dates\": [\"20260408\", \"20260407\", \"20260404\", \"20260401\"]}"
                            }
                        }
                    ]
                }
            ).encode()

    monkeypatch.setattr(mlx_vlm_history.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(mlx_vlm_history, "_encode_image_data_url", lambda image: "data:image/png;base64,abc")
    monkeypatch.setattr(mlx_vlm_history, "_build_history_crop", lambda image, candidates: image)

    ocr_results = [
        ([[280, 586], [336, 586], [336, 608], [280, 608]], "2026年』月の日", 0.91),
        ([[305, 694], [361, 694], [361, 716], [305, 716]], "2026年ざ月昭日 15.52", 0.92),
        ([[280, 804], [336, 804], [336, 826], [280, 826]], "2026年』月の日", 0.93),
        ([[280, 917], [336, 917], [336, 939], [280, 939]], "2026年74月1日", 0.94),
    ]

    assert mlx_vlm_history.find_history_date_with_vlm("20260408", ocr_results, image=object()) == (308, 597)


def test_parse_ordered_dates_normalizes_japanese_date_strings():
    content = '{"dates": ["2026年4月8日", "2026年04月07日", "20260406"]}'

    assert mlx_vlm_history._parse_ordered_dates(content) == [
        "20260408",
        "20260407",
        "20260406",
    ]


def test_load_ocr_engine_uses_easyocr_reader(monkeypatch):
    sentinel = object()
    config = SimpleNamespace(ocr_languages=["ja", "en"], ocr_use_gpu=False)

    monkeypatch.setattr(ehr_input, "load_ocr_reader", lambda languages, use_gpu: sentinel)

    assert ehr_input._load_ocr_engine(config) is sentinel


def test_click_history_uses_mlx_vlm_ocr_pipeline(monkeypatch):
    frame = object()
    events = []

    class FakeClient:
        def switch_to_mouse_mode(self):
            events.append("mode")
            return True

        def move_mouse_to_position(self, x, y):
            events.append(("move", x, y))
            return True

        def click(self):
            events.append("click")
            return True

    monkeypatch.setattr(
        ehr_input,
        "load_config",
        lambda skip_password=True: SimpleNamespace(
            capture_device_index=0,
            capture_width=1920,
            capture_height=1080,
            detection_confidence=0.2,
        ),
    )
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kwargs: frame)
    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda: FakeClient())
    monkeypatch.setattr(
        mlx_vlm_history,
        "find_history_date_in_image",
        lambda date_str, image, **kwargs: (335, 924) if date_str == "20260410" and image is frame else None,
    )

    ehr_input.click_history("20260410")

    assert events == ["mode", ("move", 335, 924), "click"]


def test_click_history_surfaces_mlx_vlm_errors(monkeypatch):
    frame = object()

    monkeypatch.setattr(
        ehr_input,
        "load_config",
        lambda skip_password=True: SimpleNamespace(
            capture_device_index=0,
            capture_width=1920,
            capture_height=1080,
            detection_confidence=0.2,
        ),
    )
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kwargs: frame)
    monkeypatch.setattr(
        mlx_vlm_history,
        "find_history_date_in_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            mlx_vlm_history.MlxVlmHistoryError("server unavailable")
        ),
    )

    try:
        ehr_input.click_history("20260410")
    except RuntimeError as exc:
        assert str(exc) == "過去カルテ日付の検出に失敗しました: server unavailable"
    else:
        raise AssertionError("RuntimeError was not raised")


def test_find_history_date_in_image_uses_full_image_easyocr(monkeypatch):
    frame = object()
    ocr_results = [("full", "full", 1.0)]

    monkeypatch.setattr(
        mlx_vlm_history,
        "_run_full_image_ocr",
        lambda image, languages=None: ocr_results if image is frame else None,
    )
    monkeypatch.setattr(
        mlx_vlm_history,
        "find_history_date_with_vlm",
        lambda date_str, actual_results, **kwargs: (
            (335, 753) if actual_results == ocr_results and date_str == "20260312"
            else None
        ),
    )

    assert mlx_vlm_history.find_history_date_in_image("20260312", frame) == (335, 753)


def test_open_test_patient_chart_uses_direct_ocr(monkeypatch):
    frame = object()
    events = []

    class FakeClient:
        def switch_to_mouse_mode(self):
            events.append("mouse")
            return True

        def move_mouse_to_position(self, x, y):
            events.append(("move", x, y))
            return True

        def click(self):
            events.append("click")
            return True

        def switch_to_keyboard_mode(self):
            events.append("keyboard")
            return True

        def press_key(self, key):
            events.append(("key", key))
            return True

    monkeypatch.setattr(
        ehr_input,
        "load_config",
        lambda skip_password=True: SimpleNamespace(
            capture_device_index=0,
            capture_width=1920,
            capture_height=1080,
            ocr_languages=["ja", "en"],
            ocr_backend="easyocr",
            ocr_use_gpu=False,
        ),
    )
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kwargs: frame)
    monkeypatch.setattr(
        ehr_input,
        "_request_ocr_results",
        lambda image, config: [
            ([[70, 450], [90, 450], [90, 474], [70, 474]], "患者検索", 0.99),
        ] if image is frame else [],
    )
    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda: FakeClient())
    monkeypatch.setattr(ehr_input, "input_text_to_field", lambda input_text, label: events.append(("input", input_text, label)))
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)

    ehr_input.open_test_patient_chart()

    assert ("move", 80, 462) in events
    assert ("input", "tesuto", "フリガナ") in events
