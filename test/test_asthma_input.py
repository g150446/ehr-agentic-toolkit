from __future__ import annotations

from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "test_asthma_input.py"
_SPEC = importlib.util.spec_from_file_location("test_asthma_input_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
asthma_input = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(asthma_input)


def test_build_ehr_input_command_adds_openrouter_and_clear():
    assert asthma_input._build_ehr_input_command(
        "咳嗽あり。",
        clear=True,
        openrouter_model="google/gemma-4-26b-a4b-it",
    ) == [
        asthma_input._PYTHON,
        "-m",
        "automation.ehr_input",
        "--openrouter",
        "google/gemma-4-26b-a4b-it",
        "--clear",
        "咳嗽あり。",
    ]


def test_resolve_python_executable_prefers_repo_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(asthma_input, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(asthma_input.sys, "executable", "/usr/bin/python3")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    assert asthma_input._resolve_python_executable() == str(venv_python)


def test_resolve_python_executable_falls_back_to_current_python(monkeypatch, tmp_path):
    monkeypatch.setattr(asthma_input, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(asthma_input.sys, "executable", "/usr/bin/python3")

    assert asthma_input._resolve_python_executable() == "/usr/bin/python3"


def test_resolve_record_path_defaults_to_known_patient_records_dir(monkeypatch, tmp_path):
    patient_records_dir = tmp_path / "data" / "patient_records"
    monkeypatch.setattr(asthma_input, "_PATIENT_RECORDS_DIR", patient_records_dir)

    assert asthma_input._resolve_record_path("asthma_2") == patient_records_dir / "asthma_2.txt"


def test_resolve_record_path_accepts_numeric_shortcut(monkeypatch, tmp_path):
    patient_records_dir = tmp_path / "data" / "patient_records"
    monkeypatch.setattr(asthma_input, "_PATIENT_RECORDS_DIR", patient_records_dir)

    assert asthma_input._resolve_record_path("3") == patient_records_dir / "asthma_3.txt"


def test_resolve_record_path_rejects_unknown_record():
    with pytest.raises(ValueError, match="--record は asthma_1, asthma_2, asthma_3 のいずれかを指定してください"):
        asthma_input._resolve_record_path("asthma_9")


def test_build_fragments_preserves_headers_punctuation_and_blank_lines(tmp_path):
    record = tmp_path / "asthma.txt"
    record.write_text("[S]\n咳嗽あり。\n\n[A]\n# 気管支喘息\n", encoding="utf-8")

    assert asthma_input._build_fragments(record) == [
        "[S]",
        "\n",
        "咳嗽あり。",
        "\n",
        "\n",
        "[A]",
        "\n",
        "# 気管支喘息",
        "\n",
    ]


def test_build_fragments_preserves_last_line_without_trailing_newline(tmp_path):
    record = tmp_path / "asthma.txt"
    record.write_text("[O]\nVitals: BT 36.5℃", encoding="utf-8")

    assert asthma_input._build_fragments(record) == [
        "[O]",
        "\n",
        "Vitals: BT 36.5℃",
    ]


def test_build_ehr_input_command_adds_google_ai_studio():
    assert asthma_input._build_ehr_input_command(
        "咳嗽あり。",
        google_ai_studio=True,
    ) == [
        asthma_input._PYTHON,
        "-m",
        "automation.ehr_input",
        "--google-ai-studio",
        "咳嗽あり。",
    ]


def test_build_ehr_input_command_adds_fireworks():
    assert asthma_input._build_ehr_input_command(
        "咳嗽あり。",
        fireworks_model="accounts/fireworks/models/gemma-4-26b-a4b-it",
    ) == [
        asthma_input._PYTHON,
        "-m",
        "automation.ehr_input",
        "--fireworks",
        "accounts/fireworks/models/gemma-4-26b-a4b-it",
        "咳嗽あり。",
    ]


def test_select_target_fragments_returns_single_fragment():
    start, end, target = asthma_input._select_target_fragments(
        ["一つ目。", "二つ目。", "三つ目。"],
        fragment=2,
    )

    assert (start, end, target) == (2, 2, ["二つ目。"])


def test_main_runs_only_selected_fragment(monkeypatch):
    calls = []
    paths = []
    monkeypatch.setattr(
        asthma_input,
        "_build_fragments",
        lambda path: paths.append(path) or ["一つ目。", "二つ目。", "三つ目。"],
    )
    monkeypatch.setattr(asthma_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        asthma_input,
        "_run_fragment",
        lambda fragment, index, total, **kwargs: calls.append((fragment, index, total, kwargs)) or True,
    )

    assert asthma_input.main(["--record", "asthma_2", "--fragment", "2", "--openrouter", "google/gemma-4-26b-a4b-it"]) == 0
    assert paths == [asthma_input._PATIENT_RECORDS_DIR / "asthma_2.txt"]
    assert calls == [
        (
            "二つ目。",
            2,
            3,
            {
                "clear": False,
                "fireworks_model": None,
                "google_ai_studio": False,
                "openrouter_model": "google/gemma-4-26b-a4b-it",
            },
        )
    ]


def test_main_uses_default_record_when_not_specified(monkeypatch):
    paths = []
    monkeypatch.setattr(
        asthma_input,
        "_build_fragments",
        lambda path: paths.append(path) or ["一つ目。"],
    )
    monkeypatch.setattr(asthma_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(asthma_input, "_run_fragment", lambda *args, **kwargs: True)

    assert asthma_input.main([]) == 0
    assert paths == [asthma_input._PATIENT_RECORDS_DIR / "asthma_1.txt"]


def test_main_dry_run_shows_preserved_headers_and_enters(monkeypatch, capsys):
    monkeypatch.setattr(asthma_input, "_build_fragments", lambda path: ["[S]", "\n", "咳嗽あり。"])

    assert asthma_input.main(["--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "  1: [S]" in output
    assert "  2: [Enter]" in output
    assert "  3: 咳嗽あり。" in output


def test_main_rejects_fragment_with_start(monkeypatch):
    monkeypatch.setattr(asthma_input, "_build_fragments", lambda path: ["一つ目。", "二つ目。"])
    stderr = io.StringIO()

    with redirect_stderr(stderr), pytest.raises(SystemExit) as excinfo:
        asthma_input.main(["--fragment", "2", "--start", "1"])

    assert excinfo.value.code == 2
    assert "--fragment は --start/--end と同時に使えません" in stderr.getvalue()


def test_main_no_longer_accepts_win10_option():
    stderr = io.StringIO()

    with redirect_stderr(stderr), pytest.raises(SystemExit) as excinfo:
        asthma_input.main(["--win10"])

    assert excinfo.value.code == 2
    assert "unrecognized arguments: --win10" in stderr.getvalue()


def test_main_rejects_unknown_record(monkeypatch):
    monkeypatch.setattr(asthma_input, "_build_fragments", lambda path: ["一つ目。"])
    stderr = io.StringIO()

    with redirect_stderr(stderr), pytest.raises(SystemExit) as excinfo:
        asthma_input.main(["--record", "asthma_9"])

    assert excinfo.value.code == 2
    assert "--record は asthma_1, asthma_2, asthma_3 のいずれかを指定してください" in stderr.getvalue()


def test_run_fragment_passes_command_to_subprocess(monkeypatch):
    captured = {}

    def fake_run(cmd, cwd, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(asthma_input.subprocess, "run", fake_run)

    assert asthma_input._run_fragment(
        "咳嗽あり。",
        1,
        3,
        clear=True,
        openrouter_model="google/gemma-4-26b-a4b-it",
    )
    assert captured == {
        "cmd": [
            asthma_input._PYTHON,
            "-m",
            "automation.ehr_input",
            "--openrouter",
            "google/gemma-4-26b-a4b-it",
            "--clear",
            "咳嗽あり。",
        ],
        "cwd": asthma_input._PROJECT_ROOT,
        "timeout": 900,
    }


def test_main_rejects_google_ai_studio_with_openrouter(monkeypatch):
    monkeypatch.setattr(asthma_input, "_build_fragments", lambda path: ["一つ目。", "二つ目。"])
    stderr = io.StringIO()

    with redirect_stderr(stderr), pytest.raises(SystemExit) as excinfo:
        asthma_input.main(["--google-ai-studio", "--openrouter", "google/gemma-4-26b-a4b-it"])

    assert excinfo.value.code == 2
    assert "--google-ai-studio / --openrouter / --fireworks は同時に使えません" in stderr.getvalue()


def test_main_rejects_fireworks_with_openrouter(monkeypatch):
    monkeypatch.setattr(asthma_input, "_build_fragments", lambda path: ["一つ目。", "二つ目。"])
    stderr = io.StringIO()

    with redirect_stderr(stderr), pytest.raises(SystemExit) as excinfo:
        asthma_input.main(["--fireworks", "accounts/fireworks/models/gemma-4-26b-a4b-it", "--openrouter", "google/gemma-4-26b-a4b-it"])

    assert excinfo.value.code == 2
    assert "--google-ai-studio / --openrouter / --fireworks は同時に使えません" in stderr.getvalue()
