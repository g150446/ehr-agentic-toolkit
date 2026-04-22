import builtins
import importlib.util
from pathlib import Path


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "automation" / "config.py"


def test_config_module_allows_missing_python_dotenv(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dotenv":
            raise ModuleNotFoundError("No module named 'dotenv'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location("config_without_dotenv", _CONFIG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = module.AutomationConfig(env_file=str(tmp_path / ".env"))

    assert module.load_dotenv(str(tmp_path / ".env"), override=True) is False
    assert config.capture_device_index == 0
