import os
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def fake_nvidia_smi(monkeypatch):
    """Prepend the fake nvidia-smi directory to PATH.

    Returns a setter function for switching scenarios. The default scenario is
    'default' (two GPUs).
    """
    monkeypatch.setenv("PATH", FIXTURES_DIR + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_SCENARIO", "default")

    def set_scenario(name):
        monkeypatch.setenv("FAKE_NVIDIA_SMI_SCENARIO", name)

    return set_scenario
