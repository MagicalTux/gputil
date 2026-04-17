import math

import pytest

import GPUtil


def test_getgpus_parses_default(fake_nvidia_smi):
    gpus = GPUtil.getGPUs()
    assert len(gpus) == 2

    busy, idle = gpus
    assert busy.id == 0
    assert busy.uuid == "GPU-2fc5ef53-da2a-d55d-9630-545b861a9b04"
    assert busy.load == 0.34
    assert busy.memoryTotal == 16376
    assert busy.memoryUsed == 2681
    assert busy.memoryFree == 13389
    assert busy.name == "NVIDIA GeForce RTX 4080 SUPER"
    assert busy.driver == "550.120"
    # Consumer GPUs commonly report [N/A] for the serial; keep it as a string.
    assert busy.serial == "[N/A]"
    assert busy.display_active == "Enabled"
    assert busy.display_mode == "Enabled"
    assert busy.temperature == 42

    assert idle.id == 1
    assert idle.load == 0.0


def test_getgpus_empty_scenario(fake_nvidia_smi):
    fake_nvidia_smi("empty")
    assert GPUtil.getGPUs() == []


def test_getgpus_handles_failure(fake_nvidia_smi):
    fake_nvidia_smi("fail")
    assert GPUtil.getGPUs() == []


def test_getgpus_nan_sensors(fake_nvidia_smi):
    fake_nvidia_smi("nan_values")
    gpus = GPUtil.getGPUs()
    assert len(gpus) == 1
    # Both utilization.gpu and temperature.gpu arrive as "[Not Supported]"
    # and must become NaN floats without crashing the parser.
    assert math.isnan(gpus[0].load)
    assert math.isnan(gpus[0].temperature)


def test_getgpus_datacenter_numeric_serial(fake_nvidia_smi):
    fake_nvidia_smi("datacenter")
    gpus = GPUtil.getGPUs()
    assert len(gpus) == 1
    assert gpus[0].name == "Tesla A100-SXM4-80GB"
    # Datacenter GPUs expose a real serial number rather than [N/A].
    assert gpus[0].serial == "1324920112345"


def test_getavailable_orders_first(fake_nvidia_smi):
    fake_nvidia_smi("three_mixed")
    # id 0 (load=0.1) and id 2 (load=0.3) are available; 'first' sorts by id.
    assert GPUtil.getAvailable(order="first", limit=2, maxLoad=0.5, maxMemory=0.9) == [0, 2]


def test_getavailable_orders_last(fake_nvidia_smi):
    fake_nvidia_smi("three_mixed")
    assert GPUtil.getAvailable(order="last", limit=2, maxLoad=0.5, maxMemory=0.9) == [2, 0]


def test_getavailable_orders_by_load(fake_nvidia_smi):
    fake_nvidia_smi("three_mixed")
    # Sorted by load: id 0 (0.1) < id 2 (0.3).
    assert GPUtil.getAvailable(order="load", limit=2, maxLoad=0.5, maxMemory=0.9) == [0, 2]


def test_getavailable_orders_by_memory(fake_nvidia_smi):
    fake_nvidia_smi("three_mixed")
    # Lowest memoryUtil first: id 2 (~1.2%) then id 0 (~6.1%).
    assert GPUtil.getAvailable(order="memory", limit=2, maxLoad=0.5, maxMemory=0.9) == [2, 0]


def test_getavailable_respects_limit(fake_nvidia_smi):
    fake_nvidia_smi("three_mixed")
    assert GPUtil.getAvailable(order="first", limit=1, maxLoad=0.5, maxMemory=0.9) == [0]


def test_getavailable_filters_high_load(fake_nvidia_smi):
    fake_nvidia_smi("three_mixed")
    # Default thresholds keep id 1 (load=0.9) out.
    assert 1 not in GPUtil.getAvailable(order="first", limit=3)


def test_getfirstavailable_returns_id(fake_nvidia_smi):
    # Both GPUs pass the default thresholds; order='first' picks the lowest id.
    result = GPUtil.getFirstAvailable(maxLoad=0.5, maxMemory=0.9, attempts=1, interval=0)
    assert result == [0]


def test_getfirstavailable_raises_when_none(fake_nvidia_smi):
    fake_nvidia_smi("empty")
    with pytest.raises(RuntimeError):
        GPUtil.getFirstAvailable(attempts=1, interval=0)


def test_showutilization_prints(fake_nvidia_smi, capsys):
    GPUtil.showUtilization()
    captured = capsys.readouterr().out
    assert "ID" in captured
    assert "GPU" in captured
    # Both GPU ids should appear in the rendered table.
    assert " 0 " in captured
    assert " 1 " in captured
