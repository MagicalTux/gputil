import math

import GPUtil
from GPUtil.GPUtil import GPU, safeFloatCast, getAvailability


def make_gpu(gid=0, uuid="GPU-x", load=0.1, mem_total=1000, mem_used=100, mem_free=900):
    return GPU(
        gid, uuid, load, mem_total, mem_used, mem_free,
        "driver", "name", "serial", "Disabled", "Disabled", 50,
    )


def test_safefloatcast_valid():
    assert safeFloatCast("3.14") == 3.14
    assert safeFloatCast("42") == 42.0


def test_safefloatcast_invalid_returns_nan():
    assert math.isnan(safeFloatCast("[Not Supported]"))
    assert math.isnan(safeFloatCast("abc"))


def test_gpu_memory_util_ratio():
    gpu = make_gpu(mem_total=1000, mem_used=250, mem_free=750)
    assert gpu.memoryUtil == 0.25


def test_availability_all_available():
    gpus = [make_gpu(0, load=0.1), make_gpu(1, load=0.2)]
    assert getAvailability(gpus) == [1, 1]


def test_availability_high_load_excluded():
    gpus = [make_gpu(0, load=0.9), make_gpu(1, load=0.1)]
    assert getAvailability(gpus, maxLoad=0.5) == [0, 1]


def test_availability_high_memory_excluded():
    busy = make_gpu(0, mem_total=1000, mem_used=800, mem_free=200)
    idle = make_gpu(1, mem_total=1000, mem_used=100, mem_free=900)
    assert getAvailability([busy, idle], maxMemory=0.5) == [0, 1]


def test_availability_memoryfree_threshold():
    low = make_gpu(0, mem_total=1000, mem_used=900, mem_free=100)
    high = make_gpu(1, mem_total=1000, mem_used=100, mem_free=900)
    assert getAvailability([low, high], memoryFree=500) == [0, 1]


def test_availability_exclude_id():
    gpus = [make_gpu(0), make_gpu(1)]
    assert getAvailability(gpus, excludeID=[0]) == [0, 1]


def test_availability_exclude_uuid():
    gpus = [make_gpu(0, uuid="GPU-a"), make_gpu(1, uuid="GPU-b")]
    assert getAvailability(gpus, excludeUUID=["GPU-b"]) == [1, 0]


def test_availability_nan_load_respects_includeNan():
    gpu = make_gpu(0, load=float("nan"))
    assert getAvailability([gpu], includeNan=False) == [0]
    assert getAvailability([gpu], includeNan=True) == [1]


def test_public_api_surface():
    # Ensure the package re-exports the documented names.
    for name in ("GPU", "getGPUs", "getAvailable", "getAvailability",
                 "getFirstAvailable", "showUtilization", "__version__"):
        assert hasattr(GPUtil, name), name
