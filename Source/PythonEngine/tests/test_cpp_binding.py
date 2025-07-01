# ==============================================================================
# 文件名称：tests/test_cpp_binding.py
# 接口名称：CppBindingUnitTest
# 作者：beilsm
# 版本号：v1.0.0
# 创建时间：2025-07-01 18:10 (Asia/Taipei)
# 最新更改时间：2025-07-01 18:10 (Asia/Taipei)
# ==============================================================================
# ✅ 功能简介：
#    - 对 OceanSim C++ Core 的 Python 绑定 (oceansim) 进行快速单元测试
#   - 验证以下内容：
#       1. 模块能被成功 import；
#       2. 可创建 GridDataStructure；
#       3. CurrentFieldSolver 可以初始化并前进一步；
#       4. 结果对象非空且包含预期属性。
# ==============================================================================
# ✅ 更新说明：
#   - v1.0.0 首次实现，可独立运行或由 pytest 调用。
# ==============================================================================
# ✅ 运行方式：
#   1) 先确保已按以下命令编译并安装 Python 绑定：
#        cmake -B build -DBUILD_PYTHON_BINDINGS=ON && cmake --build build --target install
#   2) 激活虚拟环境并安装 pytest（requirements.txt 已新增 pytest）：
#        pip install -r requirements.txt
#   3) CLI 独立运行：
#        python -m tests.test_cpp_binding
#   4) 或项目根目录执行：
#        pytest tests/test_cpp_binding.py
# ==============================================================================

from __future__ import annotations

import importlib
import sys
from typing import Any, Optional


def _try_import_module(name: str):
    """Attempt to import a module and return (module | None, error | None)."""
    try:
        module = importlib.import_module(name)
        return module, None
    except Exception as exc:  # noqa: BLE001  # broad catch is intentional for test
        return None, exc


def _create_basic_grid(oc: Any):
    """Create a small 3‑D cartesian grid for smoke testing."""
    return oc.GridDataStructure(
        nx=10,
        ny=10,
        nz=1,
        coord_sys=oc.GridDataStructure.CoordinateSystem.CARTESIAN,
    )


# ------------------------------------------------------------------------------
# PyTest‑style unit test (auto‑discoverable)
# ------------------------------------------------------------------------------

def test_cpp_binding_basic():
    """Smoke test for the oceansim binding."""

    oceansim, err = _try_import_module("oceansim")
    assert oceansim is not None, (
        "Cannot import oceansim – ensure BUILD_PYTHON_BINDINGS is ON.\n"  # noqa: E501
        f"Original exception: {err}"
    )

    # 1. Create a small grid
    grid = _create_basic_grid(oceansim)
    assert grid is not None, "Grid creation failed"

    # 2. Instantiate solver with default physical parameters
    solver = oceansim.CurrentFieldSolver(grid, oceansim.CurrentFieldSolver.PhysicalParameters())
    solver.initialize()

    # 3. Step forward once and fetch state
    solver.step_forward()
    state = solver.get_current_state()

    # 4. Basic assertions – implementation‑dependent but at least non‑null
    assert state is not None, "State retrieval returned None"
    if hasattr(state, "shape"):
        assert all(dim > 0 for dim in state.shape), "State array has invalid shape"


# ------------------------------------------------------------------------------
# Stand‑alone CLI execution
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    try:
        test_cpp_binding_basic()
        print("✅ oceansim binding smoke test passed.")
        sys.exit(0)
    except AssertionError as ae:
        print(f"❌ Assertion failed: {ae}")
        sys.exit(1)
    except Exception:  # noqa: BLE001 – catch‑all for unexpected errors
        traceback.print_exc()
        sys.exit(2)
