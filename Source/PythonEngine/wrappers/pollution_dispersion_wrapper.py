#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: pollution_dispersion_wrapper.py
位置: Source/PythonEngine/wrappers/pollution_dispersion_wrapper.py
功能: 为C#提供污染物扩散模拟包装器
用法: python pollution_dispersion_wrapper.py input.json output.json
"""

import sys
import json
import traceback
from pathlib import Path



np = None
plt = None


# 确保PythonEngine在路径中
current_dir = Path(__file__).parent
python_engine_root = current_dir.parent
if str(python_engine_root) not in sys.path:
    sys.path.insert(0, str(python_engine_root))



def run_pollution_dispersion(input_data: dict) -> dict:
    """运行一个简单的污染物扩散模拟并生成图像"""
    try:
        params = input_data.get("parameters", {})
        grid = params.get(
            "grid",
            {"nx": 60, "ny": 60, "nz": 15, "dx": 100.0, "dy": 100.0, "dz": 10.0},
        )
        diffusion = params.get(
            "diffusion",
            {"horizontal_diffusion": 10.0, "vertical_diffusion": 1.0, "degradation_rate": 1e-6},
        )
        sources = params.get(
            "sources",
            [
                {
                    "position": [grid["nx"] // 2, grid["ny"] // 2, grid["nz"] // 2],
                    "release_rate": 100.0,
                    "duration": 3600.0,
                }
            ],
        )
        total_time = float(params.get("total_time", 7200.0))
        dt = float(params.get("dt", 60.0))
        depth_index = int(params.get("depth_index", grid["nz"] // 2))
        output_path = params.get("output_path", "pollution_dispersion.png")

        sim = PollutionDispersionSimulator(grid, diffusion)
        for src in sources:
            sim.add_pollution_source(
                tuple(src.get("position", [0, 0, 0])),
                float(src.get("release_rate", 0.0)),
                float(src.get("duration", np.inf)),
                src.get("type", "generic"),
            )

        u = np.zeros((grid["nx"], grid["ny"], grid["nz"]))
        v = np.zeros_like(u)
        w = np.zeros_like(u)

        current = 0.0
        while current < total_time:
            sim.solve_advection_diffusion((u, v, w), dt, current)
            current += dt

        conc = sim.get_concentration_at_depth(depth_index)
        plt.figure(figsize=(6, 4))
        plt.contourf(conc, levels=20, cmap="YlOrRd")
        plt.colorbar(label="Concentration")
        plt.title("Pollution Dispersion")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        stats = sim.get_concentration_statistics()
        return {
            "success": True,
            "output_path": output_path,
            "statistics": stats,
            "message": "pollution dispersion completed",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"污染物扩散模拟失败: {str(e)}",
            "error_trace": traceback.format_exc(),
        }


def main():
    if len(sys.argv) != 3:
        print("Usage: pollution_dispersion_wrapper.py input.json output.json")
        sys.exit(1)
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    try:
        import numpy as _np
        import matplotlib.pyplot as _plt
        from PythonEngine.simulation.pollution_dispersion import (
            PollutionDispersionSimulator as _Sim,
        )
    except Exception as e:
        err = {
            "success": False,
            "message": f"依赖加载失败: {e}",
            "error_trace": traceback.format_exc(),
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(err, f, ensure_ascii=False, indent=2)
        sys.exit(1)
    
    globals()["np"] = _np
    globals()["plt"] = _plt
    globals()["PollutionDispersionSimulator"] = _Sim

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        result = run_pollution_dispersion(input_data)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        sys.exit(0 if result.get("success") else 1)
    except Exception as e:
        err = {
            "success": False,
            "message": str(e),
            "error_trace": traceback.format_exc(),
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(err, f, ensure_ascii=False, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()