#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lagrangian particle tracking wrapper for C# integration"""

import sys
import json
import os
import traceback
from pathlib import Path

import numpy as np

try:
    import oceansim
except ImportError:
    oceansim = None

from PythonEngine.core.netcdf_handler import NetCDFHandler


def _plot_particle_tracks(trajectories, lon, lat, output_path):
    """Plot particle trajectories using cartopy."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from PythonEngine.utils.chinese_config import setup_chinese_all

    setup_chinese_all(font_size=12, dpi=120)

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_extent([
        float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
    ])

    for traj in trajectories:
        arr = np.array(traj)
        ax.plot(arr[:, 0], arr[:, 1], '-', transform=ccrs.PlateCarree(), linewidth=1)
        ax.plot(arr[0, 0], arr[0, 1], 'go', markersize=3, transform=ccrs.PlateCarree())
        ax.plot(arr[-1, 0], arr[-1, 1], 'ro', markersize=3, transform=ccrs.PlateCarree())

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def simulate_particle_tracking(params):
    """Run Lagrangian particle simulation and save trajectory plot."""
    netcdf_path = params.get('netcdf_path')
    time_index = params.get('time_index', 0)
    depth_index = params.get('depth_index', 0)
    dt = params.get('dt', 3600.0)
    steps = params.get('steps', 24)
    initial_positions = np.array(params.get('initial_positions', []), dtype=float)
    output_path = params.get('output_path', 'particle_tracks.png')

    if oceansim is None:
        raise RuntimeError('oceansim module not found')
    if initial_positions.size == 0:
        raise ValueError('initial_positions cannot be empty')

    handler = NetCDFHandler(netcdf_path)
    try:
        u, v, lat, lon = handler.get_uv(time_idx=time_index, depth_idx=depth_index)
        grid = oceansim.GridDataStructure(len(lon), len(lat), 1)
        grid.add_field2d('u_velocity', u.astype(float))
        grid.add_field2d('v_velocity', v.astype(float))
        try:
            grid.add_vector_field('velocity', [u.astype(float), v.astype(float), np.zeros_like(u, dtype=float)])
        except Exception:
            pass
        rk_solver = oceansim.RungeKuttaSolver()
        simulator = oceansim.ParticleSimulator(grid, rk_solver)

        lon0, lat0 = float(lon.min()), float(lat.min())
        dx = float(lon[1] - lon[0]) if len(lon) > 1 else 1.0
        dy = float(lat[1] - lat[0]) if len(lat) > 1 else 1.0

        init_particles = []
        for p in initial_positions:
            ix = float((p[0] - lon0) / dx)
            iy = float((p[1] - lat0) / dy)
            init_particles.append([ix, iy, 0.0])

        simulator.initialize_particles(init_particles)

        trajectories = []
        for _ in range(steps):
            simulator.step_forward(dt)
            parts = simulator.get_particles()
            frame = []
            for pt in parts:
                x = lon0 + pt.position[0] * dx
                y = lat0 + pt.position[1] * dy
                frame.append([x, y])
            trajectories.append(frame)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        _plot_particle_tracks(trajectories, lon, lat, output_path)

        return {
            'success': True,
            'message': '粒子追踪模拟完成',
            'output_path': output_path,
            'trajectories': trajectories
        }
    finally:
        handler.close()


def main():
    if len(sys.argv) != 3:
        print('用法: python lagrangian_particle_wrapper.py input.json output.json')
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        params = input_data.get('parameters', {})
        result = simulate_particle_tracking(params)
    except Exception as e:
        result = {
            'success': False,
            'message': f'粒子追踪模拟失败: {e}',
            'error_trace': traceback.format_exc()
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    sys.exit(0 if result.get('success') else 1)


if __name__ == "__main__":
    import subprocess
    from pathlib import Path
    import json

    # 测试输入参数
    test_input = {
        "action": "simulate_particle_tracking",
        "parameters": {
            "netcdf_path": str(Path("../../Source/Data/NetCDF/merged_data.nc").resolve()),
            "time_index": 0,
            "depth_index": 0,
            "dt": 3600.0,
            "steps": 12,
            "initial_positions": [
                [120.5, 30.5],
                [121.0, 30.8]
            ],
            "output_path": "test_outputs/particle_tracks_test.png"
        }
    }

    input_file = Path("test_inputs/input_particle.json")
    output_file = Path("test_outputs/output_particle.json")
    input_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(test_input, f, indent=2, ensure_ascii=False)

    # 调用 lagrangian_particle_wrapper
    process = subprocess.run(
        [
            "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/.venv/bin/python",
            "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/wrappers/lagrangian.py",
            str(input_file),
            str(output_file)
        ],
        capture_output=True,
        text=True
    )

    print("\n================ STDOUT ================\n")
    print(process.stdout)
    print("\n================ STDERR ================\n")
    print(process.stderr)

    # 检查结果
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            result = json.load(f)
        print("\n================ RESULT ================\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if result.get("success"):
            print("✅ 粒子追踪包装器执行成功，正在预览生成的粒子轨迹图。")

            # 显示图像
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            img = mpimg.imread(test_input["parameters"]["output_path"])
            plt.imshow(img)
            plt.axis("off")
            plt.title("Lagrangian Particle Tracks")
            plt.show()

        else:
            print("❌ 粒子追踪包装器执行失败，未生成可视化。")
    else:
        print("❌ 未生成输出 JSON 文件，包装器执行出错。")
