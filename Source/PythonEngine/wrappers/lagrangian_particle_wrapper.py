import numpy as np
import traceback
import oceansim
from PythonEngine.wrappers.ocean_data_wrapper import NetCDFHandler
from PythonEngine.wrappers.ocean_data_wrapper import check_grid_data_formats
from PythonEngine.wrappers.ocean_data_wrapper import calculate_vorticity_divergence
from PythonEngine.wrappers.ocean_data_wrapper import calculate_flow_statistics
import os


def validate_particle_positions_and_time(netcdf_path, initial_positions, time_index=0, simulation_days=1):
    """
    验证粒子位置是否在水域内，并检查时间范围
    """
    try:
        handler = NetCDFHandler(netcdf_path)
        try:
            # 获取时间信息
            ds = handler.ds
            total_time_steps = ds.sizes.get('time', 1)

            # 验证时间范围
            max_available_days = total_time_steps - time_index
            if time_index >= total_time_steps:
                return {
                    "success": False,
                    "error": f"起始时间索引{time_index}超出数据范围(0-{total_time_steps-1})"
                }

            if simulation_days > max_available_days:
                return {
                    "success": False,
                    "error": f"模拟天数{simulation_days}超出可用数据范围，最多可模拟{max_available_days}天",
                    "max_days": max_available_days,
                    "time_info": {
                        "total_time_steps": total_time_steps,
                        "start_index": time_index,
                        "available_days": max_available_days
                    }
                }

            # 获取速度场创建水域掩膜
            u, v, lat, lon = handler.get_uv(time_idx=time_index, depth_idx=0)
            water_mask = ~np.isnan(u) & ~np.isnan(v) & np.isfinite(u) & np.isfinite(v)

            # 验证粒子位置
            valid_positions = []
            invalid_positions = []

            for i, pos in enumerate(initial_positions):
                lon_val, lat_val = float(pos[0]), float(pos[1])

                # 检查是否在地理范围内
                if (lon_val < lon.min() or lon_val > lon.max() or
                        lat_val < lat.min() or lat_val > lat.max()):
                    invalid_positions.append({
                        "index": i,
                        "position": [lon_val, lat_val],
                        "reason": "超出数据地理范围"
                    })
                    continue

                # 转换为网格索引
                lon_idx = np.argmin(np.abs(lon - lon_val))
                lat_idx = np.argmin(np.abs(lat - lat_val))

                # 检查是否在水域
                if water_mask[lat_idx, lon_idx]:
                    valid_positions.append([lon_val, lat_val])
                else:
                    # 尝试寻找附近的水域点
                    found_water = False
                    search_radius = 3  # 搜索半径（网格点）

                    for di in range(-search_radius, search_radius + 1):
                        for dj in range(-search_radius, search_radius + 1):
                            new_lat_idx = lat_idx + di
                            new_lon_idx = lon_idx + dj

                            if (0 <= new_lat_idx < len(lat) and
                                    0 <= new_lon_idx < len(lon) and
                                    water_mask[new_lat_idx, new_lon_idx]):

                                suggested_pos = [float(lon[new_lon_idx]), float(lat[new_lat_idx])]
                                invalid_positions.append({
                                    "index": i,
                                    "position": [lon_val, lat_val],
                                    "reason": "位于陆地区域",
                                    "suggested_position": suggested_pos,
                                    "distance_km": np.sqrt(
                                        ((lon[new_lon_idx] - lon_val) * 111.32 * np.cos(np.radians(lat_val)))**2 +
                                        ((lat[new_lat_idx] - lat_val) * 111.32)**2
                                    )
                                })
                                found_water = True
                                break
                        if found_water:
                            break

                    if not found_water:
                        invalid_positions.append({
                            "index": i,
                            "position": [lon_val, lat_val],
                            "reason": "位于陆地区域且附近无水域"
                        })

            return {
                "success": len(invalid_positions) == 0,
                "valid_positions": valid_positions,
                "invalid_positions": invalid_positions,
                "time_info": {
                    "total_time_steps": total_time_steps,
                    "max_available_days": max_available_days,
                    "start_index": time_index
                },
                "message": f"验证完成: {len(valid_positions)}个有效位置, {len(invalid_positions)}个无效位置"
            }

        finally:
            handler.close()

    except Exception as e:
        return {
            "success": False,
            "error": f"位置和时间验证失败: {str(e)}"
        }

def simulate_particle_tracking(input_data):
    """拉格朗日粒子追踪模拟"""
    try:
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        time_index = params.get('time_index', 0)
        depth_index = params.get('depth_index', 0)
        dt = params.get('dt', 3600.0)
        steps = params.get('steps', 24)
        initial_positions = np.array(params.get('initial_positions', []), dtype=float)
        output_path = params.get('output_path', 'particle_tracks.png')

        # 计算模拟天数
        simulation_days = (steps * dt) / (24 * 3600)

        if initial_positions.size == 0:
            raise ValueError('initial_positions 不能为空')



        if initial_positions.size == 0:
            raise ValueError('initial_positions 不能为空')

        print(f"[INFO] 验证粒子位置和时间范围...")

        # 新增：验证粒子位置和时间范围
        validation_result = validate_particle_positions_and_time(
            netcdf_path, initial_positions, time_index, simulation_days
        )

        if not validation_result["success"]:
            return {
                "success": False,
                "message": validation_result.get("error", "验证失败"),
                "validation_details": validation_result,
                "suggested_alternatives": validation_result.get("invalid_positions", [])
            }

        # 如果有无效位置，返回详细信息
        if validation_result.get("invalid_positions"):
            return {
                "success": False,
                "message": "部分粒子位置无效",
                "validation_details": validation_result,
                "invalid_positions": validation_result["invalid_positions"]
            }



        print(f"[INFO] 位置验证通过，运行粒子追踪模拟: 时间索引{time_index}, 深度索引{depth_index}")

        handler = NetCDFHandler(netcdf_path)
        try:
            u, v, lat, lon = handler.get_uv(time_idx=time_index, depth_idx=depth_index)

            test_result = check_grid_data_formats(u, v, lat, lon)
            if not test_result["success"]:
                raise ValueError(f"无法找到兼容的网格数据格式: {test_result.get('error', '未知错误')}")

            grid_params = test_result["grid_params"]
            u_data = test_result.get("u_data")
            v_data = test_result.get("v_data")
            if u_data is None or v_data is None:
                u_data = u.astype(np.float64)
                v_data = v.astype(np.float64)

            grid = oceansim.GridDataStructure(*grid_params)
            grid.add_field2d("u_velocity", u_data)
            grid.add_field2d("v_velocity", v_data)
            try:
                grid.add_vector_field("velocity", [u_data, v_data, np.zeros_like(u_data)])
            except Exception:
                pass

            rk_solver = oceansim.RungeKuttaSolver()
            simulator = oceansim.ParticleSimulator(grid, rk_solver)

            lon0, lat0 = float(lon.min()), float(lat.min())
            dx = float(lon[1]-lon[0]) if len(lon) > 1 else 1.0
            dy = float(lat[1]-lat[0]) if len(lat) > 1 else 1.0

            init_particles = []
            for p in initial_positions:
                ix = (p[0] - lon0) / dx
                iy = (p[1] - lat0) / dy
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

            _plot_particle_tracks(trajectories, lon, lat, output_path)

            return {
                "success": True,
                "message": "粒子追踪模拟完成",
                "output_path": output_path,
                "trajectories": trajectories,
            }
        finally:
            handler.close()
    except Exception as e:
        return {
            "success": False,
            "message": f"粒子追踪模拟失败: {str(e)}",
            "error_trace": traceback.format_exc(),
        }


def _plot_particle_tracks(trajectories, lon, lat, output_path):
    """绘制粒子轨迹"""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from Source.PythonEngine.utils.chinese_config import setup_chinese_all

    setup_chinese_all(font_size=12, dpi=120)

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())])

    for traj in trajectories:
        arr = np.array(traj)
        ax.plot(arr[:,0], arr[:,1], '-', transform=ccrs.PlateCarree(), linewidth=1)
        ax.plot(arr[0,0], arr[0,1], 'go', markersize=3, transform=ccrs.PlateCarree())
        ax.plot(arr[-1,0], arr[-1,1], 'ro', markersize=3, transform=ccrs.PlateCarree())

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
if __name__ == '__main__':
    # ========== 测试3: 拉格朗日粒子追踪 ==========

    test_netcdf_path = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/data/raw_data/merged_data.nc"


    test_input_vort = {
        "action": "calculate_vorticity_divergence",
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "output_path": "test_outputs/vorticity_divergence_analysis.png",
            "time_index": 0,
            "depth_index": 0
        }
    }

    test_input_flow = {
        "action": "calculate_flow_statistics",
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "time_index": 0,
            "depth_index": 0
        }
    }
    
    # 创建输出目录
    os.makedirs("test_outputs", exist_ok=True)


    result_vort = calculate_vorticity_divergence(test_input_vort)
    result_flow = calculate_flow_statistics(test_input_flow)

    print("🔄 测试3: 拉格朗日粒子追踪")
    print("-" * 40)

    # 选择两个初始粒子位置用于示例
    handler = NetCDFHandler(test_netcdf_path)
    u_tmp, v_tmp, lat_tmp, lon_tmp = handler.get_uv(time_idx=0, depth_idx=0)
    handler.close()
    init_positions = [
        [float(lon_tmp[len(lon_tmp)//2]), float(lat_tmp[len(lat_tmp)//2])],
        [float(lon_tmp[len(lon_tmp)//3]), float(lat_tmp[len(lat_tmp)//3])]
    ]

    test_input_particles = {
        "action": "simulate_particle_tracking",
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "time_index": 0,
            "depth_index": 0,
            "initial_positions": init_positions,
            "dt": 3600.0,
            "steps": 12,
            "output_path": "test_outputs/particle_tracks.png"
        }
    }

    print(f"⚙️  粒子数: {len(init_positions)}, 步数: {test_input_particles['parameters']['steps']}")

    result_particles = simulate_particle_tracking(test_input_particles)

    print("📊 粒子追踪结果:")
    if result_particles["success"]:
        print("✅ 模拟成功")
        print(f"📈 轨迹图: {result_particles.get('output_path', '未生成')}")
    else:
        print("❌ 模拟失败")
        print(f"错误信息: {result_particles.get('message', '未知错误')}")

    print("\n" + "=" * 60)
    print("🎯 测试完成总结")
    print("-" * 40)

    # 测试结果总结
    vort_success = result_vort.get("success", False)
    flow_success = result_flow.get("success", False)
    particle_success = result_particles.get("success", False)

    print(f"涡度散度场计算: {'✅ 成功' if vort_success else '❌ 失败'}")
    print(f"流速统计分析: {'✅ 成功' if flow_success else '❌ 失败'}")
    print(f"粒子追踪模拟: {'✅ 成功' if particle_success else '❌ 失败'}")

    if vort_success and flow_success and particle_success:
        print("\n🎉 所有测试通过！海洋统计分析模块运行正常。")

        # 显示生成的文件
        print("\n📁 生成的输出文件:")
        output_dir = "test_outputs"
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size_kb = os.path.getsize(file_path) / 1024
                    print(f"   {file} ({size_kb:.1f} KB)")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复相关问题。")

    print("=" * 60)
