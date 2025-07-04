# ==============================================================================
# wrappers/geo_animation_wrapper.py
# ==============================================================================
"""
地理动画包装器 - 处理C#调用的地理粒子动画请求
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Dict, Any, List, Optional, Tuple
import logging
import traceback
import json
from pathlib import Path
import sys

# 导入相关模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # ensure 'Source' on path

from PythonEngine.visualization.geo_particle_animator import GeoParticleAnimator, create_simple_particle_animation
from PythonEngine.wrappers.lagrangian_particle_wrapper import simulate_particle_tracking
from PythonEngine.wrappers.ocean_data_wrapper import NetCDFHandler


def create_geo_particle_animation(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建地理底图粒子轨迹动画
    
    Args:
        input_data: 输入参数字典
        
    Returns:
        包含成功状态和结果的字典
    """
    try:
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        initial_positions = params.get('initial_positions', [])
        simulation_hours = params.get('simulation_hours', 24.0)
        time_step_hours = params.get('time_step_hours', 1.0)
        output_path = params.get('output_path', 'geo_particle_animation.gif')
        title = params.get('title', '海洋粒子轨迹模拟')

        # 动画配置参数
        fps = params.get('fps', 10)
        trail_length = params.get('trail_length', 15)
        show_coastlines = params.get('show_coastlines', True)
        show_land = params.get('show_land', True)
        show_ocean = params.get('show_ocean', True)
        show_gridlines = params.get('show_gridlines', True)

        logging.info(f"开始创建地理粒子动画: {len(initial_positions)}个粒子, {simulation_hours}小时")

        # 验证输入参数
        if not initial_positions:
            raise ValueError("初始粒子位置不能为空")

        if not netcdf_path or not Path(netcdf_path).exists():
            raise FileNotFoundError(f"NetCDF文件不存在: {netcdf_path}")

        # 使用简化接口创建动画
        result = create_simple_particle_animation(
            initial_positions=initial_positions,
            netcdf_path=netcdf_path,
            simulation_hours=simulation_hours,
            time_step_hours=time_step_hours,
            output_path=output_path,
            title=title
        )

        if result["success"]:
            # 添加地理范围信息
            animator = GeoParticleAnimator()
            if animator.extent:
                result["geographic_extent"] = animator.extent

            return {
                "success": True,
                "message": "地理粒子动画创建成功",
                "output_path": result["output_path"],
                "simulation_stats": result["simulation_stats"],
                "geographic_extent": result.get("geographic_extent")
            }
        else:
            return {
                "success": False,
                "message": result["message"],
                "error_details": result
            }

    except Exception as e:
        logging.error(f"创建地理粒子动画失败: {e}")
        return {
            "success": False,
            "message": f"创建地理粒子动画失败: {str(e)}",
            "error_trace": traceback.format_exc()
        }


def create_interactive_release_animation(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建交互式粒子释放动画
    
    Args:
        input_data: 输入参数字典
        
    Returns:
        包含成功状态和结果的字典
    """
    try:
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        release_schedule = params.get('release_schedule', [])
        total_simulation_hours = params.get('total_simulation_hours', 48.0)
        time_step_hours = params.get('time_step_hours', 1.0)
        output_path = params.get('output_path', 'interactive_release_animation.gif')
        title = params.get('title', '交互式粒子释放模拟')

        logging.info(f"创建交互式粒子释放动画: {len(release_schedule)}个释放事件")

        # 模拟交互式释放
        all_trajectories = []
        all_time_steps = []
        particle_metadata = []

        # 计算总时间步数
        total_steps = int(total_simulation_hours / time_step_hours)
        time_steps = [i * time_step_hours for i in range(total_steps)]

        # 处理每个释放事件
        for event in release_schedule:
            release_time = event.get('time_hours', 0)
            positions = event.get('positions', [])
            properties = event.get('particle_properties', {})

            if not positions:
                continue

            # 计算释放时间对应的步数
            release_step = int(release_time / time_step_hours)

            # 执行粒子追踪（从释放时间开始）
            remaining_hours = total_simulation_hours - release_time
            if remaining_hours <= 0:
                continue

            tracking_result = simulate_particle_tracking({
                "parameters": {
                    "netcdf_path": netcdf_path,
                    "initial_positions": positions,
                    "dt": time_step_hours * 3600,
                    "steps": int(remaining_hours / time_step_hours),
                    "time_index": 0,
                    "depth_index": 0
                }
            })

            if tracking_result.get("success"):
                trajectories = tracking_result.get("trajectories", [])

                # 将轨迹插入到正确的时间位置
                for i, traj_frame in enumerate(trajectories):
                    step_index = release_step + i
                    if step_index < len(all_trajectories):
                        all_trajectories[step_index].extend(traj_frame)
                    else:
                        # 扩展轨迹列表
                        while len(all_trajectories) <= step_index:
                            all_trajectories.append([])
                        all_trajectories[step_index] = traj_frame

                # 记录粒子元数据
                for pos in positions:
                    particle_metadata.append({
                        "release_time": release_time,
                        "initial_position": pos,
                        "properties": properties
                    })

        # 确保所有时间步都有轨迹数据
        for i in range(len(time_steps)):
            if i >= len(all_trajectories):
                all_trajectories.append([])

        # 获取所有初始位置
        all_initial_positions = []
        for event in release_schedule:
            all_initial_positions.extend(event.get('positions', []))

        # 创建动画
        animator = GeoParticleAnimator()
        anim = animator.create_particle_trajectory_animation(
            trajectories=all_trajectories,
            time_steps=time_steps,
            initial_positions=all_initial_positions,
            title=title,
            trail_length=params.get('trail_length', 20),
            save_path=output_path,
            fps=params.get('fps', 10),
            show_coastlines=params.get('show_coastlines', True),
            show_land=params.get('show_land', True),
            show_ocean=params.get('show_ocean', True),
            show_gridlines=params.get('show_gridlines', True)
        )

        return {
            "success": True,
            "message": "交互式粒子释放动画创建成功",
            "output_path": output_path,
            "simulation_stats": {
                "total_particles": len(all_initial_positions),
                "release_events": len(release_schedule),
                "simulation_hours": total_simulation_hours,
                "n_time_steps": len(time_steps)
            },
            "release_schedule": release_schedule,
            "geographic_extent": animator.extent
        }

    except Exception as e:
        logging.error(f"创建交互式粒子释放动画失败: {e}")
        return {
            "success": False,
            "message": f"创建交互式粒子释放动画失败: {str(e)}",
            "error_trace": traceback.format_exc()
        }


def create_particle_density_heatmap(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建粒子密度热力图动画
    
    Args:
        input_data: 输入参数字典
        
    Returns:
        包含成功状态和结果的字典
    """
    try:
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        initial_positions = params.get('initial_positions', [])
        simulation_hours = params.get('simulation_hours', 24.0)
        time_step_hours = params.get('time_step_hours', 1.0)
        grid_resolution = params.get('grid_resolution', 0.01)
        output_path = params.get('output_path', 'density_heatmap.gif')
        title = params.get('title', '粒子密度分布动画')
        colormap = params.get('colormap', 'hot')
        show_particles = params.get('show_particles', True)
        density_smoothing = params.get('density_smoothing', 1.0)

        logging.info(f"创建粒子密度热力图: {len(initial_positions)}个粒子")

        # 执行粒子追踪
        tracking_result = simulate_particle_tracking({
            "parameters": {
                "netcdf_path": netcdf_path,
                "initial_positions": initial_positions,
                "dt": time_step_hours * 3600,
                "steps": int(simulation_hours / time_step_hours),
                "time_index": 0,
                "depth_index": 0
            }
        })

        if not tracking_result.get("success"):
            raise ValueError(f"粒子追踪失败: {tracking_result.get('message')}")

        trajectories = tracking_result.get("trajectories", [])
        time_steps = [i * time_step_hours for i in range(len(trajectories))]

        # 创建密度热力图动画
        density_anim = _create_density_heatmap_animation(
            trajectories=trajectories,
            time_steps=time_steps,
            grid_resolution=grid_resolution,
            title=title,
            colormap=colormap,
            show_particles=show_particles,
            density_smoothing=density_smoothing,
            output_path=output_path
        )

        return {
            "success": True,
            "message": "粒子密度热力图动画创建成功",
            "output_path": output_path,
            "simulation_stats": {
                "n_particles": len(initial_positions),
                "simulation_hours": simulation_hours,
                "n_time_steps": len(trajectories),
                "grid_resolution": grid_resolution
            }
        }

    except Exception as e:
        logging.error(f"创建粒子密度热力图失败: {e}")
        return {
            "success": False,
            "message": f"创建粒子密度热力图失败: {str(e)}",
            "error_trace": traceback.format_exc()
        }


def _create_density_heatmap_animation(trajectories: List[List[List[float]]],
                                      time_steps: List[float],
                                      grid_resolution: float,
                                      title: str,
                                      colormap: str,
                                      show_particles: bool,
                                      density_smoothing: float,
                                      output_path: str) -> plt.Figure:
    """创建密度热力图动画的内部函数"""

    # 计算地理范围
    all_lons = []
    all_lats = []
    for frame in trajectories:
        for particle in frame:
            all_lons.append(particle[0])
            all_lats.append(particle[1])

    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)

    # 创建密度网格
    lon_grid = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
    lat_grid = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 添加地理要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    def animate(frame):
        ax.clear()

        # 重新添加地理要素
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # 计算当前帧的密度
        current_positions = trajectories[frame]
        if current_positions:
            positions_array = np.array(current_positions)

            # 计算二维直方图（密度）
            density, _, _ = np.histogram2d(
                positions_array[:, 1], positions_array[:, 0],  # lat, lon
                bins=[lat_grid, lon_grid]
            )

            # 应用平滑
            if density_smoothing > 0:
                from scipy import ndimage
                density = ndimage.gaussian_filter(density, sigma=density_smoothing)

            # 绘制密度热力图
            if np.max(density) > 0:
                cs = ax.contourf(LON, LAT, density, levels=20, cmap=colormap,
                                 alpha=0.8, transform=ccrs.PlateCarree())

            # 可选：显示粒子点
            if show_particles:
                ax.scatter(positions_array[:, 0], positions_array[:, 1],
                           s=10, c='red', alpha=0.6, transform=ccrs.PlateCarree())

        ax.set_title(f'{title} - 时间: {time_steps[frame]:.1f}h', fontsize=14)

        return []

    # 创建动画
    import matplotlib.animation as animation
    anim = animation.FuncAnimation(fig, animate, frames=len(trajectories),
                                   interval=100, blit=False, repeat=True)

    # 保存动画
    if output_path.endswith('.gif'):
        anim.save(output_path, writer='pillow', fps=10)
    elif output_path.endswith('.mp4'):
        anim.save(output_path, writer='ffmpeg', fps=10)

    plt.close(fig)
    return fig


# 主要动作分发器
def handle_geo_animation_request(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理地理动画请求的主分发器
    
    Args:
        input_data: 包含action和parameters的请求数据
        
    Returns:
        处理结果
    """
    action = input_data.get("action", "")

    action_handlers = {
        "create_geo_particle_animation": create_geo_particle_animation,
        "create_interactive_release_animation": create_interactive_release_animation,
        "create_particle_density_heatmap": create_particle_density_heatmap,
    }

    handler = action_handlers.get(action)
    if handler:
        return handler(input_data)
    else:
        return {
            "success": False,
            "message": f"未知的动作类型: {action}",
            "available_actions": list(action_handlers.keys())
        }


if __name__ == "__main__":
    # 测试地理动画包装器
    import os

    # 测试配置
    test_netcdf_path = "/path/to/your/ocean_data.nc"  # 需要替换为实际路径

    print("🎬 测试地理粒子动画包装器")
    print("-" * 50)

    # 创建测试输出目录
    os.makedirs("test_outputs", exist_ok=True)

    # 测试1: 基础地理粒子动画
    print("📍 测试1: 基础地理粒子动画")
    test_input_1 = {
        "action": "create_geo_particle_animation",
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "initial_positions": [
                [120.5, 31.2],
                [121.0, 31.0],
                [120.8, 30.8]
            ],
            "simulation_hours": 24.0,
            "time_step_hours": 2.0,
            "output_path": "test_outputs/basic_geo_animation.gif",
            "title": "基础粒子轨迹测试"
        }
    }

    result_1 = handle_geo_animation_request(test_input_1)
    print(f"结果: {'✅ 成功' if result_1['success'] else '❌ 失败'}")
    if not result_1['success']:
        print(f"错误: {result_1['message']}")

    # 测试2: 交互式释放动画
    print("\n🎯 测试2: 交互式释放动画")
    test_input_2 = {
        "action": "create_interactive_release_animation",
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "release_schedule": [
                {
                    "time_hours": 0,
                    "positions": [[120.5, 31.2], [120.6, 31.1]],
                    "particle_properties": {"type": "oil", "volume": 100}
                },
                {
                    "time_hours": 12,
                    "positions": [[121.0, 31.0]],
                    "particle_properties": {"type": "debris", "mass": 50}
                }
            ],
            "total_simulation_hours": 36.0,
            "time_step_hours": 2.0,
            "output_path": "test_outputs/interactive_release.gif",
            "title": "交互式粒子释放测试"
        }
    }

    result_2 = handle_geo_animation_request(test_input_2)
    print(f"结果: {'✅ 成功' if result_2['success'] else '❌ 失败'}")
    if not result_2['success']:
        print(f"错误: {result_2['message']}")

    # 测试3: 密度热力图
    print("\n🔥 测试3: 粒子密度热力图")
    test_input_3 = {
        "action": "create_particle_density_heatmap",
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "initial_positions": [
                [120.4, 31.3], [120.5, 31.2], [120.6, 31.1],
                [120.7, 31.0], [120.8, 30.9], [120.9, 30.8]
            ],
            "simulation_hours": 48.0,
            "time_step_hours": 3.0,
            "grid_resolution": 0.02,
            "output_path": "test_outputs/density_heatmap.gif",
            "title": "粒子密度热力图测试",
            "colormap": "hot",
            "show_particles": True,
            "density_smoothing": 1.5
        }
    }

    result_3 = handle_geo_animation_request(test_input_3)
    print(f"结果: {'✅ 成功' if result_3['success'] else '❌ 失败'}")
    if not result_3['success']:
        print(f"错误: {result_3['message']}")

    # 总结测试结果
    print("\n" + "=" * 50)
    print("🎯 测试总结")
    print("-" * 30)

    success_count = sum([result_1['success'], result_2['success'], result_3['success']])
    total_tests = 3

    print(f"成功: {success_count}/{total_tests}")
    print(f"基础动画: {'✅' if result_1['success'] else '❌'}")
    print(f"交互释放: {'✅' if result_2['success'] else '❌'}")
    print(f"密度热力图: {'✅' if result_3['success'] else '❌'}")

    if success_count == total_tests:
        print("\n🎉 所有测试通过！地理动画包装器运行正常。")

        # 显示生成的文件
        print("\n📁 生成的动画文件:")
        for filename in ["basic_geo_animation.gif", "interactive_release.gif", "density_heatmap.gif"]:
            filepath = f"test_outputs/{filename}"
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"   {filename} ({size_mb:.1f} MB)")
    else:
        print(f"\n⚠️  {total_tests - success_count} 个测试失败，请检查错误信息。")

    print("=" * 50)