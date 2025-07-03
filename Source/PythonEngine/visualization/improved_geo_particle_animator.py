# ==============================================================================
# visualization/improved_geo_particle_animator.py
# ==============================================================================
"""
改进的地理粒子动画生成器 - 集成验证修复功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta
import json

# 导入验证修复模块
sys.path.append(str(Path(__file__).parent.parent))
from PythonEngine.wrappers.particle_validation_fix import (
    debug_netcdf_data,
    auto_fix_particle_positions,
    robust_particle_tracking_with_validation,
    create_test_positions_in_valid_area
)

try:
    from PythonEngine.utils.chinese_config import ChineseConfig
    chinese_config = ChineseConfig()
except ImportError:
    chinese_config = None


def create_robust_particle_animation(initial_positions: List[List[float]],
                                     netcdf_path: str,
                                     simulation_hours: float = 24.0,
                                     time_step_hours: float = 1.0,
                                     output_path: str = "particle_animation.gif",
                                     title: str = "海洋粒子轨迹模拟",
                                     auto_fix_positions: bool = True,
                                     fallback_to_test_positions: bool = True) -> Dict[str, Any]:
    """
    创建强健的地理粒子动画 - 自动处理验证和修复
    
    Args:
        initial_positions: 初始粒子位置 [[经度, 纬度], ...]
        netcdf_path: NetCDF数据文件路径
        simulation_hours: 模拟时长（小时）
        time_step_hours: 时间步长（小时）
        output_path: 输出路径
        title: 动画标题
        auto_fix_positions: 是否自动修复位置
        fallback_to_test_positions: 是否在失败时使用测试位置
    
    Returns:
        包含成功状态和结果信息的字典
    """
    try:
        logging.info(f"🌊 开始创建强健粒子动画: {len(initial_positions)}个粒子")

        # 步骤1: 调试和验证数据
        logging.info("📊 调试NetCDF数据...")
        debug_info = debug_netcdf_data(netcdf_path)

        if not debug_info['success']:
            return {
                'success': False,
                'message': f"NetCDF数据无效: {debug_info['error']}",
                'debug_info': debug_info
            }

        # 显示数据范围信息
        if 'time_0_data' in debug_info and 'lat_range' in debug_info['time_0_data']:
            extent = debug_info['time_0_data']
            logging.info(f"   数据范围: 经度 {extent['lon_range']}, 纬度 {extent['lat_range']}")
            logging.info(f"   有效点: {extent.get('u_valid_points', 0)}/{extent.get('total_points', 0)}")

        # 步骤2: 执行强健粒子追踪
        logging.info("🎯 执行强健粒子追踪...")

        tracking_input = {
            "parameters": {
                "netcdf_path": netcdf_path,
                "initial_positions": initial_positions,
                "simulation_hours": simulation_hours,
                "time_step_hours": time_step_hours,
                "auto_fix_positions": auto_fix_positions,
                "debug_mode": True
            }
        }

        tracking_result = robust_particle_tracking_with_validation(tracking_input)

        # 如果失败且允许备选方案，使用测试位置
        if (not tracking_result['success'] and
                fallback_to_test_positions and
                'position_fixes' not in tracking_result):

            logging.info("⚠️  使用原始位置失败，尝试生成测试位置...")

            try:
                test_positions = create_test_positions_in_valid_area(netcdf_path, len(initial_positions))
                logging.info(f"   生成了 {len(test_positions)} 个测试位置")

                # 使用测试位置重新尝试
                tracking_input["parameters"]["initial_positions"] = test_positions
                tracking_result = robust_particle_tracking_with_validation(tracking_input)

                if tracking_result['success']:
                    logging.info("✅ 使用测试位置成功")
                    tracking_result['used_test_positions'] = True
                    tracking_result['original_user_positions'] = initial_positions
                    tracking_result['generated_test_positions'] = test_positions

            except Exception as e:
                logging.error(f"生成测试位置失败: {e}")

        if not tracking_result['success']:
            return {
                'success': False,
                'message': f"粒子追踪失败: {tracking_result['message']}",
                'tracking_result': tracking_result,
                'debug_info': debug_info
            }

        # 步骤3: 创建地理动画
        logging.info("🎬 创建地理动画...")

        trajectories = tracking_result.get('trajectories', [])
        if not trajectories:
            return {
                'success': False,
                'message': "没有获得有效的轨迹数据"
            }

        # 生成时间步数组
        time_steps = [i * time_step_hours for i in range(len(trajectories))]

        # 获取使用的粒子位置
        used_positions = tracking_result.get('used_positions',
                                             tracking_result.get('generated_test_positions', initial_positions))

        # 创建动画
        anim_result = _create_geographic_animation(
            trajectories=trajectories,
            time_steps=time_steps,
            initial_positions=used_positions,
            title=title,
            output_path=output_path,
            debug_info=debug_info
        )

        if anim_result['success']:
            # 合并所有结果信息
            final_result = {
                'success': True,
                'message': "强健粒子动画创建成功",
                'output_path': anim_result['output_path'],
                'animation_stats': anim_result['animation_stats'],
                'tracking_stats': tracking_result.get('simulation_stats', {}),
                'position_fixes': tracking_result.get('position_fixes', []),
                'used_test_positions': tracking_result.get('used_test_positions', False),
                'debug_info': debug_info
            }

            # 如果使用了测试位置，添加说明
            if tracking_result.get('used_test_positions'):
                final_result['position_note'] = "由于原始位置验证失败，使用了自动生成的测试位置"
                final_result['original_positions'] = tracking_result.get('original_user_positions', [])
                final_result['test_positions'] = tracking_result.get('generated_test_positions', [])

            logging.info(f"✅ 强健粒子动画创建成功: {anim_result['output_path']}")
            return final_result
        else:
            return {
                'success': False,
                'message': f"动画创建失败: {anim_result['message']}",
                'tracking_result': tracking_result,
                'debug_info': debug_info
            }

    except Exception as e:
        logging.error(f"创建强健粒子动画失败: {e}")
        return {
            'success': False,
            'message': f"创建强健粒子动画失败: {str(e)}",
            'error_trace': str(e)
        }


def _create_geographic_animation(trajectories: List[List[List[float]]],
                                 time_steps: List[float],
                                 initial_positions: List[List[float]],
                                 title: str,
                                 output_path: str,
                                 debug_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建地理动画的内部函数
    
    Args:
        trajectories: 粒子轨迹数据
        time_steps: 时间步数组
        initial_positions: 初始位置
        title: 动画标题
        output_path: 输出路径
        debug_info: 调试信息
        
    Returns:
        动画创建结果
    """
    try:
        # 计算地理范围
        all_lons = []
        all_lats = []

        for frame in trajectories:
            for particle in frame:
                if len(particle) >= 2:
                    all_lons.append(particle[0])
                    all_lats.append(particle[1])

        if not all_lons or not all_lats:
            return {
                'success': False,
                'message': "轨迹数据为空或格式错误"
            }

        # 计算显示范围（增加边距）
        lon_margin = (max(all_lons) - min(all_lons)) * 0.1
        lat_margin = (max(all_lats) - min(all_lats)) * 0.1

        extent = [
            min(all_lons) - lon_margin,
            max(all_lons) + lon_margin,
            min(all_lats) - lat_margin,
            max(all_lats) + lat_margin
        ]

        # 创建图形
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # 添加地理要素
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray', alpha=0.8)

        # 网格线
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        # 设置粒子颜色
        n_particles = len(initial_positions)
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_particles, 10)))
        particle_colors = [colors[i % 10] for i in range(n_particles)]

        # 初始化粒子显示
        particles = ax.scatter([], [], s=50, c=particle_colors[:n_particles],
                               alpha=0.9, zorder=5, transform=ccrs.PlateCarree(),
                               edgecolors='black', linewidths=0.5)

        # 初始位置标记
        if initial_positions:
            initial_scatter = ax.scatter([pos[0] for pos in initial_positions],
                                         [pos[1] for pos in initial_positions],
                                         s=80, marker='*', c='yellow',
                                         edgecolors='black', linewidths=1,
                                         zorder=6, transform=ccrs.PlateCarree(),
                                         label='初始位置')

        # 轨迹线存储
        trail_lines = []
        trail_length = min(15, len(trajectories)//2)

        for i in range(n_particles):
            line, = ax.plot([], [], '-', color=particle_colors[i],
                            alpha=0.7, linewidth=2,
                            transform=ccrs.PlateCarree())
            trail_lines.append(line)

        # 信息显示
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            verticalalignment='top', fontsize=12,
                            bbox=dict(boxstyle='round,pad=0.5',
                                      facecolor='white', alpha=0.9),
                            zorder=10)

        # 标题
        ax.set_title(title, fontsize=16, pad=20)

        # 图例
        if initial_positions:
            ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.95))

        def animate(frame):
            """动画更新函数"""
            current_time = time_steps[frame]
            current_positions = trajectories[frame]

            # 更新粒子位置
            if len(current_positions) > 0:
                valid_positions = [pos for pos in current_positions if len(pos) >= 2]
                if valid_positions:
                    lons = [pos[0] for pos in valid_positions]
                    lats = [pos[1] for pos in valid_positions]
                    particles.set_offsets(np.column_stack([lons, lats]))

            # 更新轨迹尾迹
            start_frame = max(0, frame - trail_length)
            for i, line in enumerate(trail_lines):
                if i < len(current_positions):
                    # 获取该粒子的历史轨迹
                    traj_lons = []
                    traj_lats = []

                    for t in range(start_frame, frame + 1):
                        if (t < len(trajectories) and
                                i < len(trajectories[t]) and
                                len(trajectories[t][i]) >= 2):
                            pos = trajectories[t][i]
                            traj_lons.append(pos[0])
                            traj_lats.append(pos[1])

                    line.set_data(traj_lons, traj_lats)

                    # 设置透明度渐变
                    alpha = 0.8 * (len(traj_lons) / trail_length) if len(traj_lons) > 0 else 0
                    line.set_alpha(min(alpha, 0.8))

            # 更新信息显示
            active_particles = len([pos for pos in current_positions if len(pos) >= 2])
            elapsed_time = current_time

            # 计算粒子扩散统计
            if len(current_positions) > 1:
                valid_pos = [pos for pos in current_positions if len(pos) >= 2]
                if len(valid_pos) > 1:
                    positions_array = np.array(valid_pos)
                    center_lon = np.mean(positions_array[:, 0])
                    center_lat = np.mean(positions_array[:, 1])

                    # 计算扩散距离
                    distances = []
                    for pos in valid_pos:
                        dist = np.sqrt(
                            ((pos[0] - center_lon) * 111.32 * np.cos(np.radians(center_lat)))**2 +
                            ((pos[1] - center_lat) * 111.32)**2
                        )
                        distances.append(dist)

                    max_spread = np.max(distances) if distances else 0
                    mean_spread = np.mean(distances) if distances else 0
                else:
                    max_spread = mean_spread = 0
            else:
                max_spread = mean_spread = 0

            # 时间格式化
            if elapsed_time < 24:
                time_str = f"{elapsed_time:.1f} 小时"
            else:
                days = int(elapsed_time // 24)
                hours = elapsed_time % 24
                time_str = f"{days} 天 {hours:.1f} 小时"

            info_text_str = f'模拟时间: {time_str}\n'
            info_text_str += f'活跃粒子: {active_particles}/{n_particles}\n'
            info_text_str += f'最大扩散: {max_spread:.1f} km\n'
            info_text_str += f'平均扩散: {mean_spread:.1f} km'

            info_text.set_text(info_text_str)

            return [particles, info_text] + trail_lines

        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(trajectories),
                                       interval=1000//10, blit=False, repeat=True)

        plt.tight_layout()

        # 保存动画
        try:
            if output_path.endswith('.gif'):
                anim.save(output_path, writer='pillow', fps=10, dpi=100)
            elif output_path.endswith('.mp4'):
                anim.save(output_path, writer='ffmpeg', fps=10, dpi=100)
            else:
                output_path_gif = output_path + '.gif'
                anim.save(output_path_gif, writer='pillow', fps=10, dpi=100)
                output_path = output_path_gif

            # 保存轨迹数据
            _save_animation_metadata(trajectories, time_steps, initial_positions,
                                     output_path, debug_info)

            plt.close(fig)

            return {
                'success': True,
                'output_path': output_path,
                'animation_stats': {
                    'n_particles': n_particles,
                    'n_frames': len(trajectories),
                    'simulation_hours': time_steps[-1] if time_steps else 0,
                    'geographic_extent': extent
                }
            }

        except Exception as e:
            plt.close(fig)
            return {
                'success': False,
                'message': f"保存动画失败: {str(e)}"
            }

    except Exception as e:
        return {
            'success': False,
            'message': f"创建地理动画失败: {str(e)}"
        }


def _save_animation_metadata(trajectories: List[List[List[float]]],
                             time_steps: List[float],
                             initial_positions: List[List[float]],
                             output_path: str,
                             debug_info: Dict[str, Any]):
    """保存动画元数据"""
    try:
        data_path = output_path.replace('.gif', '_metadata.json').replace('.mp4', '_metadata.json')

        metadata = {
            "creation_time": datetime.now().isoformat(),
            "animation_info": {
                "n_particles": len(initial_positions),
                "n_frames": len(trajectories),
                "simulation_duration_hours": time_steps[-1] if time_steps else 0,
                "time_step_hours": time_steps[1] - time_steps[0] if len(time_steps) > 1 else 1.0
            },
            "initial_positions": initial_positions,
            "final_positions": trajectories[-1] if trajectories else [],
            "data_info": {
                "data_extent": debug_info.get('time_0_data', {}).get('lat_range', []) +
                               debug_info.get('time_0_data', {}).get('lon_range', []),
                "valid_data_points": debug_info.get('time_0_data', {}).get('u_valid_points', 0),
                "total_data_points": debug_info.get('time_0_data', {}).get('total_points', 0)
            }
        }

        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logging.info(f"动画元数据已保存: {data_path}")

    except Exception as e:
        logging.warning(f"保存动画元数据失败: {e}")


if __name__ == "__main__":
    # 测试改进的地理动画生成器
    import os

    print("🌊 测试改进的地理粒子动画生成器")
    print("-" * 50)

    # 测试配置
    test_netcdf_path = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/data/raw_data/merged_data.nc"

    # 测试原始位置（可能在陆地上）
    original_positions = [
        [120.5, 31.2],   # 上海附近
        [121.0, 31.0],   # 长江口
        [120.8, 30.8],   # 杭州湾
        [121.2, 31.5],   # 崇明岛附近
    ]

    print(f"初始位置: {len(original_positions)}个粒子")
    print(f"模拟时长: 48小时")

    # 创建输出目录
    os.makedirs("test_outputs", exist_ok=True)

    # 创建强健粒子动画
    result = create_robust_particle_animation(
        initial_positions=original_positions,
        netcdf_path=test_netcdf_path,
        simulation_hours=48.0,
        time_step_hours=2.0,
        output_path="test_outputs/robust_geo_particle_animation.gif",
        title="改进的长江口海域粒子轨迹模拟",
        auto_fix_positions=True,
        fallback_to_test_positions=True
    )

    print("\n📊 结果分析:")
    if result["success"]:
        print("✅ 动画创建成功")
        print(f"   输出文件: {result['output_path']}")

        # 显示位置处理信息
        if result.get('used_test_positions'):
            print("⚠️  使用了自动生成的测试位置")
            print(f"   原始位置数: {len(result.get('original_positions', []))}")
            print(f"   测试位置数: {len(result.get('test_positions', []))}")
        elif result.get('position_fixes'):
            print(f"🔧 应用了 {len(result['position_fixes'])} 个位置修复")
            for fix in result['position_fixes']:
                print(f"   粒子{fix['particle']}: {fix['reason']}")
        else:
            print("✨ 原始位置验证通过，无需修复")

        # 显示动画统计
        if 'animation_stats' in result:
            stats = result['animation_stats']
            print(f"📈 动画统计:")
            print(f"   帧数: {stats['n_frames']}")
            print(f"   粒子数: {stats['n_particles']}")
            print(f"   模拟时长: {stats['simulation_hours']} 小时")

        # 显示数据信息
        if 'debug_info' in result and result['debug_info'].get('success'):
            debug = result['debug_info']
            if 'time_0_data' in debug:
                data_info = debug['time_0_data']
                print(f"📊 数据质量:")
                print(f"   有效数据覆盖: {data_info.get('u_valid_points', 0)}/{data_info.get('total_points', 0)} "
                      f"({100*data_info.get('u_valid_points', 0)/max(data_info.get('total_points', 1), 1):.1f}%)")
                if 'lat_range' in data_info:
                    print(f"   地理范围: 纬度 {data_info['lat_range']}, 经度 {data_info['lon_range']}")
    else:
        print("❌ 动画创建失败")
        print(f"   错误信息: {result['message']}")

        # 显示调试信息
        if 'debug_info' in result:
            debug = result['debug_info']
            if debug.get('success'):
                print("📊 NetCDF数据调试成功，数据本身是有效的")
            else:
                print(f"📊 NetCDF数据问题: {debug.get('error', '未知')}")

    print("\n🎯 测试完成")

    if result["success"]:
        print("🎉 改进的地理动画生成器运行正常！")
        print(f"📁 查看输出文件: {result['output_path']}")
        print(f"📄 元数据文件: {result['output_path'].replace('.gif', '_metadata.json')}")
    else:
        print("⚠️  系统需要进一步调试，请检查错误信息。")