# ==============================================================================
# visualization/pure_python_particle_animator.py
# ==============================================================================
"""
纯Python粒子动画生成器 - 避免C++模块段错误问题
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
from scipy import interpolate

# 导入NetCDF处理
sys.path.append(str(Path(__file__).parent.parent))
from PythonEngine.wrappers.ocean_data_wrapper import NetCDFHandler

try:
    from PythonEngine.utils.chinese_config import ChineseConfig
    chinese_config = ChineseConfig()
except ImportError:
    chinese_config = None


class PurePythonParticleTracker:
    """纯Python实现的粒子追踪器，避免C++模块问题"""

    def __init__(self, netcdf_path: str):
        """
        初始化粒子追踪器
        
        Args:
            netcdf_path: NetCDF数据文件路径
        """
        self.netcdf_path = netcdf_path
        self.handler = None
        self.u_data = None
        self.v_data = None
        self.lat = None
        self.lon = None
        self.water_mask = None

    def initialize(self, time_idx: int = 0, depth_idx: int = 0) -> bool:
        """
        初始化数据
        
        Args:
            time_idx: 时间索引
            depth_idx: 深度索引
            
        Returns:
            是否成功初始化
        """
        try:
            self.handler = NetCDFHandler(self.netcdf_path)

            # 获取速度场数据
            self.u_data, self.v_data, self.lat, self.lon = self.handler.get_uv(
                time_idx=time_idx, depth_idx=depth_idx
            )

            # 创建水域掩膜
            self.water_mask = (~np.isnan(self.u_data) &
                               ~np.isnan(self.v_data) &
                               np.isfinite(self.u_data) &
                               np.isfinite(self.v_data))

            # 将NaN替换为0以避免插值问题
            self.u_data = np.nan_to_num(self.u_data, nan=0.0)
            self.v_data = np.nan_to_num(self.v_data, nan=0.0)

            logging.info(f"初始化成功: {np.sum(self.water_mask)} 个有效水域点")
            return True

        except Exception as e:
            logging.error(f"初始化失败: {e}")
            return False

    def validate_and_fix_positions(self, positions: List[List[float]]) -> Tuple[List[List[float]], List[Dict]]:
        """
        验证和修复粒子位置
        
        Args:
            positions: 原始粒子位置 [[lon, lat], ...]
            
        Returns:
            (修复后的位置, 修复日志)
        """
        fixed_positions = []
        fix_log = []

        for i, pos in enumerate(positions):
            lon_val, lat_val = float(pos[0]), float(pos[1])

            # 检查是否在数据范围内
            if (lon_val < self.lon.min() or lon_val > self.lon.max() or
                    lat_val < self.lat.min() or lat_val > self.lat.max()):

                # 调整到数据范围内
                fixed_lon = np.clip(lon_val, self.lon.min(), self.lon.max())
                fixed_lat = np.clip(lat_val, self.lat.min(), self.lat.max())

                fix_log.append({
                    'particle': i,
                    'original': [lon_val, lat_val],
                    'fixed': [float(fixed_lon), float(fixed_lat)],
                    'reason': '调整到数据范围内'
                })

                lon_val, lat_val = fixed_lon, fixed_lat

            # 检查是否在水域中
            lon_idx = np.argmin(np.abs(self.lon - lon_val))
            lat_idx = np.argmin(np.abs(self.lat - lat_val))

            if self.water_mask[lat_idx, lon_idx]:
                # 已在水域中
                fixed_positions.append([lon_val, lat_val])
            else:
                # 寻找最近的水域点
                water_indices = np.where(self.water_mask)
                if len(water_indices[0]) > 0:
                    # 计算到所有水域点的距离
                    distances = np.sqrt(
                        ((self.lon[water_indices[1]] - lon_val) * 111.32 * np.cos(np.radians(lat_val)))**2 +
                        ((self.lat[water_indices[0]] - lat_val) * 111.32)**2
                    )

                    # 找到最近的水域点
                    nearest_idx = np.argmin(distances)
                    nearest_lat_idx = water_indices[0][nearest_idx]
                    nearest_lon_idx = water_indices[1][nearest_idx]

                    fixed_pos = [float(self.lon[nearest_lon_idx]), float(self.lat[nearest_lat_idx])]
                    fixed_positions.append(fixed_pos)

                    fix_log.append({
                        'particle': i,
                        'original': [lon_val, lat_val],
                        'fixed': fixed_pos,
                        'distance_km': distances[nearest_idx],
                        'reason': '移动到最近水域'
                    })
                else:
                    # 备选方案：使用数据中心点
                    center_pos = [float(self.lon[len(self.lon)//2]), float(self.lat[len(self.lat)//2])]
                    fixed_positions.append(center_pos)

                    fix_log.append({
                        'particle': i,
                        'original': [lon_val, lat_val],
                        'fixed': center_pos,
                        'reason': '使用数据中心点'
                    })

        return fixed_positions, fix_log

    def interpolate_velocity(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        插值计算指定位置的速度
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            (u速度, v速度)
        """
        try:
            # 双线性插值
            lon_idx = np.interp(lon, self.lon, np.arange(len(self.lon)))
            lat_idx = np.interp(lat, self.lat, np.arange(len(self.lat)))

            # 确保索引在有效范围内
            lon_idx = np.clip(lon_idx, 0, len(self.lon) - 1)
            lat_idx = np.clip(lat_idx, 0, len(self.lat) - 1)

            # 获取周围四个点的索引
            lon_i0 = int(np.floor(lon_idx))
            lon_i1 = min(lon_i0 + 1, len(self.lon) - 1)
            lat_j0 = int(np.floor(lat_idx))
            lat_j1 = min(lat_j0 + 1, len(self.lat) - 1)

            # 插值权重
            wx = lon_idx - lon_i0
            wy = lat_idx - lat_j0

            # 双线性插值
            u_interp = (
                    (1 - wx) * (1 - wy) * self.u_data[lat_j0, lon_i0] +
                    wx * (1 - wy) * self.u_data[lat_j0, lon_i1] +
                    (1 - wx) * wy * self.u_data[lat_j1, lon_i0] +
                    wx * wy * self.u_data[lat_j1, lon_i1]
            )

            v_interp = (
                    (1 - wx) * (1 - wy) * self.v_data[lat_j0, lon_i0] +
                    wx * (1 - wy) * self.v_data[lat_j0, lon_i1] +
                    (1 - wx) * wy * self.v_data[lat_j1, lon_i0] +
                    wx * wy * self.v_data[lat_j1, lon_i1]
            )

            return float(u_interp), float(v_interp)

        except Exception as e:
            logging.warning(f"速度插值失败: {e}")
            return 0.0, 0.0

    def rk4_step(self, positions: np.ndarray, dt: float) -> np.ndarray:
        """
        RK4时间积分步
        
        Args:
            positions: 粒子位置数组 (N, 2)
            dt: 时间步长（秒）
            
        Returns:
            更新后的位置数组
        """
        # 转换时间步长为小时（用于经纬度单位转换）
        dt_hours = dt / 3600.0

        # RK4积分
        k1 = np.zeros_like(positions)
        k2 = np.zeros_like(positions)
        k3 = np.zeros_like(positions)
        k4 = np.zeros_like(positions)

        for i, pos in enumerate(positions):
            lon, lat = pos[0], pos[1]

            # k1
            u, v = self.interpolate_velocity(lon, lat)
            # 转换速度单位：m/s -> 度/小时
            k1[i, 0] = u * dt_hours / (111320 * np.cos(np.radians(lat)))  # 经度变化
            k1[i, 1] = v * dt_hours / 111320  # 纬度变化

            # k2
            u, v = self.interpolate_velocity(lon + k1[i, 0]/2, lat + k1[i, 1]/2)
            k2[i, 0] = u * dt_hours / (111320 * np.cos(np.radians(lat + k1[i, 1]/2)))
            k2[i, 1] = v * dt_hours / 111320

            # k3
            u, v = self.interpolate_velocity(lon + k2[i, 0]/2, lat + k2[i, 1]/2)
            k3[i, 0] = u * dt_hours / (111320 * np.cos(np.radians(lat + k2[i, 1]/2)))
            k3[i, 1] = v * dt_hours / 111320

            # k4
            u, v = self.interpolate_velocity(lon + k3[i, 0], lat + k3[i, 1])
            k4[i, 0] = u * dt_hours / (111320 * np.cos(np.radians(lat + k3[i, 1])))
            k4[i, 1] = v * dt_hours / 111320

        # 更新位置
        new_positions = positions + (k1 + 2*k2 + 2*k3 + k4) / 6

        return new_positions

    def track_particles(self, initial_positions: List[List[float]],
                        simulation_hours: float, time_step_hours: float) -> List[List[List[float]]]:
        """
        追踪粒子轨迹
        
        Args:
            initial_positions: 初始位置
            simulation_hours: 模拟时长（小时）
            time_step_hours: 时间步长（小时）
            
        Returns:
            轨迹数据 [时间步][粒子][lon, lat]
        """
        # 验证和修复位置
        fixed_positions, fix_log = self.validate_and_fix_positions(initial_positions)

        if fix_log:
            logging.info(f"应用了 {len(fix_log)} 个位置修复")
            for fix in fix_log:
                logging.info(f"  粒子{fix['particle']}: {fix['reason']}")

        # 初始化粒子位置
        positions = np.array(fixed_positions, dtype=float)
        trajectories = []

        # 计算时间步数
        n_steps = int(simulation_hours / time_step_hours)
        dt_seconds = time_step_hours * 3600

        logging.info(f"开始追踪 {len(positions)} 个粒子，{n_steps} 个时间步")

        # 记录初始位置
        trajectories.append(positions.copy().tolist())

        # 时间积分循环
        for step in range(n_steps):
            try:
                # RK4时间步进
                positions = self.rk4_step(positions, dt_seconds)

                # 边界处理：确保粒子在数据范围内
                positions[:, 0] = np.clip(positions[:, 0], self.lon.min(), self.lon.max())
                positions[:, 1] = np.clip(positions[:, 1], self.lat.min(), self.lat.max())

                # 记录当前位置
                trajectories.append(positions.copy().tolist())

                if (step + 1) % 10 == 0:
                    logging.info(f"完成时间步 {step + 1}/{n_steps}")

            except Exception as e:
                logging.error(f"时间步 {step} 计算失败: {e}")
                break

        logging.info(f"粒子追踪完成，生成 {len(trajectories)} 个时间步的轨迹")
        return trajectories

    def close(self):
        """关闭NetCDF处理器"""
        if self.handler:
            self.handler.close()


def create_safe_particle_animation(initial_positions: List[List[float]],
                                   netcdf_path: str,
                                   simulation_hours: float = 24.0,
                                   time_step_hours: float = 1.0,
                                   output_path: str = "safe_particle_animation.gif",
                                   title: str = "纯Python粒子轨迹模拟") -> Dict[str, Any]:
    """
    创建安全的粒子动画（纯Python实现）
    
    Args:
        initial_positions: 初始粒子位置 [[经度, 纬度], ...]
        netcdf_path: NetCDF数据文件路径
        simulation_hours: 模拟时长（小时）
        time_step_hours: 时间步长（小时）
        output_path: 输出路径
        title: 动画标题
    
    Returns:
        包含成功状态和结果信息的字典
    """
    tracker = None

    try:
        logging.info(f"🌊 开始创建安全粒子动画: {len(initial_positions)}个粒子")

        # 初始化追踪器
        tracker = PurePythonParticleTracker(netcdf_path)
        if not tracker.initialize():
            return {
                'success': False,
                'message': '粒子追踪器初始化失败'
            }

        # 执行粒子追踪
        logging.info("🎯 执行纯Python粒子追踪...")
        trajectories = tracker.track_particles(initial_positions, simulation_hours, time_step_hours)

        if not trajectories:
            return {
                'success': False,
                'message': '未生成有效轨迹'
            }

        # 生成时间步数组
        time_steps = [i * time_step_hours for i in range(len(trajectories))]

        # 创建地理动画
        logging.info("🎬 创建地理动画...")
        anim_result = _create_safe_geographic_animation(
            trajectories=trajectories,
            time_steps=time_steps,
            initial_positions=trajectories[0],  # 使用修复后的初始位置
            title=title,
            output_path=output_path
        )

        if anim_result['success']:
            logging.info(f"✅ 安全粒子动画创建成功: {anim_result['output_path']}")
            return {
                'success': True,
                'message': "安全粒子动画创建成功",
                'output_path': anim_result['output_path'],
                'animation_stats': anim_result['animation_stats'],
                'trajectories': trajectories,
                'time_steps': time_steps
            }
        else:
            return {
                'success': False,
                'message': f"动画创建失败: {anim_result['message']}"
            }

    except Exception as e:
        logging.error(f"创建安全粒子动画失败: {e}")
        return {
            'success': False,
            'message': f"创建安全粒子动画失败: {str(e)}"
        }

    finally:
        if tracker:
            tracker.close()


def _create_safe_geographic_animation(trajectories: List[List[List[float]]],
                                      time_steps: List[float],
                                      initial_positions: List[List[float]],
                                      title: str,
                                      output_path: str) -> Dict[str, Any]:
    """创建安全的地理动画"""

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
                'message': "轨迹数据为空"
            }

        # 计算显示范围
        lon_margin = (max(all_lons) - min(all_lons)) * 0.1
        lat_margin = (max(all_lats) - min(all_lats)) * 0.1

        extent = [
            min(all_lons) - lon_margin,
            max(all_lons) + lon_margin,
            min(all_lats) - lat_margin,
            max(all_lats) + lat_margin
        ]

        # 创建图形
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # 添加地理要素
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

        # 网格线
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}

        # 设置粒子颜色
        n_particles = len(initial_positions)
        colors = plt.cm.Set1(np.linspace(0, 1, n_particles))

        # 初始化显示元素
        particles = ax.scatter([], [], s=60, c=colors, alpha=0.9, zorder=5,
                               transform=ccrs.PlateCarree(), edgecolors='black', linewidths=1)

        # 初始位置标记
        initial_scatter = ax.scatter([pos[0] for pos in initial_positions],
                                     [pos[1] for pos in initial_positions],
                                     s=100, marker='*', c='yellow',
                                     edgecolors='red', linewidths=2,
                                     zorder=6, transform=ccrs.PlateCarree(),
                                     label='初始位置')

        # 轨迹线
        trail_lines = []
        trail_length = min(20, len(trajectories)//2)

        for i in range(n_particles):
            line, = ax.plot([], [], '-', color=colors[i], alpha=0.6,
                            linewidth=2, transform=ccrs.PlateCarree())
            trail_lines.append(line)

        # 信息显示
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            verticalalignment='top', fontsize=11,
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                            zorder=10)

        # 标题和图例
        ax.set_title(title, fontsize=14, pad=15)
        ax.legend(loc='upper right')

        def animate(frame):
            """动画更新函数"""
            current_time = time_steps[frame]
            current_positions = trajectories[frame]

            # 更新粒子位置
            if current_positions:
                lons = [pos[0] for pos in current_positions]
                lats = [pos[1] for pos in current_positions]
                particles.set_offsets(np.column_stack([lons, lats]))

            # 更新轨迹
            start_frame = max(0, frame - trail_length)
            for i, line in enumerate(trail_lines):
                if i < len(current_positions):
                    traj_lons = []
                    traj_lats = []

                    for t in range(start_frame, frame + 1):
                        if t < len(trajectories) and i < len(trajectories[t]):
                            pos = trajectories[t][i]
                            traj_lons.append(pos[0])
                            traj_lats.append(pos[1])

                    line.set_data(traj_lons, traj_lats)
                    alpha = 0.8 * (len(traj_lons) / trail_length) if traj_lons else 0
                    line.set_alpha(min(alpha, 0.8))

            # 更新信息
            active_particles = len(current_positions)

            if elapsed_time := current_time:
                if elapsed_time < 24:
                    time_str = f"{elapsed_time:.1f} 小时"
                else:
                    days = int(elapsed_time // 24)
                    hours = elapsed_time % 24
                    time_str = f"{days} 天 {hours:.1f} 小时"
            else:
                time_str = "0 小时"

            info_text_str = f'模拟时间: {time_str}\n'
            info_text_str += f'活跃粒子: {active_particles}/{n_particles}\n'
            info_text_str += f'当前帧: {frame + 1}/{len(trajectories)}'

            info_text.set_text(info_text_str)

            return [particles, info_text] + trail_lines

        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(trajectories),
                                       interval=500, blit=False, repeat=True)

        # 保存动画
        try:
            plt.tight_layout()

            if output_path.endswith('.gif'):
                anim.save(output_path, writer='pillow', fps=8, dpi=100)
            else:
                output_path += '.gif'
                anim.save(output_path, writer='pillow', fps=8, dpi=100)

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
            'message': f"创建动画失败: {str(e)}"
        }


if __name__ == "__main__":
    # 测试纯Python粒子动画生成器
    import os

    print("🔒 测试纯Python粒子动画生成器（避免C++段错误）")
    print("-" * 60)

    # 测试配置
    test_netcdf_path = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/data/raw_data/merged_data.nc"

    test_positions = [
        [120.5, 31.2],   # 上海附近
        [121.0, 31.0],   # 长江口
        [120.8, 30.8],   # 杭州湾
        [121.2, 31.5],   # 崇明岛附近
    ]

    print(f"初始位置: {len(test_positions)}个粒子")
    print(f"模拟时长: 24小时")

    # 创建输出目录
    os.makedirs("test_outputs", exist_ok=True)

    # 创建安全的粒子动画
    result = create_safe_particle_animation(
        initial_positions=test_positions,
        netcdf_path=test_netcdf_path,
        simulation_hours=24.0,
        time_step_hours=2.0,
        output_path="test_outputs/safe_particle_animation.gif",
        title="纯Python长江口粒子轨迹模拟"
    )

    print("\n📊 结果分析:")
    if result["success"]:
        print("✅ 安全动画创建成功")
        print(f"   输出文件: {result['output_path']}")

        if 'animation_stats' in result:
            stats = result['animation_stats']
            print(f"📈 动画统计:")
            print(f"   帧数: {stats['n_frames']}")
            print(f"   粒子数: {stats['n_particles']}")
            print(f"   模拟时长: {stats['simulation_hours']} 小时")
            print(f"   地理范围: {stats['geographic_extent']}")

        print("🎉 纯Python实现成功避免了C++段错误问题！")
    else:
        print("❌ 动画创建失败")
        print(f"   错误信息: {result['message']}")

    print("\n🎯 测试完成")
    print("💡 此版本完全使用Python实现，避免了C++模块的内存问题")