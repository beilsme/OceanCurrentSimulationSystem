# ==============================================================================
# visualization/geo_particle_animator.py
# ==============================================================================
"""
地理底图粒子轨迹动画生成器 - 在真实地理底图上显示拉格朗日粒子运动
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

# 导入中文配置
sys.path.append(str(Path(__file__).parent.parent / "utils"))
try:
    from PythonEngine.utils.chinese_config import ChineseConfig
    chinese_config = ChineseConfig()
except ImportError:
    chinese_config = None


class GeoParticleAnimator:
    """地理底图粒子轨迹动画生成器"""

    def __init__(self, extent: Optional[List[float]] = None,
                 chinese_support: bool = True,
                 projection: ccrs.Projection = None):
        """
        初始化地理粒子动画器
        
        Args:
            extent: 地理范围 [lon_min, lon_max, lat_min, lat_max]
            chinese_support: 中文支持
            projection: 地图投影，默认为PlateCarree
        """
        self.extent = extent
        self.projection = projection or ccrs.PlateCarree()

        # 中文支持
        if chinese_support and chinese_config:
            self.font_config = chinese_config.setup_chinese_support()
            allowed_keys = {"family", "size", "weight", "color"}
            self.font_config = {k: v for k, v in self.font_config.items() if k in allowed_keys}
        else:
            self.font_config = {}

    def create_particle_trajectory_animation(self,
                                             trajectories: List[List[List[float]]],
                                             time_steps: List[float],
                                             initial_positions: List[List[float]],
                                             title: str = "拉格朗日粒子轨迹动画",
                                             trail_length: int = 10,
                                             particle_colors: Optional[List[str]] = None,
                                             background_data: Optional[Dict] = None,
                                             save_path: Optional[str] = None,
                                             fps: int = 15,
                                             show_coastlines: bool = True,
                                             show_land: bool = True,
                                             show_ocean: bool = True,
                                             show_gridlines: bool = True) -> animation.FuncAnimation:
        """
        创建地理底图上的粒子轨迹动画
        
        Args:
            trajectories: 粒子轨迹时间序列 [time_step][particle_id][lon, lat]
            time_steps: 时间步数组 (小时)
            initial_positions: 初始粒子位置 [[lon, lat], ...]
            title: 动画标题
            trail_length: 轨迹尾迹长度
            particle_colors: 粒子颜色列表
            background_data: 背景场数据 (速度场、温度等)
            save_path: 保存路径
            fps: 帧率
            show_coastlines: 显示海岸线
            show_land: 显示陆地
            show_ocean: 显示海洋
            show_gridlines: 显示网格线
        """

        # 计算地理范围
        if self.extent is None:
            all_lons = []
            all_lats = []
            for frame in trajectories:
                for particle in frame:
                    all_lons.append(particle[0])
                    all_lats.append(particle[1])

            lon_margin = (max(all_lons) - min(all_lons)) * 0.1
            lat_margin = (max(all_lats) - min(all_lats)) * 0.1

            self.extent = [
                min(all_lons) - lon_margin,
                max(all_lons) + lon_margin,
                min(all_lats) - lat_margin,
                max(all_lats) + lat_margin
            ]

        # 创建图形
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=self.projection)
        ax.set_extent(self.extent, crs=ccrs.PlateCarree())

        # 添加地理要素
        if show_coastlines:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        if show_land:
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
        if show_ocean:
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

        # 添加国家边界和主要河流
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray', alpha=0.8)
        ax.add_feature(cfeature.RIVERS, linewidth=0.3, color='blue', alpha=0.6)

        # 网格线
        if show_gridlines:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()

        # 粒子颜色设置
        n_particles = len(initial_positions)
        if particle_colors is None:
            # 使用不同颜色区分粒子
            colors = plt.cm.tab10(np.linspace(0, 1, min(n_particles, 10)))
            particle_colors = [colors[i % 10] for i in range(n_particles)]

        # 初始化粒子显示
        particles = ax.scatter([], [], s=50, c=particle_colors[:n_particles],
                               alpha=0.9, zorder=5, transform=ccrs.PlateCarree(),
                               edgecolors='black', linewidths=0.5)

        # 初始位置标记
        initial_scatter = ax.scatter([pos[0] for pos in initial_positions],
                                     [pos[1] for pos in initial_positions],
                                     s=80, marker='*', c='yellow',
                                     edgecolors='black', linewidths=1,
                                     zorder=6, transform=ccrs.PlateCarree(),
                                     label='初始位置')

        # 轨迹线存储
        trail_lines = []
        for i in range(n_particles):
            line, = ax.plot([], [], '-', color=particle_colors[i],
                            alpha=0.7, linewidth=2,
                            transform=ccrs.PlateCarree())
            trail_lines.append(line)

        # 背景场显示
        background_im = None
        if background_data:
            self._add_background_field(ax, background_data)

        # 时间和统计信息显示
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            verticalalignment='top', **self.font_config,
                            bbox=dict(boxstyle='round,pad=0.5',
                                      facecolor='white', alpha=0.9),
                            fontsize=12, zorder=10)

        # 标题
        ax.set_title(title, **self.font_config, fontsize=16, pad=20)

        # 图例
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.95))

        def animate(frame):
            """动画更新函数"""
            current_time = time_steps[frame]
            current_positions = trajectories[frame]

            # 更新粒子位置
            if len(current_positions) > 0:
                lons = [pos[0] for pos in current_positions]
                lats = [pos[1] for pos in current_positions]
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
                                i < len(trajectories[t])):
                            pos = trajectories[t][i]
                            traj_lons.append(pos[0])
                            traj_lats.append(pos[1])

                    line.set_data(traj_lons, traj_lats)

                    # 设置透明度渐变
                    alpha = 0.8 * (len(traj_lons) / trail_length) if len(traj_lons) > 0 else 0
                    line.set_alpha(min(alpha, 0.8))

            # 更新信息显示
            active_particles = len(current_positions)
            elapsed_time = current_time

            # 计算粒子扩散统计
            if len(current_positions) > 1:
                positions_array = np.array(current_positions)
                center_lon = np.mean(positions_array[:, 0])
                center_lat = np.mean(positions_array[:, 1])

                # 计算扩散距离 (粗略估算，单位：km)
                distances = []
                for pos in current_positions:
                    dist = np.sqrt(
                        ((pos[0] - center_lon) * 111.32 * np.cos(np.radians(center_lat)))**2 +
                        ((pos[1] - center_lat) * 111.32)**2
                    )
                    distances.append(dist)

                max_spread = np.max(distances) if distances else 0
                mean_spread = np.mean(distances) if distances else 0
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
                                       interval=1000//fps, blit=False, repeat=True)

        plt.tight_layout()

        # 保存动画
        if save_path:
            try:
                if save_path.endswith('.gif'):
                    anim.save(save_path, writer='pillow', fps=fps, dpi=100)
                elif save_path.endswith('.mp4'):
                    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=100)
                else:
                    save_path_gif = save_path + '.gif'
                    anim.save(save_path_gif, writer='pillow', fps=fps, dpi=100)
                    save_path = save_path_gif

                logging.info(f"地理粒子轨迹动画保存至: {save_path}")

                # 保存轨迹数据
                self._save_trajectory_data(trajectories, time_steps,
                                           initial_positions, save_path)

            except Exception as e:
                logging.error(f"保存动画失败: {e}")

        return anim

    def _add_background_field(self, ax, background_data: Dict):
        """添加背景场数据"""
        field_type = background_data.get('type', 'velocity')
        data = background_data.get('data')
        lons = background_data.get('lons')
        lats = background_data.get('lats')

        if data is None or lons is None or lats is None:
            return

        try:
            if field_type == 'velocity':
                # 速度场矢量图
                u = data.get('u', np.zeros_like(lons))
                v = data.get('v', np.zeros_like(lats))

                # 降采样以避免过密的箭头
                skip = max(1, len(lons) // 20)
                ax.quiver(lons[::skip], lats[::skip],
                          u[::skip, ::skip], v[::skip, ::skip],
                          scale=50, alpha=0.6, color='gray',
                          transform=ccrs.PlateCarree())

            elif field_type == 'scalar':
                # 标量场等值线图
                levels = background_data.get('levels', 20)
                cmap = background_data.get('colormap', 'viridis')

                cs = ax.contourf(lons, lats, data, levels=levels,
                                 cmap=cmap, alpha=0.5,
                                 transform=ccrs.PlateCarree())
                plt.colorbar(cs, ax=ax, shrink=0.8, pad=0.05)

        except Exception as e:
            logging.warning(f"添加背景场失败: {e}")

    def _save_trajectory_data(self, trajectories, time_steps,
                              initial_positions, save_path):
        """保存轨迹数据为JSON格式"""
        try:
            data_path = save_path.replace('.gif', '_data.json').replace('.mp4', '_data.json')

            trajectory_data = {
                "metadata": {
                    "creation_time": datetime.now().isoformat(),
                    "n_particles": len(initial_positions),
                    "n_time_steps": len(time_steps),
                    "simulation_duration_hours": time_steps[-1] if time_steps else 0
                },
                "initial_positions": initial_positions,
                "time_steps": time_steps,
                "trajectories": trajectories,
                "geographic_extent": self.extent
            }

            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

            logging.info(f"轨迹数据保存至: {data_path}")

        except Exception as e:
            logging.warning(f"保存轨迹数据失败: {e}")


def create_simple_particle_animation(initial_positions: List[List[float]],
                                     netcdf_path: str,
                                     simulation_hours: float = 24.0,
                                     time_step_hours: float = 1.0,
                                     output_path: str = "particle_animation.gif",
                                     title: str = "海洋粒子轨迹模拟") -> Dict[str, Any]:
    """
    简化的粒子动画创建函数 - 用户友好接口
    
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
    try:
        # 导入必要的模块
        from PythonEngine.wrappers.lagrangian_particle_wrapper import simulate_particle_tracking

        # 计算时间步数
        n_steps = int(simulation_hours / time_step_hours)
        dt_seconds = time_step_hours * 3600  # 转换为秒

        # 准备粒子追踪输入
        tracking_input = {
            "action": "simulate_particle_tracking",
            "parameters": {
                "netcdf_path": netcdf_path,
                "time_index": 0,
                "depth_index": 0,
                "initial_positions": initial_positions,
                "dt": dt_seconds,
                "steps": n_steps,
                "output_path": "temp_particle_tracks.png"
            }
        }

        # 执行粒子追踪模拟
        logging.info(f"开始粒子追踪模拟: {len(initial_positions)}个粒子, {simulation_hours}小时")
        result = simulate_particle_tracking(tracking_input)

        if not result.get("success", False):
            return {
                "success": False,
                "message": f"粒子追踪模拟失败: {result.get('message', '未知错误')}",
                "error_details": result
            }

        # 获取轨迹数据
        trajectories = result.get("trajectories", [])
        if not trajectories:
            return {
                "success": False,
                "message": "未获取到轨迹数据"
            }

        # 生成时间步数组
        time_steps = [i * time_step_hours for i in range(len(trajectories))]

        # 创建地理动画
        animator = GeoParticleAnimator()

        logging.info("创建地理底图动画...")
        anim = animator.create_particle_trajectory_animation(
            trajectories=trajectories,
            time_steps=time_steps,
            initial_positions=initial_positions,
            title=title,
            trail_length=min(15, len(trajectories)//2),
            save_path=output_path,
            fps=10
        )

        # 显示动画（如果在交互环境中）
        try:
            plt.show()
        except:
            pass  # 非交互环境

        return {
            "success": True,
            "message": "粒子轨迹动画创建成功",
            "output_path": output_path,
            "animation_object": anim,
            "simulation_stats": {
                "n_particles": len(initial_positions),
                "simulation_hours": simulation_hours,
                "n_time_steps": len(trajectories),
                "final_positions": trajectories[-1] if trajectories else []
            }
        }

    except Exception as e:
        logging.error(f"创建粒子动画失败: {e}")
        return {
            "success": False,
            "message": f"创建粒子动画失败: {str(e)}",
            "error_trace": str(e)
        }


if __name__ == "__main__":
    # 测试地理粒子动画生成器
    import os

    # 测试配置
    test_netcdf_path = "../data/raw_data/merged_data.nc"  

    # 定义初始粒子位置（经度、纬度）
    initial_positions = [
        [120.5, 31.2],   # 上海附近
        [121.0, 31.0],   # 长江口
        [120.8, 30.8],   # 杭州湾
        [121.2, 31.5],   # 崇明岛附近
    ]

    print("🌊 开始地理粒子轨迹动画测试")
    print(f"初始位置: {len(initial_positions)}个粒子")
    print(f"模拟时长: 48小时")

    # 创建输出目录
    os.makedirs("test_outputs", exist_ok=True)

    # 创建粒子动画
    result = create_simple_particle_animation(
        initial_positions=initial_positions,
        netcdf_path=test_netcdf_path,
        simulation_hours=48.0,
        time_step_hours=2.0,
        output_path="test_outputs/geo_particle_animation.gif",
        title="长江口海域粒子轨迹模拟"
    )

    if result["success"]:
        print(f"✅ 动画创建成功: {result['output_path']}")
        stats = result["simulation_stats"]
        print(f"📊 统计信息:")
        print(f"   - 粒子数量: {stats['n_particles']}")
        print(f"   - 时间步数: {stats['n_time_steps']}")
        print(f"   - 模拟时长: {stats['simulation_hours']} 小时")
    else:
        print(f"❌ 动画创建失败: {result['message']}")

    print("🎯 测试完成")