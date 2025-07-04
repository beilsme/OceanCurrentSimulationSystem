# ==============================================================================
# visualization/pollution_diffusion_animator.py
# ==============================================================================
"""
纯Python污染物扩散模拟动画系统
基于对流-扩散方程的数值求解，包含多种污染源类型和可视化效果
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class PollutionDiffusionSimulator:
    """污染物扩散模拟器"""

    def __init__(self, netcdf_path: str, grid_resolution: float = 0.01):
        """
        初始化污染物扩散模拟器
        
        Args:
            netcdf_path: NetCDF数据文件路径
            grid_resolution: 网格分辨率（度）
        """
        self.netcdf_path = netcdf_path
        self.grid_resolution = grid_resolution

        # 海洋数据
        self.nc_data = None
        self.lat = None
        self.lon = None
        self.water_u = None
        self.water_v = None
        self.time_readable = None

        # 模拟网格
        self.sim_lat = None
        self.sim_lon = None
        self.sim_u = None
        self.sim_v = None
        self.concentration = None

        # 物理参数
        self.diffusion_coeff = 100.0  # 扩散系数 (m²/s)
        self.decay_rate = 0.0         # 衰减率 (1/s)
        self.settling_velocity = 0.0  # 沉降速度 (m/s)

        # 污染源
        self.pollution_sources = []

        self.is_initialized = False

    def initialize(self,
                   lon_range: Tuple[float, float] = (118, 124),
                   lat_range: Tuple[float, float] = (21, 26.5),
                   time_index: int = 0) -> Dict[str, Any]:
        """
        初始化模拟环境
        
        Args:
            lon_range: 经度范围
            lat_range: 纬度范围
            time_index: 时间索引
            
        Returns:
            初始化结果
        """
        try:
            logging.info("初始化污染物扩散模拟环境")

            # 加载NetCDF数据
            self.nc_data = Dataset(self.netcdf_path, mode='r')

            # 提取基础数据
            self.lat = self.nc_data.variables['lat'][:]
            self.lon = self.nc_data.variables['lon'][:]
            self.water_u = self.nc_data.variables['water_u'][time_index, 0, :, :]  # 表层
            self.water_v = self.nc_data.variables['water_v'][time_index, 0, :, :]

            # 处理掩码数组
            if isinstance(self.water_u, np.ma.MaskedArray):
                self.water_u = self.water_u.filled(0)
            if isinstance(self.water_v, np.ma.MaskedArray):
                self.water_v = self.water_v.filled(0)

            # 应用地理范围过滤
            lon_min, lon_max = lon_range
            lat_min, lat_max = lat_range

            lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)
            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)

            self.lon = self.lon[lon_mask]
            self.lat = self.lat[lat_mask]
            self.water_u = self.water_u[np.ix_(lat_mask, lon_mask)]
            self.water_v = self.water_v[np.ix_(lat_mask, lon_mask)]

            # 创建模拟网格
            self._create_simulation_grid(lon_range, lat_range)

            # 插值速度场到模拟网格
            self._interpolate_velocity_field()

            # 初始化浓度场
            self._initialize_concentration_field()

            self.is_initialized = True

            logging.info("污染物扩散模拟环境初始化成功")

            return {
                "success": True,
                "message": "初始化成功",
                "grid_info": {
                    "sim_grid_size": (len(self.sim_lat), len(self.sim_lon)),
                    "grid_resolution": self.grid_resolution,
                    "domain_size_km": (
                        (lon_range[1] - lon_range[0]) * 111.32,
                        (lat_range[1] - lat_range[0]) * 111.32
                    )
                }
            }

        except Exception as e:
            logging.error(f"初始化失败: {e}")
            return {
                "success": False,
                "message": f"初始化失败: {str(e)}"
            }

    def _create_simulation_grid(self, lon_range: Tuple[float, float], lat_range: Tuple[float, float]):
        """创建高分辨率模拟网格"""
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range

        # 创建等间距网格
        self.sim_lon = np.arange(lon_min, lon_max + self.grid_resolution, self.grid_resolution)
        self.sim_lat = np.arange(lat_min, lat_max + self.grid_resolution, self.grid_resolution)

        logging.info(f"创建模拟网格: {len(self.sim_lat)} x {len(self.sim_lon)}")

    def _interpolate_velocity_field(self):
        """插值速度场到模拟网格"""
        # 创建插值器
        u_interp = RegularGridInterpolator(
            (self.lat, self.lon), self.water_u,
            bounds_error=False, fill_value=0, method='linear'
        )
        v_interp = RegularGridInterpolator(
            (self.lat, self.lon), self.water_v,
            bounds_error=False, fill_value=0, method='linear'
        )

        # 创建模拟网格的坐标网格
        sim_lon_grid, sim_lat_grid = np.meshgrid(self.sim_lon, self.sim_lat, indexing='xy')

        # 插值速度场
        points = np.column_stack([sim_lat_grid.ravel(), sim_lon_grid.ravel()])
        self.sim_u = u_interp(points).reshape(sim_lat_grid.shape)
        self.sim_v = v_interp(points).reshape(sim_lat_grid.shape)

        logging.info("速度场插值完成")

    def _initialize_concentration_field(self):
        """初始化浓度场"""
        self.concentration = np.zeros((len(self.sim_lat), len(self.sim_lon)))
        logging.info("浓度场初始化完成")

    def add_pollution_source(self,
                             source_type: str,
                             location: Tuple[float, float],
                             intensity: float,
                             duration: float = None,
                             radius: float = 0.01,
                             **kwargs) -> int:
        """
        添加污染源
        
        Args:
            source_type: 污染源类型 ('point', 'continuous', 'area', 'moving')
            location: 位置 (lat, lon)
            intensity: 强度 (kg/s)
            duration: 持续时间 (小时)
            radius: 影响半径 (度)
            **kwargs: 其他参数
            
        Returns:
            污染源ID
        """
        source = {
            'id': len(self.pollution_sources),
            'type': source_type,
            'location': location,
            'intensity': intensity,
            'duration': duration,
            'radius': radius,
            'start_time': kwargs.get('start_time', 0),
            'active': True,
            **kwargs
        }

        self.pollution_sources.append(source)

        logging.info(f"添加污染源 {source['id']}: {source_type} at {location}")

        return source['id']

    def add_oil_spill(self, location: Tuple[float, float], volume: float,
                      spill_duration: float = 1.0) -> int:
        """
        添加石油泄漏源
        
        Args:
            location: 泄漏位置 (lat, lon)
            volume: 泄漏总量 (m³)
            spill_duration: 泄漏持续时间 (小时)
            
        Returns:
            污染源ID
        """
        # 石油密度约0.85 kg/L = 850 kg/m³
        oil_density = 850.0
        total_mass = volume * oil_density
        intensity = total_mass / (spill_duration * 3600)  # kg/s

        return self.add_pollution_source(
            source_type='oil_spill',
            location=location,
            intensity=intensity,
            duration=spill_duration,
            radius=0.005,
            volume=volume,
            density=oil_density,
            weathering_rate=0.1  # 风化率
        )

    def add_industrial_discharge(self, location: Tuple[float, float],
                                 discharge_rate: float, pollutant_type: str = 'chemical') -> int:
        """
        添加工业排放源
        
        Args:
            location: 排放位置 (lat, lon)
            discharge_rate: 排放率 (kg/s)
            pollutant_type: 污染物类型
            
        Returns:
            污染源ID
        """
        return self.add_pollution_source(
            source_type='industrial',
            location=location,
            intensity=discharge_rate,
            duration=None,  # 连续排放
            radius=0.002,
            pollutant_type=pollutant_type,
            decay_rate=0.001  # 生物降解率
        )

    def simulate_diffusion(self,
                           simulation_hours: float = 24.0,
                           time_step_minutes: float = 10.0,
                           **physics_params) -> Dict[str, Any]:
        """
        执行污染物扩散模拟
        
        Args:
            simulation_hours: 模拟时长 (小时)
            time_step_minutes: 时间步长 (分钟)
            **physics_params: 物理参数
            
        Returns:
            模拟结果
        """
        if not self.is_initialized:
            raise ValueError("模拟器未初始化")

        # 更新物理参数
        self.diffusion_coeff = physics_params.get('diffusion_coeff', 100.0)
        self.decay_rate = physics_params.get('decay_rate', 0.0)
        self.settling_velocity = physics_params.get('settling_velocity', 0.0)

        # 时间参数
        dt = time_step_minutes * 60  # 转换为秒
        n_steps = int(simulation_hours * 3600 / dt)

        # 空间参数
        dx = self.grid_resolution * 111320  # 转换为米
        dy = self.grid_resolution * 111320

        # 存储结果
        concentration_history = []
        time_history = []

        logging.info(f"开始扩散模拟: {simulation_hours}小时, {n_steps}步")

        for step in range(n_steps):
            current_time = step * dt / 3600  # 小时

            # 添加污染源
            self._apply_pollution_sources(current_time, dt)

            # 对流-扩散方程求解
            self._solve_advection_diffusion(dt, dx, dy)

            # 记录结果 (每隔一定步数记录一次以节省内存)
            if step % max(1, n_steps // 100) == 0:
                concentration_history.append(self.concentration.copy())
                time_history.append(current_time)

            if step % (n_steps // 10) == 0:
                logging.info(f"模拟进度: {step/n_steps*100:.1f}%")

        # 添加最终状态
        concentration_history.append(self.concentration.copy())
        time_history.append(simulation_hours)

        logging.info("扩散模拟完成")

        return {
            "success": True,
            "concentration_history": concentration_history,
            "time_history": time_history,
            "simulation_params": {
                "simulation_hours": simulation_hours,
                "time_step_minutes": time_step_minutes,
                "n_steps": n_steps,
                "diffusion_coeff": self.diffusion_coeff,
                "decay_rate": self.decay_rate
            },
            "grid_params": {
                "lat": self.sim_lat,
                "lon": self.sim_lon,
                "resolution": self.grid_resolution
            },
            "pollution_sources": self.pollution_sources
        }

    def _apply_pollution_sources(self, current_time: float, dt: float):
        """应用污染源"""
        for source in self.pollution_sources:
            if not source['active']:
                continue

            # 检查时间范围
            start_time = source.get('start_time', 0)
            duration = source.get('duration')

            if current_time < start_time:
                continue
            if duration and current_time > start_time + duration:
                source['active'] = False
                continue

            # 计算源位置在网格中的索引
            lat_idx = np.argmin(np.abs(self.sim_lat - source['location'][0]))
            lon_idx = np.argmin(np.abs(self.sim_lon - source['location'][1]))

            # 应用源强度
            if source['type'] == 'point':
                self.concentration[lat_idx, lon_idx] += source['intensity'] * dt / (111320 * self.grid_resolution)**2

            elif source['type'] == 'area' or source['type'] == 'oil_spill':
                # 面源：在半径范围内分布
                radius_grid = int(source['radius'] / self.grid_resolution)

                for di in range(-radius_grid, radius_grid + 1):
                    for dj in range(-radius_grid, radius_grid + 1):
                        ni, nj = lat_idx + di, lon_idx + dj
                        if 0 <= ni < len(self.sim_lat) and 0 <= nj < len(self.sim_lon):
                            distance = np.sqrt(di**2 + dj**2) * self.grid_resolution
                            if distance <= source['radius']:
                                # 高斯分布
                                weight = np.exp(-distance**2 / (2 * (source['radius']/3)**2))
                                self.concentration[ni, nj] += (source['intensity'] * dt * weight /
                                                               (111320 * self.grid_resolution)**2)

            elif source['type'] == 'industrial':
                # 连续点源
                self.concentration[lat_idx, lon_idx] += source['intensity'] * dt / (111320 * self.grid_resolution)**2

    def _solve_advection_diffusion(self, dt: float, dx: float, dy: float):
        """求解对流-扩散方程"""
        # 保存当前浓度
        c_old = self.concentration.copy()

        # 计算梯度
        dc_dx = np.gradient(c_old, dx, axis=1)
        dc_dy = np.gradient(c_old, dy, axis=0)

        # 对流项 (负号表示浓度被流场输运)
        advection_x = -self.sim_u * dc_dx
        advection_y = -self.sim_v * dc_dy

        # 扩散项
        d2c_dx2 = np.gradient(np.gradient(c_old, dx, axis=1), dx, axis=1)
        d2c_dy2 = np.gradient(np.gradient(c_old, dy, axis=0), dy, axis=0)
        diffusion = self.diffusion_coeff * (d2c_dx2 + d2c_dy2)

        # 衰减项
        decay = -self.decay_rate * c_old

        # 时间积分 (前向欧拉法)
        dc_dt = advection_x + advection_y + diffusion + decay
        self.concentration = c_old + dt * dc_dt

        # 确保浓度非负
        self.concentration = np.maximum(0, self.concentration)

        # 边界条件：开边界
        self.concentration[0, :] = 0
        self.concentration[-1, :] = 0
        self.concentration[:, 0] = 0
        self.concentration[:, -1] = 0

    def create_diffusion_animation(self,
                                   simulation_result: Dict[str, Any],
                                   output_path: str = "pollution_diffusion.gif",
                                   title: str = "污染物扩散模拟",
                                   fps: int = 10,
                                   show_sources: bool = True,
                                   show_velocity: bool = False,
                                   colormap: str = 'Reds') -> Dict[str, Any]:
        """
        创建污染物扩散动画
        
        Args:
            simulation_result: 模拟结果
            output_path: 输出路径
            title: 动画标题
            fps: 帧率
            show_sources: 是否显示污染源
            show_velocity: 是否显示速度场
            colormap: 颜色映射
            
        Returns:
            动画创建结果
        """
        try:
            logging.info("开始创建污染物扩散动画")

            concentration_history = simulation_result['concentration_history']
            time_history = simulation_result['time_history']
            lat = simulation_result['grid_params']['lat']
            lon = simulation_result['grid_params']['lon']

            # 计算浓度范围
            max_concentration = np.max([np.max(c) for c in concentration_history])
            if max_concentration == 0:
                max_concentration = 1e-6

            # 创建图形
            fig = plt.figure(figsize=(14, 10))
            ax = plt.axes(projection=ccrs.PlateCarree())

            # 设置地理范围
            extent = [lon.min(), lon.max(), lat.min(), lat.max()]
            ax.set_extent(extent, crs=ccrs.PlateCarree())

            # 添加地理要素
            ax.add_feature(cfeature.COASTLINE, linewidth=1.2, color='black')
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

            # 网格线
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}

            # 创建颜色映射
            if colormap == 'Reds':
                colors = ['white', 'yellow', 'orange', 'red', 'darkred']
            elif colormap == 'Blues':
                colors = ['white', 'lightblue', 'blue', 'darkblue', 'navy']
            else:
                colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, 5))

            cmap = ListedColormap(colors)
            norm = plt.Normalize(0, max_concentration)

            # 初始化图像
            LON, LAT = np.meshgrid(lon, lat)
            im = ax.contourf(LON, LAT, concentration_history[0],
                             levels=20, cmap=cmap, norm=norm,
                             transform=ccrs.PlateCarree(), alpha=0.8)

            # 颜色条
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
            cbar.set_label('浓度 (kg/m³)', fontsize=12)

            # 显示污染源
            source_patches = []
            if show_sources:
                for source in self.pollution_sources:
                    lat_src, lon_src = source['location']

                    if source['type'] == 'oil_spill':
                        marker = ax.plot(lon_src, lat_src, 'ko', markersize=12,
                                         markerfacecolor='red', markeredgecolor='black',
                                         transform=ccrs.PlateCarree(), label='石油泄漏')[0]
                    elif source['type'] == 'industrial':
                        marker = ax.plot(lon_src, lat_src, 's', markersize=10,
                                         markerfacecolor='purple', markeredgecolor='black',
                                         transform=ccrs.PlateCarree(), label='工业排放')[0]
                    else:
                        marker = ax.plot(lon_src, lat_src, '^', markersize=10,
                                         markerfacecolor='orange', markeredgecolor='black',
                                         transform=ccrs.PlateCarree(), label='污染源')[0]

                    source_patches.append(marker)

            # 显示速度场
            velocity_arrows = None
            if show_velocity:
                # 降采样速度场
                skip = max(1, len(lon) // 20)
                velocity_arrows = ax.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
                                            self.sim_u[::skip, ::skip], self.sim_v[::skip, ::skip],
                                            scale=50, alpha=0.6, color='gray',
                                            transform=ccrs.PlateCarree())

            # 信息文本
            info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                verticalalignment='top', fontsize=12,
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

            # 标题
            ax.set_title(title, fontsize=16, pad=20)

            # 图例
            if show_sources:
                ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.95))

            def animate(frame):
                """动画更新函数"""
                # 清除之前的等值线
                for coll in ax.collections:
                    if hasattr(coll, 'set_array'):
                        coll.remove()

                # 更新浓度场
                current_concentration = concentration_history[frame]
                current_time = time_history[frame]

                # 绘制新的等值线
                cs = ax.contourf(LON, LAT, current_concentration,
                                 levels=20, cmap=cmap, norm=norm,
                                 transform=ccrs.PlateCarree(), alpha=0.8)

                # 计算统计信息
                total_mass = np.sum(current_concentration) * (111320 * self.grid_resolution)**2
                max_conc = np.max(current_concentration)
                affected_area = np.sum(current_concentration > max_conc * 0.01) * (self.grid_resolution * 111.32)**2

                # 更新信息文本
                if current_time < 24:
                    time_str = f"{current_time:.1f} 小时"
                else:
                    days = int(current_time // 24)
                    hours = current_time % 24
                    time_str = f"{days} 天 {hours:.1f} 小时"

                info_str = f'模拟时间: {time_str}\n'
                info_str += f'最大浓度: {max_conc:.2e} kg/m³\n'
                info_str += f'总质量: {total_mass:.1e} kg\n'
                info_str += f'影响面积: {affected_area:.1f} km²'

                info_text.set_text(info_str)

                return [info_text]

            # 创建动画
            anim = animation.FuncAnimation(fig, animate, frames=len(concentration_history),
                                           interval=1000//fps, blit=False, repeat=True)

            # 保存动画
            plt.tight_layout()

            if output_path.endswith('.gif'):
                anim.save(output_path, writer='pillow', fps=fps, dpi=100)
            elif output_path.endswith('.mp4'):
                anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            else:
                output_path += '.gif'
                anim.save(output_path, writer='pillow', fps=fps, dpi=100)

            plt.close(fig)

            # 保存模拟数据
            self._save_simulation_data(simulation_result, output_path)

            logging.info(f"污染物扩散动画创建成功: {output_path}")

            return {
                "success": True,
                "output_path": output_path,
                "animation_stats": {
                    "n_frames": len(concentration_history),
                    "max_concentration": max_concentration,
                    "simulation_hours": time_history[-1],
                    "n_sources": len(self.pollution_sources)
                }
            }

        except Exception as e:
            logging.error(f"创建污染物扩散动画失败: {e}")
            return {
                "success": False,
                "message": f"动画创建失败: {str(e)}"
            }

    def _save_simulation_data(self, simulation_result: Dict, output_path: str):
        """保存模拟数据"""
        try:
            data_path = output_path.replace('.gif', '_data.json').replace('.mp4', '_data.json')

            # 准备保存的数据 (不包含大数组)
            save_data = {
                "metadata": {
                    "creation_time": datetime.now().isoformat(),
                    "simulation_type": "pollution_diffusion",
                    "grid_resolution": self.grid_resolution
                },
                "simulation_params": simulation_result["simulation_params"],
                "pollution_sources": simulation_result["pollution_sources"],
                "statistics": {
                    "max_concentration": float(np.max([np.max(c) for c in simulation_result['concentration_history']])),
                    "final_total_mass": float(np.sum(simulation_result['concentration_history'][-1])),
                    "simulation_domain_km2": float((len(self.sim_lat) * len(self.sim_lon)) * (self.grid_resolution * 111.32)**2)
                }
            }

            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            logging.info(f"模拟数据保存至: {data_path}")

        except Exception as e:
            logging.warning(f"保存模拟数据失败: {e}")

    def close(self):
        """关闭资源"""
        if self.nc_data:
            self.nc_data.close()


def create_pollution_diffusion_animation(
        netcdf_path: str,
        pollution_config: Dict[str, Any],
        simulation_config: Dict[str, Any],
        output_path: str = "pollution_diffusion.gif",
        title: str = "污染物扩散模拟") -> Dict[str, Any]:
    """
    创建污染物扩散动画的便捷函数
    
    Args:
        netcdf_path: NetCDF数据文件路径
        pollution_config: 污染配置
        simulation_config: 模拟配置
        output_path: 输出路径
        title: 动画标题
        
    Returns:
        创建结果
    """
    simulator = None

    try:
        # 初始化模拟器
        simulator = PollutionDiffusionSimulator(
            netcdf_path,
            grid_resolution=simulation_config.get('grid_resolution', 0.01)
        )

        # 初始化环境
        init_result = simulator.initialize(
            lon_range=simulation_config.get('lon_range', (118, 124)),
            lat_range=simulation_config.get('lat_range', (21, 26.5)),
            time_index=simulation_config.get('time_index', 0)
        )

        if not init_result["success"]:
            return init_result

        # 添加污染源
        for source_config in pollution_config.get('sources', []):
            source_type = source_config.get('type')

            if source_type == 'oil_spill':
                simulator.add_oil_spill(
                    location=tuple(source_config['location']),
                    volume=source_config['volume'],
                    spill_duration=source_config.get('duration', 1.0)
                )
            elif source_type == 'industrial':
                simulator.add_industrial_discharge(
                    location=tuple(source_config['location']),
                    discharge_rate=source_config['intensity'],
                    pollutant_type=source_config.get('pollutant_type', 'chemical')
                )
            else:
                simulator.add_pollution_source(
                    source_type=source_type,
                    location=tuple(source_config['location']),
                    intensity=source_config['intensity'],
                    duration=source_config.get('duration'),
                    radius=source_config.get('radius', 0.01)
                )

        # 执行模拟
        simulation_result = simulator.simulate_diffusion(
            simulation_hours=simulation_config.get('simulation_hours', 24.0),
            time_step_minutes=simulation_config.get('time_step_minutes', 10.0),
            diffusion_coeff=simulation_config.get('diffusion_coeff', 100.0),
            decay_rate=simulation_config.get('decay_rate', 0.0)
        )

        if not simulation_result["success"]:
            return simulation_result

        # 创建动画
        animation_result = simulator.create_diffusion_animation(
            simulation_result=simulation_result,
            output_path=output_path,
            title=title,
            fps=simulation_config.get('fps', 10),
            show_sources=simulation_config.get('show_sources', True),
            show_velocity=simulation_config.get('show_velocity', False),
            colormap=simulation_config.get('colormap', 'Reds')
        )

        return animation_result

    except Exception as e:
        logging.error(f"创建污染物扩散动画失败: {e}")
        return {
            "success": False,
            "message": f"动画创建失败: {str(e)}"
        }

    finally:
        if simulator:
            simulator.close()


if __name__ == "__main__":
    # 测试污染物扩散动画系统
    import os

    logging.basicConfig(level=logging.INFO)

    print("🏭 测试污染物扩散模拟动画系统")
    print("-" * 60)

    # 测试数据路径
    test_netcdf_path = "../data/raw_data/merged_data.nc"

    if not os.path.exists(test_netcdf_path):
        print(f"❌ 测试文件不存在: {test_netcdf_path}")
        print("请确保NetCDF文件路径正确")
        exit(1)

    # 创建输出目录
    os.makedirs("test_outputs", exist_ok=True)

    # 测试1: 石油泄漏事故模拟
    print("🛢️ 测试1: 石油泄漏事故模拟")

    oil_spill_config = {
        "sources": [
            {
                "type": "oil_spill",
                "location": [22.5, 119.5],  # 台湾海峡中部
                "volume": 1000,  # 1000立方米石油
                "duration": 2.0  # 2小时泄漏
            }
        ]
    }

    oil_simulation_config = {
        "lon_range": [118.5, 120.5],
        "lat_range": [21.5, 23.5],
        "grid_resolution": 0.008,
        "simulation_hours": 48.0,
        "time_step_minutes": 15.0,
        "diffusion_coeff": 150.0,  # 石油扩散系数
        "decay_rate": 0.0001,      # 风化率
        "fps": 12,
        "show_sources": True,
        "show_velocity": False,
        "colormap": "Reds"
    }

    result_1 = create_pollution_diffusion_animation(
        netcdf_path=test_netcdf_path,
        pollution_config=oil_spill_config,
        simulation_config=oil_simulation_config,
        output_path="test_outputs/oil_spill_diffusion.gif",
        title="台湾海峡石油泄漏扩散模拟"
    )

    print(f"结果: {'✅ 成功' if result_1['success'] else '❌ 失败'}")
    if result_1['success']:
        stats = result_1['animation_stats']
        print(f"   输出文件: {result_1['output_path']}")
        print(f"   动画帧数: {stats['n_frames']}")
        print(f"   最大浓度: {stats['max_concentration']:.2e} kg/m³")
        print(f"   模拟时长: {stats['simulation_hours']} 小时")
    else:
        print(f"   错误: {result_1['message']}")

    # 测试2: 工业排放连续污染
    print("\n🏭 测试2: 工业排放连续污染")

    industrial_config = {
        "sources": [
            {
                "type": "industrial",
                "location": [23.0, 119.2],  # 福建沿海
                "intensity": 0.5,  # 0.5 kg/s 连续排放
                "pollutant_type": "heavy_metals"
            }
        ]
    }

    industrial_simulation_config = {
        "lon_range": [118.8, 119.8],
        "lat_range": [22.5, 23.5],
        "grid_resolution": 0.005,
        "simulation_hours": 72.0,  # 3天
        "time_step_minutes": 20.0,
        "diffusion_coeff": 80.0,
        "decay_rate": 0.0005,  # 生物降解
        "fps": 10,
        "show_sources": True,
        "show_velocity": True,
        "colormap": "Blues"
    }

    result_2 = create_pollution_diffusion_animation(
        netcdf_path=test_netcdf_path,
        pollution_config=industrial_config,
        simulation_config=industrial_simulation_config,
        output_path="test_outputs/industrial_pollution.gif",
        title="工业重金属排放扩散模拟"
    )

    print(f"结果: {'✅ 成功' if result_2['success'] else '❌ 失败'}")
    if result_2['success']:
        stats = result_2['animation_stats']
        print(f"   输出文件: {result_2['output_path']}")
        print(f"   污染源数: {stats['n_sources']}")
        print(f"   模拟时长: {stats['simulation_hours']} 小时")
    else:
        print(f"   错误: {result_2['message']}")

    # 测试3: 多点污染源综合场景
    print("\n🌊 测试3: 多点污染源综合场景")

    multi_source_config = {
        "sources": [
            {
                "type": "oil_spill",
                "location": [22.3, 119.3],
                "volume": 500,
                "duration": 1.0
            },
            {
                "type": "industrial",
                "location": [22.8, 119.7],
                "intensity": 0.3,
                "pollutant_type": "chemical"
            },
            {
                "type": "area",
                "location": [23.1, 119.1],
                "intensity": 2.0,
                "duration": 6.0,
                "radius": 0.02
            }
        ]
    }

    multi_simulation_config = {
        "lon_range": [119.0, 120.0],
        "lat_range": [22.0, 23.5],
        "grid_resolution": 0.006,
        "simulation_hours": 96.0,  # 4天
        "time_step_minutes": 12.0,
        "diffusion_coeff": 120.0,
        "decay_rate": 0.0002,
        "fps": 15,
        "show_sources": True,
        "show_velocity": False,
        "colormap": "plasma"
    }

    result_3 = create_pollution_diffusion_animation(
        netcdf_path=test_netcdf_path,
        pollution_config=multi_source_config,
        simulation_config=multi_simulation_config,
        output_path="test_outputs/multi_source_pollution.gif",
        title="多源污染综合扩散模拟"
    )

    print(f"结果: {'✅ 成功' if result_3['success'] else '❌ 失败'}")
    if result_3['success']:
        stats = result_3['animation_stats']
        print(f"   输出文件: {result_3['output_path']}")
        print(f"   污染源数: {stats['n_sources']}")
        print(f"   最大浓度: {stats['max_concentration']:.2e} kg/m³")
    else:
        print(f"   错误: {result_3['message']}")

    # 测试4: 高分辨率近海污染详细模拟
    print("\n🔬 测试4: 高分辨率近海污染详细模拟")

    detailed_config = {
        "sources": [
            {
                "type": "point",
                "location": [22.6, 119.4],
                "intensity": 5.0,
                "duration": 0.5,  # 30分钟突发事故
                "radius": 0.002
            }
        ]
    }

    detailed_simulation_config = {
        "lon_range": [119.2, 119.6],
        "lat_range": [22.4, 22.8],
        "grid_resolution": 0.002,  # 高分辨率
        "simulation_hours": 24.0,
        "time_step_minutes": 5.0,  # 高时间分辨率
        "diffusion_coeff": 200.0,
        "decay_rate": 0.001,
        "fps": 20,
        "show_sources": True,
        "show_velocity": True,
        "colormap": "YlOrRd"
    }

    result_4 = create_pollution_diffusion_animation(
        netcdf_path=test_netcdf_path,
        pollution_config=detailed_config,
        simulation_config=detailed_simulation_config,
        output_path="test_outputs/detailed_pollution.gif",
        title="高分辨率近海污染扩散详细模拟"
    )

    print(f"结果: {'✅ 成功' if result_4['success'] else '❌ 失败'}")
    if result_4['success']:
        stats = result_4['animation_stats']
        print(f"   输出文件: {result_4['output_path']}")
        print(f"   动画帧数: {stats['n_frames']}")
    else:
        print(f"   错误: {result_4['message']}")

    # 总结测试结果
    print("\n" + "=" * 60)
    print("🎯 污染物扩散模拟测试总结")
    print("-" * 40)

    results = [result_1, result_2, result_3, result_4]
    test_names = ["石油泄漏", "工业排放", "多源污染", "高分辨率"]

    success_count = sum(1 for r in results if r['success'])

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"{i+1}. {name}: {status}")

    print(f"\n总计: {success_count}/{len(results)} 个测试成功")

    if success_count == len(results):
        print("🎉 所有测试通过！污染物扩散模拟系统运行正常。")
        print("\n📁 生成的动画文件:")
        output_files = [
            "oil_spill_diffusion.gif",
            "industrial_pollution.gif",
            "multi_source_pollution.gif",
            "detailed_pollution.gif"
        ]

        for filename in output_files:
            filepath = f"test_outputs/{filename}"
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"   {filename} ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  {len(results) - success_count} 个测试失败，请检查错误信息。")

    print("\n💡 使用说明:")
    print("1. 调整 grid_resolution 控制模拟精度")
    print("2. 修改 diffusion_coeff 调整扩散速度")
    print("3. 设置 decay_rate 模拟污染物降解")
    print("4. 使用不同 colormap 突出显示效果")
    print("5. 启用 show_velocity 显示海流影响")

    print("=" * 60)