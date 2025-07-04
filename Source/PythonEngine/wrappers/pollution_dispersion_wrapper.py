#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: pollution_dispersion_wrapper.py
位置: Source/PythonEngine/wrappers/pollution_dispersion_wrapper.py
功能: C#调用污染物扩散模拟的包装器脚本
用法: python pollution_dispersion_wrapper.py input.json output.json
"""

import sys
import json
from pathlib import Path
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import os


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加Python引擎路径到sys.path
current_dir = Path(__file__).parent
python_engine_root = current_dir.parent
sys.path.insert(0, str(python_engine_root))

class AdaptivePollutionAnimator:
    """自适应污染扩散动画生成器 - 自动适配NetCDF数据范围"""

    def __init__(self, netcdf_path: str, grid_resolution: float = 0.01):
        """
        初始化自适应污染动画生成器
        
        Args:
            netcdf_path: NetCDF数据文件路径
            grid_resolution: 网格分辨率（度）
        """
        self.netcdf_path = netcdf_path
        self.grid_resolution = grid_resolution

        # 数据范围（将从NetCDF文件自动读取）
        self.lon_range = None
        self.lat_range = None
        self.data_shape = None

        # 海洋数据
        self.water_u = None
        self.water_v = None
        self.lat = None
        self.lon = None

        # 模拟网格
        self.sim_lon = None
        self.sim_lat = None
        self.sim_u = None
        self.sim_v = None

        # 污染物浓度场
        self.concentration = None
        self.concentration_history = []

        # 修复的物理参数
        self.diffusion_coeff = 500.0  # 增加扩散系数 m²/s
        self.decay_rate = 0.00005     # 减少衰减率 1/s

        self.is_initialized = False

    def analyze_netcdf_data(self) -> Dict[str, Any]:
        """
        分析NetCDF数据，获取地理范围和基本信息
        
        Returns:
            数据分析结果
        """
        try:
            with Dataset(self.netcdf_path, 'r') as nc_data:
                # 获取维度和变量信息
                dims = dict(nc_data.dimensions)
                variables = list(nc_data.variables.keys())

                # 获取坐标信息
                lat = nc_data.variables['lat'][:]
                lon = nc_data.variables['lon'][:]

                # 获取地理范围
                self.lat_range = (float(lat.min()), float(lat.max()))
                self.lon_range = (float(lon.min()), float(lon.max()))

                # 获取数据形状
                if 'water_u' in nc_data.variables:
                    u_shape = nc_data.variables['water_u'].shape
                    self.data_shape = u_shape

                # 获取时间信息
                time_steps = len(nc_data.variables['time'][:]) if 'time' in nc_data.variables else 1

                # 计算地理范围（公里）
                lat_span_km = (self.lat_range[1] - self.lat_range[0]) * 111.32
                lon_span_km = (self.lon_range[1] - self.lon_range[0]) * 111.32 * np.cos(np.radians(np.mean(self.lat_range)))

                analysis_result = {
                    "success": True,
                    "geographic_info": {
                        "lat_range": self.lat_range,
                        "lon_range": self.lon_range,
                        "lat_span_km": lat_span_km,
                        "lon_span_km": lon_span_km,
                        "center_lat": np.mean(self.lat_range),
                        "center_lon": np.mean(self.lon_range)
                    },
                    "data_info": {
                        "dimensions": dims,
                        "variables": variables,
                        "data_shape": self.data_shape,
                        "time_steps": time_steps,
                        "lat_points": len(lat),
                        "lon_points": len(lon)
                    }
                }

                logging.info("NetCDF数据分析完成:")
                logging.info(f"  地理范围: {self.lat_range[0]:.2f}°-{self.lat_range[1]:.2f}°N, {self.lon_range[0]:.2f}°-{self.lon_range[1]:.2f}°E")
                logging.info(f"  空间跨度: {lat_span_km:.1f} x {lon_span_km:.1f} km")
                logging.info(f"  数据网格: {len(lat)} x {len(lon)}")

                return analysis_result

        except Exception as e:
            logging.error(f"NetCDF数据分析失败: {e}")
            return {
                "success": False,
                "message": f"数据分析失败: {str(e)}"
            }

    def initialize(self, time_index: int = 0, depth_index: int = 0) -> bool:
        """
        初始化环境，自动适配NetCDF数据范围
        
        Args:
            time_index: 时间索引
            depth_index: 深度索引
            
        Returns:
            是否初始化成功
        """
        try:
            logging.info("初始化自适应污染扩散环境")

            # 先分析数据
            analysis = self.analyze_netcdf_data()
            if not analysis["success"]:
                return False

            # 加载NetCDF数据
            with Dataset(self.netcdf_path, 'r') as nc_data:
                # 获取原始数据
                self.lat = nc_data.variables['lat'][:]
                self.lon = nc_data.variables['lon'][:]

                # 获取速度场数据
                if len(nc_data.variables['water_u'].shape) == 4:  # (time, depth, lat, lon)
                    self.water_u = nc_data.variables['water_u'][time_index, depth_index, :, :]
                    self.water_v = nc_data.variables['water_v'][time_index, depth_index, :, :]
                elif len(nc_data.variables['water_u'].shape) == 3:  # (time, lat, lon)
                    self.water_u = nc_data.variables['water_u'][time_index, :, :]
                    self.water_v = nc_data.variables['water_v'][time_index, :, :]
                else:  # (lat, lon)
                    self.water_u = nc_data.variables['water_u'][:, :]
                    self.water_v = nc_data.variables['water_v'][:, :]

                # 处理掩码数组
                if isinstance(self.water_u, np.ma.MaskedArray):
                    self.water_u = self.water_u.filled(0)
                if isinstance(self.water_v, np.ma.MaskedArray):
                    self.water_v = self.water_v.filled(0)

            # 创建模拟网格
            self._create_simulation_grid()

            # 插值速度场到模拟网格
            self._interpolate_velocity_field()

            # 初始化浓度场
            self._initialize_concentration_field()

            self.is_initialized = True
            logging.info(f"环境初始化成功，模拟网格: {len(self.sim_lat)} x {len(self.sim_lon)}")

            return True

        except Exception as e:
            logging.error(f"初始化失败: {e}")
            return False

    def _create_simulation_grid(self):
        """创建模拟网格"""
        # 基于数据范围创建网格
        self.sim_lon = np.arange(self.lon_range[0], self.lon_range[1] + self.grid_resolution, self.grid_resolution)
        self.sim_lat = np.arange(self.lat_range[0], self.lat_range[1] + self.grid_resolution, self.grid_resolution)

        logging.info(f"创建模拟网格: {len(self.sim_lat)} x {len(self.sim_lon)} (分辨率: {self.grid_resolution}°)")

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

        # 创建模拟网格坐标
        sim_lon_grid, sim_lat_grid = np.meshgrid(self.sim_lon, self.sim_lat, indexing='xy')

        # 插值
        points = np.column_stack([sim_lat_grid.ravel(), sim_lon_grid.ravel()])
        self.sim_u = u_interp(points).reshape(sim_lat_grid.shape)
        self.sim_v = v_interp(points).reshape(sim_lat_grid.shape)

        logging.info("海流速度场插值完成")

    def _initialize_concentration_field(self):
        """初始化浓度场"""
        self.concentration = np.zeros((len(self.sim_lat), len(self.sim_lon)))
        self.concentration_history = []

    def add_pollution_source(self, location, intensity, radius=None):
        """修复版本的污染源添加函数 - 实现真实海洋扩散形状"""
        try:
            lat_src, lon_src = location

            if not (self.lat_range[0] <= lat_src <= self.lat_range[1] and
                    self.lon_range[0] <= lon_src <= self.lon_range[1]):
                logging.warning(f"污染源位置超出范围")
                return False

            # 设置初始扩散参数
            if radius is None:
                lat_span = self.lat_range[1] - self.lat_range[0]
                lon_span = self.lon_range[1] - self.lon_range[0]
                radius = min(lat_span, lon_span) * 0.03  # 减小初始半径

            # 转换为网格索引
            lat_idx = np.argmin(np.abs(self.sim_lat - lat_src))
            lon_idx = np.argmin(np.abs(self.sim_lon - lon_src))

            # 创建更真实的海洋污染扩散形状
            radius_grid = max(2, int(radius / self.grid_resolution))

            # 创建椭圆形初始扩散（模拟油膜等污染物的自然形状）
            y_indices, x_indices = np.ogrid[-lat_idx:len(self.sim_lat)-lat_idx,
                                   -lon_idx:len(self.sim_lon)-lon_idx]

            # 椭圆参数 - 考虑海流方向
            if hasattr(self, 'sim_u') and hasattr(self, 'sim_v'):
                # 根据当地海流方向调整椭圆形状
                local_u = self.sim_u[lat_idx, lon_idx] if self.sim_u is not None else 0
                local_v = self.sim_v[lat_idx, lon_idx] if self.sim_v is not None else 0

                # 计算海流方向角度
                flow_angle = np.arctan2(local_v, local_u)

                # 椭圆长短轴比例（长轴沿海流方向）
                a = radius_grid * 1.5  # 长轴
                b = radius_grid * 0.8  # 短轴

                # 旋转椭圆以对齐海流方向
                cos_angle = np.cos(flow_angle)
                sin_angle = np.sin(flow_angle)

                # 坐标变换
                x_rot = x_indices * cos_angle + y_indices * sin_angle
                y_rot = -x_indices * sin_angle + y_indices * cos_angle

                # 椭圆方程
                ellipse_mask = (x_rot/a)**2 + (y_rot/b)**2
            else:
                # 如果没有海流数据，使用圆形
                ellipse_mask = (x_indices**2 + y_indices**2) / (radius_grid**2)

            # 创建更真实的浓度分布
            sigma_factor = 2.0  # 控制扩散的平缓程度

            # 多层次浓度分布（模拟真实污染物扩散）
            # 核心高浓度区域
            core_concentration = np.exp(-ellipse_mask / (sigma_factor**2))

            # 外围低浓度区域（模拟溶解扩散）
            outer_mask = ellipse_mask * 2.0
            outer_concentration = 0.3 * np.exp(-outer_mask / (sigma_factor**2))

            # 合并浓度分布
            total_concentration = np.maximum(core_concentration, outer_concentration)

            # 添加随机扰动（模拟海洋湍流影响）
            if total_concentration.shape[0] > 0 and total_concentration.shape[1] > 0:
                noise_scale = 0.1
                noise = noise_scale * np.random.randn(*total_concentration.shape)
                total_concentration += noise
                total_concentration = np.maximum(0, total_concentration)  # 确保非负

            # 应用强度
            pollution_field = total_concentration * intensity / np.max(total_concentration) if np.max(total_concentration) > 0 else total_concentration * intensity

            # 添加到浓度场
            self.concentration += pollution_field

            logging.info(f"添加海洋污染源成功，位置: ({lat_src:.3f}°N, {lon_src:.3f}°E), 半径: {radius:.4f}°")
            return True

        except Exception as e:
            logging.error(f"添加污染源失败: {e}")
            return False

    def simulate_diffusion_step(self, dt=600.0):
        """修复版本的扩散步骤 - 自动调整时间步长以满足CFL条件"""
        c_old = self.concentration.copy()

        # 空间步长（米）
        dx = self.grid_resolution * 111320
        dy = self.grid_resolution * 111320

        # 检查CFL条件并自动调整时间步长
        max_u = np.max(np.abs(self.sim_u))
        max_v = np.max(np.abs(self.sim_v))
        max_velocity = max(max_u, max_v, 1e-10)

        # CFL条件：dt < dx / (2 * max_velocity)
        cfl_limit = 0.3 * min(dx, dy) / max_velocity  # 使用更保守的0.3系数

        if dt > cfl_limit:
            # 自动将时间步长分割成多个小步
            n_substeps = int(np.ceil(dt / cfl_limit))
            sub_dt = dt / n_substeps

            # 多步积分
            for _ in range(n_substeps):
                self._single_diffusion_step(sub_dt)
        else:
            self._single_diffusion_step(dt)

    def _single_diffusion_step(self, dt):
        """执行单个扩散时间步 - 增强海洋扩散真实性"""
        c_old = self.concentration.copy()

        # 空间步长（米）
        dx = self.grid_resolution * 111320
        dy = self.grid_resolution * 111320

        # 计算梯度
        dc_dy, dc_dx = np.gradient(c_old, dy, dx)

        # 对流项（海流输运）
        advection_x = -self.sim_u * dc_dx
        advection_y = -self.sim_v * dc_dy

        # 各向异性扩散（海洋中的扩散不是各向同性的）
        # 主扩散方向沿海流方向，垂直方向扩散较弱
        u_magnitude = np.sqrt(self.sim_u**2 + self.sim_v**2) + 1e-10

        # 海流方向的单位向量
        u_dir = self.sim_u / u_magnitude
        v_dir = self.sim_v / u_magnitude

        # 沿海流方向的扩散系数更大
        parallel_diffusion = self.diffusion_coeff * 2.0
        perpendicular_diffusion = self.diffusion_coeff * 0.5

        # 各向异性扩散张量
        d2c_dx2 = np.gradient(np.gradient(c_old, dx, axis=1), dx, axis=1)
        d2c_dy2 = np.gradient(np.gradient(c_old, dy, axis=0), dy, axis=0)
        d2c_dxdy = np.gradient(np.gradient(c_old, dy, axis=0), dx, axis=1)

        # 扩散项（考虑海流方向的各向异性）
        diffusion_parallel = parallel_diffusion * (
                u_dir**2 * d2c_dx2 + v_dir**2 * d2c_dy2 + 2*u_dir*v_dir*d2c_dxdy
        )
        diffusion_perpendicular = perpendicular_diffusion * (
                v_dir**2 * d2c_dx2 + u_dir**2 * d2c_dy2 - 2*u_dir*v_dir*d2c_dxdy
        )
        diffusion = diffusion_parallel + diffusion_perpendicular

        # 湍流扩散（模拟海洋小尺度湍流）
        turbulent_diffusion = 50.0 * (d2c_dx2 + d2c_dy2)

        # 风剪切影响（表面污染物受风影响）
        wind_effect = 0.1 * self.diffusion_coeff * (d2c_dx2 + d2c_dy2)

        # 总扩散
        total_diffusion = diffusion + turbulent_diffusion + wind_effect

        # 非线性衰减（浓度越高衰减越快，模拟微生物降解等）
        nonlinear_decay = -self.decay_rate * c_old * (1 + 0.1 * c_old / np.max(c_old + 1e-10))

        # 时间积分
        dc_dt = advection_x + advection_y + total_diffusion + nonlinear_decay
        self.concentration = c_old + dt * dc_dt

        # 确保浓度非负
        self.concentration = np.maximum(0, self.concentration)

        # 海洋边界条件（开放边界，污染物可以流出）
        boundary_width = 2
        for i in range(boundary_width):
            factor = (i + 1) / boundary_width * 0.9  # 边界处轻微衰减
            self.concentration[i, :] *= factor
            self.concentration[-1-i, :] *= factor
            self.concentration[:, i] *= factor
            self.concentration[:, -1-i] *= factor

        # 适度平滑（模拟海洋中的自然混合）
        self.concentration = gaussian_filter(self.concentration, sigma=0.2)

    def create_pollution_animation(self, pollution_sources, simulation_hours=48.0,
                                   time_step_minutes=10.0, output_path="pollution_diffusion.gif",
                                   title="海洋污染扩散模拟", colormap="custom_pollution",
                                   show_velocity=False, fps=15):
        """修复版本的动画创建函数"""
        if not self.is_initialized:
            raise ValueError("环境未初始化")

        try:
            logging.info("开始创建污染扩散动画")

            # 添加污染源
            for source in pollution_sources:
                self.add_pollution_source(
                    location=tuple(source['location']),
                    intensity=source['intensity'],
                    radius=source.get('radius')
                )

            # 模拟参数
            dt = time_step_minutes * 60
            n_steps = int(simulation_hours * 3600 / dt)
            save_interval = max(1, n_steps // 150)  # 最多保存150帧

            logging.info(f"执行扩散模拟: {n_steps} 步，每 {save_interval} 步保存一帧")

            # 执行模拟并保存历史
            for step in range(n_steps):
                self.simulate_diffusion_step(dt)

                # 保存历史
                if step % save_interval == 0:
                    self.concentration_history.append(self.concentration.copy())

                if step % (n_steps // 10) == 0:
                    logging.info(f"模拟进度: {step/n_steps*100:.1f}%")

            # 确保保存最终状态
            self.concentration_history.append(self.concentration.copy())

            # 创建动画
            anim_result = self._create_geographic_animation(
                title=title,
                output_path=output_path,
                fps=fps,
                pollution_sources=pollution_sources,
                time_step_minutes=time_step_minutes * save_interval,
                colormap=colormap,
                show_velocity=show_velocity
            )

            return anim_result

        except Exception as e:
            logging.error(f"创建动画失败: {e}")
            return {
                "success": False,
                "message": f"创建动画失败: {str(e)}"
            }

    def _create_geographic_animation(self, title, output_path, fps, pollution_sources,
                                     time_step_minutes, colormap, show_velocity):
        """修复版本的地理动画创建函数"""

        # 创建自定义颜色映射
        if colormap == "custom_pollution":
            colors = ['white', 'lightblue', 'yellow', 'orange', 'red', 'darkred', 'maroon']
            cmap = LinearSegmentedColormap.from_list('pollution', colors, N=256)
            cmap.set_under('white', alpha=0)  # 设置低于最小值的颜色为透明
        else:
            cmap = plt.get_cmap(colormap)

        # 计算浓度范围 - 修复：更合理的范围设置
        max_concentration = np.max([np.max(c) for c in self.concentration_history])
        if max_concentration == 0:
            max_concentration = 1e-6

        # 修复：使用线性归一化，设置合理的阈值
        threshold = max_concentration * 0.001  # 0.1%作为显示阈值
        vmin = threshold
        vmax = max_concentration

        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # 设置地理范围
        extent = list(self.lon_range) + list(self.lat_range)
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # 添加地理要素
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, color='black', zorder=10)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.8, zorder=5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1.0, color='darkgray', zorder=10)

        # 网格线
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.8, color='gray', alpha=0.6, linestyle='--')
        gl.xlabel_style = {'size': 12, 'color': 'black'}
        gl.ylabel_style = {'size': 12, 'color': 'black'}
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        # 创建网格坐标
        LON, LAT = np.meshgrid(self.sim_lon, self.sim_lat)

        # 修复：使用线性归一化，只显示有污染的区域
        norm = Normalize(vmin=vmin, vmax=vmax)

        # 修复：只显示超过阈值的污染区域
        initial_data = self.concentration_history[0].copy()
        # 创建掩码：低于阈值的区域设为NaN（不显示）
        initial_data_masked = np.where(initial_data >= threshold, initial_data, np.nan)

        im = ax.pcolormesh(LON, LAT, initial_data_masked,
                           cmap=cmap, norm=norm, alpha=0.85,
                           transform=ccrs.PlateCarree(), zorder=3,
                           shading='auto')

        # 颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=30, extend='min')
        cbar.set_label('污染物浓度 (kg/m³)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)

        # 显示速度场（可选）
        velocity_arrows = None
        if show_velocity:
            skip = max(1, len(self.sim_lon) // 25)  # 适当降采样
            velocity_arrows = ax.quiver(
                LON[::skip, ::skip], LAT[::skip, ::skip],
                self.sim_u[::skip, ::skip], self.sim_v[::skip, ::skip],
                scale=50, alpha=0.6, color='gray', width=0.002,
                transform=ccrs.PlateCarree(), zorder=4
            )

        # 标记污染源
        for i, source in enumerate(pollution_sources):
            lat_src, lon_src = source['location']
            ax.plot(lon_src, lat_src, marker='o', markersize=15,
                    markerfacecolor='red', markeredgecolor='black',
                    markeredgewidth=2, transform=ccrs.PlateCarree(),
                    zorder=15, label=f'污染源 {i+1}' if i < 5 else "")

        # 信息文本框
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            verticalalignment='top', fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.8',
                                      facecolor='white', alpha=0.95, edgecolor='black'),
                            zorder=20)

        # 标题
        ax.set_title(title, fontsize=18, fontweight='bold', pad=25)

        # 图例
        if pollution_sources:
            ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.92),
                      fontsize=12, framealpha=0.9)

        def animate(frame):
            """修复版本的动画更新函数"""
            current_concentration = self.concentration_history[frame].copy()
            current_time = frame * time_step_minutes / 60

            # 修复：只显示超过阈值的污染区域
            current_data_masked = np.where(current_concentration >= threshold, current_concentration, np.nan)

            # 修复：正确更新数组，处理NaN值
            # 将NaN转换为一个很小的值用于set_array
            display_data = np.where(np.isnan(current_data_masked), vmin * 0.1, current_data_masked)
            im.set_array(display_data.ravel())

            # 计算统计信息
            grid_area = (111320 * self.grid_resolution)**2  # 单个网格面积 m²

            # 只计算有效污染区域的统计
            valid_pollution = current_concentration[current_concentration >= threshold]

            if len(valid_pollution) > 0:
                total_mass = np.sum(valid_pollution) * grid_area
                max_conc = np.max(valid_pollution)
                affected_cells = len(valid_pollution)
                affected_area = affected_cells * grid_area / 1e6  # 转换为km²
            else:
                total_mass = 0
                max_conc = 0
                affected_area = 0

            # 更新信息文本
            if current_time < 24:
                time_str = f"{current_time:.1f} 小时"
            else:
                days = int(current_time // 24)
                hours = current_time % 24
                time_str = f"{days} 天 {hours:.1f} 小时"

            # 获取数据区域描述
            center_lat = np.mean(self.lat_range)
            center_lon = np.mean(self.lon_range)
            region_desc = f"({center_lat:.1f}°N, {center_lon:.1f}°E)"

            info_str = f'🕐 模拟时间: {time_str}\n'
            info_str += f'🔴 最高浓度: {max_conc:.2e} kg/m³\n'
            info_str += f'⚖️ 总质量: {total_mass:.1e} kg\n'
            info_str += f'📍 影响面积: {affected_area:.1f} km²\n'
            info_str += f'🌊 海域范围: {region_desc}'

            info_text.set_text(info_str)

            return [im, info_text]

        # 创建动画
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.concentration_history),
            interval=1000//fps, blit=False, repeat=True
        )

        # 保存动画
        plt.tight_layout()

        try:
            if output_path.endswith('.gif'):
                anim.save(output_path, writer='pillow', fps=fps, dpi=120)
            elif output_path.endswith('.mp4'):
                anim.save(output_path, writer='ffmpeg', fps=fps, dpi=120)
            else:
                output_path += '.gif'
                anim.save(output_path, writer='pillow', fps=fps, dpi=120)

            plt.close(fig)

            logging.info(f"污染扩散动画保存成功: {output_path}")

            return {
                "success": True,
                "output_path": output_path,
                "animation_stats": {
                    "frames": len(self.concentration_history),
                    "max_concentration": max_concentration,
                    "simulation_hours": len(self.concentration_history) * time_step_minutes / 60,
                    "pollution_sources": len(pollution_sources),
                    "geographic_range": {
                        "lat_range": self.lat_range,
                        "lon_range": self.lon_range
                    }
                }
            }

        except Exception as e:
            plt.close(fig)
            return {
                "success": False,
                "message": f"保存动画失败: {str(e)}"
            }


def create_adaptive_pollution_animation(
        netcdf_path: str,
        pollution_sources: List[Dict[str, Any]],
        simulation_hours: float = 48.0,
        time_step_minutes: float = 10.0,
        output_path: str = "adaptive_pollution_diffusion.gif",
        title: str = "海洋污染扩散模拟",
        grid_resolution: float = 0.008,
        show_velocity: bool = True,
        colormap: str = "custom_pollution") -> Dict[str, Any]:
    """
    创建自适应污染扩散动画的便捷函数
    
    Args:
        netcdf_path: NetCDF数据文件路径
        pollution_sources: 污染源列表，格式: [{"location": [lat, lon], "intensity": float}, ...]
        simulation_hours: 模拟时长（小时）
        time_step_minutes: 时间步长（分钟）
        output_path: 输出路径
        title: 动画标题
        grid_resolution: 网格分辨率（度）
        show_velocity: 是否显示海流矢量场
        colormap: 颜色映射
        
    Returns:
        动画创建结果
    """
    try:
        # 创建自适应动画生成器
        animator = AdaptivePollutionAnimator(netcdf_path, grid_resolution)

        # 初始化
        if not animator.initialize():
            return {"success": False, "message": "初始化失败"}

        # 创建动画
        result = animator.create_pollution_animation(
            pollution_sources=pollution_sources,
            simulation_hours=simulation_hours,
            time_step_minutes=time_step_minutes,
            output_path=output_path,
            title=title,
            colormap=colormap,
            show_velocity=show_velocity,
            fps=12
        )

        return result

    except Exception as e:
        logging.error(f"创建自适应污染扩散动画失败: {e}")
        return {
            "success": False,
            "message": f"动画创建失败: {str(e)}"
        }


def run_pollution_dispersion(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行污染物扩散模拟
    
    Args:
        input_data: 输入参数字典
        
    Returns:
        模拟结果字典
    """
    try:
        logger.info("开始执行污染物扩散模拟")

        # 获取参数
        parameters = input_data.get('parameters', {})
        netcdf_path = parameters.get('netcdf_path')
        output_path = parameters.get('output_path', 'pollution_dispersion.gif')

        # 验证NetCDF文件路径
        if not netcdf_path:
            return {
                "success": False,
                "message": "未提供NetCDF文件路径",
                "error_code": "MISSING_NETCDF_PATH"
            }

        if not os.path.exists(netcdf_path):
            return {
                "success": False,
                "message": f"NetCDF文件不存在: {netcdf_path}",
                "error_code": "NETCDF_FILE_NOT_FOUND"
            }

        # 提取模拟参数
        pollution_sources = parameters.get('pollution_sources', [])
        if not pollution_sources:
            # 使用默认污染源配置
            pollution_sources = [
                {
                    "location": [23.0, 120.0],  # 台湾海峡中部
                    "intensity": 1000.0,
                    "name": "默认污染源"
                }
            ]

        simulation_hours = float(parameters.get('simulation_hours', 24.0))
        time_step_minutes = float(parameters.get('time_step_minutes', 10.0))
        grid_resolution = float(parameters.get('grid_resolution', 0.01))
        title = parameters.get('title', '海洋污染扩散模拟')
        show_velocity = parameters.get('show_velocity', True)
        colormap = parameters.get('colormap', 'custom_pollution')

        logger.info(f"模拟参数: 时长={simulation_hours}小时, 步长={time_step_minutes}分钟")
        logger.info(f"污染源数量: {len(pollution_sources)}")

        # 创建污染扩散动画
        result = create_adaptive_pollution_animation(
            netcdf_path=netcdf_path,
            pollution_sources=pollution_sources,
            simulation_hours=simulation_hours,
            time_step_minutes=time_step_minutes,
            output_path=output_path,
            title=title,
            grid_resolution=grid_resolution,
            show_velocity=show_velocity,
            colormap=colormap
        )

        if result["success"]:
            # 获取动画统计信息
            stats = result.get("animation_stats", {})
            file_size = 0
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024 / 1024  # MB

            return {
                "success": True,
                "message": "污染物扩散模拟成功完成",
                "output_path": output_path,
                "statistics": {
                    "max_concentration": stats.get("max_concentration", 0.0),
                    "mean_concentration": 0.0,  # 可以从浓度场计算
                    "total_mass": stats.get("max_concentration", 0.0) * 1000,  # 估算值
                    "simulation_frames": stats.get("frames", 0),
                    "simulation_hours": stats.get("simulation_hours", simulation_hours),
                    "file_size_mb": file_size,
                    "geographic_range": stats.get("geographic_range", {}),
                    "pollution_sources_count": len(pollution_sources)
                },
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "netcdf_source": netcdf_path,
                    "grid_resolution": grid_resolution,
                    "time_step_minutes": time_step_minutes
                }
            }
        else:
            return {
                "success": False,
                "message": f"污染物扩散模拟失败: {result.get('message', '未知错误')}",
                "error_code": "SIMULATION_FAILED"
            }

    except Exception as e:
        logger.error(f"污染物扩散模拟异常: {e}")
        return {
            "success": False,
            "message": f"模拟过程发生异常: {str(e)}",
            "error_code": "SIMULATION_EXCEPTION",
            "traceback": traceback.format_exc()
        }


def analyze_netcdf_for_pollution(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析NetCDF文件，为污染扩散模拟提供信息
    
    Args:
        input_data: 输入参数字典
        
    Returns:
        分析结果字典
    """
    try:
        logger.info("分析NetCDF文件结构")

        parameters = input_data.get('parameters', {})
        netcdf_path = parameters.get('netcdf_path')

        if not netcdf_path or not os.path.exists(netcdf_path):
            return {
                "success": False,
                "message": "NetCDF文件路径无效",
                "error_code": "INVALID_NETCDF_PATH"
            }

        # 创建分析器
        animator = AdaptivePollutionAnimator(netcdf_path)
        analysis_result = animator.analyze_netcdf_data()

        if analysis_result["success"]:
            geo_info = analysis_result["geographic_info"]
            data_info = analysis_result["data_info"]

            # 推荐污染源位置（基于数据中心区域）
            center_lat = geo_info["center_lat"]
            center_lon = geo_info["center_lon"]
            lat_span = geo_info["lat_range"][1] - geo_info["lat_range"][0]
            lon_span = geo_info["lon_range"][1] - geo_info["lon_range"][0]

            suggested_sources = [
                {
                    "location": [center_lat, center_lon],
                    "intensity": 1000.0,
                    "description": "中心区域主要污染源"
                },
                {
                    "location": [center_lat + lat_span * 0.2, center_lon - lon_span * 0.2],
                    "intensity": 800.0,
                    "description": "次要污染源1"
                },
                {
                    "location": [center_lat - lat_span * 0.2, center_lon + lon_span * 0.2],
                    "intensity": 600.0,
                    "description": "次要污染源2"
                }
            ]

            return {
                "success": True,
                "message": "NetCDF文件分析完成",
                "analysis_result": analysis_result,
                "recommendations": {
                    "suggested_pollution_sources": suggested_sources,
                    "optimal_grid_resolution": min(lat_span, lon_span) / 100,  # 1%的区域跨度
                    "recommended_simulation_hours": 48.0,
                    "recommended_time_step_minutes": 15.0
                }
            }
        else:
            return {
                "success": False,
                "message": f"NetCDF文件分析失败: {analysis_result.get('message', '未知错误')}",
                "error_code": "ANALYSIS_FAILED"
            }

    except Exception as e:
        logger.error(f"NetCDF分析异常: {e}")
        return {
            "success": False,
            "message": f"分析过程发生异常: {str(e)}",
            "error_code": "ANALYSIS_EXCEPTION",
            "traceback": traceback.format_exc()
        }


def create_custom_pollution_scenario(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建自定义污染场景
    
    Args:
        input_data: 输入参数字典
        
    Returns:
        场景创建结果
    """
    try:
        logger.info("创建自定义污染场景")

        parameters = input_data.get('parameters', {})
        scenario_config = parameters.get('scenario_config', {})

        # 场景类型
        scenario_type = scenario_config.get('type', 'oil_spill')

        # 根据场景类型配置污染源
        if scenario_type == 'oil_spill':
            # 油污泄漏场景
            pollution_sources = [
                {
                    "location": scenario_config.get('spill_location', [23.5, 120.5]),
                    "intensity": scenario_config.get('spill_volume', 5000.0),
                    "name": "海上溢油事故"
                }
            ]
            simulation_hours = scenario_config.get('duration_hours', 72.0)

        elif scenario_type == 'industrial_discharge':
            # 工业排放场景
            discharge_locations = scenario_config.get('discharge_points', [[23.0, 120.0]])
            pollution_sources = []
            for i, location in enumerate(discharge_locations):
                pollution_sources.append({
                    "location": location,
                    "intensity": scenario_config.get('discharge_rate', 2000.0),
                    "name": f"工业排放点{i+1}"
                })
            simulation_hours = scenario_config.get('duration_hours', 48.0)

        elif scenario_type == 'multiple_sources':
            # 多源污染场景
            pollution_sources = scenario_config.get('sources', [
                {"location": [23.0, 120.0], "intensity": 1000.0, "name": "污染源1"},
                {"location": [24.0, 121.0], "intensity": 800.0, "name": "污染源2"},
                {"location": [22.5, 119.5], "intensity": 600.0, "name": "污染源3"}
            ])
            simulation_hours = scenario_config.get('duration_hours', 24.0)

        else:
            return {
                "success": False,
                "message": f"不支持的场景类型: {scenario_type}",
                "error_code": "UNSUPPORTED_SCENARIO_TYPE"
            }

        # 其他参数
        output_path = scenario_config.get('output_path', f'{scenario_type}_scenario.gif')
        netcdf_path = parameters.get('netcdf_path')

        if not netcdf_path or not os.path.exists(netcdf_path):
            return {
                "success": False,
                "message": "需要提供有效的NetCDF文件路径",
                "error_code": "MISSING_NETCDF_PATH"
            }

        # 执行模拟
        result = create_adaptive_pollution_animation(
            netcdf_path=netcdf_path,
            pollution_sources=pollution_sources,
            simulation_hours=simulation_hours,
            time_step_minutes=scenario_config.get('time_step_minutes', 12.0),
            output_path=output_path,
            title=f"{scenario_type.replace('_', ' ').title()} - 污染扩散模拟",
            grid_resolution=scenario_config.get('grid_resolution', 0.01),
            show_velocity=scenario_config.get('show_velocity', True),
            colormap=scenario_config.get('colormap', 'custom_pollution')
        )

        if result["success"]:
            return {
                "success": True,
                "message": f"{scenario_type}场景模拟完成",
                "scenario_type": scenario_type,
                "output_path": output_path,
                "pollution_sources": pollution_sources,
                "simulation_result": result
            }
        else:
            return {
                "success": False,
                "message": f"场景模拟失败: {result.get('message', '未知错误')}",
                "error_code": "SCENARIO_SIMULATION_FAILED"
            }

    except Exception as e:
        logger.error(f"自定义场景创建异常: {e}")
        return {
            "success": False,
            "message": f"场景创建过程发生异常: {str(e)}",
            "error_code": "SCENARIO_CREATION_EXCEPTION",
            "traceback": traceback.format_exc()
        }


def get_pollution_dispersion_capabilities() -> Dict[str, Any]:
    """
    获取污染物扩散模拟能力信息
    
    Returns:
        能力信息字典
    """
    return {
        "success": True,
        "capabilities": {
            "supported_actions": [
                "run_pollution_dispersion",
                "analyze_netcdf_for_pollution",
                "create_custom_pollution_scenario",
                "get_capabilities"
            ],
            "supported_pollution_types": [
                "oil_spill",
                "chemical_discharge",
                "industrial_waste",
                "generic_pollutant"
            ],
            "supported_output_formats": [
                "gif",
                "mp4",
                "png_sequence"
            ],
            "scenario_types": [
                "oil_spill",
                "industrial_discharge",
                "multiple_sources",
                "custom"
            ],
            "features": [
                "自适应网格分辨率",
                "海流驱动扩散",
                "多污染源支持",
                "实时动画生成",
                "统计信息输出",
                "地理坐标系支持"
            ],
            "parameter_ranges": {
                "simulation_hours": {"min": 1.0, "max": 168.0, "default": 24.0},
                "time_step_minutes": {"min": 1.0, "max": 60.0, "default": 10.0},
                "grid_resolution": {"min": 0.001, "max": 0.1, "default": 0.01},
                "pollution_intensity": {"min": 100.0, "max": 100000.0, "default": 1000.0}
            }
        },
        "version": "1.0.0",
        "last_updated": datetime.now().isoformat()
    }


def main():
    """主函数 - 处理C#传入的JSON参数"""
    try:
        # 检查命令行参数
        if len(sys.argv) != 3:
            logger.error("用法: python pollution_dispersion_wrapper.py input.json output.json")
            sys.exit(1)

        input_file = sys.argv[1]
        output_file = sys.argv[2]

        # 检查输入文件
        if not os.path.exists(input_file):
            error_result = {
                "success": False,
                "message": f"输入文件不存在: {input_file}",
                "error_code": "INPUT_FILE_NOT_FOUND"
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, ensure_ascii=False, indent=2)
            sys.exit(1)

        # 读取输入参数
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        logger.info(f"处理请求: {input_data.get('action', 'unknown')}")

        # 根据action分发处理
        action = input_data.get('action', '')

        if action == 'run_pollution_dispersion':
            result = run_pollution_dispersion(input_data)
        elif action == 'analyze_netcdf_for_pollution':
            result = analyze_netcdf_for_pollution(input_data)
        elif action == 'create_custom_pollution_scenario':
            result = create_custom_pollution_scenario(input_data)
        elif action == 'get_capabilities':
            result = get_pollution_dispersion_capabilities()
        else:
            result = {
                "success": False,
                "message": f"不支持的操作: {action}",
                "error_code": "UNSUPPORTED_ACTION",
                "supported_actions": [
                    "run_pollution_dispersion",
                    "analyze_netcdf_for_pollution",
                    "create_custom_pollution_scenario",
                    "get_capabilities"
                ]
            }

        # 写入输出结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"处理完成, 结果已写入: {output_file}")

        # 根据处理结果设置退出码
        sys.exit(0 if result.get("success", False) else 1)

    except Exception as e:
        logger.error(f"主函数异常: {e}")

        # 确保有错误输出
        error_result = {
            "success": False,
            "message": f"包装器执行异常: {str(e)}",
            "error_code": "WRAPPER_EXCEPTION",
            "traceback": traceback.format_exc()
        }

        try:
            if len(sys.argv) >= 3:
                with open(sys.argv[2], 'w', encoding='utf-8') as f:
                    json.dump(error_result, f, ensure_ascii=False, indent=2)
        except:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()