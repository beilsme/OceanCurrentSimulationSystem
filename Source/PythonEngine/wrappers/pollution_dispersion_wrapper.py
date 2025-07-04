#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: pollution_dispersion_wrapper.py
位置: Source/PythonEngine/wrappers/pollution_dispersion_wrapper.py
功能: C#调用污染物扩散模拟的包装器脚本 - 自包含版本
用法: python pollution_dispersion_wrapper.py input.json output.json
"""

import sys
import json
import numpy as np
from pathlib import Path
import traceback
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import math

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 使用与ocean_data_wrapper.py相同的路径设置方法
this_dir = Path(__file__).parent
python_engine_root = this_dir.parent.parent  # points to Source directory

if str(python_engine_root) not in sys.path:
    sys.path.insert(0, str(python_engine_root))

# 添加多个可能的Python引擎路径
current_dir = Path(__file__).parent
python_engine_paths = [
    current_dir.parent.parent,  # Source directory
    current_dir.parent,         # PythonEngine directory
    current_dir                 # wrappers directory
]

for path in python_engine_paths:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# 导入必要的模块
try:
    from PythonEngine.core.data_processor import DataProcessor
    from PythonEngine.core.netcdf_handler import NetCDFHandler
    logger.info("基础数据处理模块导入成功")
except ImportError as e:
    logger.error(f"基础模块导入失败: {e}")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator
    logger.info("可视化和科学计算模块导入成功")
except ImportError as e:
    logger.error(f"可视化模块导入失败: {e}")
    sys.exit(1)
import matplotlib.pyplot as plt



def nan_to_none(obj):
    """将NaN值转换为None，确保JSON序列化正常"""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(x) for x in obj]
    return obj


def setup_chinese_font():
    """设置中文字体显示"""
    try:
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti',  # Mac字体
                                           'SimHei', 'Microsoft YaHei',  # Windows字体
                                           'WenQuanYi Micro Hei', 'Noto Sans CJK SC',  # Linux字体
                                           'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False


class IntegratedPollutionSimulator:
    """集成的污染扩散模拟器 - 自包含实现"""

    def __init__(self, netcdf_path: str, grid_resolution: float = 0.01):
        self.netcdf_path = netcdf_path
        self.grid_resolution = grid_resolution

        # 数据范围（将从NetCDF文件自动读取）
        self.lon_range = None
        self.lat_range = None

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

        # 物理参数
        self.diffusion_coeff = 500.0
        self.decay_rate = 0.00005

        self.is_initialized = False

    def analyze_netcdf_data(self) -> Dict[str, Any]:
        """分析NetCDF数据，获取地理范围和基本信息"""
        try:
            handler = NetCDFHandler(self.netcdf_path)
            try:
                # 获取坐标信息
                u, v, lat, lon = handler.get_uv(time_idx=0, depth_idx=0)

                # 获取地理范围
                self.lat_range = (float(lat.min()), float(lat.max()))
                self.lon_range = (float(lon.min()), float(lon.max()))

                # 计算地理范围（公里）
                lat_span_km = (self.lat_range[1] - self.lat_range[0]) * 111.32
                lon_span_km = (self.lon_range[1] - self.lon_range[0]) * 111.32 * np.cos(np.radians(np.mean(self.lat_range)))

                return {
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
                        "lat_points": len(lat),
                        "lon_points": len(lon),
                        "data_shape": u.shape
                    }
                }
            finally:
                handler.close()

        except Exception as e:
            logger.error(f"NetCDF数据分析失败: {e}")
            return {
                "success": False,
                "message": f"数据分析失败: {str(e)}"
            }

    def initialize(self, time_index: int = 0, depth_index: int = 0) -> bool:
        """初始化环境，自动适配NetCDF数据范围"""
        try:
            logger.info("初始化污染扩散环境")

            # 先分析数据
            analysis = self.analyze_netcdf_data()
            if not analysis["success"]:
                return False

            # 加载NetCDF数据
            handler = NetCDFHandler(self.netcdf_path)
            try:
                # 获取原始数据
                self.water_u, self.water_v, self.lat, self.lon = handler.get_uv(
                    time_idx=time_index, depth_idx=depth_index
                )

                # 处理掩码数组
                if isinstance(self.water_u, np.ma.MaskedArray):
                    self.water_u = self.water_u.filled(0)
                if isinstance(self.water_v, np.ma.MaskedArray):
                    self.water_v = self.water_v.filled(0)
            finally:
                handler.close()

            # 创建模拟网格
            self._create_simulation_grid()

            # 插值速度场到模拟网格
            self._interpolate_velocity_field()

            # 初始化浓度场
            self._initialize_concentration_field()

            self.is_initialized = True
            logger.info(f"环境初始化成功，模拟网格: {len(self.sim_lat)} x {len(self.sim_lon)}")

            return True

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False

    def _create_simulation_grid(self):
        """创建模拟网格"""
        self.sim_lon = np.arange(self.lon_range[0], self.lon_range[1] + self.grid_resolution, self.grid_resolution)
        self.sim_lat = np.arange(self.lat_range[0], self.lat_range[1] + self.grid_resolution, self.grid_resolution)
        logger.info(f"创建模拟网格: {len(self.sim_lat)} x {len(self.sim_lon)} (分辨率: {self.grid_resolution}°)")

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

        logger.info("海流速度场插值完成")

    def _initialize_concentration_field(self):
        """初始化浓度场"""
        self.concentration = np.zeros((len(self.sim_lat), len(self.sim_lon)))
        self.concentration_history = []

    def add_pollution_source(self, location, intensity, radius=None):
        """添加污染源"""
        try:
            lat_src, lon_src = location

            if not (self.lat_range[0] <= lat_src <= self.lat_range[1] and
                    self.lon_range[0] <= lon_src <= self.lon_range[1]):
                logger.warning(f"污染源位置超出范围")
                return False

            # 设置初始扩散参数
            if radius is None:
                lat_span = self.lat_range[1] - self.lat_range[0]
                lon_span = self.lon_range[1] - self.lon_range[0]
                radius = min(lat_span, lon_span) * 0.03

            # 转换为网格索引
            lat_idx = np.argmin(np.abs(self.sim_lat - lat_src))
            lon_idx = np.argmin(np.abs(self.sim_lon - lon_src))

            # 创建污染源
            radius_grid = max(2, int(radius / self.grid_resolution))

            y_indices, x_indices = np.ogrid[-lat_idx:len(self.sim_lat)-lat_idx,
                                   -lon_idx:len(self.sim_lon)-lon_idx]

            # 椭圆形初始扩散
            if hasattr(self, 'sim_u') and hasattr(self, 'sim_v'):
                local_u = self.sim_u[lat_idx, lon_idx] if self.sim_u is not None else 0
                local_v = self.sim_v[lat_idx, lon_idx] if self.sim_v is not None else 0

                flow_angle = np.arctan2(local_v, local_u)
                a = radius_grid * 1.5
                b = radius_grid * 0.8

                cos_angle = np.cos(flow_angle)
                sin_angle = np.sin(flow_angle)

                x_rot = x_indices * cos_angle + y_indices * sin_angle
                y_rot = -x_indices * sin_angle + y_indices * cos_angle

                ellipse_mask = (x_rot/a)**2 + (y_rot/b)**2
            else:
                ellipse_mask = (x_indices**2 + y_indices**2) / (radius_grid**2)

            # 创建浓度分布
            sigma_factor = 2.0
            core_concentration = np.exp(-ellipse_mask / (sigma_factor**2))
            outer_mask = ellipse_mask * 2.0
            outer_concentration = 0.3 * np.exp(-outer_mask / (sigma_factor**2))

            total_concentration = np.maximum(core_concentration, outer_concentration)

            # 添加随机扰动
            if total_concentration.shape[0] > 0 and total_concentration.shape[1] > 0:
                noise_scale = 0.1
                noise = noise_scale * np.random.randn(*total_concentration.shape)
                total_concentration += noise
                total_concentration = np.maximum(0, total_concentration)

            # 应用强度
            pollution_field = total_concentration * intensity / np.max(total_concentration) if np.max(total_concentration) > 0 else total_concentration * intensity

            # 添加到浓度场
            self.concentration += pollution_field

            logger.info(f"添加污染源成功，位置: ({lat_src:.3f}°N, {lon_src:.3f}°E)")
            return True

        except Exception as e:
            logger.error(f"添加污染源失败: {e}")
            return False

    def simulate_diffusion_step(self, dt=600.0):
        """执行扩散步骤"""
        c_old = self.concentration.copy()

        # 空间步长（米）
        dx = self.grid_resolution * 111320
        dy = self.grid_resolution * 111320

        # 检查CFL条件并自动调整时间步长
        max_u = np.max(np.abs(self.sim_u))
        max_v = np.max(np.abs(self.sim_v))
        max_velocity = max(max_u, max_v, 1e-10)

        cfl_limit = 0.3 * min(dx, dy) / max_velocity

        if dt > cfl_limit:
            n_substeps = int(np.ceil(dt / cfl_limit))
            sub_dt = dt / n_substeps

            for _ in range(n_substeps):
                self._single_diffusion_step(sub_dt)
        else:
            self._single_diffusion_step(dt)

    def _single_diffusion_step(self, dt):
        """执行单个扩散时间步"""
        c_old = self.concentration.copy()

        # 空间步长（米）
        dx = self.grid_resolution * 111320
        dy = self.grid_resolution * 111320

        # 计算梯度
        dc_dy, dc_dx = np.gradient(c_old, dy, dx)

        # 对流项（海流输运）
        advection_x = -self.sim_u * dc_dx
        advection_y = -self.sim_v * dc_dy

        # 扩散项
        d2c_dx2 = np.gradient(np.gradient(c_old, dx, axis=1), dx, axis=1)
        d2c_dy2 = np.gradient(np.gradient(c_old, dy, axis=0), dy, axis=0)

        diffusion = self.diffusion_coeff * (d2c_dx2 + d2c_dy2)

        # 衰减项
        decay = -self.decay_rate * c_old

        # 时间积分
        dc_dt = advection_x + advection_y + diffusion + decay
        self.concentration = c_old + dt * dc_dt

        # 确保浓度非负
        self.concentration = np.maximum(0, self.concentration)

        # 边界条件
        boundary_width = 2
        for i in range(boundary_width):
            factor = (i + 1) / boundary_width * 0.9
            self.concentration[i, :] *= factor
            self.concentration[-1-i, :] *= factor
            self.concentration[:, i] *= factor
            self.concentration[:, -1-i] *= factor

        # 平滑
        self.concentration = gaussian_filter(self.concentration, sigma=0.2)

    def create_static_visualization(self, pollution_sources: List[Dict], output_path: str, title: str = "海洋污染扩散模拟"):
        """创建静态可视化图像"""
        try:
            # 设置中文字体
            setup_chinese_font()

            # 执行短期模拟以展示扩散效果
            logger.info("执行短期扩散模拟...")
            for step in range(20):  # 模拟20个时间步
                self.simulate_diffusion_step(dt=300.0)

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

            # 创建网格坐标
            LON, LAT = np.meshgrid(self.sim_lon, self.sim_lat)

            # 绘制污染浓度场
            max_concentration = np.max(self.concentration)
            if max_concentration > 0:
                threshold = max_concentration * 0.001
                concentration_masked = np.where(self.concentration >= threshold, self.concentration, np.nan)

                # 创建自定义颜色映射
                colors = ['white', 'lightblue', 'yellow', 'orange', 'red', 'darkred', 'maroon']
                cmap = LinearSegmentedColormap.from_list('pollution', colors, N=256)

                norm = Normalize(vmin=threshold, vmax=max_concentration)

                im = ax.pcolormesh(LON, LAT, concentration_masked,
                                   cmap=cmap, norm=norm, alpha=0.85,
                                   transform=ccrs.PlateCarree(), zorder=3,
                                   shading='auto')

                # 颜色条
                cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=30)
                cbar.set_label('污染物浓度 (kg/m³)', fontsize=14, fontweight='bold')
                cbar.ax.tick_params(labelsize=12)

            # 绘制海流矢量场
            skip = max(1, len(self.sim_lon) // 25)
            ax.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
                      self.sim_u[::skip, ::skip], self.sim_v[::skip, ::skip],
                      scale=50, alpha=0.6, color='gray', width=0.002,
                      transform=ccrs.PlateCarree(), zorder=4)

            # 标记污染源
            for i, source in enumerate(pollution_sources):
                lat_src, lon_src = source['location']
                ax.plot(lon_src, lat_src, marker='o', markersize=15,
                        markerfacecolor='red', markeredgecolor='black',
                        markeredgewidth=2, transform=ccrs.PlateCarree(),
                        zorder=15, label=f'污染源 {i+1}' if i < 5 else "")

            # 网格线
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.8, color='gray', alpha=0.6, linestyle='--')
            gl.xlabel_style = {'size': 12, 'color': 'black'}
            gl.ylabel_style = {'size': 12, 'color': 'black'}
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()

            # 统计信息文本框
            total_mass = np.sum(self.concentration) * (111320 * self.grid_resolution)**2
            max_conc = np.max(self.concentration)
            affected_area = np.sum(self.concentration > max_conc * 0.001) * (self.grid_resolution * 111.32)**2

            center_lat = np.mean(self.lat_range)
            center_lon = np.mean(self.lon_range)

            info_str = f'🔴 最高浓度: {max_conc:.2e} kg/m³\n'
            info_str += f'⚖️ 总质量: {total_mass:.1e} kg\n'
            info_str += f'📍 影响面积: {affected_area:.1f} km²\n'
            info_str += f'🌊 海域中心: ({center_lat:.2f}°N, {center_lon:.2f}°E)\n'
            info_str += f'📊 污染源数量: {len(pollution_sources)}'

            ax.text(0.02, 0.98, info_str, transform=ax.transAxes,
                    verticalalignment='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.8',
                              facecolor='white', alpha=0.95, edgecolor='black'),
                    zorder=20)

            # 标题
            ax.set_title(title, fontsize=18, fontweight='bold', pad=25)

            # 图例
            if pollution_sources:
                ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.85),
                          fontsize=12, framealpha=0.9)

            # 保存图像
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"静态可视化保存成功: {output_path}")
            return True

        except Exception as e:
            logger.error(f"创建静态可视化失败: {e}")
            return False


def run_pollution_dispersion(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """执行污染物扩散模拟"""
    try:
        logger.info("开始执行污染物扩散模拟")

        # 获取参数
        parameters = input_data.get('parameters', {})
        netcdf_path = parameters.get('netcdf_path')
        output_path = parameters.get('output_path', 'pollution_dispersion.png')

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

        logger.info(f"使用NetCDF文件: {netcdf_path}")
        logger.info(f"输出路径: {output_path}")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 如果输出路径是gif，改为png
        if output_path.endswith('.gif'):
            output_path = output_path.replace('.gif', '.png')

        # 提取模拟参数
        pollution_sources = parameters.get('pollution_sources', [])
        if not pollution_sources:
            # 自动分析NetCDF文件确定污染源位置
            temp_simulator = IntegratedPollutionSimulator(netcdf_path)
            analysis = temp_simulator.analyze_netcdf_data()

            if analysis["success"]:
                geo_info = analysis["geographic_info"]
                center_lat = geo_info["center_lat"]
                center_lon = geo_info["center_lon"]

                pollution_sources = [
                    {
                        "location": [center_lat, center_lon],
                        "intensity": 1000.0,
                        "name": "自动检测的中心污染源"
                    }
                ]
                logger.info(f"自动设置污染源位置: ({center_lat:.3f}°N, {center_lon:.3f}°E)")
            else:
                pollution_sources = [
                    {
                        "location": [23.0, 120.0],
                        "intensity": 1000.0,
                        "name": "默认污染源"
                    }
                ]
                logger.warning("无法分析NetCDF文件，使用默认污染源位置")

        grid_resolution = float(parameters.get('grid_resolution', 0.015))
        title = parameters.get('title', '海洋污染扩散模拟')

        logger.info(f"网格分辨率: {grid_resolution}°")
        logger.info(f"污染源数量: {len(pollution_sources)}")

        # 创建集成污染扩散模拟器
        simulator = IntegratedPollutionSimulator(netcdf_path, grid_resolution)

        # 初始化
        if not simulator.initialize():
            return {
                "success": False,
                "message": "污染扩散模拟器初始化失败",
                "error_code": "SIMULATOR_INIT_FAILED"
            }

        # 添加污染源
        for source in pollution_sources:
            simulator.add_pollution_source(
                location=tuple(source['location']),
                intensity=source['intensity']
            )

        # 创建可视化
        success = simulator.create_static_visualization(pollution_sources, output_path, title)

        if success and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024 / 1024  # MB

            max_conc = np.max(simulator.concentration)
            mean_conc = np.mean(simulator.concentration)
            total_mass = np.sum(simulator.concentration) * (111320 * grid_resolution)**2

            statistics = {
                "max_concentration": float(max_conc) if max_conc is not None and not np.isnan(max_conc) else 0.0,
                "mean_concentration": float(mean_conc) if mean_conc is not None and not np.isnan(mean_conc) else 0.0,
                "total_mass": float(total_mass) if total_mass is not None and not np.isnan(total_mass) else 0.0,
                "simulation_frames": 1,
                "simulation_hours": 1.0,
                "file_size_mb": file_size,
                "pollution_sources_count": len(pollution_sources)
            }

            return {
                "success": True,
                "message": "污染物扩散模拟成功完成",
                "output_path": output_path,
                "statistics": statistics,
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "simulation_type": "集成静态模拟",
                    "netcdf_source": netcdf_path,
                    "grid_resolution": grid_resolution
                }
            }
        else:
            return {
                "success": False,
                "message": "无法生成输出文件",
                "error_code": "OUTPUT_GENERATION_FAILED"
            }

    except Exception as e:
        logger.error(f"污染物扩散模拟异常: {e}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "message": f"模拟过程发生异常: {str(e)}",
            "error_code": "SIMULATION_EXCEPTION",
            "traceback": traceback.format_exc()
        }


def analyze_netcdf_for_pollution(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """分析NetCDF文件，为污染扩散模拟提供信息"""
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
        simulator = IntegratedPollutionSimulator(netcdf_path)
        analysis_result = simulator.analyze_netcdf_data()

        if analysis_result["success"]:
            geo_info = analysis_result["geographic_info"]
            data_info = analysis_result["data_info"]

            # 推荐污染源位置
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
                    "optimal_grid_resolution": min(lat_span, lon_span) / 100,
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


def get_pollution_dispersion_capabilities() -> Dict[str, Any]:
    """获取污染物扩散模拟能力信息"""
    return {
        "success": True,
        "capabilities": {
            "supported_actions": [
                "run_pollution_dispersion",
                "analyze_netcdf_for_pollution",
                "get_capabilities"
            ],
            "simulation_modes": [
                "集成静态模拟 - 使用内置污染扩散算法"
            ],
            "supported_output_formats": [
                "png", "jpg"
            ],
            "features": [
                "海流驱动扩散",
                "多污染源支持",
                "地理坐标系支持",
                "自适应网格分辨率",
                "统计信息输出",
                "自动污染源定位"
            ],
            "parameter_ranges": {
                "grid_resolution": {"min": 0.005, "max": 0.05, "default": 0.015},
                "pollution_intensity": {"min": 100.0, "max": 10000.0, "default": 1000.0}
            }
        },
        "version": "1.2.0 - 集成版",
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

        logger.info(f"读取输入文件: {input_file}")
        logger.info(f"输出文件: {output_file}")

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
                    "get_capabilities"
                ]
            }

        # 处理NaN值并写入输出结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(nan_to_none(result), f, ensure_ascii=False, indent=2)

        logger.info(f"处理完成, 结果已写入: {output_file}")

        # 根据处理结果设置退出码
        sys.exit(0 if result.get("success", False) else 1)

    except Exception as e:
        logger.error(f"主函数异常: {e}")
        logger.error(traceback.format_exc())

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
    if len(sys.argv) == 3:
        # 当从C#调用时，使用命令行参数模式
        main()
    else:
        import json
        import os