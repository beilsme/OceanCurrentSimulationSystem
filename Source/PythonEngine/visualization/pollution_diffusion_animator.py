# ==============================================================================
# visualization/adaptive_pollution_animator.py
# ==============================================================================
"""
自适应污染扩散动画生成器
自动读取NetCDF文件的地理范围，实现：
- 自动适配数据的地理边界
- 地图打底显示
- 污染物从点逐渐扩散成面
- 根据浓度不同显示不同颜色
- 流畅的动画效果
"""

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

# 导入中文设置函数
def setup_chinese_font():
    """设置中文字体显示"""
    import matplotlib.font_manager as fm
    import platform

    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 根据操作系统选择合适的中文字体
    system = platform.system()
    if system == "Windows":
        # Windows系统
        font_names = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    elif system == "Darwin":  # macOS
        font_names = ['Heiti TC', 'PingFang SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        font_names = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Liberation Sans']

    # 尝试找到可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font_name in font_names:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            print(f"✅ 中文字体设置成功: {font_name}")
            return True

    print("⚠️  未找到合适的中文字体，使用默认字体")
    return False

# 在模块加载时设置中文字体
setup_chinese_font()


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
        else:
            cmap = plt.get_cmap(colormap)

        # 计算浓度范围 - 关键修复：设置更合理的范围
        max_concentration = np.max([np.max(c) for c in self.concentration_history])
        if max_concentration == 0:
            max_concentration = 1e-6

        # 使用对数范围以更好显示扩散
        vmin = max_concentration * 1e-4  # 最小值设为峰值的万分之一
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

        # 关键修复：使用LogNorm来更好显示扩散过程
        norm = LogNorm(vmin=vmin, vmax=vmax)

        # 初始化污染物显示 - 修复：确保初始数据不为零
        initial_data = self.concentration_history[0].copy()
        initial_data[initial_data <= 0] = vmin  # 避免对数归一化问题

        im = ax.pcolormesh(LON, LAT, initial_data,
                           cmap=cmap, norm=norm, alpha=0.85,
                           transform=ccrs.PlateCarree(), zorder=3,
                           shading='auto')  # 关键修复：使用auto shading

        # 颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=30)
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

            # 关键修复：处理零值和负值
            current_concentration = np.maximum(current_concentration, vmin)

            # 关键修复：使用ravel()并正确更新数组
            im.set_array(current_concentration.ravel())

            # 计算统计信息
            grid_area = (111320 * self.grid_resolution)**2  # 单个网格面积 m²
            total_mass = np.sum(current_concentration) * grid_area
            max_conc = np.max(current_concentration)

            # 计算污染影响面积
            threshold = max_conc * 0.01 if max_conc > 0 else 0
            affected_cells = np.sum(current_concentration > threshold)
            affected_area = affected_cells * grid_area / 1e6  # 转换为km²

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


if __name__ == "__main__":
    """主测试函数 - 演示自适应污染扩散动画生成器的使用"""

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("🌊 自适应海洋污染扩散动画生成器测试")
    print("=" * 80)

    # 测试数据文件路径（请替换为实际的NetCDF文件路径）
    netcdf_path = "../data/raw_data/merged_data.nc"

   

    try:
        print(f"\n📂 加载NetCDF数据: {netcdf_path}")

        # 1. 数据分析测试
        print("\n" + "─" * 60)
        print("🔍 步骤1: 分析NetCDF数据结构")
        print("─" * 60)

        animator = AdaptivePollutionAnimator(netcdf_path, grid_resolution=0.01)
        analysis = animator.analyze_netcdf_data()

        if analysis["success"]:
            geo_info = analysis["geographic_info"]
            data_info = analysis["data_info"]

            print(f"✅ 数据分析成功!")
            print(f"📍 地理范围: {geo_info['lat_range'][0]:.2f}°-{geo_info['lat_range'][1]:.2f}°N")
            print(f"📍 经度范围: {geo_info['lon_range'][0]:.2f}°-{geo_info['lon_range'][1]:.2f}°E")
            print(f"📏 空间跨度: {geo_info['lat_span_km']:.1f} x {geo_info['lon_span_km']:.1f} km")
            print(f"🗂️  数据网格: {data_info['lat_points']} x {data_info['lon_points']}")
            print(f"⏰ 时间步数: {data_info['time_steps']}")
            print(f"🌊 中心位置: ({geo_info['center_lat']:.2f}°N, {geo_info['center_lon']:.2f}°E)")
        else:
            print(f"❌ 数据分析失败: {analysis['message']}")
       

        # 2. 环境初始化测试
        print("\n" + "─" * 60)
        print("⚙️  步骤2: 初始化模拟环境")
        print("─" * 60)

        success = animator.initialize()
        if success:
            print(f"✅ 环境初始化成功!")
            print(f"🔢 模拟网格: {len(animator.sim_lat)} x {len(animator.sim_lon)}")
            print(f"📐 网格分辨率: {animator.grid_resolution}°")
            print(f"🌀 速度场形状: {animator.sim_u.shape}")
        else:
            print("❌ 环境初始化失败")


        # 3. 污染源设置测试
        print("\n" + "─" * 60)
        print("☢️  步骤3: 设置污染源")
        print("─" * 60)

        # 根据数据范围设置海域中的污染源位置
        center_lat = geo_info['center_lat']
        center_lon = geo_info['center_lon']
        lat_span = geo_info['lat_range'][1] - geo_info['lat_range'][0]
        lon_span = geo_info['lon_range'][1] - geo_info['lon_range'][0]

        # 选择海域位置作为污染源（避开陆地）
        pollution_sources = [
            {
                "location": [23.0, 120.0],  # 台湾海峡中部海域
                "intensity": 8000.0,
                "name": "海上溢油事故点"
            },
            {
                "location": [24.5, 119.5],  # 台湾海峡北部海域
                "intensity": 5000.0,
                "name": "船舶排污点"
            },
            {
                "location": [22.5, 121.5],  # 台湾东南海域
                "intensity": 3000.0,
                "name": "工业排放点"
            }
        ]

        for i, source in enumerate(pollution_sources):
            success = animator.add_pollution_source(
                location=tuple(source['location']),
                intensity=source['intensity']
            )
            if success:
                print(f"✅ {source['name']}: ({source['location'][0]:.3f}°N, {source['location'][1]:.3f}°E), 强度: {source['intensity']}")
            else:
                print(f"❌ 添加污染源失败: {source['name']}")

        print(f"🎯 总计设置 {len(pollution_sources)} 个污染源")

        # 4. 扩散模拟测试
        print("\n" + "─" * 60)
        print("🧪 步骤4: 扩散过程模拟测试")
        print("─" * 60)

        print("执行短期扩散模拟...")
        initial_max = np.max(animator.concentration)
        initial_sum = np.sum(animator.concentration)

        # 执行10个时间步
        for step in range(10):
            animator.simulate_diffusion_step(dt=300.0)  # 减少时间步长
            if step % 3 == 0:
                current_max = np.max(animator.concentration)
                current_sum = np.sum(animator.concentration)
                print(f"  步骤 {step+1}: 最高浓度 {current_max:.2e}, 总质量 {current_sum:.2e}")

        final_max = np.max(animator.concentration)
        final_sum = np.sum(animator.concentration)

        print(f"📊 模拟结果:")
        print(f"  初始最高浓度: {initial_max:.2e} → 最终: {final_max:.2e}")
        print(f"  总质量变化: {initial_sum:.2e} → {final_sum:.2e} (衰减: {(1-final_sum/initial_sum)*100:.1f}%)")

        # 5. 动画生成测试
        print("\n" + "─" * 60)
        print("🎬 步骤5: 生成污染扩散动画")
        print("─" * 60)

        # 重新初始化以获得干净的环境
        animator.initialize()

        # 测试多种动画配置
        test_configs = [
            {
                "name": "快速预览版",
                "hours": 6.0,
                "time_step": 10.0,  # 减少时间步长
                "filename": "quick_preview.gif",
                "show_velocity": False
            },
            {
                "name": "标准版本",
                "hours": 24.0,
                "time_step": 8.0,  # 减少时间步长
                "filename": "standard_simulation.gif",
                "show_velocity": True
            },
            {
                "name": "长期观察版",
                "hours": 48.0,
                "time_step": 15.0,  # 减少时间步长
                "filename": "long_term_simulation.gif",
                "show_velocity": True
            }
        ]

        for config in test_configs:
            print(f"\n🎯 生成{config['name']}...")
            print(f"   模拟时长: {config['hours']} 小时")
            print(f"   时间步长: {config['time_step']} 分钟")
            print(f"   显示流场: {'是' if config['show_velocity'] else '否'}")

            result = animator.create_pollution_animation(
                pollution_sources=pollution_sources,
                simulation_hours=config['hours'],
                time_step_minutes=config['time_step'],
                output_path=config['filename'],
                title=f"海洋污染扩散模拟 - {config['name']}",
                show_velocity=config['show_velocity'],
                colormap="custom_pollution",
                fps=15
            )

            if result["success"]:
                stats = result["animation_stats"]
                print(f"   ✅ 动画生成成功: {config['filename']}")
                print(f"   📹 动画帧数: {stats['frames']}")
                print(f"   📊 最高浓度: {stats['max_concentration']:.2e}")
                print(f"   📂 文件大小: {os.path.getsize(config['filename'])/1024/1024:.1f} MB")
            else:
                print(f"   ❌ 动画生成失败: {result['message']}")

            # 重新初始化环境以确保每个动画都是独立的
            animator.initialize()

        # 6. 便捷函数测试
        print("\n" + "─" * 60)
        print("🛠️  步骤6: 测试便捷函数")
        print("─" * 60)

        print("使用便捷函数生成动画...")

        convenience_result = create_adaptive_pollution_animation(
            netcdf_path=netcdf_path,
            pollution_sources=[
                {"location": [23.2, 120.2], "intensity": 6000.0},  # 海域位置
                {"location": [24.0, 119.8], "intensity": 4000.0}   # 海域位置
            ],
            simulation_hours=12.0,
            time_step_minutes=12.0,  # 减少时间步长
            output_path="convenience_function_test.gif",
            title="便捷函数测试 - 海洋污染扩散",
            grid_resolution=0.015,
            show_velocity=True,
            colormap="custom_pollution"
        )

        if convenience_result["success"]:
            print("✅ 便捷函数测试成功!")
            print(f"📁 输出文件: convenience_function_test.gif")
        else:
            print(f"❌ 便捷函数测试失败: {convenience_result['message']}")

        # 7. 结果总结
        print("\n" + "=" * 80)
        print("📋 测试结果总结")
        print("=" * 80)

        generated_files = []
        for config in test_configs:
            if os.path.exists(config['filename']):
                generated_files.append(config['filename'])

        if os.path.exists("convenience_function_test.gif"):
            generated_files.append("convenience_function_test.gif")

        print(f"✅ 成功生成 {len(generated_files)} 个动画文件:")
        for file in generated_files:
            size_mb = os.path.getsize(file) / 1024 / 1024
            print(f"   📹 {file} ({size_mb:.1f} MB)")

        print(f"\n🎯 测试完成! 所有功能正常运行。")
        print(f"📍 数据覆盖区域: {geo_info['lat_range'][0]:.2f}°-{geo_info['lat_range'][1]:.2f}°N, {geo_info['lon_range'][0]:.2f}°-{geo_info['lon_range'][1]:.2f}°E")
        print(f"💾 输出文件保存在当前目录")

    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        logging.error(f"测试失败: {e}", exc_info=True)

    finally:
        print(f"\n🧹 清理临时文件...")
        # 可选择是否删除演示数据
        # if os.path.exists("demo_ocean_data.nc"):
        #     os.remove("demo_ocean_data.nc")
        #     print("演示数据已清理")


def create_demo_netcdf_data(output_path: str):
    """创建演示用的NetCDF海洋数据"""
    print("正在创建演示用海洋流场数据...")

    # 创建台湾海峡区域的演示数据
    lat_range = np.linspace(23.0, 25.5, 60)
    lon_range = np.linspace(119.5, 122.0, 80)
    time_range = np.arange(0, 72, 3)  # 72小时，每3小时一个数据点
    depth_range = np.array([0, 5, 10, 20])

    with Dataset(output_path, 'w', format='NETCDF4') as nc:
        # 创建维度
        nc.createDimension('lat', len(lat_range))
        nc.createDimension('lon', len(lon_range))
        nc.createDimension('time', len(time_range))
        nc.createDimension('depth', len(depth_range))

        # 创建坐标变量
        lat_var = nc.createVariable('lat', 'f4', ('lat',))
        lon_var = nc.createVariable('lon', 'f4', ('lon',))
        time_var = nc.createVariable('time', 'f4', ('time',))
        depth_var = nc.createVariable('depth', 'f4', ('depth',))

        # 创建数据变量
        u_var = nc.createVariable('water_u', 'f4', ('time', 'depth', 'lat', 'lon'))
        v_var = nc.createVariable('water_v', 'f4', ('time', 'depth', 'lat', 'lon'))

        # 填充坐标
        lat_var[:] = lat_range
        lon_var[:] = lon_range
        time_var[:] = time_range
        depth_var[:] = depth_range

        # 创建复杂的海流场
        LAT, LON = np.meshgrid(lat_range, lon_range, indexing='ij')

        for t, time_val in enumerate(time_range):
            for d, depth_val in enumerate(depth_range):
                # 基础环流
                center_lat, center_lon = 24.2, 120.7
                dx = (LON - center_lon) * 111320
                dy = (LAT - center_lat) * 111320
                r = np.sqrt(dx**2 + dy**2)

                # 旋转流场（台湾海峡环流）
                omega = 2e-5 * np.exp(-r/80000) * (1 + 0.3*np.sin(2*np.pi*time_val/24))
                u_rotation = -omega * dy
                v_rotation = omega * dx

                # 潮汐流
                tidal_phase_m2 = 2 * np.pi * time_val / 12.42  # M2 潮汐
                tidal_phase_s2 = 2 * np.pi * time_val / 12.00  # S2 潮汐

                u_tidal = (0.4 * np.sin(tidal_phase_m2) + 0.2 * np.sin(tidal_phase_s2)) * \
                          np.cos(LAT * np.pi / 180) * np.exp(-depth_val/10)
                v_tidal = (0.3 * np.cos(tidal_phase_m2) + 0.15 * np.cos(tidal_phase_s2)) * \
                          np.sin(LON * np.pi / 180) * np.exp(-depth_val/10)

                # 季风影响
                monsoon_strength = 0.3 * np.sin(2 * np.pi * time_val / (24*30))  # 月周期
                u_monsoon = monsoon_strength * np.exp(-depth_val/15)
                v_monsoon = monsoon_strength * 0.7 * np.exp(-depth_val/15)

                # 合成流场
                u_total = u_rotation + u_tidal + u_monsoon
                v_total = v_rotation + v_tidal + v_monsoon

                # 添加噪声
                noise_scale = 0.05 * np.exp(-depth_val/5)
                u_noise = noise_scale * np.random.randn(*u_total.shape)
                v_noise = noise_scale * np.random.randn(*v_total.shape)

                u_var[t, d, :, :] = u_total + u_noise
                v_var[t, d, :, :] = v_total + v_noise

        # 添加全局属性
        nc.title = "Demo Ocean Current Data for Taiwan Strait"
        nc.description = "Synthetic ocean current data for pollution diffusion simulation testing"
        nc.source = "Generated by AdaptivePollutionAnimator demo"
        nc.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"✅ 演示数据创建完成: {output_path}")
    print(f"   覆盖区域: 台湾海峡 ({lat_range[0]:.1f}°-{lat_range[-1]:.1f}°N)")
    print(f"   时间跨度: {len(time_range)} 个时间点 ({time_range[-1]} 小时)")
    print(f"   深度层次: {len(depth_range)} 层 (0-{depth_range[-1]} 米)")