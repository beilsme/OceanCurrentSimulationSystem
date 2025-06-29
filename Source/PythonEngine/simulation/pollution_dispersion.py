"""
污染物扩散模拟器
基于粒子追踪和扩散方程的污染物传播模拟
支持多种污染物类型和复杂的海洋环境
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

# 导入相关模块
from .particle_tracking_wrapper import ParticleTrackingWrapper, ParticleState, TrackingParameters
from .current_simulation_wrapper import CurrentSimulationWrapper

# 设置日志
logger = logging.getLogger(__name__)

class PollutantType(Enum):
    """污染物类型"""
    OIL = "oil"
    CHEMICAL = "chemical"
    PLASTIC = "plastic"
    SEDIMENT = "sediment"
    THERMAL = "thermal"
    BIOLOGICAL = "biological"

class DischargeType(Enum):
    """排放类型"""
    INSTANTANEOUS = "instantaneous"  # 瞬时排放
    CONTINUOUS = "continuous"        # 连续排放
    VARIABLE = "variable"           # 变化排放

class WeatheringProcess(Enum):
    """风化过程"""
    EVAPORATION = "evaporation"
    DISSOLUTION = "dissolution"
    BIODEGRADATION = "biodegradation"
    SEDIMENTATION = "sedimentation"
    PHOTO_OXIDATION = "photo_oxidation"

@dataclass
class PollutantProperties:
    """污染物属性"""
    pollutant_type: PollutantType
    density: float  # 密度 (kg/m³)
    viscosity: float  # 粘度 (Pa·s)
    solubility: float  # 溶解度 (kg/m³)
    vapor_pressure: float  # 蒸汽压 (Pa)
    molecular_weight: float  # 分子量 (g/mol)
    boiling_point: float  # 沸点 (K)
    degradation_rate: float = 0.0  # 降解速率 (1/day)
    volatilization_rate: float = 0.0  # 挥发速率 (1/day)
    settling_velocity: float = 0.0  # 沉降速度 (m/s)
    sorption_coefficient: float = 0.0  # 吸附系数

@dataclass
class DischargeSource:
    """排放源"""
    x: float  # x坐标
    y: float  # y坐标
    z: float  # z坐标
    discharge_rate: float  # 排放速率 (kg/s)
    discharge_type: DischargeType
    start_time: float = 0.0  # 开始时间 (s)
    end_time: float = float('inf')  # 结束时间 (s)
    temperature: float = 298.15  # 温度 (K)
    properties: Optional[PollutantProperties] = None
    time_series: Optional[List[Tuple[float, float]]] = None  # 时间-排放速率序列

@dataclass
class EnvironmentalConditions:
    """环境条件"""
    water_temperature: Union[float, np.ndarray] = 298.15  # 水温 (K)
    salinity: Union[float, np.ndarray] = 35.0  # 盐度 (psu)
    ph: Union[float, np.ndarray] = 8.1  # pH值
    dissolved_oxygen: Union[float, np.ndarray] = 8.0  # 溶解氧 (mg/L)
    turbidity: Union[float, np.ndarray] = 1.0  # 浊度 (NTU)
    wind_speed: Union[float, np.ndarray] = 5.0  # 风速 (m/s)
    solar_radiation: Union[float, np.ndarray] = 200.0  # 太阳辐射 (W/m²)
    wave_height: Union[float, np.ndarray] = 1.0  # 波高 (m)

@dataclass
class PollutantParticle:
    """污染物粒子"""
    x: float
    y: float
    z: float
    mass: float  # 质量 (kg)
    age: float = 0.0  # 年龄 (s)
    active: bool = True
    properties: Optional[PollutantProperties] = None
    weathering_state: Dict[str, float] = field(default_factory=dict)

class PollutionDispersionSimulator:
    """
    污染物扩散模拟器
    基于拉格朗日粒子追踪方法模拟污染物在海洋中的传播
    """

    def __init__(self, particle_tracker: Optional[ParticleTrackingWrapper] = None):
        """
        初始化污染物扩散模拟器
        
        Args:
            particle_tracker: 粒子追踪器实例
        """
        self.particle_tracker = particle_tracker or ParticleTrackingWrapper()
        self.discharge_sources = []
        self.pollutant_particles = []
        self.environmental_conditions = EnvironmentalConditions()
        self.simulation_time = 0.0
        self.time_step = 300.0  # 默认时间步长5分钟
        self.concentration_grid = None
        self.concentration_history = []
        self._lock = threading.Lock()

        logger.info("污染物扩散模拟器初始化完成")

    def add_discharge_source(self, source: DischargeSource):
        """
        添加排放源
        
        Args:
            source: 排放源对象
        """
        self.discharge_sources.append(source)
        logger.info(f"添加排放源: 位置({source.x}, {source.y}, {source.z}), "
                    f"排放速率: {source.discharge_rate} kg/s")

    def set_environmental_conditions(self, conditions: EnvironmentalConditions):
        """
        设置环境条件
        
        Args:
            conditions: 环境条件对象
        """
        self.environmental_conditions = conditions
        logger.info("环境条件已更新")

    def initialize_particles(self, initial_particle_count: int = 1000):
        """
        初始化污染物粒子
        
        Args:
            initial_particle_count: 初始粒子数量
        """
        self.pollutant_particles = []

        # 为每个排放源分配粒子
        particles_per_source = max(1, initial_particle_count // len(self.discharge_sources))

        for source in self.discharge_sources:
            for i in range(particles_per_source):
                # 在排放源周围随机分布粒子
                x_offset = np.random.normal(0, 10)  # 10米标准差
                y_offset = np.random.normal(0, 10)
                z_offset = np.random.normal(0, 1)   # 1米标准差

                particle = PollutantParticle(
                    x=source.x + x_offset,
                    y=source.y + y_offset,
                    z=source.z + z_offset,
                    mass=source.discharge_rate * self.time_step / particles_per_source,
                    properties=source.properties
                )

                # 初始化风化状态
                if source.properties:
                    particle.weathering_state = {
                        'evaporated_fraction': 0.0,
                        'dissolved_fraction': 0.0,
                        'degraded_fraction': 0.0,
                        'settled_fraction': 0.0
                    }

                self.pollutant_particles.append(particle)

        logger.info(f"初始化 {len(self.pollutant_particles)} 个污染物粒子")

    def inject_new_particles(self, current_time: float):
        """
        根据排放源注入新粒子
        
        Args:
            current_time: 当前时间
        """
        new_particles = []

        for source in self.discharge_sources:
            # 检查排放源是否在活跃期
            if not (source.start_time <= current_time <= source.end_time):
                continue

            # 计算当前排放速率
            discharge_rate = self._get_discharge_rate(source, current_time)
            if discharge_rate <= 0:
                continue

            # 计算新粒子数量和质量
            particles_to_inject = max(1, int(discharge_rate * self.time_step / 1000))  # 假设每个粒子代表1kg
            mass_per_particle = discharge_rate * self.time_step / particles_to_inject

            for _ in range(particles_to_inject):
                # 在排放源周围随机分布
                x_offset = np.random.normal(0, 5)
                y_offset = np.random.normal(0, 5)
                z_offset = np.random.normal(0, 0.5)

                particle = PollutantParticle(
                    x=source.x + x_offset,
                    y=source.y + y_offset,
                    z=source.z + z_offset,
                    mass=mass_per_particle,
                    properties=source.properties
                )

                # 初始化风化状态
                if source.properties:
                    particle.weathering_state = {
                        'evaporated_fraction': 0.0,
                        'dissolved_fraction': 0.0,
                        'degraded_fraction': 0.0,
                        'settled_fraction': 0.0
                    }

                new_particles.append(particle)

        self.pollutant_particles.extend(new_particles)
        if new_particles:
            logger.debug(f"注入 {len(new_particles)} 个新粒子")

    def _get_discharge_rate(self, source: DischargeSource, current_time: float) -> float:
        """获取当前时刻的排放速率"""
        if source.discharge_type == DischargeType.INSTANTANEOUS:
            # 瞬时排放，只在开始时刻排放
            return source.discharge_rate if abs(current_time - source.start_time) < self.time_step else 0.0

        elif source.discharge_type == DischargeType.CONTINUOUS:
            # 连续排放
            return source.discharge_rate

        elif source.discharge_type == DischargeType.VARIABLE:
            # 变化排放，根据时间序列插值
            if source.time_series:
                return self._interpolate_time_series(source.time_series, current_time)
            else:
                return source.discharge_rate

        return 0.0

    def _interpolate_time_series(self, time_series: List[Tuple[float, float]],
                                 current_time: float) -> float:
        """时间序列插值"""
        if not time_series:
            return 0.0

        # 找到当前时间的位置
        for i in range(len(time_series) - 1):
            t1, rate1 = time_series[i]
            t2, rate2 = time_series[i + 1]

            if t1 <= current_time <= t2:
                # 线性插值
                if t2 == t1:
                    return rate1
                return rate1 + (rate2 - rate1) * (current_time - t1) / (t2 - t1)

        # 超出范围，返回边界值
        if current_time < time_series[0][0]:
            return time_series[0][1]
        else:
            return time_series[-1][1]

    def advect_particles(self, velocity_field: Dict[str, np.ndarray],
                         grid_info: Dict[str, Union[float, int]]):
        """
        使用速度场平流粒子
        
        Args:
            velocity_field: 速度场字典 {'u': u_array, 'v': v_array, 'w': w_array}
            grid_info: 网格信息
        """
        if not self.pollutant_particles:
            return

        # 转换为粒子追踪器需要的格式
        particle_states = []
        for particle in self.pollutant_particles:
            if particle.active:
                state = ParticleState(
                    x=particle.x, y=particle.y, z=particle.z,
                    u=0.0, v=0.0, w=0.0, age=particle.age
                )
                particle_states.append(state)

        if not particle_states:
            return

        # 设置速度场
        if not self.particle_tracker.set_velocity_field(
                velocity_field['u'], velocity_field['v'], velocity_field['w'], grid_info
        ):
            logger.error("设置速度场失败")
            return

        # 配置追踪参数
        tracking_params = TrackingParameters(
            time_step=self.time_step,
            max_time=self.time_step,  # 只推进一个时间步
            output_interval=self.time_step
        )

        # 执行粒子追踪
        try:
            trajectories = self.particle_tracker.track_multiple_particles(
                particle_states, tracking_params, parallel=True
            )

            # 更新粒子位置
            active_particle_idx = 0
            for i, particle in enumerate(self.pollutant_particles):
                if particle.active and active_particle_idx < len(trajectories):
                    trajectory = trajectories[active_particle_idx]
                    if trajectory and len(trajectory) > 1:
                        final_state = trajectory[-1]
                        particle.x = final_state.x
                        particle.y = final_state.y
                        particle.z = final_state.z
                    active_particle_idx += 1

        except Exception as e:
            logger.error(f"粒子平流失败: {e}")

    def apply_diffusion(self, diffusion_coefficient: float = 10.0):
        """
        应用湍流扩散
        
        Args:
            diffusion_coefficient: 扩散系数 (m²/s)
        """
        if diffusion_coefficient <= 0:
            return

        # 计算扩散距离
        diffusion_distance = np.sqrt(2 * diffusion_coefficient * self.time_step)

        for particle in self.pollutant_particles:
            if particle.active:
                # 随机游走
                dx = np.random.normal(0, diffusion_distance)
                dy = np.random.normal(0, diffusion_distance)
                dz = np.random.normal(0, diffusion_distance * 0.1)  # 垂直扩散较小

                particle.x += dx
                particle.y += dy
                particle.z += dz

    def apply_weathering_processes(self):
        """应用风化过程"""
        for particle in self.pollutant_particles:
            if not particle.active or not particle.properties:
                continue

            # 更新粒子年龄
            particle.age += self.time_step

            # 计算各种风化过程
            self._apply_evaporation(particle)
            self._apply_dissolution(particle)
            self._apply_biodegradation(particle)
            self._apply_sedimentation(particle)

            # 检查粒子是否仍然活跃
            total_lost = sum([
                particle.weathering_state.get('evaporated_fraction', 0),
                particle.weathering_state.get('dissolved_fraction', 0),
                particle.weathering_state.get('degraded_fraction', 0),
                particle.weathering_state.get('settled_fraction', 0)
            ])

            if total_lost >= 0.99:  # 99%消失
                particle.active = False

    def _apply_evaporation(self, particle: PollutantParticle):
        """应用蒸发过程"""
        if not particle.properties or particle.properties.volatilization_rate <= 0:
            return

        # 蒸发速率取决于温度、风速和蒸汽压
        temp_factor = self._get_temperature_factor(particle.x, particle.y)
        wind_factor = self._get_wind_factor(particle.x, particle.y)

        evaporation_rate = (particle.properties.volatilization_rate *
                            temp_factor * wind_factor * self.time_step / 86400)  # 转换为秒

        current_evaporated = particle.weathering_state.get('evaporated_fraction', 0)
        new_evaporated = min(1.0, current_evaporated + evaporation_rate)
        particle.weathering_state['evaporated_fraction'] = new_evaporated

        # 更新粒子质量
        particle.mass *= (1 - evaporation_rate)

    def _apply_dissolution(self, particle: PollutantParticle):
        """应用溶解过程"""
        if not particle.properties or particle.properties.solubility <= 0:
            return

        # 溶解速率取决于溶解度和环境条件
        solubility_factor = min(1.0, particle.properties.solubility / 1000)  # 标准化
        dissolution_rate = solubility_factor * 0.01 * self.time_step / 86400  # 1%/天的基础速率

        current_dissolved = particle.weathering_state.get('dissolved_fraction', 0)
        new_dissolved = min(1.0, current_dissolved + dissolution_rate)
        particle.weathering_state['dissolved_fraction'] = new_dissolved

        # 更新粒子质量
        particle.mass *= (1 - dissolution_rate)

    def _apply_biodegradation(self, particle: PollutantParticle):
        """应用生物降解过程"""
        if not particle.properties or particle.properties.degradation_rate <= 0:
            return

        # 生物降解速率
        degradation_rate = particle.properties.degradation_rate * self.time_step / 86400

        current_degraded = particle.weathering_state.get('degraded_fraction', 0)
        new_degraded = min(1.0, current_degraded + degradation_rate)
        particle.weathering_state['degraded_fraction'] = new_degraded

        # 更新粒子质量
        particle.mass *= (1 - degradation_rate)

    def _apply_sedimentation(self, particle: PollutantParticle):
        """应用沉降过程"""
        if not particle.properties or particle.properties.settling_velocity <= 0:
            return

        # 计算沉降距离
        settling_distance = particle.properties.settling_velocity * self.time_step
        particle.z += settling_distance  # 向下沉降

        # 检查是否沉降到海底
        if particle.z > 200:  # 假设海深200米
            settled_fraction = particle.weathering_state.get('settled_fraction', 0)
            particle.weathering_state['settled_fraction'] = 1.0
            particle.active = False

    def _get_temperature_factor(self, x: float, y: float) -> float:
        """获取温度因子"""
        # 简化的温度因子计算
        if isinstance(self.environmental_conditions.water_temperature, np.ndarray):
            # 如果是数组，需要插值
            return 1.0  # 简化处理
        else:
            # 温度越高，蒸发越快
            temp_celsius = self.environmental_conditions.water_temperature - 273.15
            return max(0.1, temp_celsius / 25.0)  # 25°C为基准

    def _get_wind_factor(self, x: float, y: float) -> float:
        """获取风速因子"""
        if isinstance(self.environmental_conditions.wind_speed, np.ndarray):
            return 1.0  # 简化处理
        else:
            # 风速越大，蒸发越快
            return max(0.1, self.environmental_conditions.wind_speed / 10.0)  # 10m/s为基准

    def calculate_concentration_field(self, grid_bounds: Tuple[float, float, float, float],
                                      grid_resolution: Tuple[int, int]) -> np.ndarray:
        """
        计算浓度场
        
        Args:
            grid_bounds: 网格边界 (x_min, y_min, x_max, y_max)
            grid_resolution: 网格分辨率 (nx, ny)
            
        Returns:
            浓度场数组 [ny, nx]
        """
        x_min, y_min, x_max, y_max = grid_bounds
        nx, ny = grid_resolution

        # 创建网格
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)

        # 初始化浓度场
        concentration = np.zeros((ny, nx))

        # 将粒子质量分配到网格
        for particle in self.pollutant_particles:
            if not particle.active:
                continue

            # 找到粒子在网格中的位置
            i = int((particle.x - x_min) / dx)
            j = int((particle.y - y_min) / dy)

            # 检查边界
            if 0 <= i < nx and 0 <= j < ny:
                # 简单的点分配，可以改进为双线性插值
                grid_area = dx * dy
                concentration[j, i] += particle.mass / grid_area

        return concentration

    def calculate_statistics(self) -> Dict[str, Any]:
        """计算统计量"""
        active_particles = [p for p in self.pollutant_particles if p.active]

        if not active_particles:
            return {
                'total_particles': 0,
                'active_particles': 0,
                'total_mass': 0.0,
                'center_of_mass': (0.0, 0.0),
                'extent': 0.0,
                'weathering_summary': {}
            }

        # 基本统计
        total_mass = sum(p.mass for p in active_particles)
        x_coords = [p.x for p in active_particles]
        y_coords = [p.y for p in active_particles]

        center_x = np.average(x_coords, weights=[p.mass for p in active_particles])
        center_y = np.average(y_coords, weights=[p.mass for p in active_particles])

        # 计算扩散范围
        distances = [np.sqrt((p.x - center_x)**2 + (p.y - center_y)**2)
                     for p in active_particles]
        extent = np.max(distances) if distances else 0.0

        # 风化统计
        weathering_summary = {}
        if active_particles and active_particles[0].properties:
            for process in ['evaporated_fraction', 'dissolved_fraction',
                            'degraded_fraction', 'settled_fraction']:
                fractions = [p.weathering_state.get(process, 0) for p in active_particles]
                weathering_summary[process] = {
                    'mean': np.mean(fractions),
                    'max': np.max(fractions),
                    'std': np.std(fractions)
                }

        return {
            'total_particles': len(self.pollutant_particles),
            'active_particles': len(active_particles),
            'total_mass': total_mass,
            'center_of_mass': (center_x, center_y),
            'extent': extent,
            'weathering_summary': weathering_summary,
            'simulation_time': self.simulation_time
        }

    def run_simulation(self, total_time: float, velocity_field: Dict[str, np.ndarray],
                       grid_info: Dict[str, Union[float, int]],
                       callback_func: Optional[callable] = None) -> bool:
        """
        运行污染物扩散模拟
        
        Args:
            total_time: 总模拟时间 (秒)
            velocity_field: 速度场
            grid_info: 网格信息
            callback_func: 回调函数
            
        Returns:
            是否成功
        """
        logger.info(f"开始污染物扩散模拟，总时间: {total_time/3600:.1f} 小时")

        try:
            start_time = time.time()
            num_steps = int(total_time / self.time_step)

            for step in range(num_steps):
                self.simulation_time = step * self.time_step

                # 注入新粒子
                self.inject_new_particles(self.simulation_time)

                # 平流
                self.advect_particles(velocity_field, grid_info)

                # 扩散
                self.apply_diffusion()

                # 风化过程
                self.apply_weathering_processes()

                # 计算浓度场（定期）
                if step % 10 == 0:
                    bounds = (
                        grid_info.get('x0', 0), grid_info.get('y0', 0),
                        grid_info.get('x0', 0) + grid_info.get('nx', 100) * grid_info.get('dx', 1000),
                        grid_info.get('y0', 0) + grid_info.get('ny', 80) * grid_info.get('dy', 1000)
                    )
                    concentration = self.calculate_concentration_field(
                        bounds, (grid_info.get('nx', 100), grid_info.get('ny', 80))
                    )
                    self.concentration_history.append((self.simulation_time, concentration))

                # 回调函数
                if callback_func and step % 100 == 0:
                    try:
                        stats = self.calculate_statistics()
                        callback_func(step, num_steps, stats)
                    except Exception as e:
                        logger.warning(f"回调函数执行异常: {e}")

                # 进度报告
                if step % 1000 == 0:
                    elapsed = time.time() - start_time
                    progress = (step + 1) / num_steps * 100
                    stats = self.calculate_statistics()
                    logger.info(f"进度: {progress:.1f}%, 活跃粒子: {stats['active_particles']}, "
                                f"总质量: {stats['total_mass']:.2e} kg, 耗时: {elapsed:.1f}s")

            total_elapsed = time.time() - start_time
            logger.info(f"污染物扩散模拟完成，总耗时: {total_elapsed:.1f}s")
            return True

        except Exception as e:
            logger.error(f"模拟运行异常: {e}")
            return False

    def export_results(self, filename_prefix: str, format: str = 'netcdf') -> bool:
        """
        导出模拟结果
        
        Args:
            filename_prefix: 文件名前缀
            format: 输出格式 ('netcdf', 'csv', 'hdf5')
            
        Returns:
            是否导出成功
        """
        try:
            if format.lower() == 'csv':
                return self._export_csv(filename_prefix)
            elif format.lower() == 'netcdf':
                return self._export_netcdf(filename_prefix)
            elif format.lower() == 'hdf5':
                return self._export_hdf5(filename_prefix)
            else:
                logger.error(f"不支持的输出格式: {format}")
                return False
        except Exception as e:
            logger.error(f"导出结果失败: {e}")
            return False

    def _export_csv(self, filename_prefix: str) -> bool:
        """导出为CSV格式"""
        import csv

        # 导出粒子数据
        particle_filename = f"{filename_prefix}_particles.csv"
        with open(particle_filename, 'w', newline='') as csvfile:
            fieldnames = ['particle_id', 'x', 'y', 'z', 'mass', 'age', 'active',
                          'evaporated_fraction', 'dissolved_fraction', 'degraded_fraction', 'settled_fraction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, particle in enumerate(self.pollutant_particles):
                row = {
                    'particle_id': i,
                    'x': particle.x,
                    'y': particle.y,
                    'z': particle.z,
                    'mass': particle.mass,
                    'age': particle.age,
                    'active': particle.active,
                    'evaporated_fraction': particle.weathering_state.get('evaporated_fraction', 0),
                    'dissolved_fraction': particle.weathering_state.get('dissolved_fraction', 0),
                    'degraded_fraction': particle.weathering_state.get('degraded_fraction', 0),
                    'settled_fraction': particle.weathering_state.get('settled_fraction', 0)
                }
                writer.writerow(row)

        # 导出统计数据
        stats_filename = f"{filename_prefix}_statistics.csv"
        stats = self.calculate_statistics()
        with open(stats_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Statistic', 'Value'])
            for key, value in stats.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        writer.writerow([f"{key}_{subkey}", subvalue])
                else:
                    writer.writerow([key, value])

        logger.info(f"结果已导出到: {particle_filename}, {stats_filename}")
        return True

    def _export_netcdf(self, filename_prefix: str) -> bool:
        """导出为NetCDF格式"""
        try:
            import netCDF4 as nc

            filename = f"{filename_prefix}_pollution.nc"
            with nc.Dataset(filename, 'w') as dataset:
                # 创建维度
                n_particles = len(self.pollutant_particles)
                dataset.createDimension('particle', n_particles)
                dataset.createDimension('time', len(self.concentration_history))

                if self.concentration_history:
                    _, first_concentration = self.concentration_history[0]
                    ny, nx = first_concentration.shape
                    dataset.createDimension('x', nx)
                    dataset.createDimension('y', ny)

                # 粒子数据
                x_var = dataset.createVariable('particle_x', 'f8', ('particle',))
                y_var = dataset.createVariable('particle_y', 'f8', ('particle',))
                z_var = dataset.createVariable('particle_z', 'f8', ('particle',))
                mass_var = dataset.createVariable('particle_mass', 'f8', ('particle',))
                age_var = dataset.createVariable('particle_age', 'f8', ('particle',))
                active_var = dataset.createVariable('particle_active', 'i4', ('particle',))

                # 填充粒子数据
                for i, particle in enumerate(self.pollutant_particles):
                    x_var[i] = particle.x
                    y_var[i] = particle.y
                    z_var[i] = particle.z
                    mass_var[i] = particle.mass
                    age_var[i] = particle.age
                    active_var[i] = 1 if particle.active else 0

                # 浓度场历史
                if self.concentration_history:
                    time_var = dataset.createVariable('time', 'f8', ('time',))
                    conc_var = dataset.createVariable('concentration', 'f8', ('time', 'y', 'x'))

                    for i, (time_val, concentration) in enumerate(self.concentration_history):
                        time_var[i] = time_val
                        conc_var[i, :, :] = concentration

                    # 添加属性
                    conc_var.units = 'kg/m^2'
                    conc_var.long_name = 'Pollutant concentration'
                    time_var.units = 'seconds since simulation start'

                # 全局属性
                dataset.title = 'Pollution Dispersion Simulation Results'
                dataset.source = 'C# Multi-language Ocean Current Simulation System'
                dataset.simulation_time = self.simulation_time

                # 统计信息
                stats = self.calculate_statistics()
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        setattr(dataset, f"stat_{key}", value)

            logger.info(f"结果已导出到: {filename}")
            return True

        except ImportError:
            logger.error("需要安装netCDF4库")
            return False
        except Exception as e:
            logger.error(f"NetCDF导出失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        self.pollutant_particles.clear()
        self.concentration_history.clear()
        if self.particle_tracker:
            self.particle_tracker.cleanup()
        logger.info("污染物扩散模拟器已清理")


# 预定义的污染物属性
OIL_PROPERTIES = {
    'light_crude': PollutantProperties(
        pollutant_type=PollutantType.OIL,
        density=850.0,
        viscosity=0.01,
        solubility=0.1,
        vapor_pressure=1000.0,
        molecular_weight=200.0,
        boiling_point=473.15,
        degradation_rate=0.1,
        volatilization_rate=0.2,
        settling_velocity=0.0
    ),
    'heavy_crude': PollutantProperties(
        pollutant_type=PollutantType.OIL,
        density=950.0,
        viscosity=0.5,
        solubility=0.01,
        vapor_pressure=100.0,
        molecular_weight=400.0,
        boiling_point=573.15,
        degradation_rate=0.05,
        volatilization_rate=0.02,
        settling_velocity=0.001
    )
}

CHEMICAL_PROPERTIES = {
    'benzene': PollutantProperties(
        pollutant_type=PollutantType.CHEMICAL,
        density=879.0,
        viscosity=0.0006,
        solubility=1780.0,
        vapor_pressure=12700.0,
        molecular_weight=78.11,
        boiling_point=353.15,
        degradation_rate=0.3,
        volatilization_rate=0.8,
        settling_velocity=0.0
    ),
    'toluene': PollutantProperties(
        pollutant_type=PollutantType.CHEMICAL,
        density=867.0,
        viscosity=0.0006,
        solubility=526.0,
        vapor_pressure=3800.0,
        molecular_weight=92.14,
        boiling_point=383.15,
        degradation_rate=0.25,
        volatilization_rate=0.6,
        settling_velocity=0.0
    )
}


# 使用示例
if __name__ == "__main__":
    # 创建污染物扩散模拟器
    simulator = PollutionDispersionSimulator()

    # 创建排放源
    oil_spill = DischargeSource(
        x=50000.0, y=40000.0, z=0.0,
        discharge_rate=1000.0,  # 1000 kg/s
        discharge_type=DischargeType.INSTANTANEOUS,
        properties=OIL_PROPERTIES['light_crude']
    )

    chemical_discharge = DischargeSource(
        x=60000.0, y=45000.0, z=0.0,
        discharge_rate=10.0,  # 10 kg/s
        discharge_type=DischargeType.CONTINUOUS,
        start_time=3600.0,  # 1小时后开始
        end_time=7200.0,    # 2小时后结束
        properties=CHEMICAL_PROPERTIES['benzene']
    )

    # 添加排放源
    simulator.add_discharge_source(oil_spill)
    simulator.add_discharge_source(chemical_discharge)

    # 设置环境条件
    env_conditions = EnvironmentalConditions(
        water_temperature=288.15,  # 15°C
        salinity=35.0,
        wind_speed=8.0,
        wave_height=2.0
    )
    simulator.set_environmental_conditions(env_conditions)

    # 初始化粒子
    simulator.initialize_particles(2000)

    # 创建示例速度场
    nx, ny, nz = 100, 80, 10
    u = np.random.randn(nz, ny, nx) * 0.5
    v = np.random.randn(nz, ny, nx) * 0.3
    w = np.random.randn(nz, ny, nx) * 0.1

    velocity_field = {'u': u, 'v': v, 'w': w}
    grid_info = {
        'nx': nx, 'ny': ny, 'nz': nz,
        'dx': 1000.0, 'dy': 1000.0, 'dz': 10.0,
        'x0': 0.0, 'y0': 0.0, 'z0': 0.0
    }

    # 定义回调函数
    def progress_callback(step, total_steps, stats):
        progress = step / total_steps * 100
        print(f"进度: {progress:.1f}%, 活跃粒子: {stats['active_particles']}, "
              f"质心位置: ({stats['center_of_mass'][0]:.0f}, {stats['center_of_mass'][1]:.0f}), "
              f"扩散范围: {stats['extent']:.0f} m")

    # 运行模拟
    total_time = 24 * 3600  # 24小时
    print("开始污染物扩散模拟...")

    success = simulator.run_simulation(
        total_time, velocity_field, grid_info, progress_callback
    )

    if success:
        print("模拟完成!")

        # 获取最终统计
        final_stats = simulator.calculate_statistics()
        print(f"最终活跃粒子数: {final_stats['active_particles']}")
        print(f"总质量: {final_stats['total_mass']:.2e} kg")
        print(f"扩散范围: {final_stats['extent']:.0f} m")

        # 输出风化统计
        if final_stats['weathering_summary']:
            print("\n风化过程统计:")
            for process, stats in final_stats['weathering_summary'].items():
                print(f"  {process}: 平均 {stats['mean']:.3f}, 最大 {stats['max']:.3f}")

        # 导出结果
        simulator.export_results("pollution_simulation", "csv")
        simulator.export_results("pollution_simulation", "netcdf")
    else:
        print("模拟失败!")

    # 清理
    simulator.cleanup()
    print("污染物扩散模拟示例完成")