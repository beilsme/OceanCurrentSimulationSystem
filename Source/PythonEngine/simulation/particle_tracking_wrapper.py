"""
粒子追踪包装器
调用C++核心模块进行高性能粒子追踪计算
支持拉格朗日粒子追踪、轨迹预测和集合模拟
"""

import numpy as np
import ctypes
from ctypes import Structure, c_double, c_int, c_float, POINTER, byref, c_void_p
from typing import List, Tuple, Optional, Dict, Any, Union
import os
import sys
from dataclasses import dataclass
from enum import Enum
import logging
import threading
import time

# 设置日志
logger = logging.getLogger(__name__)

class IntegrationMethod(Enum):
    """数值积分方法"""
    EULER = 0
    RK2 = 1
    RK4 = 2
    ADAPTIVE_RK45 = 3

class BoundaryCondition(Enum):
    """边界条件"""
    REFLECTIVE = 0
    ABSORBING = 1
    PERIODIC = 2

@dataclass
class ParticleState:
    """粒子状态"""
    x: float
    y: float
    z: float
    u: float  # x方向速度
    v: float  # y方向速度
    w: float  # z方向速度
    age: float  # 粒子年龄
    active: bool = True
    properties: Optional[Dict[str, float]] = None

@dataclass
class TrackingParameters:
    """追踪参数"""
    time_step: float = 3600.0  # 时间步长(秒)
    max_time: float = 864000.0  # 最大追踪时间(秒，10天)
    integration_method: IntegrationMethod = IntegrationMethod.RK4
    boundary_condition: BoundaryCondition = BoundaryCondition.REFLECTIVE
    diffusion_coefficient: float = 10.0  # 扩散系数(m²/s)
    enable_turbulence: bool = True
    enable_stokes_drift: bool = False
    enable_wind_drift: bool = False
    wind_drift_factor: float = 0.03
    output_interval: float = 3600.0  # 输出间隔(秒)

# C++结构体定义
class CParticleState(Structure):
    """C++粒子状态结构体"""
    _fields_ = [
        ("x", c_double),
        ("y", c_double),
        ("z", c_double),
        ("u", c_double),
        ("v", c_double),
        ("w", c_double),
        ("age", c_double),
        ("active", c_int),
    ]

class CTrackingParams(Structure):
    """C++追踪参数结构体"""
    _fields_ = [
        ("time_step", c_double),
        ("max_time", c_double),
        ("integration_method", c_int),
        ("boundary_condition", c_int),
        ("diffusion_coefficient", c_double),
        ("enable_turbulence", c_int),
        ("enable_stokes_drift", c_int),
        ("enable_wind_drift", c_int),
        ("wind_drift_factor", c_double),
        ("output_interval", c_double),
    ]

class CVelocityField(Structure):
    """C++速度场结构体"""
    _fields_ = [
        ("data", POINTER(c_double)),
        ("nx", c_int),
        ("ny", c_int),
        ("nz", c_int),
        ("nt", c_int),
        ("dx", c_double),
        ("dy", c_double),
        ("dz", c_double),
        ("dt", c_double),
        ("x0", c_double),
        ("y0", c_double),
        ("z0", c_double),
        ("t0", c_double),
    ]

class ParticleTrackingWrapper:
    """
    粒子追踪包装器
    提供对C++粒子追踪模块的Python接口
    """

    def __init__(self, cpp_lib_path: Optional[str] = None):
        """
        初始化粒子追踪包装器
        
        Args:
            cpp_lib_path: C++动态库路径
        """
        self.lib = None
        self.velocity_field = None
        self.is_initialized = False
        self.tracking_threads = []
        self._lock = threading.Lock()

        # 加载C++动态库
        self._load_cpp_library(cpp_lib_path)

        # 配置函数签名
        self._configure_function_signatures()

        logger.info("粒子追踪包装器初始化完成")

    def _load_cpp_library(self, lib_path: Optional[str]):
        """加载C++动态库"""
        if lib_path is None:
            # 自动查找库文件
            possible_paths = [
                "./build/lib/libparticle_tracking.so",
                "./build/lib/libparticle_tracking.dll",
                "../CppCore/build/lib/libparticle_tracking.so",
                "../CppCore/build/lib/libparticle_tracking.dll",
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    lib_path = path
                    break

            if lib_path is None:
                raise FileNotFoundError("找不到粒子追踪C++库文件")

        try:
            self.lib = ctypes.CDLL(lib_path)
            logger.info(f"成功加载C++库: {lib_path}")
        except OSError as e:
            logger.error(f"加载C++库失败: {e}")
            raise

    def _configure_function_signatures(self):
        """配置C++函数签名"""
        if self.lib is None:
            return

        # 初始化函数
        self.lib.initialize_particle_tracker.argtypes = []
        self.lib.initialize_particle_tracker.restype = c_int

        # 设置速度场
        self.lib.set_velocity_field.argtypes = [POINTER(CVelocityField)]
        self.lib.set_velocity_field.restype = c_int

        # 追踪单个粒子
        self.lib.track_particle.argtypes = [
            POINTER(CParticleState), POINTER(CTrackingParams),
            POINTER(POINTER(CParticleState)), POINTER(c_int)
        ]
        self.lib.track_particle.restype = c_int

        # 批量追踪粒子
        self.lib.track_particles_batch.argtypes = [
            POINTER(CParticleState), c_int, POINTER(CTrackingParams),
            POINTER(POINTER(CParticleState)), POINTER(c_int), POINTER(c_int)
        ]
        self.lib.track_particles_batch.restype = c_int

        # 释放内存
        self.lib.free_trajectory_memory.argtypes = [POINTER(CParticleState), c_int]
        self.lib.free_trajectory_memory.restype = None

        # 清理函数
        self.lib.cleanup_particle_tracker.argtypes = []
        self.lib.cleanup_particle_tracker.restype = None

    def initialize(self) -> bool:
        """
        初始化粒子追踪器
        
        Returns:
            是否初始化成功
        """
        if self.lib is None:
            logger.error("C++库未加载")
            return False

        try:
            result = self.lib.initialize_particle_tracker()
            self.is_initialized = (result == 0)

            if self.is_initialized:
                logger.info("粒子追踪器初始化成功")
            else:
                logger.error("粒子追踪器初始化失败")

            return self.is_initialized
        except Exception as e:
            logger.error(f"初始化异常: {e}")
            return False

    def set_velocity_field(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                           grid_info: Dict[str, Union[float, int]]) -> bool:
        """
        设置速度场数据
        
        Args:
            u, v, w: 三维速度场数组 [nt, nz, ny, nx]
            grid_info: 网格信息字典
            
        Returns:
            是否设置成功
        """
        if not self.is_initialized:
            logger.error("追踪器未初始化")
            return False

        try:
            # 验证数组维度
            if u.shape != v.shape or u.shape != w.shape:
                raise ValueError("速度场数组维度不匹配")

            nt, nz, ny, nx = u.shape

            # 创建合并的速度场数据 [nt, nz, ny, nx, 3]
            velocity_data = np.stack([u, v, w], axis=-1)
            velocity_data = velocity_data.ascontiguousarray(dtype=np.float64)

            # 创建C++速度场结构体
            c_field = CVelocityField()
            c_field.data = velocity_data.ctypes.data_as(POINTER(c_double))
            c_field.nx = nx
            c_field.ny = ny
            c_field.nz = nz
            c_field.nt = nt
            c_field.dx = grid_info.get('dx', 1000.0)
            c_field.dy = grid_info.get('dy', 1000.0)
            c_field.dz = grid_info.get('dz', 10.0)
            c_field.dt = grid_info.get('dt', 3600.0)
            c_field.x0 = grid_info.get('x0', 0.0)
            c_field.y0 = grid_info.get('y0', 0.0)
            c_field.z0 = grid_info.get('z0', 0.0)
            c_field.t0 = grid_info.get('t0', 0.0)

            # 调用C++函数
            result = self.lib.set_velocity_field(byref(c_field))

            if result == 0:
                self.velocity_field = velocity_data  # 保持引用
                logger.info("速度场设置成功")
                return True
            else:
                logger.error("速度场设置失败")
                return False

        except Exception as e:
            logger.error(f"设置速度场异常: {e}")
            return False

    def track_single_particle(self, initial_state: ParticleState,
                              params: TrackingParameters) -> List[ParticleState]:
        """
        追踪单个粒子
        
        Args:
            initial_state: 初始粒子状态
            params: 追踪参数
            
        Returns:
            粒子轨迹列表
        """
        if not self.is_initialized:
            raise RuntimeError("追踪器未初始化")

        # 转换为C++结构体
        c_particle = self._python_to_c_particle(initial_state)
        c_params = self._python_to_c_params(params)

        # 准备输出参数
        trajectory_ptr = POINTER(CParticleState)()
        trajectory_length = c_int()

        try:
            # 调用C++函数
            result = self.lib.track_particle(
                byref(c_particle), byref(c_params),
                byref(trajectory_ptr), byref(trajectory_length)
            )

            if result != 0:
                raise RuntimeError(f"粒子追踪失败，错误码: {result}")

            # 转换结果
            trajectory = self._c_to_python_trajectory(
                trajectory_ptr, trajectory_length.value
            )

            # 释放C++内存
            self.lib.free_trajectory_memory(trajectory_ptr, trajectory_length.value)

            return trajectory

        except Exception as e:
            logger.error(f"单粒子追踪异常: {e}")
            raise

    def track_multiple_particles(self, initial_states: List[ParticleState],
                                 params: TrackingParameters,
                                 parallel: bool = True) -> List[List[ParticleState]]:
        """
        追踪多个粒子
        
        Args:
            initial_states: 初始粒子状态列表
            params: 追踪参数
            parallel: 是否并行计算
            
        Returns:
            粒子轨迹列表的列表
        """
        if not self.is_initialized:
            raise RuntimeError("追踪器未初始化")

        if not initial_states:
            return []

        if parallel and len(initial_states) > 1:
            return self._track_parallel(initial_states, params)
        else:
            return self._track_sequential(initial_states, params)

    def _track_sequential(self, initial_states: List[ParticleState],
                          params: TrackingParameters) -> List[List[ParticleState]]:
        """顺序追踪多个粒子"""
        results = []

        for i, state in enumerate(initial_states):
            try:
                trajectory = self.track_single_particle(state, params)
                results.append(trajectory)
                logger.debug(f"完成粒子 {i+1}/{len(initial_states)} 的追踪")
            except Exception as e:
                logger.error(f"粒子 {i+1} 追踪失败: {e}")
                results.append([])

        return results

    def _track_parallel(self, initial_states: List[ParticleState],
                        params: TrackingParameters) -> List[List[ParticleState]]:
        """并行追踪多个粒子"""
        num_particles = len(initial_states)

        # 转换为C++数组
        c_particles = (CParticleState * num_particles)()
        for i, state in enumerate(initial_states):
            c_particles[i] = self._python_to_c_particle(state)

        c_params = self._python_to_c_params(params)

        # 准备输出参数
        trajectories_ptr = POINTER(POINTER(CParticleState))()
        trajectory_lengths = (c_int * num_particles)()
        total_points = c_int()

        try:
            # 调用C++批量追踪函数
            result = self.lib.track_particles_batch(
                c_particles, num_particles, byref(c_params),
                byref(trajectories_ptr), trajectory_lengths, byref(total_points)
            )

            if result != 0:
                raise RuntimeError(f"批量粒子追踪失败，错误码: {result}")

            # 转换结果
            results = []
            for i in range(num_particles):
                if trajectory_lengths[i] > 0:
                    trajectory = self._c_to_python_trajectory(
                        trajectories_ptr[i], trajectory_lengths[i]
                    )
                    results.append(trajectory)
                else:
                    results.append([])

            # 释放C++内存
            for i in range(num_particles):
                if trajectory_lengths[i] > 0:
                    self.lib.free_trajectory_memory(
                        trajectories_ptr[i], trajectory_lengths[i]
                    )

            return results

        except Exception as e:
            logger.error(f"并行粒子追踪异常: {e}")
            raise

    def calculate_ensemble_statistics(self, trajectories: List[List[ParticleState]],
                                      time_indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        计算集合统计量
        
        Args:
            trajectories: 粒子轨迹列表
            time_indices: 时间索引列表
            
        Returns:
            统计量字典
        """
        if not trajectories:
            return {}

        # 确定时间索引
        if time_indices is None:
            max_length = max(len(traj) for traj in trajectories if traj)
            time_indices = list(range(max_length))

        stats = {
            'mean_x': [],
            'mean_y': [],
            'std_x': [],
            'std_y': [],
            'center_of_mass': [],
            'spread': [],
            'active_count': []
        }

        for t_idx in time_indices:
            positions_x = []
            positions_y = []
            active_count = 0

            for trajectory in trajectories:
                if trajectory and t_idx < len(trajectory):
                    particle = trajectory[t_idx]
                    if particle.active:
                        positions_x.append(particle.x)
                        positions_y.append(particle.y)
                        active_count += 1

            if positions_x:
                x_array = np.array(positions_x)
                y_array = np.array(positions_y)

                stats['mean_x'].append(np.mean(x_array))
                stats['mean_y'].append(np.mean(y_array))
                stats['std_x'].append(np.std(x_array))
                stats['std_y'].append(np.std(y_array))

                # 质心
                center_x = np.mean(x_array)
                center_y = np.mean(y_array)
                stats['center_of_mass'].append((center_x, center_y))

                # 扩散范围
                distances = np.sqrt((x_array - center_x)**2 + (y_array - center_y)**2)
                stats['spread'].append(np.mean(distances))

                stats['active_count'].append(active_count)
            else:
                stats['mean_x'].append(np.nan)
                stats['mean_y'].append(np.nan)
                stats['std_x'].append(np.nan)
                stats['std_y'].append(np.nan)
                stats['center_of_mass'].append((np.nan, np.nan))
                stats['spread'].append(np.nan)
                stats['active_count'].append(0)

        # 转换为numpy数组
        for key in ['mean_x', 'mean_y', 'std_x', 'std_y', 'spread', 'active_count']:
            stats[key] = np.array(stats[key])

        return stats

    def export_trajectories(self, trajectories: List[List[ParticleState]],
                            filename: str, format: str = 'csv') -> bool:
        """
        导出轨迹数据
        
        Args:
            trajectories: 粒子轨迹列表
            filename: 输出文件名
            format: 输出格式 ('csv', 'netcdf', 'hdf5')
            
        Returns:
            是否导出成功
        """
        try:
            if format.lower() == 'csv':
                return self._export_csv(trajectories, filename)
            elif format.lower() == 'netcdf':
                return self._export_netcdf(trajectories, filename)
            elif format.lower() == 'hdf5':
                return self._export_hdf5(trajectories, filename)
            else:
                logger.error(f"不支持的输出格式: {format}")
                return False
        except Exception as e:
            logger.error(f"导出轨迹失败: {e}")
            return False

    def _export_csv(self, trajectories: List[List[ParticleState]], filename: str) -> bool:
        """导出为CSV格式"""
        import csv

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['particle_id', 'time_step', 'x', 'y', 'z', 'u', 'v', 'w', 'age', 'active']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for particle_id, trajectory in enumerate(trajectories):
                for time_step, state in enumerate(trajectory):
                    writer.writerow({
                        'particle_id': particle_id,
                        'time_step': time_step,
                        'x': state.x,
                        'y': state.y,
                        'z': state.z,
                        'u': state.u,
                        'v': state.v,
                        'w': state.w,
                        'age': state.age,
                        'active': state.active
                    })

        logger.info(f"轨迹数据已导出到: {filename}")
        return True

    def _python_to_c_particle(self, state: ParticleState) -> CParticleState:
        """转换Python粒子状态为C++结构体"""
        c_particle = CParticleState()
        c_particle.x = state.x
        c_particle.y = state.y
        c_particle.z = state.z
        c_particle.u = state.u
        c_particle.v = state.v
        c_particle.w = state.w
        c_particle.age = state.age
        c_particle.active = 1 if state.active else 0
        return c_particle

    def _python_to_c_params(self, params: TrackingParameters) -> CTrackingParams:
        """转换Python追踪参数为C++结构体"""
        c_params = CTrackingParams()
        c_params.time_step = params.time_step
        c_params.max_time = params.max_time
        c_params.integration_method = params.integration_method.value
        c_params.boundary_condition = params.boundary_condition.value
        c_params.diffusion_coefficient = params.diffusion_coefficient
        c_params.enable_turbulence = 1 if params.enable_turbulence else 0
        c_params.enable_stokes_drift = 1 if params.enable_stokes_drift else 0
        c_params.enable_wind_drift = 1 if params.enable_wind_drift else 0
        c_params.wind_drift_factor = params.wind_drift_factor
        c_params.output_interval = params.output_interval
        return c_params

    def _c_to_python_trajectory(self, trajectory_ptr: POINTER(CParticleState),
                                length: int) -> List[ParticleState]:
        """转换C++轨迹为Python对象列表"""
        trajectory = []

        for i in range(length):
            c_state = trajectory_ptr[i]
            state = ParticleState(
                x=c_state.x,
                y=c_state.y,
                z=c_state.z,
                u=c_state.u,
                v=c_state.v,
                w=c_state.w,
                age=c_state.age,
                active=bool(c_state.active)
            )
            trajectory.append(state)

        return trajectory

    def cleanup(self):
        """清理资源"""
        if self.lib and self.is_initialized:
            self.lib.cleanup_particle_tracker()
            self.is_initialized = False
            logger.info("粒子追踪器已清理")

    def __del__(self):
        """析构函数"""
        self.cleanup()


# 使用示例
if __name__ == "__main__":
    # 创建粒子追踪包装器
    tracker = ParticleTrackingWrapper()

    # 初始化
    if not tracker.initialize():
        print("初始化失败")
        exit(1)

    # 创建示例速度场
    nx, ny, nz, nt = 100, 80, 20, 24
    u = np.random.randn(nt, nz, ny, nx) * 0.5
    v = np.random.randn(nt, nz, ny, nx) * 0.3
    w = np.random.randn(nt, nz, ny, nx) * 0.1

    grid_info = {
        'dx': 1000.0, 'dy': 1000.0, 'dz': 10.0, 'dt': 3600.0,
        'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 't0': 0.0
    }

    # 设置速度场
    if not tracker.set_velocity_field(u, v, w, grid_info):
        print("设置速度场失败")
        exit(1)

    # 创建初始粒子
    initial_states = [
        ParticleState(x=10000.0, y=15000.0, z=5.0, u=0.0, v=0.0, w=0.0, age=0.0),
        ParticleState(x=12000.0, y=18000.0, z=8.0, u=0.0, v=0.0, w=0.0, age=0.0),
        ParticleState(x=8000.0, y=12000.0, z=3.0, u=0.0, v=0.0, w=0.0, age=0.0),
    ]

    # 设置追踪参数
    params = TrackingParameters(
        time_step=1800.0,
        max_time=172800.0,  # 2天
        integration_method=IntegrationMethod.RK4,
        diffusion_coefficient=10.0,
        output_interval=3600.0
    )

    # 执行追踪
    print("开始粒子追踪...")
    trajectories = tracker.track_multiple_particles(initial_states, params)

    print(f"追踪完成，获得 {len(trajectories)} 条轨迹")
    for i, traj in enumerate(trajectories):
        print(f"粒子 {i+1}: {len(traj)} 个时间步")

    # 计算统计量
    stats = tracker.calculate_ensemble_statistics(trajectories)
    print(f"平均扩散范围: {np.nanmean(stats['spread']):.2f} m")

    # 导出结果
    tracker.export_trajectories(trajectories, "particle_trajectories.csv")

    # 清理
    tracker.cleanup()
    print("粒子追踪示例完成")