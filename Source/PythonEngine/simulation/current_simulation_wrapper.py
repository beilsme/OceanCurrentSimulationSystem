"""
洋流模拟包装器
调用C++核心模块进行高性能洋流数值模拟
支持有限差分、有限元和谱方法求解
"""

import numpy as np
import ctypes
from ctypes import Structure, c_double, c_int, c_float, POINTER, byref, c_void_p, c_char_p
from typing import List, Tuple, Optional, Dict, Any, Union
import os
import logging
from dataclasses import dataclass
from enum import Enum
import threading
import time


# 设置日志
logger = logging.getLogger(__name__)

class SolverMethod(Enum):
    """求解器方法"""
    FINITE_DIFFERENCE = 0
    FINITE_ELEMENT = 1
    SPECTRAL = 2
    FINITE_VOLUME = 3

class BoundaryType(Enum):
    """边界条件类型"""
    DIRICHLET = 0  # 第一类边界条件
    NEUMANN = 1    # 第二类边界条件
    ROBIN = 2      # 第三类边界条件
    PERIODIC = 3   # 周期边界条件

class TimeScheme(Enum):
    """时间积分格式"""
    EXPLICIT_EULER = 0
    IMPLICIT_EULER = 1
    CRANK_NICOLSON = 2
    RUNGE_KUTTA_4 = 3
    ADAMS_BASHFORTH = 4

@dataclass
class GridConfiguration:
    """网格配置"""
    nx: int  # x方向网格数
    ny: int  # y方向网格数
    nz: int  # z方向网格数
    dx: float  # x方向网格间距
    dy: float  # y方向网格间距
    dz: float  # z方向网格间距
    x_min: float  # x方向最小值
    y_min: float  # y方向最小值
    z_min: float  # z方向最小值
    x_max: float  # x方向最大值
    y_max: float  # y方向最大值
    z_max: float  # z方向最大值

@dataclass
class SimulationParameters:
    """模拟参数"""
    time_step: float = 300.0  # 时间步长(秒)
    total_time: float = 86400.0  # 总模拟时间(秒)
    viscosity: float = 1e-6  # 运动粘度(m²/s)
    density: float = 1025.0  # 密度(kg/m³)
    coriolis_parameter: float = 1e-4  # 科里奥利参数(1/s)
    gravity: float = 9.81  # 重力加速度(m/s²)
    solver_method: SolverMethod = SolverMethod.FINITE_DIFFERENCE
    time_scheme: TimeScheme = TimeScheme.RUNGE_KUTTA_4
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    enable_nonlinear: bool = True
    enable_turbulence: bool = False

@dataclass
class BoundaryCondition:
    """边界条件"""
    boundary_type: BoundaryType
    value: float
    location: str  # 'north', 'south', 'east', 'west', 'top', 'bottom'

@dataclass
class ForceField:
    """外力场"""
    wind_stress_x: Optional[np.ndarray] = None
    wind_stress_y: Optional[np.ndarray] = None
    pressure_gradient_x: Optional[np.ndarray] = None
    pressure_gradient_y: Optional[np.ndarray] = None
    temperature_gradient: Optional[np.ndarray] = None
    salinity_gradient: Optional[np.ndarray] = None

# C++结构体定义
class CGridConfig(Structure):
    """C++网格配置结构体"""
    _fields_ = [
        ("nx", c_int),
        ("ny", c_int),
        ("nz", c_int),
        ("dx", c_double),
        ("dy", c_double),
        ("dz", c_double),
        ("x_min", c_double),
        ("y_min", c_double),
        ("z_min", c_double),
        ("x_max", c_double),
        ("y_max", c_double),
        ("z_max", c_double),
    ]

class CSimParams(Structure):
    """C++模拟参数结构体"""
    _fields_ = [
        ("time_step", c_double),
        ("total_time", c_double),
        ("viscosity", c_double),
        ("density", c_double),
        ("coriolis_parameter", c_double),
        ("gravity", c_double),
        ("solver_method", c_int),
        ("time_scheme", c_int),
        ("max_iterations", c_int),
        ("convergence_tolerance", c_double),
        ("enable_nonlinear", c_int),
        ("enable_turbulence", c_int),
    ]

class CBoundaryCondition(Structure):
    """C++边界条件结构体"""
    _fields_ = [
        ("boundary_type", c_int),
        ("value", c_double),
        ("location", c_char_p),
    ]

class CCurrentField(Structure):
    """C++洋流场结构体"""
    _fields_ = [
        ("u", POINTER(c_double)),  # x方向速度
        ("v", POINTER(c_double)),  # y方向速度
        ("w", POINTER(c_double)),  # z方向速度
        ("p", POINTER(c_double)),  # 压力场
        ("nx", c_int),
        ("ny", c_int),
        ("nz", c_int),
        ("nt", c_int),
    ]

class CurrentSimulationWrapper:
    """
    洋流模拟包装器
    提供对C++洋流模拟核心的Python接口
    """

    def __init__(self, cpp_lib_path: Optional[str] = None):
        """
        初始化洋流模拟包装器
        
        Args:
            cpp_lib_path: C++动态库路径
        """
        self.lib = None
        self.grid_config = None
        self.sim_params = None
        self.boundary_conditions = []
        self.is_initialized = False
        self.current_field = None
        self._lock = threading.Lock()

        # 加载C++动态库
        self._load_cpp_library(cpp_lib_path)

        # 配置函数签名
        self._configure_function_signatures()

        logger.info("洋流模拟包装器初始化完成")

    def _load_cpp_library(self, lib_path: Optional[str]):
        """加载C++动态库"""
        if lib_path is None:
            # 自动查找库文件
            possible_paths = [
                "./build/lib/libcurrent_solver.so",
                "./build/lib/libcurrent_solver.dll",
                "../CppCore/build/lib/libcurrent_solver.so",
                "../CppCore/build/lib/libcurrent_solver.dll",
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    lib_path = path
                    break

            if lib_path is None:
                raise FileNotFoundError("找不到洋流模拟C++库文件")

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

        # 初始化求解器
        self.lib.initialize_current_solver.argtypes = []
        self.lib.initialize_current_solver.restype = c_int

        # 设置网格
        self.lib.set_grid_configuration.argtypes = [POINTER(CGridConfig)]
        self.lib.set_grid_configuration.restype = c_int

        # 设置模拟参数
        self.lib.set_simulation_parameters.argtypes = [POINTER(CSimParams)]
        self.lib.set_simulation_parameters.restype = c_int

        # 设置边界条件
        self.lib.set_boundary_condition.argtypes = [POINTER(CBoundaryCondition)]
        self.lib.set_boundary_condition.restype = c_int

        # 设置初始条件
        self.lib.set_initial_velocity_field.argtypes = [
            POINTER(c_double), POINTER(c_double), POINTER(c_double)
        ]
        self.lib.set_initial_velocity_field.restype = c_int

        # 设置外力
        self.lib.set_external_forces.argtypes = [
            POINTER(c_double), POINTER(c_double)  # wind_stress_x, wind_stress_y
        ]
        self.lib.set_external_forces.restype = c_int

        # 执行时间步
        self.lib.advance_time_step.argtypes = []
        self.lib.advance_time_step.restype = c_int

        # 运行模拟
        self.lib.run_simulation.argtypes = []
        self.lib.run_simulation.restype = c_int

        # 获取结果
        self.lib.get_current_field.argtypes = [POINTER(CCurrentField)]
        self.lib.get_current_field.restype = c_int

        # 获取诊断信息
        self.lib.get_kinetic_energy.argtypes = []
        self.lib.get_kinetic_energy.restype = c_double

        self.lib.get_max_velocity.argtypes = []
        self.lib.get_max_velocity.restype = c_double

        self.lib.get_residual.argtypes = []
        self.lib.get_residual.restype = c_double

        # 清理函数
        self.lib.cleanup_current_solver.argtypes = []
        self.lib.cleanup_current_solver.restype = None

    def initialize(self, grid_config: GridConfiguration,
                   sim_params: SimulationParameters) -> bool:
        """
        初始化洋流求解器
        
        Args:
            grid_config: 网格配置
            sim_params: 模拟参数
            
        Returns:
            是否初始化成功
        """
        if self.lib is None:
            logger.error("C++库未加载")
            return False

        try:
            # 初始化求解器
            result = self.lib.initialize_current_solver()
            if result != 0:
                logger.error("求解器初始化失败")
                return False

            # 设置网格配置
            c_grid = self._python_to_c_grid(grid_config)
            result = self.lib.set_grid_configuration(byref(c_grid))
            if result != 0:
                logger.error("设置网格配置失败")
                return False

            # 设置模拟参数
            c_params = self._python_to_c_params(sim_params)
            result = self.lib.set_simulation_parameters(byref(c_params))
            if result != 0:
                logger.error("设置模拟参数失败")
                return False

            self.grid_config = grid_config
            self.sim_params = sim_params
            self.is_initialized = True

            logger.info("洋流求解器初始化成功")
            return True

        except Exception as e:
            logger.error(f"初始化异常: {e}")
            return False

    def set_boundary_conditions(self, boundary_conditions: List[BoundaryCondition]) -> bool:
        """
        设置边界条件
        
        Args:
            boundary_conditions: 边界条件列表
            
        Returns:
            是否设置成功
        """
        if not self.is_initialized:
            logger.error("求解器未初始化")
            return False

        try:
            for bc in boundary_conditions:
                c_bc = CBoundaryCondition()
                c_bc.boundary_type = bc.boundary_type.value
                c_bc.value = bc.value
                c_bc.location = bc.location.encode('utf-8')

                result = self.lib.set_boundary_condition(byref(c_bc))
                if result != 0:
                    logger.error(f"设置边界条件失败: {bc.location}")
                    return False

            self.boundary_conditions = boundary_conditions
            logger.info(f"成功设置 {len(boundary_conditions)} 个边界条件")
            return True

        except Exception as e:
            logger.error(f"设置边界条件异常: {e}")
            return False

    def set_initial_velocity_field(self, u: np.ndarray, v: np.ndarray,
                                   w: Optional[np.ndarray] = None) -> bool:
        """
        设置初始速度场
        
        Args:
            u: x方向速度场 [nz, ny, nx]
            v: y方向速度场 [nz, ny, nx]
            w: z方向速度场 [nz, ny, nx] (可选)
            
        Returns:
            是否设置成功
        """
        if not self.is_initialized:
            logger.error("求解器未初始化")
            return False

        try:
            # 验证数组维度
            expected_shape = (self.grid_config.nz, self.grid_config.ny, self.grid_config.nx)
            if u.shape != expected_shape or v.shape != expected_shape:
                raise ValueError(f"速度场维度不匹配，期望: {expected_shape}")

            # 如果没有提供w分量，设置为零
            if w is None:
                w = np.zeros_like(u)
            elif w.shape != expected_shape:
                raise ValueError(f"w分量维度不匹配，期望: {expected_shape}")

            # 确保数组是连续的
            u = np.ascontiguousarray(u, dtype=np.float64)
            v = np.ascontiguousarray(v, dtype=np.float64)
            w = np.ascontiguousarray(w, dtype=np.float64)

            # 调用C++函数
            result = self.lib.set_initial_velocity_field(
                u.ctypes.data_as(POINTER(c_double)),
                v.ctypes.data_as(POINTER(c_double)),
                w.ctypes.data_as(POINTER(c_double))
            )

            if result == 0:
                logger.info("初始速度场设置成功")
                return True
            else:
                logger.error("设置初始速度场失败")
                return False

        except Exception as e:
            logger.error(f"设置初始速度场异常: {e}")
            return False

    def set_external_forces(self, force_field: ForceField) -> bool:
        """
        设置外力场
        
        Args:
            force_field: 外力场对象
            
        Returns:
            是否设置成功
        """
        if not self.is_initialized:
            logger.error("求解器未初始化")
            return False

        try:
            # 准备风应力数据
            grid_size = self.grid_config.nx * self.grid_config.ny

            if force_field.wind_stress_x is not None:
                wind_stress_x = np.ascontiguousarray(
                    force_field.wind_stress_x.flatten(), dtype=np.float64
                )
            else:
                wind_stress_x = np.zeros(grid_size, dtype=np.float64)

            if force_field.wind_stress_y is not None:
                wind_stress_y = np.ascontiguousarray(
                    force_field.wind_stress_y.flatten(), dtype=np.float64
                )
            else:
                wind_stress_y = np.zeros(grid_size, dtype=np.float64)

            # 调用C++函数
            result = self.lib.set_external_forces(
                wind_stress_x.ctypes.data_as(POINTER(c_double)),
                wind_stress_y.ctypes.data_as(POINTER(c_double))
            )

            if result == 0:
                logger.info("外力场设置成功")
                return True
            else:
                logger.error("设置外力场失败")
                return False

        except Exception as e:
            logger.error(f"设置外力场异常: {e}")
            return False

    def advance_time_step(self) -> bool:
        """
        推进一个时间步
        
        Returns:
            是否成功
        """
        if not self.is_initialized:
            logger.error("求解器未初始化")
            return False

        try:
            result = self.lib.advance_time_step()
            return result == 0
        except Exception as e:
            logger.error(f"时间步推进异常: {e}")
            return False

    def run_simulation(self, callback_func: Optional[callable] = None,
                       callback_interval: int = 10) -> bool:
        """
        运行完整模拟
        
        Args:
            callback_func: 回调函数(可选)
            callback_interval: 回调间隔(时间步数)
            
        Returns:
            是否成功
        """
        if not self.is_initialized:
            logger.error("求解器未初始化")
            return False

        try:
            total_steps = int(self.sim_params.total_time / self.sim_params.time_step)
            logger.info(f"开始模拟，总时间步数: {total_steps}")

            start_time = time.time()

            for step in range(total_steps):
                success = self.advance_time_step()
                if not success:
                    logger.error(f"时间步 {step+1} 失败")
                    return False

                # 执行回调函数
                if callback_func and (step + 1) % callback_interval == 0:
                    try:
                        callback_func(step + 1, total_steps, self.get_diagnostics())
                    except Exception as e:
                        logger.warning(f"回调函数执行异常: {e}")

                # 进度报告
                if (step + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    progress = (step + 1) / total_steps * 100
                    logger.info(f"进度: {progress:.1f}% ({step+1}/{total_steps}), "
                                f"耗时: {elapsed:.1f}s")

            total_time = time.time() - start_time
            logger.info(f"模拟完成，总耗时: {total_time:.1f}s")
            return True

        except Exception as e:
            logger.error(f"模拟运行异常: {e}")
            return False

    def get_current_field(self) -> Optional[Dict[str, np.ndarray]]:
        """
        获取当前洋流场
        
        Returns:
            洋流场字典，包含u, v, w, p
        """
        if not self.is_initialized:
            logger.error("求解器未初始化")
            return None

        try:
            # 分配内存
            grid_size = self.grid_config.nx * self.grid_config.ny * self.grid_config.nz

            u_array = np.zeros(grid_size, dtype=np.float64)
            v_array = np.zeros(grid_size, dtype=np.float64)
            w_array = np.zeros(grid_size, dtype=np.float64)
            p_array = np.zeros(grid_size, dtype=np.float64)

            # 创建C++结构体
            c_field = CCurrentField()
            c_field.u = u_array.ctypes.data_as(POINTER(c_double))
            c_field.v = v_array.ctypes.data_as(POINTER(c_double))
            c_field.w = w_array.ctypes.data_as(POINTER(c_double))
            c_field.p = p_array.ctypes.data_as(POINTER(c_double))
            c_field.nx = self.grid_config.nx
            c_field.ny = self.grid_config.ny
            c_field.nz = self.grid_config.nz
            c_field.nt = 1

            # 调用C++函数
            result = self.lib.get_current_field(byref(c_field))

            if result != 0:
                logger.error("获取洋流场失败")
                return None

            # 重新整形为3D数组
            shape = (self.grid_config.nz, self.grid_config.ny, self.grid_config.nx)

            field_data = {
                'u': u_array.reshape(shape),
                'v': v_array.reshape(shape),
                'w': w_array.reshape(shape),
                'p': p_array.reshape(shape)
            }

            self.current_field = field_data
            return field_data

        except Exception as e:
            logger.error(f"获取洋流场异常: {e}")
            return None

    def get_diagnostics(self) -> Dict[str, float]:
        """
        获取诊断信息
        
        Returns:
            诊断信息字典
        """
        if not self.is_initialized:
            return {}

        try:
            diagnostics = {
                'kinetic_energy': self.lib.get_kinetic_energy(),
                'max_velocity': self.lib.get_max_velocity(),
                'residual': self.lib.get_residual()
            }
            return diagnostics
        except Exception as e:
            logger.error(f"获取诊断信息异常: {e}")
            return {}

    def calculate_vorticity(self) -> Optional[np.ndarray]:
        """
        计算涡度场
        
        Returns:
            涡度场 [nz, ny, nx]
        """
        field = self.get_current_field()
        if field is None:
            return None

        u = field['u']
        v = field['v']

        # 计算涡度 ζ = ∂v/∂x - ∂u/∂y
        dy, dx = self.grid_config.dy, self.grid_config.dx

        # 使用中心差分
        dvdx = np.gradient(v, dx, axis=2)  # ∂v/∂x
        dudy = np.gradient(u, dy, axis=1)  # ∂u/∂y

        vorticity = dvdx - dudy

        return vorticity

    def calculate_divergence(self) -> Optional[np.ndarray]:
        """
        计算散度场
        
        Returns:
            散度场 [nz, ny, nx]
        """
        field = self.get_current_field()
        if field is None:
            return None

        u = field['u']
        v = field['v']
        w = field['w']

        # 计算散度 ∇·v = ∂u/∂x + ∂v/∂y + ∂w/∂z
        dx, dy, dz = self.grid_config.dx, self.grid_config.dy, self.grid_config.dz

        dudx = np.gradient(u, dx, axis=2)  # ∂u/∂x
        dvdy = np.gradient(v, dy, axis=1)  # ∂v/∂y
        dwdz = np.gradient(w, dz, axis=0)  # ∂w/∂z

        divergence = dudx + dvdy + dwdz

        return divergence

    def export_field_data(self, filename: str, format: str = 'netcdf') -> bool:
        """
        导出场数据
        
        Args:
            filename: 输出文件名
            format: 输出格式 ('netcdf', 'hdf5', 'vtk')
            
        Returns:
            是否导出成功
        """
        field = self.get_current_field()
        if field is None:
            logger.error("无法获取洋流场数据")
            return False

        try:
            if format.lower() == 'netcdf':
                return self._export_netcdf(field, filename)
            elif format.lower() == 'hdf5':
                return self._export_hdf5(field, filename)
            elif format.lower() == 'vtk':
                return self._export_vtk(field, filename)
            else:
                logger.error(f"不支持的输出格式: {format}")
                return False
        except Exception as e:
            logger.error(f"导出场数据失败: {e}")
            return False

    def _export_netcdf(self, field: Dict[str, np.ndarray], filename: str) -> bool:
        """导出为NetCDF格式"""
        try:
            import netCDF4 as nc

            with nc.Dataset(filename, 'w') as dataset:
                # 创建维度
                dataset.createDimension('x', self.grid_config.nx)
                dataset.createDimension('y', self.grid_config.ny)
                dataset.createDimension('z', self.grid_config.nz)

                # 创建坐标变量
                x = dataset.createVariable('x', 'f8', ('x',))
                y = dataset.createVariable('y', 'f8', ('y',))
                z = dataset.createVariable('z', 'f8', ('z',))

                x[:] = np.linspace(self.grid_config.x_min, self.grid_config.x_max,
                                   self.grid_config.nx)
                y[:] = np.linspace(self.grid_config.y_min, self.grid_config.y_max,
                                   self.grid_config.ny)
                z[:] = np.linspace(self.grid_config.z_min, self.grid_config.z_max,
                                   self.grid_config.nz)

                # 创建变量
                for var_name, var_data in field.items():
                    var = dataset.createVariable(var_name, 'f8', ('z', 'y', 'x'))
                    var[:] = var_data

                    # 添加属性
                    if var_name in ['u', 'v', 'w']:
                        var.units = 'm/s'
                        var.long_name = f'Velocity component {var_name}'
                    elif var_name == 'p':
                        var.units = 'Pa'
                        var.long_name = 'Pressure'

                # 添加全局属性
                dataset.title = 'Ocean Current Simulation Results'
                dataset.source = 'C# Multi-language Ocean Current Simulation System'

            logger.info(f"场数据已导出到: {filename}")
            return True

        except ImportError:
            logger.error("需要安装netCDF4库")
            return False
        except Exception as e:
            logger.error(f"NetCDF导出失败: {e}")
            return False

    def _python_to_c_grid(self, grid_config: GridConfiguration) -> CGridConfig:
        """转换Python网格配置为C++结构体"""
        c_grid = CGridConfig()
        c_grid.nx = grid_config.nx
        c_grid.ny = grid_config.ny
        c_grid.nz = grid_config.nz
        c_grid.dx = grid_config.dx
        c_grid.dy = grid_config.dy
        c_grid.dz = grid_config.dz
        c_grid.x_min = grid_config.x_min
        c_grid.y_min = grid_config.y_min
        c_grid.z_min = grid_config.z_min
        c_grid.x_max = grid_config.x_max
        c_grid.y_max = grid_config.y_max
        c_grid.z_max = grid_config.z_max
        return c_grid

    def _python_to_c_params(self, sim_params: SimulationParameters) -> CSimParams:
        """转换Python模拟参数为C++结构体"""
        c_params = CSimParams()
        c_params.time_step = sim_params.time_step
        c_params.total_time = sim_params.total_time
        c_params.viscosity = sim_params.viscosity
        c_params.density = sim_params.density
        c_params.coriolis_parameter = sim_params.coriolis_parameter
        c_params.gravity = sim_params.gravity
        c_params.solver_method = sim_params.solver_method.value
        c_params.time_scheme = sim_params.time_scheme.value
        c_params.max_iterations = sim_params.max_iterations
        c_params.convergence_tolerance = sim_params.convergence_tolerance
        c_params.enable_nonlinear = 1 if sim_params.enable_nonlinear else 0
        c_params.enable_turbulence = 1 if sim_params.enable_turbulence else 0
        return c_params

    def cleanup(self):
        """清理资源"""
        if self.lib and self.is_initialized:
            self.lib.cleanup_current_solver()
            self.is_initialized = False
            logger.info("洋流求解器已清理")

    def __del__(self):
        """析构函数"""
        self.cleanup()


# 使用示例
if __name__ == "__main__":
    # 创建洋流模拟包装器
    simulator = CurrentSimulationWrapper()

    # 配置网格
    grid_config = GridConfiguration(
        nx=100, ny=80, nz=20,
        dx=1000.0, dy=1000.0, dz=10.0,
        x_min=0.0, y_min=0.0, z_min=0.0,
        x_max=100000.0, y_max=80000.0, z_max=200.0
    )

    # 配置模拟参数
    sim_params = SimulationParameters(
        time_step=300.0,
        total_time=86400.0,  # 1天
        viscosity=1e-6,
        density=1025.0,
        coriolis_parameter=1e-4,
        solver_method=SolverMethod.FINITE_DIFFERENCE,
        time_scheme=TimeScheme.RUNGE_KUTTA_4
    )

    # 初始化
    if not simulator.initialize(grid_config, sim_params):
        print("初始化失败")
        exit(1)

    # 设置边界条件
    boundary_conditions = [
        BoundaryCondition(BoundaryType.DIRICHLET, 0.0, 'north'),
        BoundaryCondition(BoundaryType.DIRICHLET, 0.0, 'south'),
        BoundaryCondition(BoundaryType.NEUMANN, 0.0, 'east'),
        BoundaryCondition(BoundaryType.NEUMANN, 0.0, 'west'),
    ]

    if not simulator.set_boundary_conditions(boundary_conditions):
        print("设置边界条件失败")
        exit(1)

    # 设置初始速度场
    u_init = np.random.randn(grid_config.nz, grid_config.ny, grid_config.nx) * 0.1
    v_init = np.random.randn(grid_config.nz, grid_config.ny, grid_config.nx) * 0.1
    w_init = np.zeros((grid_config.nz, grid_config.ny, grid_config.nx))

    if not simulator.set_initial_velocity_field(u_init, v_init, w_init):
        print("设置初始速度场失败")
        exit(1)

    # 设置风应力
    wind_stress_x = np.ones((grid_config.ny, grid_config.nx)) * 0.1
    wind_stress_y = np.zeros((grid_config.ny, grid_config.nx))

    force_field = ForceField(
        wind_stress_x=wind_stress_x,
        wind_stress_y=wind_stress_y
    )

    if not simulator.set_external_forces(force_field):
        print("设置外力场失败")
        exit(1)

    # 定义回调函数
    def progress_callback(step, total_steps, diagnostics):
        progress = step / total_steps * 100
        print(f"进度: {progress:.1f}%, 动能: {diagnostics.get('kinetic_energy', 0):.2e}, "
              f"最大速度: {diagnostics.get('max_velocity', 0):.3f} m/s")

    # 运行模拟
    print("开始洋流模拟...")
    success = simulator.run_simulation(progress_callback, callback_interval=50)

    if success:
        print("模拟完成!")

        # 获取最终结果
        final_field = simulator.get_current_field()
        if final_field:
            print(f"最大u速度: {np.max(final_field['u']):.3f} m/s")
            print(f"最大v速度: {np.max(final_field['v']):.3f} m/s")

            # 计算涡度
            vorticity = simulator.calculate_vorticity()
            if vorticity is not None:
                print(f"最大涡度: {np.max(np.abs(vorticity)):.2e} s⁻¹")

            # 导出结果
            simulator.export_field_data("ocean_current_results.nc", "netcdf")
    else:
        print("模拟失败!")

    # 清理
    simulator.cleanup()
    print("洋流模拟示例完成")