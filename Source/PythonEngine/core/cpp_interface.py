#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: cpp_interface.py
模块: Source.PythonEngine.core.cpp_interface
功能: C++高性能计算引擎的Python接口包装器。负责与C++动态库通信，实现粒子追踪、洋流场数值求解、平流扩散、性能采集等高性能任务的跨语言调用。
作者: beilsm
版本: v1.0.0
创建时间: 2025-06-28
最近更新: 2025-06-28
主要功能:
    - 跨平台动态库自动加载与签名绑定
    - 网格参数、仿真参数结构体对接
    - 支持异步粒子模拟、场求解与扩散模拟
    - 性能数据自动采集
    - 兼容未来C++算法接口扩展
较上一版改进:
    - 首发版，接口全量实现，结构体参数类型细致对齐
测试方法:
    见文件底部 `if __name__ == "__main__"` 区域
接口说明:
    - initialize(), simulate_particles(), solve_current_field(), solve_advection_diffusion(), get_performance_metrics(), cleanup()
"""

# ...后续为正文...


import ctypes
import numpy as np
import numpy.ctypeslib as npct
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import asyncio
import os
from pathlib import Path
import platform
import json

class CppInterface:
    """C++计算引擎接口包装器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化C++接口
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # C++库句柄
        self.lib = None
        self.is_initialized = False

        # 数据类型定义
        self._setup_ctypes()

        # 性能统计
        self.performance_stats = {
            "total_calls": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "last_error": None
        }

    def _setup_ctypes(self):
        """设置ctypes数据类型"""
        # 基础数据类型
        self.c_double_p = npct.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.c_float_p = npct.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        self.c_int_p = npct.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')

        # 结构体定义
        class GridParams(ctypes.Structure):
            """网格参数结构体"""
            _fields_ = [
                ("nx", ctypes.c_int),
                ("ny", ctypes.c_int),
                ("nz", ctypes.c_int),
                ("dx", ctypes.c_double),
                ("dy", ctypes.c_double),
                ("dz", ctypes.c_double),
                ("x_min", ctypes.c_double),
                ("y_min", ctypes.c_double),
                ("z_min", ctypes.c_double)
            ]

        class SimulationParams(ctypes.Structure):
            """仿真参数结构体"""
            _fields_ = [
                ("dt", ctypes.c_double),
                ("total_time", ctypes.c_double),
                ("num_particles", ctypes.c_int),
                ("diffusion_coeff", ctypes.c_double),
                ("viscosity", ctypes.c_double),
                ("enable_3d", ctypes.c_bool),
                ("enable_diffusion", ctypes.c_bool)
            ]

        class PerformanceConfig(ctypes.Structure):
            """性能配置结构体"""
            _fields_ = [
                ("num_threads", ctypes.c_int),
                ("use_vectorization", ctypes.c_bool),
                ("use_gpu", ctypes.c_bool),
                ("memory_pool_size", ctypes.c_size_t),
                ("chunk_size", ctypes.c_int)
            ]

        self.GridParams = GridParams
        self.SimulationParams = SimulationParams
        self.PerformanceConfig = PerformanceConfig

    async def initialize(self) -> bool:
        """
        初始化C++计算引擎
        
        Returns:
            初始化是否成功
        """
        try:
            # 确定库文件路径
            library_path = self._get_library_path()
            if not library_path.exists():
                self.logger.warning(f"C++库文件不存在: {library_path}")
                return False

            # 加载动态库
            self.lib = ctypes.CDLL(str(library_path))

            # 设置函数签名
            self._setup_function_signatures()

            # 初始化C++引擎
            perf_config = self.PerformanceConfig()
            perf_config.num_threads = self.config.get("num_threads", -1)  # -1表示自动检测
            perf_config.use_vectorization = self.config.get("use_vectorization", True)
            perf_config.use_gpu = self.config.get("enable_cuda", False)
            perf_config.memory_pool_size = self.config.get("memory_pool_size", 1024 * 1024 * 1024)  # 1GB
            perf_config.chunk_size = self.config.get("chunk_size", 1000)

            result = self.lib.cpp_engine_initialize(ctypes.byref(perf_config))

            if result == 0:
                self.is_initialized = True
                self.logger.info("C++计算引擎初始化成功")

                # 获取引擎信息
                engine_info = self._get_engine_info()
                self.logger.info(f"C++引擎信息: {engine_info}")

                return True
            else:
                self.logger.error(f"C++引擎初始化失败，错误代码: {result}")
                return False

        except Exception as e:
            self.logger.error(f"加载C++库失败: {e}")
            self.performance_stats["last_error"] = str(e)
            return False

    def _get_library_path(self) -> Path:
        """获取动态库路径"""
        library_dir = Path(self.config.get("library_path", "./Build/Release/Cpp"))

        if platform.system() == "Windows":
            library_name = "OceanSimCore.dll"
        elif platform.system() == "Darwin":
            library_name = "libOceanSimCore.dylib"
        else:
            library_name = "libOceanSimCore.so"

        return library_dir / library_name

    def _setup_function_signatures(self):
        """设置C++函数签名"""
        if not self.lib:
            return

        # 引擎管理函数
        self.lib.cpp_engine_initialize.argtypes = [ctypes.POINTER(self.PerformanceConfig)]
        self.lib.cpp_engine_initialize.restype = ctypes.c_int

        self.lib.cpp_engine_cleanup.argtypes = []
        self.lib.cpp_engine_cleanup.restype = ctypes.c_int

        self.lib.cpp_engine_get_info.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.cpp_engine_get_info.restype = ctypes.c_int

        # 粒子仿真函数
        self.lib.cpp_particle_simulation.argtypes = [
            self.c_double_p,  # u velocity field
            self.c_double_p,  # v velocity field
            self.c_double_p,  # w velocity field (可选)
            self.c_double_p,  # particle_x (输入/输出)
            self.c_double_p,  # particle_y (输入/输出)
            self.c_double_p,  # particle_z (输入/输出，可选)
            ctypes.POINTER(self.GridParams),
            ctypes.POINTER(self.SimulationParams)
        ]
        self.lib.cpp_particle_simulation.restype = ctypes.c_int

        # 洋流场求解函数
        self.lib.cpp_current_field_solver.argtypes = [
            self.c_double_p,  # input field
            self.c_double_p,  # output field
            ctypes.POINTER(self.GridParams),
            ctypes.c_double,  # time step
            ctypes.c_int      # solver type
        ]
        self.lib.cpp_current_field_solver.restype = ctypes.c_int

        # 平流扩散求解函数
        self.lib.cpp_advection_diffusion_solver.argtypes = [
            self.c_double_p,  # concentration field (输入/输出)
            self.c_double_p,  # u velocity
            self.c_double_p,  # v velocity
            ctypes.POINTER(self.GridParams),
            ctypes.c_double,  # diffusion coefficient
            ctypes.c_double,  # time step
            ctypes.c_int      # boundary condition type
        ]
        self.lib.cpp_advection_diffusion_solver.restype = ctypes.c_int

        # 性能分析函数
        self.lib.cpp_get_performance_metrics.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.cpp_get_performance_metrics.restype = ctypes.c_int

    def _get_engine_info(self) -> Dict[str, Any]:
        """获取C++引擎信息"""
        if not self.is_initialized:
            return {}

        try:
            buffer_size = 1024
            buffer = ctypes.create_string_buffer(buffer_size)
            result = self.lib.cpp_engine_get_info(buffer, buffer_size)

            if result == 0:
                info_str = buffer.value.decode('utf-8')
                return json.loads(info_str)
            else:
                return {"error": f"获取引擎信息失败，错误代码: {result}"}

        except Exception as e:
            self.logger.warning(f"获取引擎信息失败: {e}")
            return {"error": str(e)}

    async def simulate_particles(
            self,
            velocity_field: Dict[str, np.ndarray],
            initial_positions: np.ndarray,
            grid_params: Dict[str, Any],
            simulation_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        粒子追踪仿真
        
        Args:
            velocity_field: 速度场 {"u": array, "v": array, "w": array(可选)}
            initial_positions: 初始位置 [[x1,y1,z1], [x2,y2,z2], ...]
            grid_params: 网格参数
            simulation_params: 仿真参数
            
        Returns:
            粒子轨迹数组
        """
        if not self.is_initialized:
            raise RuntimeError("C++引擎未初始化")

        try:
            import time
            start_time = time.time()

            # 准备输入数据
            u_field = np.ascontiguousarray(velocity_field["u"], dtype=np.float64)
            v_field = np.ascontiguousarray(velocity_field["v"], dtype=np.float64)
            w_field = np.ascontiguousarray(
                velocity_field.get("w", np.zeros_like(u_field)),
                dtype=np.float64
            )

            # 粒子位置
            num_particles = initial_positions.shape[0]
            particle_x = np.ascontiguousarray(initial_positions[:, 0], dtype=np.float64)
            particle_y = np.ascontiguousarray(initial_positions[:, 1], dtype=np.float64)
            particle_z = np.ascontiguousarray(
                initial_positions[:, 2] if initial_positions.shape[1] > 2
                else np.zeros(num_particles),
                dtype=np.float64
            )

            # 设置网格参数
            grid_params_c = self.GridParams()
            grid_params_c.nx = grid_params["nx"]
            grid_params_c.ny = grid_params["ny"]
            grid_params_c.nz = grid_params.get("nz", 1)
            grid_params_c.dx = grid_params["dx"]
            grid_params_c.dy = grid_params["dy"]
            grid_params_c.dz = grid_params.get("dz", 1.0)
            grid_params_c.x_min = grid_params["x_min"]
            grid_params_c.y_min = grid_params["y_min"]
            grid_params_c.z_min = grid_params.get("z_min", 0.0)

            # 设置仿真参数
            sim_params_c = self.SimulationParams()
            sim_params_c.dt = simulation_params["dt"]
            sim_params_c.total_time = simulation_params["total_time"]
            sim_params_c.num_particles = num_particles
            sim_params_c.diffusion_coeff = simulation_params.get("diffusion_coeff", 0.0)
            sim_params_c.viscosity = simulation_params.get("viscosity", 1e-6)
            sim_params_c.enable_3d = simulation_params.get("enable_3d", False)
            sim_params_c.enable_diffusion = simulation_params.get("enable_diffusion", False)

            # 调用C++函数
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.lib.cpp_particle_simulation,
                u_field, v_field, w_field,
                particle_x, particle_y, particle_z,
                ctypes.byref(grid_params_c),
                ctypes.byref(sim_params_c)
            )

            # 更新统计信息
            elapsed_time = time.time() - start_time
            self.performance_stats["total_calls"] += 1
            self.performance_stats["total_time"] += elapsed_time
            self.performance_stats["average_time"] = (
                    self.performance_stats["total_time"] / self.performance_stats["total_calls"]
            )

            if result != 0:
                error_msg = f"粒子仿真失败，错误代码: {result}"
                self.logger.error(error_msg)
                self.performance_stats["last_error"] = error_msg
                raise RuntimeError(error_msg)

            # 返回结果
            if sim_params_c.enable_3d:
                final_positions = np.column_stack([particle_x, particle_y, particle_z])
            else:
                final_positions = np.column_stack([particle_x, particle_y])

            self.logger.debug(f"粒子仿真完成，耗时: {elapsed_time:.3f}s")
            return final_positions

        except Exception as e:
            error_msg = f"粒子仿真异常: {e}"
            self.logger.error(error_msg)
            self.performance_stats["last_error"] = error_msg
            raise

    async def solve_current_field(
            self,
            input_field: np.ndarray,
            grid_params: Dict[str, Any],
            time_step: float,
            solver_type: int = 0
    ) -> np.ndarray:
        """
        洋流场求解
        
        Args:
            input_field: 输入场
            grid_params: 网格参数
            time_step: 时间步长
            solver_type: 求解器类型
            
        Returns:
            求解后的场
        """
        if not self.is_initialized:
            raise RuntimeError("C++引擎未初始化")

        try:
            import time
            start_time = time.time()

            # 准备数据
            input_array = np.ascontiguousarray(input_field, dtype=np.float64)
            output_array = np.zeros_like(input_array, dtype=np.float64)

            # 设置网格参数
            grid_params_c = self.GridParams()
            grid_params_c.nx = grid_params["nx"]
            grid_params_c.ny = grid_params["ny"]
            grid_params_c.nz = grid_params.get("nz", 1)
            grid_params_c.dx = grid_params["dx"]
            grid_params_c.dy = grid_params["dy"]
            grid_params_c.dz = grid_params.get("dz", 1.0)

            # 调用C++函数
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.lib.cpp_current_field_solver,
                input_array, output_array,
                ctypes.byref(grid_params_c),
                ctypes.c_double(time_step),
                ctypes.c_int(solver_type)
            )

            elapsed_time = time.time() - start_time
            self.performance_stats["total_calls"] += 1
            self.performance_stats["total_time"] += elapsed_time
            self.performance_stats["average_time"] = (
                    self.performance_stats["total_time"] / self.performance_stats["total_calls"]
            )

            if result != 0:
                error_msg = f"洋流场求解失败，错误代码: {result}"
                self.logger.error(error_msg)
                self.performance_stats["last_error"] = error_msg
                raise RuntimeError(error_msg)

            self.logger.debug(f"洋流场求解完成，耗时: {elapsed_time:.3f}s")
            return output_array

        except Exception as e:
            error_msg = f"洋流场求解异常: {e}"
            self.logger.error(error_msg)
            self.performance_stats["last_error"] = error_msg
            raise

    async def solve_advection_diffusion(
            self,
            concentration_field: np.ndarray,
            velocity_field: Dict[str, np.ndarray],
            grid_params: Dict[str, Any],
            diffusion_coeff: float,
            time_step: float,
            boundary_condition: int = 0
    ) -> np.ndarray:
        """
        平流扩散求解
        
        Args:
            concentration_field: 浓度场
            velocity_field: 速度场
            grid_params: 网格参数
            diffusion_coeff: 扩散系数
            time_step: 时间步长
            boundary_condition: 边界条件类型
            
        Returns:
            更新后的浓度场
        """
        if not self.is_initialized:
            raise RuntimeError("C++引擎未初始化")

        try:
            import time
            start_time = time.time()

            # 准备数据
            concentration = np.ascontiguousarray(concentration_field, dtype=np.float64)
            u_field = np.ascontiguousarray(velocity_field["u"], dtype=np.float64)
            v_field = np.ascontiguousarray(velocity_field["v"], dtype=np.float64)

            # 设置网格参数
            grid_params_c = self.GridParams()
            grid_params_c.nx = grid_params["nx"]
            grid_params_c.ny = grid_params["ny"]
            grid_params_c.nz = grid_params.get("nz", 1)
            grid_params_c.dx = grid_params["dx"]
            grid_params_c.dy = grid_params["dy"]
            grid_params_c.dz = grid_params.get("dz", 1.0)

            # 调用C++函数
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.lib.cpp_advection_diffusion_solver,
                concentration, u_field, v_field,
                ctypes.byref(grid_params_c),
                ctypes.c_double(diffusion_coeff),
                ctypes.c_double(time_step),
                ctypes.c_int(boundary_condition)
            )

            elapsed_time = time.time() - start_time
            self.performance_stats["total_calls"] += 1
            self.performance_stats["total_time"] += elapsed_time
            self.performance_stats["average_time"] = (
                    self.performance_stats["total_time"] / self.performance_stats["total_calls"]
            )

            if result != 0:
                error_msg = f"平流扩散求解失败，错误代码: {result}"
                self.logger.error(error_msg)
                self.performance_stats["last_error"] = error_msg
                raise RuntimeError(error_msg)

            self.logger.debug(f"平流扩散求解完成，耗时: {elapsed_time:.3f}s")
            return concentration

        except Exception as e:
            error_msg = f"平流扩散求解异常: {e}"
            self.logger.error(error_msg)
            self.performance_stats["last_error"] = error_msg
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        cpp_metrics = {}

        if self.is_initialized:
            try:
                buffer_size = 2048
                buffer = ctypes.create_string_buffer(buffer_size)
                result = self.lib.cpp_get_performance_metrics(buffer, buffer_size)

                if result == 0:
                    metrics_str = buffer.value.decode('utf-8')
                    cpp_metrics = json.loads(metrics_str)
            except Exception as e:
                self.logger.warning(f"获取C++性能指标失败: {e}")

        return {
            "python_interface": self.performance_stats,
            "cpp_engine": cpp_metrics,
            "is_available": self.is_available()
        }

    def is_available(self) -> bool:
        """检查C++引擎是否可用"""
        return self.is_initialized and self.lib is not None

    async def cleanup(self):
        """清理资源"""
        if self.is_initialized and self.lib:
            try:
                result = self.lib.cpp_engine_cleanup()
                if result == 0:
                    self.logger.info("C++引擎清理完成")
                else:
                    self.logger.warning(f"C++引擎清理失败，错误代码: {result}")
            except Exception as e:
                self.logger.error(f"C++引擎清理异常: {e}")

            self.is_initialized = False
            self.lib = None

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    # 伪造 config 测试路径
    config = {"library_path": "./Build/Release/Cpp", "num_threads": 2}
    cpp = CppInterface(config)
    asyncio.run(cpp.initialize())
    print("Cpp引擎可用?", cpp.is_available())
    # 你可以继续补充实际测试数据
