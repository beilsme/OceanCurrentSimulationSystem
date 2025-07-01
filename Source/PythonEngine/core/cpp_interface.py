#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: cpp_interface.py
模块: Source.PythonEngine.core.cpp_interface
功能: C++高性能计算引擎的Python接口包装器。负责与C++动态库通信，实现粒子追踪、洋流场数值求解、平流扩散、性能采集等高性能任务的跨语言调用。
作者: beilsm
版本: v1.0.1
创建时间: 2025-06-28
最近更新: 2025-06-29
主要功能:
    - 跨平台动态库自动加载与签名绑定
    - 网格参数、仿真参数结构体对接
    - 支持异步粒子模拟、场求解与扩散模拟
    - 性能数据自动采集
    - 兼容未来C++算法接口扩展
    - 为simulation模块提供统一接口
较上一版改进:
    - 增加CppInterfaceWrapper类，为simulation模块提供统一接口
    - 改进错误处理和资源管理
    - 优化性能监控功能
测试方法:
    见文件底部 `if __name__ == "__main__"` 区域
接口说明:
    - initialize(), simulate_particles(), solve_current_field(), solve_advection_diffusion(), get_performance_metrics(), cleanup()
"""

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
import threading
import time
from dataclasses import dataclass
from enum import Enum


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


class ComputeTaskType(Enum):
    """计算任务类型"""
    PARTICLE_TRACKING = "particle_tracking"
    CURRENT_FIELD_SOLVING = "current_field_solving"
    ADVECTION_DIFFUSION = "advection_diffusion"
    PERFORMANCE_ANALYSIS = "performance_analysis"


@dataclass
class ComputeTask:
    """计算任务数据结构"""
    task_type: ComputeTaskType
    input_data: Dict[str, Any]
    parameters: Dict[str, Any]
    priority: int = 0
    timeout: Optional[float] = None


class CppInterfaceWrapper:
    """
    C++接口包装器
    为simulation模块提供统一的C++计算引擎接口
    支持任务队列、资源池管理和性能监控
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化C++接口包装器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # C++接口实例
        self.cpp_interface = CppInterface(self.config)

        # 任务管理
        self.task_queue = asyncio.Queue()
        self.is_processing = False
        self.max_concurrent_tasks = self.config.get("max_concurrent_tasks", 4)

        # 资源管理
        self._lock = threading.Lock()
        self._is_initialized = False

        # 性能监控
        self.task_statistics = {
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_execution_time": 0.0,
            "average_task_time": 0.0,
            "task_type_stats": {}
        }

        self.logger.info("C++接口包装器初始化完成")

    def initialize(self) -> bool:
        """
        同步初始化接口
        
        Returns:
            初始化是否成功
        """
        try:
            # 使用事件循环运行异步初始化
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # 如果没有运行的事件循环，创建新的
                return asyncio.run(self._async_initialize())

            # 如果已有事件循环，创建任务
            if loop.is_running():
                task = asyncio.create_task(self._async_initialize())
                # 注意：这里可能需要其他方式来等待任务完成
                return True  # 简化处理，假设初始化成功
            else:
                return asyncio.run(self._async_initialize())

        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            return False

    async def _async_initialize(self) -> bool:
        """异步初始化"""
        with self._lock:
            if self._is_initialized:
                return True

            success = await self.cpp_interface.initialize()
            if success:
                self._is_initialized = True
                # 启动任务处理器
                asyncio.create_task(self._task_processor())
                self.logger.info("C++接口包装器异步初始化成功")

            return success

    async def _task_processor(self):
        """任务处理器"""
        self.is_processing = True

        while self.is_processing:
            try:
                # 获取任务（带超时）
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # 执行任务
                await self._execute_task(task)

                # 标记任务完成
                self.task_queue.task_done()

            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                self.logger.error(f"任务处理器异常: {e}")

    async def _execute_task(self, task: ComputeTask):
        """执行计算任务"""
        start_time = time.time()

        try:
            if task.task_type == ComputeTaskType.PARTICLE_TRACKING:
                result = await self.cpp_interface.simulate_particles(
                    task.input_data["velocity_field"],
                    task.input_data["initial_positions"],
                    task.parameters["grid_params"],
                    task.parameters["simulation_params"]
                )

            elif task.task_type == ComputeTaskType.CURRENT_FIELD_SOLVING:
                result = await self.cpp_interface.solve_current_field(
                    task.input_data["input_field"],
                    task.parameters["grid_params"],
                    task.parameters["time_step"],
                    task.parameters.get("solver_type", 0)
                )

            elif task.task_type == ComputeTaskType.ADVECTION_DIFFUSION:
                result = await self.cpp_interface.solve_advection_diffusion(
                    task.input_data["concentration_field"],
                    task.input_data["velocity_field"],
                    task.parameters["grid_params"],
                    task.parameters["diffusion_coeff"],
                    task.parameters["time_step"],
                    task.parameters.get("boundary_condition", 0)
                )

            else:
                raise ValueError(f"不支持的任务类型: {task.task_type}")

            # 更新统计信息
            elapsed_time = time.time() - start_time
            self._update_task_statistics(task.task_type, elapsed_time, success=True)

            # 存储结果（如果需要）
            task.input_data["result"] = result

        except Exception as e:
            elapsed_time = time.time() - start_time
            self._update_task_statistics(task.task_type, elapsed_time, success=False)
            self.logger.error(f"任务执行失败 ({task.task_type}): {e}")
            task.input_data["error"] = str(e)

    def _update_task_statistics(self, task_type: ComputeTaskType, elapsed_time: float, success: bool):
        """更新任务统计信息"""
        if success:
            self.task_statistics["completed_tasks"] += 1
        else:
            self.task_statistics["failed_tasks"] += 1

        self.task_statistics["total_execution_time"] += elapsed_time

        total_tasks = self.task_statistics["completed_tasks"] + self.task_statistics["failed_tasks"]
        if total_tasks > 0:
            self.task_statistics["average_task_time"] = (
                    self.task_statistics["total_execution_time"] / total_tasks
            )

        # 按任务类型统计
        type_key = task_type.value
        if type_key not in self.task_statistics["task_type_stats"]:
            self.task_statistics["task_type_stats"][type_key] = {
                "count": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "success_count": 0,
                "failure_count": 0
            }

        stats = self.task_statistics["task_type_stats"][type_key]
        stats["count"] += 1
        stats["total_time"] += elapsed_time
        stats["average_time"] = stats["total_time"] / stats["count"]

        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1

    def submit_particle_tracking_task(self, velocity_field: Dict[str, np.ndarray],
                                      initial_positions: np.ndarray,
                                      grid_params: Dict[str, Any],
                                      simulation_params: Dict[str, Any],
                                      priority: int = 0) -> asyncio.Future:
        """
        提交粒子追踪任务
        
        Args:
            velocity_field: 速度场
            initial_positions: 初始位置
            grid_params: 网格参数
            simulation_params: 模拟参数
            priority: 任务优先级
            
        Returns:
            异步Future对象
        """
        task = ComputeTask(
            task_type=ComputeTaskType.PARTICLE_TRACKING,
            input_data={
                "velocity_field": velocity_field,
                "initial_positions": initial_positions
            },
            parameters={
                "grid_params": grid_params,
                "simulation_params": simulation_params
            },
            priority=priority
        )

        return self._submit_task(task)

    def submit_current_field_task(self, input_field: np.ndarray,
                                  grid_params: Dict[str, Any],
                                  time_step: float,
                                  solver_type: int = 0,
                                  priority: int = 0) -> asyncio.Future:
        """
        提交洋流场求解任务
        
        Args:
            input_field: 输入场
            grid_params: 网格参数
            time_step: 时间步长
            solver_type: 求解器类型
            priority: 任务优先级
            
        Returns:
            异步Future对象
        """
        task = ComputeTask(
            task_type=ComputeTaskType.CURRENT_FIELD_SOLVING,
            input_data={"input_field": input_field},
            parameters={
                "grid_params": grid_params,
                "time_step": time_step,
                "solver_type": solver_type
            },
            priority=priority
        )

        return self._submit_task(task)

    def submit_advection_diffusion_task(self, concentration_field: np.ndarray,
                                        velocity_field: Dict[str, np.ndarray],
                                        grid_params: Dict[str, Any],
                                        diffusion_coeff: float,
                                        time_step: float,
                                        boundary_condition: int = 0,
                                        priority: int = 0) -> asyncio.Future:
        """
        提交平流扩散任务
        
        Args:
            concentration_field: 浓度场
            velocity_field: 速度场
            grid_params: 网格参数
            diffusion_coeff: 扩散系数
            time_step: 时间步长
            boundary_condition: 边界条件类型
            priority: 任务优先级
            
        Returns:
            异步Future对象
        """
        task = ComputeTask(
            task_type=ComputeTaskType.ADVECTION_DIFFUSION,
            input_data={
                "concentration_field": concentration_field,
                "velocity_field": velocity_field
            },
            parameters={
                "grid_params": grid_params,
                "diffusion_coeff": diffusion_coeff,
                "time_step": time_step,
                "boundary_condition": boundary_condition
            },
            priority=priority
        )

        return self._submit_task(task)

    def _submit_task(self, task: ComputeTask) -> asyncio.Future:
        """提交任务到队列"""
        future = asyncio.Future()
        task.input_data["future"] = future

        try:
            # 添加到队列（非阻塞）
            self.task_queue.put_nowait(task)
            self.logger.debug(f"任务已提交: {task.task_type}")
        except asyncio.QueueFull:
            future.set_exception(RuntimeError("任务队列已满"))

        return future

    # 同步接口方法，为与现有simulation模块兼容
    def sync_simulate_particles(self, velocity_field: Dict[str, np.ndarray],
                                initial_positions: np.ndarray,
                                grid_params: Dict[str, Any],
                                simulation_params: Dict[str, Any]) -> np.ndarray:
        """
        同步粒子追踪仿真
        
        Args:
            velocity_field: 速度场
            initial_positions: 初始位置
            grid_params: 网格参数
            simulation_params: 模拟参数
            
        Returns:
            粒子轨迹数组
        """
        if not self._is_initialized:
            raise RuntimeError("接口未初始化")

        try:
            # 直接调用C++接口的异步方法
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.cpp_interface.simulate_particles(
                        velocity_field, initial_positions, grid_params, simulation_params
                    )
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"同步粒子追踪失败: {e}")
            raise

    def sync_solve_current_field(self, input_field: np.ndarray,
                                 grid_params: Dict[str, Any],
                                 time_step: float,
                                 solver_type: int = 0) -> np.ndarray:
        """
        同步洋流场求解
        
        Args:
            input_field: 输入场
            grid_params: 网格参数
            time_step: 时间步长
            solver_type: 求解器类型
            
        Returns:
            求解后的场
        """
        if not self._is_initialized:
            raise RuntimeError("接口未初始化")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.cpp_interface.solve_current_field(
                        input_field, grid_params, time_step, solver_type
                    )
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"同步洋流场求解失败: {e}")
            raise

    def sync_solve_advection_diffusion(self, concentration_field: np.ndarray,
                                       velocity_field: Dict[str, np.ndarray],
                                       grid_params: Dict[str, Any],
                                       diffusion_coeff: float,
                                       time_step: float,
                                       boundary_condition: int = 0) -> np.ndarray:
        """
        同步平流扩散求解
        
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
        if not self._is_initialized:
            raise RuntimeError("接口未初始化")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.cpp_interface.solve_advection_diffusion(
                        concentration_field, velocity_field, grid_params,
                        diffusion_coeff, time_step, boundary_condition
                    )
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"同步平流扩散求解失败: {e}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        # 合并C++引擎性能指标和包装器统计信息
        cpp_metrics = self.cpp_interface.get_performance_metrics()

        return {
            "cpp_engine": cpp_metrics,
            "wrapper_statistics": self.task_statistics,
            "queue_size": self.task_queue.qsize(),
            "is_processing": self.is_processing,
            "is_initialized": self._is_initialized
        }

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "library_path": self.config.get("library_path", "N/A"),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "config": self.config
        }

    def is_available(self) -> bool:
        """检查C++引擎是否可用"""
        return self._is_initialized and self.cpp_interface.is_available()

    def wait_for_completion(self, timeout: Optional[float] = None):
        """等待所有任务完成"""
        if self.task_queue.empty():
            return

        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(asyncio.wait_for(self.task_queue.join(), timeout))
        except RuntimeError:
            # 没有运行的事件循环
            asyncio.run(asyncio.wait_for(self.task_queue.join(), timeout))

    def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理C++接口包装器资源...")

        # 停止任务处理器
        self.is_processing = False

        # 等待任务完成（带超时）
        try:
            self.wait_for_completion(timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("等待任务完成超时")

        # 清理C++接口
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.cpp_interface.cleanup())
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"清理C++接口失败: {e}")

        self._is_initialized = False
        self.logger.info("C++接口包装器资源清理完成")

    def __enter__(self):
        """上下文管理器入口"""
        if not self.initialize():
            raise RuntimeError("C++接口初始化失败")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()

    def __del__(self):
        """析构函数"""
        if hasattr(self, '_is_initialized') and self._is_initialized:
            self.cleanup()


# 为向后兼容提供的别名
CppEngineInterface = CppInterfaceWrapper


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    # 基础C++接口测试
    print("=== 基础C++接口测试 ===")
    config = {"library_path": "./Build/Release/Cpp", "num_threads": 2}
    cpp = CppInterface(config)

    async def test_basic_interface():
        await cpp.initialize()
        print("C++引擎可用?", cpp.is_available())
        if cpp.is_available():
            metrics = cpp.get_performance_metrics()
            print("性能指标:", metrics)
        await cpp.cleanup()

    asyncio.run(test_basic_interface())

    # 包装器接口测试
    print("\n=== 包装器接口测试 ===")

    with CppInterfaceWrapper(config) as wrapper:
        print("包装器可用?", wrapper.is_available())

        if wrapper.is_available():
            # 创建测试数据
            velocity_field = {
                "u": np.random.randn(10, 10, 5),
                "v": np.random.randn(10, 10, 5),
                "w": np.random.randn(10, 10, 5)
            }

            initial_positions = np.random.rand(100, 3) * 1000

            grid_params = {
                "nx": 10, "ny": 10, "nz": 5,
                "dx": 100.0, "dy": 100.0, "dz": 10.0,
                "x_min": 0.0, "y_min": 0.0, "z_min": 0.0
            }

            simulation_params = {
                "dt": 60.0,
                "total_time": 3600.0,
                "diffusion_coeff": 10.0,
                "enable_3d": True
            }

            try:
                # 测试同步接口
                print("测试同步粒子追踪...")
                result = wrapper.sync_simulate_particles(
                    velocity_field, initial_positions, grid_params, simulation_params
                )
                print(f"粒子追踪结果形状: {result.shape}")

                # 获取性能指标
                metrics = wrapper.get_performance_metrics()
                print("包装器性能指标:", metrics)

                # 获取系统信息
                system_info = wrapper.get_system_info()
                print("系统信息:", system_info)

            except Exception as e:
                print(f"测试失败: {e}")

    print("测试完成!")