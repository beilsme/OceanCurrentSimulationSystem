"""
混合求解器
结合多种数值方法和物理模型的综合求解器
支持欧拉-拉格朗日混合方法、多尺度耦合和自适应网格细化
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# 导入相关模块
from .particle_tracking_wrapper import ParticleTrackingWrapper, ParticleState, TrackingParameters
from .current_simulation_wrapper import CurrentSimulationWrapper, GridConfiguration, SimulationParameters
from .pollution_dispersion import PollutionDispersionSimulator
from ..core.cpp_interface import CppInterfaceWrapper

# 设置日志
logger = logging.getLogger(__name__)

class SolverCoupling(Enum):
    """求解器耦合方式"""
    LOOSE_COUPLING = "loose"      # 松耦合
    TIGHT_COUPLING = "tight"      # 紧耦合
    ITERATIVE_COUPLING = "iterative"  # 迭代耦合

class TimeIntegration(Enum):
    """时间积分方式"""
    OPERATOR_SPLITTING = "splitting"  # 算子分裂
    STRANG_SPLITTING = "strang"      # Strang分裂
    IMPLICIT_EXPLICIT = "imex"       # 隐式-显式
    FULLY_IMPLICIT = "implicit"      # 全隐式

class AdaptationCriteria(Enum):
    """自适应准则"""
    GRADIENT_BASED = "gradient"      # 梯度判据
    ERROR_BASED = "error"           # 误差判据
    FEATURE_BASED = "feature"       # 特征判据
    CURVATURE_BASED = "curvature"   # 曲率判据

@dataclass
class CouplingParameters:
    """耦合参数"""
    coupling_type: SolverCoupling = SolverCoupling.LOOSE_COUPLING
    time_integration: TimeIntegration = TimeIntegration.OPERATOR_SPLITTING
    iteration_tolerance: float = 1e-6
    max_coupling_iterations: int = 10
    relaxation_factor: float = 0.8
    feedback_strength: float = 1.0

@dataclass
class AdaptiveParameters:
    """自适应参数"""
    enable_adaptation: bool = True
    adaptation_criteria: AdaptationCriteria = AdaptationCriteria.GRADIENT_BASED
    adaptation_threshold: float = 0.1
    min_grid_size: float = 100.0
    max_grid_size: float = 10000.0
    refinement_ratio: int = 2
    coarsening_ratio: int = 2
    adaptation_frequency: int = 10

@dataclass
class MultiScaleParameters:
    """多尺度参数"""
    enable_multiscale: bool = True
    scale_separation_ratio: float = 10.0
    upscaling_method: str = "averaging"
    downscaling_method: str = "interpolation"
    scale_coupling_strength: float = 1.0

@dataclass
class SolverConfiguration:
    """求解器配置"""
    grid_config: GridConfiguration
    sim_params: SimulationParameters
    coupling_params: CouplingParameters = field(default_factory=CouplingParameters)
    adaptive_params: AdaptiveParameters = field(default_factory=AdaptiveParameters)
    multiscale_params: MultiScaleParameters = field(default_factory=MultiScaleParameters)

class RegionManager:
    """区域管理器"""

    def __init__(self):
        self.regions = {}
        self.interfaces = {}
        self.region_counter = 0

    def add_region(self, bounds: Tuple[float, float, float, float],
                   grid_resolution: Tuple[int, int], region_type: str = "eulerian") -> int:
        """
        添加计算区域
        
        Args:
            bounds: 区域边界 (x_min, y_min, x_max, y_max)
            grid_resolution: 网格分辨率 (nx, ny)
            region_type: 区域类型 ("eulerian", "lagrangian", "hybrid")
            
        Returns:
            区域ID
        """
        region_id = self.region_counter
        self.regions[region_id] = {
            'bounds': bounds,
            'resolution': grid_resolution,
            'type': region_type,
            'grid': None,
            'active': True
        }
        self.region_counter += 1
        logger.info(f"添加区域 {region_id}: {region_type}, 范围 {bounds}")
        return region_id

    def create_interface(self, region1_id: int, region2_id: int,
                         interface_type: str = "overlap") -> int:
        """
        创建区域间接口
        
        Args:
            region1_id: 区域1 ID
            region2_id: 区域2 ID
            interface_type: 接口类型 ("overlap", "boundary", "embedding")
            
        Returns:
            接口ID
        """
        interface_id = len(self.interfaces)
        self.interfaces[interface_id] = {
            'region1': region1_id,
            'region2': region2_id,
            'type': interface_type,
            'transfer_data': None
        }
        logger.info(f"创建接口 {interface_id}: 区域{region1_id} <-> 区域{region2_id}")
        return interface_id

    def get_overlapping_regions(self, x: float, y: float) -> List[int]:
        """获取包含指定点的所有区域"""
        overlapping = []
        for region_id, region in self.regions.items():
            if not region['active']:
                continue
            x_min, y_min, x_max, y_max = region['bounds']
            if x_min <= x <= x_max and y_min <= y <= y_max:
                overlapping.append(region_id)
        return overlapping

class HybridSolver:
    """
    混合求解器
    集成欧拉和拉格朗日方法，支持多尺度耦合和自适应网格
    """

    def __init__(self, config: SolverConfiguration):
        """
        初始化混合求解器
        
        Args:
            config: 求解器配置
        """
        self.config = config
        self.current_solver = None
        self.particle_tracker = None
        self.pollution_simulator = None
        self.cpp_interface = None

        # 区域管理
        self.region_manager = RegionManager()

        # 求解器状态
        self.current_time = 0.0
        self.time_step = config.sim_params.time_step
        self.solution_fields = {}
        self.coupling_residuals = []

        # 自适应网格
        self.adaptive_grids = {}
        self.refinement_history = []

        # 多尺度数据
        self.scale_hierarchy = {}

        # 线程管理
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()

        logger.info("混合求解器初始化完成")

    def initialize(self) -> bool:
        """
        初始化所有子求解器
        
        Returns:
            是否初始化成功
        """
        try:
            # 初始化欧拉求解器
            self.current_solver = CurrentSimulationWrapper()
            if not self.current_solver.initialize(self.config.grid_config, self.config.sim_params):
                logger.error("欧拉求解器初始化失败")
                return False

            # 初始化拉格朗日粒子追踪器
            self.particle_tracker = ParticleTrackingWrapper()
            if not self.particle_tracker.initialize():
                logger.error("粒子追踪器初始化失败")
                return False

            # 初始化污染物扩散模拟器
            self.pollution_simulator = PollutionDispersionSimulator(self.particle_tracker)

            # 初始化C++接口（如果可用）
            try:
                self.cpp_interface = CppInterfaceWrapper()
                self.cpp_interface.initialize()
            except Exception as e:
                logger.warning(f"C++接口初始化失败: {e}")
                self.cpp_interface = None

            # 设置初始区域
            self._setup_initial_regions()

            logger.info("所有子求解器初始化成功")
            return True

        except Exception as e:
            logger.error(f"混合求解器初始化异常: {e}")
            return False

    def _setup_initial_regions(self):
        """设置初始计算区域"""
        grid = self.config.grid_config

        # 主欧拉区域
        main_bounds = (grid.x_min, grid.y_min, grid.x_max, grid.y_max)
        main_resolution = (grid.nx, grid.ny)
        main_region = self.region_manager.add_region(main_bounds, main_resolution, "eulerian")

        # 如果启用多尺度，创建不同分辨率的区域
        if self.config.multiscale_params.enable_multiscale:
            ratio = self.config.multiscale_params.scale_separation_ratio

            # 粗网格区域
            coarse_resolution = (max(10, grid.nx // int(ratio)), max(8, grid.ny // int(ratio)))
            coarse_region = self.region_manager.add_region(main_bounds, coarse_resolution, "eulerian")

            # 细网格区域（子区域）
            fine_bounds = (
                grid.x_min + (grid.x_max - grid.x_min) * 0.3,
                grid.y_min + (grid.y_max - grid.y_min) * 0.3,
                grid.x_min + (grid.x_max - grid.x_min) * 0.7,
                grid.y_min + (grid.y_max - grid.y_min) * 0.7
            )
            fine_resolution = (int(grid.nx * ratio), int(grid.ny * ratio))
            fine_region = self.region_manager.add_region(fine_bounds, fine_resolution, "eulerian")

            # 创建接口
            self.region_manager.create_interface(main_region, coarse_region, "overlap")
            self.region_manager.create_interface(main_region, fine_region, "embedding")

        # 拉格朗日区域（用于粒子追踪）
        lagrangian_region = self.region_manager.add_region(main_bounds, (0, 0), "lagrangian")

        logger.info(f"设置了 {len(self.region_manager.regions)} 个计算区域")

    def set_initial_conditions(self, velocity_field: Dict[str, np.ndarray],
                               scalar_fields: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """
        设置初始条件
        
        Args:
            velocity_field: 初始速度场 {'u': u_array, 'v': v_array, 'w': w_array}
            scalar_fields: 标量场字典（可选）
            
        Returns:
            是否设置成功
        """
        try:
            # 设置欧拉求解器的初始条件
            if not self.current_solver.set_initial_velocity_field(
                    velocity_field['u'], velocity_field['v'], velocity_field.get('w')
            ):
                logger.error("设置欧拉求解器初始条件失败")
                return False

            # 设置粒子追踪器的速度场
            grid_info = {
                'nx': self.config.grid_config.nx,
                'ny': self.config.grid_config.ny,
                'nz': self.config.grid_config.nz,
                'dx': self.config.grid_config.dx,
                'dy': self.config.grid_config.dy,
                'dz': self.config.grid_config.dz,
                'x0': self.config.grid_config.x_min,
                'y0': self.config.grid_config.y_min,
                'z0': self.config.grid_config.z_min,
                'dt': self.time_step
            }

            if not self.particle_tracker.set_velocity_field(
                    velocity_field['u'], velocity_field['v'], velocity_field.get('w',
                                                                                 np.zeros_like(velocity_field['u'])), grid_info
            ):
                logger.error("设置粒子追踪器速度场失败")
                return False

            # 存储解场
            self.solution_fields = velocity_field.copy()
            if scalar_fields:
                self.solution_fields.update(scalar_fields)

            logger.info("初始条件设置成功")
            return True

        except Exception as e:
            logger.error(f"设置初始条件异常: {e}")
            return False

    def advance_time_step(self) -> bool:
        """
        推进一个时间步
        
        Returns:
            是否成功
        """
        try:
            if self.config.coupling_params.coupling_type == SolverCoupling.LOOSE_COUPLING:
                return self._advance_loose_coupling()
            elif self.config.coupling_params.coupling_type == SolverCoupling.TIGHT_COUPLING:
                return self._advance_tight_coupling()
            elif self.config.coupling_params.coupling_type == SolverCoupling.ITERATIVE_COUPLING:
                return self._advance_iterative_coupling()
            else:
                logger.error(f"不支持的耦合类型: {self.config.coupling_params.coupling_type}")
                return False

        except Exception as e:
            logger.error(f"时间步推进异常: {e}")
            return False

    def _advance_loose_coupling(self) -> bool:
        """松耦合时间步推进"""
        # 1. 更新欧拉场
        if not self.current_solver.advance_time_step():
            logger.error("欧拉求解器时间步失败")
            return False

        # 2. 获取更新后的速度场
        current_field = self.current_solver.get_current_field()
        if current_field is None:
            logger.error("无法获取当前速度场")
            return False

        # 3. 更新拉格朗日粒子
        if hasattr(self, 'active_particles') and self.active_particles:
            grid_info = self._get_grid_info()

            # 使用新的速度场追踪粒子
            self.particle_tracker.set_velocity_field(
                current_field['u'], current_field['v'], current_field['w'], grid_info
            )

            # 推进粒子一个时间步
            tracking_params = TrackingParameters(
                time_step=self.time_step,
                max_time=self.time_step,
                output_interval=self.time_step
            )

            try:
                new_trajectories = self.particle_tracker.track_multiple_particles(
                    self.active_particles, tracking_params, parallel=True
                )

                # 更新粒子状态
                for i, trajectory in enumerate(new_trajectories):
                    if trajectory and len(trajectory) > 1:
                        final_state = trajectory[-1]
                        self.active_particles[i] = final_state

            except Exception as e:
                logger.error(f"粒子追踪失败: {e}")

        # 4. 应用自适应网格细化（如果启用）
        if self.config.adaptive_params.enable_adaptation:
            self._apply_grid_adaptation(current_field)

        # 5. 处理多尺度耦合（如果启用）
        if self.config.multiscale_params.enable_multiscale:
            self._apply_multiscale_coupling(current_field)

        # 6. 更新解场
        self.solution_fields.update(current_field)
        self.current_time += self.time_step

        return True

    def _advance_tight_coupling(self) -> bool:
        """紧耦合时间步推进"""
        # 紧耦合需要在每个时间步内同时求解欧拉和拉格朗日方程
        max_iterations = self.config.coupling_params.max_coupling_iterations
        tolerance = self.config.coupling_params.iteration_tolerance
        relaxation = self.config.coupling_params.relaxation_factor

        # 保存上一时间步的解
        old_field = self.solution_fields.copy()

        for iteration in range(max_iterations):
            # 1. 欧拉步
            if not self.current_solver.advance_time_step():
                logger.error(f"欧拉求解器迭代 {iteration} 失败")
                return False

            new_field = self.current_solver.get_current_field()
            if new_field is None:
                return False

            # 2. 拉格朗日步
            if hasattr(self, 'active_particles') and self.active_particles:
                self._update_lagrangian_particles(new_field)

            # 3. 计算耦合反馈
            feedback = self._calculate_coupling_feedback(old_field, new_field)

            # 4. 应用松弛
            for key in new_field:
                new_field[key] = (1 - relaxation) * old_field[key] + relaxation * new_field[key]

            # 5. 检查收敛
            residual = np.linalg.norm(feedback)
            self.coupling_residuals.append(residual)

            if residual < tolerance:
                logger.debug(f"紧耦合在迭代 {iteration+1} 收敛，残差: {residual:.2e}")
                break

            old_field = new_field.copy()

        self.solution_fields.update(new_field)
        self.current_time += self.time_step

        return True

    def _advance_iterative_coupling(self) -> bool:
        """迭代耦合时间步推进"""
        # 类似紧耦合，但使用更复杂的迭代策略
        return self._advance_tight_coupling()  # 简化实现

    def _update_lagrangian_particles(self, velocity_field: Dict[str, np.ndarray]):
        """更新拉格朗日粒子"""
        if not hasattr(self, 'active_particles') or not self.active_particles:
            return

        grid_info = self._get_grid_info()

        # 设置速度场
        self.particle_tracker.set_velocity_field(
            velocity_field['u'], velocity_field['v'], velocity_field['w'], grid_info
        )

        # 追踪粒子
        tracking_params = TrackingParameters(
            time_step=self.time_step,
            max_time=self.time_step,
            output_interval=self.time_step
        )

        try:
            trajectories = self.particle_tracker.track_multiple_particles(
                self.active_particles, tracking_params, parallel=False
            )

            # 更新粒子位置
            for i, trajectory in enumerate(trajectories):
                if trajectory and len(trajectory) > 1:
                    self.active_particles[i] = trajectory[-1]

        except Exception as e:
            logger.error(f"拉格朗日粒子更新失败: {e}")

    def _calculate_coupling_feedback(self, old_field: Dict[str, np.ndarray],
                                     new_field: Dict[str, np.ndarray]) -> np.ndarray:
        """计算耦合反馈"""
        feedback = []

        for key in ['u', 'v', 'w']:
            if key in old_field and key in new_field:
                diff = new_field[key] - old_field[key]
                feedback.append(np.mean(np.abs(diff)))

        return np.array(feedback)

    def _apply_grid_adaptation(self, solution_field: Dict[str, np.ndarray]):
        """应用网格自适应细化"""
        if not self.config.adaptive_params.enable_adaptation:
            return

        # 检查是否需要细化
        if (self.current_time / self.time_step) % self.config.adaptive_params.adaptation_frequency != 0:
            return

        try:
            # 计算适应指标
            criteria = self._compute_adaptation_criteria(solution_field)

            # 确定需要细化和粗化的区域
            refinement_regions, coarsening_regions = self._identify_adaptation_regions(criteria)

            # 执行网格细化
            if refinement_regions:
                self._refine_grid_regions(refinement_regions)

            # 执行网格粗化
            if coarsening_regions:
                self._coarsen_grid_regions(coarsening_regions)

            logger.debug(f"网格自适应: 细化 {len(refinement_regions)} 个区域, "
                         f"粗化 {len(coarsening_regions)} 个区域")

        except Exception as e:
            logger.error(f"网格自适应失败: {e}")

    def _compute_adaptation_criteria(self, solution_field: Dict[str, np.ndarray]) -> np.ndarray:
        """计算自适应判据"""
        criteria_type = self.config.adaptive_params.adaptation_criteria

        if criteria_type == AdaptationCriteria.GRADIENT_BASED:
            # 基于梯度的判据
            u = solution_field['u']
            v = solution_field['v']

            # 计算速度梯度
            grad_u_x = np.gradient(u, axis=2)
            grad_u_y = np.gradient(u, axis=1)
            grad_v_x = np.gradient(v, axis=2)
            grad_v_y = np.gradient(v, axis=1)

            # 计算梯度模长
            grad_magnitude = np.sqrt(grad_u_x**2 + grad_u_y**2 + grad_v_x**2 + grad_v_y**2)

            return grad_magnitude

        elif criteria_type == AdaptationCriteria.CURVATURE_BASED:
            # 基于曲率的判据
            u = solution_field['u']

            # 计算二阶导数
            u_xx = np.gradient(np.gradient(u, axis=2), axis=2)
            u_yy = np.gradient(np.gradient(u, axis=1), axis=1)
            u_xy = np.gradient(np.gradient(u, axis=2), axis=1)

            # 计算曲率
            curvature = np.abs(u_xx + u_yy)

            return curvature

        else:
            # 默认使用梯度判据
            return self._compute_adaptation_criteria_gradient(solution_field)

    def _identify_adaptation_regions(self, criteria: np.ndarray) -> Tuple[List, List]:
        """识别需要细化和粗化的区域"""
        threshold = self.config.adaptive_params.adaptation_threshold

        # 归一化判据
        max_criteria = np.max(criteria)
        if max_criteria > 0:
            normalized_criteria = criteria / max_criteria
        else:
            normalized_criteria = criteria

        # 识别细化区域（高梯度区域）
        refinement_mask = normalized_criteria > threshold
        refinement_regions = self._extract_connected_regions(refinement_mask, "refinement")

        # 识别粗化区域（低梯度区域）
        coarsening_mask = normalized_criteria < threshold * 0.1
        coarsening_regions = self._extract_connected_regions(coarsening_mask, "coarsening")

        return refinement_regions, coarsening_regions

    def _extract_connected_regions(self, mask: np.ndarray, region_type: str) -> List[Dict]:
        """提取连通区域"""
        from scipy import ndimage

        # 标记连通区域
        labeled_array, num_features = ndimage.label(mask)

        regions = []
        for i in range(1, num_features + 1):
            region_mask = labeled_array == i

            # 获取区域边界
            indices = np.where(region_mask)
            if len(indices[0]) > 0:
                y_min, y_max = np.min(indices[0]), np.max(indices[0])
                x_min, x_max = np.min(indices[1]), np.max(indices[1])

                region = {
                    'type': region_type,
                    'bounds': (x_min, y_min, x_max, y_max),
                    'mask': region_mask,
                    'size': np.sum(region_mask)
                }
                regions.append(region)

        return regions

    def _refine_grid_regions(self, regions: List[Dict]):
        """细化网格区域"""
        for region in regions:
            # 实现网格细化逻辑
            # 这里是简化版本，实际实现需要重新网格化
            logger.debug(f"细化区域: 边界 {region['bounds']}, 大小 {region['size']}")

            # 记录细化历史
            self.refinement_history.append({
                'time': self.current_time,
                'type': 'refinement',
                'region': region['bounds']
            })

    def _coarsen_grid_regions(self, regions: List[Dict]):
        """粗化网格区域"""
        for region in regions:
            # 实现网格粗化逻辑
            logger.debug(f"粗化区域: 边界 {region['bounds']}, 大小 {region['size']}")

            # 记录粗化历史
            self.refinement_history.append({
                'time': self.current_time,
                'type': 'coarsening',
                'region': region['bounds']
            })

    def _apply_multiscale_coupling(self, solution_field: Dict[str, np.ndarray]):
        """应用多尺度耦合"""
        if not self.config.multiscale_params.enable_multiscale:
            return

        try:
            # 上尺度传递（细网格到粗网格）
            self._upscale_solution(solution_field)

            # 下尺度传递（粗网格到细网格）
            self._downscale_solution(solution_field)

        except Exception as e:
            logger.error(f"多尺度耦合失败: {e}")

    def _upscale_solution(self, solution_field: Dict[str, np.ndarray]):
        """上尺度传递"""
        method = self.config.multiscale_params.upscaling_method

        if method == "averaging":
            # 使用平均值上尺度
            for key, field in solution_field.items():
                if key in ['u', 'v', 'w']:
                    # 简化的区域平均
                    coarse_field = self._spatial_average(field, 2)
                    self.scale_hierarchy[f"{key}_coarse"] = coarse_field

        elif method == "filtering":
            # 使用滤波上尺度
            for key, field in solution_field.items():
                if key in ['u', 'v', 'w']:
                    coarse_field = self._apply_filter(field, "gaussian")
                    self.scale_hierarchy[f"{key}_coarse"] = coarse_field

    def _downscale_solution(self, solution_field: Dict[str, np.ndarray]):
        """下尺度传递"""
        method = self.config.multiscale_params.downscaling_method

        if method == "interpolation":
            # 使用插值下尺度
            for key in ['u', 'v', 'w']:
                coarse_key = f"{key}_coarse"
                if coarse_key in self.scale_hierarchy:
                    fine_field = self._interpolate_field(
                        self.scale_hierarchy[coarse_key], solution_field[key].shape
                    )
                    # 混合粗细网格解
                    strength = self.config.multiscale_params.scale_coupling_strength
                    solution_field[key] = (1 - strength) * solution_field[key] + strength * fine_field

    def _spatial_average(self, field: np.ndarray, factor: int) -> np.ndarray:
        """空间平均"""
        nz, ny, nx = field.shape
        new_ny, new_nx = ny // factor, nx // factor

        averaged = np.zeros((nz, new_ny, new_nx))

        for i in range(new_ny):
            for j in range(new_nx):
                averaged[:, i, j] = np.mean(
                    field[:, i*factor:(i+1)*factor, j*factor:(j+1)*factor], axis=(1, 2)
                )

        return averaged

    def _apply_filter(self, field: np.ndarray, filter_type: str) -> np.ndarray:
        """应用滤波器"""
        from scipy import ndimage

        if filter_type == "gaussian":
            return ndimage.gaussian_filter(field, sigma=1.0)
        else:
            return field

    def _interpolate_field(self, coarse_field: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """插值场到目标形状"""
        from scipy import ndimage

        zoom_factors = [target_shape[i] / coarse_field.shape[i] for i in range(len(target_shape))]
        return ndimage.zoom(coarse_field, zoom_factors)

    def _get_grid_info(self) -> Dict[str, Union[float, int]]:
        """获取网格信息"""
        grid = self.config.grid_config
        return {
            'nx': grid.nx, 'ny': grid.ny, 'nz': grid.nz,
            'dx': grid.dx, 'dy': grid.dy, 'dz': grid.dz,
            'x0': grid.x_min, 'y0': grid.y_min, 'z0': grid.z_min,
            'dt': self.time_step
        }

    def add_lagrangian_particles(self, initial_states: List[ParticleState]):
        """添加拉格朗日粒子"""
        if not hasattr(self, 'active_particles'):
            self.active_particles = []

        self.active_particles.extend(initial_states)
        logger.info(f"添加 {len(initial_states)} 个拉格朗日粒子")

    def run_simulation(self, total_time: float,
                       callback_func: Optional[Callable] = None) -> bool:
        """
        运行完整混合模拟
        
        Args:
            total_time: 总模拟时间
            callback_func: 回调函数
            
        Returns:
            是否成功
        """
        logger.info(f"开始混合模拟，总时间: {total_time/3600:.1f} 小时")

        try:
            start_time = time.time()
            num_steps = int(total_time / self.time_step)

            for step in range(num_steps):
                # 推进时间步
                if not self.advance_time_step():
                    logger.error(f"时间步 {step+1} 失败")
                    return False

                # 回调函数
                if callback_func and step % 10 == 0:
                    try:
                        diagnostics = self.get_diagnostics()
                        callback_func(step, num_steps, diagnostics)
                    except Exception as e:
                        logger.warning(f"回调函数执行异常: {e}")

                # 进度报告
                if step % 100 == 0:
                    elapsed = time.time() - start_time
                    progress = (step + 1) / num_steps * 100
                    diagnostics = self.get_diagnostics()

                    logger.info(f"进度: {progress:.1f}% ({step+1}/{num_steps}), "
                                f"最大速度: {diagnostics.get('max_velocity', 0):.3f} m/s, "
                                f"耗时: {elapsed:.1f}s")

            total_elapsed = time.time() - start_time
            logger.info(f"混合模拟完成，总耗时: {total_elapsed:.1f}s")
            return True

        except Exception as e:
            logger.error(f"模拟运行异常: {e}")
            return False

    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息"""
        diagnostics = {}

        # 欧拉求解器诊断
        if self.current_solver:
            euler_diag = self.current_solver.get_diagnostics()
            diagnostics.update(euler_diag)

        # 拉格朗日粒子诊断
        if hasattr(self, 'active_particles') and self.active_particles:
            active_count = sum(1 for p in self.active_particles if p.active)
            diagnostics['active_particles'] = active_count
            diagnostics['total_particles'] = len(self.active_particles)

        # 耦合诊断
        if self.coupling_residuals:
            diagnostics['coupling_residual'] = self.coupling_residuals[-1]
            diagnostics['coupling_iterations'] = len(self.coupling_residuals)

        # 自适应网格诊断
        if self.refinement_history:
            recent_refinements = [r for r in self.refinement_history
                                  if self.current_time - r['time'] < 3600]  # 最近1小时
            diagnostics['recent_refinements'] = len(recent_refinements)

        # 多尺度诊断
        if self.scale_hierarchy:
            diagnostics['scale_levels'] = len(self.scale_hierarchy)

        diagnostics['simulation_time'] = self.current_time
        diagnostics['time_step'] = self.time_step

        return diagnostics

    def export_solution(self, filename: str, format: str = 'netcdf') -> bool:
        """导出解场"""
        try:
            if format.lower() == 'netcdf':
                return self._export_netcdf_solution(filename)
            elif format.lower() == 'vtk':
                return self._export_vtk_solution(filename)
            else:
                logger.error(f"不支持的输出格式: {format}")
                return False
        except Exception as e:
            logger.error(f"导出解场失败: {e}")
            return False

    def _export_netcdf_solution(self, filename: str) -> bool:
        """导出NetCDF格式解场"""
        try:
            import netCDF4 as nc

            with nc.Dataset(filename, 'w') as dataset:
                # 创建维度
                grid = self.config.grid_config
                dataset.createDimension('x', grid.nx)
                dataset.createDimension('y', grid.ny)
                dataset.createDimension('z', grid.nz)

                # 创建坐标变量
                x = dataset.createVariable('x', 'f8', ('x',))
                y = dataset.createVariable('y', 'f8', ('y',))
                z = dataset.createVariable('z', 'f8', ('z',))

                x[:] = np.linspace(grid.x_min, grid.x_max, grid.nx)
                y[:] = np.linspace(grid.y_min, grid.y_max, grid.ny)
                z[:] = np.linspace(grid.z_min, grid.z_max, grid.nz)

                # 创建解变量
                for var_name, var_data in self.solution_fields.items():
                    if isinstance(var_data, np.ndarray) and var_data.ndim == 3:
                        var = dataset.createVariable(var_name, 'f8', ('z', 'y', 'x'))
                        var[:] = var_data

                        if var_name in ['u', 'v', 'w']:
                            var.units = 'm/s'
                            var.long_name = f'Velocity component {var_name}'
                        elif var_name == 'p':
                            var.units = 'Pa'
                            var.long_name = 'Pressure'

                # 拉格朗日粒子数据
                if hasattr(self, 'active_particles') and self.active_particles:
                    n_particles = len(self.active_particles)
                    dataset.createDimension('particle', n_particles)

                    px = dataset.createVariable('particle_x', 'f8', ('particle',))
                    py = dataset.createVariable('particle_y', 'f8', ('particle',))
                    pz = dataset.createVariable('particle_z', 'f8', ('particle',))

                    for i, particle in enumerate(self.active_particles):
                        px[i] = particle.x
                        py[i] = particle.y
                        pz[i] = particle.z

                # 全局属性
                dataset.title = 'Hybrid Solver Simulation Results'
                dataset.simulation_time = self.current_time
                dataset.coupling_type = self.config.coupling_params.coupling_type.value

            logger.info(f"解场已导出到: {filename}")
            return True

        except ImportError:
            logger.error("需要安装netCDF4库")
            return False
        except Exception as e:
            logger.error(f"NetCDF导出失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        if self.current_solver:
            self.current_solver.cleanup()

        if self.particle_tracker:
            self.particle_tracker.cleanup()

        if self.pollution_simulator:
            self.pollution_simulator.cleanup()

        if self.cpp_interface:
            self.cpp_interface.cleanup()

        if self.executor:
            self.executor.shutdown(wait=True)

        logger.info("混合求解器已清理")

    def __del__(self):
        """析构函数"""
        self.cleanup()


# 使用示例
if __name__ == "__main__":
    # 创建网格配置
    grid_config = GridConfiguration(
        nx=100, ny=80, nz=20,
        dx=1000.0, dy=1000.0, dz=10.0,
        x_min=0.0, y_min=0.0, z_min=0.0,
        x_max=100000.0, y_max=80000.0, z_max=200.0
    )

    # 创建模拟参数
    sim_params = SimulationParameters(
        time_step=300.0,
        total_time=86400.0,
        viscosity=1e-6,
        density=1025.0
    )

    # 创建耦合参数
    coupling_params = CouplingParameters(
        coupling_type=SolverCoupling.LOOSE_COUPLING,
        time_integration=TimeIntegration.OPERATOR_SPLITTING
    )

    # 创建自适应参数
    adaptive_params = AdaptiveParameters(
        enable_adaptation=True,
        adaptation_criteria=AdaptationCriteria.GRADIENT_BASED,
        adaptation_frequency=50
    )

    # 创建多尺度参数
    multiscale_params = MultiScaleParameters(
        enable_multiscale=True,
        scale_separation_ratio=5.0
    )

    # 创建求解器配置
    config = SolverConfiguration(
        grid_config=grid_config,
        sim_params=sim_params,
        coupling_params=coupling_params,
        adaptive_params=adaptive_params,
        multiscale_params=multiscale_params
    )

    # 创建混合求解器
    solver = HybridSolver(config)

    # 初始化
    if not solver.initialize():
        print("初始化失败")
        exit(1)

    # 设置初始条件
    u_init = np.random.randn(grid_config.nz, grid_config.ny, grid_config.nx) * 0.5
    v_init = np.random.randn(grid_config.nz, grid_config.ny, grid_config.nx) * 0.3
    w_init = np.random.randn(grid_config.nz, grid_config.ny, grid_config.nx) * 0.1

    velocity_field = {'u': u_init, 'v': v_init, 'w': w_init}

    if not solver.set_initial_conditions(velocity_field):
        print("设置初始条件失败")
        exit(1)

    # 添加拉格朗日粒子
    initial_particles = [
        ParticleState(x=25000.0, y=20000.0, z=5.0, u=0.0, v=0.0, w=0.0, age=0.0),
        ParticleState(x=50000.0, y=40000.0, z=10.0, u=0.0, v=0.0, w=0.0, age=0.0),
        ParticleState(x=75000.0, y=60000.0, z=15.0, u=0.0, v=0.0, w=0.0, age=0.0),
    ]
    solver.add_lagrangian_particles(initial_particles)

    # 定义回调函数
    def progress_callback(step, total_steps, diagnostics):
        progress = step / total_steps * 100
        print(f"进度: {progress:.1f}%, 最大速度: {diagnostics.get('max_velocity', 0):.3f} m/s, "
              f"活跃粒子: {diagnostics.get('active_particles', 0)}, "
              f"耦合残差: {diagnostics.get('coupling_residual', 0):.2e}")

    # 运行模拟
    total_time = 24 * 3600  # 24小时
    print("开始混合模拟...")

    success = solver.run_simulation(total_time, progress_callback)

    if success:
        print("模拟完成!")

        # 获取最终诊断
        final_diagnostics = solver.get_diagnostics()
        print(f"最终时间: {final_diagnostics.get('simulation_time', 0)/3600:.1f} 小时")
        print(f"活跃粒子数: {final_diagnostics.get('active_particles', 0)}")
        print(f"网格细化次数: {final_diagnostics.get('recent_refinements', 0)}")

        # 导出结果
        solver.export_solution("hybrid_simulation_results.nc", "netcdf")
    else:
        print("模拟失败!")

    # 清理
    solver.cleanup()
    print("混合求解器示例完成")