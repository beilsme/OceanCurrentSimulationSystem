"""
simulation模块综合使用示例
展示如何使用各个模拟组件进行海洋模拟
包括完整的工作流程和最佳实践
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional
import time
import os
import sys

# 添加模块路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入simulation模块
from simulation import (
    ParticleTrackingWrapper, CurrentSimulationWrapper,
    PollutionDispersionSimulator, HybridSolver
)
from simulation.particle_tracking_wrapper import ParticleState, TrackingParameters, IntegrationMethod
from simulation.current_simulation_wrapper import (
    GridConfiguration, SimulationParameters, BoundaryCondition,
    BoundaryType, SolverMethod, TimeScheme, ForceField
)
from simulation.pollution_dispersion import (
    DischargeSource, DischargeType, PollutantProperties, PollutantType,
    EnvironmentalConditions, OIL_PROPERTIES, CHEMICAL_PROPERTIES
)
from simulation.hybrid_solver import (
    SolverConfiguration, CouplingParameters, AdaptiveParameters,
    MultiScaleParameters, SolverCoupling, AdaptationCriteria
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OceanSimulationWorkflow:
    """
    海洋模拟工作流程
    整合各个模拟组件的完整工作流程
    """

    def __init__(self, output_dir: str = "simulation_results"):
        """
        初始化工作流程
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.ensure_output_directory()

        # 模拟组件
        self.current_solver = None
        self.particle_tracker = None
        self.pollution_simulator = None
        self.hybrid_solver = None

        # 模拟数据
        self.grid_config = None
        self.velocity_fields = {}
        self.simulation_results = {}

        logger.info(f"海洋模拟工作流程初始化，输出目录: {output_dir}")

    def ensure_output_directory(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"创建输出目录: {self.output_dir}")

    def setup_simulation_domain(self, domain_config: Dict) -> GridConfiguration:
        """
        设置模拟区域
        
        Args:
            domain_config: 区域配置字典
            
        Returns:
            网格配置对象
        """
        self.grid_config = GridConfiguration(
            nx=domain_config.get('nx', 100),
            ny=domain_config.get('ny', 80),
            nz=domain_config.get('nz', 20),
            dx=domain_config.get('dx', 1000.0),
            dy=domain_config.get('dy', 1000.0),
            dz=domain_config.get('dz', 10.0),
            x_min=domain_config.get('x_min', 0.0),
            y_min=domain_config.get('y_min', 0.0),
            z_min=domain_config.get('z_min', 0.0),
            x_max=domain_config.get('x_max', 100000.0),
            y_max=domain_config.get('y_max', 80000.0),
            z_max=domain_config.get('z_max', 200.0)
        )

        logger.info(f"设置模拟区域: {self.grid_config.nx}×{self.grid_config.ny}×{self.grid_config.nz}")
        return self.grid_config

    def create_synthetic_ocean_field(self) -> Dict[str, np.ndarray]:
        """
        创建合成海洋场
        包括洋流、温度、盐度等
        
        Returns:
            海洋场字典
        """
        if self.grid_config is None:
            raise ValueError("请先设置模拟区域")

        nx, ny, nz = self.grid_config.nx, self.grid_config.ny, self.grid_config.nz

        # 创建坐标网格
        x = np.linspace(self.grid_config.x_min, self.grid_config.x_max, nx)
        y = np.linspace(self.grid_config.y_min, self.grid_config.y_max, ny)
        z = np.linspace(self.grid_config.z_min, self.grid_config.z_max, nz)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        X = np.transpose(X, (2, 1, 0))  # [nz, ny, nx]
        Y = np.transpose(Y, (2, 1, 0))
        Z = np.transpose(Z, (2, 1, 0))

        # 创建环流模式（类似墨西哥湾流）
        # 主流从西向东，带有涡旋结构
        center_x = (self.grid_config.x_max + self.grid_config.x_min) / 2
        center_y = (self.grid_config.y_max + self.grid_config.y_min) / 2

        # 基础流场
        u_base = 0.5 * np.ones((nz, ny, nx))  # 基础东向流 0.5 m/s
        v_base = 0.1 * np.sin(2 * np.pi * X / (self.grid_config.x_max - self.grid_config.x_min))

        # 添加涡旋
        for i, (vortex_x, vortex_y, strength, radius) in enumerate([
            (center_x - 20000, center_y + 15000, 0.8, 15000),  # 气旋涡
            (center_x + 25000, center_y - 10000, -0.6, 12000),  # 反气旋涡
            (center_x - 10000, center_y - 20000, 0.4, 8000),   # 小涡旋
        ]):
            r = np.sqrt((X - vortex_x)**2 + (Y - vortex_y)**2)

            # 兰金涡模型
            mask_inner = r < radius * 0.3
            mask_outer = (r >= radius * 0.3) & (r < radius)

            # 内核：固体旋转
            theta_inner = np.arctan2(Y - vortex_y, X - vortex_x)
            u_vortex_inner = -strength * r * np.sin(theta_inner) / (radius * 0.3)
            v_vortex_inner = strength * r * np.cos(theta_inner) / (radius * 0.3)

            # 外环：自由涡
            theta_outer = np.arctan2(Y - vortex_y, X - vortex_x)
            u_vortex_outer = -strength * (radius * 0.3) * np.sin(theta_outer) / r
            v_vortex_outer = strength * (radius * 0.3) * np.cos(theta_outer) / r

            # 衰减函数
            decay = np.exp(-(r / radius)**2)

            u_base += (u_vortex_inner * mask_inner + u_vortex_outer * mask_outer) * decay
            v_base += (v_vortex_inner * mask_inner + v_vortex_outer * mask_outer) * decay

        # 垂直结构（表层流强，深层流弱）
        depth_factor = np.exp(-Z / 50.0)  # 特征深度50米
        u = u_base * depth_factor
        v = v_base * depth_factor

        # 垂直速度（连续性方程的简化满足）
        w = np.zeros_like(u)
        for k in range(1, nz):
            # 简化的垂直速度计算
            du_dx = np.gradient(u[k], axis=1)
            dv_dy = np.gradient(v[k], axis=0)
            w[k] = w[k-1] - (du_dx + dv_dy) * self.grid_config.dz

        # 温度场（表层暖，深层冷）
        temperature = 25.0 - 15.0 * (Z / self.grid_config.z_max) + \
                      2.0 * np.sin(2 * np.pi * X / (self.grid_config.x_max - self.grid_config.x_min)) * \
                      np.exp(-Z / 30.0)

        # 盐度场（相对均匀，略有梯度）
        salinity = 35.0 + 1.0 * (Y - center_y) / (self.grid_config.y_max - self.grid_config.y_min) + \
                   0.5 * (Z / self.grid_config.z_max)

        # 添加随机扰动
        noise_scale = 0.05
        u += np.random.normal(0, noise_scale, u.shape)
        v += np.random.normal(0, noise_scale, v.shape)
        w += np.random.normal(0, noise_scale * 0.1, w.shape)

        ocean_field = {
            'u': u, 'v': v, 'w': w,
            'temperature': temperature,
            'salinity': salinity,
            'pressure': np.zeros_like(u)  # 压力场初始为零
        }

        self.velocity_fields = ocean_field
        logger.info("合成海洋场创建完成")

        return ocean_field

    def run_current_simulation_example(self) -> Dict[str, np.ndarray]:
        """
        运行洋流模拟示例
        
        Returns:
            模拟结果
        """
        logger.info("开始洋流模拟示例...")

        # 创建洋流模拟器
        self.current_solver = CurrentSimulationWrapper()

        # 模拟参数
        sim_params = SimulationParameters(
            time_step=600.0,  # 10分钟
            total_time=7200.0,  # 2小时
            viscosity=1e-6,
            density=1025.0,
            coriolis_parameter=1e-4,
            solver_method=SolverMethod.FINITE_DIFFERENCE,
            time_scheme=TimeScheme.RUNGE_KUTTA_4
        )

        # 初始化求解器
        if not self.current_solver.initialize(self.grid_config, sim_params):
            raise RuntimeError("洋流求解器初始化失败")

        # 设置边界条件
        boundary_conditions = [
            BoundaryCondition(BoundaryType.NEUMANN, 0.0, 'north'),
            BoundaryCondition(BoundaryType.NEUMANN, 0.0, 'south'),
            BoundaryCondition(BoundaryType.DIRICHLET, 0.5, 'west'),  # 西边界流入
            BoundaryCondition(BoundaryType.NEUMANN, 0.0, 'east'),
        ]
        self.current_solver.set_boundary_conditions(boundary_conditions)

        # 设置初始速度场
        ocean_field = self.velocity_fields
        self.current_solver.set_initial_velocity_field(
            ocean_field['u'], ocean_field['v'], ocean_field['w']
        )

        # 设置风应力
        wind_stress_x = 0.1 * np.ones((self.grid_config.ny, self.grid_config.nx))
        wind_stress_y = 0.05 * np.sin(
            2 * np.pi * np.arange(self.grid_config.ny) / self.grid_config.ny
        )[:, np.newaxis] * np.ones((self.grid_config.ny, self.grid_config.nx))

        force_field = ForceField(
            wind_stress_x=wind_stress_x,
            wind_stress_y=wind_stress_y
        )
        self.current_solver.set_external_forces(force_field)

        # 运行模拟
        def progress_callback(step, total_steps, diagnostics):
            if step % 5 == 0:
                progress = step / total_steps * 100
                logger.info(f"洋流模拟进度: {progress:.1f}%, "
                            f"动能: {diagnostics.get('kinetic_energy', 0):.2e}")

        success = self.current_solver.run_simulation(progress_callback, callback_interval=1)

        if success:
            # 获取最终结果
            final_field = self.current_solver.get_current_field()

            # 导出结果
            output_file = os.path.join(self.output_dir, "current_simulation.nc")
            self.current_solver.export_field_data(output_file, "netcdf")

            logger.info("洋流模拟完成")
            return final_field
        else:
            raise RuntimeError("洋流模拟失败")

    def run_particle_tracking_example(self) -> List[List[ParticleState]]:
        """
        运行粒子追踪示例
        
        Returns:
            粒子轨迹列表
        """
        logger.info("开始粒子追踪示例...")

        # 创建粒子追踪器
        self.particle_tracker = ParticleTrackingWrapper()

        if not self.particle_tracker.initialize():
            raise RuntimeError("粒子追踪器初始化失败")

        # 设置速度场
        ocean_field = self.velocity_fields
        grid_info = {
            'nx': self.grid_config.nx, 'ny': self.grid_config.ny, 'nz': self.grid_config.nz,
            'dx': self.grid_config.dx, 'dy': self.grid_config.dy, 'dz': self.grid_config.dz,
            'x0': self.grid_config.x_min, 'y0': self.grid_config.y_min, 'z0': self.grid_config.z_min,
            'dt': 600.0
        }

        self.particle_tracker.set_velocity_field(
            ocean_field['u'], ocean_field['v'], ocean_field['w'], grid_info
        )

        # 创建初始粒子（在不同位置释放）
        initial_particles = []
        release_points = [
            (20000, 25000, 5),   # 释放点1
            (30000, 35000, 10),  # 释放点2
            (50000, 45000, 15),  # 释放点3
            (70000, 55000, 8),   # 释放点4
        ]

        for x, y, z in release_points:
            # 每个释放点创建多个粒子
            for i in range(10):
                offset_x = np.random.normal(0, 1000)  # 1km标准差
                offset_y = np.random.normal(0, 1000)
                offset_z = np.random.normal(0, 2)     # 2m标准差

                particle = ParticleState(
                    x=x + offset_x, y=y + offset_y, z=z + offset_z,
                    u=0.0, v=0.0, w=0.0, age=0.0
                )
                initial_particles.append(particle)

        # 追踪参数
        tracking_params = TrackingParameters(
            time_step=600.0,  # 10分钟
            max_time=24*3600.0,  # 24小时
            integration_method=IntegrationMethod.RK4,
            diffusion_coefficient=50.0,  # 增大扩散系数
            output_interval=3600.0  # 1小时输出一次
        )

        # 执行粒子追踪
        trajectories = self.particle_tracker.track_multiple_particles(
            initial_particles, tracking_params, parallel=True
        )

        # 计算统计量
        stats = self.particle_tracker.calculate_ensemble_statistics(trajectories)

        logger.info(f"粒子追踪完成，获得 {len(trajectories)} 条轨迹")
        logger.info(f"平均扩散距离: {np.nanmean(stats['spread']):.0f} m")

        # 导出轨迹
        output_file = os.path.join(self.output_dir, "particle_trajectories.csv")
        self.particle_tracker.export_trajectories(trajectories, output_file, "csv")

        return trajectories

    def run_pollution_dispersion_example(self) -> Dict:
        """
        运行污染物扩散示例
        
        Returns:
            模拟统计结果
        """
        logger.info("开始污染物扩散示例...")

        # 创建污染物扩散模拟器
        self.pollution_simulator = PollutionDispersionSimulator(self.particle_tracker)

        # 创建石油泄漏源
        oil_spill = DischargeSource(
            x=40000.0, y=30000.0, z=0.0,
            discharge_rate=500.0,  # 500 kg/s
            discharge_type=DischargeType.INSTANTANEOUS,
            properties=OIL_PROPERTIES['light_crude']
        )

        # 创建化学品连续排放源
        chemical_discharge = DischargeSource(
            x=60000.0, y=50000.0, z=0.0,
            discharge_rate=50.0,  # 50 kg/s
            discharge_type=DischargeType.CONTINUOUS,
            start_time=3600.0,  # 1小时后开始
            end_time=10800.0,   # 3小时后结束
            properties=CHEMICAL_PROPERTIES['benzene']
        )

        # 添加排放源
        self.pollution_simulator.add_discharge_source(oil_spill)
        self.pollution_simulator.add_discharge_source(chemical_discharge)

        # 设置环境条件
        env_conditions = EnvironmentalConditions(
            water_temperature=288.15,  # 15°C
            salinity=35.0,
            wind_speed=10.0,  # 10 m/s风速
            wave_height=2.5,
            solar_radiation=300.0
        )
        self.pollution_simulator.set_environmental_conditions(env_conditions)

        # 初始化粒子
        self.pollution_simulator.initialize_particles(1000)

        # 运行模拟
        ocean_field = self.velocity_fields
        grid_info = {
            'nx': self.grid_config.nx, 'ny': self.grid_config.ny, 'nz': self.grid_config.nz,
            'dx': self.grid_config.dx, 'dy': self.grid_config.dy, 'dz': self.grid_config.dz,
            'x0': self.grid_config.x_min, 'y0': self.grid_config.y_min, 'z0': self.grid_config.z_min
        }

        def pollution_callback(step, total_steps, stats):
            if step % 100 == 0:
                progress = step / total_steps * 100
                logger.info(f"污染扩散进度: {progress:.1f}%, "
                            f"活跃粒子: {stats['active_particles']}, "
                            f"扩散范围: {stats['extent']:.0f} m")

        total_time = 48 * 3600  # 48小时
        success = self.pollution_simulator.run_simulation(
            total_time, ocean_field, grid_info, pollution_callback
        )

        if success:
            # 获取最终统计
            final_stats = self.pollution_simulator.calculate_statistics()

            logger.info("污染物扩散模拟完成")
            logger.info(f"最终活跃粒子: {final_stats['active_particles']}")
            logger.info(f"总质量: {final_stats['total_mass']:.2e} kg")
            logger.info(f"最大扩散范围: {final_stats['extent']:.0f} m")

            # 导出结果
            output_prefix = os.path.join(self.output_dir, "pollution_simulation")
            self.pollution_simulator.export_results(output_prefix, "csv")

            return final_stats
        else:
            raise RuntimeError("污染物扩散模拟失败")

    def run_hybrid_simulation_example(self) -> Dict:
        """
        运行混合模拟示例
        
        Returns:
            模拟诊断结果
        """
        logger.info("开始混合模拟示例...")

        # 创建求解器配置
        sim_params = SimulationParameters(
            time_step=300.0,
            total_time=12*3600.0,  # 12小时
            viscosity=1e-6,
            density=1025.0
        )

        coupling_params = CouplingParameters(
            coupling_type=SolverCoupling.LOOSE_COUPLING,
            iteration_tolerance=1e-6
        )

        adaptive_params = AdaptiveParameters(
            enable_adaptation=True,
            adaptation_criteria=AdaptationCriteria.GRADIENT_BASED,
            adaptation_frequency=20
        )

        multiscale_params = MultiScaleParameters(
            enable_multiscale=True,
            scale_separation_ratio=3.0
        )

        config = SolverConfiguration(
            grid_config=self.grid_config,
            sim_params=sim_params,
            coupling_params=coupling_params,
            adaptive_params=adaptive_params,
            multiscale_params=multiscale_params
        )

        # 创建混合求解器
        self.hybrid_solver = HybridSolver(config)

        if not self.hybrid_solver.initialize():
            raise RuntimeError("混合求解器初始化失败")

        # 设置初始条件
        ocean_field = self.velocity_fields
        self.hybrid_solver.set_initial_conditions(ocean_field)

        # 添加拉格朗日粒子
        lagrangian_particles = [
            ParticleState(x=25000.0, y=20000.0, z=5.0, u=0.0, v=0.0, w=0.0, age=0.0),
            ParticleState(x=50000.0, y=40000.0, z=10.0, u=0.0, v=0.0, w=0.0, age=0.0),
            ParticleState(x=75000.0, y=60000.0, z=15.0, u=0.0, v=0.0, w=0.0, age=0.0),
        ]
        self.hybrid_solver.add_lagrangian_particles(lagrangian_particles)

        # 运行模拟
        def hybrid_callback(step, total_steps, diagnostics):
            if step % 20 == 0:
                progress = step / total_steps * 100
                logger.info(f"混合模拟进度: {progress:.1f}%, "
                            f"最大速度: {diagnostics.get('max_velocity', 0):.3f} m/s, "
                            f"活跃粒子: {diagnostics.get('active_particles', 0)}")

        success = self.hybrid_solver.run_simulation(sim_params.total_time, hybrid_callback)

        if success:
            # 获取最终诊断
            final_diagnostics = self.hybrid_solver.get_diagnostics()

            logger.info("混合模拟完成")
            logger.info(f"模拟时间: {final_diagnostics.get('simulation_time', 0)/3600:.1f} 小时")
            logger.info(f"活跃粒子数: {final_diagnostics.get('active_particles', 0)}")

            # 导出结果
            output_file = os.path.join(self.output_dir, "hybrid_simulation.nc")
            self.hybrid_solver.export_solution(output_file, "netcdf")

            return final_diagnostics
        else:
            raise RuntimeError("混合模拟失败")

    def create_visualization(self):
        """创建可视化图表"""
        logger.info("创建可视化图表...")

        try:
            # 可视化速度场
            self._plot_velocity_field()

            # 可视化粒子轨迹（如果有）
            if hasattr(self, 'particle_trajectories'):
                self._plot_particle_trajectories()

            logger.info("可视化图表创建完成")

        except Exception as e:
            logger.error(f"创建可视化失败: {e}")

    def _plot_velocity_field(self):
        """绘制速度场"""
        if not self.velocity_fields:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 表层流场
        u_surface = self.velocity_fields['u'][0]  # 表层
        v_surface = self.velocity_fields['v'][0]

        # 速度大小
        speed = np.sqrt(u_surface**2 + v_surface**2)

        # 绘制速度大小
        im1 = axes[0, 0].imshow(speed, origin='lower', aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Surface Current Speed (m/s)')
        plt.colorbar(im1, ax=axes[0, 0])

        # 绘制矢量场（下采样）
        step = 5
        x_indices = np.arange(0, self.grid_config.nx, step)
        y_indices = np.arange(0, self.grid_config.ny, step)
        X_sub, Y_sub = np.meshgrid(x_indices, y_indices)
        U_sub = u_surface[::step, ::step]
        V_sub = v_surface[::step, ::step]

        axes[0, 1].quiver(X_sub, Y_sub, U_sub, V_sub, scale=10)
        axes[0, 1].set_title('Surface Current Vectors')
        axes[0, 1].set_aspect('equal')

        # 温度场
        if 'temperature' in self.velocity_fields:
            temp_surface = self.velocity_fields['temperature'][0]
            im3 = axes[1, 0].imshow(temp_surface, origin='lower', aspect='auto', cmap='RdYlBu_r')
            axes[1, 0].set_title('Surface Temperature (°C)')
            plt.colorbar(im3, ax=axes[1, 0])

        # 盐度场
        if 'salinity' in self.velocity_fields:
            sal_surface = self.velocity_fields['salinity'][0]
            im4 = axes[1, 1].imshow(sal_surface, origin='lower', aspect='auto', cmap='plasma')
            axes[1, 1].set_title('Surface Salinity (psu)')
            plt.colorbar(im4, ax=axes[1, 1])

        plt.tight_layout()
        output_file = os.path.join(self.output_dir, "velocity_field_visualization.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"速度场可视化保存到: {output_file}")

    def run_complete_workflow(self):
        """运行完整的工作流程"""
        logger.info("开始完整海洋模拟工作流程...")

        start_time = time.time()

        try:
            # 1. 设置模拟区域
            domain_config = {
                'nx': 120, 'ny': 100, 'nz': 25,
                'dx': 1000.0, 'dy': 1000.0, 'dz': 8.0,
                'x_min': 0.0, 'y_min': 0.0, 'z_min': 0.0,
                'x_max': 120000.0, 'y_max': 100000.0, 'z_max': 200.0
            }
            self.setup_simulation_domain(domain_config)

            # 2. 创建合成海洋场
            self.create_synthetic_ocean_field()

            # 3. 运行洋流模拟
            current_results = self.run_current_simulation_example()
            self.simulation_results['current'] = current_results

            # 4. 运行粒子追踪
            particle_trajectories = self.run_particle_tracking_example()
            self.simulation_results['particles'] = particle_trajectories

            # 5. 运行污染物扩散
            pollution_stats = self.run_pollution_dispersion_example()
            self.simulation_results['pollution'] = pollution_stats

            # 6. 运行混合模拟
            hybrid_diagnostics = self.run_hybrid_simulation_example()
            self.simulation_results['hybrid'] = hybrid_diagnostics

            # 7. 创建可视化
            self.create_visualization()

            # 8. 生成综合报告
            self.generate_summary_report()

            total_time = time.time() - start_time
            logger.info(f"完整工作流程完成，总耗时: {total_time:.1f} 秒")

        except Exception as e:
            logger.error(f"工作流程执行失败: {e}")
            raise

        finally:
            # 清理资源
            self.cleanup()

    def generate_summary_report(self):
        """生成综合报告"""
        logger.info("生成综合报告...")

        report_file = os.path.join(self.output_dir, "simulation_summary_report.md")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 海洋模拟综合报告\n\n")
            f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 模拟区域信息
            f.write("## 模拟区域配置\n\n")
            f.write(f"- 网格尺寸: {self.grid_config.nx} × {self.grid_config.ny} × {self.grid_config.nz}\n")
            f.write(f"- 空间分辨率: {self.grid_config.dx/1000:.1f} km × {self.grid_config.dy/1000:.1f} km × {self.grid_config.dz} m\n")
            f.write(f"- 模拟区域: {self.grid_config.x_max/1000:.0f} km × {self.grid_config.y_max/1000:.0f} km × {self.grid_config.z_max:.0f} m\n\n")

            # 洋流模拟结果
            if 'current' in self.simulation_results:
                f.write("## 洋流模拟结果\n\n")
                current_field = self.simulation_results['current']
                max_u = np.max(np.abs(current_field['u']))
                max_v = np.max(np.abs(current_field['v']))
                max_w = np.max(np.abs(current_field['w']))

                f.write(f"- 最大东向速度: {max_u:.3f} m/s\n")
                f.write(f"- 最大北向速度: {max_v:.3f} m/s\n")
                f.write(f"- 最大垂直速度: {max_w:.4f} m/s\n")
                f.write(f"- 平均水平速度: {np.mean(np.sqrt(current_field['u']**2 + current_field['v']**2)):.3f} m/s\n\n")

            # 粒子追踪结果
            if 'particles' in self.simulation_results:
                f.write("## 粒子追踪结果\n\n")
                trajectories = self.simulation_results['particles']
                total_particles = len(trajectories)
                successful_particles = len([t for t in trajectories if t])

                f.write(f"- 总粒子数: {total_particles}\n")
                f.write(f"- 成功追踪粒子数: {successful_particles}\n")
                f.write(f"- 成功率: {successful_particles/total_particles*100:.1f}%\n")

                if trajectories:
                    # 计算平均追踪时间
                    avg_time = np.mean([len(t) for t in trajectories if t]) * 3600  # 假设1小时间隔
                    f.write(f"- 平均追踪时间: {avg_time/3600:.1f} 小时\n")

                    # 计算扩散距离
                    final_positions = []
                    for traj in trajectories:
                        if traj and len(traj) > 1:
                            start = traj[0]
                            end = traj[-1]
                            distance = np.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
                            final_positions.append(distance)

                    if final_positions:
                        f.write(f"- 平均扩散距离: {np.mean(final_positions)/1000:.1f} km\n")
                        f.write(f"- 最大扩散距离: {np.max(final_positions)/1000:.1f} km\n\n")

            # 污染物扩散结果
            if 'pollution' in self.simulation_results:
                f.write("## 污染物扩散结果\n\n")
                pollution_stats = self.simulation_results['pollution']

                f.write(f"- 初始粒子数: {pollution_stats['total_particles']}\n")
                f.write(f"- 最终活跃粒子数: {pollution_stats['active_particles']}\n")
                f.write(f"- 存活率: {pollution_stats['active_particles']/pollution_stats['total_particles']*100:.1f}%\n")
                f.write(f"- 剩余总质量: {pollution_stats['total_mass']:.2e} kg\n")
                f.write(f"- 质心位置: ({pollution_stats['center_of_mass'][0]/1000:.1f}, {pollution_stats['center_of_mass'][1]/1000:.1f}) km\n")
                f.write(f"- 扩散范围: {pollution_stats['extent']/1000:.1f} km\n")

                if 'weathering_summary' in pollution_stats and pollution_stats['weathering_summary']:
                    f.write("\n### 风化过程统计\n\n")
                    for process, stats in pollution_stats['weathering_summary'].items():
                        process_name = {
                            'evaporated_fraction': '蒸发',
                            'dissolved_fraction': '溶解',
                            'degraded_fraction': '降解',
                            'settled_fraction': '沉降'
                        }.get(process, process)
                        f.write(f"- {process_name}: 平均 {stats['mean']:.1%}, 最大 {stats['max']:.1%}\n")
                f.write("\n")

            # 混合模拟结果
            if 'hybrid' in self.simulation_results:
                f.write("## 混合模拟结果\n\n")
                hybrid_diag = self.simulation_results['hybrid']

                f.write(f"- 模拟时间: {hybrid_diag.get('simulation_time', 0)/3600:.1f} 小时\n")
                f.write(f"- 时间步长: {hybrid_diag.get('time_step', 0)/60:.0f} 分钟\n")
                f.write(f"- 最大速度: {hybrid_diag.get('max_velocity', 0):.3f} m/s\n")
                f.write(f"- 动能: {hybrid_diag.get('kinetic_energy', 0):.2e} J\n")

                if 'active_particles' in hybrid_diag:
                    f.write(f"- 拉格朗日粒子数: {hybrid_diag['active_particles']}\n")

                if 'coupling_residual' in hybrid_diag:
                    f.write(f"- 耦合残差: {hybrid_diag['coupling_residual']:.2e}\n")

                if 'recent_refinements' in hybrid_diag:
                    f.write(f"- 近期网格细化次数: {hybrid_diag['recent_refinements']}\n")
                f.write("\n")

            # 文件列表
            f.write("## 输出文件列表\n\n")
            output_files = []
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    if file != "simulation_summary_report.md":
                        file_path = os.path.relpath(os.path.join(root, file), self.output_dir)
                        output_files.append(file_path)

            for file in sorted(output_files):
                f.write(f"- `{file}`\n")

            f.write("\n## 技术说明\n\n")
            f.write("本次模拟使用了C#主导的多语言洋流模拟系统，集成了以下技术:\n\n")
            f.write("- **C++核心**: 高性能数值计算和粒子追踪\n")
            f.write("- **Python引擎**: 科学计算、机器学习和数据处理\n")
            f.write("- **混合求解器**: 欧拉-拉格朗日耦合方法\n")
            f.write("- **自适应网格**: 基于梯度的网格细化\n")
            f.write("- **多尺度耦合**: 不同分辨率网格间的信息传递\n")
            f.write("- **污染物模拟**: 物理化学过程和风化机制\n\n")

            f.write("模拟结果仅供科研和教学使用，实际应用需要更详细的验证和校准。\n")

        logger.info(f"综合报告已保存到: {report_file}")

    def cleanup(self):
        """清理所有资源"""
        logger.info("清理模拟资源...")

        if self.current_solver:
            self.current_solver.cleanup()

        if self.particle_tracker:
            self.particle_tracker.cleanup()

        if self.pollution_simulator:
            self.pollution_simulator.cleanup()

        if self.hybrid_solver:
            self.hybrid_solver.cleanup()

        logger.info("资源清理完成")


def run_performance_benchmark():
    """运行性能基准测试"""
    logger.info("开始性能基准测试...")

    # 不同网格尺寸的测试
    grid_sizes = [
        (50, 40, 10),   # 小网格
        (100, 80, 20),  # 中等网格
        (200, 160, 30), # 大网格
    ]

    results = {}

    for nx, ny, nz in grid_sizes:
        logger.info(f"测试网格尺寸: {nx}×{ny}×{nz}")

        domain_config = {
            'nx': nx, 'ny': ny, 'nz': nz,
            'dx': 1000.0, 'dy': 1000.0, 'dz': 10.0,
            'x_min': 0.0, 'y_min': 0.0, 'z_min': 0.0,
            'x_max': nx*1000.0, 'y_max': ny*1000.0, 'z_max': nz*10.0
        }

        workflow = OceanSimulationWorkflow(f"benchmark_{nx}x{ny}x{nz}")

        try:
            start_time = time.time()

            # 设置并创建海洋场
            workflow.setup_simulation_domain(domain_config)
            workflow.create_synthetic_ocean_field()

            # 简化的洋流模拟
            grid_config = workflow.grid_config
            current_solver = CurrentSimulationWrapper()

            sim_params = SimulationParameters(
                time_step=300.0,
                total_time=3600.0,  # 仅1小时用于基准测试
                viscosity=1e-6
            )

            if current_solver.initialize(grid_config, sim_params):
                ocean_field = workflow.velocity_fields
                current_solver.set_initial_velocity_field(
                    ocean_field['u'], ocean_field['v'], ocean_field['w']
                )

                # 运行几个时间步
                for _ in range(5):
                    current_solver.advance_time_step()

                current_solver.cleanup()

            elapsed_time = time.time() - start_time
            grid_points = nx * ny * nz

            results[f"{nx}x{ny}x{nz}"] = {
                'grid_points': grid_points,
                'elapsed_time': elapsed_time,
                'points_per_second': grid_points / elapsed_time
            }

            logger.info(f"网格 {nx}×{ny}×{nz}: {elapsed_time:.2f}s, "
                        f"{grid_points/elapsed_time:.0f} 点/秒")

            workflow.cleanup()

        except Exception as e:
            logger.error(f"基准测试失败 ({nx}×{ny}×{nz}): {e}")

    # 输出基准测试结果
    print("\n=== 性能基准测试结果 ===")
    for grid_size, result in results.items():
        print(f"网格 {grid_size}:")
        print(f"  网格点数: {result['grid_points']:,}")
        print(f"  耗时: {result['elapsed_time']:.2f} 秒")
        print(f"  处理速度: {result['points_per_second']:.0f} 点/秒")
        print()


def main():
    """主函数 - 演示完整工作流程"""
    print("=" * 60)
    print("C# 多语言海洋模拟系统 - Python引擎演示")
    print("=" * 60)

    # 选择运行模式
    mode = input("请选择运行模式:\n"
                 "1. 完整工作流程演示\n"
                 "2. 性能基准测试\n"
                 "3. 单独模块测试\n"
                 "请输入选择 (1-3): ").strip()

    if mode == "1":
        # 完整工作流程
        workflow = OceanSimulationWorkflow("complete_simulation_demo")
        workflow.run_complete_workflow()

    elif mode == "2":
        # 性能基准测试
        run_performance_benchmark()

    elif mode == "3":
        # 单独模块测试
        module = input("请选择测试模块:\n"
                       "1. 洋流模拟 (CurrentSimulationWrapper)\n"
                       "2. 粒子追踪 (ParticleTrackingWrapper)\n"
                       "3. 污染扩散 (PollutionDispersionSimulator)\n"
                       "4. 混合求解器 (HybridSolver)\n"
                       "请输入选择 (1-4): ").strip()

        workflow = OceanSimulationWorkflow("single_module_test")

        # 基础设置
        domain_config = {
            'nx': 80, 'ny': 60, 'nz': 15,
            'dx': 1000.0, 'dy': 1000.0, 'dz': 10.0,
            'x_min': 0.0, 'y_min': 0.0, 'z_min': 0.0,
            'x_max': 80000.0, 'y_max': 60000.0, 'z_max': 150.0
        }
        workflow.setup_simulation_domain(domain_config)
        workflow.create_synthetic_ocean_field()

        try:
            if module == "1":
                workflow.run_current_simulation_example()
            elif module == "2":
                workflow.run_particle_tracking_example()
            elif module == "3":
                workflow.run_pollution_dispersion_example()
            elif module == "4":
                workflow.run_hybrid_simulation_example()
            else:
                print("无效选择")
                return

            workflow.create_visualization()
            print(f"测试完成，结果保存在: {workflow.output_dir}")

        finally:
            workflow.cleanup()

    else:
        print("无效选择")
        return

    print("\n演示完成！")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ocean_simulation.log')
        ]
    )

    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        logger.exception("程序执行异常")
        print(f"程序异常: {e}")