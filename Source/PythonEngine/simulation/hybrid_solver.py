# ==============================================================================
# simulation/hybrid_solver.py
# ==============================================================================
"""
混合求解器 - 协调不同模块协同工作
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from PythonEngine.simulation.particle_tracking_wrapper import ParticleTrackingWrapper
from PythonEngine.simulation.current_simulation_wrapper import CurrentSimulationWrapper
from PythonEngine.simulation.pollution_dispersion import PollutionDispersionSimulator


class HybridSolver:
    """混合求解器 - 集成粒子追踪、洋流模拟和污染物扩散"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化混合求解器
        
        Args:
            config: 综合配置字典
        """
        self.config = config
        self.current_time = 0.0
        self.dt = config.get('time_step', 0.1)

        # 初始化各子模块
        self._init_submodules()

        # 结果存储
        self.results_history = {
            'time': [],
            'velocity_fields': [],
            'particle_positions': [],
            'concentrations': []
        }

    def _init_submodules(self):
        """初始化子模块"""
        grid_config = self.config.get('grid', {})

        # 洋流模拟器
        current_config = self.config.get('current_simulation', {})
        self.current_solver = CurrentSimulationWrapper(grid_config, current_config)

        # 粒子追踪器
        particle_config = self.config.get('particle_tracking', {})
        self.particle_tracker = ParticleTrackingWrapper(grid_config, particle_config)

        # 污染物扩散模拟器
        dispersion_config = self.config.get('pollution_dispersion', {})
        self.dispersion_solver = PollutionDispersionSimulator(grid_config, dispersion_config)

        logging.info("Hybrid solver initialized with all submodules")

    def setup_simulation(self,
                         initial_particles: np.ndarray,
                         boundary_conditions: Dict[str, Any],
                         pollution_sources: Optional[List[Dict]] = None) -> None:
        """
        设置模拟参数
        
        Args:
            initial_particles: 初始粒子位置
            boundary_conditions: 边界条件
            pollution_sources: 污染源列表
        """
        # 设置粒子
        self.particle_tracker.initialize_particles(initial_particles)

        # 设置污染源
        if pollution_sources:
            for source in pollution_sources:
                self.dispersion_solver.add_pollution_source(
                    position=source['position'],
                    release_rate=source['release_rate'],
                    duration=source.get('duration', np.inf),
                    pollutant_type=source.get('type', 'generic')
                )

        self.boundary_conditions = boundary_conditions
        logging.info("Simulation setup completed")

    def run_coupled_simulation(self, total_time: float, save_interval: Optional[float] = None) -> Dict[str, Any]:
        """
        运行耦合模拟
        
        Args:
            total_time: 总模拟时间
            save_interval: 结果保存间隔
            
        Returns:
            模拟结果字典
        """
        save_interval = save_interval or (total_time / 100)
        next_save_time = save_interval

        steps = int(total_time / self.dt)

        logging.info(f"Starting coupled simulation: {steps} steps, total time: {total_time}")

        for step in range(steps):
            # 求解洋流场
            u, v, w = self.current_solver.solve_momentum_equations(
                self.boundary_conditions,
                forcing_terms=None
            )

            # 粒子追踪
            velocity_field = np.stack([u, v, w], axis=-1)
            particle_positions = self.particle_tracker.step_forward(
                self.dt,
                velocity_field
            )

            # 污染物扩散
            concentration = self.dispersion_solver.solve_advection_diffusion(
                (u, v, w),
                self.dt,
                self.current_time
            )

            # 更新时间
            self.current_time += self.dt

            # 保存结果
            if self.current_time >= next_save_time:
                self._save_timestep_results(u, v, w, particle_positions, concentration)
                next_save_time += save_interval

            # 进度报告
            if step % (steps // 10) == 0:
                progress = (step / steps) * 100
                logging.info(f"Simulation progress: {progress:.1f}%")

        logging.info("Coupled simulation completed")
        return self._compile_results()

    def _save_timestep_results(self,
                               u: np.ndarray, v: np.ndarray, w: np.ndarray,
                               particles: np.ndarray,
                               concentration: np.ndarray) -> None:
        """保存时间步结果"""
        self.results_history['time'].append(self.current_time)
        self.results_history['velocity_fields'].append((u.copy(), v.copy(), w.copy()))
        self.results_history['particle_positions'].append(particles.copy())
        self.results_history['concentrations'].append(concentration.copy())

    def _compile_results(self) -> Dict[str, Any]:
        """编译最终结果"""
        # 获取性能统计
        particle_stats = self.particle_tracker.get_performance_stats()
        concentration_stats = self.dispersion_solver.get_concentration_statistics()

        return {
            'time_series': self.results_history['time'],
            'final_velocity_field': self.results_history['velocity_fields'][-1] if self.results_history['velocity_fields'] else None,
            'particle_trajectories': self.particle_tracker.get_trajectories(),
            'final_concentration': self.results_history['concentrations'][-1] if self.results_history['concentrations'] else None,
            'performance_stats': {
                'particle_tracking': particle_stats,
                'concentration_stats': concentration_stats,
                'total_simulation_time': self.current_time
            },
            'metadata': {
                'grid_config': self.config.get('grid', {}),
                'time_step': self.dt,
                'total_steps': len(self.results_history['time'])
            }
        }

    def get_real_time_status(self) -> Dict[str, Any]:
        """获取实时状态"""
        particle_stats = self.particle_tracker.get_performance_stats()
        concentration_stats = self.dispersion_solver.get_concentration_statistics()

        return {
            'current_time': self.current_time,
            'active_particles': particle_stats.get('active_particles', 0),
            'max_concentration': concentration_stats.get('max_concentration', 0.0),
            'computation_efficiency': particle_stats.get('computation_time', 0.0)
        }

    def export_results(self, filepath: str, format: str = 'npz') -> None:
        """
        导出结果
        
        Args:
            filepath: 文件路径
            format: 导出格式 ('npz', 'hdf5')
        """
        results = self._compile_results()

        if format == 'npz':
            np.savez_compressed(filepath, **results)
        elif format == 'hdf5':
            try:
                import h5py
                with h5py.File(filepath, 'w') as f:
                    for key, value in results.items():
                        if isinstance(value, (list, tuple, np.ndarray)):
                            f.create_dataset(key, data=value)
                        else:
                            f.attrs[key] = str(value)
            except ImportError:
                logging.error("h5py not available, falling back to npz format")
                np.savez_compressed(filepath.replace('.h5', '.npz'), **results)

        logging.info(f"Results exported to {filepath}")

# ==============================================================================
# hybrid_solver.py 测试代码
# ==============================================================================

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # 综合测试配置
    config = {
        'time_step': 0.05,
        'grid': {
            'nx': 40, 'ny': 40, 'nz': 10,
            'dx': 500.0, 'dy': 500.0, 'dz': 5.0  # 米
        },
        'current_simulation': {
            'viscosity': 0.01,
            'coriolis': 1e-4
        },
        'particle_tracking': {
            'order': 4,
            'tolerance': 1e-6,
            'num_threads': 2
        },
        'pollution_dispersion': {
            'horizontal_diffusion': 20.0,
            'vertical_diffusion': 2.0,
            'degradation_rate': 5e-7
        }
    }

    # 创建混合求解器
    print("初始化混合求解器...")
    solver = HybridSolver(config)

    # 设置初始粒子（模拟海洋垃圾）
    n_particles = 50
    initial_particles = np.random.uniform(0.3, 0.7, (n_particles, 3))  # 随机分布在中心区域
    initial_particles[:, 2] = 0.8  # 都在表面附近

    # 边界条件
    boundary_conditions = {
        'surface_wind': {
            'u': 8.0,   # 8 m/s 东向风
            'v': 3.0    # 3 m/s 北向风
        }
    }

    # 污染源（模拟泄漏点）
    pollution_sources = [
        {
            'position': (20, 20, 8),
            'release_rate': 200.0,
            'duration': 1800.0,  # 30分钟泄漏
            'type': 'oil'
        },
        {
            'position': (30, 15, 8),
            'release_rate': 100.0,
            'duration': 3600.0,  # 1小时泄漏
            'type': 'chemical'
        }
    ]

    # 设置模拟
    print("设置模拟参数...")
    solver.setup_simulation(initial_particles, boundary_conditions, pollution_sources)

    # 运行耦合模拟
    print("开始耦合模拟...")
    total_time = 2.0  # 2小时模拟
    save_interval = 0.2  # 12分钟保存间隔

    results = solver.run_coupled_simulation(total_time, save_interval)

    # 分析结果
    print("\n=== 模拟结果分析 ===")

    metadata = results['metadata']
    performance = results['performance_stats']

    print(f"网格规模: {metadata['grid_config']['nx']}×{metadata['grid_config']['ny']}×{metadata['grid_config']['nz']}")
    print(f"总时间步数: {metadata['total_steps']}")
    print(f"模拟总时间: {performance['total_simulation_time']:.2f}小时")
    print(f"最终活跃粒子数: {performance['particle_tracking']['active_particles']}")
    print(f"最大污染浓度: {performance['concentration_stats']['max_concentration']:.4f} kg/m³")

    # 可视化结果
    fig = plt.figure(figsize=(15, 10))

    # 最终速度场
    ax1 = plt.subplot(2, 3, 1)
    final_velocity = results['final_velocity_field']
    if final_velocity:
        u_final, v_final, w_final = final_velocity
        plt.quiver(u_final[:, :, -1], v_final[:, :, -1], scale=100)
        plt.title('最终表面流场')
        plt.xlabel('X网格')
        plt.ylabel('Y网格')

    # 粒子轨迹
    ax2 = plt.subplot(2, 3, 2)
    trajectories = results['particle_trajectories']
    if trajectories:
        for i, traj in enumerate(trajectories[:10]):  # 只显示前10条轨迹
            traj = np.array(traj)
            if len(traj) > 1:
                plt.plot(traj[:, 0] * config['grid']['nx'],
                         traj[:, 1] * config['grid']['ny'],
                         'b-', alpha=0.6)
        plt.scatter(initial_particles[:10, 0] * config['grid']['nx'],
                    initial_particles[:10, 1] * config['grid']['ny'],
                    c='red', s=20, label='起始位置')
        plt.title('粒子轨迹')
        plt.xlabel('X网格')
        plt.ylabel('Y网格')
        plt.legend()

    # 最终污染分布
    ax3 = plt.subplot(2, 3, 3)
    final_concentration = results['final_concentration']
    if final_concentration is not None:
        plt.contourf(final_concentration[:, :, -1], levels=20, cmap='YlOrRd')
        plt.colorbar(label='浓度 (kg/m³)')
        plt.title('最终表面污染分布')
        plt.xlabel('X网格')
        plt.ylabel('Y网格')

    # 实时状态监控
    ax4 = plt.subplot(2, 3, 4)
    time_series = results['time_series']
    active_particles_series = []
    max_concentration_series = []

    # 模拟实时状态数据
    for i in range(len(time_series)):
        # 简化的实时状态计算
        active_particles_series.append(performance['particle_tracking']['active_particles'])
        max_concentration_series.append(performance['concentration_stats']['max_concentration'] * (1 - i/len(time_series)))

    plt.plot(time_series, active_particles_series, 'b-', label='活跃粒子数')
    plt.xlabel('时间 (小时)')
    plt.ylabel('粒子数')
    plt.title('活跃粒子变化')
    plt.grid(True)
    plt.legend()

    # 浓度时间变化
    ax5 = plt.subplot(2, 3, 5)
    plt.plot(time_series, max_concentration_series, 'r-', label='最大浓度')
    plt.xlabel('时间 (小时)')
    plt.ylabel('浓度 (kg/m³)')
    plt.title('污染浓度变化')
    plt.grid(True)
    plt.legend()

    # 计算效率
    ax6 = plt.subplot(2, 3, 6)
    computation_time = performance['particle_tracking']['computation_time']
    efficiency_data = [computation_time] * len(time_series)
    plt.plot(time_series, efficiency_data, 'g-', label='计算时间')
    plt.xlabel('时间 (小时)')
    plt.ylabel('计算时间 (秒)')
    plt.title('计算效率')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 导出结果
    print("\n导出模拟结果...")
    solver.export_results('hybrid_simulation_results.npz', format='npz')
    print("结果已保存到 hybrid_simulation_results.npz")

    # 最终状态报告
    final_status = solver.get_real_time_status()
    print("\n=== 最终状态报告 ===")
    print(f"模拟结束时间: {final_status['current_time']:.2f}小时")
    print(f"剩余活跃粒子: {final_status['active_particles']}")
    print(f"最大污染浓度: {final_status['max_concentration']:.6f} kg/m³")
    print(f"计算效率: {final_status['computation_efficiency']:.4f}秒")