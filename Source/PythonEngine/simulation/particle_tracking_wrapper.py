# ==============================================================================
# simulation/particle_tracking_wrapper.py
# ==============================================================================
"""
粒子追踪包装器 - 调用C++高性能计算核心
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

try:
    import oceansim  # C++模块绑定
except ImportError:
    logging.warning("C++ oceansim module not found, using fallback implementation")
    oceansim = None


class ParticleTrackingWrapper:
    """粒子追踪模拟包装器"""

    def __init__(self, grid_data: Dict[str, Any], solver_config: Optional[Dict] = None):
        """
        初始化粒子追踪器
        
        Args:
            grid_data: 网格数据配置
            solver_config: 求解器配置
        """
        self.grid_data = grid_data
        self.solver_config = solver_config or {}
        self._cpp_simulator = None
        self._particles = []

        # 初始化C++模拟器
        if oceansim is not None:
            self._init_cpp_simulator()

    def _init_cpp_simulator(self):
        """初始化C++粒子模拟器"""
        try:
            # 创建网格数据结构
            grid = oceansim.GridDataStructure(
                self.grid_data.get('nx', 100),
                self.grid_data.get('ny', 100),
                self.grid_data.get('nz', 50)
            )

            # 创建求解器
            solver = oceansim.RungeKuttaSolver(
                self.solver_config.get('order', 4),
                self.solver_config.get('tolerance', 1e-6)
            )

            # 创建粒子模拟器
            self._cpp_simulator = oceansim.ParticleSimulator(grid, solver)

            # 设置并行线程数
            threads = self.solver_config.get('num_threads', 4)
            self._cpp_simulator.setNumThreads(threads)

        except Exception as e:
            logging.error(f"Failed to initialize C++ simulator: {e}")
            self._cpp_simulator = None

    def initialize_particles(self, positions: np.ndarray) -> None:
        """
        初始化粒子位置
        
        Args:
            positions: 粒子初始位置数组 (N, 3)
        """
        self._particles = positions.copy()

        if self._cpp_simulator:
            # 转换为C++格式
            cpp_positions = [list(pos) for pos in positions]
            self._cpp_simulator.initializeParticles(cpp_positions)
        else:
            # 使用Python fallback
            logging.info("Using Python fallback for particle initialization")

    def step_forward(self, dt: float, velocity_field: np.ndarray) -> np.ndarray:
        """
        时间步进
        
        Args:
            dt: 时间步长
            velocity_field: 速度场数据
            
        Returns:
            更新后的粒子位置
        """
        if self._cpp_simulator:
            # 使用C++高性能计算
            self._cpp_simulator.stepForward(dt)
            particles = self._cpp_simulator.getParticles()
            # 转换回numpy数组
            return np.array([[p.position[0], p.position[1], p.position[2]] for p in particles])
        else:
            # Python fallback实现
            return self._step_forward_python(dt, velocity_field)

    def _step_forward_python(self, dt: float, velocity_field: np.ndarray) -> np.ndarray:
        """Python版本的时间步进（备用实现）"""
        # 简化的RK4实现
        positions = self._particles

        # RK4积分
        k1 = self._interpolate_velocity(positions, velocity_field)
        k2 = self._interpolate_velocity(positions + 0.5*dt*k1, velocity_field)
        k3 = self._interpolate_velocity(positions + 0.5*dt*k2, velocity_field)
        k4 = self._interpolate_velocity(positions + dt*k3, velocity_field)

        # 更新位置
        self._particles = positions + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self._particles

    def _interpolate_velocity(self, positions: np.ndarray, velocity_field: np.ndarray) -> np.ndarray:
        """简化的速度场插值"""
        # 这里使用最近邻插值作为示例
        # 实际应用中应使用更精确的插值方法
        velocities = np.zeros_like(positions)

        for i, pos in enumerate(positions):
            # 转换为网格索引
            ix = int(np.clip(pos[0] * self.grid_data.get('nx', 100), 0, self.grid_data.get('nx', 100)-1))
            iy = int(np.clip(pos[1] * self.grid_data.get('ny', 100), 0, self.grid_data.get('ny', 100)-1))
            iz = int(np.clip(pos[2] * self.grid_data.get('nz', 50), 0, self.grid_data.get('nz', 50)-1))

            if velocity_field.shape[0] > ix and velocity_field.shape[1] > iy and velocity_field.shape[2] > iz:
                velocities[i] = velocity_field[ix, iy, iz]

        return velocities

    def get_trajectories(self) -> List[np.ndarray]:
        """获取粒子轨迹"""
        if self._cpp_simulator:
            return self._cpp_simulator.getTrajectories()
        else:
            # 返回当前位置作为简化轨迹
            return [self._particles]

    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if self._cpp_simulator:
            return {
                'computation_time': self._cpp_simulator.getComputationTime(),
                'active_particles': self._cpp_simulator.getActiveParticleCount()
            }
        return {'computation_time': 0.0, 'active_particles': len(self._particles)}


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # 测试配置
    grid_data = {
        'nx': 50, 'ny': 50, 'nz': 20,
        'dx': 0.1, 'dy': 0.1, 'dz': 0.05
    }

    solver_config = {
        'order': 4,
        'tolerance': 1e-6,
        'num_threads': 2
    }

    # 创建粒子追踪器
    tracker = ParticleTrackingWrapper(grid_data, solver_config)

    # 初始化粒子（圆形分布）
    n_particles = 100
    theta = np.linspace(0, 2*np.pi, n_particles)
    radius = 0.1
    positions = np.column_stack([
        0.5 + radius * np.cos(theta),  # x
        0.5 + radius * np.sin(theta),  # y
        np.full(n_particles, 0.5)      # z
    ])

    tracker.initialize_particles(positions)

    # 创建简单的旋转速度场
    u = np.zeros((grid_data['nx'], grid_data['ny'], grid_data['nz']))
    v = np.zeros((grid_data['nx'], grid_data['ny'], grid_data['nz']))
    w = np.zeros((grid_data['nx'], grid_data['ny'], grid_data['nz']))

    # 旋转流场
    for i in range(grid_data['nx']):
        for j in range(grid_data['ny']):
            x = i / grid_data['nx'] - 0.5
            y = j / grid_data['ny'] - 0.5
            u[i, j, :] = -y * 2.0  # 旋转速度
            v[i, j, :] = x * 2.0

    velocity_field = np.stack([u, v, w], axis=-1)

    # 运行模拟
    print("开始粒子追踪模拟...")
    positions_history = [tracker._particles.copy()]

    for step in range(50):
        new_positions = tracker.step_forward(0.01, velocity_field)
        positions_history.append(new_positions.copy())
        if step % 10 == 0:
            print(f"步骤 {step}: {len(new_positions)} 个活跃粒子")

    # 可视化结果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(positions_history[0][:, 0], positions_history[0][:, 1], 'bo', label='初始位置')
    plt.plot(positions_history[-1][:, 0], positions_history[-1][:, 1], 'ro', label='最终位置')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('粒子位置变化')
    plt.legend()
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    # 绘制轨迹
    for i in range(0, n_particles, 10):
        trajectory = np.array([pos[i] for pos in positions_history])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'g-', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('粒子轨迹')
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    # 性能统计
    stats = tracker.get_performance_stats()
    print(f"\n性能统计:")
    print(f"计算时间: {stats['computation_time']:.4f}s")
    print(f"活跃粒子数: {stats['active_particles']}")



