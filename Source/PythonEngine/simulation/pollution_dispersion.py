# ==============================================================================
# simulation/pollution_dispersion.py
# ==============================================================================
"""
污染物扩散模拟器
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy.ndimage import gaussian_filter
import logging


class PollutionDispersionSimulator:
    """污染物扩散模拟器"""

    def __init__(self, grid_config: Dict[str, Any], diffusion_config: Optional[Dict] = None):
        """
        初始化污染物扩散模拟器
        
        Args:
            grid_config: 网格配置
            diffusion_config: 扩散参数配置
        """
        self.grid_config = grid_config
        self.diffusion_config = diffusion_config or {}

        # 网格参数
        self.nx = grid_config.get('nx', 100)
        self.ny = grid_config.get('ny', 100)
        self.nz = grid_config.get('nz', 50)

        # 扩散参数
        self.diffusion_coeff = self.diffusion_config.get('horizontal_diffusion', 1.0)
        self.vertical_diffusion = self.diffusion_config.get('vertical_diffusion', 0.1)
        self.degradation_rate = self.diffusion_config.get('degradation_rate', 0.0)

        # 浓度场
        self.concentration = np.zeros((self.nx, self.ny, self.nz))
        self.source_terms = []

    def add_pollution_source(self,
                             position: Tuple[int, int, int],
                             release_rate: float,
                             duration: float = np.inf,
                             pollutant_type: str = "generic") -> None:
        """
        添加污染源
        
        Args:
            position: 源位置 (i, j, k)
            release_rate: 释放速率
            duration: 持续时间
            pollutant_type: 污染物类型
        """
        source = {
            'position': position,
            'release_rate': release_rate,
            'duration': duration,
            'pollutant_type': pollutant_type,
            'start_time': 0.0
        }
        self.source_terms.append(source)

    def solve_advection_diffusion(self,
                                  velocity_field: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  dt: float,
                                  current_time: float = 0.0) -> np.ndarray:
        """
        求解平流扩散方程
        
        Args:
            velocity_field: 速度场 (u, v, w)
            dt: 时间步长
            current_time: 当前时间
            
        Returns:
            更新后的浓度场
        """
        u, v, w = velocity_field

        # 平流项计算
        concentration_new = self._compute_advection(self.concentration, u, v, w, dt)

        # 扩散项计算
        concentration_new = self._compute_diffusion(concentration_new, dt)

        # 源项
        concentration_new = self._apply_sources(concentration_new, dt, current_time)

        # 降解项
        if self.degradation_rate > 0:
            concentration_new *= np.exp(-self.degradation_rate * dt)

        self.concentration = concentration_new
        return self.concentration

    def _compute_advection(self,
                           concentration: np.ndarray,
                           u: np.ndarray, v: np.ndarray, w: np.ndarray,
                           dt: float) -> np.ndarray:
        """计算平流项"""
        # 使用上风差分格式
        dx = self.grid_config.get('dx', 1.0)
        dy = self.grid_config.get('dy', 1.0)
        dz = self.grid_config.get('dz', 1.0)

        # 上风差分
        concentration_new = concentration.copy()

        # x方向平流
        for i in range(1, self.nx-1):
            for j in range(self.ny):
                for k in range(self.nz):
                    if u[i, j, k] > 0:
                        grad_x = (concentration[i, j, k] - concentration[i-1, j, k]) / dx
                    else:
                        grad_x = (concentration[i+1, j, k] - concentration[i, j, k]) / dx
                    concentration_new[i, j, k] -= u[i, j, k] * grad_x * dt

        # y方向平流
        for i in range(self.nx):
            for j in range(1, self.ny-1):
                for k in range(self.nz):
                    if v[i, j, k] > 0:
                        grad_y = (concentration[i, j, k] - concentration[i, j-1, k]) / dy
                    else:
                        grad_y = (concentration[i, j+1, k] - concentration[i, j, k]) / dy
                    concentration_new[i, j, k] -= v[i, j, k] * grad_y * dt

        return concentration_new

    def _compute_diffusion(self, concentration: np.ndarray, dt: float) -> np.ndarray:
        """计算扩散项"""
        # 使用高斯滤波近似扩散
        sigma_h = np.sqrt(2 * self.diffusion_coeff * dt)
        sigma_v = np.sqrt(2 * self.vertical_diffusion * dt)

        # 分别在水平和垂直方向应用扩散
        result = concentration.copy()

        # 水平扩散
        for k in range(self.nz):
            result[:, :, k] = gaussian_filter(result[:, :, k], sigma=sigma_h)

        # 垂直扩散
        for i in range(self.nx):
            for j in range(self.ny):
                result[i, j, :] = gaussian_filter(result[i, j, :], sigma=sigma_v)

        return result

    def _apply_sources(self,
                       concentration: np.ndarray,
                       dt: float,
                       current_time: float) -> np.ndarray:
        """应用污染源项"""
        result = concentration.copy()

        for source in self.source_terms:
            # 检查源是否活跃
            elapsed_time = current_time - source['start_time']
            if 0 <= elapsed_time <= source['duration']:
                i, j, k = source['position']
                if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
                    result[i, j, k] += source['release_rate'] * dt

        return result

    def get_concentration_statistics(self) -> Dict[str, float]:
        """获取浓度统计信息"""
        return {
            'max_concentration': np.max(self.concentration),
            'mean_concentration': np.mean(self.concentration),
            'total_mass': np.sum(self.concentration),
            'peak_location': np.unravel_index(np.argmax(self.concentration), self.concentration.shape)
        }

    def get_concentration_at_depth(self, depth_index: int) -> np.ndarray:
        """获取指定深度的浓度分布"""
        if 0 <= depth_index < self.nz:
            return self.concentration[:, :, depth_index]
        else:
            raise ValueError(f"Depth index {depth_index} out of range [0, {self.nz-1}]")

# ==============================================================================
# pollution_dispersion.py 测试代码
# ==============================================================================

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # 测试配置
    grid_config = {
        'nx': 60, 'ny': 60, 'nz': 15,
        'dx': 100.0, 'dy': 100.0, 'dz': 10.0  # 米
    }

    diffusion_config = {
        'horizontal_diffusion': 10.0,  # m²/s
        'vertical_diffusion': 1.0,     # m²/s
        'degradation_rate': 1e-6       # s⁻¹
    }

    # 创建污染扩散模拟器
    dispersion = PollutionDispersionSimulator(grid_config, diffusion_config)

    # 添加污染源
    dispersion.add_pollution_source(
        position=(30, 30, 5),  # 网格中心附近
        release_rate=100.0,    # kg/s
        duration=3600.0,       # 1小时
        pollutant_type="oil"
    )

    dispersion.add_pollution_source(
        position=(20, 40, 5),  # 第二个源
        release_rate=50.0,
        duration=1800.0,       # 30分钟
        pollutant_type="chemical"
    )

    # 创建简单的流场（东北向流）
    u = np.full((grid_config['nx'], grid_config['ny'], grid_config['nz']), 0.5)  # 0.5 m/s 东向
    v = np.full((grid_config['nx'], grid_config['ny'], grid_config['nz']), 0.3)  # 0.3 m/s 北向
    w = np.zeros((grid_config['nx'], grid_config['ny'], grid_config['nz']))      # 无垂直流动

    velocity_field = (u, v, w)

    print("开始污染物扩散模拟...")

    # 时间参数
    dt = 60.0  # 1分钟时间步
    total_time = 7200.0  # 2小时模拟
    save_interval = 600.0  # 10分钟保存一次

    time_series = []
    concentration_series = []

    current_time = 0.0
    next_save = save_interval

    while current_time < total_time:
        # 求解扩散方程
        concentration = dispersion.solve_advection_diffusion(velocity_field, dt, current_time)

        # 保存结果
        if current_time >= next_save:
            time_series.append(current_time / 3600.0)  # 转换为小时
            concentration_series.append(concentration[:, :, 5].copy())  # 保存中层浓度
            next_save += save_interval

            # 打印统计
            stats = dispersion.get_concentration_statistics()
            print(f"时间: {current_time/3600:.2f}h, 最大浓度: {stats['max_concentration']:.3f} kg/m³")

        current_time += dt

    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 显示不同时间的浓度分布
    time_indices = [0, len(concentration_series)//3, 2*len(concentration_series)//3, -1]

    for i, idx in enumerate(time_indices[:4]):
        row = i // 2
        col = i % 2

        if idx < len(concentration_series):
            im = axes[row, col].contourf(concentration_series[idx], levels=20, cmap='YlOrRd')
            axes[row, col].set_title(f'浓度分布 t={time_series[idx]:.1f}h')
            axes[row, col].set_xlabel('X网格')
            axes[row, col].set_ylabel('Y网格')
            plt.colorbar(im, ax=axes[row, col])

    # 时间序列图
    max_concentrations = [np.max(conc) for conc in concentration_series]
    total_masses = [np.sum(conc) for conc in concentration_series]

    axes[1, 2].plot(time_series, max_concentrations, 'r-', label='最大浓度')
    axes[1, 2].set_xlabel('时间 (小时)')
    axes[1, 2].set_ylabel('浓度 (kg/m³)')
    axes[1, 2].set_title('浓度时间变化')
    axes[1, 2].grid(True)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()

    # 最终统计
    final_stats = dispersion.get_concentration_statistics()
    print(f"\n最终统计:")
    print(f"最大浓度: {final_stats['max_concentration']:.3f} kg/m³")
    print(f"平均浓度: {final_stats['mean_concentration']:.6f} kg/m³")
    print(f"总质量: {final_stats['total_mass']:.1f} kg")
    print(f"峰值位置: {final_stats['peak_location']}")