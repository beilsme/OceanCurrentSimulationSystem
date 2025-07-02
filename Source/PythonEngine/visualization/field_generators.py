# ==============================================================================
# visualization/field_generators.py
# ==============================================================================
"""
场数据生成器 - 为可视化生成各种海洋物理场
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Rectangle
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from pathlib import Path
import sys

# 导入中文配置
sys.path.append(str(Path(__file__).parent.parent / "utils"))
try:
    from chinese_config import ChineseConfig
    chinese_config = ChineseConfig()
except ImportError:
    logging.warning("无法导入中文配置，将使用默认字体设置")
    chinese_config = None


class FieldGenerator:
    """海洋物理场数据生成器"""

    def __init__(self, grid_config: Dict[str, Any], chinese_support: bool = True):
        """
        初始化场生成器
        
        Args:
            grid_config: 网格配置参数
            chinese_support: 是否启用中文支持
        """
        self.grid_config = grid_config
        self.nx = grid_config.get('nx', 100)
        self.ny = grid_config.get('ny', 100)
        self.nz = grid_config.get('nz', 20)
        self.dx = grid_config.get('dx', 1.0)
        self.dy = grid_config.get('dy', 1.0)
        self.dz = grid_config.get('dz', 1.0)

        # 坐标网格
        self.x = np.linspace(0, self.nx * self.dx, self.nx)
        self.y = np.linspace(0, self.ny * self.dy, self.ny)
        self.z = np.linspace(0, self.nz * self.dz, self.nz)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # 中文支持
        if chinese_support and chinese_config:
            self.font_config = chinese_config.setup_chinese_support()
        else:
            self.font_config = {}

    def generate_velocity_field(self,
                                velocity_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                layer_index: int = -1,
                                title: str = "海流速度场",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        生成速度场可视化
        
        Args:
            velocity_data: (u, v, w) 速度分量
            layer_index: 显示的层级索引，-1表示表层
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        u, v, w = velocity_data

        # 选择显示层级
        u_layer = u[:, :, layer_index]
        v_layer = v[:, :, layer_index]

        # 计算速度幅值
        speed = np.sqrt(u_layer**2 + v_layer**2)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 流线图
        ax1 = axes[0, 0]
        strm = ax1.streamplot(self.X, self.Y, u_layer, v_layer,
                              color=speed, cmap='viridis', density=2)
        ax1.set_title('流线分布', **self.font_config)
        ax1.set_xlabel('X (m)', **self.font_config)
        ax1.set_ylabel('Y (m)', **self.font_config)
        plt.colorbar(strm.lines, ax=ax1, label='速度 (m/s)')

        # 速度矢量图
        ax2 = axes[0, 1]
        skip = max(1, min(self.nx, self.ny) // 20)  # 控制箭头密度
        Q = ax2.quiver(self.X[::skip, ::skip], self.Y[::skip, ::skip],
                       u_layer[::skip, ::skip], v_layer[::skip, ::skip],
                       speed[::skip, ::skip], cmap='plasma', scale=50)
        ax2.set_title('速度矢量', **self.font_config)
        ax2.set_xlabel('X (m)', **self.font_config)
        ax2.set_ylabel('Y (m)', **self.font_config)
        plt.colorbar(Q, ax=ax2, label='速度 (m/s)')

        # U分量等值线
        ax3 = axes[1, 0]
        cs1 = ax3.contourf(self.X, self.Y, u_layer, levels=20, cmap='RdBu_r')
        ax3.contour(self.X, self.Y, u_layer, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax3.set_title('U分量 (东向)', **self.font_config)
        ax3.set_xlabel('X (m)', **self.font_config)
        ax3.set_ylabel('Y (m)', **self.font_config)
        plt.colorbar(cs1, ax=ax3, label='速度 (m/s)')

        # V分量等值线
        ax4 = axes[1, 1]
        cs2 = ax4.contourf(self.X, self.Y, v_layer, levels=20, cmap='RdBu_r')
        ax4.contour(self.X, self.Y, v_layer, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax4.set_title('V分量 (北向)', **self.font_config)
        ax4.set_xlabel('X (m)', **self.font_config)
        ax4.set_ylabel('Y (m)', **self.font_config)
        plt.colorbar(cs2, ax=ax4, label='速度 (m/s)')

        fig.suptitle(f'{title} (第{layer_index+1}层)', **self.font_config, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"速度场图保存至: {save_path}")

        return fig

    def generate_concentration_field(self,
                                     concentration_data: np.ndarray,
                                     layer_index: int = -1,
                                     title: str = "污染物浓度分布",
                                     colormap: str = 'YlOrRd',
                                     log_scale: bool = False,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        生成浓度场可视化
        
        Args:
            concentration_data: 浓度数据
            layer_index: 显示层级
            title: 图表标题
            colormap: 色彩映射
            log_scale: 是否使用对数刻度
            save_path: 保存路径
        """
        conc_layer = concentration_data[:, :, layer_index]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 主要浓度分布
        ax1 = axes[0, 0]
        if log_scale and np.any(conc_layer > 0):
            # 对数刻度处理
            conc_plot = np.log10(conc_layer + 1e-10)
            levels = np.logspace(np.log10(np.min(conc_layer[conc_layer > 0])),
                                 np.log10(np.max(conc_layer)), 20)
            cs1 = ax1.contourf(self.X, self.Y, conc_plot, levels=20, cmap=colormap)
        else:
            cs1 = ax1.contourf(self.X, self.Y, conc_layer, levels=20, cmap=colormap)

        ax1.set_title('浓度分布', **self.font_config)
        ax1.set_xlabel('X (m)', **self.font_config)
        ax1.set_ylabel('Y (m)', **self.font_config)
        cbar1 = plt.colorbar(cs1, ax=ax1)
        cbar1.set_label('浓度 (kg/m³)', **self.font_config)

        # 3D视图
        ax2 = axes[0, 1]
        im2 = ax2.imshow(conc_layer.T, origin='lower', cmap=colormap,
                         extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
        ax2.set_title('浓度图像', **self.font_config)
        ax2.set_xlabel('X (m)', **self.font_config)
        ax2.set_ylabel('Y (m)', **self.font_config)
        plt.colorbar(im2, ax=ax2, label='浓度 (kg/m³)')

        # 浓度等值线
        ax3 = axes[1, 0]
        cs3 = ax3.contour(self.X, self.Y, conc_layer, levels=15, colors='black')
        ax3.clabel(cs3, inline=True, fontsize=8)
        ax3.set_title('浓度等值线', **self.font_config)
        ax3.set_xlabel('X (m)', **self.font_config)
        ax3.set_ylabel('Y (m)', **self.font_config)

        # 浓度统计直方图
        ax4 = axes[1, 1]
        valid_conc = conc_layer[conc_layer > 0]
        if len(valid_conc) > 0:
            ax4.hist(valid_conc, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax4.set_xlabel('浓度 (kg/m³)', **self.font_config)
            ax4.set_ylabel('频次', **self.font_config)
            ax4.set_title('浓度分布直方图', **self.font_config)
            if log_scale:
                ax4.set_xscale('log')

        fig.suptitle(f'{title} (第{layer_index+1}层)', **self.font_config, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"浓度场图保存至: {save_path}")

        return fig

    def generate_particle_field(self,
                                particle_positions: np.ndarray,
                                particle_properties: Optional[Dict] = None,
                                background_field: Optional[np.ndarray] = None,
                                title: str = "粒子分布",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        生成粒子分布可视化
        
        Args:
            particle_positions: 粒子位置数组 (N, 3)
            particle_properties: 粒子属性字典
            background_field: 背景场数据
            title: 图表标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 提取位置坐标
        x_particles = particle_positions[:, 0] * self.dx
        y_particles = particle_positions[:, 1] * self.dy
        z_particles = particle_positions[:, 2] * self.dz if particle_positions.shape[1] > 2 else None

        # 粒子散点图
        ax1 = axes[0, 0]
        if background_field is not None:
            # 显示背景场
            im = ax1.contourf(self.X, self.Y, background_field[:, :, -1],
                              levels=20, cmap='Blues', alpha=0.6)
            plt.colorbar(im, ax=ax1, label='背景场')

        scatter = ax1.scatter(x_particles, y_particles,
                              c=z_particles if z_particles is not None else 'red',
                              cmap='viridis', s=20, alpha=0.7)
        ax1.set_title('粒子水平分布', **self.font_config)
        ax1.set_xlabel('X (m)', **self.font_config)
        ax1.set_ylabel('Y (m)', **self.font_config)
        if z_particles is not None:
            plt.colorbar(scatter, ax=ax1, label='深度 (m)')

        # 粒子密度图
        ax2 = axes[0, 1]
        hist, xedges, yedges = np.histogram2d(x_particles, y_particles, bins=30)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im2 = ax2.imshow(hist.T, origin='lower', extent=extent, cmap='hot')
        ax2.set_title('粒子密度分布', **self.font_config)
        ax2.set_xlabel('X (m)', **self.font_config)
        ax2.set_ylabel('Y (m)', **self.font_config)
        plt.colorbar(im2, ax=ax2, label='粒子密度')

        # 深度分布直方图
        ax3 = axes[1, 0]
        if z_particles is not None:
            ax3.hist(z_particles, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax3.set_xlabel('深度 (m)', **self.font_config)
            ax3.set_ylabel('粒子数量', **self.font_config)
            ax3.set_title('深度分布', **self.font_config)
        else:
            ax3.text(0.5, 0.5, '无深度数据', ha='center', va='center',
                     transform=ax3.transAxes, **self.font_config)
            ax3.set_title('深度分布', **self.font_config)

        # 粒子属性散点图
        ax4 = axes[1, 1]
        if particle_properties and 'age' in particle_properties:
            ages = particle_properties['age']
            scatter4 = ax4.scatter(x_particles, y_particles, c=ages,
                                   cmap='plasma', s=30, alpha=0.8)
            plt.colorbar(scatter4, ax=ax4, label='粒子年龄')
            ax4.set_title('粒子年龄分布', **self.font_config)
        else:
            # 显示粒子聚集情况
            from scipy.spatial.distance import pdist
            if len(particle_positions) > 1:
                distances = pdist(particle_positions[:, :2])
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                ax4.text(0.1, 0.9, f'平均距离: {mean_dist:.2f}m',
                         transform=ax4.transAxes, **self.font_config)
                ax4.text(0.1, 0.8, f'距离标准差: {std_dist:.2f}m',
                         transform=ax4.transAxes, **self.font_config)
                ax4.text(0.1, 0.7, f'粒子总数: {len(particle_positions)}',
                         transform=ax4.transAxes, **self.font_config)
            ax4.set_title('粒子统计信息', **self.font_config)

        ax4.set_xlabel('X (m)', **self.font_config)
        ax4.set_ylabel('Y (m)', **self.font_config)

        fig.suptitle(title, **self.font_config, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"粒子场图保存至: {save_path}")

        return fig

if __name__ == "__main__":
    # field_generators.py 测试
    import numpy as np
    import matplotlib.pyplot as plt

    # 测试配置
    grid_config = {
        'nx': 60, 'ny': 60, 'nz': 15,
        'dx': 100.0, 'dy': 100.0, 'dz': 5.0
    }

    # 创建场生成器
    field_gen = FieldGenerator(grid_config)

    # 生成测试数据
    print("生成测试速度场数据...")
    u = np.random.normal(0, 1, (60, 60, 15))
    v = np.random.normal(0, 1, (60, 60, 15))
    w = np.random.normal(0, 0.1, (60, 60, 15))

    # 添加一些结构
    X, Y = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60), indexing='ij')
    for k in range(15):
        u[:, :, k] += np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
        v[:, :, k] += np.cos(2*np.pi*X) * np.sin(2*np.pi*Y)

    # 生成速度场可视化
    velocity_data = (u, v, w)
    fig1 = field_gen.generate_velocity_field(velocity_data, title="测试海流速度场")
    plt.show()

    # 生成浓度场数据
    print("生成测试浓度场数据...")
    concentration = np.zeros((60, 60, 15))

    # 添加污染源
    source_x, source_y = 30, 30
    for i in range(60):
        for j in range(60):
            for k in range(15):
                dist = np.sqrt((i-source_x)**2 + (j-source_y)**2)
                concentration[i, j, k] = 100 * np.exp(-dist/10) * np.exp(-k/5)

    # 生成浓度场可视化
    fig2 = field_gen.generate_concentration_field(concentration, title="测试污染物浓度分布")
    plt.show()

    # 生成粒子数据
    print("生成测试粒子数据...")
    n_particles = 200
    particle_positions = np.random.uniform(0, 1, (n_particles, 3))
    particle_properties = {'age': np.random.exponential(2, n_particles)}

    # 生成粒子场可视化
    fig3 = field_gen.generate_particle_field(particle_positions, particle_properties,
                                             background_field=concentration,
                                             title="测试粒子分布")
    plt.show()

    print("场生成器测试完成!")

